// Phase 5: the two-stage momentum-exchange reduction (forces.wgsl) against
// the CPU reference momentumExchangeForce(), in real Chromium WebGPU.
//
// Both read the SAME distribution buffer, so the comparison isolates the
// reduction and the force formula from any streaming/collision ordering: a
// realistic non-trivial field is produced by running the CPU solver a few
// steps with a cylinder, quantized to f32, uploaded to the GPU, and reduced.

import { describe, expect, it } from 'vitest';
import { CpuLbm, equilibrium, momentumExchangeForce } from '../src/solver/cpu-lbm.ts';
import { createLbmBuffers, writeMask, writeParams, type SimParams } from '../src/gpu/buffers.ts';
import { createForcePipelines } from '../src/gpu/pipelines.ts';
import { rasterizeCircle } from '../src/geometry/presets.ts';
import { CX, CY, Q } from '../src/solver/constants.ts';
import { assertShaderCompiles } from './gpu-test-utils.ts';

const NX = 96;
const NY = 48;
const N = NX * NY;
const TAU = 0.6;
const U0 = 0.08;

async function getDevice(): Promise<GPUDevice> {
  expect(navigator.gpu, 'WebGPU unavailable in the test browser').toBeDefined();
  const adapter = await navigator.gpu.requestAdapter();
  expect(adapter, 'no WebGPU adapter in the test browser').not.toBeNull();
  return adapter!.requestDevice();
}

const baseParams = (): SimParams => ({
  nx: NX,
  ny: NY,
  tau: TAU,
  inletU: U0,
  periodicY: false,
  dyeEnabled: false,
  viewMode: 'velocity',
  stepIndex: 0,
  substeps: 1,
});

/**
 * Develop a non-trivial wake and return the f32-quantized POST-COLLISION
 * distributions -- exactly what the fused GPU kernel stores each step, so the
 * momentum-exchange force computed from it is physical (positive drag).
 */
function developedField(mask: Uint8Array, steps: number): Float32Array {
  const cpu = new CpuLbm({
    nx: NX,
    ny: NY,
    tau: TAU,
    xBoundary: 'inlet-outflow',
    yBoundary: 'free-slip',
    inletVelocity: U0,
  });
  cpu.mask.set(mask);
  cpu.initUniformEquilibrium(1, U0, 0);
  cpu.step(steps);
  // Collide the current (post-stream) state once, without streaming, to get
  // the post-collision populations the GPU force pass reads.
  const omega = 1 / TAU;
  const post = new Float32Array(Q * N);
  for (let cell = 0; cell < N; cell++) {
    if (mask[cell] !== 0) continue;
    let r = 0;
    let mx = 0;
    let my = 0;
    for (let i = 0; i < Q; i++) {
      const v = cpu.f[i * N + cell]!;
      r += v;
      mx += v * CX[i]!;
      my += v * CY[i]!;
    }
    const ux = mx / r;
    const uy = my / r;
    for (let i = 0; i < Q; i++) {
      const k = i * N + cell;
      post[k] = cpu.f[k]! - omega * (cpu.f[k]! - equilibrium(i, r, ux, uy));
    }
  }
  return post;
}

async function reduceForceOnGpu(f: Float32Array, mask: Uint8Array): Promise<[number, number]> {
  const device = await getDevice();
  const buffers = createLbmBuffers(device, NX, NY);
  writeParams(device, buffers, baseParams());
  writeMask(device, buffers, mask);
  device.queue.writeBuffer(buffers.f[0], 0, f);

  const forces = createForcePipelines(device, buffers);
  await assertShaderCompiles(forces.module);

  const encoder = device.createCommandEncoder();
  // Separate passes: WebGPU synchronizes storage writes across pass
  // boundaries, so stage 2 sees stage 1's partials.
  const acc = encoder.beginComputePass();
  acc.setPipeline(forces.accumulate);
  acc.setBindGroup(0, forces.accumulateBindGroups[0]);
  acc.dispatchWorkgroups(forces.workgroupsX, forces.workgroupsY);
  acc.end();
  const red = encoder.beginComputePass();
  red.setPipeline(forces.reduce);
  red.setBindGroup(0, forces.reduceBindGroup);
  red.dispatchWorkgroups(1);
  red.end();

  const staging = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(buffers.forceResult, 0, staging, 0, 8);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  device.destroy();
  return [out[0]!, out[1]!];
}

describe('forces.wgsl two-stage momentum-exchange reduction', () => {
  it('matches the CPU reference for a cylinder wake', async () => {
    const mask = new Uint8Array(N);
    rasterizeCircle(mask, NX, NY, 24, 24, 8); // D = 16, centered
    const f = developedField(mask, 40);
    const expected = momentumExchangeForce(f, mask, NX, NY);
    const [fx, fy] = await reduceForceOnGpu(f, mask);
    // f32 reduction vs f64 reference; the force magnitude is O(0.1..1) here.
    expect(fx).toBeCloseTo(expected.fx, 4);
    expect(fy).toBeCloseTo(expected.fy, 4);
    // Physical sanity: drag is positive (+x) for west-to-east inflow.
    expect(expected.fx).toBeGreaterThan(0);
  });

  it('reports near-zero lift for a symmetric cylinder in symmetric flow', async () => {
    const mask = new Uint8Array(N);
    rasterizeCircle(mask, NX, NY, 24, 24, 8);
    const f = developedField(mask, 40);
    const [, fy] = await reduceForceOnGpu(f, mask);
    // Free-slip walls + centered cylinder + uniform inflow stays symmetric,
    // so Fy cancels to roundoff.
    expect(Math.abs(fy)).toBeLessThan(1e-3);
  });

  it('reports zero force with no obstacle', async () => {
    const mask = new Uint8Array(N); // all fluid
    const f = developedField(mask, 10);
    const [fx, fy] = await reduceForceOnGpu(f, mask);
    expect(fx).toBeCloseTo(0, 5);
    expect(fy).toBeCloseTo(0, 5);
  });

  it('sums partials across many workgroups (off-8 grid)', async () => {
    // A large off-multiple obstacle spans many 8x8 tiles, exercising the
    // stage-2 stride loop and the ceil() workgroup count together.
    const mask = new Uint8Array(N);
    rasterizeCircle(mask, NX, NY, 30, 24, 13);
    const f = developedField(mask, 30);
    const expected = momentumExchangeForce(f, mask, NX, NY);
    const [fx, fy] = await reduceForceOnGpu(f, mask);
    expect(fx).toBeCloseTo(expected.fx, 4);
    expect(fy).toBeCloseTo(expected.fy, 4);
  });
});
