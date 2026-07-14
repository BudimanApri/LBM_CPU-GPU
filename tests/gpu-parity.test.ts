// Phase 2 parity gate: the fused WGSL pull-scheme kernel against the CPU
// reference solver. Runs in real Chromium (Vitest browser mode) with actual
// WebGPU -- 500 steps on identical 128x64 initial conditions with a
// cylinder; max abs difference in rho, ux, uy must stay below 1e-4 (the
// f32-vs-f64 accumulation tolerance).
//
// Correspondence note: the GPU kernel fuses stream->BC->collide and stores
// post-collision states, while the CPU does collide->stream->BC. Starting
// both from the same equilibrium (a collision fixed point), the GPU's
// PRE-collision moments at dispatch k equal the CPU's post-step moments at
// step k exactly -- so the comparison is over the moment fields the kernel
// writes, which is also what Phase 4 rendering consumes.

import { describe, expect, it } from 'vitest';
import { CpuLbm, type YBoundary } from '../src/solver/cpu-lbm.ts';
import {
  createLbmBuffers,
  writeDistributions,
  writeMask,
  writeParams,
} from '../src/gpu/buffers.ts';
import { createLbmPipeline } from '../src/gpu/pipelines.ts';
import { rasterizeCircle } from '../src/geometry/presets.ts';

const NX = 128;
const NY = 64;
const TAU = 0.6;
const U0 = 0.08;
// Cylinder at 25% chord, vertically centered, D = 16 cells.
const MASK = new Uint8Array(NX * NY);
rasterizeCircle(MASK, NX, NY, 32, 32, 8);

async function getDevice(): Promise<GPUDevice> {
  expect(navigator.gpu, 'WebGPU unavailable in the test browser').toBeDefined();
  const adapter = await navigator.gpu.requestAdapter();
  expect(adapter, 'no WebGPU adapter in the test browser').not.toBeNull();
  return adapter!.requestDevice();
}

function makeCpu(yBoundary: YBoundary): CpuLbm {
  const cpu = new CpuLbm({
    nx: NX,
    ny: NY,
    tau: TAU,
    xBoundary: 'inlet-outflow',
    yBoundary,
    inletVelocity: U0,
  });
  cpu.mask.set(MASK);
  cpu.initUniformEquilibrium(1, U0, 0);
  // Quantize the IC to f32 so both solvers start from bit-identical states
  // and the comparison measures accumulation differences only.
  cpu.f.set(new Float32Array(cpu.f));
  return cpu;
}

interface GpuMoments {
  rho: Float32Array;
  ux: Float32Array;
  uy: Float32Array;
}

async function runGpu(
  device: GPUDevice,
  cpuInitialF: Float64Array,
  yBoundary: YBoundary,
  steps: number,
): Promise<GpuMoments> {
  const buffers = createLbmBuffers(device, NX, NY);
  writeParams(device, buffers, {
    nx: NX,
    ny: NY,
    tau: TAU,
    inletU: U0,
    periodicY: yBoundary === 'periodic',
    dyeEnabled: false,
    viewMode: 'velocity',
    stepIndex: 0,
    substeps: 1,
  });
  writeMask(device, buffers, MASK);
  writeDistributions(device, buffers, 0, cpuInitialF);

  const lbm = createLbmPipeline(device, buffers);
  const encoder = device.createCommandEncoder();
  for (let s = 0; s < steps; s++) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(lbm.pipeline);
    pass.setBindGroup(0, lbm.bindGroups[s % 2]);
    pass.dispatchWorkgroups(lbm.workgroupsX, lbm.workgroupsY);
    pass.end();
  }
  const byteSize = buffers.n * 4;
  const staging = [0, 1, 2].map(() =>
    device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
  );
  encoder.copyBufferToBuffer(buffers.rho, 0, staging[0]!, 0, byteSize);
  encoder.copyBufferToBuffer(buffers.ux, 0, staging[1]!, 0, byteSize);
  encoder.copyBufferToBuffer(buffers.uy, 0, staging[2]!, 0, byteSize);
  device.queue.submit([encoder.finish()]);

  await Promise.all(staging.map((b) => b.mapAsync(GPUMapMode.READ)));
  const [rho, ux, uy] = staging.map((b) => {
    const data = new Float32Array(b.getMappedRange().slice(0));
    b.unmap();
    return data;
  });
  return { rho: rho!, ux: ux!, uy: uy! };
}

function maxAbsDiff(gpu: Float32Array, cpu: Float64Array): number {
  let m = 0;
  for (let k = 0; k < cpu.length; k++) {
    m = Math.max(m, Math.abs(gpu[k]! - cpu[k]!));
  }
  return m;
}

async function runParity(
  yBoundary: YBoundary,
  steps: number,
): Promise<{ dRho: number; dUx: number; dUy: number }> {
  const cpu = makeCpu(yBoundary);
  const device = await getDevice();
  const gpu = await runGpu(device, cpu.f, yBoundary, steps);
  cpu.step(steps);
  cpu.computeMoments();
  const result = {
    dRho: maxAbsDiff(gpu.rho, cpu.rho),
    dUx: maxAbsDiff(gpu.ux, cpu.ux),
    dUy: maxAbsDiff(gpu.uy, cpu.uy),
  };
  device.destroy();
  return result;
}

describe('GPU kernel parity with the CPU reference (128x64, cylinder)', () => {
  it.each(['free-slip', 'periodic'] as const)(
    'matches after 1 step (%s walls)',
    async (yBoundary) => {
      const { dRho, dUx, dUy } = await runParity(yBoundary, 1);
      // A single step admits only a handful of f32 roundings.
      expect(dRho).toBeLessThan(1e-5);
      expect(dUx).toBeLessThan(1e-5);
      expect(dUy).toBeLessThan(1e-5);
    },
  );

  it.each(['free-slip', 'periodic'] as const)(
    'gate: max abs moment difference < 1e-4 after 500 steps (%s walls)',
    async (yBoundary) => {
      const { dRho, dUx, dUy } = await runParity(yBoundary, 500);
      console.log(
        `parity 500 steps, ${yBoundary}: dRho=${dRho.toExponential(2)} dUx=${dUx.toExponential(2)} dUy=${dUy.toExponential(2)}`,
      );
      expect(dRho).toBeLessThan(1e-4);
      expect(dUx).toBeLessThan(1e-4);
      expect(dUy).toBeLessThan(1e-4);
    },
  );
});
