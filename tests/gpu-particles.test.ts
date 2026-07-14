import { describe, expect, it } from 'vitest';
import { createLbmBuffers, writeParams, type SimParams } from '../src/gpu/buffers.ts';
import { createParticlePipelines } from '../src/gpu/pipelines.ts';
import { assertShaderCompiles } from './gpu-test-utils.ts';

const NX = 32;
const NY = 16;
const N = NX * NY;

async function getDevice(): Promise<GPUDevice> {
  const adapter = await navigator.gpu.requestAdapter();
  expect(adapter).not.toBeNull();
  return adapter!.requestDevice();
}

const baseParams = (stepIndex = 0): SimParams => ({
  nx: NX,
  ny: NY,
  tau: 0.6,
  inletU: 0.05,
  periodicY: false,
  dyeEnabled: false,
  viewMode: 'velocity',
  stepIndex,
});

async function runParticleSteps(
  device: GPUDevice,
  ux: number[],
  uy: number[],
  mask: number[],
  positions: number[],
  steps: number,
): Promise<Float32Array> {
  const buffers = createLbmBuffers(device, NX, NY);
  writeParams(device, buffers, baseParams(steps));
  device.queue.writeBuffer(buffers.ux, 0, new Float32Array(ux));
  device.queue.writeBuffer(buffers.uy, 0, new Float32Array(uy));
  device.queue.writeBuffer(buffers.mask, 0, Uint32Array.from(mask));
  device.queue.writeBuffer(buffers.particles, 0, new Float32Array(positions));

  const particleCount = positions.length / 2;
  const particles = createParticlePipelines(device, buffers, 'rgba8unorm');
  await assertShaderCompiles(particles.module);

  const encoder = device.createCommandEncoder();
  for (let s = 0; s < steps; s++) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(particles.computePipeline);
    pass.setBindGroup(0, particles.computeBindGroup);
    pass.dispatchWorkgroups(Math.ceil(particleCount / 64));
    pass.end();
  }
  const staging = device.createBuffer({
    size: positions.length * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(buffers.particles, 0, staging, 0, positions.length * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  return new Float32Array(staging.getMappedRange().slice(0));
}

describe('particles.wgsl RK2 advection', () => {
  it('moves exactly by the uniform velocity per step (RK2 exact for constant fields)', async () => {
    const device = await getDevice();
    const ux = new Array<number>(N).fill(0.2);
    const uy = new Array<number>(N).fill(0.1);
    const positions = [10, 6]; // well clear of every edge
    const result = await runParticleSteps(
      device,
      ux,
      uy,
      new Array<number>(N).fill(0),
      positions,
      3,
    );
    expect(result[0]).toBeCloseTo(10 + 3 * 0.2, 5);
    expect(result[1]).toBeCloseTo(6 + 3 * 0.1, 5);
    device.destroy();
  });

  it('respawns at x=0.5 when advected out of bounds', async () => {
    const device = await getDevice();
    const ux = new Array<number>(N).fill(5); // guarantees an out-of-bounds step
    const uy = new Array<number>(N).fill(0);
    const positions = [NX - 2, 8];
    const result = await runParticleSteps(
      device,
      ux,
      uy,
      new Array<number>(N).fill(0),
      positions,
      1,
    );
    expect(result[0]).toBeCloseTo(0.5, 5);
    expect(result[1]).toBeGreaterThanOrEqual(0);
    expect(result[1]).toBeLessThan(NY);
    device.destroy();
  });

  it('respawns when advected into a solid cell', async () => {
    const device = await getDevice();
    const mask = new Array<number>(N).fill(0);
    mask[8 * NX + 16] = 1; // directly in the particle's path
    const ux = new Array<number>(N).fill(1);
    const uy = new Array<number>(N).fill(0);
    const positions = [15, 8];
    const result = await runParticleSteps(device, ux, uy, mask, positions, 1);
    expect(result[0]).toBeCloseTo(0.5, 5);
    device.destroy();
  });

  it('does not respawn a particle advecting through open fluid', async () => {
    const device = await getDevice();
    const ux = new Array<number>(N).fill(0.3);
    const uy = new Array<number>(N).fill(0);
    const positions = [10, 8];
    const result = await runParticleSteps(
      device,
      ux,
      uy,
      new Array<number>(N).fill(0),
      positions,
      1,
    );
    expect(result[0]).toBeCloseTo(10.3, 5);
    device.destroy();
  });

  it('hashes distinct respawn y-values across particles and steps (no stacking)', async () => {
    const device = await getDevice();
    const count = 64;
    const positions: number[] = [];
    for (let i = 0; i < count; i++) positions.push(NX + 1, 8); // all out of bounds immediately
    const ux = new Array<number>(N).fill(0);
    const uy = new Array<number>(N).fill(0);
    const result = await runParticleSteps(
      device,
      ux,
      uy,
      new Array<number>(N).fill(0),
      positions,
      1,
    );
    const ys = new Set<number>();
    for (let i = 0; i < count; i++) ys.add(Math.round(result[2 * i + 1]! * 1000));
    expect(ys.size).toBeGreaterThan(count / 2); // overwhelmingly distinct
    device.destroy();
  });
});

describe('particles.wgsl render pipeline', () => {
  it('draws a bright, alpha-blended dot at the particle position, transparent elsewhere', async () => {
    const device = await getDevice();
    const buffers = createLbmBuffers(device, NX, NY);
    writeParams(device, buffers, baseParams());
    device.queue.writeBuffer(buffers.ux, 0, new Float32Array(N).fill(0.05));
    device.queue.writeBuffer(buffers.uy, 0, new Float32Array(N));
    device.queue.writeBuffer(buffers.mask, 0, new Uint32Array(N));
    // One particle, centered in the domain.
    device.queue.writeBuffer(buffers.particles, 0, new Float32Array([NX / 2, NY / 2]));

    // The quad's half-size is a fixed NDC constant tuned to read as a small
    // dot at the real ~1024x512 canvas; render into a comparably large
    // target here so it isn't sub-pixel (the lattice fields above stay at
    // the tiny NX/NY -- only the render target's pixel resolution matters
    // for the quad's on-screen size).
    const format = 'rgba8unorm';
    // The quad's NDC half-size (0.0018) is tuned to read as a small ~2px
    // dot at the real ~1024px-wide canvas -- at that scale it's meant to
    // be tiny. A render target has to be proportionally large here for
    // the dot to cover several pixels and survive readback reliably.
    const RENDER_W = 3000;
    const RENDER_H = 1500;
    const bytesPerRow = Math.ceil((RENDER_W * 4) / 256) * 256;
    const particles = createParticlePipelines(device, buffers, format);
    const tex = device.createTexture({
      size: [RENDER_W, RENDER_H],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: tex.createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        },
      ],
    });
    pass.setPipeline(particles.renderPipeline);
    pass.setBindGroup(0, particles.renderBindGroup);
    pass.draw(6, 1);
    pass.end();

    const buf = device.createBuffer({
      size: bytesPerRow * RENDER_H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow }, [
      RENDER_W,
      RENDER_H,
      1,
    ]);
    device.queue.submit([encoder.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(buf.getMappedRange().slice(0));

    // The particle sits at the exact center of the lattice, which maps to
    // the exact center of NDC regardless of any y-flip convention -- check
    // a small neighborhood (robust to sub-pixel rounding of the quad edge)
    // for the brightest alpha, and a far corner for full transparency.
    let maxAlpha = 0;
    let maxAt = [-1, -1];
    for (let y = 0; y < RENDER_H; y++) {
      for (let x = 0; x < RENDER_W; x++) {
        const a = data[y * bytesPerRow + x * 4 + 3]!;
        if (a > maxAlpha) {
          maxAlpha = a;
          maxAt = [x, y];
        }
      }
    }
    console.log('particles render debug: maxAlpha', maxAlpha, 'at', maxAt, 'expected center', [
      RENDER_W / 2,
      RENDER_H / 2,
    ]);
    const cornerAlpha = data[3]!;
    expect(maxAlpha).toBeGreaterThan(50);
    expect(cornerAlpha).toBe(0);
    device.destroy();
  });
});
