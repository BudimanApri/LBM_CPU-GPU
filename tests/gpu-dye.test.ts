import { describe, expect, it } from 'vitest';
import { createLbmBuffers, writeParams, type SimParams } from '../src/gpu/buffers.ts';
import { createDyePipeline } from '../src/gpu/pipelines.ts';
import { assertShaderCompiles } from './gpu-test-utils.ts';

const NX = 16;
const NY = 8;
const N = NX * NY;

async function getDevice(): Promise<GPUDevice> {
  const adapter = await navigator.gpu.requestAdapter();
  expect(adapter).not.toBeNull();
  return adapter!.requestDevice();
}

const baseParams = (dyeEnabled: boolean, substeps = 1): SimParams => ({
  nx: NX,
  ny: NY,
  tau: 0.6,
  inletU: 0.05,
  periodicY: false,
  dyeEnabled,
  viewMode: 'dye',
  stepIndex: 0,
  substeps,
});

async function runDyeSteps(
  device: GPUDevice,
  dyeEnabled: boolean,
  ux: number[],
  uy: number[],
  mask: number[],
  dye0: number[],
  steps: number,
  substeps = 1,
): Promise<Float32Array> {
  const buffers = createLbmBuffers(device, NX, NY);
  writeParams(device, buffers, baseParams(dyeEnabled, substeps));
  device.queue.writeBuffer(buffers.ux, 0, new Float32Array(ux));
  device.queue.writeBuffer(buffers.uy, 0, new Float32Array(uy));
  device.queue.writeBuffer(buffers.mask, 0, Uint32Array.from(mask));
  device.queue.writeBuffer(buffers.dye[0], 0, new Float32Array(dye0));

  const dye = createDyePipeline(device, buffers);
  await assertShaderCompiles(dye.module);

  const encoder = device.createCommandEncoder();
  for (let s = 0; s < steps; s++) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(dye.pipeline);
    pass.setBindGroup(0, dye.bindGroups[s % 2]);
    pass.dispatchWorkgroups(dye.workgroupsX, dye.workgroupsY);
    pass.end();
  }
  const finalBuf = buffers.dye[steps % 2]!;
  const staging = device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(finalBuf, 0, staging, 0, N * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  return new Float32Array(staging.getMappedRange().slice(0));
}

describe('dye.wgsl semi-Lagrangian advection', () => {
  it('dissipates by exactly 0.999^steps with zero velocity, no emitters', async () => {
    const device = await getDevice();
    const dye0 = new Array<number>(N).fill(0);
    dye0[3 * NX + 5] = 0.8;
    const result = await runDyeSteps(
      device,
      false,
      new Array<number>(N).fill(0),
      new Array<number>(N).fill(0),
      new Array<number>(N).fill(0),
      dye0,
      10,
    );
    expect(result[3 * NX + 5]).toBeCloseTo(0.8 * 0.999 ** 10, 6);
    // Untouched cells stay at zero.
    expect(result[0]).toBe(0);
    device.destroy();
  });

  it('advects a dye spike downstream by the local velocity each step', async () => {
    const device = await getDevice();
    const ux = new Array<number>(N).fill(0.5); // 0.5 cells/step, exact half-integer shift
    const uy = new Array<number>(N).fill(0);
    const dye0 = new Array<number>(N).fill(0);
    dye0[3 * NX + 5] = 1.0;
    // Back-trace pulls from upstream (x - u), so the peak intensity moves
    // downstream (+x) over time as neighboring cells sample the spike.
    const result = await runDyeSteps(device, false, ux, uy, new Array<number>(N).fill(0), dye0, 1);
    // At x=6 (one cell downstream), back-trace samples x=5.5 -> bilinear
    // average of cell 5 (1.0) and cell 6 (0.0) = 0.5, times dissipation.
    expect(result[3 * NX + 6]).toBeCloseTo(0.5 * 0.999, 5);
    // The spike's own cell back-traces to x=4.5 -> average of cell 4 (0.0)
    // and cell 5 itself (1.0) = 0.5 too (both neighbors of the spike see
    // half its intensity after a half-cell shift), same dissipation.
    expect(result[3 * NX + 5]).toBeCloseTo(0.5 * 0.999, 5);
    // Two cells downstream sees none of the spike yet.
    expect(result[3 * NX + 7]).toBeCloseTo(0, 6);
    device.destroy();
  });

  it('scales the back-trace and dissipation by the substep count K', async () => {
    const device = await getDevice();
    const ux = new Array<number>(N).fill(0.25); // K=4 -> exactly one cell per dispatch
    const uy = new Array<number>(N).fill(0);
    const dye0 = new Array<number>(N).fill(0);
    dye0[3 * NX + 5] = 1.0;
    const result = await runDyeSteps(
      device,
      false,
      ux,
      uy,
      new Array<number>(N).fill(0),
      dye0,
      1,
      4,
    );
    // One cell downstream back-traces 4 * 0.25 = 1.0 cells, landing exactly
    // on the spike; dissipation compounds once per covered substep.
    expect(result[3 * NX + 6]).toBeCloseTo(0.999 ** 4, 5);
    // The spike's own cell back-traces to empty x=4.
    expect(result[3 * NX + 5]).toBeCloseTo(0, 6);
    device.destroy();
  });

  it('forces dye to 1.0 in emitter bands at the emitter column when enabled', async () => {
    const device = await getDevice();
    const result = await runDyeSteps(
      device,
      true,
      new Array<number>(N).fill(0),
      new Array<number>(N).fill(0),
      new Array<number>(N).fill(0),
      new Array<number>(N).fill(0),
      1,
    );
    // period = max(1, 8/5) = 1, band_height = max(1, 1/2) = 1 -> every row
    // in the emitter column is forced to 1 (degenerate but deterministic
    // for this small NY; the important check is the column itself).
    for (let y = 0; y < NY; y++) {
      expect(result[y * NX + 1]).toBe(1);
    }
    // Off the emitter column, nothing was injected (stays at the
    // dissipated advected value, here 0).
    expect(result[3 * NX + 5]).toBe(0);
    device.destroy();
  });

  it('does not force dye when disabled, even in the emitter column', async () => {
    const device = await getDevice();
    const result = await runDyeSteps(
      device,
      false,
      new Array<number>(N).fill(0),
      new Array<number>(N).fill(0),
      new Array<number>(N).fill(0),
      new Array<number>(N).fill(0),
      1,
    );
    for (let y = 0; y < NY; y++) {
      expect(result[y * NX + 1]).toBe(0);
    }
    device.destroy();
  });

  it('always reports zero dye in solid cells', async () => {
    const device = await getDevice();
    const mask = new Array<number>(N).fill(0);
    mask[3 * NX + 5] = 1;
    const dye0 = new Array<number>(N).fill(0);
    dye0[3 * NX + 5] = 1.0; // stale dye in a cell that just became solid
    const result = await runDyeSteps(
      device,
      false,
      new Array<number>(N).fill(0),
      new Array<number>(N).fill(0),
      mask,
      dye0,
      1,
    );
    expect(result[3 * NX + 5]).toBe(0);
    device.destroy();
  });
});
