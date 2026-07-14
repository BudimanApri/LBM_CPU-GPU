import { describe, expect, it } from 'vitest';
import {
  createLbmBuffers,
  writeMask,
  writeParams,
  writeUniformEquilibrium,
} from '../src/gpu/buffers.ts';
import {
  LBM_WORKGROUP_CANDIDATES,
  benchmarkLbmWorkgroups,
  createLbmPipeline,
  createSentinelPipeline,
} from '../src/gpu/pipelines.ts';
import { nacaPreset } from '../src/geometry/presets.ts';
import { solveTauForRe } from '../src/solver/units.ts';
import { assertShaderCompiles } from './gpu-test-utils.ts';

async function getDevice(): Promise<GPUDevice> {
  const adapter = await navigator.gpu.requestAdapter();
  expect(adapter).not.toBeNull();
  return adapter!.requestDevice();
}

function params(nx: number, ny: number, tau = 0.51) {
  return {
    nx,
    ny,
    tau,
    inletU: 0.05,
    periodicY: false,
    dyeEnabled: false,
    viewMode: 'vorticity' as const,
    stepIndex: 0,
    substeps: 3,
    lesEnabled: true,
    smagorinskyCs: 0.1,
    spongeOutlet: 0.04,
    spongeWall: 0.02,
  };
}

async function readBuffer(
  device: GPUDevice,
  source: GPUBuffer,
  size: number,
): Promise<ArrayBuffer> {
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = staging.getMappedRange().slice(0);
  staging.destroy();
  return result;
}

describe('Phase 6 GPU robustness', () => {
  it('compiles and completion-times every LBM workgroup specialization', async () => {
    const device = await getDevice();
    const buffers = createLbmBuffers(device, 64, 32);
    writeParams(device, buffers, params(64, 32, 0.6));
    writeMask(device, buffers, new Uint8Array(buffers.n), true);
    writeUniformEquilibrium(device, buffers, 0, 1, 0.05, 0);

    for (const workgroup of LBM_WORKGROUP_CANDIDATES) {
      const pipeline = createLbmPipeline(device, buffers, workgroup);
      await assertShaderCompiles(pipeline.module);
    }
    const benchmark = await benchmarkLbmWorkgroups(device, buffers, 4);
    expect(benchmark.results).toHaveLength(3);
    for (const result of benchmark.results) {
      expect(result.millisecondsPerStep).toBeGreaterThan(0);
      expect(Number.isFinite(result.mlups)).toBe(true);
    }
    device.destroy();
  });

  it('sets the sentinel flag for NaN density and leaves finite fields clear', async () => {
    const device = await getDevice();
    const buffers = createLbmBuffers(device, 16, 8);
    writeParams(device, buffers, params(16, 8));
    const sentinel = createSentinelPipeline(device, buffers);
    await assertShaderCompiles(sentinel.module);

    const run = async (rho: Float32Array): Promise<number> => {
      device.queue.writeBuffer(buffers.rho, 0, rho);
      const encoder = device.createCommandEncoder();
      encoder.clearBuffer(buffers.nanFlag);
      const pass = encoder.beginComputePass();
      pass.setPipeline(sentinel.pipeline);
      pass.setBindGroup(0, sentinel.bindGroup);
      pass.dispatchWorkgroups(sentinel.workgroups);
      pass.end();
      device.queue.submit([encoder.finish()]);
      return new Uint32Array(await readBuffer(device, buffers.nanFlag, 4))[0]!;
    };

    expect(await run(new Float32Array(buffers.n).fill(1))).toBe(0);
    const broken = new Float32Array(buffers.n).fill(1);
    broken[37] = NaN;
    expect(await run(broken)).toBe(1);
    device.destroy();
  });

  it(
    'keeps a Re=5000 NACA 4412 at 10 degrees finite with LES enabled',
    { timeout: 60_000 },
    async () => {
      const nx = 256;
      const ny = 128;
      const preset = nacaPreset(nx, ny, '4412', 49, 10);
      const tau = solveTauForRe(5000, 0.05, preset.d).tau;
      expect(tau).toBe(0.51);

      const device = await getDevice();
      const buffers = createLbmBuffers(device, nx, ny);
      writeParams(device, buffers, params(nx, ny, tau));
      writeMask(device, buffers, preset.mask, true);
      writeUniformEquilibrium(device, buffers, 0, 1, 0.05, 0, 0.05);
      const lbm = createLbmPipeline(device, buffers, LBM_WORKGROUP_CANDIDATES[2]);

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(lbm.pipeline);
      for (let step = 0; step < 20_000; step++) {
        pass.setBindGroup(0, lbm.bindGroups[step % 2]);
        pass.dispatchWorkgroups(lbm.workgroupsX, lbm.workgroupsY);
      }
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();

      const rho = new Float32Array(await readBuffer(device, buffers.rho, buffers.n * 4));
      let min = Infinity;
      let max = -Infinity;
      for (let cell = 0; cell < rho.length; cell++) {
        expect(Number.isFinite(rho[cell]!)).toBe(true);
        if (preset.mask[cell] === 0) {
          min = Math.min(min, rho[cell]!);
          max = Math.max(max, rho[cell]!);
        }
      }
      expect(min).toBeGreaterThan(0.7);
      expect(max).toBeLessThan(1.3);
      device.destroy();
    },
  );
});
