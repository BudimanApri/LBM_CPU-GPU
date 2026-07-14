import { describe, expect, it } from 'vitest';
import {
  createLbmBuffers,
  writeDistributions,
  writeMask,
  writeParams,
} from '../src/gpu/buffers.ts';
import { LBM_WORKGROUP_CANDIDATES, createLbmPipeline } from '../src/gpu/pipelines.ts';
import { equilibrium } from '../src/solver/cpu-lbm.ts';
import { assertShaderCompiles } from './gpu-test-utils.ts';

const NX = 256;
const NY = 128;
const N = NX * NY;
const STEPS = 3000;

async function getDevice(): Promise<GPUDevice> {
  const adapter = await navigator.gpu.requestAdapter();
  expect(adapter).not.toBeNull();
  return adapter!.requestDevice();
}

function pressurePulse(): Float64Array {
  const f = new Float64Array(9 * N);
  for (let y = 0; y < NY; y++) {
    for (let x = 0; x < NX; x++) {
      const dx = x - NX * 0.38;
      const dy = y - NY * 0.5;
      const rho = 1 + 0.02 * Math.exp(-(dx * dx + dy * dy) / 180);
      const cell = y * NX + x;
      for (let i = 0; i < 9; i++) f[i * N + cell] = equilibrium(i, rho, 0.05, 0);
    }
  }
  return f;
}

async function run(device: GPUDevice, sponge: boolean): Promise<Float32Array> {
  const buffers = createLbmBuffers(device, NX, NY);
  writeParams(device, buffers, {
    nx: NX,
    ny: NY,
    tau: 0.56,
    inletU: 0.05,
    periodicY: false,
    dyeEnabled: false,
    viewMode: 'density',
    stepIndex: 0,
    substeps: 1,
    spongeOutlet: sponge ? 0.04 : 0,
    spongeWall: sponge ? 0.02 : 0,
  });
  writeMask(device, buffers, new Uint8Array(N), true);
  writeDistributions(device, buffers, 0, pressurePulse());
  const lbm = createLbmPipeline(device, buffers, LBM_WORKGROUP_CANDIDATES[2]);
  await assertShaderCompiles(lbm.module);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(lbm.pipeline);
  for (let step = 0; step < STEPS; step++) {
    pass.setBindGroup(0, lbm.bindGroups[step % 2]);
    pass.dispatchWorkgroups(lbm.workgroupsX, lbm.workgroupsY);
  }
  pass.end();
  const staging = device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(buffers.rho, 0, staging, 0, N * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const rho = new Float32Array(staging.getMappedRange().slice(0));
  staging.destroy();
  return rho;
}

function coreRms(rho: Float32Array): number {
  let squared = 0;
  let count = 0;
  // Exclude the sponge cells themselves: this measures reflected energy
  // returning to the useful tunnel core, not the trivially damped fringe.
  for (let y = 12; y < NY - 12; y++) {
    for (let x = 12; x < Math.floor(NX * 0.82); x++) {
      const dev = rho[y * NX + x]! - 1;
      squared += dev * dev;
      count++;
    }
  }
  return Math.sqrt(squared / count);
}

describe('acoustic sponge fringe', () => {
  it('removes a pressure pulse instead of reflecting it into the tunnel core', async () => {
    const device = await getDevice();
    const undamped = coreRms(await run(device, false));
    const damped = coreRms(await run(device, true));
    console.log(
      `sponge acoustic RMS: undamped=${undamped.toExponential(3)}, damped=${damped.toExponential(3)}`,
    );
    expect(damped).toBeLessThan(undamped * 0.6);
    device.destroy();
  });
});
