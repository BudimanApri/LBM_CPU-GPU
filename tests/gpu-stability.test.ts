// Phase 3 gate: cylinder at Re=20 develops steady, symmetric twin
// recirculation vortices and stays stable over 100k steps (no blow-up).
//
// Domain 256x128 with D=16 keeps the blockage ratio at 1/8 (ny = 8D), the
// bound Phase 5's Cd validation needs. The cylinder is centered on the
// half-integer row (ny-1)/2 = 63.5 so the geometry -- and therefore the
// steady Re=20 wake -- is exactly mirror-symmetric between rows y and
// 127-y.

import { describe, expect, it } from 'vitest';
import { solveTauForRe } from '../src/solver/units.ts';
import { equilibrium } from '../src/solver/cpu-lbm.ts';
import { rasterizeCircle } from '../src/geometry/presets.ts';
import { createLbmBuffers, writeMask, writeParams } from '../src/gpu/buffers.ts';
import { createLbmPipeline } from '../src/gpu/pipelines.ts';

const NX = 256;
const NY = 128;
const N = NX * NY;
const D = 16;
const U0 = 0.05;
const RE = 20;
const CX0 = 64; // 25% chord
const CY0 = (NY - 1) / 2;
const TOTAL_STEPS = 100_000;
const CHUNK = 2_000;

async function readField(device: GPUDevice, src: GPUBuffer): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(src, 0, staging, 0, N * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(staging.getMappedRange().slice(0));
  staging.destroy();
  return data;
}

describe('Re=20 cylinder long-run stability (Phase 3 gate)', () => {
  it(
    'holds steady symmetric twin recirculation over 100k steps',
    { timeout: 120_000 },
    async () => {
      const tauSolution = solveTauForRe(RE, U0, D);
      expect(tauSolution.clamped).toBe(false); // Re=20 must be exactly realizable

      const adapter = await navigator.gpu.requestAdapter();
      expect(adapter).not.toBeNull();
      const device = await adapter!.requestDevice();

      const mask = new Uint8Array(N);
      rasterizeCircle(mask, NX, NY, CX0, CY0, D / 2);
      const buffers = createLbmBuffers(device, NX, NY);
      writeParams(device, buffers, {
        nx: NX,
        ny: NY,
        tau: tauSolution.tau,
        inletU: U0,
        periodicY: false,
        dyeEnabled: false,
        viewMode: 'velocity',
        stepIndex: 0,
        substeps: 1,
      });
      writeMask(device, buffers, mask, true);
      const f = new Float32Array(9 * N);
      for (let i = 0; i < 9; i++) f.fill(equilibrium(i, 1, U0, 0), i * N, (i + 1) * N);
      device.queue.writeBuffer(buffers.f[0], 0, f);

      const lbm = createLbmPipeline(device, buffers);
      let step = 0;
      const runChunk = (count: number): void => {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(lbm.pipeline);
        for (let s = 0; s < count; s++) {
          pass.setBindGroup(0, lbm.bindGroups[step % 2]);
          pass.dispatchWorkgroups(lbm.workgroupsX, lbm.workgroupsY);
          step++;
        }
        pass.end();
        device.queue.submit([encoder.finish()]);
      };

      const t0 = performance.now();
      while (step < 90_000) {
        runChunk(CHUNK);
        await device.queue.onSubmittedWorkDone();
      }
      const ux90k = await readField(device, buffers.ux);
      while (step < TOTAL_STEPS) {
        runChunk(CHUNK);
        await device.queue.onSubmittedWorkDone();
      }
      const elapsed = (performance.now() - t0) / 1000;
      const mlups = (N * TOTAL_STEPS) / elapsed / 1e6;

      const rho = await readField(device, buffers.rho);
      const ux = await readField(device, buffers.ux);
      const uy = await readField(device, buffers.uy);

      // -- no blow-up: everything finite, density bounded near 1 --
      let rhoMin = Infinity;
      let rhoMax = -Infinity;
      for (let k = 0; k < N; k++) {
        const r = rho[k]!;
        expect(Number.isFinite(r)).toBe(true);
        if (mask[k] === 0) {
          rhoMin = Math.min(rhoMin, r);
          rhoMax = Math.max(rhoMax, r);
        }
      }
      expect(rhoMin).toBeGreaterThan(0.8);
      expect(rhoMax).toBeLessThan(1.2);

      // -- twin recirculation: reversed flow in the near wake --
      let uxMin = Infinity;
      for (let y = 56; y <= 71; y++) {
        for (let x = 73; x <= 113; x++) {
          if (mask[y * NX + x] === 0) uxMin = Math.min(uxMin, ux[y * NX + x]!);
        }
      }
      console.log(
        `Re=20 gate: wake ux_min=${uxMin.toExponential(2)}, rho in [${rhoMin.toFixed(4)}, ${rhoMax.toFixed(4)}], ${mlups.toFixed(0)} MLUPS`,
      );
      expect(uxMin).toBeLessThan(-1e-4);

      // -- steady & symmetric: the Re=20 wake neither oscillates nor tilts --
      let maxDrift = 0;
      for (let k = 0; k < N; k++) {
        maxDrift = Math.max(maxDrift, Math.abs(ux[k]! - ux90k[k]!));
      }
      expect(maxDrift).toBeLessThan(1e-5);

      let maxAsym = 0;
      for (let y = 0; y < NY / 2; y++) {
        for (let x = 0; x < NX; x++) {
          const mirror = (NY - 1 - y) * NX + x;
          if (mask[y * NX + x] !== 0) continue;
          maxAsym = Math.max(
            maxAsym,
            Math.abs(ux[y * NX + x]! - ux[mirror]!),
            Math.abs(uy[y * NX + x]! + uy[mirror]!),
          );
        }
      }
      console.log(
        `Re=20 gate: steadiness drift=${maxDrift.toExponential(2)}, asymmetry=${maxAsym.toExponential(2)}`,
      );
      expect(maxAsym).toBeLessThan(1e-3);

      device.destroy();
    },
  );
});
