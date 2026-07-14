// GPU buffer allocation and the params-uniform encoding for the LBM solver.
//
// Distributions use the same SoA layout as the CPU reference:
// f[i * n + y * nx + x], n = nx * ny, two copies (A and B) for ping-pong.
// Macroscopic fields are three separate f32 buffers (rho, ux, uy) -- plain
// storage buffers everywhere, one consistent binding style.

import { equilibrium } from '../solver/cpu-lbm.ts';

/**
 * Byte layout of the WGSL `Params` uniform struct in lbm.wgsl. The struct is
 * scalars-only (u32/f32, each 4-byte aligned) padded to 32 bytes. This table
 * is the TS mirror that tests/params-layout.test.ts verifies byte-for-byte
 * (CLAUDE.md gotcha #5); extend BOTH the WGSL struct and this table in the
 * same commit whenever a field is added, and mind vec2/vec3 alignment rules
 * if vectors ever join the struct.
 */
export const PARAMS_BYTE_SIZE = 32;
export const PARAMS_OFFSETS = {
  nx: 0, // u32
  ny: 4, // u32
  tau: 8, // f32
  inletU: 12, // f32
  flags: 16, // u32
  stepIndex: 20, // u32
  // 24..31: explicit padding to a 16-byte multiple
} as const;

/** Params.flags bit 0: periodic top/bottom walls (otherwise free-slip). */
export const FLAG_PERIODIC_Y = 1;

/**
 * Byte layout of the WGSL `BrushParams` uniform in brush.wgsl. The leading
 * vec2f has 8-byte alignment -- exactly the trap gotcha #5 warns about --
 * so the layout is pinned in tests/params-layout.test.ts alongside Params.
 */
export const BRUSH_PARAMS_BYTE_SIZE = 32;
export const BRUSH_PARAMS_OFFSETS = {
  centerX: 0, // vec2f.x
  centerY: 4, // vec2f.y
  radius: 8, // f32
  mode: 12, // u32 (1 = paint solid, 0 = erase)
  nx: 16, // u32
  ny: 20, // u32
  // 24..31: explicit padding
} as const;

export interface BrushStamp {
  centerX: number;
  centerY: number;
  radius: number;
  paint: boolean;
  nx: number;
  ny: number;
}

export function encodeBrushParams(s: BrushStamp): ArrayBuffer {
  const buf = new ArrayBuffer(BRUSH_PARAMS_BYTE_SIZE);
  const dv = new DataView(buf);
  dv.setFloat32(BRUSH_PARAMS_OFFSETS.centerX, s.centerX, true);
  dv.setFloat32(BRUSH_PARAMS_OFFSETS.centerY, s.centerY, true);
  dv.setFloat32(BRUSH_PARAMS_OFFSETS.radius, s.radius, true);
  dv.setUint32(BRUSH_PARAMS_OFFSETS.mode, s.paint ? 1 : 0, true);
  dv.setUint32(BRUSH_PARAMS_OFFSETS.nx, s.nx, true);
  dv.setUint32(BRUSH_PARAMS_OFFSETS.ny, s.ny, true);
  return buf;
}

export interface SimParams {
  nx: number;
  ny: number;
  tau: number;
  inletU: number;
  periodicY: boolean;
  stepIndex: number;
}

export function encodeParams(p: SimParams): ArrayBuffer {
  const buf = new ArrayBuffer(PARAMS_BYTE_SIZE);
  const dv = new DataView(buf);
  dv.setUint32(PARAMS_OFFSETS.nx, p.nx, true);
  dv.setUint32(PARAMS_OFFSETS.ny, p.ny, true);
  dv.setFloat32(PARAMS_OFFSETS.tau, p.tau, true);
  dv.setFloat32(PARAMS_OFFSETS.inletU, p.inletU, true);
  dv.setUint32(PARAMS_OFFSETS.flags, p.periodicY ? FLAG_PERIODIC_Y : 0, true);
  dv.setUint32(PARAMS_OFFSETS.stepIndex, p.stepIndex, true);
  return buf;
}

export interface LbmBuffers {
  nx: number;
  ny: number;
  /** Cells per field: nx * ny. */
  n: number;
  /** Ping-pong distribution buffers, 9 * n f32 each (post-collision values). */
  f: readonly [GPUBuffer, GPUBuffer];
  /** Obstacle mask, n u32 (0 = fluid, 1 = solid). */
  mask: GPUBuffer;
  /** Mask snapshot from before the latest edit; mask_diff compares. */
  maskPrev: GPUBuffer;
  /** Per-cell changed flags produced by mask_diff. */
  changed: GPUBuffer;
  rho: GPUBuffer;
  ux: GPUBuffer;
  uy: GPUBuffer;
  params: GPUBuffer;
  brushParams: GPUBuffer;
}

export function createLbmBuffers(device: GPUDevice, nx: number, ny: number): LbmBuffers {
  const n = nx * ny;
  const fUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const fieldUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  return {
    nx,
    ny,
    n,
    f: [
      device.createBuffer({ label: 'f-A', size: 9 * n * 4, usage: fUsage }),
      device.createBuffer({ label: 'f-B', size: 9 * n * 4, usage: fUsage }),
    ],
    mask: device.createBuffer({
      label: 'obstacle-mask',
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    maskPrev: device.createBuffer({
      label: 'obstacle-mask-prev',
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    changed: device.createBuffer({
      label: 'mask-changed',
      size: n * 4,
      usage: GPUBufferUsage.STORAGE,
    }),
    rho: device.createBuffer({ label: 'rho', size: n * 4, usage: fieldUsage }),
    ux: device.createBuffer({ label: 'ux', size: n * 4, usage: fieldUsage }),
    uy: device.createBuffer({ label: 'uy', size: n * 4, usage: fieldUsage }),
    params: device.createBuffer({
      label: 'params',
      size: PARAMS_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    brushParams: device.createBuffer({
      label: 'brush-params',
      size: BRUSH_PARAMS_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
  };
}

export function writeParams(device: GPUDevice, buffers: LbmBuffers, params: SimParams): void {
  device.queue.writeBuffer(buffers.params, 0, encodeParams(params));
}

/**
 * Upload an obstacle mask (0/1 per cell) into the u32 mask buffer. With
 * `syncPrev` the snapshot buffer is written too (initial setup, where no
 * reconcile pass should fire); without it, the next mask_diff/mask_reconcile
 * chain sees the difference and re-equilibrates around the edit.
 */
export function writeMask(
  device: GPUDevice,
  buffers: LbmBuffers,
  mask: Uint8Array,
  syncPrev = false,
): void {
  const words = Uint32Array.from(mask);
  device.queue.writeBuffer(buffers.mask, 0, words);
  if (syncPrev) device.queue.writeBuffer(buffers.maskPrev, 0, words);
}

export function writeBrushParams(device: GPUDevice, buffers: LbmBuffers, s: BrushStamp): void {
  device.queue.writeBuffer(buffers.brushParams, 0, encodeBrushParams(s));
}

/**
 * Upload distributions (e.g. the CPU reference's Float64 state) into f[which],
 * converting to f32.
 */
export function writeDistributions(
  device: GPUDevice,
  buffers: LbmBuffers,
  which: 0 | 1,
  f: Float64Array,
): void {
  device.queue.writeBuffer(buffers.f[which], 0, new Float32Array(f));
}

/**
 * Reset the flow: fill f[which] with the equilibrium of a uniform state and
 * write the matching moment fields. The moment buffers must be valid before
 * the first solver step -- both the render pass and mask_reconcile read them.
 */
export function writeUniformEquilibrium(
  device: GPUDevice,
  buffers: LbmBuffers,
  which: 0 | 1,
  rho0: number,
  ux0: number,
  uy0: number,
): void {
  const { n } = buffers;
  const f = new Float32Array(9 * n);
  for (let i = 0; i < 9; i++) {
    f.fill(equilibrium(i, rho0, ux0, uy0), i * n, (i + 1) * n);
  }
  device.queue.writeBuffer(buffers.f[which], 0, f);
  const field = new Float32Array(n);
  device.queue.writeBuffer(buffers.rho, 0, field.fill(rho0));
  device.queue.writeBuffer(buffers.ux, 0, field.fill(ux0));
  device.queue.writeBuffer(buffers.uy, 0, field.fill(uy0));
}
