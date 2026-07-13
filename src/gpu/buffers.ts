// GPU buffer allocation and the params-uniform encoding for the LBM solver.
//
// Distributions use the same SoA layout as the CPU reference:
// f[i * n + y * nx + x], n = nx * ny, two copies (A and B) for ping-pong.
// Macroscopic fields are three separate f32 buffers (rho, ux, uy) -- plain
// storage buffers everywhere, one consistent binding style.

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
  rho: GPUBuffer;
  ux: GPUBuffer;
  uy: GPUBuffer;
  params: GPUBuffer;
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
    rho: device.createBuffer({ label: 'rho', size: n * 4, usage: fieldUsage }),
    ux: device.createBuffer({ label: 'ux', size: n * 4, usage: fieldUsage }),
    uy: device.createBuffer({ label: 'uy', size: n * 4, usage: fieldUsage }),
    params: device.createBuffer({
      label: 'params',
      size: PARAMS_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
  };
}

export function writeParams(device: GPUDevice, buffers: LbmBuffers, params: SimParams): void {
  device.queue.writeBuffer(buffers.params, 0, encodeParams(params));
}

/** Upload an obstacle mask (0/1 per cell) into the u32 mask buffer. */
export function writeMask(device: GPUDevice, buffers: LbmBuffers, mask: Uint8Array): void {
  device.queue.writeBuffer(buffers.mask, 0, Uint32Array.from(mask));
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
