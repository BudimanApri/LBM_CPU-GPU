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
  substeps: 24, // u32 -- K, so once-per-frame passes scale advection to match
  // 28..31: explicit padding to a 16-byte multiple
} as const;

/** Params.flags bit 0: periodic top/bottom walls (otherwise free-slip). */
export const FLAG_PERIODIC_Y = 1;
/** Params.flags bit 1: dye advection pass runs and emitters inject. */
export const FLAG_DYE_ENABLED = 2;
/** Params.flags bits 4-5: the render view mode (2 bits, 4 values). */
export const VIEW_MODE_SHIFT = 4;
export const VIEW_MODE_MASK = 0b11;

export type ViewMode = 'velocity' | 'vorticity' | 'density' | 'dye';
const VIEW_MODE_CODES: Record<ViewMode, number> = {
  velocity: 0,
  vorticity: 1,
  density: 2,
  dye: 3,
};

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
  dyeEnabled: boolean;
  viewMode: ViewMode;
  stepIndex: number;
  /** Solver substeps per rendered frame (K); dye/particles scale by this. */
  substeps: number;
}

export function encodeParams(p: SimParams): ArrayBuffer {
  const buf = new ArrayBuffer(PARAMS_BYTE_SIZE);
  const dv = new DataView(buf);
  dv.setUint32(PARAMS_OFFSETS.nx, p.nx, true);
  dv.setUint32(PARAMS_OFFSETS.ny, p.ny, true);
  dv.setFloat32(PARAMS_OFFSETS.tau, p.tau, true);
  dv.setFloat32(PARAMS_OFFSETS.inletU, p.inletU, true);
  const flags =
    (p.periodicY ? FLAG_PERIODIC_Y : 0) |
    (p.dyeEnabled ? FLAG_DYE_ENABLED : 0) |
    (VIEW_MODE_CODES[p.viewMode] << VIEW_MODE_SHIFT);
  dv.setUint32(PARAMS_OFFSETS.flags, flags, true);
  dv.setUint32(PARAMS_OFFSETS.stepIndex, p.stepIndex, true);
  dv.setUint32(PARAMS_OFFSETS.substeps, p.substeps, true);
  return buf;
}

/** Tracer particle count, per CLAUDE.md's "~20k particles" target. */
export const PARTICLE_COUNT = 20_000;

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
  /** Dye density ping-pong pair, n f32 each. */
  dye: readonly [GPUBuffer, GPUBuffer];
  /** Tracer positions, PARTICLE_COUNT * vec2f (8 bytes each). */
  particles: GPUBuffer;
  /** One vec2f partial per force-reduction workgroup (stage-1 output). */
  forcePartials: GPUBuffer;
  /** Reduced total momentum-exchange force (Fx, Fy), one vec2f. */
  forceResult: GPUBuffer;
  params: GPUBuffer;
  brushParams: GPUBuffer;
}

/**
 * Force-reduction workgroup grid: one thread per cell in 8x8 tiles, matching
 * forces.wgsl's @workgroup_size. Stage 1 emits one partial per workgroup, so
 * this also sizes the partials buffer. Kept here so the host allocation and
 * the shader's num_workgroups indexing can never disagree.
 */
export function forceWorkgroups(nx: number, ny: number): { x: number; y: number } {
  return { x: Math.ceil(nx / 8), y: Math.ceil(ny / 8) };
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
      // COPY_SRC so the debounced D (frontal height) readback on brush edits
      // can snapshot the mask (Phase 5); presets/clear supply D CPU-side.
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
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
    dye: [
      device.createBuffer({ label: 'dye-A', size: n * 4, usage: fieldUsage }),
      device.createBuffer({ label: 'dye-B', size: n * 4, usage: fieldUsage }),
    ],
    particles: device.createBuffer({
      label: 'particles',
      size: PARTICLE_COUNT * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    }),
    forcePartials: device.createBuffer({
      label: 'force-partials',
      size: forceWorkgroups(nx, ny).x * forceWorkgroups(nx, ny).y * 8, // vec2f
      usage: GPUBufferUsage.STORAGE,
    }),
    forceResult: device.createBuffer({
      label: 'force-result',
      size: 8, // one vec2<f32> (Fx, Fy)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    }),
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
 * Scatter tracer particles uniformly at random across the domain. Particles
 * seeded inside an obstacle self-correct on the first compute step (their
 * post-step position is still inside the solid, which the respawn check
 * already treats as a hit), so the mask doesn't need consulting here.
 */
export function seedParticlesRandom(device: GPUDevice, buffers: LbmBuffers): void {
  const positions = new Float32Array(PARTICLE_COUNT * 2);
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    positions[2 * i] = Math.random() * buffers.nx;
    positions[2 * i + 1] = Math.random() * buffers.ny;
  }
  device.queue.writeBuffer(buffers.particles, 0, positions);
}

/**
 * Reset the flow: fill f[which] with the equilibrium of a uniform state and
 * write the matching moment fields. The moment buffers must be valid before
 * the first solver step -- both the render pass and mask_reconcile read them.
 *
 * `transversePerturbation` (fraction of ux0, typically ~0.05) adds a gentle
 * sinusoidal uy component that breaks the top-bottom symmetry of the start.
 * A perfectly symmetric impulsive start only sheds vortices once f32
 * roundoff amplifies -- tens of thousands of steps -- while a small explicit
 * seed starts the physical Karman instability within a couple of shedding
 * periods. The wave is weak and viscously damped; it does not change the
 * developed flow, only how fast it appears.
 */
export function writeUniformEquilibrium(
  device: GPUDevice,
  buffers: LbmBuffers,
  which: 0 | 1,
  rho0: number,
  ux0: number,
  uy0: number,
  transversePerturbation = 0,
): void {
  const { n, nx } = buffers;
  const f = new Float32Array(9 * n);
  const uyField = new Float32Array(n);
  if (transversePerturbation === 0) {
    for (let i = 0; i < 9; i++) {
      f.fill(equilibrium(i, rho0, ux0, uy0), i * n, (i + 1) * n);
    }
    uyField.fill(uy0);
  } else {
    const amp = transversePerturbation * ux0;
    const waveNumber = (2 * Math.PI) / (nx / 4);
    // uy varies only with x: precompute one row of per-column equilibria
    // and stamp it down every row (5000x fewer equilibrium() calls than a
    // naive per-cell loop at 1024x512).
    const rowF = new Float32Array(9 * nx);
    const rowUy = new Float32Array(nx);
    for (let x = 0; x < nx; x++) {
      const uy = uy0 + amp * Math.sin(waveNumber * x);
      rowUy[x] = uy;
      for (let i = 0; i < 9; i++) {
        rowF[i * nx + x] = equilibrium(i, rho0, ux0, uy);
      }
    }
    const ny = n / nx;
    for (let y = 0; y < ny; y++) {
      for (let i = 0; i < 9; i++) {
        f.set(rowF.subarray(i * nx, (i + 1) * nx), i * n + y * nx);
      }
      uyField.set(rowUy, y * nx);
    }
  }
  device.queue.writeBuffer(buffers.f[which], 0, f);
  device.queue.writeBuffer(buffers.uy, 0, uyField);
  const field = new Float32Array(n);
  device.queue.writeBuffer(buffers.rho, 0, field.fill(rho0));
  device.queue.writeBuffer(buffers.ux, 0, field.fill(ux0));
}
