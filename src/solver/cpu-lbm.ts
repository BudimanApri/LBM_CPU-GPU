// CPU reference D2Q9 BGK solver -- the ground truth the GPU kernels are
// verified against (Phase 2 parity gate). Written for clarity, not speed:
// classic collide-then-stream on two Float64 buffers, deliberately higher
// precision than the GPU's mandated f32 so the Phase 2 tolerance (1e-4)
// measures the GPU's rounding against a trustworthy reference.
//
// Data layout (shared with the GPU): f[i * n + y * nx + x], n = nx * ny.
//
// Boundary configuration:
//   x: 'periodic', or 'inlet-outflow' = Zou-He velocity inlet (west, x=0)
//      plus zero-gradient outflow (east, x=nx-1), both applied post-stream.
//   y: 'periodic' | 'bounce-back' (no-slip walls, halfway bounce-back --
//      walls sit at y = -0.5 and y = ny - 0.5, so channel height H = ny)
//      | 'free-slip' (specular reflection, the wind-tunnel default).
//   obstacles: halfway bounce-back via `mask` (0 = fluid, 1 = solid).
//
// Corner note (CLAUDE.md gotcha #2): with free-slip or periodic top/bottom,
// inlet corners are NOT underdetermined -- the wall reflection supplies the
// wall-adjacent populations before Zou-He runs, leaving exactly the same
// three unknowns (f1, f5, f8) as interior inlet rows, so the standard
// west-wall equations apply on the whole column. The classic underdetermined
// corner only arises for inlet + no-slip outer walls, a combination no
// configuration here uses (Poiseuille validation is periodic in x). If
// no-slip tunnel walls are ever added, add the standard corner treatment.

import { CX, CY, OPPOSITE, Q, SPECULAR_Y, WEIGHTS } from './constants.ts';

export type XBoundary = 'periodic' | 'inlet-outflow';
export type YBoundary = 'periodic' | 'bounce-back' | 'free-slip';

export interface CpuLbmOptions {
  nx: number;
  ny: number;
  tau: number;
  xBoundary: XBoundary;
  yBoundary: YBoundary;
  /** Zou-He inlet speed U; required when xBoundary is 'inlet-outflow'. */
  inletVelocity?: number;
  /** Uniform body force (fx, fy), applied with Guo (2002) forcing. */
  bodyForce?: readonly [number, number];
}

/** Factored D2Q9 equilibrium (gotcha #4 -- keep exactly this form). */
export function equilibrium(i: number, rho: number, ux: number, uy: number): number {
  const cu = CX[i]! * ux + CY[i]! * uy;
  return WEIGHTS[i]! * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy));
}

export class CpuLbm {
  readonly nx: number;
  readonly ny: number;
  /** Cells per field: nx * ny. */
  readonly n: number;
  readonly tau: number;
  readonly xBoundary: XBoundary;
  readonly yBoundary: YBoundary;
  readonly inletVelocity: number;
  readonly forceX: number;
  readonly forceY: number;
  /**
   * Obstacle mask, 0 = fluid, 1 = solid, indexed y * nx + x. Set cells
   * before running steps; mid-run edits leave stale populations in newly
   * solid cells (gotcha #7's re-equilibration is a Phase 3 GPU concern).
   * Do not place solids in the two easternmost columns when using
   * 'inlet-outflow': the outflow copy reads column nx-2.
   */
  readonly mask: Uint8Array;
  /** Current distributions, f[i * n + y * nx + x]. Swapped every step. */
  f: Float64Array;
  private fNext: Float64Array;
  /** Macroscopic fields, filled by computeMoments(), indexed y * nx + x. */
  readonly rho: Float64Array;
  readonly ux: Float64Array;
  readonly uy: Float64Array;

  constructor(opts: CpuLbmOptions) {
    const { nx, ny, tau } = opts;
    if (!Number.isInteger(nx) || nx < 3 || !Number.isInteger(ny) || ny < 3) {
      throw new RangeError(`grid must be integer and at least 3x3, got ${nx}x${ny}`);
    }
    if (!Number.isFinite(tau) || tau <= 0.5) {
      throw new RangeError(`tau must be finite and > 0.5, got ${tau}`);
    }
    const u0 = opts.inletVelocity ?? 0;
    if (opts.xBoundary === 'inlet-outflow' && !(u0 > 0 && u0 < 1)) {
      throw new RangeError(`inlet-outflow requires inletVelocity in (0, 1), got ${u0}`);
    }
    const [fx, fy] = opts.bodyForce ?? [0, 0];
    this.nx = nx;
    this.ny = ny;
    this.n = nx * ny;
    this.tau = tau;
    this.xBoundary = opts.xBoundary;
    this.yBoundary = opts.yBoundary;
    this.inletVelocity = u0;
    this.forceX = fx;
    this.forceY = fy;
    this.mask = new Uint8Array(this.n);
    this.f = new Float64Array(Q * this.n);
    this.fNext = new Float64Array(Q * this.n);
    this.rho = new Float64Array(this.n);
    this.ux = new Float64Array(this.n);
    this.uy = new Float64Array(this.n);
  }

  /** Initialize every cell to the equilibrium of the given uniform state. */
  initUniformEquilibrium(rho0: number, ux0: number, uy0: number): void {
    this.initEquilibriumWith(() => [rho0, ux0, uy0]);
  }

  /** Initialize every cell to the equilibrium of per-cell (rho, ux, uy). */
  initEquilibriumWith(fields: (x: number, y: number) => readonly [number, number, number]): void {
    const { nx, ny, n, f } = this;
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const [r, u, v] = fields(x, y);
        const cell = y * nx + x;
        for (let i = 0; i < Q; i++) {
          f[i * n + cell] = equilibrium(i, r, u, v);
        }
      }
    }
  }

  step(count = 1): void {
    for (let s = 0; s < count; s++) {
      this.collide();
      this.stream();
      if (this.xBoundary === 'inlet-outflow') this.applyInletOutflowBCs();
    }
  }

  /**
   * Recompute rho/ux/uy from the current distributions. The velocity
   * includes the half-force shift, u = (sum f c + F/2) / rho -- that is the
   * physical velocity under Guo forcing. Solid cells get (rho=1, u=0).
   */
  computeMoments(): void {
    const { f, mask, rho, ux, uy, n, forceX, forceY } = this;
    for (let cell = 0; cell < n; cell++) {
      if (mask[cell]! !== 0) {
        rho[cell] = 1;
        ux[cell] = 0;
        uy[cell] = 0;
        continue;
      }
      let r = 0;
      let mx = 0;
      let my = 0;
      for (let i = 0; i < Q; i++) {
        const v = f[i * n + cell]!;
        r += v;
        mx += v * CX[i]!;
        my += v * CY[i]!;
      }
      rho[cell] = r;
      ux[cell] = (mx + 0.5 * forceX) / r;
      uy[cell] = (my + 0.5 * forceY) / r;
    }
  }

  /** Total mass over fluid cells (solid cells hold inert values). */
  totalMass(): number {
    const { f, mask, n } = this;
    let m = 0;
    for (let cell = 0; cell < n; cell++) {
      if (mask[cell]! !== 0) continue;
      for (let i = 0; i < Q; i++) {
        m += f[i * n + cell]!;
      }
    }
    return m;
  }

  /** Bare lattice momentum (sum f c, no half-force shift) over fluid cells. */
  totalMomentum(): { px: number; py: number } {
    const { f, mask, n } = this;
    let px = 0;
    let py = 0;
    for (let cell = 0; cell < n; cell++) {
      if (mask[cell]! !== 0) continue;
      for (let i = 0; i < Q; i++) {
        const v = f[i * n + cell]!;
        px += v * CX[i]!;
        py += v * CY[i]!;
      }
    }
    return { px, py };
  }

  private collide(): void {
    const { f, mask, nx, ny, n, tau, forceX, forceY } = this;
    const omega = 1 / tau;
    const hasForce = forceX !== 0 || forceY !== 0;
    // Guo (2002): S_i = (1 - 1/(2 tau)) w_i [3 (c_i - u) . F + 9 (c_i . u)(c_i . F)]
    const guoPrefactor = 1 - 0.5 * omega;
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const cell = y * nx + x;
        if (mask[cell]! !== 0) continue;
        let r = 0;
        let mx = 0;
        let my = 0;
        for (let i = 0; i < Q; i++) {
          const v = f[i * n + cell]!;
          r += v;
          mx += v * CX[i]!;
          my += v * CY[i]!;
        }
        // Physical velocity includes the half-force shift (Guo forcing).
        const u = (mx + 0.5 * forceX) / r;
        const v = (my + 0.5 * forceY) / r;
        for (let i = 0; i < Q; i++) {
          const k = i * n + cell;
          let value = f[k]! - omega * (f[k]! - equilibrium(i, r, u, v));
          if (hasForce) {
            const cx = CX[i]!;
            const cy = CY[i]!;
            const cu = cx * u + cy * v;
            value +=
              guoPrefactor *
              WEIGHTS[i]! *
              (3 * ((cx - u) * forceX + (cy - v) * forceY) + 9 * cu * (cx * forceX + cy * forceY));
          }
          f[k] = value;
        }
      }
    }
  }

  private stream(): void {
    const { f, fNext, mask, nx, ny, n, xBoundary, yBoundary } = this;
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const cell = y * nx + x;
        if (mask[cell]! !== 0) continue;
        fNext[cell] = f[cell]!; // rest population (i = 0)
        for (let i = 1; i < Q; i++) {
          const value = f[i * n + cell]!;
          let dir = i;
          let xd = x + CX[i]!;
          let yd = y + CY[i]!;
          if (yd < 0 || yd >= ny) {
            if (yBoundary === 'periodic') {
              yd = (yd + ny) % ny;
            } else if (yBoundary === 'bounce-back') {
              // Halfway bounce-back: the wall sits between lattice rows
              // (y = -0.5 / ny - 0.5); the population returns to its own
              // cell fully reversed (no-slip).
              fNext[OPPOSITE[i]! * n + cell] = value;
              continue;
            } else {
              // Free-slip: specular reflection. cy flips, cx is kept, so the
              // population lands beside its source in the same row.
              dir = SPECULAR_Y[i]!;
              yd = y;
            }
          }
          if (xd < 0 || xd >= nx) {
            if (xBoundary === 'periodic') {
              xd = (xd + nx) % nx;
            } else {
              // Exits through the inlet/outlet face. The unknown slots this
              // leaves are reconstructed post-stream by applyInletOutflowBCs.
              continue;
            }
          }
          const target = yd * nx + xd;
          if (mask[target]! !== 0) {
            // Halfway bounce-back off an obstacle: reflect into the source
            // cell's opposite direction (gotcha #3, push-scheme mirror).
            fNext[OPPOSITE[dir]! * n + cell] = value;
            continue;
          }
          fNext[dir * n + target] = value;
        }
      }
    }
    this.fNext = this.f;
    this.f = fNext;
  }

  private applyInletOutflowBCs(): void {
    const { f, mask, nx, ny, n, inletVelocity: u0 } = this;
    // West (x = 0): Zou-He velocity inlet with prescribed (U, 0). Standard
    // D2Q9 west-wall reconstruction of the unknowns f1, f5, f8.
    for (let y = 0; y < ny; y++) {
      const cell = y * nx;
      if (mask[cell]! !== 0) continue;
      const f0 = f[cell]!;
      const f2 = f[2 * n + cell]!;
      const f3 = f[3 * n + cell]!;
      const f4 = f[4 * n + cell]!;
      const f6 = f[6 * n + cell]!;
      const f7 = f[7 * n + cell]!;
      const r = (f0 + f2 + f4 + 2 * (f3 + f6 + f7)) / (1 - u0);
      f[n + cell] = f3 + (2 / 3) * r * u0;
      f[5 * n + cell] = f7 - 0.5 * (f2 - f4) + (1 / 6) * r * u0;
      f[8 * n + cell] = f6 + 0.5 * (f2 - f4) + (1 / 6) * r * u0;
    }
    // East (x = nx - 1): zero-gradient outflow -- copy the westward-moving
    // unknowns (f3, f6, f7) from the neighbouring interior column.
    for (let y = 0; y < ny; y++) {
      const cell = y * nx + (nx - 1);
      if (mask[cell]! !== 0) continue;
      const src = cell - 1;
      f[3 * n + cell] = f[3 * n + src]!;
      f[6 * n + cell] = f[6 * n + src]!;
      f[7 * n + cell] = f[7 * n + src]!;
    }
  }
}
