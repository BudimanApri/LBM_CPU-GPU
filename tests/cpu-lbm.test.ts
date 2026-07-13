import { describe, expect, it } from 'vitest';
import { CpuLbm } from '../src/solver/cpu-lbm.ts';

// Phase 1 gate (a): mass conserved to 1e-10 (relative) in a periodic box.
describe('conservation in a periodic box', () => {
  function makePerturbedBox(): CpuLbm {
    const lbm = new CpuLbm({
      nx: 64,
      ny: 32,
      tau: 0.8,
      xBoundary: 'periodic',
      yBoundary: 'periodic',
    });
    // Smooth density + shear-wave perturbation with a nonzero mean flow so
    // the momentum check is meaningful.
    lbm.initEquilibriumWith((x, y) => [
      1 + 0.02 * Math.sin((2 * Math.PI * x) / 64),
      0.03 + 0.02 * Math.sin((2 * Math.PI * y) / 32),
      0.01 * Math.cos((2 * Math.PI * x) / 64),
    ]);
    return lbm;
  }

  it('conserves mass over 1000 steps', () => {
    const lbm = makePerturbedBox();
    const m0 = lbm.totalMass();
    lbm.step(1000);
    expect(Math.abs(lbm.totalMass() - m0) / m0).toBeLessThan(1e-10);
  });

  it('conserves momentum over 1000 steps (force-free)', () => {
    const lbm = makePerturbedBox();
    const p0 = lbm.totalMomentum();
    lbm.step(1000);
    const p1 = lbm.totalMomentum();
    expect(Math.abs(p1.px - p0.px) / Math.abs(p0.px)).toBeLessThan(1e-10);
    // Net y-momentum of the IC is ~0 by symmetry; check absolute drift
    // against the x-momentum scale instead of a meaningless relative error.
    expect(Math.abs(p1.py - p0.py) / Math.abs(p0.px)).toBeLessThan(1e-10);
  });

  it('conserves mass with solids touching a free-slip wall', () => {
    // Exercises the blocked-specular corner links: a population whose
    // free-slip reflection lands in a solid cell must fully reverse into
    // its own cell (opp of the ORIGINAL direction). Getting this wrong
    // double-writes one slot and leaves another stale, leaking mass.
    const nx = 32;
    const ny = 16;
    const lbm = new CpuLbm({ nx, ny, tau: 0.7, xBoundary: 'periodic', yBoundary: 'free-slip' });
    // Block touching the bottom wall, block touching the top wall.
    for (let y = 0; y <= 3; y++) {
      for (let x = 10; x <= 14; x++) lbm.mask[y * nx + x] = 1;
    }
    for (let y = ny - 4; y < ny; y++) {
      for (let x = 20; x <= 24; x++) lbm.mask[y * nx + x] = 1;
    }
    // Single-cell solids at each wall: their diagonal neighbors are fluid,
    // which is exactly the double-write configuration.
    lbm.mask[0 * nx + 5] = 1;
    lbm.mask[(ny - 1) * nx + 27] = 1;
    lbm.initUniformEquilibrium(1, 0.05, 0.01);
    const m0 = lbm.totalMass();
    lbm.step(400);
    expect(Math.abs(lbm.totalMass() - m0) / m0).toBeLessThan(1e-10);
  });

  it('conserves mass with a bounce-back obstacle in the box', () => {
    const nx = 64;
    const ny = 64;
    const lbm = new CpuLbm({ nx, ny, tau: 0.75, xBoundary: 'periodic', yBoundary: 'periodic' });
    // Solid disc, radius 10, centered.
    const cx = 32;
    const cy = 32;
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= 100) {
          lbm.mask[y * nx + x] = 1;
        }
      }
    }
    lbm.initUniformEquilibrium(1, 0.04, 0.015);
    const m0 = lbm.totalMass();
    lbm.step(500);
    expect(Math.abs(lbm.totalMass() - m0) / m0).toBeLessThan(1e-10);
  });
});

// Phase 1 gate (c): uniform flow past no obstacle stays uniform. Also proves
// the Zou-He inlet, outflow copy, and wall reflections are all consistent
// with uniform equilibrium -- including at the inlet/outlet corners, which
// would drift or leak mass if mishandled (gotcha #2).
describe('uniform flow through the tunnel', () => {
  it.each(['free-slip', 'periodic'] as const)('stays uniform with %s walls', (yBoundary) => {
    const u0 = 0.05;
    const nx = 64;
    const ny = 32;
    const lbm = new CpuLbm({
      nx,
      ny,
      tau: 0.7,
      xBoundary: 'inlet-outflow',
      yBoundary,
      inletVelocity: u0,
    });
    lbm.initUniformEquilibrium(1, u0, 0);
    lbm.step(300);
    lbm.computeMoments();
    let maxDu = 0;
    let maxDv = 0;
    let maxDr = 0;
    for (let cell = 0; cell < nx * ny; cell++) {
      maxDu = Math.max(maxDu, Math.abs(lbm.ux[cell]! - u0));
      maxDv = Math.max(maxDv, Math.abs(lbm.uy[cell]!));
      maxDr = Math.max(maxDr, Math.abs(lbm.rho[cell]! - 1));
    }
    expect(maxDu).toBeLessThan(1e-12);
    expect(maxDv).toBeLessThan(1e-12);
    expect(maxDr).toBeLessThan(1e-12);
  });
});

// Phase 1 gate (b): body-force-driven Poiseuille channel matches the
// analytic parabolic profile within 1% at the centerline after convergence.
describe('Poiseuille channel (body force, no-slip walls)', () => {
  it('converges to the analytic parabola', () => {
    const nx = 4; // periodic in x and x-invariant, so a sliver is enough
    const ny = 32;
    // BGK "magic" tau: (tau - 0.5)^2 = 3/16 puts the halfway bounce-back
    // wall exactly at y = -0.5, eliminating the numerical slip artifact.
    const tau = 0.5 + Math.sqrt(3 / 16);
    const nu = (tau - 0.5) / 3;
    const H = ny; // walls at y = -0.5 and y = ny - 0.5
    const uTarget = 0.05;
    const fx = (8 * nu * uTarget) / (H * H);
    const lbm = new CpuLbm({
      nx,
      ny,
      tau,
      xBoundary: 'periodic',
      yBoundary: 'bounce-back',
      bodyForce: [fx, 0],
    });
    lbm.initUniformEquilibrium(1, 0, 0);
    const m0 = lbm.totalMass();

    // March to steady state: converged when the profile stops changing
    // between 500-step checkpoints (measured ~13.5k steps; capped at 30k).
    const profile = new Float64Array(ny);
    const prev = new Float64Array(ny);
    let converged = false;
    for (let chunk = 0; chunk < 60 && !converged; chunk++) {
      lbm.step(500);
      lbm.computeMoments();
      for (let y = 0; y < ny; y++) profile[y] = lbm.ux[y * nx + 1]!;
      if (chunk > 0) {
        let maxDelta = 0;
        for (let y = 0; y < ny; y++) {
          maxDelta = Math.max(maxDelta, Math.abs(profile[y]! - prev[y]!));
        }
        converged = maxDelta / uTarget < 1e-8;
      }
      prev.set(profile);
    }
    expect(converged).toBe(true);

    // Guo forcing conserves mass exactly too.
    expect(Math.abs(lbm.totalMass() - m0) / m0).toBeLessThan(1e-10);

    const uMax = (fx * H * H) / (8 * nu);
    const analytic = (y: number): number => {
      const yw = y + 0.5; // distance from the bottom wall
      return (4 * uMax * yw * (H - yw)) / (H * H);
    };

    // Gate: centerline within 1% (ny even -> the two center nodes).
    for (const yc of [ny / 2 - 1, ny / 2]) {
      expect(Math.abs(profile[yc]! - analytic(yc)) / analytic(yc)).toBeLessThan(0.01);
    }
    // Whole profile within 1% of u_max (absolute-scaled: near-wall relative
    // error is dominated by the tiny analytic value, not solver quality).
    for (let y = 0; y < ny; y++) {
      expect(Math.abs(profile[y]! - analytic(y))).toBeLessThan(0.01 * uMax);
    }
    // Mirror symmetry about the channel center.
    for (let y = 0; y < ny / 2; y++) {
      expect(Math.abs(profile[y]! - profile[ny - 1 - y]!)).toBeLessThan(1e-10);
    }
    // No transverse flow, and x-invariance across the periodic direction.
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        expect(Math.abs(lbm.uy[y * nx + x]!)).toBeLessThan(1e-10);
        expect(Math.abs(lbm.ux[y * nx + x]! - profile[y]!)).toBeLessThan(1e-12);
      }
    }
  });
});
