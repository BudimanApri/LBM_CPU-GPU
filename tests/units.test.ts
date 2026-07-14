import { describe, expect, it } from 'vitest';
import {
  TAU_MAX,
  TAU_MIN,
  U_MAX,
  U_MIN,
  clampTau,
  clampU,
  normalizeForce,
  nuFromTau,
  reynolds,
  solveTauForRe,
  smagorinskyTau,
  tauFromNu,
} from '../src/solver/units.ts';

describe('nu <-> tau', () => {
  it('matches the CLAUDE.md relation nu = (tau - 0.5)/3', () => {
    expect(nuFromTau(0.8)).toBeCloseTo(0.1, 15);
    expect(nuFromTau(0.5)).toBe(0);
    expect(nuFromTau(1.1)).toBeCloseTo(0.2, 15);
  });

  it('round-trips through both directions', () => {
    for (const tau of [0.51, 0.65, 0.9330127018922193, 1.2, 1.5]) {
      expect(tauFromNu(nuFromTau(tau))).toBeCloseTo(tau, 14);
    }
    for (const nu of [0.005, 0.05, 0.2, 1 / 3]) {
      expect(nuFromTau(tauFromNu(nu))).toBeCloseTo(nu, 14);
    }
  });
});

describe('stability-envelope clamps', () => {
  it('clampTau enforces [0.51, 1.5]', () => {
    expect(clampTau(0.2)).toBe(TAU_MIN);
    expect(clampTau(0.5)).toBe(TAU_MIN);
    expect(clampTau(0.7)).toBe(0.7);
    expect(clampTau(9)).toBe(TAU_MAX);
  });

  it('clampU enforces [0.02, 0.1]', () => {
    expect(clampU(0)).toBe(U_MIN);
    expect(clampU(0.05)).toBe(0.05);
    expect(clampU(0.5)).toBe(U_MAX);
  });
});

describe('solveTauForRe', () => {
  it('solves an unclamped mid-range case exactly', () => {
    // Re = 20, U = 0.05, D = 20 -> nu = 0.05 -> tau = 0.65
    const s = solveTauForRe(20, 0.05, 20);
    expect(s.tau).toBeCloseTo(0.65, 14);
    expect(s.nu).toBeCloseTo(0.05, 14);
    expect(s.reEffective).toBeCloseTo(20, 12);
    expect(s.clamped).toBe(false);
  });

  it('clamps at high Re and reports the effective Re honestly', () => {
    // Re = 10,000 at U = 0.1, D = 20 wants nu = 2e-4 -> tau = 0.5006: unstable.
    const s = solveTauForRe(10_000, 0.1, 20);
    expect(s.tau).toBe(TAU_MIN);
    expect(s.clamped).toBe(true);
    // Effective Re realized by tau = 0.51: U*D/nu(0.51) = 2 / (0.01/3) = 600.
    expect(s.reEffective).toBeCloseTo(600, 10);
    expect(s.reEffective).toBeLessThan(10_000);
  });

  it('clamps at very low Re', () => {
    // Re = 1, U = 0.02, D = 30 wants nu = 0.6 -> tau = 2.3: over-damped.
    const s = solveTauForRe(1, 0.02, 30);
    expect(s.tau).toBe(TAU_MAX);
    expect(s.clamped).toBe(true);
    expect(s.reEffective).toBeGreaterThan(1);
  });

  it('never yields tau <= 0.5 for any positive input', () => {
    for (const re of [1e-6, 1, 100, 1e12]) {
      for (const u of [1e-6, 0.05, 0.1, 10]) {
        for (const d of [1e-3, 1, 64, 1e6]) {
          expect(solveTauForRe(re, u, d).tau).toBeGreaterThan(0.5);
        }
      }
    }
  });

  it('throws RangeError on nonpositive or nonfinite inputs', () => {
    expect(() => solveTauForRe(0, 0.05, 20)).toThrow(RangeError);
    expect(() => solveTauForRe(-5, 0.05, 20)).toThrow(RangeError);
    expect(() => solveTauForRe(NaN, 0.05, 20)).toThrow(RangeError);
    expect(() => solveTauForRe(Infinity, 0.05, 20)).toThrow(RangeError);
    expect(() => solveTauForRe(100, 0, 20)).toThrow(RangeError);
    expect(() => solveTauForRe(100, 0.05, -1)).toThrow(RangeError);
  });
});

describe('reynolds', () => {
  it('computes Re = U D / nu', () => {
    expect(reynolds(0.05, 40, 0.02)).toBeCloseTo(100, 12);
  });
});

describe('force coefficient normalization', () => {
  it('uses the supplied aerodynamic reference length', () => {
    const frontal = normalizeForce(1.2, -0.3, 1, 0.05, 20);
    const chord = normalizeForce(1.2, -0.3, 1, 0.05, 100);
    expect(frontal.cd).toBeCloseTo(48, 12);
    expect(frontal.cl).toBeCloseTo(-12, 12);
    expect(chord.cd).toBeCloseTo(frontal.cd / 5, 12);
    expect(chord.cl).toBeCloseTo(frontal.cl / 5, 12);
  });

  it('rejects a nonpositive normalization denominator', () => {
    expect(() => normalizeForce(1, 1, 1, 0.05, 0)).toThrow(RangeError);
    expect(() => normalizeForce(1, 1, 1, Number.NaN, 20)).toThrow(RangeError);
  });
});

describe('Smagorinsky relaxation', () => {
  it('adds no eddy viscosity at equilibrium and grows with stress', () => {
    expect(smagorinskyTau(0.51, 0.1, 0, 1)).toBeCloseTo(0.51, 15);
    const lowStress = smagorinskyTau(0.51, 0.1, 0.001, 1);
    const highStress = smagorinskyTau(0.51, 0.1, 0.1, 1);
    expect(lowStress).toBeGreaterThan(0.51);
    expect(highStress).toBeGreaterThan(lowStress);
    expect(highStress).toBeLessThanOrEqual(TAU_MAX);
  });
});
