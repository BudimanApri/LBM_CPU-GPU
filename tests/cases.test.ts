import { describe, expect, it } from 'vitest';
import {
  MAX_BLOCKAGE_RATIO,
  RE100_CYLINDER,
  RE20_CYLINDER,
  VALIDATION_CASES,
  assertBlockage,
  blockageRatio,
  caseTau,
} from '../src/validation/cases.ts';
import { TAU_MIN } from '../src/solver/units.ts';

describe('validation cases', () => {
  it('keeps blockage ratio at or below 1/8 for every case', () => {
    for (const c of VALIDATION_CASES) {
      expect(blockageRatio(c)).toBeLessThanOrEqual(MAX_BLOCKAGE_RATIO);
      expect(() => {
        assertBlockage(c);
      }).not.toThrow();
    }
  });

  it('realizes the requested Re without clamping tau (stays in envelope)', () => {
    for (const c of VALIDATION_CASES) {
      const sol = caseTau(c);
      expect(sol.clamped).toBe(false);
      expect(sol.tau).toBeGreaterThan(TAU_MIN);
    }
  });

  it('assertBlockage throws when the obstacle is too tall for the domain', () => {
    expect(() => {
      assertBlockage({ ...RE20_CYLINDER, diameter: RE20_CYLINDER.ny / 4 });
    }).toThrow(/blockage ratio/);
  });

  it('pins the expected gate targets', () => {
    expect(RE20_CYLINDER.expected.cd).toEqual({ value: 2.0, tol: 0.15 });
    expect(RE100_CYLINDER.expected.st).toEqual({ min: 0.16, max: 0.17 });
  });
});
