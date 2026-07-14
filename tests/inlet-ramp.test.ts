import { describe, expect, it } from 'vitest';
import { inletRampVelocity } from '../src/solver/inlet-ramp.ts';

describe('inlet velocity ramp', () => {
  it('starts and ends exactly with a smooth midpoint', () => {
    expect(inletRampVelocity(0.01, 0.05, 0, 1000)).toBe(0.01);
    expect(inletRampVelocity(0.01, 0.05, 500, 1000)).toBeCloseTo(0.03, 15);
    expect(inletRampVelocity(0.01, 0.05, 1000, 1000)).toBe(0.05);
  });

  it('clamps progress and supports ramping downward', () => {
    expect(inletRampVelocity(0.09, 0.04, -50, 1000)).toBe(0.09);
    expect(inletRampVelocity(0.09, 0.04, 5000, 1000)).toBeCloseTo(0.04, 15);
    expect(inletRampVelocity(0.09, 0.04, 1, 0)).toBe(0.04);
  });
});
