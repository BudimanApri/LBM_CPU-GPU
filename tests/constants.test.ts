import { describe, expect, it } from 'vitest';
import { CX, CY, OPPOSITE, Q, SPECULAR_Y, WEIGHTS, CS2 } from '../src/solver/constants.ts';

describe('D2Q9 constants', () => {
  it('has 9 discrete velocities', () => {
    expect(Q).toBe(9);
    expect(CX).toHaveLength(9);
    expect(CY).toHaveLength(9);
    expect(WEIGHTS).toHaveLength(9);
    expect(OPPOSITE).toHaveLength(9);
  });

  it('weights sum to 1', () => {
    const sum = WEIGHTS.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 12);
  });

  it('OPPOSITE is an involution (opposite of opposite is self)', () => {
    for (let i = 0; i < Q; i++) {
      const opp = OPPOSITE[i]!;
      expect(OPPOSITE[opp]).toBe(i);
    }
  });

  it('opposite direction reverses the velocity vector', () => {
    // Plain === (not toBe/toEqual, both Object.is-based) deliberately: i=0
    // is its own opposite with zero velocity, where CX[opp] is +0 but
    // -CX[i] is -0 -- Object.is treats those as different, though 0 === -0.
    for (let i = 0; i < Q; i++) {
      const opp = OPPOSITE[i]!;
      expect(CX[opp] === -CX[i]!).toBe(true);
      expect(CY[opp] === -CY[i]!).toBe(true);
    }
  });

  it('rest particle (i=0) has zero velocity and weight 4/9', () => {
    expect(CX[0]).toBe(0);
    expect(CY[0]).toBe(0);
    expect(WEIGHTS[0]).toBeCloseTo(4 / 9, 15);
    expect(OPPOSITE[0]).toBe(0);
  });

  it('lattice speed of sound squared is 1/3', () => {
    expect(CS2).toBeCloseTo(1 / 3, 15);
  });

  it('SPECULAR_Y is an involution that preserves cx and flips cy', () => {
    expect(SPECULAR_Y).toHaveLength(9);
    for (let i = 0; i < Q; i++) {
      const s = SPECULAR_Y[i]!;
      expect(SPECULAR_Y[s]).toBe(i);
      expect(CX[s] === CX[i]!).toBe(true);
      expect(CY[s] === -CY[i]!).toBe(true);
    }
  });
});
