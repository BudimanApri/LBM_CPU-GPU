import { describe, expect, it } from 'vitest';
import { meanCoefficients, strouhalFromSamples, type ForceSample } from '../src/ui/chart.ts';

// Build Cl samples from a sinusoid of a known period (in solver steps),
// sampled at a fixed step spacing, so St = D / (period * U) is exact.
function sinusoidSamples(
  periodSteps: number,
  cycles: number,
  spacing: number,
  amplitude: number,
  offset: number,
): ForceSample[] {
  const out: ForceSample[] = [];
  const total = periodSteps * cycles;
  for (let step = 0; step <= total; step += spacing) {
    out.push({
      step,
      cd: 2.0,
      cl: offset + amplitude * Math.sin((2 * Math.PI * step) / periodSteps),
    });
  }
  return out;
}

describe('strouhalFromSamples', () => {
  it('recovers St = D / (period * U) from a clean oscillation', () => {
    const period = 6000; // lattice steps per shedding cycle
    const D = 48;
    const U = 0.05;
    const samples = sinusoidSamples(period, 10, 80, 0.5, 0);
    const st = strouhalFromSamples(samples, D, U);
    expect(st).not.toBeNull();
    // Expected St = 48 / (6000 * 0.05) = 0.16.
    expect(st!).toBeCloseTo(D / (period * U), 3);
  });

  it('is unaffected by a nonzero mean (offset Cl)', () => {
    const period = 5000;
    const D = 40;
    const U = 0.04;
    const samples = sinusoidSamples(period, 8, 50, 0.3, 1.7);
    const st = strouhalFromSamples(samples, D, U);
    expect(st).not.toBeNull();
    expect(st!).toBeCloseTo(D / (period * U), 2);
  });

  it('lands in the Re=100 gate window for a representative signal', () => {
    // period ~ 5818 steps gives St ~ 0.165 at D=48, U=0.05.
    const samples = sinusoidSamples(5818, 9, 70, 0.4, 0.05);
    const st = strouhalFromSamples(samples, 48, 0.05);
    expect(st).not.toBeNull();
    expect(st!).toBeGreaterThanOrEqual(0.16);
    expect(st!).toBeLessThanOrEqual(0.17);
  });

  it('returns null for a flat (non-oscillating) signal', () => {
    const samples: ForceSample[] = [];
    for (let step = 0; step <= 2000; step += 50) samples.push({ step, cd: 2.0, cl: 0.0 });
    expect(strouhalFromSamples(samples, 48, 0.05)).toBeNull();
  });

  it('returns null without enough samples or with bad D/U', () => {
    const few: ForceSample[] = [
      { step: 0, cd: 2, cl: 0 },
      { step: 10, cd: 2, cl: 0.1 },
    ];
    expect(strouhalFromSamples(few, 48, 0.05)).toBeNull();
    const enough = sinusoidSamples(5000, 8, 50, 0.3, 0);
    expect(strouhalFromSamples(enough, 0, 0.05)).toBeNull();
    expect(strouhalFromSamples(enough, 48, 0)).toBeNull();
  });
});

describe('meanCoefficients', () => {
  it('averages Cd and cancels a zero-mean oscillation in Cl', () => {
    // Cl is a full-cycle sinusoid (mean 0); Cd is constant 2.03.
    const samples = sinusoidSamples(4000, 6, 40, 1.0, 0).map((s) => ({ ...s, cd: 2.03 }));
    const m = meanCoefficients(samples);
    expect(m).not.toBeNull();
    expect(m!.cd).toBeCloseTo(2.03, 6);
    expect(Math.abs(m!.cl)).toBeLessThan(0.05); // acoustic-like ripple averages out
  });

  it('returns null for an empty buffer', () => {
    expect(meanCoefficients([])).toBeNull();
  });
});
