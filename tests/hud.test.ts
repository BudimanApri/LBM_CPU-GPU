import { describe, expect, it } from 'vitest';
import { formatHud } from '../src/ui/hud.ts';

const base = {
  fps: 60.4,
  mlups: 3191.2,
  kSubsteps: 2,
  re: 100,
  tau: 0.536,
  d: 24,
  presetLabel: 'cylinder',
  steps: 12345,
  cd: 2.014,
  cl: -0.317,
  st: 0.165,
};

describe('formatHud', () => {
  it('formats all fields on separate lines', () => {
    const text = formatHud(base);
    expect(text).toContain('fps    60');
    expect(text).toContain('MLUPS  3191');
    expect(text).toContain('K      2');
    expect(text).toContain('Re     100');
    expect(text).toContain('tau    0.5360');
    expect(text).toContain('D      24 cells (cylinder)');
    expect(text).toContain('Cd     2.014');
    expect(text).toContain('Cl     -0.317');
    expect(text).toContain('St     0.165');
    expect(text).toContain('steps  12345');
  });

  it('shows -- for Strouhal when no oscillation is detected yet', () => {
    expect(formatHud({ ...base, st: null })).toContain('St     --');
  });

  it('omits the effective-Re note when not clamped', () => {
    expect(formatHud({ ...base, reEffective: 100 })).not.toContain('eff');
  });

  it('shows the effective-Re note when the requested Re was clamped', () => {
    const text = formatHud({ ...base, re: 10_000, reEffective: 600.3 });
    expect(text).toContain('Re     10000 (eff 600)');
  });
});
