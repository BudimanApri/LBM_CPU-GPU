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
    expect(text).toContain('steps  12345');
  });

  it('omits the effective-Re note when not clamped', () => {
    expect(formatHud({ ...base, reEffective: 100 })).not.toContain('eff');
  });

  it('shows the effective-Re note when the requested Re was clamped', () => {
    const text = formatHud({ ...base, re: 10_000, reEffective: 600.3 });
    expect(text).toContain('Re     10000 (eff 600)');
  });
});
