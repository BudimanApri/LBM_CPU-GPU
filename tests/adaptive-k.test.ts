import { describe, expect, it } from 'vitest';
import { AdaptiveKController } from '../src/solver/adaptive-k.ts';

function feed(
  controller: AdaptiveKController,
  frameMs: number,
  gpuMs: number | null,
  frames = 45,
): number | null {
  let changed: number | null = null;
  for (let i = 0; i < frames; i++) changed = controller.observe(frameMs, gpuMs) ?? changed;
  return changed;
}

describe('AdaptiveKController', () => {
  it('grows under sustained CPU/GPU headroom and respects its maximum', () => {
    const c = new AdaptiveKController(1, 4, 1000 / 60, 2);
    expect(feed(c, 8, 5)).toBe(3);
    expect(feed(c, 8, 5)).toBe(4);
    expect(feed(c, 8, 5)).toBeNull();
    expect(c.value()).toBe(4);
  });

  it('shrinks on rAF misses or queued GPU work and respects its minimum', () => {
    const frameBound = new AdaptiveKController(1, 8, 1000 / 60, 4);
    expect(feed(frameBound, 25, 5)).toBe(3);

    const gpuBound = new AdaptiveKController(1, 8, 1000 / 60, 4);
    expect(feed(gpuBound, 16.7, 25)).toBe(3);
    expect(feed(gpuBound, 16.7, 25, 45 * 4)).toBe(1);
    expect(gpuBound.value()).toBe(1);
  });

  it('supports a clamped validation override', () => {
    const c = new AdaptiveKController(1, 8, 1000 / 60, 3);
    expect(c.set(99)).toBe(8);
    expect(c.set(0)).toBe(1);
    expect(c.set(3.4)).toBe(3);
  });
});
