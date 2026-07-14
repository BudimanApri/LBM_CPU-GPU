import { describe, expect, it } from 'vitest';
import { nacaCamber, nacaHalfThickness, nacaPolygon } from '../src/geometry/naca.ts';
import { cylinderPreset, nacaPreset, backwardStepPreset } from '../src/geometry/presets.ts';

describe('NACA thickness and camber functions', () => {
  it('closed trailing edge: half-thickness is ~0 at x/c = 1', () => {
    expect(Math.abs(nacaHalfThickness(0.12, 1))).toBeLessThan(1e-12);
  });

  it('max total thickness matches the XX digits near x/c = 0.30', () => {
    const t = 0.12;
    let best = 0;
    let bestX = 0;
    for (let x = 0.01; x <= 0.99; x += 0.001) {
      const th = 2 * nacaHalfThickness(t, x);
      if (th > best) {
        best = th;
        bestX = x;
      }
    }
    expect(best).toBeCloseTo(t, 3);
    expect(bestX).toBeGreaterThan(0.27);
    expect(bestX).toBeLessThan(0.33);
  });

  it('4412 camber peaks at 4% of chord at x/c = 0.4', () => {
    const { yc, dyc } = nacaCamber(0.04, 0.4, 0.4);
    expect(yc).toBeCloseTo(0.04, 12);
    expect(dyc).toBeCloseTo(0, 12);
    // Camber line is continuous across the p-junction.
    const before = nacaCamber(0.04, 0.4, 0.4 - 1e-9).yc;
    expect(Math.abs(before - yc)).toBeLessThan(1e-8);
  });
});

describe('nacaPolygon', () => {
  it('rejects malformed designations', () => {
    expect(() => nacaPolygon('44', 100, 0)).toThrow(RangeError);
    expect(() => nacaPolygon('44a2', 100, 0)).toThrow(RangeError);
    expect(() => nacaPolygon('4012', 100, 0)).toThrow(RangeError); // camber with no position
    expect(() => nacaPolygon('4400', 100, 0)).toThrow(RangeError); // zero thickness
  });

  it('is a closed loop', () => {
    const pts = nacaPolygon('4412', 100, 0);
    const first = pts[0]!;
    const last = pts[pts.length - 1]!;
    expect(Math.hypot(first.x - last.x, first.y - last.y)).toBeLessThan(1e-9);
  });

  it('symmetric 0012 mirrors exactly about the chord line', () => {
    const pts = nacaPolygon('0012', 100, 0);
    // TE->upper->LE (81 points), then lower LE->TE (80 points).
    const upper = pts.slice(0, 81);
    const lower = pts.slice(80); // shares the LE point
    expect(lower).toHaveLength(81);
    for (let k = 0; k < 81; k++) {
      const u = upper[80 - k]!; // LE -> TE ordering
      const l = lower[k]!;
      expect(Math.abs(u.x - l.x)).toBeLessThan(1e-12);
      expect(Math.abs(u.y + l.y)).toBeLessThan(1e-12);
    }
  });

  it('positive AoA pitches the nose up about the quarter-chord', () => {
    const chord = 100;
    const level = nacaPolygon('0012', chord, 0);
    const pitched = nacaPolygon('0012', chord, 10);
    // Leading edge is the point with minimum x.
    const le = (pts: { x: number; y: number }[]) => pts.reduce((a, b) => (b.x < a.x ? b : a));
    expect(le(pitched).y).toBeGreaterThan(le(level).y);
    // The pivot (quarter-chord, at the origin) stays fixed: rotating back
    // recovers the level polygon.
    const back = nacaPolygon('0012', chord, 0);
    expect(le(back).y).toBeCloseTo(le(level).y, 12);
  });

  it('frontal extent grows with AoA', () => {
    const height = (alpha: number) => {
      const pts = nacaPolygon('0012', 100, alpha);
      const ys = pts.map((p) => p.y);
      return Math.max(...ys) - Math.min(...ys);
    };
    expect(height(15)).toBeGreaterThan(height(0));
  });
});

describe('presets', () => {
  it('cylinder mask has ~pi r^2 solid cells centered at 25% chord', () => {
    const nx = 256;
    const ny = 128;
    const { mask, d } = cylinderPreset(nx, ny, 24);
    expect(d).toBe(24);
    let count = 0;
    for (const v of mask) count += v;
    expect(count).toBeGreaterThan(Math.PI * 12 * 12 * 0.95);
    expect(count).toBeLessThan(Math.PI * 12 * 12 * 1.1);
    // Mirror symmetry about the (ny-1)/2 centerline.
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        expect(mask[y * nx + x]).toBe(mask[(ny - 1 - y) * nx + x]);
      }
    }
  });

  it('airfoil preset rasterizes a solid, leak-free interior row', () => {
    const nx = 256;
    const ny = 128;
    const { mask, d } = nacaPreset(nx, ny, '4412', 80, 0);
    expect(d).toBeGreaterThan(6); // 12% of 80 ~ 10 cells thick
    // The row through the thickest section must be contiguous solid
    // (a gap would let flow leak through the airfoil).
    let count = 0;
    for (const v of mask) count += v;
    expect(count).toBeGreaterThan(200);
  });

  it('presets never touch the two easternmost columns', () => {
    const nx = 64;
    const ny = 32;
    for (const { mask } of [
      cylinderPreset(nx, ny, 20),
      nacaPreset(nx, ny, '0012', 30, 10),
      backwardStepPreset(nx, ny, 10),
    ]) {
      for (let y = 0; y < ny; y++) {
        expect(mask[y * nx + (nx - 1)]).toBe(0);
        expect(mask[y * nx + (nx - 2)]).toBe(0);
      }
    }
  });

  it('backward step spans from the inlet along the bottom', () => {
    const nx = 128;
    const ny = 64;
    const { mask, d } = backwardStepPreset(nx, ny, 20);
    expect(d).toBe(20);
    expect(mask[0]).toBe(1); // inlet-bottom corner is solid
    expect(mask[19 * nx + 0]).toBe(1);
    expect(mask[20 * nx + 0]).toBe(0); // above the step is fluid
  });
});
