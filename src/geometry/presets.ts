// Obstacle preset rasterizers. Every preset returns a fresh full-domain
// mask (0 = fluid, 1 = solid) plus the characteristic size D (frontal
// height in cells) used for the Re <-> tau conversion and, later, Cd/Cl.
//
// Placement convention (CLAUDE.md): obstacles sit at ~25% chord of the
// domain, vertically centered. The vertical center is (ny - 1) / 2 -- a
// half-integer for even ny -- so shapes are mirror-symmetric about the
// centerline, which the Re=20 twin-vortex gate relies on.
//
// The two easternmost columns are never solidified: the outflow copy reads
// column nx-2, and solids there would propagate garbage.

import { nacaPolygon, type Point } from './naca.ts';

export interface PresetResult {
  mask: Uint8Array;
  /** Characteristic obstacle size D: frontal (y) extent in cells. */
  d: number;
}

/** Rasterize a filled circle into an existing mask. */
export function rasterizeCircle(
  mask: Uint8Array,
  nx: number,
  ny: number,
  cx: number,
  cy: number,
  r: number,
): void {
  const r2 = r * r;
  const xMax = Math.min(nx - 3, Math.ceil(cx + r));
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(ny - 1, Math.ceil(cy + r)); y++) {
    for (let x = Math.max(0, Math.floor(cx - r)); x <= xMax; x++) {
      const dx = x - cx;
      const dy = y - cy;
      if (dx * dx + dy * dy <= r2) mask[y * nx + x] = 1;
    }
  }
}

/** Even-odd point-in-polygon test. */
function insidePolygon(px: number, py: number, pts: readonly Point[]): boolean {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const a = pts[i]!;
    const b = pts[j]!;
    if (a.y > py !== b.y > py && px < ((b.x - a.x) * (py - a.y)) / (b.y - a.y) + a.x) {
      inside = !inside;
    }
  }
  return inside;
}

/** Rasterize a closed polygon (cell centers, even-odd rule) into a mask. */
export function rasterizePolygon(
  mask: Uint8Array,
  nx: number,
  ny: number,
  pts: readonly Point[],
): void {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const x0 = Math.max(0, Math.floor(minX));
  const x1 = Math.min(nx - 3, Math.ceil(maxX));
  const y0 = Math.max(0, Math.floor(minY));
  const y1 = Math.min(ny - 1, Math.ceil(maxY));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      if (insidePolygon(x, y, pts)) mask[y * nx + x] = 1;
    }
  }
}

function anchor(nx: number, ny: number): { ax: number; ay: number } {
  return { ax: Math.round(0.25 * nx), ay: (ny - 1) / 2 };
}

/**
 * Frontal height D of an arbitrary mask: the y-extent of its solid bounding
 * box (CLAUDE.md's "bounding box column-height"). Used to recompute D after
 * brush edits, where no analytic D is available. Returns 0 for an empty mask.
 */
export function maskFrontalHeight(mask: ArrayLike<number>, nx: number, ny: number): number {
  let minY = Infinity;
  let maxY = -Infinity;
  for (let y = 0; y < ny; y++) {
    const row = y * nx;
    for (let x = 0; x < nx; x++) {
      if (mask[row + x] !== 0) {
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        break; // one solid in the row is enough to bound it
      }
    }
  }
  return maxY < minY ? 0 : maxY - minY + 1;
}

function polygonFrontalHeight(pts: readonly Point[]): number {
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of pts) {
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  return Math.ceil(maxY - minY);
}

export function cylinderPreset(nx: number, ny: number, diameter: number): PresetResult {
  const { ax, ay } = anchor(nx, ny);
  const mask = new Uint8Array(nx * ny);
  rasterizeCircle(mask, nx, ny, ax, ay, diameter / 2);
  return { mask, d: diameter };
}

export function nacaPreset(
  nx: number,
  ny: number,
  digits: string,
  chord: number,
  alphaDeg: number,
): PresetResult {
  const { ax, ay } = anchor(nx, ny);
  const pts = nacaPolygon(digits, chord, alphaDeg).map((p) => ({ x: p.x + ax, y: p.y + ay }));
  const mask = new Uint8Array(nx * ny);
  rasterizePolygon(mask, nx, ny, pts);
  return { mask, d: polygonFrontalHeight(pts) };
}

const PLATE_THICKNESS = 3;

/** Flat plate normal to the flow (a thin vertical rectangle). */
export function plateNormalPreset(nx: number, ny: number, height: number): PresetResult {
  const { ax, ay } = anchor(nx, ny);
  const h = height / 2;
  const t = PLATE_THICKNESS / 2;
  const pts: Point[] = [
    { x: ax - t, y: ay - h },
    { x: ax + t, y: ay - h },
    { x: ax + t, y: ay + h },
    { x: ax - t, y: ay + h },
  ];
  const mask = new Uint8Array(nx * ny);
  rasterizePolygon(mask, nx, ny, pts);
  return { mask, d: height };
}

/** Flat plate inclined to the flow (rotated thin rectangle). */
export function plateInclinedPreset(
  nx: number,
  ny: number,
  length: number,
  angleDeg: number,
): PresetResult {
  const { ax, ay } = anchor(nx, ny);
  const a = (angleDeg * Math.PI) / 180;
  const cosA = Math.cos(a);
  const sinA = Math.sin(a);
  const rot = (x: number, y: number): Point => ({
    x: ax + x * cosA + y * sinA,
    y: ay - x * sinA + y * cosA,
  });
  const l = length / 2;
  const t = PLATE_THICKNESS / 2;
  const pts = [rot(-l, -t), rot(l, -t), rot(l, t), rot(-l, t)];
  const mask = new Uint8Array(nx * ny);
  rasterizePolygon(mask, nx, ny, pts);
  return { mask, d: polygonFrontalHeight(pts) };
}

/**
 * Backward-facing step: a solid block along the bottom from the inlet to
 * 25% chord. Touches both the inlet column and the bottom wall -- the
 * blocked-path bounce-back rules handle wall-touching solids, and Zou-He
 * skips solid inlet cells (partial-height inflow).
 */
export function backwardStepPreset(nx: number, ny: number, height: number): PresetResult {
  const mask = new Uint8Array(nx * ny);
  const xEnd = Math.round(0.25 * nx);
  const h = Math.min(Math.round(height), ny - 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x <= xEnd; x++) {
      mask[y * nx + x] = 1;
    }
  }
  return { mask, d: h };
}
