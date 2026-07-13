// Shared helpers for tests. Deliberately minimal: the real preset
// rasterizers (with UI integration) arrive in Phase 3's src/geometry/.

/** Rasterize a filled circle into a 0/1 mask (1 = solid), indexed y*nx + x. */
export function rasterizeCircleMask(
  nx: number,
  ny: number,
  cx: number,
  cy: number,
  r: number,
): Uint8Array {
  const mask = new Uint8Array(nx * ny);
  const r2 = r * r;
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const dx = x - cx;
      const dy = y - cy;
      if (dx * dx + dy * dy <= r2) mask[y * nx + x] = 1;
    }
  }
  return mask;
}
