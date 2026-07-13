// D2Q9 lattice tables -- single source of truth for both the CPU reference
// solver and the WGSL codegen (see scripts/wgsl-codegen.ts). Ordering must
// match CLAUDE.md's D2Q9 table exactly: index i, velocity (cx,cy), weight w,
// opposite direction index.
export const Q = 9;

export const CX: readonly number[] = [0, 1, 0, -1, 0, 1, -1, -1, 1];
export const CY: readonly number[] = [0, 0, 1, 0, -1, 1, 1, -1, -1];

export const WEIGHTS: readonly number[] = [
  4 / 9,
  1 / 9,
  1 / 9,
  1 / 9,
  1 / 9,
  1 / 36,
  1 / 36,
  1 / 36,
  1 / 36,
];

export const OPPOSITE: readonly number[] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

// Specular reflection about a horizontal wall (cy flips, cx is preserved).
// Used by free-slip top/bottom boundaries. Derived from CX/CY rather than
// hand-written so it can never diverge from the velocity set.
export const SPECULAR_Y: readonly number[] = CX.map((cx, i) => {
  for (let j = 0; j < Q; j++) {
    if (CX[j] === cx && CY[j] === -CY[i]!) return j;
  }
  throw new Error(`no specular partner for direction ${i}`);
});

// Lattice speed of sound squared, cs^2 = 1/3.
export const CS2 = 1 / 3;
