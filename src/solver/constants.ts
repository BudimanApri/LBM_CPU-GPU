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

// Lattice speed of sound squared, cs^2 = 1/3.
export const CS2 = 1 / 3;
