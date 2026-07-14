// NACA 4-digit airfoil generator. Digits MPXX: M = max camber in % of
// chord, P = camber position in tenths of chord, XX = thickness in % of
// chord (e.g. 4412: 4% camber at 0.4c, 12% thick).
//
// Uses the CLOSED trailing-edge thickness coefficient (-0.1036 instead of
// the classic open-TE -0.1015): the ~0.02%c shape difference is irrelevant
// to the flow at lattice resolution, but a closed polygon is required for a
// gap-free rasterized mask -- an open TE risks a one-cell leak through the
// obstacle, which is a correctness bug, not a cosmetic one.

export interface Point {
  x: number;
  y: number;
}

/** Half-thickness distribution y_t(x/c), closed trailing edge. */
export function nacaHalfThickness(t: number, xc: number): number {
  return (
    5 *
    t *
    (0.2969 * Math.sqrt(xc) -
      0.126 * xc -
      0.3516 * xc * xc +
      0.2843 * xc * xc * xc -
      0.1036 * xc * xc * xc * xc)
  );
}

/** Camber line y_c(x/c) and its slope for camber m at position p. */
export function nacaCamber(m: number, p: number, xc: number): { yc: number; dyc: number } {
  if (m === 0) return { yc: 0, dyc: 0 };
  if (xc < p) {
    return {
      yc: (m / (p * p)) * (2 * p * xc - xc * xc),
      dyc: ((2 * m) / (p * p)) * (p - xc),
    };
  }
  const q = 1 - p;
  return {
    yc: (m / (q * q)) * (1 - 2 * p + 2 * p * xc - xc * xc),
    dyc: ((2 * m) / (q * q)) * (p - xc),
  };
}

/**
 * Closed polygon for a NACA 4-digit airfoil.
 *
 * Coordinates: chord units of `chord`, quarter-chord point at the origin
 * (the natural placement anchor and AoA pivot), y up. Positive `alphaDeg`
 * pitches the nose up (clockwise rotation for flow in +x). Points run from
 * the trailing edge over the upper surface to the leading edge, then back
 * along the lower surface; cosine spacing clusters stations at both ends.
 */
export function nacaPolygon(digits: string, chord: number, alphaDeg: number): Point[] {
  if (!/^\d{4}$/.test(digits)) {
    throw new RangeError(`NACA designation must be 4 digits, got "${digits}"`);
  }
  const m = Number(digits[0]) / 100;
  const p = Number(digits[1]) / 10;
  const t = Number(digits.slice(2)) / 100;
  if (m > 0 && p === 0) {
    throw new RangeError(`NACA ${digits} is invalid: camber ${digits[0]} needs a position digit`);
  }
  if (t === 0) {
    throw new RangeError(`NACA ${digits} has zero thickness`);
  }

  const N = 80; // stations per surface
  const upper: Point[] = [];
  const lower: Point[] = [];
  for (let k = 0; k <= N; k++) {
    const xc = (1 - Math.cos((k * Math.PI) / N)) / 2; // cosine spacing, 0..1
    const yt = nacaHalfThickness(t, xc);
    const { yc, dyc } = nacaCamber(m, p, xc);
    const theta = Math.atan(dyc);
    upper.push({ x: xc - yt * Math.sin(theta), y: yc + yt * Math.cos(theta) });
    lower.push({ x: xc + yt * Math.sin(theta), y: yc - yt * Math.cos(theta) });
  }

  // TE -> upper -> LE, then LE -> lower -> TE (skip the duplicated LE point).
  const unit = [...upper.reverse(), ...lower.slice(1)];

  // Scale to chord, pivot to quarter-chord, rotate clockwise by alpha
  // (positive alpha = nose up for flow in +x).
  const a = (alphaDeg * Math.PI) / 180;
  const cosA = Math.cos(a);
  const sinA = Math.sin(a);
  return unit.map(({ x, y }) => {
    const px = (x - 0.25) * chord;
    const py = y * chord;
    return { x: px * cosA + py * sinA, y: -px * sinA + py * cosA };
  });
}
