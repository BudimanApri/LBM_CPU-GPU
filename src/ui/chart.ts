// Cd/Cl strip chart (hand-rolled Canvas2D, no chart lib -- CLAUDE.md's "no
// frameworks / thin UI" directive) plus the Strouhal estimator. The chart is
// the project's signature instrument: oscillating Cl at Re~100 is how the
// vortex shedding is read quantitatively, and its frequency gives St.
//
// Samples carry the absolute solver step at which the force was measured, not
// a frame or wall-clock time -- so the shedding period comes out in lattice
// time units regardless of substep count K or frame rate, and St = f D / U is
// exact.

export interface ForceSample {
  /** Total solver steps elapsed when this force was measured. */
  step: number;
  cd: number;
  cl: number;
}

const MAX_SAMPLES = 1000;

/**
 * Strouhal number from the Cl oscillation: St = f D / U, with f the shedding
 * frequency in inverse lattice time units. Frequency comes from the mean
 * interval between successive upward zero-crossings of (Cl - mean), linearly
 * interpolated to sub-step precision. Zero-crossing intervals are simpler
 * than an FFT and robust for the clean near-sinusoidal Re~100 signal
 * (CLAUDE.md). Returns null until a stable oscillation is present.
 */
export function strouhalFromSamples(
  samples: readonly ForceSample[],
  d: number,
  u: number,
): number | null {
  if (samples.length < 8 || d <= 0 || u <= 0) return null;
  let mean = 0;
  for (const s of samples) mean += s.cl;
  mean /= samples.length;

  // Upward crossings of the mean-subtracted signal, interpolated in `step`.
  const crossings: number[] = [];
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1]!.cl - mean;
    const b = samples[i]!.cl - mean;
    if (a <= 0 && b > 0) {
      const s0 = samples[i - 1]!.step;
      const s1 = samples[i]!.step;
      const frac = a === b ? 0 : a / (a - b); // a<=0, b>0 => frac in [0,1)
      crossings.push(s0 + frac * (s1 - s0));
    }
  }
  if (crossings.length < 2) return null;

  let period = 0;
  for (let i = 1; i < crossings.length; i++) period += crossings[i]! - crossings[i - 1]!;
  period /= crossings.length - 1;
  if (period <= 0) return null;

  // Reject a flat/noise signal: amplitude must be a meaningful fraction of
  // the drag scale, else "crossings" are just roundoff wander.
  let min = Infinity;
  let max = -Infinity;
  for (const s of samples) {
    min = Math.min(min, s.cl);
    max = Math.max(max, s.cl);
  }
  if (max - min < 1e-3) return null;

  return d / (period * u); // f = 1/period, St = f D / U
}

/**
 * Time-mean of Cd and Cl over the buffered samples. Cd's validation target is
 * a time-mean (CLAUDE.md: "Cd converges to ~2.0"), and averaging also cancels
 * the domain acoustic ripple that rides on the instantaneous force, so this is
 * the number to trust for the coefficient readout. Returns null when empty.
 */
export function meanCoefficients(
  samples: readonly ForceSample[],
): { cd: number; cl: number } | null {
  if (samples.length === 0) return null;
  let cd = 0;
  let cl = 0;
  for (const s of samples) {
    cd += s.cd;
    cl += s.cl;
  }
  return { cd: cd / samples.length, cl: cl / samples.length };
}

export interface StripChart {
  readonly canvas: HTMLCanvasElement;
  push(sample: ForceSample): void;
  render(): void;
  latest(): ForceSample | null;
  mean(): { cd: number; cl: number } | null;
  sampleCount(): number;
  clear(): void;
  strouhal(d: number, u: number): number | null;
}

const COLORS = {
  bg: '#0b0e13',
  grid: '#1a2029',
  zero: '#2a333f',
  cd: '#3ba7c9',
  cl: '#d98a3b',
  text: '#7d8794',
};

export function createStripChart(cssWidth = 272, cssHeight = 110): StripChart {
  const canvas = document.createElement('canvas');
  canvas.id = 'force-chart';
  const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);

  const w = cssWidth;
  const h = cssHeight;
  const samples: ForceSample[] = [];

  function push(sample: ForceSample): void {
    samples.push(sample);
    if (samples.length > MAX_SAMPLES) samples.shift();
  }

  function latest(): ForceSample | null {
    return samples.length > 0 ? samples[samples.length - 1]! : null;
  }

  function mean(): { cd: number; cl: number } | null {
    return meanCoefficients(samples);
  }

  function sampleCount(): number {
    return samples.length;
  }

  function clear(): void {
    samples.length = 0;
    render();
  }

  function strouhal(d: number, u: number): number | null {
    return strouhalFromSamples(samples, d, u);
  }

  // Symmetric autoscale about zero so the Cl sign (up/down shedding) reads
  // directly and Cd sits in the same frame.
  function drawSeries(pick: (s: ForceSample) => number, color: string, half: number): void {
    if (samples.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < samples.length; i++) {
      const x = (i / (samples.length - 1)) * w;
      const y = h / 2 - (pick(samples[i]!) / half) * (h / 2 - 4);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function render(): void {
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);
    // Zero line.
    ctx.strokeStyle = COLORS.zero;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    let half = 0.5;
    for (const s of samples) half = Math.max(half, Math.abs(s.cd), Math.abs(s.cl));
    half *= 1.1;

    drawSeries((s) => s.cd, COLORS.cd, half);
    drawSeries((s) => s.cl, COLORS.cl, half);

    ctx.font = '10px ui-monospace, Menlo, Consolas, monospace';
    ctx.textBaseline = 'top';
    ctx.fillStyle = COLORS.cd;
    ctx.fillText('Cd', 4, 3);
    ctx.fillStyle = COLORS.cl;
    ctx.fillText('Cl', 26, 3);
    ctx.fillStyle = COLORS.text;
    ctx.textAlign = 'right';
    ctx.fillText(`±${half.toFixed(2)}`, w - 4, 3);
    ctx.textAlign = 'left';
  }

  render();
  return { canvas, push, render, latest, mean, sampleCount, clear, strouhal };
}
