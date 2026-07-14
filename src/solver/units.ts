// Re <-> tau <-> nu conversions and the stability-envelope clamps that
// CLAUDE.md requires to be enforced in code: tau in [0.51, 1.5] (never
// tau <= 0.5), inlet velocity U in [0.02, 0.1] (low-Mach limit).
import { CS2 } from './constants.ts';

export const TAU_MIN = 0.51;
export const TAU_MAX = 1.5;
export const U_MIN = 0.02;
export const U_MAX = 0.1;

/** Kinematic viscosity in lattice units: nu = cs^2 (tau - 0.5). */
export function nuFromTau(tau: number): number {
  return CS2 * (tau - 0.5);
}

export function tauFromNu(nu: number): number {
  return nu / CS2 + 0.5;
}

/** Re = U D / nu, with D the characteristic obstacle size in lattice cells. */
export function reynolds(u: number, d: number, nu: number): number {
  return (u * d) / nu;
}

export function clampTau(tau: number): number {
  return Math.min(TAU_MAX, Math.max(TAU_MIN, tau));
}

export function clampU(u: number): number {
  return Math.min(U_MAX, Math.max(U_MIN, u));
}

export interface TauSolution {
  /** Relaxation time, clamped into [TAU_MIN, TAU_MAX]. */
  tau: number;
  /** Viscosity actually realized by the clamped tau. */
  nu: number;
  /** Reynolds number actually realized (differs from the request when clamped). */
  reEffective: number;
  /**
   * True when the requested Re was unreachable at this U and D. The caller
   * should respond by raising resolution (larger D) or enabling the
   * Smagorinsky LES term (Phase 6) -- never by letting tau collapse.
   */
  clamped: boolean;
}

/**
 * Reynolds number is the user-facing control: hold U fixed and solve for tau.
 * nu = U D / Re, tau = nu / cs^2 + 0.5, then clamp into the stability envelope.
 */
export function solveTauForRe(re: number, u: number, d: number): TauSolution {
  if (!Number.isFinite(re) || re <= 0) {
    throw new RangeError(`Re must be a positive finite number, got ${re}`);
  }
  if (!Number.isFinite(u) || u <= 0) {
    throw new RangeError(`U must be a positive finite number, got ${u}`);
  }
  if (!Number.isFinite(d) || d <= 0) {
    throw new RangeError(`D must be a positive finite number, got ${d}`);
  }
  const tauRequested = tauFromNu((u * d) / re);
  const tau = clampTau(tauRequested);
  const nu = nuFromTau(tau);
  return { tau, nu, reEffective: reynolds(u, d, nu), clamped: tau !== tauRequested };
}

/** Normalize 2D force per unit span: C = 2F / (rho U^2 Lref). */
export function normalizeForce(
  fx: number,
  fy: number,
  rho: number,
  u: number,
  referenceLength: number,
): { cd: number; cl: number } {
  const denominator = rho * u * u * referenceLength;
  if (!Number.isFinite(denominator) || denominator <= 0) {
    throw new RangeError('rho, U, and coefficient reference length must form a positive value');
  }
  return { cd: (2 * fx) / denominator, cl: (2 * fy) / denominator };
}

/** CPU mirror of the local WGSL Smagorinsky relaxation, useful for tests. */
export function smagorinskyTau(
  molecularTau: number,
  cs: number,
  piNorm: number,
  rho: number,
): number {
  const tauTurb =
    0.5 *
    (Math.sqrt(molecularTau * molecularTau + (18 * cs * cs * piNorm) / Math.max(rho, 1e-12)) -
      molecularTau);
  return clampTau(molecularTau + tauTurb);
}
