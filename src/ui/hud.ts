// Pure formatting for the HUD status readout -- fps, MLUPS, substeps,
// Re/tau, current obstacle, step count. No DOM/solver coupling: main.ts
// gathers the numbers, this just lays them out consistently.

export interface HudStats {
  fps: number;
  mlups: number;
  kSubsteps: number;
  re: number;
  /** Present only when the requested Re was clamped to a different tau. */
  reEffective?: number;
  tau: number;
  d: number;
  presetLabel: string;
  steps: number;
}

export function formatHud(s: HudStats): string {
  const eff =
    s.reEffective !== undefined && Math.round(s.reEffective) !== s.re
      ? ` (eff ${s.reEffective.toFixed(0)})`
      : '';
  return [
    `fps    ${s.fps.toFixed(0)}`,
    `MLUPS  ${s.mlups.toFixed(0)}`,
    `K      ${s.kSubsteps}`,
    `Re     ${s.re}${eff}`,
    `tau    ${s.tau.toFixed(4)}`,
    `D      ${s.d} cells (${s.presetLabel})`,
    `steps  ${s.steps}`,
  ].join('\n');
}
