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
  coefficientLength: number;
  coefficientReference: 'frontal' | 'chord';
  steps: number;
  /** Drag coefficient from the latest force readback. */
  cd: number;
  /** Lift coefficient from the latest force readback. */
  cl: number;
  /** Strouhal number once a stable Cl oscillation is detected, else null. */
  st: number | null;
  resolution: string;
  workgroup: string;
  lesEnabled: boolean;
  nanDetected: boolean;
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
    `grid   ${s.resolution}  wg ${s.workgroup}`,
    `LES    ${s.lesEnabled ? 'on' : 'off'}`,
    `Re     ${s.re}${eff}`,
    `tau    ${s.tau.toFixed(4)}`,
    `D      ${s.d} cells (${s.presetLabel})`,
    `Lref   ${s.coefficientLength} cells (${s.coefficientReference})`,
    `Cd     ${s.cd.toFixed(3)}`,
    `Cl     ${s.cl.toFixed(3)}`,
    `St     ${s.st !== null ? s.st.toFixed(3) : '--'}`,
    `steps  ${s.steps}`,
    ...(s.nanDetected ? ['ERROR  non-finite density; paused'] : []),
  ].join('\n');
}
