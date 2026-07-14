/** Cubic smoothstep ramp evaluated in solver-step space. */
export function inletRampVelocity(
  start: number,
  target: number,
  completedSteps: number,
  totalSteps: number,
): number {
  if (totalSteps <= 0) return target;
  const linear = Math.max(0, Math.min(1, completedSteps / totalSteps));
  const smooth = linear * linear * (3 - 2 * linear);
  return start + (target - start) * smooth;
}
