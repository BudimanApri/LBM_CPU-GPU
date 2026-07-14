/** Hysteresis controller for solver substeps per rendered frame. */
export class AdaptiveKController {
  private frameAverageMs: number;
  private framesInWindow = 0;
  readonly min: number;
  readonly max: number;
  readonly targetFrameMs: number;
  private current: number;

  constructor(min: number, max: number, targetFrameMs: number, current: number) {
    this.min = min;
    this.max = max;
    this.targetFrameMs = targetFrameMs;
    this.current = current;
    this.frameAverageMs = targetFrameMs;
  }

  value(): number {
    return this.current;
  }

  set(value: number): number {
    this.current = Math.max(this.min, Math.min(this.max, Math.round(value)));
    this.framesInWindow = 0;
    return this.current;
  }

  /**
   * Returns a new K only when a 45-frame evidence window crosses a hysteresis
   * boundary. gpuDrainMs is measured asynchronously with
   * queue.onSubmittedWorkDone(), so queued GPU overload can lower K even when
   * rAF remains deceptively vsync-limited.
   */
  observe(frameMs: number, gpuDrainMs: number | null): number | null {
    this.frameAverageMs = 0.9 * this.frameAverageMs + 0.1 * frameMs;
    this.framesInWindow++;
    if (this.framesInWindow < 45) return null;
    this.framesInWindow = 0;

    const overloaded =
      this.frameAverageMs > this.targetFrameMs * 1.12 ||
      (gpuDrainMs !== null && gpuDrainMs > this.targetFrameMs * 1.05);
    const hasHeadroom =
      this.frameAverageMs < this.targetFrameMs * 1.03 &&
      (gpuDrainMs === null || gpuDrainMs < this.targetFrameMs * 0.8);

    const previous = this.current;
    if (overloaded && this.current > this.min) this.current--;
    else if (hasHeadroom && this.current < this.max) this.current++;
    return this.current === previous ? null : this.current;
  }
}
