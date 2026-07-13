export interface GpuContext {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}

export async function initGPU(canvas: HTMLCanvasElement): Promise<GpuContext | null> {
  // @webgpu/types declares `navigator.gpu` as always-present (it types the
  // API shape, not runtime availability), so TS considers this check
  // tautological -- but on a non-WebGPU browser it is genuinely undefined,
  // which is exactly the case this feature-detects.
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
  if (!navigator.gpu) return null;

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;

  // Request the adapter's full storage-buffer limit up front: Phase 6's
  // 2048x1024 mode needs it, and it's cheap to ask for now vs. awkward to
  // retrofit onto an already-created device later.
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize },
  });
  device.lost
    .then((info) => {
      console.error('WebGPU device lost:', info.message);
    })
    .catch((err: unknown) => {
      console.error('Error awaiting device.lost:', err);
    });

  const context = canvas.getContext('webgpu');
  if (!context) return null;

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  return { device, context, format };
}

// Tracks the display canvas's backing-store size against its CSS/layout size
// (DPR-aware). This is unrelated to the LBM lattice resolution (512x256 /
// 1024x512 / 2048x1024), which is an explicit, rarely-changed user control
// requiring a full buffer reallocation -- that's Phase 3+ UI work, not a
// "resize" concern. WebGPU auto-tracks canvas.width/height on the next
// getCurrentTexture() call, so no context.configure() re-call is needed here.
export function observeCanvasResize(canvas: HTMLCanvasElement, device: GPUDevice): void {
  const maxDim = device.limits.maxTextureDimension2D;
  new ResizeObserver((entries) => {
    for (const entry of entries) {
      const dpr = window.devicePixelRatio || 1;
      const inlineSize = entry.contentBoxSize[0]?.inlineSize ?? canvas.clientWidth;
      const blockSize = entry.contentBoxSize[0]?.blockSize ?? canvas.clientHeight;
      const w = Math.max(1, Math.min(maxDim, Math.round(inlineSize * dpr)));
      const h = Math.max(1, Math.min(maxDim, Math.round(blockSize * dpr)));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    }
  }).observe(canvas);
}
