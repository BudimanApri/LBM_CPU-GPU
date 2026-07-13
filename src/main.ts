import './style.css';
import { initGPU, observeCanvasResize } from './gpu/context.ts';
import { createGradientPipeline } from './gpu/pipelines.ts';

function showFallback(): void {
  const fallback = document.getElementById('webgpu-fallback');
  const canvas = document.getElementById('gpu-canvas');
  fallback?.classList.remove('hidden');
  canvas?.remove();
}

async function main(): Promise<void> {
  const canvas = document.getElementById('gpu-canvas');
  if (!(canvas instanceof HTMLCanvasElement)) {
    showFallback();
    return;
  }

  const gpu = await initGPU(canvas);
  if (!gpu) {
    showFallback();
    return;
  }

  const { device, context, format } = gpu;
  observeCanvasResize(canvas, device);

  const pipeline = createGradientPipeline(device, format);

  function frame(): void {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((err: unknown) => {
  console.error('Fatal error during startup:', err);
  showFallback();
});
