import renderSource from './shaders/render.wgsl?raw';
import lbmSource from './shaders/lbm.wgsl?raw';
import d2q9Constants from './shaders/generated/d2q9-constants.wgsl?raw';
import type { LbmBuffers } from './buffers.ts';

export function createGradientPipeline(
  device: GPUDevice,
  format: GPUTextureFormat,
): GPURenderPipeline {
  const module = device.createShaderModule({ code: renderSource });
  return device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_main' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
}

export interface LbmPipeline {
  pipeline: GPUComputePipeline;
  /**
   * Prebuilt ping-pong bind groups (gotcha #8 -- never recreate per frame).
   * bindGroups[step % 2] reads f[step % 2] and writes f[(step + 1) % 2].
   */
  bindGroups: readonly [GPUBindGroup, GPUBindGroup];
  workgroupsX: number;
  workgroupsY: number;
}

export function createLbmPipeline(device: GPUDevice, buffers: LbmBuffers): LbmPipeline {
  const module = device.createShaderModule({
    label: 'lbm-stream-collide',
    code: d2q9Constants + '\n' + lbmSource,
  });
  const pipeline = device.createComputePipeline({
    label: 'lbm-stream-collide',
    layout: 'auto',
    compute: { module, entryPoint: 'lbm_step' },
  });
  const layout = pipeline.getBindGroupLayout(0);
  const makeBindGroup = (src: GPUBuffer, dst: GPUBuffer, label: string): GPUBindGroup =>
    device.createBindGroup({
      label,
      layout,
      entries: [
        { binding: 0, resource: { buffer: buffers.params } },
        { binding: 1, resource: { buffer: src } },
        { binding: 2, resource: { buffer: dst } },
        { binding: 3, resource: { buffer: buffers.mask } },
        { binding: 4, resource: { buffer: buffers.rho } },
        { binding: 5, resource: { buffer: buffers.ux } },
        { binding: 6, resource: { buffer: buffers.uy } },
      ],
    });
  return {
    pipeline,
    bindGroups: [
      makeBindGroup(buffers.f[0], buffers.f[1], 'lbm-A-to-B'),
      makeBindGroup(buffers.f[1], buffers.f[0], 'lbm-B-to-A'),
    ],
    workgroupsX: Math.ceil(buffers.nx / 8),
    workgroupsY: Math.ceil(buffers.ny / 8),
  };
}
