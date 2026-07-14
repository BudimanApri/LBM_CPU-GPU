import renderSource from './shaders/render.wgsl?raw';
import lbmSource from './shaders/lbm.wgsl?raw';
import brushSource from './shaders/brush.wgsl?raw';
import paramsSource from './shaders/params.wgsl?raw';
import d2q9Constants from './shaders/generated/d2q9-constants.wgsl?raw';
import type { LbmBuffers } from './buffers.ts';

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
    code: d2q9Constants + '\n' + paramsSource + '\n' + lbmSource,
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

export interface BrushPipelines {
  apply: GPUComputePipeline;
  applyBindGroup: GPUBindGroup;
  diff: GPUComputePipeline;
  diffBindGroup: GPUBindGroup;
  reconcile: GPUComputePipeline;
  /** Parity variants: reconcileBindGroups[step % 2] writes f[step % 2]. */
  reconcileBindGroups: readonly [GPUBindGroup, GPUBindGroup];
  workgroupsX: number;
  workgroupsY: number;
}

export function createBrushPipelines(device: GPUDevice, buffers: LbmBuffers): BrushPipelines {
  const module = device.createShaderModule({
    label: 'brush',
    code: d2q9Constants + '\n' + brushSource,
  });
  const makePipeline = (entryPoint: string): GPUComputePipeline =>
    device.createComputePipeline({
      label: `brush-${entryPoint}`,
      layout: 'auto',
      compute: { module, entryPoint },
    });
  const apply = makePipeline('brush_apply');
  const diff = makePipeline('mask_diff');
  const reconcile = makePipeline('mask_reconcile');

  const applyBindGroup = device.createBindGroup({
    label: 'brush-apply',
    layout: apply.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.brushParams } },
      { binding: 1, resource: { buffer: buffers.mask } },
    ],
  });
  const diffBindGroup = device.createBindGroup({
    label: 'brush-diff',
    layout: diff.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.brushParams } },
      { binding: 1, resource: { buffer: buffers.mask } },
      { binding: 2, resource: { buffer: buffers.maskPrev } },
      { binding: 3, resource: { buffer: buffers.changed } },
    ],
  });
  const makeReconcileBindGroup = (fCur: GPUBuffer, label: string): GPUBindGroup =>
    device.createBindGroup({
      label,
      layout: reconcile.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.brushParams } },
        { binding: 1, resource: { buffer: buffers.mask } },
        { binding: 2, resource: { buffer: buffers.maskPrev } },
        { binding: 3, resource: { buffer: buffers.changed } },
        { binding: 4, resource: { buffer: fCur } },
        { binding: 5, resource: { buffer: buffers.rho } },
        { binding: 6, resource: { buffer: buffers.ux } },
        { binding: 7, resource: { buffer: buffers.uy } },
      ],
    });
  return {
    apply,
    applyBindGroup,
    diff,
    diffBindGroup,
    reconcile,
    reconcileBindGroups: [
      makeReconcileBindGroup(buffers.f[0], 'brush-reconcile-A'),
      makeReconcileBindGroup(buffers.f[1], 'brush-reconcile-B'),
    ],
    workgroupsX: Math.ceil(buffers.nx / 8),
    workgroupsY: Math.ceil(buffers.ny / 8),
  };
}

export interface RenderPipeline {
  pipeline: GPURenderPipeline;
  bindGroup: GPUBindGroup;
}

export function createRenderPipeline(
  device: GPUDevice,
  buffers: LbmBuffers,
  format: GPUTextureFormat,
): RenderPipeline {
  const module = device.createShaderModule({
    label: 'render',
    code: paramsSource + '\n' + renderSource,
  });
  const pipeline = device.createRenderPipeline({
    label: 'render',
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_main' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const bindGroup = device.createBindGroup({
    label: 'render',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.params } },
      { binding: 1, resource: { buffer: buffers.ux } },
      { binding: 2, resource: { buffer: buffers.uy } },
      { binding: 3, resource: { buffer: buffers.mask } },
    ],
  });
  return { pipeline, bindGroup };
}
