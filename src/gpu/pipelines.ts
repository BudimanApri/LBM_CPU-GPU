import renderSource from './shaders/render.wgsl?raw';
import lbmSource from './shaders/lbm.wgsl?raw';
import brushSource from './shaders/brush.wgsl?raw';
import dyeSource from './shaders/dye.wgsl?raw';
import particlesSource from './shaders/particles.wgsl?raw';
import forcesSource from './shaders/forces.wgsl?raw';
import paramsSource from './shaders/params.wgsl?raw';
import commonSource from './shaders/common.wgsl?raw';
import d2q9Constants from './shaders/generated/d2q9-constants.wgsl?raw';
import { PARTICLE_COUNT, forceWorkgroups, type LbmBuffers } from './buffers.ts';

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

export interface DyePipeline {
  pipeline: GPUComputePipeline;
  /** Exposed for tests -- GPUComputePipeline doesn't expose its source module. */
  module: GPUShaderModule;
  /** Parity variants: bindGroups[step % 2] reads dye[step % 2], writes dye[(step+1) % 2]. */
  bindGroups: readonly [GPUBindGroup, GPUBindGroup];
  workgroupsX: number;
  workgroupsY: number;
}

export function createDyePipeline(device: GPUDevice, buffers: LbmBuffers): DyePipeline {
  const module = device.createShaderModule({
    label: 'dye',
    code: commonSource + '\n' + paramsSource + '\n' + dyeSource,
  });
  const pipeline = device.createComputePipeline({
    label: 'dye',
    layout: 'auto',
    compute: { module, entryPoint: 'dye_step' },
  });
  const layout = pipeline.getBindGroupLayout(0);
  const makeBindGroup = (src: GPUBuffer, dst: GPUBuffer, label: string): GPUBindGroup =>
    device.createBindGroup({
      label,
      layout,
      entries: [
        { binding: 0, resource: { buffer: buffers.params } },
        { binding: 1, resource: { buffer: buffers.ux } },
        { binding: 2, resource: { buffer: buffers.uy } },
        { binding: 3, resource: { buffer: buffers.mask } },
        { binding: 4, resource: { buffer: src } },
        { binding: 5, resource: { buffer: dst } },
      ],
    });
  return {
    pipeline,
    module,
    bindGroups: [
      makeBindGroup(buffers.dye[0], buffers.dye[1], 'dye-A-to-B'),
      makeBindGroup(buffers.dye[1], buffers.dye[0], 'dye-B-to-A'),
    ],
    workgroupsX: Math.ceil(buffers.nx / 8),
    workgroupsY: Math.ceil(buffers.ny / 8),
  };
}

export interface ParticlePipelines {
  computePipeline: GPUComputePipeline;
  computeBindGroup: GPUBindGroup;
  renderPipeline: GPURenderPipeline;
  renderBindGroup: GPUBindGroup;
  /** Exposed for tests -- GPU*Pipeline objects don't expose their source module. */
  module: GPUShaderModule;
  workgroups: number;
}

export function createParticlePipelines(
  device: GPUDevice,
  buffers: LbmBuffers,
  format: GPUTextureFormat,
): ParticlePipelines {
  const module = device.createShaderModule({
    label: 'particles',
    code: commonSource + '\n' + paramsSource + '\n' + particlesSource,
  });
  const computePipeline = device.createComputePipeline({
    label: 'particles-step',
    layout: 'auto',
    compute: { module, entryPoint: 'particles_step' },
  });
  const computeBindGroup = device.createBindGroup({
    label: 'particles-step',
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.params } },
      { binding: 1, resource: { buffer: buffers.ux } },
      { binding: 2, resource: { buffer: buffers.uy } },
      { binding: 3, resource: { buffer: buffers.mask } },
      { binding: 4, resource: { buffer: buffers.particles } },
    ],
  });

  const renderPipeline = device.createRenderPipeline({
    label: 'particles-render',
    layout: 'auto',
    vertex: { module, entryPoint: 'particles_vs' },
    fragment: {
      module,
      entryPoint: 'particles_fs',
      targets: [
        {
          format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-list' },
  });
  const renderBindGroup = device.createBindGroup({
    label: 'particles-render',
    // layout: 'auto' derives the bind group layout from only the bindings
    // particles_vs/particles_fs actually reference -- solid_mask (binding
    // 3) is declared in the module but unused by these entry points, so it
    // is NOT part of this pipeline's layout; omit it here to match.
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.params } },
      { binding: 1, resource: { buffer: buffers.ux } },
      { binding: 2, resource: { buffer: buffers.uy } },
      { binding: 5, resource: { buffer: buffers.particles } },
    ],
  });

  return {
    computePipeline,
    computeBindGroup,
    renderPipeline,
    renderBindGroup,
    module,
    workgroups: Math.ceil(PARTICLE_COUNT / 64),
  };
}

export interface ForcePipelines {
  accumulate: GPUComputePipeline;
  /** Parity variants: accumulateBindGroups[step % 2] reads f[step % 2]. */
  accumulateBindGroups: readonly [GPUBindGroup, GPUBindGroup];
  reduce: GPUComputePipeline;
  reduceBindGroup: GPUBindGroup;
  /** Exposed for tests -- GPUComputePipeline doesn't expose its source module. */
  module: GPUShaderModule;
  /** Stage-1 dispatch dims; forces.wgsl indexes partials by workgroup id. */
  workgroupsX: number;
  workgroupsY: number;
}

export function createForcePipelines(device: GPUDevice, buffers: LbmBuffers): ForcePipelines {
  const module = device.createShaderModule({
    label: 'forces',
    code: d2q9Constants + '\n' + paramsSource + '\n' + forcesSource,
  });
  const accumulate = device.createComputePipeline({
    label: 'forces-accumulate',
    layout: 'auto',
    compute: { module, entryPoint: 'forces_accumulate' },
  });
  const reduce = device.createComputePipeline({
    label: 'forces-reduce',
    layout: 'auto',
    compute: { module, entryPoint: 'forces_reduce' },
  });
  const accLayout = accumulate.getBindGroupLayout(0);
  const makeAcc = (fbuf: GPUBuffer, label: string): GPUBindGroup =>
    device.createBindGroup({
      label,
      layout: accLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.params } },
        { binding: 1, resource: { buffer: fbuf } },
        { binding: 2, resource: { buffer: buffers.mask } },
        { binding: 3, resource: { buffer: buffers.forcePartials } },
      ],
    });
  // Stage 2 uses only partials + result (auto-layout drops params/f/mask).
  const reduceBindGroup = device.createBindGroup({
    label: 'forces-reduce',
    layout: reduce.getBindGroupLayout(0),
    entries: [
      { binding: 3, resource: { buffer: buffers.forcePartials } },
      { binding: 4, resource: { buffer: buffers.forceResult } },
    ],
  });
  const wg = forceWorkgroups(buffers.nx, buffers.ny);
  return {
    accumulate,
    accumulateBindGroups: [
      makeAcc(buffers.f[0], 'forces-acc-A'),
      makeAcc(buffers.f[1], 'forces-acc-B'),
    ],
    reduce,
    reduceBindGroup,
    module,
    workgroupsX: wg.x,
    workgroupsY: wg.y,
  };
}

export interface RenderPipeline {
  pipeline: GPURenderPipeline;
  /** Parity variants: bindGroups[dyeIndex] reads buffers.dye[dyeIndex]. */
  bindGroups: readonly [GPUBindGroup, GPUBindGroup];
}

export function createRenderPipeline(
  device: GPUDevice,
  buffers: LbmBuffers,
  format: GPUTextureFormat,
): RenderPipeline {
  const module = device.createShaderModule({
    label: 'render',
    code: commonSource + '\n' + paramsSource + '\n' + renderSource,
  });
  const pipeline = device.createRenderPipeline({
    label: 'render',
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_main' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const layout = pipeline.getBindGroupLayout(0);
  const makeBindGroup = (dye: GPUBuffer, label: string): GPUBindGroup =>
    device.createBindGroup({
      label,
      layout,
      entries: [
        { binding: 0, resource: { buffer: buffers.params } },
        { binding: 1, resource: { buffer: buffers.ux } },
        { binding: 2, resource: { buffer: buffers.uy } },
        { binding: 3, resource: { buffer: buffers.rho } },
        { binding: 4, resource: { buffer: buffers.mask } },
        { binding: 5, resource: { buffer: dye } },
      ],
    });
  return {
    pipeline,
    bindGroups: [
      makeBindGroup(buffers.dye[0], 'render-dye-A'),
      makeBindGroup(buffers.dye[1], 'render-dye-B'),
    ],
  };
}
