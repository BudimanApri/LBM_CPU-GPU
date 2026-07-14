// Exercises the actual production render pipeline (createRenderPipeline)
// against known field values, rather than re-deriving the shader math in
// JS. Colormap ground truth comes from a tiny compute dispatch of the same
// common.wgsl functions the fragment shader calls, so this only checks
// render.wgsl's own wiring (which view mode reads which field, the
// normalization constants, the edge/solid overlay) -- the colormap
// functions themselves are covered by gpu-colormaps.test.ts.
import { describe, expect, it } from 'vitest';
import commonSource from '../src/gpu/shaders/common.wgsl?raw';
import { createLbmBuffers, writeParams, type SimParams } from '../src/gpu/buffers.ts';
import { createRenderPipeline } from '../src/gpu/pipelines.ts';

const NX = 6;
const NY = 6;

async function colormapRgb(
  device: GPUDevice,
  fn: 'turbo' | 'diverging',
  t: number,
): Promise<number[]> {
  const module = device.createShaderModule({
    code:
      commonSource +
      `
@group(0) @binding(0) var<storage, read_write> out_rgb: array<vec4f>;
@compute @workgroup_size(1)
fn probe() {
  out_rgb[0] = vec4f(${fn}(${t}), 0.0);
}
`,
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'probe' },
  });
  const buf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: buf } }],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  const staging = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(buf, 0, staging, 0, 16);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(staging.getMappedRange().slice(0));
  return [data[0]!, data[1]!, data[2]!];
}

async function renderPixel(
  device: GPUDevice,
  params: SimParams,
  fields: { rho?: number[]; ux?: number[]; uy?: number[]; dye?: number[]; mask?: number[] },
  px: number,
  py: number,
): Promise<number[]> {
  const buffers = createLbmBuffers(device, NX, NY);
  writeParams(device, buffers, params);
  const n = NX * NY;
  device.queue.writeBuffer(
    buffers.rho,
    0,
    new Float32Array(fields.rho ?? new Array<number>(n).fill(1)),
  );
  device.queue.writeBuffer(
    buffers.ux,
    0,
    new Float32Array(fields.ux ?? new Array<number>(n).fill(0)),
  );
  device.queue.writeBuffer(
    buffers.uy,
    0,
    new Float32Array(fields.uy ?? new Array<number>(n).fill(0)),
  );
  device.queue.writeBuffer(
    buffers.dye[0],
    0,
    new Float32Array(fields.dye ?? new Array<number>(n).fill(0)),
  );
  device.queue.writeBuffer(
    buffers.mask,
    0,
    Uint32Array.from(fields.mask ?? new Array<number>(n).fill(0)),
  );

  const format = 'rgba8unorm';
  const render = createRenderPipeline(device, buffers, format);
  const tex = device.createTexture({
    size: [NX, NY],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: tex.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      },
    ],
  });
  pass.setPipeline(render.pipeline);
  pass.setBindGroup(0, render.bindGroups[0]);
  pass.draw(3);
  pass.end();

  const bytesPerRow = 256;
  const buf = device.createBuffer({
    size: bytesPerRow * NY,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow }, [NX, NY, 1]);
  device.queue.submit([encoder.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const data = new Uint8Array(buf.getMappedRange().slice(0));
  const o = py * bytesPerRow + px * 4;
  return [data[o]!, data[o + 1]!, data[o + 2]!];
}

const baseParams = (viewMode: SimParams['viewMode']): SimParams => ({
  nx: NX,
  ny: NY,
  tau: 0.6,
  inletU: 0.05,
  periodicY: false,
  dyeEnabled: false,
  viewMode,
  stepIndex: 0,
  substeps: 1,
});

// uv->cell mapping: uv.x = (px+0.5)/NX -> cell x = px exactly for a render
// target the same size as the lattice. uv.y is DELIBERATELY flipped in the
// fragment shader (framebuffer top = highest lattice row, matching the
// wind-tunnel's on-screen "up"), so framebuffer row py maps to lattice row
// NY-1-py, not py. Probe lattice cell (3, 3) -- interior enough that its
// 4-neighbors are all valid (no edge clamping) -- read back at framebuffer
// pixel (3, NY-1-3).
const CELL_X = 3;
const CELL_Y = 3;
const PX = CELL_X;
const PY = NY - 1 - CELL_Y;

function toRgb255(c: number[]): number[] {
  return c.map((v) => Math.round(Math.max(0, Math.min(1, v)) * 255));
}

describe('render.wgsl view modes (production pipeline, known fields)', () => {
  it('velocity mode: turbo(speed / 1.8*u0) at the probed cell', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();
    const n = NX * NY;
    const ux = new Array<number>(n).fill(0);
    const uy = new Array<number>(n).fill(0);
    const idx = CELL_Y * NX + CELL_X;
    ux[idx] = 0.03;
    uy[idx] = 0.04; // speed = 0.05
    const pixel = await renderPixel(device, baseParams('velocity'), { ux, uy }, PX, PY);
    const expected = toRgb255(await colormapRgb(device, 'turbo', 0.05 / (1.8 * 0.05)));
    for (let k = 0; k < 3; k++) expect(Math.abs(pixel[k]! - expected[k]!)).toBeLessThanOrEqual(1);
    device.destroy();
  });

  it('vorticity mode: diverging(central-diff vorticity / 0.5*u0)', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();
    const n = NX * NY;
    // uy = 0.01 * x (linear ramp) -> d(uy)/dx = 0.01 exactly via central
    // difference; ux = 0 -> d(ux)/dy = 0. vorticity = 0.01.
    const uy = Array.from({ length: n }, (_, k) => 0.01 * (k % NX));
    const pixel = await renderPixel(device, baseParams('vorticity'), { uy }, PX, PY);
    const expected = toRgb255(await colormapRgb(device, 'diverging', 0.01 / (0.5 * 0.05)));
    for (let k = 0; k < 3; k++) expect(Math.abs(pixel[k]! - expected[k]!)).toBeLessThanOrEqual(1);
    device.destroy();
  });

  it('density mode: diverging((rho-1) / 2*u0^2)', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();
    const n = NX * NY;
    const rho = new Array<number>(n).fill(1);
    rho[CELL_Y * NX + CELL_X] = 1.002;
    const pixel = await renderPixel(device, baseParams('density'), { rho }, PX, PY);
    const expected = toRgb255(await colormapRgb(device, 'diverging', 0.002 / (2 * 0.05 * 0.05)));
    for (let k = 0; k < 3; k++) expect(Math.abs(pixel[k]! - expected[k]!)).toBeLessThanOrEqual(1);
    device.destroy();
  });

  it('paints non-finite density magenta for an unmistakable stability failure', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();
    const rho = new Array<number>(NX * NY).fill(1);
    rho[CELL_Y * NX + CELL_X] = NaN;
    const pixel = await renderPixel(device, baseParams('density'), { rho }, PX, PY);
    expect(pixel[0]).toBeGreaterThan(250);
    expect(pixel[1]).toBeLessThan(5);
    expect(pixel[2]).toBeGreaterThan(250);
    device.destroy();
  });

  it('dye mode: dark background blended toward bright by dye intensity', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();
    const n = NX * NY;
    const dye = new Array<number>(n).fill(0);
    dye[CELL_Y * NX + CELL_X] = 0.6;
    const pixel = await renderPixel(device, baseParams('dye'), { dye }, PX, PY);
    const expected = toRgb255([0.02 + 0.6 * 0.85, 0.03 + 0.6 * 0.87, 0.05 + 0.6 * 0.9]);
    for (let k = 0; k < 3; k++) expect(Math.abs(pixel[k]! - expected[k]!)).toBeLessThanOrEqual(1);
    device.destroy();
  });

  it('obstacle overlay: flat fill inside, bright outline at the boundary', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();
    const n = NX * NY;
    const mask = new Array<number>(n).fill(0);
    // 3x3 solid block at x,y in [1,3]: (2,2) is the one cell with all 4
    // neighbors solid (true flat-fill interior); (2,1) has a fluid neighbor
    // at (2,0) below it (a boundary cell). Framebuffer y is flipped from
    // lattice y (top of screen = highest lattice row).
    for (let y = 1; y <= 3; y++) {
      for (let x = 1; x <= 3; x++) mask[y * NX + x] = 1;
    }
    const interior = await renderPixel(device, baseParams('velocity'), { mask }, 2, NY - 1 - 2);
    const edge = await renderPixel(device, baseParams('velocity'), { mask }, 2, NY - 1 - 1);
    // Flat solid fill: (0.16, 0.17, 0.19) in [0,1] -> ~(41, 43, 48) in u8.
    expect(interior[0]).toBeGreaterThan(30);
    expect(interior[0]).toBeLessThan(55);
    expect(interior[1]! - interior[0]!).toBeLessThan(5); // near-neutral gray
    // Edge outline is much brighter than the flat interior fill.
    expect(edge[0]!).toBeGreaterThan(interior[0]! + 100);
    device.destroy();
  });
});
