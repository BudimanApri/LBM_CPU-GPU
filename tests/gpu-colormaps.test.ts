// Probes common.wgsl's colormap and bilinear-sample functions by actually
// compiling and dispatching them on the GPU (WGSL correctness can't be
// checked any other way -- there's no Node-side WGSL interpreter). Checks
// structural properties (range, monotonic trend, symmetry) rather than
// exact reference pixels, since the polynomial coefficients are a ported
// external fit, not derived here.
import { describe, expect, it } from 'vitest';
import commonSource from '../src/gpu/shaders/common.wgsl?raw';

const PROBE_N = 33; // t = 0, 1/32, ..., 1

const PROBE_SHADER = `
@group(0) @binding(0) var<storage, read_write> out_rgb: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> out_div: array<vec4f>;
@group(0) @binding(2) var<storage, read> field: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_bilinear: array<f32>;

@compute @workgroup_size(1)
fn probe() {
  let n = ${PROBE_N};
  for (var i = 0; i < n; i++) {
    let t = f32(i) / f32(n - 1);
    let turbo_c = turbo(t);
    let viridis_c = viridis(t);
    out_rgb[i] = vec4f(turbo_c, 0.0);
    out_rgb[i + n] = vec4f(viridis_c, 0.0);
    let s = t * 2.0 - 1.0; // -1..1
    out_div[i] = vec4f(diverging(s), 0.0);
  }
  // field (binding 2, written from JS before dispatch): 3x3, values =
  // 10*y + x (0..22). Bilinear-sample it at fractional/boundary coords.
  out_bilinear[0] = bilinear_sample(&field, 3, 3, 0.5, 0.5); // avg of 0,1,10,11 = 5.5
  out_bilinear[1] = bilinear_sample(&field, 3, 3, 1.0, 1.0); // exact node = 11
  out_bilinear[2] = bilinear_sample(&field, 3, 3, -5.0, -5.0); // clamps to (0,0) = 0
  out_bilinear[3] = bilinear_sample(&field, 3, 3, 50.0, 50.0); // clamps to (2,2) = 22
}
`;

interface Probe {
  turbo: Float32Array[];
  viridis: Float32Array[];
  diverging: Float32Array[];
  bilinear: Float32Array;
}

async function runProbe(): Promise<Probe> {
  const adapter = await navigator.gpu.requestAdapter();
  expect(adapter).not.toBeNull();
  const device = await adapter!.requestDevice();

  const module = device.createShaderModule({ code: commonSource + '\n' + PROBE_SHADER });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'probe' },
  });

  const rgbBuf = device.createBuffer({
    size: 2 * PROBE_N * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const divBuf = device.createBuffer({
    size: PROBE_N * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const fieldBuf = device.createBuffer({
    size: 9 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const bilinearBuf = device.createBuffer({
    size: 4 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: rgbBuf } },
      { binding: 1, resource: { buffer: divBuf } },
      { binding: 2, resource: { buffer: fieldBuf } },
      { binding: 3, resource: { buffer: bilinearBuf } },
    ],
  });
  device.queue.writeBuffer(fieldBuf, 0, new Float32Array([0, 1, 2, 10, 11, 12, 20, 21, 22]));

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();

  const stagingRgb = device.createBuffer({
    size: rgbBuf.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const stagingDiv = device.createBuffer({
    size: divBuf.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const stagingBilinear = device.createBuffer({
    size: bilinearBuf.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(rgbBuf, 0, stagingRgb, 0, rgbBuf.size);
  encoder.copyBufferToBuffer(divBuf, 0, stagingDiv, 0, divBuf.size);
  encoder.copyBufferToBuffer(bilinearBuf, 0, stagingBilinear, 0, bilinearBuf.size);
  device.queue.submit([encoder.finish()]);

  await Promise.all([
    stagingRgb.mapAsync(GPUMapMode.READ),
    stagingDiv.mapAsync(GPUMapMode.READ),
    stagingBilinear.mapAsync(GPUMapMode.READ),
  ]);
  const rgb = new Float32Array(stagingRgb.getMappedRange().slice(0));
  const div = new Float32Array(stagingDiv.getMappedRange().slice(0));
  const bilinear = new Float32Array(stagingBilinear.getMappedRange().slice(0));
  device.destroy();

  const chunk = (arr: Float32Array, offset: number): Float32Array[] =>
    Array.from({ length: PROBE_N }, (_, i) => arr.subarray((offset + i) * 4, (offset + i) * 4 + 3));
  return {
    turbo: chunk(rgb, 0),
    viridis: chunk(rgb, PROBE_N),
    diverging: chunk(div, 0),
    bilinear,
  };
}

describe('common.wgsl colormaps and bilinear sampling', () => {
  it('turbo and viridis stay in [0,1]^3 and span a visible range', async () => {
    const { turbo, viridis } = await runProbe();
    for (const c of [...turbo, ...viridis]) {
      for (const ch of c) {
        expect(ch).toBeGreaterThanOrEqual(0);
        expect(ch).toBeLessThanOrEqual(1);
      }
    }
    // Endpoints are distinguishable from the midpoint (not a constant map).
    const dist = (a: Float32Array, b: Float32Array): number =>
      Math.hypot(a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!);
    expect(dist(turbo[0]!, turbo[16]!)).toBeGreaterThan(0.3);
    expect(dist(turbo[16]!, turbo[32]!)).toBeGreaterThan(0.3);
    expect(dist(viridis[0]!, viridis[32]!)).toBeGreaterThan(0.3);
  });

  it('diverging is white at 0, and blue/red at the negative/positive extremes', async () => {
    const { diverging } = await runProbe();
    const mid = diverging[16]!; // t=0
    const lo = diverging[0]!; // t=-1
    const hi = diverging[32]!; // t=1
    expect(mid[0]).toBeGreaterThan(0.85);
    expect(mid[1]).toBeGreaterThan(0.85);
    expect(mid[2]).toBeGreaterThan(0.85);
    expect(lo[2]).toBeGreaterThan(lo[0]!); // blue-dominant
    expect(hi[0]).toBeGreaterThan(hi[2]!); // red-dominant
    // Symmetric magnitude of deviation from white between the two extremes.
    const devLo = Math.hypot(...[0, 1, 2].map((k) => lo[k]! - mid[k]!));
    const devHi = Math.hypot(...[0, 1, 2].map((k) => hi[k]! - mid[k]!));
    expect(Math.abs(devLo - devHi)).toBeLessThan(0.05);
  });

  it('bilinear_sample interpolates correctly, exactly, and clamps at the edges', async () => {
    const { bilinear } = await runProbe();
    expect(bilinear[0]).toBeCloseTo(5.5, 5);
    expect(bilinear[1]).toBeCloseTo(11, 5);
    expect(bilinear[2]).toBeCloseTo(0, 5);
    expect(bilinear[3]).toBeCloseTo(22, 5);
  });
});
