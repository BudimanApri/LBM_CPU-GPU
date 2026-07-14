// Fullscreen visualization pass. Phase 3 placeholder: velocity-magnitude
// monochrome ramp plus a flat obstacle overlay -- just enough to see and
// draw the flow. Phase 4 replaces the fragment logic with the real view
// modes (turbo/viridis colormaps, vorticity, density, dye).
//
// The Params struct comes from params.wgsl, concatenated at module creation.

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> ux_in: array<f32>;
@group(0) @binding(2) var<storage, read> uy_in: array<f32>;
@group(0) @binding(3) var<storage, read> solid_mask: array<u32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let x = f32((vertexIndex << 1u) & 2u) * 2.0 - 1.0;
  let y = f32(vertexIndex & 2u) * 2.0 - 1.0;
  var out: VertexOutput;
  out.position = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let nx = i32(params.nx);
  let ny = i32(params.ny);
  // uv.y runs down the screen; lattice y runs up.
  let cx = clamp(i32(in.uv.x * f32(nx)), 0, nx - 1);
  let cy = clamp(i32((1.0 - in.uv.y) * f32(ny)), 0, ny - 1);
  let cell = cy * nx + cx;
  if (solid_mask[cell] != 0u) {
    return vec4f(0.42, 0.44, 0.48, 1.0);
  }
  let speed = length(vec2f(ux_in[cell], uy_in[cell]));
  let t = clamp(speed / (1.6 * max(params.inlet_u, 1e-6)), 0.0, 1.0);
  return vec4f(0.02 + 0.5 * t, 0.03 + 0.75 * t, 0.05 + 0.95 * t, 1.0);
}
