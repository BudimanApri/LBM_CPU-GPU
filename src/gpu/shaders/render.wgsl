// Fullscreen visualization pass. View mode selected by params.flags bits
// 4-5 (VIEW_MODE_SHIFT/VIEW_MODE_MASK in buffers.ts): 0 = velocity
// magnitude (turbo), 1 = vorticity (diverging), 2 = density deviation from
// 1 (diverging), 3 = dye (monochrome). Obstacles render as a crisp
// edge-detected outline over any mode. Params/colormap/bilinear helpers
// come from params.wgsl / common.wgsl, concatenated at module creation.

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> ux_in: array<f32>;
@group(0) @binding(2) var<storage, read> uy_in: array<f32>;
@group(0) @binding(3) var<storage, read> rho_in: array<f32>;
@group(0) @binding(4) var<storage, read> solid_mask: array<u32>;
@group(0) @binding(5) var<storage, read> dye_in: array<f32>;

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

fn cell_index(x: i32, y: i32, nx: i32, ny: i32) -> i32 {
  return clamp(y, 0, ny - 1) * nx + clamp(x, 0, nx - 1);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let nx = i32(params.nx);
  let ny = i32(params.ny);
  // uv.y runs down the screen; lattice y runs up.
  let x = clamp(i32(in.uv.x * f32(nx)), 0, nx - 1);
  let y = clamp(i32((1.0 - in.uv.y) * f32(ny)), 0, ny - 1);
  let cell = y * nx + x;

  // Edge-detected obstacle outline: any 4-neighbor on the opposite side of
  // the mask boundary marks this pixel as an edge, drawn over both sides.
  let here = solid_mask[cell];
  let is_edge =
    here != solid_mask[cell_index(x - 1, y, nx, ny)] ||
    here != solid_mask[cell_index(x + 1, y, nx, ny)] ||
    here != solid_mask[cell_index(x, y - 1, nx, ny)] ||
    here != solid_mask[cell_index(x, y + 1, nx, ny)];
  if (is_edge) {
    return vec4f(0.85, 0.87, 0.9, 1.0);
  }
  if (here != 0u) {
    return vec4f(0.16, 0.17, 0.19, 1.0);
  }

  let view_mode = (params.flags >> VIEW_MODE_SHIFT) & VIEW_MODE_MASK;
  let u0 = max(params.inlet_u, 1e-6);

  if (view_mode == 0u) {
    let speed = length(vec2f(ux_in[cell], uy_in[cell]));
    return vec4f(turbo(clamp(speed / (1.8 * u0), 0.0, 1.0)), 1.0);
  }
  if (view_mode == 1u) {
    // Central-difference vorticity: d(uy)/dx - d(ux)/dy.
    let duy_dx =
      (uy_in[cell_index(x + 1, y, nx, ny)] - uy_in[cell_index(x - 1, y, nx, ny)]) * 0.5;
    let dux_dy =
      (ux_in[cell_index(x, y + 1, nx, ny)] - ux_in[cell_index(x, y - 1, nx, ny)]) * 0.5;
    let vorticity = duy_dx - dux_dy;
    // Empirical normalization: lattice-unit vorticity scales with u0, not
    // with any fixed constant -- so a fraction of u0 keeps shed vortices
    // visibly saturated across the Re range without a per-case tuning knob.
    return vec4f(diverging(vorticity / (0.5 * u0)), 1.0);
  }
  if (view_mode == 2u) {
    let dev = rho_in[cell] - 1.0;
    // BGK density deviations are O(Ma^2); u0^2 gives a resolution-
    // independent normalization tied to the actual compressibility error.
    return vec4f(diverging(dev / (2.0 * u0 * u0)), 1.0);
  }
  // view_mode == 3: dye/smoke, monochrome over a dark background.
  let d = clamp(dye_in[cell], 0.0, 1.0);
  return vec4f(vec3f(0.02, 0.03, 0.05) + d * vec3f(0.85, 0.87, 0.9), 1.0);
}
