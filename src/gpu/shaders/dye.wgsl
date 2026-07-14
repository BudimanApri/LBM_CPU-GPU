// Semi-Lagrangian dye (smoke) advection: one compute pass, ping-pong dye
// buffers. Bilinear back-trace along the current velocity field, small
// uniform dissipation, and continuous emitter rakes just downstream of the
// inlet while dye is enabled (params.flags bit 1, FLAG_DYE_ENABLED --
// CLAUDE.md's single "dye emitters on/off" control). No extra distribution
// functions for the scalar -- this is cheaper and visually smoother.
// Params/bilinear_sample come from params.wgsl / common.wgsl, concatenated
// at module creation.

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> ux_in: array<f32>;
@group(0) @binding(2) var<storage, read> uy_in: array<f32>;
@group(0) @binding(3) var<storage, read> solid_mask: array<u32>;
@group(0) @binding(4) var<storage, read> dye_src: array<f32>;
@group(0) @binding(5) var<storage, read_write> dye_dst: array<f32>;

const DYE_DISSIPATION: f32 = 0.999;
const EMITTER_COLUMN: i32 = 1;
const EMITTER_BANDS: i32 = 5;

@compute @workgroup_size(8, 8)
fn dye_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nx = i32(params.nx);
  let ny = i32(params.ny);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= nx || y >= ny) {
    return;
  }
  let cell = y * nx + x;
  if (solid_mask[cell] != 0u) {
    dye_dst[cell] = 0.0;
    return;
  }

  let src_x = f32(x) - ux_in[cell];
  let src_y = f32(y) - uy_in[cell];
  var d = bilinear_sample(&dye_src, nx, ny, src_x, src_y) * DYE_DISSIPATION;

  if ((params.flags & FLAG_DYE_ENABLED) != 0u && x == EMITTER_COLUMN) {
    let period = max(1, ny / EMITTER_BANDS);
    let band_height = max(1, period / 2);
    if ((y % period) < band_height) {
      d = 1.0;
    }
  }
  dye_dst[cell] = clamp(d, 0.0, 1.0);
}
