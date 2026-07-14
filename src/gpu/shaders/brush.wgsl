// Obstacle-mask editing: brush stamping plus the mask-change reconcile pass
// (CLAUDE.md gotcha #7). The generated D2Q9 constant block is prepended at
// shader-module creation.
//
// Flow per edit (rAF-coalesced to at most one chain per frame):
//   1. brush_apply   -- paint/erase a disc into `mask` (one dispatch per
//                       stamp; presets/clears upload the mask directly and
//                       skip this kernel)
//   2. mask_diff     -- changed = (mask != mask_prev)
//   3. mask_reconcile - every cell within one lattice link of a change gets
//                       its populations reset to the local equilibrium of
//                       its last-known moments, and mask_prev is re-synced.
//
// Why the reconcile works for every transition: the solver writes neutral
// moments (rho=1, u=0) for solid cells, so cells that were solid and became
// fluid re-enter quiescent; cells that stayed fluid next to an edit keep
// their velocity but drop their (now inconsistent) non-equilibrium part;
// newly solid cells get harmless inert values that no pull ever reads.

struct BrushParams {
  center: vec2f,   // stamp center, lattice coordinates
  radius: f32,
  mode: u32,       // 1 = paint solid, 0 = erase
  nx: u32,
  ny: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> brush: BrushParams;
@group(0) @binding(1) var<storage, read_write> mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> mask_prev: array<u32>;
@group(0) @binding(3) var<storage, read_write> changed: array<u32>;
@group(0) @binding(4) var<storage, read_write> f_cur: array<f32>;
@group(0) @binding(5) var<storage, read> rho_in: array<f32>;
@group(0) @binding(6) var<storage, read> ux_in: array<f32>;
@group(0) @binding(7) var<storage, read> uy_in: array<f32>;

fn equilibrium(i: i32, rho: f32, u: f32, v: f32) -> f32 {
  let cu = f32(D2Q9_CX[i]) * u + f32(D2Q9_CY[i]) * v;
  return D2Q9_W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (u * u + v * v));
}

@compute @workgroup_size(8, 8)
fn brush_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nx = i32(brush.nx);
  let ny = i32(brush.ny);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= nx || y >= ny) {
    return;
  }
  // Keep the inlet column and the two outflow-copy columns paint-free.
  if (x < 1 || x >= nx - 2) {
    return;
  }
  let dx = f32(x) - brush.center.x;
  let dy = f32(y) - brush.center.y;
  if (dx * dx + dy * dy <= brush.radius * brush.radius) {
    mask[y * nx + x] = brush.mode;
  }
}

@compute @workgroup_size(8, 8)
fn mask_diff(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nx = i32(brush.nx);
  let ny = i32(brush.ny);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= nx || y >= ny) {
    return;
  }
  let cell = y * nx + x;
  changed[cell] = u32(mask[cell] != mask_prev[cell]);
}

@compute @workgroup_size(8, 8)
fn mask_reconcile(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nx = i32(brush.nx);
  let ny = i32(brush.ny);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= nx || y >= ny) {
    return;
  }
  let cell = y * nx + x;
  var near = false;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let sx = clamp(x + dx, 0, nx - 1);
      let sy = clamp(y + dy, 0, ny - 1);
      if (changed[sy * nx + sx] != 0u) {
        near = true;
      }
    }
  }
  if (near) {
    let n = nx * ny;
    let r = rho_in[cell];
    let u = ux_in[cell];
    let v = uy_in[cell];
    for (var i = 0; i < 9; i++) {
      f_cur[i * n + cell] = equilibrium(i, r, u, v);
    }
  }
  if (changed[cell] != 0u) {
    mask_prev[cell] = mask[cell];
  }
}
