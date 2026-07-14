// Momentum-exchange drag/lift via a two-stage parallel reduction.
//
// CLAUDE.md (Forces) + gotcha #6: WebGPU has no f32 atomics, and the spec
// prescribes the reduction itself, not merely "avoid atomics." So this is a
// genuine two-stage reduction -- workgroup-shared accumulation, in-workgroup
// tree reduce, one partial per workgroup, then a tiny second dispatch that
// sums the partials -- with no atomic shortcut anywhere.
//
// Stage 1 (forces_accumulate): one thread per lattice cell. For a fluid cell,
// every direction i whose neighbor (x + c_i) is a solid obstacle cell is a
// fluid->solid link; the momentum crossing it is CLAUDE.md's
//     dp = c_i * (f_i + f_ibar).
// Halfway bounce-back reflects the post-collision population f_i straight back
// into this same fluid node (that is exactly what lbm.wgsl's pull gather
// does), so the incoming f_ibar arriving over the link equals f_i, and the
// form reduces to the exact, robust
//     dp = 2 c_i f_i
// with f_i the post-collision population stored at the fluid cell (buffers A/B
// hold post-collision state). Reading f_ibar instead as the *opposite*
// post-collision population at the node is a different quantity that nearly
// cancels f_i and flips sign with resolution -- do not do that. Per-thread
// contributions sum in workgroup memory, tree-reduce, and thread 0 writes one
// vec2<f32> partial for the workgroup.
//
// Stage 2 (forces_reduce): a single workgroup strides over every partial,
// tree-reduces, and writes the total (Fx, Fy) to force_result[0].
//
// Sign: dp is the momentum the fluid delivers to the solid, so the total is
// the force ON the obstacle -- +x is drag for west-to-east inflow. The host
// forms C = 2 F / (rho0 U^2 Lref), where Lref is airfoil chord or frontal
// height for the other obstacle types. Normalization stays on the host.
//
// Domain-edge walls (free-slip / periodic / inlet / outflow) are NOT counted:
// only interior cells flagged solid in the mask are obstacles. A link whose
// neighbor falls outside the lattice is skipped.

// Params struct comes from params.wgsl; D2Q9 tables from the generated block.

const WG_X: u32 = 8u;
const WG_Y: u32 = 8u;
const WG_SIZE: u32 = 64u; // WG_X * WG_Y
const REDUCE_SIZE: u32 = 256u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> f_src: array<f32>;
@group(0) @binding(2) var<storage, read> solid_mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> partials: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> force_result: array<vec2<f32>>;

var<workgroup> tile: array<vec2<f32>, WG_SIZE>;
var<workgroup> rtile: array<vec2<f32>, REDUCE_SIZE>;

// Sum of c_i (f_i + f_ibar) over this fluid cell's fluid->solid links.
fn link_force(x: i32, y: i32) -> vec2<f32> {
  let nx = i32(params.nx);
  let ny = i32(params.ny);
  let n = nx * ny;
  let cell = y * nx + x;
  if (solid_mask[cell] != 0u) {
    return vec2<f32>(0.0, 0.0);
  }
  var acc = vec2<f32>(0.0, 0.0);
  // Direction 0 is the rest population and crosses no link.
  for (var i = 1; i < 9; i++) {
    let sx = x + D2Q9_CX[i];
    let sy = y + D2Q9_CY[i];
    if (sx < 0 || sx >= nx || sy < 0 || sy >= ny) {
      continue;
    }
    if (solid_mask[sy * nx + sx] == 0u) {
      continue;
    }
    let two_fi = 2.0 * f_src[i * n + cell];
    acc += vec2<f32>(f32(D2Q9_CX[i]) * two_fi, f32(D2Q9_CY[i]) * two_fi);
  }
  return acc;
}

@compute @workgroup_size(WG_X, WG_Y)
fn forces_accumulate(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let nx = i32(params.nx);
  let ny = i32(params.ny);
  let x = i32(gid.x);
  let y = i32(gid.y);
  var contrib = vec2<f32>(0.0, 0.0);
  if (x < nx && y < ny) {
    contrib = link_force(x, y);
  }
  tile[lid] = contrib;
  workgroupBarrier();

  // Tree reduction over the 64-thread workgroup (power of two).
  var stride = WG_SIZE / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (lid < stride) {
      tile[lid] += tile[lid + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid == 0u) {
    partials[wid.y * nwg.x + wid.x] = tile[0];
  }
}

@compute @workgroup_size(REDUCE_SIZE)
fn forces_reduce(@builtin(local_invocation_index) lid: u32) {
  let count = arrayLength(&partials);
  var acc = vec2<f32>(0.0, 0.0);
  var idx = lid;
  loop {
    if (idx >= count) {
      break;
    }
    acc += partials[idx];
    idx += REDUCE_SIZE;
  }
  rtile[lid] = acc;
  workgroupBarrier();

  var stride = REDUCE_SIZE / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (lid < stride) {
      rtile[lid] += rtile[lid + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid == 0u) {
    force_result[0] = rtile[0];
  }
}
