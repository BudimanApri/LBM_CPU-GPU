// Fused pull-scheme stream-collide kernel: one dispatch per LBM step.
// The generated D2Q9 constant block (d2q9-constants.wgsl) is prepended at
// shader-module creation, providing D2Q9_CX/CY/W/OPP/SPEC.
//
// Buffers A/B hold POST-COLLISION populations. Each thread, for its own
// cell: gathers the 9 streamed-in populations from f_src (resolving walls,
// obstacles, and the west/east edge conditions during the gather), computes
// moments, writes rho/ux/uy, applies BGK collision, and writes the
// post-collision populations to f_dst. Pure pull -- each thread writes only
// its own cell in f_dst, so there are no write conflicts (gotcha #1).
//
// Halfway bounce-back (gotcha #3): when the upstream neighbor in direction i
// is solid, the streamed-in value is this cell's OWN post-collision
// population in the opposite direction, f_src[opp(i)] at (x, y). The wall
// sits halfway between the fluid and solid lattice nodes -- obstacle
// diameters D for Cd are measured between those halfway planes.
//
// Moments are computed PRE-collision, i.e. from the post-stream state.
// Because BGK collision conserves rho and u, these are exactly the moments
// of the CPU reference's post-step state -- what the parity gate compares.

// The Params struct comes from params.wgsl, concatenated at module creation.

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> f_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> f_dst: array<f32>;
@group(0) @binding(3) var<storage, read> solid_mask: array<u32>;
@group(0) @binding(4) var<storage, read_write> rho_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> ux_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> uy_out: array<f32>;

// Streamed-in population for direction i at fluid cell (x, y): read the
// post-collision value from the upstream neighbor, resolving top/bottom
// walls and solid cells.
//
// Free-slip in pull form: when the upstream row is outside the domain, the
// arriving population is the specular partner (cx kept, cy flipped) from the
// same row -- f_i(x, y) <- f_src[spec(i)] at (x - cx_i, y).
//
// Blocked paths (direct upstream solid, or the free-slip source solid)
// fully reverse the ORIGINAL direction: return f_src[opp(i)] at (x, y).
// This mirrors the CPU reference exactly.
//
// The x coordinate of the source never leaves [0, nx-1] for the directions
// this is called with: west-edge unknowns (1, 5, 8) are reconstructed by
// Zou-He instead of pulled, and east-edge unknowns (3, 6, 7) are pulled at
// column nx-2 by the outflow rule below.
// Factored equilibrium at the prescribed inlet state (u = (u0, 0)).
fn inlet_equilibrium(i: i32, rho: f32, u0: f32) -> f32 {
  let cu = f32(D2Q9_CX[i]) * u0;
  return D2Q9_W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u0 * u0);
}

fn pull_population(i: i32, x: i32, y: i32) -> f32 {
  let nx = i32(params.nx);
  let ny = i32(params.ny);
  let n = nx * ny;
  var si = i;
  var sy = y - D2Q9_CY[i];
  let sx = x - D2Q9_CX[i];
  if (sy < 0 || sy >= ny) {
    if ((params.flags & FLAG_PERIODIC_Y) != 0u) {
      sy = (sy + ny) % ny;
    } else {
      si = D2Q9_SPEC[i];
      sy = y;
    }
  }
  let source = sy * nx + sx;
  if (solid_mask[source] != 0u) {
    return f_src[D2Q9_OPP[i] * n + y * nx + x];
  }
  return f_src[si * n + source];
}

@compute @workgroup_size(8, 8)
fn lbm_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nx = i32(params.nx);
  let ny = i32(params.ny);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= nx || y >= ny) {
    return;
  }
  let n = nx * ny;
  let cell = y * nx + x;

  if (solid_mask[cell] != 0u) {
    // Solid cells carry no fluid state; their f_dst slots are never read
    // (every pull checks the mask first). Neutral moments for rendering.
    rho_out[cell] = 1.0;
    ux_out[cell] = 0.0;
    uy_out[cell] = 0.0;
    return;
  }

  // -- streaming (pull gather) --
  // At the west/east edges this also pulls the unknown directions with
  // out-of-domain upstreams; WebGPU's robust buffer access makes those
  // reads safe, and the values are overwritten by the edge rules below.
  var g = array<f32, 9>();
  for (var i = 0; i < 9; i++) {
    g[i] = pull_population(i, x, y);
  }

  // -- west edge (x = 0): Zou-He velocity inlet, prescribed (U, 0), in the
  // REGULARIZED form (Latt & Chopard; mirrors the CPU reference exactly).
  // Zou-He density + bounce-back of non-equilibrium for the unknowns, then
  // all nine populations are rebuilt as equilibrium plus the second-order
  // projected non-equilibrium stress. Raw Zou-He is unstable below
  // tau ~ 0.55 with an unsteady wake (growing standing wave on the inlet
  // column); the projection filters those ghost modes while preserving the
  // prescribed density and momentum exactly.
  if (x == 0) {
    let u0 = params.inlet_u;
    let r = (g[0] + g[2] + g[4] + 2.0 * (g[3] + g[6] + g[7])) / (1.0 - u0);
    var neq = array<f32, 9>();
    for (var i = 0; i < 9; i++) {
      neq[i] = g[i] - inlet_equilibrium(i, r, u0);
    }
    neq[1] = neq[3];
    neq[5] = neq[7];
    neq[8] = neq[6];
    let pxx = neq[1] + neq[3] + neq[5] + neq[6] + neq[7] + neq[8];
    let pyy = neq[2] + neq[4] + neq[5] + neq[6] + neq[7] + neq[8];
    let pxy = neq[5] - neq[6] + neq[7] - neq[8];
    for (var i = 0; i < 9; i++) {
      let cx = f32(D2Q9_CX[i]);
      let cy = f32(D2Q9_CY[i]);
      let q = (cx * cx - 1.0 / 3.0) * pxx + 2.0 * cx * cy * pxy + (cy * cy - 1.0 / 3.0) * pyy;
      g[i] = inlet_equilibrium(i, r, u0) + D2Q9_W[i] * 4.5 * q;
    }
  }

  // -- east edge (x = nx-1): zero-gradient outflow --
  // The unknowns equal the post-stream values of the same directions at the
  // interior neighbor column, which is exactly a pull performed for (x-1, y).
  if (x == nx - 1) {
    g[3] = pull_population(3, x - 1, y);
    g[6] = pull_population(6, x - 1, y);
    g[7] = pull_population(7, x - 1, y);
  }

  // -- moments (pre-collision == CPU post-step moments) --
  var r = 0.0;
  var mx = 0.0;
  var my = 0.0;
  for (var i = 0; i < 9; i++) {
    r += g[i];
    mx += g[i] * f32(D2Q9_CX[i]);
    my += g[i] * f32(D2Q9_CY[i]);
  }
  let u = mx / r;
  let v = my / r;
  rho_out[cell] = r;
  ux_out[cell] = u;
  uy_out[cell] = v;

  // -- BGK collision (factored equilibrium, gotcha #4) --
  let omega = 1.0 / params.tau;
  let usq = u * u + v * v;
  for (var i = 0; i < 9; i++) {
    let cu = f32(D2Q9_CX[i]) * u + f32(D2Q9_CY[i]) * v;
    let feq = D2Q9_W[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
    f_dst[i * n + cell] = g[i] - omega * (g[i] - feq);
  }
}
