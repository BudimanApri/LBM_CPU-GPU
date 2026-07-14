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

// Pipeline-overridable workgroup dimensions. Phase 6 benchmarks all three
// candidates against completed GPU work and specializes these constants on
// the winning pipeline; no shader source rewriting is involved.
override WORKGROUP_X: u32 = 8u;
override WORKGROUP_Y: u32 = 8u;

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

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y)
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

  // -- BGK / Smagorinsky collision (factored equilibrium, gotcha #4) --
  // LES is local and allocation-free. The non-equilibrium stress tensor is
  // evaluated from the streamed populations, then the standard LBM
  // Smagorinsky relaxation raises tau only where unresolved shear demands
  // eddy viscosity. With LES disabled tau_eff is exactly params.tau, keeping
  // the Phase-2 CPU/GPU parity path bit-for-bit unchanged.
  let usq = u * u + v * v;
  var feq = array<f32, 9>();
  for (var i = 0; i < 9; i++) {
    let cu = f32(D2Q9_CX[i]) * u + f32(D2Q9_CY[i]) * v;
    feq[i] = D2Q9_W[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
  }

  var tau_eff = params.tau;
  if ((params.flags & FLAG_LES_ENABLED) != 0u) {
    var pi_xx = 0.0;
    var pi_yy = 0.0;
    var pi_xy = 0.0;
    for (var i = 0; i < 9; i++) {
      let cx = f32(D2Q9_CX[i]);
      let cy = f32(D2Q9_CY[i]);
      let neq = g[i] - feq[i];
      pi_xx += cx * cx * neq;
      pi_yy += cy * cy * neq;
      pi_xy += cx * cy * neq;
    }
    let pi_norm = sqrt(pi_xx * pi_xx + 2.0 * pi_xy * pi_xy + pi_yy * pi_yy);
    let cs2 = params.smagorinsky_cs * params.smagorinsky_cs;
    let tau_turb = 0.5 *
      (sqrt(params.tau * params.tau + 18.0 * cs2 * pi_norm / max(r, 1e-12)) - params.tau);
    tau_eff = clamp(params.tau + tau_turb, 0.51, 1.5);
  }

  let omega = 1.0 / tau_eff;
  // -- absorbing fringe for weakly-compressible pressure waves --
  // Free-slip walls and the simple zero-gradient outlet reflect acoustic
  // modes. A smooth, weak relaxation toward the undisturbed inlet
  // equilibrium absorbs those modes before they reach the boundary. The
  // outlet fringe occupies the final 16% of the tunnel; wall fringes occupy
  // the outer 8% and are disabled for periodic-y runs. Strengths default to
  // zero in parity/validation tests, so the original solver remains exactly
  // available. The interactive app uses conservative per-step strengths.
  let xf = (f32(x) + 0.5) / f32(nx);
  let yf = (f32(y) + 0.5) / f32(ny);
  let outlet_ramp = smoothstep(0.84, 1.0, xf);
  let wall_distance = min(yf, 1.0 - yf);
  var wall_ramp = 1.0 - smoothstep(0.0, 0.08, wall_distance);
  if ((params.flags & FLAG_PERIODIC_Y) != 0u) {
    wall_ramp = 0.0;
  }
  let sponge = clamp(
    params.sponge_outlet * outlet_ramp * outlet_ramp +
      params.sponge_wall * wall_ramp * wall_ramp,
    0.0,
    0.2,
  );
  let target_usq = params.inlet_u * params.inlet_u;
  for (var i = 0; i < 9; i++) {
    var post_collision = g[i] - omega * (g[i] - feq[i]);
    if (sponge > 0.0) {
      let target_cu = f32(D2Q9_CX[i]) * params.inlet_u;
      let target_eq = D2Q9_W[i] *
        (1.0 + 3.0 * target_cu + 4.5 * target_cu * target_cu - 1.5 * target_usq);
      post_collision = mix(post_collision, target_eq, sponge);
    }
    f_dst[i * n + cell] = post_collision;
  }
}
