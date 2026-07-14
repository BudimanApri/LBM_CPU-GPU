// Particle tracer advection (compute) and rendering (instanced quads,
// velocity-based alpha). WebGPU's point-list topology has no configurable
// point size, so quads are the only portable way to get visible dots.
// Params/bilinear_sample come from params.wgsl / common.wgsl, concatenated
// at module creation.

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> ux_in: array<f32>;
@group(0) @binding(2) var<storage, read> uy_in: array<f32>;
@group(0) @binding(3) var<storage, read> solid_mask: array<u32>;
// Two views of the same GPU buffer: WGSL forbids read_write storage in the
// vertex stage, so particles_step (compute) and particles_vs (vertex) bind
// it at different numbers with the access mode each stage is allowed.
@group(0) @binding(4) var<storage, read_write> particles: array<vec2f>;
@group(0) @binding(5) var<storage, read> particles_ro: array<vec2f>;

// Bowman/Jenkins-style integer hash (public domain, widely reproduced) --
// deterministic per (particle index, step), no shared RNG state needed.
fn hash_u32(x_in: u32) -> u32 {
  var x = x_in;
  x = (x ^ (x >> 16u)) * 0x7feb352du;
  x = (x ^ (x >> 15u)) * 0x846ca68bu;
  return x ^ (x >> 16u);
}

fn sample_velocity(pos: vec2f, nx: i32, ny: i32) -> vec2f {
  return vec2f(
    bilinear_sample(&ux_in, nx, ny, pos.x, pos.y),
    bilinear_sample(&uy_in, nx, ny, pos.x, pos.y),
  );
}

@compute @workgroup_size(64)
fn particles_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&particles)) {
    return;
  }
  let nx = i32(params.nx);
  let ny = i32(params.ny);
  let p = particles[i];

  // RK2 (midpoint) integration, one lattice time unit per step.
  let k1 = sample_velocity(p, nx, ny);
  let k2 = sample_velocity(p + 0.5 * k1, nx, ny);
  let next = p + k2;

  let out_of_bounds = next.x < 0.0 || next.x >= f32(nx) || next.y < 0.0 || next.y >= f32(ny);
  var hit_solid = false;
  if (!out_of_bounds) {
    let cx = i32(next.x);
    let cy = i32(next.y);
    hit_solid = solid_mask[cy * nx + cx] != 0u;
  }

  if (out_of_bounds || hit_solid) {
    // Respawn at the inlet with a hashed y so particles don't stack.
    let seed = hash_u32(i * 747796405u + params.step_index * 2891336453u + 12345u);
    let y = (f32(seed) / 4294967295.0) * f32(ny);
    particles[i] = vec2f(0.5, y);
  } else {
    particles[i] = next;
  }
}

struct ParticleVertexOutput {
  @builtin(position) position: vec4f,
  @location(0) alpha: f32,
};

const PARTICLE_HALF_SIZE: f32 = 0.0018;

@vertex
fn particles_vs(
  @builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32,
) -> ParticleVertexOutput {
  let nx = i32(params.nx);
  let ny = i32(params.ny);
  let p = particles_ro[instance_index];

  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, 1.0),
  );
  let corner = corners[vertex_index];

  // Same pixel-center convention as render.wgsl's uv, but lattice y and
  // NDC y both increase "up" -- unlike the fragment shader's framebuffer
  // lookup, no flip is needed here.
  let ndc_x = ((p.x + 0.5) / f32(nx)) * 2.0 - 1.0;
  let ndc_y = ((p.y + 0.5) / f32(ny)) * 2.0 - 1.0;

  var out: ParticleVertexOutput;
  out.position = vec4f(
    ndc_x + corner.x * PARTICLE_HALF_SIZE,
    ndc_y + corner.y * PARTICLE_HALF_SIZE,
    0.0,
    1.0,
  );
  let cx = clamp(i32(p.x), 0, nx - 1);
  let cy = clamp(i32(p.y), 0, ny - 1);
  let cell = cy * nx + cx;
  let speed = length(vec2f(ux_in[cell], uy_in[cell]));
  out.alpha = clamp(speed / max(params.inlet_u, 1e-6), 0.15, 1.0);
  return out;
}

@fragment
fn particles_fs(in: ParticleVertexOutput) -> @location(0) vec4f {
  return vec4f(0.9, 0.93, 0.97, in.alpha * 0.8);
}
