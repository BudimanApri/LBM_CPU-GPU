// Shared WGSL fragments concatenated into every module that needs them
// (WGSL has no #include). Colormaps as polynomial fits -- no textures --
// and a bilinear sampler over a storage-buffer scalar field, used by both
// dye.wgsl and particles.wgsl so the interpolation logic exists once.

// Turbo colormap polynomial approximation (Mikhailov, Google AI Blog 2019,
// "Turbo, An Improved Rainbow Colormap for Visualization"). t in [0, 1].
fn turbo(t_in: f32) -> vec3f {
  let x = clamp(t_in, 0.0, 1.0);
  let v4 = vec4f(1.0, x, x * x, x * x * x);
  let v2 = v4.zw * v4.z;
  let r =
    dot(v4, vec4f(0.13572138, 4.6153926, -42.66032258, 132.13108234)) +
    dot(v2, vec2f(-152.94239396, 59.28637943));
  let g =
    dot(v4, vec4f(0.09140261, 2.19418839, 4.84296658, -14.18503333)) +
    dot(v2, vec2f(4.27729857, 2.82956604));
  let b =
    dot(v4, vec4f(0.1066733, 12.64194608, -60.58204836, 110.36276771)) +
    dot(v2, vec2f(-89.90310912, 27.34824973));
  return clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0));
}

// Viridis colormap polynomial approximation (6th-order Horner form). t in [0, 1].
fn viridis(t_in: f32) -> vec3f {
  let t = clamp(t_in, 0.0, 1.0);
  let c0 = vec3f(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  let c1 = vec3f(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  let c2 = vec3f(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  let c3 = vec3f(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  let c4 = vec3f(6.228269936347081, 14.17993336680509, 56.69055260068105);
  let c5 = vec3f(4.776384997670288, -13.74514537774601, -65.35303263337234);
  let c6 = vec3f(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return clamp(
    c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))),
    vec3f(0.0),
    vec3f(1.0),
  );
}

// Diverging blue<->white<->red for symmetric-range fields (vorticity,
// density deviation). t in [-1, 1], 0 = white. Blue and red are the same
// deviation from white with R/B swapped, so both ends carry equal visual
// weight -- deliberately symmetric, not an arbitrary color pick.
fn diverging(t_in: f32) -> vec3f {
  let t = clamp(t_in, -1.0, 1.0);
  let white = vec3f(0.95, 0.95, 0.95);
  let blue = vec3f(0.1, 0.3, 0.9);
  let red = vec3f(0.9, 0.3, 0.1);
  return mix(white, select(red, blue, t < 0.0), abs(t));
}

// Bilinear sample of a scalar field stored as a flat nx*ny array, clamped
// to the domain edge (no wraparound -- callers already resolve which
// domain a coordinate belongs to before sampling).
fn bilinear_sample(buf: ptr<storage, array<f32>, read>, nx: i32, ny: i32, x: f32, y: f32) -> f32 {
  let cx = clamp(x, 0.0, f32(nx - 1));
  let cy = clamp(y, 0.0, f32(ny - 1));
  let x0 = i32(floor(cx));
  let y0 = i32(floor(cy));
  let x1 = min(x0 + 1, nx - 1);
  let y1 = min(y0 + 1, ny - 1);
  let fx = cx - f32(x0);
  let fy = cy - f32(y0);
  let v00 = buf[y0 * nx + x0];
  let v10 = buf[y0 * nx + x1];
  let v01 = buf[y1 * nx + x0];
  let v11 = buf[y1 * nx + x1];
  return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}
