// Shared simulation-params uniform struct, concatenated into every shader
// module that binds it (WGSL has no #include). Byte layout is mirrored by
// PARAMS_OFFSETS in src/gpu/buffers.ts and pinned by
// tests/params-layout.test.ts (gotcha #5) -- extend all three together.

struct Params {
  nx: u32,
  ny: u32,
  tau: f32,
  inlet_u: f32,
  flags: u32,      // bit 0: periodic top/bottom walls (else free-slip)
  step_index: u32, // hash salt for particle respawn
  substeps: u32,   // solver steps per rendered frame (K) -- the dye and
                   // particle passes run once per frame and scale their
                   // advection by K to stay synchronized with the flow
  smagorinsky_cs: f32,
}

const FLAG_PERIODIC_Y: u32 = 1u;
const FLAG_DYE_ENABLED: u32 = 2u;
const FLAG_LES_ENABLED: u32 = 4u;
const VIEW_MODE_SHIFT: u32 = 4u;
const VIEW_MODE_MASK: u32 = 3u;
