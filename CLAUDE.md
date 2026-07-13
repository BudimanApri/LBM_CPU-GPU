# Real-Time LBM Wind Tunnel — WebGPU

## Project Overview

Build a browser-based, real-time 2D computational wind tunnel using the **Lattice Boltzmann Method (D2Q9, BGK collision)** running entirely on the GPU via **WebGPU compute shaders (WGSL)**. The user draws obstacles (cylinders, airfoils, walls) directly onto the domain and watches physically correct flow develop: boundary layers, separation, wake recirculation, and Kármán vortex streets — with live velocity/vorticity color mapping, dye (smoke) advection, particle tracers, and real-time drag/lift coefficient plots.

This is **not** a particle fake or a stable-fluids visual toy. The solver must be a correct LBM implementation that passes quantitative validation (Poiseuille profile, cylinder drag at Re=20, Strouhal number at Re=100) before any visual polish is added.

**Performance target:** ≥ 60 fps at a 1024×512 lattice with 2–4 solver substeps per rendered frame on a mid-range discrete GPU (report MLUPS — million lattice-site updates per second — in the HUD).

---

## Tech Stack & Constraints

- **Language:** TypeScript, `strict: true`, no `any`.
- **Build:** Vite. WGSL shaders live in `.wgsl` files imported as raw strings (`?raw`).
- **GPU:** WebGPU only. On unsupported browsers, show a clear fallback page explaining how to enable WebGPU (Chrome/Edge stable, Safari 18+, Firefox flag). Do **not** build a WebGL2 fallback unless explicitly requested later.
- **No frameworks** for the app shell (no React). Plain TS + DOM for controls. Keep the UI layer thin and isolated from the solver.
- **No CPU↔GPU readbacks inside the frame loop.** The only readback is the force accumulator, asynchronously, every N frames (see Forces).
- **Testing:** Vitest for the CPU reference solver and unit tests. The CPU solver is the ground truth the GPU kernels are verified against.

---

## Physics Specification

### D2Q9 lattice

Nine discrete velocities. Use this exact ordering everywhere (CPU reference, WGSL, tables):

| i | cᵢ (x, y) | weight wᵢ | opposite ī |
|---|-----------|-----------|------------|
| 0 | ( 0,  0)  | 4/9       | 0 |
| 1 | ( 1,  0)  | 1/9       | 3 |
| 2 | ( 0,  1)  | 1/9       | 4 |
| 3 | (−1,  0)  | 1/9       | 1 |
| 4 | ( 0, −1)  | 1/9       | 2 |
| 5 | ( 1,  1)  | 1/36      | 7 |
| 6 | (−1,  1)  | 1/36      | 8 |
| 7 | (−1, −1)  | 1/36      | 5 |
| 8 | ( 1, −1)  | 1/36      | 6 |

Lattice speed of sound: `cs² = 1/3`.

### Macroscopic moments

```
ρ  = Σᵢ fᵢ
ρu = Σᵢ fᵢ cᵢ
```

### BGK collision with equilibrium

```
fᵢ_eq = wᵢ ρ [ 1 + 3(cᵢ·u) + 4.5(cᵢ·u)² − 1.5(u·u) ]
fᵢ*   = fᵢ − (1/τ)(fᵢ − fᵢ_eq)
```

Kinematic viscosity in lattice units: `ν = cs²(τ − 0.5) = (τ − 0.5)/3`.

### Stability envelope (enforce in code, not just docs)

- Clamp `τ ∈ [0.51, 1.5]`. Never allow τ ≤ 0.5.
- Keep inlet velocity `U ≤ 0.1` lattice units (low-Mach limit; compressibility error ~ Ma²). Clamp the UI slider accordingly.
- Reynolds number is the user-facing control: `Re = U·D/ν` where D = characteristic obstacle size in lattice cells. When the user changes Re, hold U fixed and solve for τ. If the resulting τ < 0.51, either raise resolution or enable the Smagorinsky LES term (Phase 6) rather than letting τ collapse.
- NaN guard: a debug-mode compute pass that detects non-finite ρ and paints those cells magenta; on detection, pause and report. Never let NaNs silently propagate.

### Boundary conditions

- **Obstacles:** halfway bounce-back. Solid cells reflect incoming populations to their opposite direction. Implemented inside the fused kernel via the obstacle mask.
- **Inlet (west edge):** Zou–He velocity boundary with prescribed `(U, 0)`. Implement the standard D2Q9 west-wall equations for the unknown populations f₁, f₅, f₈.
- **Outlet (east edge):** zero-gradient outflow — copy the unknown populations from the neighboring interior column after streaming. Simple and stable at these Re; do not over-engineer with characteristic BCs.
- **Top/bottom edges:** free-slip (specular reflection) by default, with a settings toggle for periodic. Free-slip better approximates an open wind tunnel and avoids spurious wall boundary layers contaminating Cd.

### Passive dye / smoke

Advect a dye field for smoke visualization with a **semi-Lagrangian advection pass** using the LBM velocity field (bilinear back-trace, small uniform dissipation factor ~0.999). Do not add extra distribution functions for the scalar — semi-Lagrangian is cheaper and visually smoother. Dye is injected in user-configurable emitter lines/rakes near the inlet.

---

## GPU Architecture

### Memory layout

- **Distributions:** Structure-of-Arrays. One `array<f32>` storage buffer per direction is acceptable, but preferred: a single buffer of size `9 × Nx × Ny` indexed as `f[i * N + (y * Nx + x)]` — better for bind-group limits. **Two copies (A and B) for ping-pong.** All f32; do not attempt f16 storage.
- **Obstacle mask:** `array<u32>` storage buffer (0 = fluid, 1 = solid), written by the brush/preset pipeline.
- **Macroscopic fields:** one `rgba32float`-equivalent storage buffer or three buffers (ρ, uₓ, u_y), written by the solver each step, read by rendering and dye advection.
- **Uniforms:** a single params uniform buffer (Nx, Ny, τ, U_inlet, flags, frame index...). **Respect WGSL std140-like alignment: pad structs to 16-byte boundaries; verify layout with a unit test that mirrors the struct in TS.**

### Kernel design

- **One fused stream–collide kernel using the pull scheme:** each thread, for its cell, gathers post-collision populations from the 9 upstream neighbors of buffer A, handles bounce-back/edges during the gather, computes moments, applies BGK collision, and writes to buffer B. Then swap. One dispatch per LBM step — no separate streaming pass, no separate moments pass.
- Boundary handling (Zou–He inlet, outflow copy) either as branches inside the fused kernel on edge threads (fine at this scale) or a tiny second dispatch over edge columns — implement the branch version first, measure, only split if it costs > 5%.
- **Workgroup size:** start with 8×8; benchmark 16×16 and 16×8 in Phase 6 and keep the winner.
- **Substepping:** run K solver steps per animation frame (K adaptive to hold 60 fps, default 3). Higher K = faster apparent flow evolution.

### Auxiliary passes

- Dye advection: one compute pass, ping-pong dye textures.
- Particle tracers: ~20k particles in a storage buffer, advected with RK2 using bilinear-sampled velocity; respawn at inlet when they exit or enter solids. Rendered as instanced quads or point-list with velocity-based alpha.
- Force reduction: see Forces.

### Rendering

Fullscreen triangle + fragment shader reading the macroscopic buffers. View modes (dropdown):

1. **Velocity magnitude** — turbo or viridis colormap (implement colormaps as WGSL polynomial fits, not textures).
2. **Vorticity** (∂u_y/∂x − ∂uₓ/∂y via central differences in the fragment shader) — diverging blue↔white↔red, symmetric range. This is the money shot for vortex streets.
3. **Density/pressure** (ρ deviation from 1).
4. **Dye/smoke** — monochrome smoke composited over a dark background, optionally blended with vorticity.

Obstacles render as a crisp overlay (mask edge-detected, subtle outline). Particles composite on top. Canvas format: use `navigator.gpu.getPreferredCanvasFormat()`.

---

## Forces: Drag & Lift

Use the **momentum-exchange method** on boundary links: for every fluid→solid link crossed during streaming, accumulate `Δp = cᵢ (fᵢ + fī)` into per-workgroup partial sums (atomics on i32 fixed-point or a two-stage parallel reduction — implement the two-stage reduction; avoid f32 atomics since WebGPU lacks them).

- Reduce to total (Fx, Fy) on GPU; async `mapAsync` readback every 10 frames.
- Report `Cd = 2Fx / (ρ₀ U² D)` and `Cl = 2Fy / (ρ₀ U² D)` where D is the current obstacle's frontal height in cells (computed from the mask's bounding box column-height; recompute when the mask changes).
- Plot Cd and Cl time histories on a small 2D-canvas strip chart (last ~1000 samples). Oscillating Cl at Re≈100 is how the user *sees* vortex shedding quantitatively — the Cl oscillation frequency directly gives the Strouhal number; display `St = f·D/U` live.

---

## Interaction & UI

### Obstacle authoring

- **Brush:** paint/erase solid cells with adjustable radius (1–30 cells), drawn on pointer events, rasterized on GPU (small compute dispatch writing the mask). Support touch.
- **Presets** (button row, placed at ~25% chord of domain, vertically centered):
  - Circular cylinder (diameter slider),
  - **NACA 4-digit airfoil** — generate from the standard thickness + camber equations parameterized by the 4 digits (default 4412) with an **angle-of-attack slider (−15° to +15°)** that re-rasterizes the polygon into the mask,
  - Flat plate (normal and inclined),
  - Backward-facing step.
- **Clear obstacles** and **Reset flow** (reinitialize f to equilibrium at inlet conditions) are separate buttons.

### Controls panel

- Reynolds number (log slider, 10 – 10,000; values above ~2,000 auto-enable LES once Phase 6 lands),
- Inlet velocity U (0.02 – 0.1 lattice units),
- View mode, colormap, dye emitters on/off, particles on/off,
- Pause / single-step / resolution select (512×256, 1024×512, 2048×1024),
- HUD: fps, MLUPS, K substeps, Re, τ, Cd, Cl, St.

### Visual design direction

Dark, instrument-like "wind tunnel control room" aesthetic: near-black background so vorticity and smoke fields carry the color; a monospaced utility face for HUD numerics; controls in a slim collapsible side panel that never overlaps the domain. The simulation canvas **is** the hero — chrome stays quiet. No decorative gradients; the one signature element is the live Cd/Cl strip chart with the Strouhal readout, styled like tunnel instrumentation.

---

## File Structure

```
lbm-wind-tunnel/
├── CLAUDE.md
├── index.html
├── package.json / tsconfig.json / vite.config.ts
├── src/
│   ├── main.ts                 # bootstrap, frame loop, substep scheduler
│   ├── gpu/
│   │   ├── context.ts          # adapter/device/canvas setup, feature checks
│   │   ├── buffers.ts          # buffer allocation, ping-pong management
│   │   ├── pipelines.ts        # pipeline + bind group construction
│   │   └── shaders/
│   │       ├── lbm.wgsl        # fused pull-scheme stream–collide + BCs
│   │       ├── dye.wgsl        # semi-Lagrangian dye advection
│   │       ├── particles.wgsl  # tracer advection
│   │       ├── forces.wgsl     # momentum exchange + reduction
│   │       ├── brush.wgsl      # mask painting / preset rasterization
│   │       └── render.wgsl     # fullscreen visualization passes
│   ├── solver/
│   │   ├── cpu-lbm.ts          # reference CPU solver (ground truth)
│   │   ├── constants.ts        # D2Q9 tables — single source of truth
│   │   └── units.ts            # Re ↔ τ ↔ ν conversions, clamping
│   ├── geometry/
│   │   ├── naca.ts             # 4-digit airfoil generator + AoA transform
│   │   └── presets.ts          # cylinder, plate, step rasterizers
│   ├── ui/
│   │   ├── controls.ts         # panel, sliders, buttons
│   │   ├── hud.ts              # fps / MLUPS / coefficients
│   │   └── chart.ts            # Cd/Cl strip chart (2D canvas)
│   └── validation/
│       └── cases.ts            # scripted validation scenarios
└── tests/
    ├── cpu-lbm.test.ts         # Poiseuille, symmetry, conservation
    ├── units.test.ts           # Re/τ math, clamps
    ├── naca.test.ts            # airfoil geometry properties
    └── gpu-parity.test.ts      # GPU vs CPU field comparison (dev harness)
```

The D2Q9 tables in `constants.ts` are the **single source of truth**; generate the WGSL constant block from them at build time (tiny Vite plugin or codegen script) so CPU and GPU can never diverge.

---

## Implementation Phases

Work strictly in order. **Each phase has a gate; do not start the next phase until the gate passes.** Commit at every gate.

**Phase 0 — Scaffold.** Vite + TS + WebGPU context, fullscreen triangle rendering a test gradient, resize handling, fallback page.
*Gate:* gradient renders; `npm run build` and `npm test` are green.

**Phase 1 — CPU reference solver.** Full D2Q9 BGK on typed arrays (small grids, e.g., 128×64), with bounce-back, Zou–He inlet, outflow. Written for clarity, not speed.
*Gate (Vitest):* (a) mass conserved to 1e-10 in a periodic box; (b) Poiseuille channel (body-force or pressure-driven, no-slip walls) matches the analytical parabolic profile within 1% at centerline after convergence; (c) uniform flow past no obstacle stays uniform.

**Phase 2 — GPU kernel parity.** Port to the fused WGSL pull-scheme kernel with ping-pong.
*Gate:* dev harness runs CPU and GPU side by side for 500 steps on identical 128×64 initial conditions with a cylinder; max abs difference in ρ, uₓ, u_y < 1e-4 (f32 accumulation-order tolerance).

**Phase 3 — Boundaries + interaction.** Zou–He/outflow/free-slip on GPU, obstacle mask buffer, brush painting, presets, NACA generator, reset/clear.
*Gate:* draw a cylinder at Re=20 → steady symmetric twin recirculation vortices develop and remain stable indefinitely (no blow-up over 100k steps).

**Phase 4 — Visualization.** Colormaps, vorticity view, dye advection + emitters, particle tracers, obstacle overlay, HUD with fps/MLUPS.
*Gate:* at Re=100, cylinder case shows a clean Kármán street in the vorticity view; 60 fps at 1024×512 with K=2.

**Phase 5 — Forces & instrumentation.** Momentum-exchange, GPU reduction, async readback, Cd/Cl strip chart, Strouhal readout.
*Gate (quantitative validation):*
- Re = 20 cylinder: Cd converges to ≈ 2.0 ± 15% (blockage ratio ≤ 1/8, i.e., domain height ≥ 8D).
- Re = 100 cylinder: sustained periodic Cl oscillation, **St = 0.16 – 0.17**.
Log both in a `VALIDATION.md` with screenshots.

**Phase 6 — Performance & robustness.** Workgroup-size benchmark, buffer layout experiments, adaptive K, Smagorinsky LES (`τ_eff = τ + τ_turb` from the local stress magnitude — enables Re into the thousands without instability), 2048×1024 mode, NaN sentinel, mobile/touch pass.
*Gate:* ≥ 60 fps at 1024×512 K=3 on the dev machine; MLUPS reported; Re=5,000 airfoil at 10° AoA runs stably with LES on.

---

## Numerical Gotchas (read before writing kernels)

1. **Pull vs push:** pull (gather) avoids write conflicts entirely — each thread writes only its own cell in buffer B. Never mix schemes.
2. **Zou–He corners** (inlet ∩ top/bottom) are underdetermined; use the standard corner treatment (copy tangential unknowns from the adjacent inlet node, enforce ρ from the neighbor). Handle explicitly or corners will slowly leak mass.
3. **Bounce-back inside the pull gather:** when the upstream neighbor in direction ī is solid, take the cell's *own* post-collision fī from the previous step instead. Halfway bounce-back places the wall between lattice nodes — document this in the kernel comments because it affects where D is measured for Cd.
4. **f32 equilibrium:** compute `fᵢ_eq` with the factored form above; do not "optimize" algebraically in ways that lose the u·u term symmetry — this is the classic source of drift.
5. **WGSL uniform alignment:** vec2/vec3 padding rules will silently corrupt the params struct. Mirror the layout in a TS test that checks byte offsets.
6. **No f32 atomics in WebGPU:** force accumulation must use the two-stage workgroup reduction (workgroup shared memory → per-workgroup partials buffer → second tiny dispatch).
7. **Obstacle edits mid-run:** newly solidified cells contain stale fluid populations; on mask change, reinitialize a 1-cell fluid shell around edited regions to local equilibrium to prevent pressure-pulse artifacts.
8. **Don't recreate bind groups per frame.** Prebuild both ping-pong bind group variants and alternate.

---

## Commands

```bash
npm run dev        # Vite dev server
npm run build      # production build (type-checks first)
npm test           # Vitest — CPU solver + units + geometry
npm run lint       # eslint + prettier check
```

---

## Definition of Done

- All six phase gates pass, with Phase 1 tests and Phase 5 validation numbers recorded in `VALIDATION.md`.
- A first-time user can, within 10 seconds of loading: see flow already running past a default cylinder at Re=100 with a visible vortex street, draw with the brush, and switch to the airfoil preset and drag the AoA slider to watch Cl respond.
- No console errors or WebGPU validation warnings; graceful message on non-WebGPU browsers.
- README with a GIF, control reference, the physics summary (one paragraph + the Re/τ relation), and the validation table.
