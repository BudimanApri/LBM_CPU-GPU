# WebGPU LBM Wind Tunnel

A real-time 2D computational wind tunnel running a D2Q9 lattice Boltzmann solver entirely on WebGPU. Draw obstacles or select a cylinder, NACA 4-digit airfoil, plate, or backward-facing step and inspect velocity, vorticity, pressure, smoke, particle paths, and momentum-exchange forces live.

![The Re=100 cylinder developing a Karman vortex street](docs/wind-tunnel.gif)

## Quick start

Requirements: Node.js 24+ and a browser with WebGPU enabled.

```bash
npm install
npm run dev
```

Chrome/Edge stable and Safari 18+ expose WebGPU by default on supported hardware. Firefox currently requires `dom.webgpu.enabled`. The app presents an explanatory fallback when WebGPU is unavailable.

## Controls

- **Re / Inlet U:** Reynolds number is the primary flow control; inlet speed remains in the low-Mach interval 0.02–0.1.
- **Walls:** free-slip open-tunnel walls or periodic top/bottom boundaries.
- **Obstacles:** cylinder, NACA 4-digit airfoil with angle of attack, normal/inclined plates, and backward-facing step.
- **Brush:** left/touch drag paints solid cells; right drag or erase mode removes them.
- **Views:** velocity magnitude, vorticity, density/pressure, and dye/smoke. About 20,000 optional particles use RK2 advection.
- **Resolution:** 512×256, 1024×512, or 2048×1024. Unsupported modes are disabled after checking the device's storage-buffer limits. A change reloads the simulation and reallocates all GPU resources once.
- **Run:** pause, single-step, reset flow, or clear obstacles.

The HUD reports FPS, MLUPS, adaptive solver substeps K, lattice/workgroup size, LES status, Re, relaxation time, frontal height, Cd, Cl, St, and total steps.

## Physics and numerics

The solver uses the D2Q9 BGK lattice Boltzmann method with a fused pull-scheme stream/collide kernel. Halfway bounce-back handles obstacles, a regularized Zou–He condition prescribes the west inlet, zero-gradient populations leave at the east outlet, and top/bottom walls default to free-slip.

The user-facing Reynolds number controls viscosity through

```text
nu  = (tau - 0.5) / 3
Re  = U D / nu
tau = 0.5 + 3 U D / Re
```

`tau` is always clamped to [0.51, 1.5]. Above Re≈2000, the app automatically enables a local Smagorinsky LES model (`Cs=0.1`):

```text
Pi_ab    = sum_i c_i,a c_i,b (f_i - f_i_eq)
tau_turb = 0.5 [sqrt(tau^2 + 18 Cs^2 |Pi| / rho) - tau]
tau_eff  = clamp(tau + tau_turb, 0.51, 1.5)
```

No extra LES buffers or neighbor reads are required. A low-frequency GPU sentinel detects non-finite density, pauses the simulation, reports the failure, and paints unstable cells magenta.

## GPU architecture

- Two f32 distribution buffers, each `9 × Nx × Ny`, ping-ponged without per-frame bind-group creation.
- One 2D compute dispatch per LBM substep; workgroup dimensions are WGSL pipeline overrides.
- Startup benchmark compares 8×8, 16×16, and 16×8 using `queue.onSubmittedWorkDone()` so timing includes completed GPU work rather than command-encoding time.
- Adaptive K uses both rAF frame time and asynchronously measured queue-drain latency, with hysteresis over 45-frame windows.
- Reset and inlet-speed changes use a 1200-lattice-step smooth ramp instead of launching an impulsive pressure pulse.
- Smooth absorbing fringes cover the final 16% of the outlet and outer 8% of free-slip walls. They relax weakly toward undisturbed equilibrium so pressure waves leave without damping the obstacle/wake core; periodic-Y runs disable wall damping.
- Forces use exact two-stage reduction: workgroup shared sums → per-workgroup partials → one final reduction. Force readback is asynchronous every ten frames.
- Dye, particles, force reduction, rendering, and stability checks remain GPU-side; no steady-state frame blocks on a readback.

## Validation

The quantitative procedure and caveats are recorded in [VALIDATION.md](VALIDATION.md).

| Gate                            |                  Result | Status |
| ------------------------------- | ----------------------: | :----: |
| Periodic-box mass conservation  |           error < 1e-10 |  Pass  |
| Poiseuille centerline profile   |              < 1% error |  Pass  |
| CPU/GPU fields after 500 steps  |        max error < 1e-4 |  Pass  |
| Cylinder Re=20, blockage 1/21   |               Cd = 2.18 |  Pass  |
| Cylinder Re=100, blockage 1/21  |              St = 0.167 |  Pass  |
| 1024×512, fixed K=3             |        60 fps, 94 MLUPS |  Pass  |
| Re≈5000 NACA 4412, AoA 10°, LES | finite; rho 0.990–1.005 |  Pass  |
| 2048×1024 allocation/run        |  60 fps at adaptive K=8 |  Pass  |

The force coefficients follow the project specification and normalize by **frontal height D** for every obstacle. Conventional airfoil polars instead normalize by chord and generally operate at Re≥50,000. Therefore low-Re airfoil Cd/Cl shown here should not be compared directly with AirfoilTools values without converting the reference length; Phase 6 extends stable runs into the thousands, not the experimental polar regime.

The underlying simple outlet/free-slip conditions reflect weak acoustic modes, so the production app adds absorbing fringes. A direct pressure-pulse A/B test reduced reflected core-density RMS by about 97% (`2.94e-4` → `8.32e-6`). Time-mean Cd remains the primary force metric; the Phase 5 Strouhal gate was independently cross-checked against wake velocity zero crossings.

## Development commands

```bash
npm test                 # CPU + browser/WebGPU tests
npm run typecheck
npm run lint
npm run build
npm run validate:phase6  # live FPS, LES, and 2048x1024 browser gate
npm run validate:acoustics # Re≈7000/U=0.09/AoA=8 boundary-reflection gate
```

WGSL constants are generated from `src/solver/constants.ts`, the single source of truth for D2Q9 ordering, weights, opposites, and specular partners.
