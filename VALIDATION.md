# Validation — Phase 5 Forces & Instrumentation

Quantitative validation of the LBM solver against the Phase 5 gate:

- **Re = 20 cylinder:** Cd converges to ≈ 2.0 ± 15% (i.e. 1.70 – 2.30), at blockage ≤ 1/8.
- **Re = 100 cylinder:** sustained periodic vortex shedding with **St = 0.16 – 0.17**.

Setups are defined in [`src/validation/cases.ts`](src/validation/cases.ts) and their
constraints (blockage ratio ≤ 1/8, τ in the stability envelope) are asserted in
[`tests/cases.test.ts`](tests/cases.test.ts). Both cases run on the app's **1024 × 512**
lattice at inlet velocity **U = 0.05** with a **D = 24** cylinder.

Why D = 24 and not the app's default D = 48: the flow is confined between free-slip walls,
and confinement raises both Cd and St by roughly `1/(1 − D/H)²`. At the default blockage
48/512 = 0.094 that correction is ~+22 %, more than the gate's ±15 % window on its own — so
the default cylinder reads Cd ≈ 2.40 and St ≈ 0.175 (correct physics, but out of band). At
D = 24 the blockage is 24/512 = 1/21 (domain height 21 D), the correction drops to ~+10 %,
and the raw numbers land inside the gate. D = 24 is still well enough resolved for a < 5 %
discretization error.

Coefficients are measured live from the running app via the dev probe
`window.__lbm.coefficients()` (instantaneous force + time-mean) and, for the shedding
frequency, `window.__lbm.readMoments()` (wake velocity).

## How the force is measured

- **Momentum exchange**, two-stage GPU reduction ([`forces.wgsl`](src/gpu/shaders/forces.wgsl)):
  per fluid→solid link, `Δp = cᵢ (fᵢ + f_ī)`. With halfway bounce-back the reflected
  population arriving back at the fluid node equals the post-collision `fᵢ`, so the sum is
  the exact, resolution-robust `2 cᵢ fᵢ`. The reduction is verified against the CPU
  reference `momentumExchangeForce()` to < 1e-4 in
  [`tests/gpu-forces.test.ts`](tests/gpu-forces.test.ts).
- **Coefficients:** `Cd = 2·Fx / (ρ₀ U² D)`, `Cl = 2·Fy / (ρ₀ U² D)`, ρ₀ = 1.
- Cd is reported as a **time-mean** over the sample buffer — that is the gate's target, and
  averaging cancels the acoustic ripple described below.

## Results

| Case                     | Metric             | Expected               | Measured                             | Pass |
| ------------------------ | ------------------ | ---------------------- | ------------------------------------ | ---- |
| Re = 20 cylinder (D=24)  | mean Cd            | 2.0 ± 15 % (1.70–2.30) | **2.18** (Cl ≈ 0, symmetric, steady) | ✅   |
| Re = 100 cylinder (D=24) | St (wake velocity) | 0.16 – 0.17            | **0.167** (period 2873 steps)        | ✅   |

Reference points at the app default **D = 48** (blockage 0.094): mean Cd ≈ **2.40**, wake
St ≈ **0.175** — both correct once the ~+22 % confinement correction is removed
(2.40·(1−0.094)² ≈ 1.97; St trends down to 0.167 as blockage falls), but out of the raw gate
band, which is why the validation cases use D = 24.

The Re = 20 wake is a steady, symmetric twin-vortex recirculation (Cl = 0.000). The Re = 100
wake velocity oscillates cleanly and periodically (period 2873 ± 10 steps across nine cycles).

## Finding: the Cl signal is acoustically contaminated

The solver's shedding physics is correct — the **wake transverse velocity** downstream of the
cylinder gives St in-band. However, the **lift signal Cl** oscillates ~3× faster (period
≈ 1774 steps ≈ L/c_s at D = 48, the domain's longitudinal acoustic transit time, c_s = 1/√3)
and at several times the amplitude of the physical shedding lift. The momentum-exchange force
integrates the surface **pressure**, which is dominated by weakly-damped **domain acoustic
modes** (a ~0.25 % density fluctuation produces Cl ≈ ±2). These modes are sustained because
the inlet (Zou–He) and outlet (zero-gradient) boundaries are **reflective** — a deliberate
spec choice (CLAUDE.md: "do not over-engineer with characteristic BCs").

Consequences:

- **Mean Cd is reliable** — the acoustic ripple is zero-mean and averages out, so the Cd gate
  is evaluated on the time-mean.
- **St is validated from the wake velocity**, which is acoustically clean. St read directly
  from Cl locks onto the acoustic period and is _not_ reliable at this domain size; the
  Cl-based readout remains in the HUD/chart for qualitative shedding visibility only.

**Recommended follow-up (Phase 6):** a convective / non-reflective outflow BC, or a sponge
(absorbing) layer near the outlet, would drain the trapped acoustic energy and make Cl
directly usable for St. This is a boundary-condition change beyond Phase 5's instrumentation
scope.

## Reproduction

1. `npm run dev`, open the app (Chrome/Edge).
2. **Re = 20 Cd:** set the cylinder-diameter slider to 24 and the Re slider to 20, click
   **Reset flow**, and let it converge (D²/ν ≈ 4 800 steps per diffusion time; steady within
   ~30 k steps). Read `mean.cd` from `window.__lbm.coefficients()` once Cl ≈ 0 and the value
   is steady → ≈ 2.18.
3. **Re = 100 St:** with D = 24, set Re = 100, **Reset flow**, let the Kármán street establish,
   then probe `uy` a little off-centreline downstream of the cylinder and measure its
   oscillation period in solver steps → St = D / (period · U) ≈ 0.167.
