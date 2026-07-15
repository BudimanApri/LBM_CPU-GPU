# Validation data

Machine-readable summaries of the quantitative cases documented in the root
[`VALIDATION.md`](../VALIDATION.md). These files are intended to make solver-to-solver
comparisons easier and to distinguish recorded measurements from target values.

## Files

- [`cylinder-summary.csv`](cylinder-summary.csv): Phase 5 cylinder drag and shedding-frequency gates.
- [`naca0012-summary.csv`](naca0012-summary.csv): manually recorded, chord-normalized NACA 0012 drag results.

## Provenance

`measurement_type=automated_probe` means the value was obtained through the project's
browser validation probe. `measurement_type=manual_hud_mean` means it was transcribed from
the converged HUD/chart by the project author. Blank fields mean that the parameter was not
recorded at measurement time; consumers must not infer a value for them.

The current airfoil CSV contains summary values, not raw Cd/Cl time histories. A future raw
export should be added without replacing these records and should include at least lattice
step, Cd, Cl, requested/effective Reynolds number, tau, U, geometry, resolution, wall mode,
LES state, browser, GPU, and averaging-window metadata.

## Reproduction and comparison

Follow the airfoil procedure in the root [`README.md`](../README.md#airfoil-validation-method)
and the cylinder procedure in [`VALIDATION.md`](../VALIDATION.md#reproduction). Compare like
with like: characteristic length, force normalization, angle of attack, blockage, wall mode,
Reynolds number, and averaging convention must match.

The airfoil literature reference is:

> G. Di Ilio, D. Chiappini, S. Ubertini, G. Bella, and S. Succi, "Fluid flow
> around NACA 0012 airfoil at low-Reynolds numbers with hybrid lattice Boltzmann
> method," Computers & Fluids 166 (2018), 200-208.
> https://doi.org/10.1016/j.compfluid.2018.02.014
