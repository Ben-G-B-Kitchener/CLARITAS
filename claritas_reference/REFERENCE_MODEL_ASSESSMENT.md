# Reference-model assessment and recommendation

## Validation status

The CPU transport passes all 10 automated tests and all 19 recorded validation
metrics. The tested properties are:

- zero-concentration ballistic transport;
- Beer–Lambert pure absorption;
- isotropic-scattering transverse symmetry;
- Mie CDF sampling error (L1, RMS, and maximum);
- low-optical-depth ballistic probability;
- exactly linear `mu_s`, `mu_a`, and `mu_t` concentration scaling;
- concentration-invariant phase functions;
- known detector-position assignments;
- exit/absorption/truncation probability closure;
- extinction-event categorical sampling;
- unit-vector preservation and isotropic direction mean;
- bitwise-equivalent selected outputs under a fixed seed.

The machine-readable evidence is in `validation_outputs/validation_results.csv`.

## Benchmark findings

The primary-only model is not a fit to all measured curves:

- Loess 0.5 g/L shape RMSE: 0.084251
- Loess 4.0 g/L shape RMSE: 0.038303
- Kaolin 0.5 g/L shape RMSE: 0.073578
- Kaolin 4.0 g/L shape RMSE: 0.087151

This is useful rather than surprising. It establishes what the declared primary
PSD, spherical particles, assumed complex index, spherical vessel, and ideal
annular detectors predict without aggregate or detector corrections.

The assumed common `k=0.001` produces absorbed fractions from 0.298 to 0.865.
Those are model predictions, not validated material measurements. They reinforce
the need for absolute transmission data and material-specific optical constants
before absorption is calibrated.

The low-concentration Loess discrepancy is concentrated in the forward region
(RMSE 0.142406), while its rear-region RMSE is only 0.026603. At high Loess
concentration the rear region becomes the larger error (0.057827). For Kaolin,
rear-region errors dominate both concentrations. This pattern is consistent
with missing aggregate, non-spherical-particle, and/or instrument geometry
physics; it does not justify inserting an angular correction.

## Can this replace CLARITAS_76 as the development foundation?

**Yes as the main scientific development foundation; no as a completed
predictive sediment model.**

It is sufficiently validated to become the reference branch because its
probabilities, geometry, phase sampling, reproducibility, and accounting are
small enough to test independently. New physics can be accepted or rejected
against explicit gates.

It is not yet sufficiently constrained for material conclusions because:

1. PSD weight semantics need instrument metadata.
2. Loess and Kaolin complex indices need independent wavelength-specific data.
3. The actual vessel is not yet represented; the reference uses a sphere.
4. Detector aperture, orientation, walls, Fresnel optics, and responsivity are
   not measured/implemented.
5. Absolute source-to-detector calibration is unavailable.
6. Flocs and primary non-sphericity are intentionally absent.
7. Monte Carlo benchmark uncertainty should be quantified with replicated seeds.

## Next decision

Adopt this package as the controlled CPU reference. Do not tune it to the four
normalised curves. Next obtain or encode the real vessel/detector geometry and
independently constrained optical inputs, then repeat the benchmark.

Only after those stages should the project approve one floc approach from
`FLOC_MODEL_DESIGN_REVIEW.md`. GPU work is justified for replicate and
uncertainty studies—the 100,000-ray four-case run took about 159 seconds, mostly
in Loess phase-table construction—but the GPU kernel must reproduce the CPU
validation distributions before becoming authoritative.

