# CLARITAS 77 changelog

CLARITAS 77 is an in-place production evolution of the CUDA/CuPy CLARITAS
code. It was copied from CLARITAS 76 and then modified; it is not a CPU
replacement or a parallel research model. `CLARITAS_76.py` remains unchanged
for comparison.

## Production transport

- Replaced planar signed-angle scattering with three-component direction
  vectors, uniform scattering azimuth, a stable local orthonormal basis, and
  direction renormalization after every turn.
- Changed the ideal reference transport domain from a 2-D circle to a 3-D
  sphere. The HDF5 output labels this geometry assumption explicitly.
- Changed angular probability construction from raw `I(theta)` to the physical
  axisymmetric solid-angle density `I(theta) sin(theta)`.
- Replaced pointwise phase-CDF accumulation with trapezoidal integration and
  fixed zero/one endpoints.
- Corrected the phase grid to the physical closed polar domain `0..180`
  degrees (inclusive), added endpoint validation, and versioned the cache so
  legacy tables containing the unused `180..181` degree tail cannot reload.
- Added linearly interpolated inverse-CDF phase sampling in CUDA.
- Replaced heatmap-controlled boundary crossing with an analytic sphere
  intersection. Heatmap samples are now derived from an immutable segment
  origin and cannot alter the transport endpoint or path length.
- Zero extinction now produces ballistic boundary exit rather than absorption.
- The maximum-event limit now produces a distinct `TRUNCATED` outcome. A ray at
  the cap may still exit if its next free path reaches the boundary before
  another event.
- Changed the albedo decision to exact `scatter iff U < albedo` semantics.
- Clamped primary Mie scattering cross sections unconditionally into
  `[0, sigma_t]` and derives nonnegative absorption by difference.
- Removed the unused/invalid PSD CDF and unused kernel/wrapper arguments.
- Removed unused reflection-probability/size-threshold constants and obsolete
  roughness aliases; no boundary reflection model is active.
- Removed the unmodelled reversal of outward-pointing source rays. Such rays
  are now classified as missed sample launches.
- Retained the existing beta/Gaussian production source under the explicit
  `production_beta` label and added a centred/collimated source mode matching
  the CPU reference. The screen's `--reference-parity` flag selects that source
  and disables flocs for controlled primary-only GPU validation.

## Probability validation and reproducibility

- Removed the empirical `ANGULAR_CDF_POWER`; no replacement tuning parameter
  was added.
- Invalid phase intensities, theta grids, extinction CDFs, phase CDFs,
  albedos, optical depths, diameters, and negative/nonfinite `mu_t` now fail
  before CUDA launch.
- Added the valid zero-extinction placeholder event CDF used by the CPU
  reference. It is not sampled because event selection is bypassed at
  `mu_t == 0`.
- Added `SIMULATION_SEED` and a seeded NumPy source generator.
- Device streams are keyed by global ray index, so resetting a CUDA chunk no
  longer repeats chunk-local ray streams.
- Global ray IDs are mapped bijectively onto nonzero xorshift32 seeds modulo
  `2^32-1`; runs exceeding the finite unique-initial-state limit fail before launch
  instead of silently wrapping or substituting a duplicate zero state. Float
  uniforms are constrained to an open interval.
- Bulk and internal exponential free paths use a double-precision open-interval
  uniform before the sampled distance is stored in the production float state.
- Replaced the unsafe `for`-range OOM resize with a gap-safe
  `while start < N` loop. Allocation or launch OOM halves the chunk and retries
  the same global ray index; the write cursor advances only after success.
- Added named minimum/maximum chunk controls and a 128-byte-per-ray planning
  allowance.
- Expanded HDF5 provenance with master/derived seeds, RNG seed mapping and
  unique-initial-state limit, RNG/source/geometry/detector labels,
  material, concentration, wavelength, PSD mode, floc switches, phase measure
  and cache hash, radius, event cap, terminal-state codes, chunk retry
  metadata, and SHA-256 hashes of the exact generated Python case and embedded
  CUDA source.

## Per-ray bookkeeping and detector accounting

- Added `exit_z`, `exit_vx`, `exit_vy`, and `exit_vz`.
- Path length is now written for escaped, absorbed, and truncated histories.
- Added explicit `absorbed`, `truncated`, and `terminal_state` arrays.
- Added `floc_extinction_count`, recorded before the albedo test, while
  retaining `floc_event_count` as successful floc outer scatters.
- Added z displacement for the existing finite-floc domain bookkeeping.
- Replaced signed planar detector angles with spherical polar boundary angle
  measured from the forward `+y` axis.
- Assigns every escaped ray to at most one nearest accepted detector. This
  removes overlap double-counting from 6.5-degree half-angle windows around
  detectors spaced by 10 degrees.
- Removed the old rule that suppressed ballistic rays in rear detectors.
- Saves the canonical detector index and full 3-D ray record in the exit CSV.

## Automatic diagnostics

`claritas_production_diagnostics.py` now writes 14 CSVs for every wavelength:

- run summary;
- scatter-count histogram;
- path-length histogram;
- ultimate absorption probability versus scatter count;
- detector contribution versus scatter order;
- per-detector absolute efficiency;
- scatter statistics;
- internal-floc scatter statistics;
- extinction statistics;
- exit-direction polar distribution;
- exit-position polar distribution;
- exit-position azimuth distribution;
- Cartesian exit-position marginals;
- diagnostic integrity checks.

Canonical fractions use launched rays as their stated denominator. Ballistic,
single, and multiple mean 0, 1, and at least 2 outer scatters. Floc and primary
event fractions use all selected outer extinction events when the new
`floc_extinction_count` is present; successful-scatter fractions are reported
separately. The helper can also post-process CLARITAS 76 HDF5 files, with
legacy limitations labelled rather than hidden.

## Measured-data comparison

- `claritas_measured_comparison.py` parses Loess 0.5, Loess 4.0, Kaolin 0.5,
  and Kaolin 4.0 directly from `sediment_data.csv`.
- Reports shape RMSE, MAE, maximum error, and forward/middle/rear RMSE and MAE.
- Separately reports per-angle hit/launched efficiency, unique absolute
  detector efficiency, total detected, escaped, absorbed, truncated, and
  unclassified fractions.
- Writes per-angle comparison CSV plus one-row metrics CSV and JSON.
- Does not invent an absolute measured residual: all four measured columns are
  unit-sum angular curves and contain no absolute radiometric calibration.
- The existing parameter-screen driver now makes the measured CSV and helper
  imports available to generated production cases, records the new absolute
  simulation fractions, exposes bounded GPU chunk overrides for invariance
  testing, and refuses to resume legacy rows lacking the new
  metrics/run-summary files.

## Validation files

- `validate_claritas_77.py`: post-run HDF5 geometry, state closure, event
  invariants, path coverage, and optional same-seed reproducibility checks.
- `validate_claritas_77_reference_parity.py`: reconstructs the existing
  primary-only CPU reference from HDF5 provenance and applies explicit
  two-sample Monte Carlo gates to terminal, scatter-order, detector,
  direction, symmetry, coefficient, and accounting metrics.
- `test_claritas_production_diagnostics.py`: CPU unit tests for canonical and
  legacy diagnostic schemas.
- `validate_claritas_measured_comparison.py`: CPU tests for dataset parsing,
  region metrics, and transport-accounting integration.
- `CLARITAS_77_TRANSFER_AUDIT.md`: itemized CPU-reference versus CLARITAS
  algorithm classification and transfer disposition.
- `CLARITAS_77_VALIDATION.md`: exact Ubuntu/CUDA execution instructions.

## Important floc limitation

The outer 3-D direction transfer requires the active finite-floc walk to accept
nonzero z directions, so its old 2-D circular walk was converted to a spherical
3-D walk. This is a real floc-physics change, not a validated floc redesign.
The CPU reference contains no flocs. Central floc impact parameters,
internal-cap projection, near-wall floc displacement, and the wider inherited
floc architecture remain explicitly unvalidated and are listed in the transfer
audit.

The current production configuration contains one wavelength (622 nm).
Although the inherited script exposes a wavelength list, cross sections and
bulk coefficients are still prepared from its first element. Multi-wavelength
runs must not be interpreted until those tables are rebuilt per wavelength, as
described in the transfer audit.

## Checks executed in this environment

- Python bytecode compilation for production, helpers, screen, and validators.
- Standalone CUDA C++ compilation of the exact embedded kernel with CUDA
  12.3 `nvcc`.
- CPU reference validation: 10/10 tests and 19/19 metrics passed.
- Production diagnostic tests: 4/4 passed.
- Measured-comparison validation: 3/3 passed.
- Synthetic CLARITAS 77 HDF5 acceptance: all checks passed.
- Synthetic zero-extinction CPU-reference parity fixture: 52/52 checks passed.

## Not executed here

The CUDA kernel could not be launched. NVML reports that GPU access is blocked,
and CuPy reports `cudaErrorInsufficientDriver`. Therefore production ray
tracing, GPU statistical parity, same-seed GPU replay, chunk-size invariance,
OOM retry under real device pressure, and new four-dataset CLARITAS 77 results
remain to be run on the target Ubuntu workstation.
