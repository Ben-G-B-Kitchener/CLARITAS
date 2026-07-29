# CLARITAS_18 behavioural comparison

## Availability of a numerical legacy curve

The workspace does not contain a provenance-labelled detector output generated
by `CLARITAS_18_17-06-2026.py` under the same Loess 0.5 g/L inputs and scoring
pipeline. The legacy script cannot be rerun in this environment because its
CuPy kernel requires a usable CUDA driver; CUDA initialisation currently fails
with `cudaErrorInsufficientDriver`.

No CLARITAS_18 curve has therefore been invented or substituted. A valid
side-by-side numerical CSV requires one of:

1. running the frozen script in its original working CUDA environment and
   supplying its `detector_hits.csv`, or
2. supplying the archived detector output that supported the historical
   “almost perfect” claim.

The new result is available at
`benchmark_outputs/loess_0p5gL/measured_vs_model.csv`; measured values are read
directly from `sediment_data.csv`.

## Quantified code-level differences

| Mechanism | CLARITAS_18 | New reference | Quantified consequence |
|---|---|---|---|
| Bulk free path | Fixed `0.1/mu_s`, scattering probability 1 | Exponential mean `1/mu_t` | With no absorption, the legacy mean event spacing is 10× too short. The correct Bernoulli probability over its step would be `1-exp(-0.1)=0.09516`, not 1. |
| Free-path variance | Zero | `1/mu_t²` | Legacy event counts are artificially regular; the reference has Poisson transport in a homogeneous medium. |
| Polar probability | Cumulative `I(theta)` | Cumulative `I(theta)sin(theta)` | Legacy cross-sections and sampled phase angles use incompatible angular measures. |
| Azimuth/direction | Random sign applied in a plane | Uniform azimuth and local 3-D vector rotation | Legacy paths cannot leave the original plane; reference paths explore the full solid angle. |
| Floc reversal | 7% of floc events add π | Absent; flocs disabled | For a ray with `N_f` floc events, the probability of at least one artificial reversal is `1-0.93^N_f`. |
| Floc optics | Pooled effective homogeneous spheres | No flocs | The reference intentionally makes no unconstrained aggregate claim. |
| Rear ballistic rule | Ballistic hits suppressed for detector centres ≥90° | No class-specific suppression | Reference detector assignment depends only on physical exit position. |

## Loess 0.5 g/L result

At 100,000 rays the new primary-only result has shape RMSE 0.084251. Its total
detected fraction is 0.698910, total exit fraction is 0.702050, and absorbed
fraction is 0.297950. These values are not expected to reproduce CLARITAS_18:
the reference deliberately removes flocs, deterministic over-scattering, planar
angle sampling, reversal, and rear-bin filtering simultaneously.

The historical fit is most plausibly explained by cancellation between those
mechanisms. It remains a behavioural datum worth preserving, but not a target
that the physical reference should reproduce before independently constrained
floc and instrument optics are added.

## Required follow-up

When the genuine CLARITAS_18 output is available, add its raw counts as a
separate column without renormalising away the total. Then report:

- unit-sum shape residuals for measured, CLARITAS_18, and reference;
- raw hit fractions and total exit/absorption fractions where legacy data permit;
- per-angle differences;
- Monte Carlo uncertainty from replicated fixed configurations.

