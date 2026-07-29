# CLARITAS reference architecture

## Coordinate system and sample

The initial reference sample is a sphere of radius 49 mm centred at the origin.
The source enters at `(0, -R, 0)` and its collimated direction is `+y`. Photon
position and direction are always three-dimensional double-precision vectors.

The spherical boundary is an explicit modelling assumption. The legacy scripts
transported in a two-dimensional circle; that geometry has no unique physical
three-dimensional extension without the actual vessel dimensions. A sphere
preserves the legacy radial path scale while permitting closed, testable 3-D
boundary intersections. A finite cylinder and vessel walls should be added only
when their measured dimensions are available.

## Detector geometry

The 18 detector centres remain 0° through 170° in 10° increments with a 6.5°
half-acceptance. An exit's polar position angle is measured from forward `+y`:

`theta_detector = arccos(y_exit / |r_exit|)`.

The initial detector is therefore a set of ideal annular bands on the spherical
boundary. Each photon can be assigned to at most one nearest detector band. This
is a clear 3-D generalisation of the legacy radial exit-position detector, but
it is not yet a finite photodiode model. Detector area, orientation, refractive
interfaces, and responsivity remain unresolved physical inputs.

## Optical and transport probabilities

PSD weights are explicitly interpreted as mass fractions:

`n_i = C w_i / m_i`.

Primary-particle Mie theory supplies mutually consistent `sigma_t`, `sigma_s`,
`sigma_a`, and angular amplitudes. Bulk coefficients are sums of `n_i sigma_i`.

Each active photon:

1. samples `s = -ln(U)/mu_t`;
2. exits if the boundary is closer than `s`;
3. otherwise selects bin `i` from `n_i sigma_t,i / mu_t`;
4. is absorbed with probability `1 - sigma_s,i/sigma_t,i`;
5. or samples polar angle from `I(theta) sin(theta)` and azimuth uniformly;
6. rotates its full 3-D direction in a local orthonormal basis.

Outcomes are mutually exclusive: exited, absorbed, or truncated. Exits are
further classified as ballistic or scattered, and may or may not intersect an
ideal detector band.

## Module structure

| Module | Responsibility |
|---|---|
| `config.py` | Immutable material, detector, and simulation inputs with units |
| `physics.py` | PSD number density, complex-index Mie optics, physical phase CDFs |
| `geometry.py` | Sphere intersections, 3-D direction rotations, detector assignment |
| `transport.py` | Seeded event-driven CPU Monte Carlo and outcome accounting |
| `validate.py` | Analytic/statistical validation suite and CSV summaries |
| `run_benchmarks.py` | Four measured-data runs, metrics, and comparison files |

## Legacy disposition

| Category | Item | Disposition |
|---|---|---|
| Retain from CLARITAS_76 | Event-driven extinction and exponential paths | Implemented |
| Retain from CLARITAS_76 | `n_i sigma_t,i` event selection | Implemented |
| Retain from CLARITAS_76 | Per-bin albedo and complex-index Mie | Implemented |
| Retain from CLARITAS_76 | Explicit units, diagnostics, measured CSV loading | Implemented |
| Retain later | GPU chunking and binary-search techniques | Only after CPU validation |
| Comparison only from CLARITAS_18 | Historical detector curve and fixed-step behavior | Never used by new transport |
| Comparison only from CLARITAS_18 | Pooled homogeneous-sphere floc response | No flocs in reference |
| Reject | Deterministic `0.1/mu_s` scattering steps with probability 1 | Replaced by exponential paths |
| Reject | Raw `I(theta)` per-degree CDF | Replaced by `I(theta) sin(theta)` |
| Reject | Signed planar polar-angle update | Replaced by 3-D rotation and uniform azimuth |
| Reject | 7% floc direction reversal | Removed |
| Reject | Ballistic rear-detector suppression | Removed |
| Reject | Detector-specific angular powers or scaling | Not present |
| Unresolved input | Meaning of source PSD weights | Currently declared mass fraction |
| Unresolved input | Material- and wavelength-specific complex indices | Defaults are explicit assumptions |
| Unresolved input | Actual vessel shape, walls, and interfaces | Sphere used as validation geometry |
| Unresolved input | Finite detector aperture and responsivity | Ideal spherical bands used |
| Unresolved input | Absolute source/detector calibration | Raw fractions retained but not instrument calibrated |

## Acceptance gates

Flocs and GPU transport are prohibited from the reference branch until:

- analytic zero-concentration and pure-absorption checks pass;
- angular sampling and 3-D isotropy checks pass;
- coefficient concentration scaling passes;
- detector assignments pass known-ray tests;
- fixed-seed output is identical;
- exit/absorption/truncation accounting closes;
- all four primary-only benchmark outputs have been recorded.

