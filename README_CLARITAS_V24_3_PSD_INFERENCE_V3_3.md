# CLARITAS V24.3 + PSD Inference V3.3

## Purpose
V24.3 returns to the **exact physical TARDIIS detector aperture** after the V24.1/V24.2 KDE experiments showed channel-dependent smoothing bias. It is deliberately a narrow engineering/statistical release, not a new optical model.

## Physics retained unchanged
- V23.2 geometric particle encounters: `sigma = pi*r^2` and exponential free paths.
- Full 3-D Snell/Fresnel spherical-particle interactions; no Mie, empirical scatter multipliers or roughness boosts.
- V24 cylindrical TARDIIS geometry: 93-mm water ID, 100-mm acrylic OD, air outside.
- Explicit water/acrylic/air refraction, Fresnel reflection, TIR and acrylic-wall multiple reflections.
- Real two-plane 4-mm radial source/detector collimator test.
- Experimental interpretation remains a continuously stirred suspension; no post-stir settling clock is introduced.

## What V24.3 changes
### 1. Streaming high-statistics accumulation
Ray outputs are scored batch-by-batch instead of retaining every ray in RAM. This makes 5M, 10M or larger exact runs practical without host arrays scaling with the full ray count. Per-ray CSV storage remains optional (`--save-rays`).

### 2. Exact symmetry variance reduction
The modeled apparatus is invariant under reflection about the y-z plane (`x -> -x`). For every traced ray, V24.3 evaluates two **hard-aperture** indicators: the native ray and its exact mirrored ray. The production score is

`H_sym = 0.5 * (H_native + H_mirror)`

where each H is still a binary passage through both physical 4-mm collimator planes. This is an unbiased symmetry/Rao-Blackwell average under the V24.3 model. It does **not** enlarge an aperture, blur detector angle, use a KDE, or extrapolate a bandwidth. Native exact counts are retained for audit.

### 3. Optional adaptive exact runs
The production driver can continue exact batches until a requested total symmetry-equivalent detector score is obtained, optionally also requiring cumulative-response stability. This is useful for convergence studies; inference itself uses fixed ray counts so that the objective remains deterministic.

### 4. Batch-jackknife uncertainty
`detector_response_normalized.csv` includes a batch-jackknife standard-error estimate for every normalized detector channel.

## First regression run
Repeat kaolin 4 g/L at 1M rays:

```bash
python CLARITAS_24_3_31-08-2026_TARDIIS_exact_streaming.py --material kaolin --concentration 4.0 --n-rays 1000000
```

Inspect `claritas_v24_3_results/kaolin_4gL/detector_response_normalized.csv`. Compare `normalized_native_exact` with the old V24 exact curve and compare it with `normalized_symmetry_exact`. They should be statistically compatible; the symmetry average should be less noisy.

## High-statistics convergence run
For example:

```bash
python CLARITAS_24_3_31-08-2026_TARDIIS_exact_streaming.py --material kaolin --concentration 4.0 --n-rays 5000000 --output-dir claritas_v24_3_results/kaolin_4gL_5M
```

Or adaptive exact accumulation:

```bash
python CLARITAS_24_3_31-08-2026_TARDIIS_exact_streaming.py --material kaolin --concentration 4.0 --n-rays 10000000 --target-detector-score 5000 --min-rays 1000000 --max-rays 10000000 --stability-l1-tolerance 0.01 --output-dir claritas_v24_3_results/kaolin_4gL_adaptive
```

## PSD Inference V3.3
The PSD/local-concentration parameterisation is deliberately carried forward rather than replaced. V3.3 uses the V24.3 exact symmetry-averaged detector response.

Baseline check:

```bash
python CLARITAS_PSD_Inference_V3_3.py --material all --baseline-only --quick
```

The normal config uses 1M coarse rays and 5M final rays because exact aperture scoring is a rare-event calculation. `--quick` is primarily for code/runtime checks; do not treat a low-hit quick optimisation as a final PSD inference.

Existing V2/V3-compatible parameter files can still be used with `--initial-parameters` / `--warm-start-v2` because the suspension parameterisation remains 7 PSD controls per concentration plus one local-concentration factor per condition.

## Validation
Run the CPU-only estimator check:

```bash
python verify_v24_3_exact_symmetry.py
```

The CUDA/NVRTC kernel still has to compile in the local CLARITAS CUDA environment. The V24 CUDA source itself is retained rather than rewritten.
