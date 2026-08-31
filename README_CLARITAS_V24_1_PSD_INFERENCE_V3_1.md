# CLARITAS V24.1 + PSD Inference V3.1

## Purpose

V24.1 is a detector-statistics release.  It **does not change the V24 optical
physics**: the 93-mm-ID / 100-mm-OD acrylic TARDIIS cell, water/acrylic/air
Snell/Fresnel interfaces, multiple wall reflections/TIR, geometric particle
cross-section `pi*r^2`, 3-D spherical particle Snell/Fresnel transport and the
physical 4-mm two-plane source/detector collimators are unchanged.

The problem addressed is that the real detector apertures are small.  In the
1,000,000-ray kaolin 4 g/L V24 test only 624 rays registered exact detector
hits, making a 25k-100k-ray PSD-optimiser evaluation far too noisy.

## V24.1 detector estimator

Every ray is still propagated physically all the way into external air.
V24.1 then computes **two detector estimates in parallel**:

1. **Exact** — the original V24 binary test: a ray must pass through both real
   2-mm-radius apertures.  Each ray can hit at most one physical detector.
2. **Variance-reduced (default)** — for each detector, compute the transverse
   miss vectors `q1` and `q2` at the two real aperture planes and estimate the
   local 4-D ray phase-space density with a product Epanechnikov kernel.

With real aperture radius `a` and kernel bandwidth `h = f*a`,

```
K = max(0, 1-|q1|^2/h^2) * max(0, 1-|q2|^2/h^2)
```

The physical-equivalent score is

```
score = sum(K) * 4*(a/h)^4
```

because each 2-D Epanechnikov kernel integrates to `pi*h^2/2`, while the two
real circular apertures have phase-space area `(pi*a^2)^2`.

The default fixed bandwidth factor is `f=3`.  It is intentionally held fixed
for every detector, concentration and optimiser evaluation.  This reduces
variance while keeping the common-random-number objective smooth.  The exact
physical hit distribution is always written alongside the VR result so that
high-statistics runs can validate estimator bias.

This is a **kernel density estimator**, so it trades finite smoothing bias for
variance reduction.  It is not claimed to be an exactly unbiased adjoint or
next-event estimator.  For final publication-quality results, compare VR and
exact responses at high ray count and reduce the bandwidth if necessary.

## Files

- `CLARITAS_24_1_31-08-2026_TARDIIS_variance_reduced.py`
- `claritas_tardiis_core_v24_1.py`
- `claritas_forward_inference_v3_1.py`
- `CLARITAS_PSD_Inference_V3_1.py`
- `psd_inference_v3_1_config.json`
- `measured_detector_responses_v3.csv`

## First validation run

Use the demanding kaolin 4 g/L condition:

```bash
python CLARITAS_24_1_31-08-2026_TARDIIS_variance_reduced.py \
  --material kaolin --concentration 4.0 --n-rays 1000000
```

Inspect `claritas_v24_1_results/kaolin_4gL/detector_response_normalized.csv`.
It contains the selected response, VR response, exact response, physical-
equivalent VR score, support-ray count and exact physical hits.

The key validation is whether `normalized_variance_reduced` approaches
`normalized_exact` at 1M+ rays while having far larger support counts in the
weak detector channels.

A useful bandwidth sensitivity check is:

```bash
python CLARITAS_24_1_31-08-2026_TARDIIS_variance_reduced.py --material kaolin --concentration 4.0 --n-rays 1000000 --vr-bandwidth-factor 2.0 --output-dir claritas_v24_1_results/kaolin_4gL_bw2
python CLARITAS_24_1_31-08-2026_TARDIIS_variance_reduced.py --material kaolin --concentration 4.0 --n-rays 1000000 --vr-bandwidth-factor 3.0 --output-dir claritas_v24_1_results/kaolin_4gL_bw3
```

If the two VR curves agree closely, bandwidth bias is small relative to the
Monte-Carlo variance being removed.

## PSD inference V3.1

Do not infer optical parameters.  V3.1 retains V3's seven non-monotonic PSD
controls per concentration plus one effective local-concentration factor per
condition.  The suspension is interpreted as continuously magnetically
stirred; there is no settling-time clock.

Baseline first:

```bash
python CLARITAS_PSD_Inference_V3_1.py --material all --baseline-only --quick
```

Then warm-start from V2 or V3 24-parameter solutions if desired.  V3.1 output
is written to `psd_inference_v3_1_results/`, so earlier inference results are
not overwritten.
