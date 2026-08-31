# CLARITAS V24.2 + PSD Inference V3.2

## Purpose

V24.2 keeps the V24.1 optical model unchanged and replaces only the detector estimator. The TARDIIS water/acrylic/air cylinder, particle geometric encounters, 3-D particle Snell/Fresnel physics, acrylic-wall Fresnel/TIR, source collimator and physical detector collimators are unchanged. The experimental interpretation remains **continuous magnetic stirring**; there is no stirrer-off settling clock.

## Why V24.2

A single fixed KDE bandwidth reduced shot noise but introduced angular smoothing bias. V24.2 scores every completed ray set at bandwidth factors **2, 3 and 4** times the real 2-mm aperture radius, converts each to a physical-equivalent score, then fits per detector

`H(h) = H0 + A (h/a)^2`

and uses the non-negative zero-bandwidth intercept `max(H0,0)` as the production detector score. Exact physical aperture hits are still computed.

This is a detector **estimator** change, not an optical-physics change.

## First validation run

```bash
python CLARITAS_24_2_31-08-2026_TARDIIS_multiband_extrapolated.py --material kaolin --concentration 4.0 --n-rays 1000000
```

Results are written to `claritas_v24_2_results/kaolin_4gL/`. Inspect `detector_response_normalized.csv` and `detector_scores_multiband.csv`. The former contains exact, each bandwidth, raw extrapolated intercept, non-negative extrapolated score, R2 and bandwidth spread.

The most useful audit is to compare the normalized columns for bw=2, bw=3, bw=4, h->0 extrapolated and exact. A good estimator should move the extrapolated curve toward the exact high-statistics physical result while using far larger support counts than exact binary scoring.

## Inference V3.2

Do not fit optical parameters. V3.2 retains the V3.1 seven-knot per-concentration PSD functions and local-concentration factors. Its only forward-model change is that `normalized_response` now uses the fixed V24.2 h->0 detector estimate.

Quick baseline:

```bash
python CLARITAS_PSD_Inference_V3_2.py --material all --baseline-only --quick
```

Warm-start from V2/V3/V3.1 compatible 24-parameter results:

```bash
python CLARITAS_PSD_Inference_V3_2.py --material kaolin --quick --warm-start-v2 psd_inference_v2_results/kaolin/best_fit_parameters.json --skip-local
```

V3.2 quick mode is deliberately raised to **100k rays/evaluation** and **500k final rays** because zero-bandwidth extrapolation still requires adequate support at the smallest bandwidth. Normal mode uses 250k/2M rays. Edit the JSON if runtime requires a different compromise.

## Key files

- `CLARITAS_24_2_31-08-2026_TARDIIS_multiband_extrapolated.py`
- `claritas_tardiis_core_v24_2.py`
- `claritas_forward_inference_v3_2.py`
- `CLARITAS_PSD_Inference_V3_2.py`
- `psd_inference_v3_2_config.json`
- `measured_detector_responses_v3_2.csv`
- `verify_v24_2_detector_estimator.py`

## Notes on extrapolation

The leading `h^2` bias assumption is a local asymptotic model, not a new physical law. V24.2 therefore exports the per-detector regression R2 and relative spread between bandwidth scores. Poor R2 or large spread flags a detector where the extrapolation is not yet in a clean local-bias regime. The raw intercept is never hidden; negative raw intercepts are reported and only the production physical score is clipped to zero.
