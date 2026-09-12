# CLARITAS V24.52.0 — validator correction and full-campaign engineering note

## Scope

V24.52 is a targeted correction to the V24.51 independent complete-apparatus one-step validator plus a campaign-orchestration correction. The production apparatus geometry introduced in V24.51 is deliberately frozen unless the corrected independent oracle proves a local production error.

## V24.51 CUDA evidence reproduced

The real V24.51 run reported 300000/300000 region classifications, 494951/500000 one-step comparisons, 100000/100000 particle-accessibility comparisons, 512/512 V24.50 failure-state regressions and 14/14 historical/local replays. Therefore 5,049 one-step states were the sole complete-geometry gate failure.

V24.52 regenerates the exact deterministic 500,000-state V24.51 population on the host. The old V24.51 oracle differs from the corrected reference in exactly 5,049 states: 3,225 WATER horizontal states and 1,824 HEADSPACE horizontal states. There is no unexplained remainder relative to the reported 5,049.

## Root cause

V24.51 treated `vz == 0` WATER and HEADSPACE rays as wall-targeting and did not test the radially varying free surface. A horizontal ray keeps z fixed but changes x and y, therefore changes radius and can cross a curved `z_free_surface(r)` level set.

V24.52 independently solves the free-surface level-set radius for horizontal rays and intersects the ray's XY line with that radius. It then competes that free-surface distance against the finite inner-wall distance. The reference remains host-only and high precision; it does not call the production CUDA free-surface intersection helper.

## Deterministic regression

Nine host-only cases cover WATER/HEADSPACE wall-first and free-surface-first paths, tangential/no-crossing, near-axis, near-wall, near-free-surface and close-but-unambiguous competition. All pass in the packaged host verifier.

## Production CUDA

`CUDA_SRC` is byte-identical to V24.51. No production geometry, optical, RNG, particle or detector physics was changed for V24.52. The recorded V24.51 CUDA source SHA-256 is stored in `references/V24_51_CUDA_SRC_SHA256.txt`, and host verification checks exact equality.

## Campaign orchestration

Production authorization remains fail-closed. The campaign now separates a fatal executable/topology failure from a release-gate/reference disagreement. If boundary transport and essential apparatus invariants are healthy, diagnostic sediment execution may proceed even if a later geometry release gate or production qualification fails. Such runs remain explicitly diagnostic and do not imply production authorization.

If the complete geometry gate passes, the runner automatically proceeds to the exact 1,000,000-ray production qualification and then the five-case integrity matrix when eligible. The sediment diagnostic phase is then attempted independently when transport is diagnostically safe.

## Sediment diagnostic inventory

The canonical executable diagnostic set remains six cases: loess and kaolin at 0.5, 2.0 and 4.0 g/L. The campaign records per-case detector outputs, H170, ray count, seed, PSD metadata, transport failures, WATER/HEADSPACE/ACRYLIC/microsurface failure counts where available, watchdog/nonfinite counts, elapsed time, output inventory, consolidated CSV/JSON, and plots when matplotlib is available.

## Validation performed in the packaging environment

Host validation passes, including the 5,049-case reconstruction, horizontal-free-surface deterministic regression, case-1650 scalar-input-parity regression, all 512 stored V24.50 host failure states, C++ CUDA syntax/scope surrogate and CUDA/Python ABI audit. CUDA is not available in the packaging environment, so the corrected real-GPU 500,000-state result, million-ray qualification, five-case matrix and sediment simulations are truthfully marked NOT RUN in the packaged candidate.

The intended CUDA-host run is:

```bash
cd "/f/Google Drive Sync/CLARITAS_code/CLARITAS/CLARITAS_V24_52"
conda activate claritas
rm -rf ~/.cupy/kernel_cache
rm -rf ~/.cupy/jitify_cache
rm -rf results
rm -rf __pycache__
python preflight_v24_52_package.py && \
python verify_v24_52_complete_geometry.py --output-dir results && \
python run_v24_52_campaign.py --output-dir results
```
