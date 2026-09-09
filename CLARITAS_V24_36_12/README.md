# CLARITAS V24.36.12 — final mixed-kernel reciprocity decision diagnostic

V24.36.12 is **diagnostic-only**. Production transport remains frozen to the V24.29 event-rate physics plus the inherited V24.36 production core. The purpose of this release is deliberately narrow: decide whether the **complete current rough + geometric-fallback boundary kernel** violates the appropriate optical reciprocity/detailed-balance relation.

## What changed from V24.36.11

Three confirmed diagnostic bugs are corrected:

1. Probability closure no longer double-counts entry reflections; `escaped_count` already includes them.
2. Reverse reflection Fresnel tests retain the same `n1 -> n2` medium orientation; only transmission reverses media.
3. Exact reciprocity invariants reconstruct deterministic outgoing directions from the recorded incident direction and microfacet normal instead of treating float32 stored directions as exact mathematical states.

A new complete-kernel equilibrium-flux Monte Carlo test includes both accepted rough interactions and geometric fallback in the **final transition counts**.

## Complete-kernel state measure

At a surface in isotropic equilibrium radiance, the boundary crossing measure is proportional to

`n^2 * mu * dOmega`,

where `mu=|v·n|`. The diagnostic therefore samples `p(mu)=2mu` on each side of the interface. For internal reflection transitions the constant `n_particle^2` cancels between forward and reverse. For cross-interface transmission the measured fluxes retain the corresponding `n_particle^2` and `n_medium^2` factors.

Internal states are binned by geometric-sphere incidence relative to the critical cosine into ordinary outside-TIR, near-critical outside-TIR, near-critical TIR, and grazing TIR. The primary decision pair is the aggregate `outside macro-TIR <-> inside macro-TIR` reflected transition.

The final mixed kernel is tested after all compatibility and fallback decisions. Thus fallback probability mass is not compared as though it were a continuous microfacet density; it is included in the actual final-state transition count.

## Decision rule

For each material, the primary complete-kernel pair fails only when the forward/reverse residual is both:

- greater than the fixed 1% engineering floor, **and**
- at least 5 estimated standard errors from zero.

Both kaolin and loess failing -> `PROCESS_DEFECT_CONFIRMED`.
Both passing -> `PROCESS_DEFECT_NOT_CONFIRMED`.
Mixed result -> `INSUFFICIENT_EVIDENCE`.

Thresholds are fixed before Ben's GPU run and are not selected from H170.

## Frozen physics

No H170 fit, angular remap, Mie moment fit, empirical loss, arbitrary escape, roughness tuning, or branch suppression is introduced. Particle `k=0` and RMS slope 25° remain frozen. The inherited production 65-reflection numerical sink also remains frozen in this diagnostic release and is explicitly reported; it is **not** claimed to be fixed.

## Canonical run

```bash
cd "/f/Google Drive Sync/CLARITAS_code/CLARITAS/CLARITAS_V24_36_12"
conda activate claritas
rm -rf ~/.cupy/kernel_cache
rm -rf ~/.cupy/jitify_cache
rm -rf results
rm -rf __pycache__
python preflight_v24_36_12_package.py && \
python verify_v24_36_12_reciprocity_fallback_audit.py && \
python run_v24_36_12_campaign.py
```

The campaign finalizer cannot run the six H170 sediment cases. It only finalizes this diagnostic.
