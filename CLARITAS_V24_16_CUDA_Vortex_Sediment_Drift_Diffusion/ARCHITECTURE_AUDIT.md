# V24.16 Architecture Audit

## CUDA-authoritative boundary

Ray-determining forward physics remains CUDA-resident. V24.16 adds GPU-resident sediment transport to the validated V24.15 free-surface architecture:

- per-PSD-bin Stokes response time and settling velocity;
- per-bin radial and vertical drift–diffusion coefficients;
- numerical volume normalisation over the curved water domain;
- exact per-bin maximum density scale and optical majorant construction;
- local size-resolved sediment density evaluation;
- Woodcock/delta tracking of spatially varying interaction rates;
- local interacting-size-bin sampling after an accepted collision.

The existing CUDA path still determines Mie/event coefficients, source launch, free-surface intersections/normals/Fresnel/TIR, acrylic/cell propagation, particle scattering and detector acceptance. Python configures runs, launches kernels, copies diagnostic products, computes reporting metrics and writes files; it does not determine a ray outcome.

## Sediment mass-conservation boundary

For each PSD bin the unnormalised steady field is evaluated in peak-scaled form

```text
E_i(r,z) = exp[A_i(f(r)-f(R)) - B_i(z-zbottom)]
f(r)     = r^2/(a^2+r^2).
```

CUDA integrates `E_i` over the actual V24.15 curved water volume and stores

```text
Nscaled_i = <E_i>_vessel
S_i       = E_i/Nscaled_i.
```

Therefore `<S_i>_vessel=1` independently for every PSD bin. This preserves the integrated number and mass of each size class, rather than only conserving total concentration after reshaping the PSD.

## Majorant boundary

For the V24.16 field, `A_i>=0`, `B_i>=0`, `f(r)<=f(R)` and `z>=zbottom`, so `E_i<=1`. The exact per-bin maximum local density factor is consequently

```text
S_i,max = 1/Nscaled_i
```

at the bottom outer wall. The constant geometric and wave Woodcock majorants are sums of the corresponding bulk per-bin optical rates multiplied by `S_i,max`. At a candidate point, the local rate is recomputed from all bins. The acceptance ratio is capped at one as a numerical guard, but the verifier checks the constructed majorant against sampled local fields.

## Uniform regression boundary

`sediment_transport_model=uniform` sets every local density scale to exactly one. The trace kernel then uses the historical V24.15 `MU_TOTAL` exponential free path and historical event-bin CDFs rather than Woodcock null collisions. This explicit branch is intended to preserve the previous random stream and detector result for fixed seed/configuration.

## Frozen-geometry boundary

The production V24.16 screen holds the independently tested V24.15 localized Scully surface at `a=20 mm`, `D=50 mm`. The old `flat` and `parabolic_vortex` selectors remain available for controlled regression, but the V24.16 D_t sweep does not refit free-surface geometry.

## Verification boundary

`verify_v24_16_cuda_sediment_transport.py` contains two levels of checks:

1. Host-only checks compare the closed-form coefficients with an independent two-dimensional vessel integration, verify per-bin mass conservation, and verify physically expected size/D_t monotonicity.
2. CUDA checks compare GPU-generated coefficients and pointwise density scales against the independent host reference, verify the uniform branch, inspect majorants, and run ray-transport regression/smoke cases.

## Deliberate scope limit

V24.16 treats unresolved stirring/recirculation through a scalar effective eddy diffusivity. It is not a solved Navier–Stokes/CFD field. No finite-Re drag correction, anisotropic diffusivity, explicit meridional velocity field, wall deposition/resuspension model, aggregation/fragmentation, sediment feedback on flow, or air-headspace re-entry is added. These mechanisms must not be inferred from the release.
