# V24.36.12 architecture audit

## Scope

This is the final narrow decision diagnostic, not another broad audit layer.

Production `claritas_tardiis_core_v24_36.py` is byte-for-byte frozen. The new `v24_36_12_complete_kernel.py` appends a diagnostic CUDA kernel only. It does not call or modify the six-case sediment campaign.

## Boundary process tested

The production process is interpreted as a mixed transition kernel:

1. sample a Beckmann rough normal with the inherited facing-rejection sampler;
2. evaluate rough-interface refraction/TIR;
3. apply geometric compatibility;
4. if compatible, retain the rough proposal;
5. if incompatible, replace the optical normal with the geometric normal and evaluate the geometric branch;
6. for reflection, apply the inherited geometric-normal fallback if the rough reflected ray points outside the ideal sphere.

The complete-kernel diagnostic records the **final** outcome after steps 1–6. This avoids treating fallback as though it had the same continuous density as an accepted microfacet transition.

## Reciprocity measure

The invariant sampling measure is equilibrium surface flux, proportional to `n^2 mu dOmega`. Sampling uses `p(mu)=2mu`. Internal reflection comparisons use the same particle medium on both sides, so the common `n_particle^2` factor cancels. Cross-interface comparisons retain `n_particle^2` vs `n_medium^2`.

## Decision

The primary diagnostic is aggregate reflected flux `outside macro-TIR <-> macro-TIR`. A material fails only when relative residual exceeds 1% and significance is >=5 standard errors. Both materials must fail for `PROCESS_DEFECT_CONFIRMED`.
