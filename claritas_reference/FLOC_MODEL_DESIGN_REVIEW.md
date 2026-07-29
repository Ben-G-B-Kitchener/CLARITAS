# Floc model design review

No floc model is implemented in the reference transport. This document defines
the decision required after primary transport validation.

| Approach | Physical consistency | Required inputs | Runtime/GPU suitability | Validation opportunities | Main limitations |
|---|---|---|---|---|---|
| Effective-medium sphere | Internally coherent if one mixing rule, complex index, and sphere solver provide `sigma_s`, `sigma_a`, and phase function together | Floc diameter distribution, solid fraction/density, constituent indices, valid effective-medium rule | Low runtime; excellent CPU/GPU lookup suitability | Compare extinction and phase functions with monodisperse aggregate measurements and higher-fidelity calculations | Misses non-sphericity, orientation, monomer correlations, and multiple internal interactions outside effective-medium validity |
| Multi-sphere T-matrix or DDA | Highest direct electromagnetic fidelity within solver limits | Realistic connected aggregate geometries, monomer sizes/indices, orientation ensemble, wavelength | Expensive offline; runtime tables are GPU-friendly; direct online use is impractical | Cross-solver benchmarks, convergence with discretisation/multipole order, laboratory aggregate data | Size parameter and monomer-count limits; substantial offline compute and storage |
| Explicit monomer-cloud photon transport | Process-interpretable in geometrical/radiative-transfer regime, but not a wave solution | Aggregate geometry, monomer extinction/phase functions, overlap/contact rules, pathwise absorption, interface treatment | Expensive; parallelisable on GPU | Analytic optically thin limit, path distributions, comparison with electromagnetic solvers where both apply | Neglects coherent interference unless added consistently; cannot simply coexist with aggregate-level scattering cross-sections |
| Hybrid precomputed aggregate response | Coherent at runtime if every table row comes from one documented solver and contains matched `sigma_s`, `sigma_a`, and phase function | Same inputs as chosen offline solver plus interpolation variables and uncertainty | Best runtime/GPU tradeoff | Validate table nodes against source solver and interpolation against held-out nodes | Accuracy limited by source solver and coverage; interpolation can violate conservation unless constrained |

## Recommendation

Begin with an **effective-medium sphere as an explicitly labelled baseline**, not
as a presumed final truth. In parallel, generate a small set of physically
connected aggregates and calculate higher-fidelity responses offline. If the
effective-medium baseline fails those comparisons materially, adopt a **hybrid
precomputed aggregate response** based on the validated higher-fidelity solver.

Do not use explicit internal photon transport merely to broaden a phase function.
It is appropriate only when aggregate dimensions and optical depth place the
problem in a radiative-transfer regime and when geometric encounter probability,
no-interaction traversal, internal scattering, and pathwise absorption are all
implemented as one model.

## Inputs required before approval

1. Measured floc size distribution for each material and concentration under the
   actual mixing and elapsed-time protocol.
2. Fractal dimension and prefactor, including size dependence and uncertainty.
3. Primary monomer size distribution within flocs.
4. Complex refractive indices of solids and liquid at each wavelength.
5. Aggregate shape/orientation assumptions.
6. Independent extinction or transmission data for flocculated suspensions.

Until these exist, collision length, pooling-kernel width, and scatter efficiency
must not be fitted to detector curves and presented as validated floc physics.

