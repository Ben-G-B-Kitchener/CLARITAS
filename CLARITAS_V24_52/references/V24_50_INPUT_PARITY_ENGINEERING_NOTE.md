# V24.50 Scalar-Input-Parity Closure Engineering Note

## Scope

V24.50 is intentionally the final geometry-validator closure release. It starts from the V24.49 package and keeps production `CUDA_SRC` frozen to the V24.47 connected-solid implementation. The release addresses one remaining validator false negative rather than changing production optical/transport physics.

## What V24.49 established

The supplied V24.49 real-GPU run passed the boundary/microsurface suite, deterministic connected-solid topology, the 500,000-state one-step CUDA/reference comparison, the corrected 100,000-state particle-accessibility comparison and the historical 129-ray geometry gate. The V24.49 local-replay closure population contained two historical residuals and twelve 4096-ray stress residuals. Thirteen of those fourteen cases were classified `LOCAL CUDA CORRECT`.

No local CUDA topology defect was demonstrated.

## The remaining case: 1650

The single failing local replay was full-path case 1650. At its first meaningful divergence, CUDA and the independent reference already agreed on:

- physical surface: 1;
- interface classification;
- TIR: true.

The sole failing quantity was the local intersection distance. V24.49 compared approximately:

- CUDA `t = 0.0222776802623251 m`;
- unquantized-reference `t = 0.0222790651161 m`;
- absolute difference approximately `1.38485e-6 m`;
- fixed validation tolerance `5e-7 m`.

The tolerance was not the defect.

## Root cause: reference input value did not equal CUDA input value

The model configuration carries `Rin = 0.0465 m` as a Python double. The validation kernel launch explicitly converts that scalar to `np.float32`. The value actually supplied through the CUDA ABI is therefore approximately:

`0.04650000110268593 m`

The V24.49 independent replay instead used the original unquantized Python value:

`0.04650000000000000 m`

The difference is only about 1.1 nm in radius. Case 1650 is extremely grazing, so this tiny parameter-domain difference is amplified into a micrometre-scale path-distance difference.

When the independent quadratic is evaluated in high precision using the exact float32-effective radius supplied to CUDA, the host result is approximately:

`0.0222776802622524 m`

which differs from the recorded CUDA distance by only about `7e-14 m`.

## V24.50 rule

The independent reference must test the same numerical problem that CUDA received.

For every relevant scalar kernel argument:

1. determine the actual CUDA ABI type from the launch/signature;
2. form the exact effective value delivered to CUDA;
3. use that effective value as the input to the independent host oracle;
4. keep the oracle mathematics high precision.

For float32 CUDA scalars this means conceptually:

```python
value_ref = float(np.float32(value_python))
# Use value_ref in double-precision independent calculations.
```

This does **not** mean converting the host reference calculation itself to float32.

## Parameters audited

The V24.50 geometry validator constructs effective reference inputs for the relevant float32 geometry/optical scalars used in local replay and one-step comparisons, including the vessel geometry, free-surface/vortex scalars and refractive-index inputs actually passed to the validation kernels. The machine-readable parity report records the original value, ABI cast type, CUDA-effective value, reference-effective value and parity result.

The implementation is driven by the actual V24.50 validation launch semantics rather than by case-specific constants.

## Deterministic regression

V24.50 packages the recorded V24.49 local-replay evidence and includes a deterministic case-1650 host regression. It must prove:

- the original unquantized scalar reproduces the previous excess distance error;
- the CUDA-effective scalar closes that error within the unchanged `5e-7 m` tolerance, normally to near roundoff;
- surface selection remains unchanged;
- TIR classification remains unchanged;
- `tolerance_relaxed=false`.

The regression fixture may identify case 1650, but normal replay acceptance remains generic and does not special-case that ray ID.

## Closure criterion

All long-path global differences continue to be discovered dynamically. Global whole-orbit difference remains diagnostic. Every global residual is release-gated by local replay from the actual CUDA pre-interaction state using CUDA-effective scalar values.

If all local residuals are `LOCAL CUDA CORRECT`, the connected-solid geometry gate passes and geometry auditing stops. The same qualification chain then proceeds directly to the exact 1,000,000-ray production qualification and, if that passes, the five-case integrity matrix.

No additional long-path audit stage is authorized merely because a separately propagated float64 orbit differs globally from float32 CUDA.

## Production physics

Production CUDA remains frozen. The expected `CUDA_SRC` SHA-256 is:

`e230658265a5e5826d06760bd49a7e10d9197ac9f9040bf414b3b413f9b8f9e4`

A production change is justified only by a concrete local CUDA decision shown incorrect after applying the corrected input-parity rule.
