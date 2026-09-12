# V24.50 Validation-Correction Evidence

## Baseline evidence

V24.50 uses the supplied V24.49 package as its sole code baseline and includes a compact copy of the supplied V24.49 GPU decision/local-replay evidence under `references/v24_49_gpu_results/`.

The V24.49 GPU evidence establishes:

- boundary/microsurface GPU validation passed;
- deterministic connected-solid geometry passed;
- the 500,000-state one-step CUDA/reference gate passed;
- particle accessibility had zero mismatches in 100,000 states;
- the historical 129-ray gate passed after local replay;
- the two historical global residuals were both locally correct;
- eleven of twelve 4096-path global residuals were locally correct;
- the sole remaining local replay failure was full-path case 1650;
- no genuine production CUDA topology error had been demonstrated.

## Independent case-1650 reconstruction

V24.50's deterministic host regression reads the recorded V24.49 local-replay state for case 1650 and evaluates the independent geometry twice:

1. using the original unquantized Python scalar value;
2. using the exact float32-effective value actually supplied to CUDA.

With the original double `Rin`, the host reconstruction reproduces an intersection-distance error of approximately `1.38485e-6 m`, larger than the fixed `5e-7 m` local replay tolerance.

With `Rin` quantized exactly as the CUDA launch does, the independent high-precision result agrees with the recorded CUDA intersection distance to approximately `7e-14 m` while retaining the same surface and TIR result.

This is evidence of a reference-input-domain mismatch, not evidence for a production geometry correction.

## General correction

The V24.50 reference path does not special-case case 1650. It derives CUDA-effective scalar inputs from the actual validation launch domain and applies them consistently to the independent one-step/local-replay geometry calculations. High-precision host mathematics is retained after input quantization.

A machine-readable scalar parity report is generated for the relevant geometry/optical inputs.

## Tolerances

The local replay distance tolerance remains exactly `5e-7 m`. No geometry tolerance is enlarged to close the case. The deterministic regression explicitly records `tolerance_relaxed=false`.

## Strong gates retained

The following remain release blockers and are not replaced by the case-1650 regression:

- boundary/microsurface GPU suite;
- deterministic connected-solid geometry suite;
- raw-CUDA memory-layout regression;
- 500,000-state one-step comparison;
- 100,000-state particle accessibility comparison;
- historical and long-path local replay correctness;
- invalid-state and watchdog semantics;
- CUDA/Python ABI consistency.

Global float32-vs-float64 whole-orbit identity remains diagnostic only.

## Production hash invariant

The packaged production `CUDA_SRC` must remain byte-identical to the frozen V24.47 implementation with SHA-256:

`e230658265a5e5826d06760bd49a7e10d9197ac9f9040bf414b3b413f9b8f9e4`

The host verifier treats a mismatch as a release blocker.

## Fail-closed outcome

If a compatible CUDA runtime is absent in the build environment, V24.50 may establish only package/host/static evidence. GPU geometry, million-ray qualification and the five-case integrity matrix remain `NOT RUN`, and production authorization remains false.

On a real CUDA machine, if all residual local replay cases pass after scalar parity correction, the geometry gate closes and the verifier proceeds automatically to the exact 1,000,000-ray qualification. No additional geometry audit stage is inserted.
