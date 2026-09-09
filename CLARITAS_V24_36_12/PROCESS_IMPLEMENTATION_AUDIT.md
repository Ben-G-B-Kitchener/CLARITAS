# V24.36.12 process implementation audit

The release makes no production physics correction.

## V24.36.11 corrections

- Count closure: `resolved = escaped + absorbed + numerical_failure + censored`. `entry_reflection_count` is diagnostic-only and is not added a second time.
- Reverse reflection Fresnel: reflected forward and reverse rays are evaluated with the same incident/transmitted medium pair. Transmission reverses the pair.
- Pair reconstruction: exact deterministic reflected/refracted direction is reconstructed from incident direction + microfacet normal for invariants. The stored CUDA float32 direction is retained only as a precision observation.

## Complete mixed-kernel CUDA diagnostic

`complete_kernel_kernel` independently samples equilibrium incident surface states and reproduces the current local rough/fallback logic. It records final reflection transition matrices, external escape, reverse external entry, accepted-rough contribution, fallback contribution, and candidate-escape probability mass redirected by fallback.

The test is deliberately local because the question is reciprocity of the boundary transition law. Free flight across the ideal sphere is included for reflected internal states to map the outgoing ray to its next geometric-sphere incidence state.

## Statistical rule

For an equilibrium sampled pair, forward and reverse transition-frequency differences are evaluated with a conservative counting standard error. Failure requires both >1% relative residual and >=5-sigma significance. This prevents Monte-Carlo noise from becoming a physics verdict.

## Known production defect retained

The inherited inclusive loop still permits 65 internal reflections and then returns numerical failure. V24.36.12 diagnoses but does not remove that sink.
