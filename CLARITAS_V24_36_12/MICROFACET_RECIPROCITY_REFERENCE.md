# Reciprocity reference used by V24.36.12

The deterministic microfacet tests use path reversal and Fresnel/BSDF reciprocity as hard local invariants. Reflection remains in the same incident medium, while a reversed transmission exchanges the two media.

For the stochastic complete boundary kernel the diagnostic does **not** compare raw forward and reverse microfacet PDFs directly. Instead it samples the optical equilibrium boundary-flux measure proportional to `n^2 |v·n| dOmega` and compares final transition flux after accepted-rough and fallback processing.

For internal reflection the medium is the same before and after the event, so the `n^2` factor is common. For transmission between particle and water, the two side flux estimates retain their respective `n^2` factors. The discrete probability mass produced by compatibility fallback is therefore naturally included through the final outcome frequencies rather than assigned a fictitious continuous Jacobian.

This is a coarse-grained detailed-balance test of the implemented mixed kernel. It is intentionally focused on the macro-TIR basin implicated by V24.36.9/V24.36.11 rather than an attempt to prove every possible microscopic surface model.
