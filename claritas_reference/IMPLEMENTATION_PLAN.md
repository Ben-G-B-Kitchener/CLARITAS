# Incremental implementation and validation plan

1. Freeze legacy scripts and measured input. Define a spherical 3-D coordinate
   system and ideal annular detector bands.
2. Build immutable configurations and a primary-only optical-medium constructor.
   Confirm Mie cross-sections and phase CDFs use the same complex index.
3. Implement seeded CPU transport with full vectors, exponential paths,
   extinction-weighted bin selection, albedo, and exact sphere crossing.
4. Validate deterministic and analytic limits before interpreting sediment data.
5. Run all four material/concentration cases from `sediment_data.csv`, preserving
   both unit-sum shape metrics and absolute transport fractions.
6. Compare the new Loess 0.5 g/L output with the stored CLARITAS_76 screen and
   any reproducible CLARITAS_18 output. If no common CLARITAS_18 output exists,
   document that limitation rather than relabelling another curve.
7. Profile only after correctness. Port the exact validated algorithm to GPU
   only if CPU runtime prevents uncertainty/replicate studies.
8. Select one independently constrained aggregate model before implementing
   flocs. Do not combine aggregate cross-sections with a second internal event
   model.

Every milestone writes machine-readable CSV evidence. Any failed statistical
validation blocks later claims of scientific validation.

