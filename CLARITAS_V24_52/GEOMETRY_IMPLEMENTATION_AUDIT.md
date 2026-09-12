# V24.52 Complete Geometry Implementation Audit

## Canonical geometry and provenance

V24.52 uses `v24_52_complete_geometry.py`, `v24_52_geometry_manifest.json` and `v24_52_region_surface_manifest.json` as the independent host description of the complete finite apparatus. Production CUDA does not call the host oracle; it receives the same physical values through the normal model configuration/ABI. The physical tube length, inner diameter and outer diameter are sourced to Kitchener et al., *HardwareX* 5 (2019) e00052; experiment-specific sensor/water geometry remains explicit in configuration.

## Production paths audited

The production trace now has one coherent surface model across all participating media:

- **water**: local free surface, finite inner acrylic cylinder, finite bottom aperture;
- **headspace**: local free surface from above, finite inner acrylic cylinder, open bore top;
- **connected acrylic**: finite inner cylinder, finite outer cylinder, external bottom face, annular top face, bottom water aperture; the sidewall/bottom seam is internal continuity;
- **external/source path**: initial source-to-vessel acquisition uses the same finite outer-cylinder support;
- **detector path**: detector scoring remains outside the vessel and retains the validated finite ring/bore geometry; apparatus dimensions are checked against the vessel radius and sensor plane;
- **particles**: centre accessibility retains finite sphere clearance from sidewall, bottom and the spatially varying free surface.

The former infinite-cylinder helpers remain only in the historical legacy core under `v24_52_legacy_core.py`; active V24.52 production sample-cell wall acquisition uses `cell_cylinder_hit_finite_forward_d` / connected-solid finite-surface logic. No runtime selector chooses the legacy core for the canonical V24.52 model.

## Defect family corrected together

The audit treats the V24.50 7,734 failures as evidence of a topology inconsistency, not as a special-case condition. The complete correction separates `Zmean` from `Zwall_top`, extends the acrylic wall to the actual tube top, adds headspace transport, gives inner and outer walls identical finite support in each medium, and adds the open-bore / annular-top distinction. All 512 stored V24.50 acrylic failure samples are retained as regression states rather than implementation conditions.

## Exact gates retained

Raw CUDA state arrays are explicitly C-contiguous. Independent reference calculations use the exact scalar value domain sent through the CUDA ABI and then compute in higher precision. Particle accessibility uses the finite spherical footprint and numerical-tolerance parity. The case-1650 input-parity regression remains. No established tolerance is relaxed to obtain a pass.

## Whole-apparatus gates

The GPU geometry suite includes deterministic topology, random region ownership throughout/around the apparatus, 500,000 one-step boundary states across sidewall/bottom/headspace/free-surface/top-rim regions, 100,000 particle accessibility states, all available V24.50 failure samples, historical residual local states, finite-surface/RNG invariants and source/detector consistency checks. After this gate passes there is no further geometry gate before production qualification.
