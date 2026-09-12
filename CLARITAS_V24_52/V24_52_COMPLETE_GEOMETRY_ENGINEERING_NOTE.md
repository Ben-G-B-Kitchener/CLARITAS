# CLARITAS V24.52.0 — Complete Apparatus Geometry Engineering Note

## Purpose

V24.52 is intentionally a whole-apparatus geometry release rather than another local-failure patch. V24.47–V24.50 progressively removed validator defects and corrected the connected acrylic bottom topology; the exact V24.50 1,000,000-ray production run then exposed 7,734 acrylic no-intersection failures. V24.52 reconstructs the finite sample cell, water, headspace and acrylic as one coherent physical model and uses that model consistently in production CUDA and independent host validation.

No empirical sediment-scattering multiplier, PSD boost factor, fitted detector gain, or result-tuning factor is introduced.

## Authoritative sample-cell dimensions

The physical sample cell is based on Kitchener et al., *HardwareX* 5 (2019) e00052, Section 5.1. The assembled sample cell is described as an extruded acrylic tube **491 mm long, 93 mm internal diameter and 100 mm external diameter**, closed at one end by a glued acrylic disc.

Reference: https://eprints.whiterose.ac.uk/id/eprint/141481/1/1-s2.0-S2468067218300762-main.pdf

CLARITAS already defines the sensor plane as 93 mm above the internal bottom for the 622 nm experiment and the mean water depth as 142 mm. Therefore the canonical V24.52 axial coordinates relative to the sensor plane are:

- inner water/acrylic bottom: `Zmin = -0.093 m`
- modelled outer bottom face: `Zlow = -0.096 m`
- mean water/free-surface elevation: `Zmean = +0.049 m`
- physical tube top: `Zwall_top = 0.491 - 0.093 = +0.398 m`

`Zmean` and `Zwall_top` are now separate physical parameters. The localized Scully free surface remains a water-surface model; it no longer truncates the acrylic tube.

The 3 mm bottom-disc thickness is an existing CLARITAS apparatus-model value documented in the project configuration. The open upper tube topology is modelled explicitly: headspace air occupies the bore above the local free surface, the acrylic annular top face exists at `Zwall_top`, and the bore is open to external air above that elevation.

## Complete region model

V24.52 explicitly distinguishes:

1. **WATER** — `r < Rin`, from `Zmin` to the local free surface.
2. **HEADSPACE AIR** — `r < Rin`, from the local free surface to `Zwall_top`.
3. **ACRYLIC SIDEWALL** — `Rin < r < Rout`, `Zmin < z < Zwall_top`.
4. **ACRYLIC BOTTOM** — `r < Rout`, `Zlow < z < Zmin`.
5. **EXTERNAL AIR** — outside the finite vessel, including above the open tube.

Interface ambiguity is treated as a numerical boundary condition rather than a sixth material.

## Complete surface model

The production and independent reference geometry use the same finite supports:

- water/headspace free surface: `z = Zfree(x,y)`, `r < Rin`
- inner acrylic cylinder: `r = Rin`, `Zmin <= z <= Zwall_top`
- outer acrylic cylinder: `r = Rout`, `Zlow <= z <= Zwall_top`
- bottom water aperture: `z = Zmin`, `r < Rin`
- internal bottom/sidewall acrylic seam: `z = Zmin`, `Rin <= r <= Rout` — **not optical; no Fresnel event and no RNG draw**
- outer bottom face: `z = Zlow`, `r <= Rout`
- annular acrylic top face: `z = Zwall_top`, `Rin <= r <= Rout`
- open bore top: `z = Zwall_top`, `r < Rin` — air-to-air geometric exit, no Fresnel event

At the inner cylinder, the medium inside the bore is selected from the local free surface: water below it and headspace air above it.

## Production geometry defects identified and corrected

The complete audit identified the following coherent defect family rather than treating the V24.50 failure signature as an isolated special case:

1. **Mean free-surface height was also used as the physical acrylic wall top.** This truncated the tube at `+0.049 m` even though the physical wall continues to approximately `+0.398 m`.
2. **Water-side wall selection treated `r=Rin` as an effectively infinite cylinder**, while the connected acrylic solver treated that surface as finite to the old wall top. This allowed water rays to enter a state that the acrylic solver then considered outside its solid.
3. **The previous production path treated transmission through the water free surface as an immediate top exit.** With the real finite tube height, this omits the physical air headspace and its possible water re-entry / acrylic-wall interactions.
4. **Outer/source wall acquisition did not share the same explicit finite axial support as the connected vessel geometry.** V24.52 uses the finite physical wall support.

Corrections are model-level, not ray-ID special cases:

- added explicit `tube_length_m` / `Zwall_top` production geometry;
- kept `Zmean` for the water/free-surface model only;
- extended connected acrylic sidewall and outer wall to the physical tube top;
- added an explicit headspace transport branch;
- made inner and outer sample-cell cylinder intersections use finite support consistently;
- retained the bonded sidewall/bottom connected-solid seam with zero Fresnel/RNG at acrylic/acrylic continuity;
- added a distinct headspace geometry failure code so future failures are visible rather than conflated.

The particle-scattering, Mie, Smith-Beckmann, PSD, source-angular and detector-response physics are not tuned by these changes.

## V24.50 failure-population regression

The supplied V24.50 production run recorded 7,734 failures in 1,000,000 rays. All 512 stored failure samples are `ACRYLIC_ANNULUS_NO_INTERSECTION`; their stored radii lie approximately `Rin + 0.2 µm` and their z positions span approximately `+0.049044 m` to `+0.060088 m` — exactly the interval above the old mean-level wall truncation and below the localized-vortex wall surface.

V24.52 host reconstruction replays **all 512 stored states** against the complete finite acrylic vessel and finds a valid forward physical acrylic boundary for **512/512**. This is a host regression only; the real production elimination is still required from the V24.52 GPU qualification.

## Validation architecture

V24.52 uses one finite geometry gate containing:

- canonical geometry manifest and provenance;
- explicit region/surface manifest;
- deterministic whole-apparatus topology cases;
- large random whole-volume region classification;
- 500,000-state complete-apparatus one-step CUDA/reference boundary comparison;
- 100,000-state finite-particle accessibility comparison;
- all available V24.50 failure samples;
- historical long-path residual pre-states as local one-step replay evidence only;
- memory-layout and CUDA scalar-input parity regressions;
- the case-1650 input-parity regression;
- finite support / headspace / acrylic-continuity invariants.

Old global trajectories constructed with the erroneous `Zmean == wall top` topology are not treated as an oracle for the corrected physical vessel. Their stored local states remain useful regression inputs.

Once this complete gate passes, geometry auditing stops and the runner proceeds to the exact 1,000,000-ray production qualification.

## Sediment execution and reporting

V24.52 inventories the six current canonical sediment cases:

- loess: 0.5, 2.0 and 4.0 g/L
- kaolin: 0.5, 2.0 and 4.0 g/L

H170 is the 170° detector channel in each case, not a seventh material/concentration case.

The top-level campaign distinguishes:

- **sediment diagnostic execution** — may run after the complete geometry gate even if later production authorization fails;
- **sediment production authorization** — remains fail-closed and requires the geometry gate, exact million-ray qualification and all five cross-case integrity cases to pass.

Each diagnostic case records detector outputs, H170 diagnostics, failure counts, seed, ray count, elapsed time and its authorization watermark. The campaign writes `sediment_case_results.csv`, `sediment_case_results.json`, `V24_52_SEDIMENT_TEST_REPORT.md`, plot data and diagnostic comparison plots when matplotlib is available.

## Build-environment qualification status

The release-build environment used here does not provide a compatible CuPy/CUDA runtime. Therefore real GPU geometry validation, the exact 1,000,000-ray V24.52 qualification, the five-case integrity matrix, and the six sediment simulations are **NOT RUN in the build environment**. No GPU result is fabricated.

Host/static validation does run, including C++ CUDA syntax/scope checking, ABI parity, complete-geometry invariants, case-1650 closure, manifest checks and the 512-state V24.50 failure host regression. The delivered campaign runner performs the full GPU + sediment sequence automatically on the user's CUDA machine.
