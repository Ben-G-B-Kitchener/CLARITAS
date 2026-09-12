# V24.52 Physics Basis

V24.52 is a **production geometry/topology correction**, not a sediment-optics tuning release. The V24.50 exact million-ray qualification demonstrated that the previous apparatus geometry was internally inconsistent: water could cross the inner cylindrical boundary above the modelled acrylic wall top, producing acrylic states with no physical continuation. The correction is therefore justified by production transport evidence.

The established process-based optical and sediment physics are retained: wavelength-dependent particle optical constants, exact-rate Mie/hybrid particle-event transport, Smith-Beckmann dielectric microsurface physics, Fresnel/Snell interfaces, source angular model, detector acceptance/response, particle accessibility, sediment transport, and diagnostic decomposition are not fitted to sediment results in this release. No material-specific scattering multiplier, PSD boost, fitted detector gain, or H170 acceptance target is introduced.

## Canonical finite apparatus

V24.52 separates the mean water elevation from the physical acrylic tube top. The canonical sample cell is an open acrylic tube with a bonded acrylic bottom disc:

- `Rin = 0.0465 m` (93 mm internal diameter)
- `Rout = 0.0500 m` (100 mm external diameter)
- physical tube length `0.491 m`
- sensor plane `z=0`, 93 mm above the internal bottom
- `Zmin = -0.093 m` internal bottom
- `Zlow = -0.096 m` modelled outer bottom face
- `Zmean = +0.049 m` mean water/free-surface elevation for the 142 mm water depth
- `Zwall_top = +0.398 m` physical tube top

The 491/93/100 mm tube dimensions are taken from Kitchener et al., *HardwareX* 5 (2019) e00052. The configured 3 mm acrylic bottom-disc thickness and detector/source machining dimensions are retained from project apparatus configuration/provenance.

The local Scully free surface divides the bore into WATER below and HEADSPACE AIR above. The acrylic sidewall exists independently of that free surface from `Zmin` to `Zwall_top`. Above `Zwall_top` the bore is open to external air. The annular acrylic top face is optical; the open-bore top is air-to-air and therefore non-optical.

## Connected acrylic and RNG semantics

The acrylic solid is the connected union of the finite sidewall and bottom disc. The annular seam at `z=Zmin`, `Rin<=r<=Rout`, is internal acrylic/acrylic continuity: it creates no Fresnel event and consumes no optical RNG draw. Real acrylic interfaces retain their existing Fresnel/Snell random-draw ordering. The obsolete low reflection-count rejection is not a physical criterion; the acrylic solver uses progress/runaway safety with a high emergency cap.

## Validation philosophy

V24.52 has one finite complete-apparatus geometry gate. It checks canonical manifests, region ownership, finite surface support, source/detector consistency, deterministic topology, large random region classification, a 500,000-state CUDA/reference one-step population spanning water/headspace/acrylic, particle accessibility, all stored V24.50 production failure states, historical local replay states, CUDA scalar-value parity and memory-layout regressions. Old globally propagated float32-vs-float64 paths are diagnostic history rather than an oracle for the corrected apparatus.

Once this comprehensive gate passes, geometry auditing stops. The workflow proceeds to the exact million-ray production qualification and then the five-case integrity matrix.

## Sediment execution

Sediment diagnostic execution is deliberately separate from production authorization. The six canonical loess/kaolin concentration cases are run after the complete geometry gate so their actual detector outputs and transport diagnostics are visible even if a later production-authorization gate fails. Diagnostic execution never implies production authorization.
