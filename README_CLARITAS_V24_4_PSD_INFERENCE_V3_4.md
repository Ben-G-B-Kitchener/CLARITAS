# CLARITAS V24.4 + PSD Inference V3.4

## Purpose
V24.4 is a narrow **TARDIIS collimator-geometry correction**. It does not change the successful V23.2 particle optics and does not return to KDE, widened detectors, or empirical scattering factors.

## Geometry taken from the supplied TARDIIS paper
The engineering drawing / machining caption gives:
- sample-cell OD: 100 mm; ID: 93 mm;
- sensor-ring ID: 101 mm -> inner radius 50.5 mm;
- sensor-ring OD: 131 mm -> outer radius 65.5 mm;
- inner through-bore: 4 mm diameter;
- counterbore/module receptacle: 8.7 mm diameter x 8 mm deep;
- therefore the narrow 4-mm bore length is 15 - 8 = 7 mm.

The section 5.3 prose says 3 mm through / 10 mm deep, conflicting with Fig. 3, Fig. 4 and the machining description. V24.4 deliberately defaults to the drawing/machining values (4 mm / 8 mm) and keeps them configurable.

## Selected V24.4 detector scorer
After a ray leaves the acrylic cell into air, a detector hit requires passage through:
1. the 4-mm circle at the inner ring face (r = 50.5 mm),
2. the 4-mm circle at the outer end of the narrow bore (r = 57.5 mm), and
3. the 8.7-mm counterbore opening at the ring outer face (r = 65.5 mm).

Each physical ray is assigned to at most one detector. The x-reflection symmetry average is retained. No KDE, angular blur, aperture widening, or bandwidth extrapolation is used. The paper does not give the photodiode active-area diameter, so V24.4 does **not** invent another detector aperture.

## Source geometry
The source launch reference is now a configurable `source_launch_radius_m`, defaulting to the sensor-ring outer radius (65.5 mm), and the ray must traverse the reconstructed 8.7-mm counterbore / 4-mm through-bore before reaching the cell. The supplied paper defines the ring and module dimensions but does **not** locate the LED die/luminous plane precisely within the installed module, so the 65.5-mm launch plane is explicitly a default geometric assumption rather than a paper-measured optical plane. The pre-existing narrow Gaussian source spot and beta angular proposal (`alpha1=1`, `alpha2=100`) are otherwise retained.

## Original CLARITAS +/-6.5 degree comparison
Original CLARITAS used `detector_acceptance_deg = 6.5` and binned by boundary exit-position angle. V24.4 computes that response from the same rays as a **diagnostic comparison channel**. It retains the old overlapping-window semantics and the old suppression of ballistic contributions to detector centres >=90 degrees. It is not used as `normalized_response`.

For the default launch reference, a centred point at 65.5 mm looking through a 2-mm-radius inner throat at 50.5 mm gives `atan(2/15) ~= 7.59 degrees`, so the old 6.5-degree value is mechanically plausible but is not substituted for the reconstructed scorer.

## Physics preserved
- geometric particle event cross-section `pi*r^2`;
- exponential free path;
- full 3-D spherical-particle Snell/Fresnel interaction and internal reflections;
- V24 water/acrylic/air cylindrical cell interfaces, Fresnel reflection, refraction and TIR;
- continuous-stirring interpretation;
- no Mie, empirical material scatter multiplier, PSD boost, KDE or detector broadening.

`verify_v24_4_geometry.py` statically checks that the particle Fresnel routine, acrylic-annulus routine and water transport loop are byte-for-byte unchanged from V24.3.

## First runs — do these before PSD inference
Cheap diagnostic, loess 0.5 g/L:
```bash
python CLARITAS_24_4_31-08-2026_TARDIIS_reconstructed_collimator.py --material loess --concentration 0.5 --n-rays 1000000 --output-dir claritas_v24_4_results/loess_0.5gL_1M
```

Regression against the previously improved dense case, kaolin 4 g/L:
```bash
python CLARITAS_24_4_31-08-2026_TARDIIS_reconstructed_collimator.py --material kaolin --concentration 4.0 --n-rays 5000000 --output-dir claritas_v24_4_results/kaolin_4gL_5M
```

Inspect `detector_response_normalized.csv`. Compare the **selected** `normalized_hardware_symmetry_exact` with the measured response, and also inspect `normalized_legacy_6p5_symmetry` to understand how the old CLARITAS acceptance approximation relates to the reconstructed hardware.

Only if those regressions are satisfactory should the six-condition V3.4 baseline be run:
```bash
python CLARITAS_PSD_Inference_V3_4.py --material all --baseline-only --config psd_inference_v3_4_config.json --measurements measured_detector_responses_v3_4.csv --output-dir psd_inference_v3_4_baseline
```

Do **not** start a full PSD/local-concentration fit merely because V3.4 is included. First establish that the corrected apparatus geometry improves or at least physically explains the baseline behaviour.
