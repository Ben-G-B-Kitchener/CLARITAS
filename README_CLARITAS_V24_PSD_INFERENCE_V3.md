# CLARITAS V24 + PSD Inference V3

## Purpose

This release moves the missing **TARDIIS apparatus optics** into the forward model rather than asking the PSD optimiser to mimic them.

### Fixed forward physics

- Mie-free particle encounter cross-section: `sigma = pi*r^2`.
- Exponential geometric free paths and particle-bin selection from `n_i*pi*r_i^2`.
- Full 3-D spherical-particle Snell + unpolarised Fresnel physics, including internal particle reflections.
- **Cylindrical water sample** with 93 mm bore.
- **Acrylic tube wall** to 100 mm OD.
- Water -> acrylic -> air and reverse Snell/Fresnel interfaces.
- Multiple acrylic-wall Fresnel reflections and total internal reflection.
- Physical TARDIIS source/detector collimators rather than spherical-cap detector acceptance.
- One physical detector maximum per ray.

The particle optical physics is unchanged in spirit from V23.2; the major change is the sample-cell/instrument geometry.

## Apparatus geometry used by default

The supplied TARDIIS design paper gives a 491 mm long extruded acrylic sample cell with 93 mm ID and 100 mm OD. The sensor ring has 36 positions at 10-degree intervals. Its manufacturing description gives a 101 mm ring bore, 4 mm inner collimator holes and 8.7 mm counterbores 8 mm deep.

V24 therefore defaults to:

- water radius: 46.5 mm
- acrylic outer radius: 50.0 mm
- ring inner/collimator inner plane radius: 50.5 mm
- collimator outer plane radius: 59.5 mm
- collimator diameter: 4.0 mm
- acrylic refractive index: 1.4906 (configurable PMMA default)

The published example water height (426 mm) and sensor-ring height above the base (102 mm) are included only as configurable axial defaults. If the original experimental runs used different values, edit `psd_inference_v3_config.json`.

## Continuous stirring

For the experimental series being reinterpreted here, the magnetic stirrer is treated as **continuously on**. V3 does **not** implement the paper's example post-stir settling experiment. The inferred effective PSD and effective local concentration are interpreted as ensemble properties of the continuously stirred optical sampling region.

## Why the apparatus optics are now explicit

The paper's water-only Device Geometry Baseline identifies responses at 10-170 degrees as internal instrument reflections and specifically notes that the back-angle response rises toward 170 degrees. V24 therefore models the acrylic interfaces directly rather than forcing the PSD optimiser to generate this feature.

## Files

- `claritas_tardiis_core_v24.py` — shared CUDA/host physics core used by both production CLARITAS and inference.
- `CLARITAS_24_31-08-2026_TARDIIS_cell_optics.py` — production forward runner.
- `claritas_forward_inference_v3.py` — lightweight direct forward-evaluation CLI/re-export.
- `CLARITAS_PSD_Inference_V3.py` — 7-knot non-monotonic effective-PSD + local-concentration inference.
- `psd_inference_v3_config.json` — apparatus + optimiser settings.
- `measured_detector_responses_v3.csv` — the six measured response curves already supplied.

## Recommended validation sequence

### 1. Water-only Device Geometry Baseline

Run this before sediment fitting:

```bash
python CLARITAS_24_31-08-2026_TARDIIS_cell_optics.py --material loess --dgb-only --n-rays 500000
```

Inspect:

- `claritas_v24_results/DGB_water_only/detector_response_normalized.csv`
- `.../detector_response_normalized.png`
- `.../diagnostics.json`

The important question is whether the explicit acrylic/collimator optics now generate the experimentally known high-angle baseline rise, especially 150-170 degrees.

### 2. V3 unchanged-PSD baseline

```bash
python CLARITAS_PSD_Inference_V3.py --material all --baseline-only --quick
```

Outputs go to `psd_inference_v3_results/`, so V1/V2 results are not overwritten.

### 3. Warm-start V3 from the V2 fit

The PSD parameterisation is deliberately kept compatible with V2 (7 knots per concentration + 3 concentration factors, 24 parameters). For example:

```bash
python CLARITAS_PSD_Inference_V3.py --material loess --quick \
  --warm-start-v2 psd_inference_v2_results/loess/best_fit_parameters.json --skip-local

python CLARITAS_PSD_Inference_V3.py --material kaolin --quick \
  --warm-start-v2 psd_inference_v2_results/kaolin/best_fit_parameters.json --skip-local
```

Then use full runs after checking the quick fits.

## Important modelling boundaries

- The cylindrical **side wall** is modelled explicitly. The top free surface and acrylic bottom disc are currently treated as axial losses, because the detector ring samples the radial plane and the exact axial geometry of the user's specific continuous-stirring runs has not yet been confirmed.
- The nylon sensor ring is treated as opaque except through the physical collimator apertures.
- The detector photodiode active-area details beyond the published collimator geometry are not separately modelled yet; passing both collimator planes constitutes a detector hit.
- No empirical detector/backscatter multiplier is fitted.
- No Mie term, floc model, roughness multiplier or material-specific optical boost is introduced.
