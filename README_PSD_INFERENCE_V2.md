# CLARITAS PSD Inference V2

This toolkit infers the **effective suspension state seen by the optical volume** while leaving the CLARITAS V23.2 optical physics unchanged.

## What changed from V1

V1 used only five diameter controls and forced all concentrations to share one
`A(d) + B(d) log(C/Cref)` modification. In practice this strongly favoured broad,
often monotonic fine-to-coarse trends.

V2 is deliberately more comprehensive:

- **7 PSD control knots per measured concentration**.
- One independent smooth PSD modification for 0.5, 2 and 4 g/L.
- Shape-preserving cubic **PCHIP interpolation in log particle diameter**.
- Non-monotonic, peaked, U-shaped and band-selective PSD changes are allowed.
- The three PSDs are **cross-regularised**, not forced to share a functional form.
- One bounded **effective local concentration factor** is inferred for each nominal concentration.
- Detector objective includes ordinary residuals, log-response residuals, angular-gradient residuals, and an explicit high-angle shape term using 140/150/160/170° channels.
- Best parameters are checkpointed during the joint optimisation.

## What is NOT fitted

The forward optics remain the V23.2 model:

- 3-D ray transport;
- spherical sample;
- geometric particle cross section `pi*r^2`;
- exponential geometric free paths;
- particle selection from `n_i*pi*r_i^2`;
- 3-D Snell refraction;
- unpolarised Fresnel reflection/transmission;
- internal Fresnel reflections;
- source and detector geometry.

There is still **no Mie model, floc model, roughness multiplier or empirical optical
backscatter factor** in the inference kernel.

## Files

- `CLARITAS_PSD_Inference_V2.py` — inverse model / optimiser.
- `claritas_forward_inference_v2.py` — streamlined V23.2 forward model.
- `psd_inference_v2_config.json` — inference settings and regularisation.
- `measured_detector_responses.csv` — six measured detector curves supplied for the current analysis.
- `psd_inference_v2_requirements.txt` — Python dependencies.

Unzip these files into the main CLARITAS repository directory.

## First run: baseline validation

```bash
conda activate claritas
python CLARITAS_PSD_Inference_V2.py --material all --baseline-only --quick
```

This uses the source PSDs and nominal concentrations. V2 uses a richer objective than
V1, so the numerical V2 objective should **not** be compared directly with the V1
objective value. Compare detector curves/RMSE instead.

## Recommended exploratory V2 fits

Because you already have useful V1 fits, use them as warm starts if the corresponding
V1 `effective_psds.csv` files are available:

```bash
python CLARITAS_PSD_Inference_V2.py --material loess --quick \
  --warm-start-v1 psd_inference_results/loess/effective_psds.csv

python CLARITAS_PSD_Inference_V2.py --material kaolin --quick \
  --warm-start-v1 psd_inference_results/kaolin/effective_psds.csv
```

If your files have different names, simply pass their paths. A warm start affects only
the starting PSD shape; the V2 local concentration factors begin at 1.0 and are free
to move during optimisation.

You can also start from the unmodified source PSDs:

```bash
python CLARITAS_PSD_Inference_V2.py --material loess --quick
python CLARITAS_PSD_Inference_V2.py --material kaolin --quick
```

## Full fits

```bash
python CLARITAS_PSD_Inference_V2.py --material loess \
  --warm-start-v1 psd_inference_results/loess/effective_psds.csv

python CLARITAS_PSD_Inference_V2.py --material kaolin \
  --warm-start-v1 psd_inference_results/kaolin/effective_psds.csv
```

The full calculation uses 100,000 rays per coarse forward evaluation and 1,000,000
rays for the final best-fit validation by default.

## Optimisation stages

V2 uses two stages:

1. **Concentration-specific seed fitting.** Each condition gets seven PSD controls plus
   one local-concentration factor. This efficiently finds different fine/coarse/banded
   solutions without yet forcing the concentrations to resemble one another.
2. **Joint refinement.** All conditions are then fitted together with cross-concentration
   regularisation. This discourages arbitrary unrelated PSDs while still allowing genuine
   non-monotonic concentration evolution.

Use `--skip-local` if you deliberately want to bypass stage 1.

## Effective local concentration

For each condition V2 infers

`C_eff = concentration_scale * C_nominal`

with the default bound

`0.25 <= concentration_scale <= 2.0`.

This is intended to represent the local solids loading in the optical sampling volume of
a continuously stirred suspension. It is **not** an optical multiplier. The geometric
particle number density is rebuilt from `C_eff` before ray tracing.

## Regularisation

The objective contains physically motivated penalties for:

- excessive departure from the measured source PSD;
- excessive curvature of the PSD modifier versus log diameter;
- unnecessary differences between neighbouring concentration PSDs;
- large departures of local concentration from nominal;
- excessive curvature of local concentration factor versus concentration.

The regularisation strengths live in `psd_inference_v2_config.json`. V2 deliberately
uses a lower size-smoothness penalty than V1 so that local enrichment/depletion bands
can be discovered.

## High-angle detector shape

The measured data contain a sharp response near 170°. A simple least-squares objective
can reproduce the integrated high-angle power with an overly broad 140–170° hump.
V2 therefore also compares adjacent **log-response slopes** at 140, 150, 160 and 170°.
This asks the optimiser to reproduce the *shape* of the backscatter feature rather than
merely its total amplitude.

## Output directory

Default:

```text
psd_inference_v2_results/
  kaolin/
  loess/
```

Important files for each material:

- `effective_psds.csv` — source and inferred mass-fraction PSD by concentration.
- `psd_control_points.csv` — the seven inferred spline controls.
- `effective_concentrations.csv` — nominal vs inferred local g/L.
- `detector_fit.csv` — measured/modelled detector values and residuals.
- `fit_summary.csv` — RMSE, MAE, correlation and transport diagnostics.
- `objective_by_condition.csv` — linear/log/gradient/high-angle objective pieces.
- `best_fit_parameters.json` — complete V2 solution for reproducibility/resume.
- `checkpoint_best_parameters.json` — best coarse point seen during the joint search.
- `psd_ratio.png` — effective/source PSD ratio versus diameter.
- `effective_psds.png` — absolute source/effective PSD curves.
- `detector_fit.png` — detector overlays.
- `detector_residuals.png` — residual by detector angle.
- `effective_concentration.png` — inferred local vs nominal concentration.

## Resume/refine

A completed or interrupted V2 fit can be restarted from a V2 parameter JSON:

```bash
python CLARITAS_PSD_Inference_V2.py --material loess \
  --initial-parameters psd_inference_v2_results/loess/checkpoint_best_parameters.json \
  --skip-local
```

The parameter file is material-specific.

## Interpretation warning

V2 has more freedom than V1, so a better detector fit alone does not prove that the
inferred PSD is physically correct. The important test is whether the inferred
concentration-dependent PSD and local concentration trends are plausible for continuous
magnetic stirring, settling, recirculation and particle residence in the optical volume.
