# CLARITAS Effective-PSD Inference Toolkit

This toolkit estimates the **effective particle-size distribution (PSD) seen by the optical measurement** while keeping the CLARITAS V23.2 optical physics fixed.

It is intended to test the hypothesis that continuous magnetic stirring plus size-dependent suspension/settling changes the PSD present in the illuminated volume, rather than requiring further changes to the optical model.

## Files

- `claritas_forward_inference.py` — streamlined V23.2 forward model for repeated optimisation calls. It retains 3D geometric particle encounters, Snell refraction and Fresnel reflection, but omits heatmaps/HDF5/plotting during each trial.
- `CLARITAS_PSD_Inference.py` — constrained inverse model / optimiser.
- `measured_detector_responses.csv` — the six measured detector curves supplied on 30 Aug 2026, plus the corresponding V23.2 baseline model curves for reference.
- `psd_inference_config.json` — control-point count, ray counts, regularisation and optimiser settings.
- `psd_inference_requirements.txt` — Python package requirements for the toolkit.

The normal `CLARITAS_23_2...py` release is not modified by this toolkit.

## What is allowed to change

For each material the source PSD is multiplied by a smooth size-dependent suspension/enrichment function and then renormalised to unit mass fraction.

The default model is

```text
log S(d,C) = A(d) + B(d) log(C/Cref)
```

where `A(d)` and `B(d)` are represented by five control points spaced uniformly in log particle diameter and linearly interpolated between them.

All three concentrations for one material are therefore fitted **together**. They cannot acquire three arbitrary unrelated PSDs.

## What is NOT fitted

The inference code deliberately keeps these fixed:

- nominal concentration in g/L;
- total PSD mass fraction (renormalised to 1);
- geometric cross-section `pi*r^2`;
- particle and medium refractive indices;
- 3D Snell refraction;
- Fresnel coefficients and internal reflections;
- sample radius and geometry;
- beam divergence;
- detector positions and acceptance angle.

There are no Mie terms, empirical scattering multipliers, material-specific optical boosts, floc terms, roughness terms or arbitrary reflection probabilities.

## Installation

Use the existing `claritas` Conda environment. The only package likely to be new is SciPy:

```bash
conda activate claritas
conda install -c conda-forge scipy -y
```

The inference forward model also requires the same working CuPy/CUDA setup as V23.2.

## 1. First run: baseline validation

Before fitting anything, verify that the streamlined forward model gives the same *statistical* detector behaviour as V23.2 when supplied with the unmodified source PSD:

```bash
python CLARITAS_PSD_Inference.py --material all --baseline-only --quick
```

For a higher-statistics check:

```bash
python CLARITAS_PSD_Inference.py --material all --baseline-only
```

Results appear under:

```text
psd_inference_results/kaolin/
psd_inference_results/loess/
```

The streamlined kernel uses fixed common random numbers and does not reproduce the exact random sequence used in the earlier V23.2 runs, so individual counts will not be bit-for-bit identical. The curves should agree statistically.

## 2. Quick optimiser smoke test

Run one material with reduced ray count and iterations:

```bash
python CLARITAS_PSD_Inference.py --material kaolin --quick
```

or

```bash
python CLARITAS_PSD_Inference.py --material loess --quick
```

This is primarily a functional check of CUDA compilation, optimiser/forward-model communication and output generation.

## 3. Normal inference

Kaolin:

```bash
python CLARITAS_PSD_Inference.py --material kaolin
```

Loess:

```bash
python CLARITAS_PSD_Inference.py --material loess
```

Both sequentially:

```bash
python CLARITAS_PSD_Inference.py --material all
```

## Output files

For each material the optimiser produces:

- `effective_psds.csv` — source and inferred PSD bin weights at 0.5, 2 and 4 g/L.
- `detector_fit.csv` — measured response, best-fit model response and residual at every detector angle.
- `fit_summary.csv` — RMSE, MAE, Pearson correlation, `mu_geom`, diameter optical depth, interaction statistics and Fresnel diagnostics.
- `best_fit_parameters.json` — fitted low-dimensional PSD parameters and optimiser status.
- `evaluation_history.csv` — objective value during optimisation.
- `psd_ratio.png` — inferred/source PSD ratio versus particle diameter.
- `effective_psds.png` — source and effective PSDs.
- `detector_fit.png` — measured and inferred-PSD CLARITAS detector curves.

## Interpreting the PSD ratio

`effective_to_source_ratio` is the most useful diagnostic column.

For example:

```text
ratio > 1   -> size bin enriched in the optical sampling volume
ratio = 1   -> unchanged relative representation
ratio < 1   -> size bin depleted in the optical sampling volume
```

Because the PSD is renormalised, these are **relative enrichments/depletions**, not independent changes in total mass concentration.

## Objective function

The default objective combines:

1. ordinary squared error between normalised detector responses;
2. a weak log-response error so low-amplitude mid-angle channels are not ignored;
3. weak regularisation towards the original PSD;
4. smoothness regularisation across particle diameter;
5. weak regularisation against unnecessarily strong concentration dependence.

All weights can be changed in `psd_inference_config.json`.

## Common random numbers

Each candidate PSD is evaluated using the same source directions and deterministic CUDA random streams for a given concentration. This is deliberate: differences in objective value should primarily reflect the candidate PSD rather than Monte Carlo noise.

## Recommended scientific workflow

1. Run `--baseline-only` and verify agreement with the normal V23.2 output.
2. Run `--quick` for each material.
3. Run the normal inference separately for kaolin and loess.
4. Inspect `psd_ratio.png` before interpreting the detector fit.
5. Ask whether the inferred size-dependent enrichment/depletion is physically plausible under continuous magnetic stirring.
6. Only then consider a hydrodynamic suspension model capable of predicting the inferred PSD changes independently.

The inverse fit should be treated as a hypothesis generator. A good detector fit alone does not prove that the inferred PSD is the unique physical PSD.
