# CLARITAS V24.52.0 — corrected curved-free-surface validator + full campaign

V24.52 inherits the complete finite apparatus geometry from V24.51 and freezes production CUDA. It corrects the independent one-step oracle for horizontal WATER/HEADSPACE rays: `vz=0` does not prevent a ray from crossing a radially varying vortex free surface because radius changes along the XY path.

The exact V24.51 deterministic 500k population is reconstructed on the host: 3,225 WATER-horizontal plus 1,824 HEADSPACE-horizontal oracle errors account for all 5,049 reported mismatches, with zero unexplained remainder. The campaign then proceeds through the complete GPU geometry gate, exact million-ray production qualification, five-case matrix, and six-case sediment diagnostic reporting when run on CUDA.

## Canonical execution

```bash
cd "/f/Google Drive Sync/CLARITAS_code/CLARITAS/CLARITAS_V24_52"
conda activate claritas
rm -rf ~/.cupy/kernel_cache
rm -rf ~/.cupy/jitify_cache
rm -rf results
rm -rf __pycache__
python preflight_v24_52_package.py && \
python verify_v24_52_complete_geometry.py --output-dir results && \
python run_v24_52_campaign.py --output-dir results
```

The campaign performs boundary/microsurface GPU validation, the complete 500k geometry gate, exact 1,000,000-ray production qualification, the five-case integrity matrix when qualified, and then the six sediment diagnostic cases. Sediment diagnostics are deliberately separate from production authorization, so results can still be inspected if the million-ray or integrity gate fails after geometry has passed.

## Sediment outputs

The six current canonical cases are loess and kaolin at 0.5, 2.0 and 4.0 g/L. H170 is the 170° detector channel in every case.

After a GPU run see:

- `results/sediment_case_inventory.csv`
- `results/sediment_case_results.csv`
- `results/sediment_case_results.json`
- `results/V24_52_SEDIMENT_TEST_REPORT.md`
- `results/sediment_plot_data.csv`
- `results/sediment_H170_hardware_score_by_case.png` (when matplotlib is available)
- `results/sediment_trusted_RMSE_by_case.png` (when matplotlib is available)

Each case directory also contains the full detector response/event-history tables and production-failure diagnostics.

## Qualification policy

`sediment_diagnostic_campaign_run=true` does **not** imply `sediment_campaign_authorized=true`. Production authorization requires the complete geometry gate, exact million-ray qualification and the complete five-case integrity matrix to pass with the defined fail-closed criteria.

See `V24_52_VALIDATOR_AND_CAMPAIGN_ENGINEERING_NOTE.md` for this release correction, and `V24_52_COMPLETE_GEOMETRY_ENGINEERING_NOTE.md`, `v24_52_geometry_manifest.json` and `v24_52_region_surface_manifest.json` for the inherited complete apparatus geometry.
