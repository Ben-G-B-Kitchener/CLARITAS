# V24.36.9 roughness provenance audit

## Parameter

The frozen finite-particle model uses:

```text
particle_surface_rms_slope_deg = 25.0
alpha = tan(25 deg) ~= 0.4663
```

## Evidence found in the available CLARITAS lineage

1. `claritas_tardiis_core_v24_36.py` contains the current default `particle_surface_rms_slope_deg=25.0` and comments identifying the local rough-optical-normal architecture as a V24.8 morphology diagnostic.
2. `references/v24_24_1_archived_case_summary.csv` records the 25-degree value in archived V24.24.1 material/concentration cases.
3. `references/v24_29_archived_case_summary.csv` shows that the validated V24.29 event-rate baseline retained the same 25-degree roughness value.

These observations establish persistence in the model lineage. They do not establish the physical origin of the numerical value.

## Evidence not located

Within the available packages, comments, README/config files and bundled references, no source was located tying the 25-degree value to:

- microscopy of the actual kaolin or loess;
- profilometry;
- a sediment-specific surface-slope measurement;
- a literature value explicitly cited for these particles;
- another independent morphology measurement.

No evidence was located proving that the value was selected from H170 either. Absence of physical provenance must not be reinterpreted as evidence of detector-data tuning.

## Classification

Until independent provenance is supplied, V24.36.9 classifies the 25-degree slope as:

> **an inherited, physically motivated but unvalidated morphology parameter**

The release does not alter this value.
