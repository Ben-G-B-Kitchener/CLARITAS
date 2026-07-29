# CLARITAS_76 transport-diagnostics audit

This audit is limited to run-time transport diagnostics and the per-ray HDF5
schema. It does not assess the floc optical model.

## Existing HDF5 record

CLARITAS_76 writes these one-dimensional datasets:

`exit_x`, `exit_y`, `exit_dir`, `exit_rpl`, `scatter_count`,
`floc_event_count`, `last_event_was_floc`, `last_scatter_bin`,
`extinction_count`, `absorbed`, `floc_domain_dx`, `floc_domain_dy`,
`floc_domain_path`, and `floc_internal_scatter_count`.

This is sufficient to reconstruct most requested count diagnostics after a
run, but it has four material limitations:

1. `exit_rpl` is written only for escaped rays. Absorbed-ray path lengths are
   left as NaN, so an all-ray path-length histogram cannot be recovered.
2. Reaching `MAX_EXTINCTIONS` sets `absorbed = 1`. A numerical truncation is
   therefore indistinguishable from physical absorption.
3. `floc_event_count` is incremented only after the albedo test succeeds. It
   counts successful outer floc scattering events, not all floc extinction
   events. The population responsible for a terminal absorption is not stored.
4. Transport is two-dimensional and only `exit_dir = atan2(vy, vx)` is stored.
   A three-dimensional exit-direction or exit-position distribution cannot be
   reconstructed.

The HDF5 file also has no configuration/provenance attributes and no
per-ray detector assignment.

## Requested diagnostics versus CLARITAS_76

| Requested diagnostic | CLARITAS_76 status | Exact issue |
|---|---|---|
| Mean scatter count | Partial/incorrect denominator | Printed for valid exits only. Separate detector overview values exist, but there is no canonical all-completed-ray value. Internal floc scatters are stored separately and are not included in the printed mean. |
| Scatter-count histogram | Partial | `scatter_count_histogram_*` includes valid exits only. Absorbed rays, truncated rays, status counts, and explicit denominators are absent. |
| Path-length histogram | Partial/incorrect scope | Uses finite `exit_rpl`, which means escaped rays only. It is written to `./`, not `OUTDIR`, and is generated after the wavelength loop from the final wavelength's arrays. |
| Absorption probability versus scatter count | Missing | It can be derived for physical absorption from `absorbed` and `scatter_count`, but CLARITAS_76 never exports it. Cap termination is incorrectly included in absorption. |
| Detector contribution versus scatter order | Partial | Exports only `0`, `1`, `2–5`, and `>5` classes, not exact scatter order. Per-centre acceptance masks can assign one ray to two overlapping detectors. |
| Floc-event fraction | Partial/ambiguous | The static extinction-weight probability is printed/exported. Realised per-ray floc counts exist, but no global realised fraction is exported. The recorded count excludes floc absorption events. |
| Primary-event fraction | Partial/ambiguous | Static extinction-weight probability is printed. A realised successful-primary-scatter fraction can be derived as `(scatter_count - floc_event_count) / scatter_count`, but is not exported. |
| Extinction statistics | Partial | Static per-bin extinction budgets and some per-detector means exist. There is no run-level distribution or statistics split by escaped, absorbed, and truncated outcomes. |
| Scattering statistics | Partial | Exit-only mean/max and exit-only histogram exist. There is no all-ray/fate-split table. Individual event scattering angles are not recorded, so event-angle statistics cannot be reconstructed. |
| Exit-angle distribution | Incorrect/partial | The bulk histogram feeds signed `exit_dir` values in `[-180, 180]` into bins covering `[0, 180]`, dropping negative angles. Its angular reference also differs from the detector polar convention. |
| Exit-position distribution | Partial | Raw x/y points and detector-binned position angles are saved, but no general position histogram is exported and no z coordinate exists. |
| Ballistic fraction | Ambiguous | Available as a fraction of valid exits in one CSV. No clearly named all-launched-ray ballistic escape fraction is provided. |
| Single-scattered fraction | Ambiguous | Available as a fraction of valid exits/scatter-class detector hits, but not as a canonical fraction of launched rays. |
| Multiply-scattered fraction | Incorrect definition | One diagnostic defines multiply scattered as `scatter_count >= 6`. The conventional requested category is `scatter_count >= 2`; the code separately labels 2–5 as “low multiple”. |
| Absorbed fraction | Incorrectly conflated | Several summaries use `total - valid_exit` and label it “absorbed_or_invalid”. The exact `absorbed` flag exists, but cap termination is also marked absorbed and no separate unclassified/truncated fraction is reported. |

## Detector-efficiency accounting issue

CLARITAS_76 reports per-detector hit counts but not a unique detected-ray
count. Detector centres are 10 degrees apart and the acceptance half-width is
6.5 degrees, so neighbouring acceptance windows overlap. Summing detector
counts is therefore an assignment count and can exceed the number of uniquely
detected rays. Absolute detector efficiency needs either:

- one deterministic detector index per escaped ray (nearest accepted centre),
  or
- separate `unique_detected_ray_count` and `detector_hit_assignment_count`
  fields if overlap is intentional.

The production helper uses a single `detector_index` per ray and reports both
names explicitly.

## Event-count semantics

CLARITAS_76's recorded relationships are:

```text
extinction_count = scatter_count                 for escaped rays
extinction_count = scatter_count + 1             for absorbed rays
floc_event_count <= scatter_count
primary successful scatters = scatter_count - floc_event_count
```

`floc_event_count / scatter_count` is consequently a valid realised fraction
of **successful outer scattering events**. It is not a realised fraction of all
extinction events. Adding a kernel-side `floc_extinction_count` before the
albedo decision is required to measure the latter exactly. CLARITAS_77 now
supplies that optional counter to the production diagnostics. The canonical
`floc_event_fraction` and `primary_event_fraction` use all outer extinction
events when it is present; separately named successful-scatter fractions are
retained.

Similarly, `scatter_count` counts an outer floc encounter once while
`floc_internal_scatter_count` records internal monomer scatters separately.
These should remain separate diagnostics until the floc architecture defines
which events constitute the physical scatter order. Silently summing them
would change the meaning of detector scatter order.

## Production helper interface

The new `claritas_production_diagnostics.py` module exposes:

```python
save_comprehensive_transport_diagnostics(
    *,
    wl_nm,
    outdir,
    exit_x, exit_y, exit_z,
    exit_vx, exit_vy, exit_vz,
    path_length,
    scatter_count,
    floc_event_count,
    floc_extinction_count=None,
    extinction_count,
    absorbed,
    truncated,
    detector_index,
    detector_angles_deg,
    floc_internal_scatter_count=None,
)
```

It writes a canonical `run_summary_<wavelength>nm.csv` plus count, path,
absorption, detector-order, extinction, direction, position, and integrity CSV
files. Fractions have explicit denominators. Ballistic, single-scattered, and
multiply-scattered use escaped rays with scatter counts 0, 1, and at least 2,
respectively, divided by all launched rays. Separate within-escaped fractions
are also saved.

The companion
`save_comprehensive_transport_diagnostics_from_hdf5(...)` wrapper accepts
CLARITAS_76 files. It reports unavailable truncation and absorbed-path
coverage rather than treating either as measured. Because CLARITAS_76 has no
floc-extinction counter, its floc/primary event fractions use the historical
successful-scatter fallback and record that basis and provenance in the run
summary. The command-line form is:

```bash
python3 claritas_production_diagnostics.py ray_exits_622nm.h5 \
  --wl-nm 622 \
  --outdir diagnostics \
  --detector-angles 0:180:10 \
  --detector-acceptance-deg 6.5
```

## Legacy four-dataset baseline check

The new measured-comparison helper was exercised against the existing
CLARITAS 76 baseline screen outputs. These are compatibility results, not new
CLARITAS 77 predictions:

| Dataset | Shape RMSE | Legacy detector assignment fraction | Reconstructed unique detected fraction | Escaped fraction | Legacy absorbed flag |
|---|---:|---:|---:|---:|---:|
| Loess 0.5 g/L | 0.024618 | 0.36185 | 0.29756 | 0.78433 | 0.21567 |
| Loess 4.0 g/L | 0.087107 | 0.05179 | 0.04056 | 0.27140 | 0.72860 |
| Kaolin 0.5 g/L | 0.070275 | 0.37623 | 0.29246 | 0.78515 | 0.21485 |
| Kaolin 4.0 g/L | 0.051903 | 0.11364 | 0.09850 | 0.47637 | 0.52363 |

The gap between assignment and unique fractions quantifies the overlapping
detector-window double counting in CLARITAS 76. Its absorbed flag can include
forced event-cap termination and must not be interpreted as a clean physical
absorption measurement.
