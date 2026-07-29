# CLARITAS 77 workstation validation

This runbook validates the production CUDA/CuPy branch in stages. Run the
stages in order and stop if an acceptance check fails. The commands assume the
returned project files are in `/home/ben`; change `CLARITAS_PROJECT` once if
they are elsewhere.

Use a fresh validation root so stale CLARITAS 76 files cannot be mistaken for
CLARITAS 77 output. Keep the same shell open for all stages so the exported
paths remain available.

```bash
set -euo pipefail
export CLARITAS_PROJECT=/home/ben
export CLARITAS_VALIDATION_ROOT="$CLARITAS_PROJECT/claritas_77_validation_$(date -u +%Y%m%dT%H%M%SZ)"
export MPLBACKEND=Agg
mkdir -p "$CLARITAS_VALIDATION_ROOT"
cd "$CLARITAS_PROJECT"
printf 'Validation output: %s\n' "$CLARITAS_VALIDATION_ROOT"
```

For long runs, use this inside an existing `tmux` or `screen` session. If a
shell must be restarted, re-export `CLARITAS_PROJECT`,
`CLARITAS_VALIDATION_ROOT`, and `MPLBACKEND` with the values printed above.

## 1. Environment and CUDA smoke check

Confirm the NVIDIA driver, Python dependencies, CuPy device access, NVRTC
compilation, and the two `miepython` APIs used by CLARITAS 77:

```bash
cd "$CLARITAS_PROJECT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

python3 - <<'PY'
import sys
import numpy as np
import pandas as pd
import matplotlib
import h5py
import miepython
import cupy as cp
import tqdm

print("Python:", sys.version.replace("\n", " "))
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("h5py:", h5py.__version__)
print("miepython:", getattr(miepython, "__version__", "version attribute unavailable"))
print("CuPy:", cp.__version__)
print("tqdm:", tqdm.__version__)

assert callable(miepython.efficiencies_mx)
assert callable(miepython.S1_S2)
assert cp.cuda.runtime.getDeviceCount() >= 1

properties = cp.cuda.runtime.getDeviceProperties(0)
device_name = properties["name"]
if isinstance(device_name, bytes):
    device_name = device_name.decode()
print("CUDA device 0:", device_name)

source = r'''
extern "C" __global__
void add_one(const float* x, float* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] = x[i] + 1.0f;
}
'''
kernel = cp.RawKernel(source, "add_one")
x = cp.arange(32, dtype=cp.float32)
y = cp.empty_like(x)
kernel((1,), (32,), (x, y, np.int32(x.size)))
cp.cuda.Stream.null.synchronize()
assert bool(cp.allclose(y, x + 1.0).item())
print("PASS: CuPy allocation, transfer, NVRTC compilation, and kernel launch")
cp.get_default_memory_pool().free_all_blocks()
PY
```

Do not continue if `nvidia-smi`, a dependency import, the Mie API checks, or
the RawKernel launch fails. The CuPy build must match the workstation's CUDA
driver/runtime.

## 2. CPU-only source and helper checks

These checks do not launch CUDA:

```bash
cd "$CLARITAS_PROJECT"

python3 -m py_compile \
  CLARITAS_77.py \
  claritas_production_diagnostics.py \
  claritas_measured_comparison.py \
  run_claritas_parameter_screen.py \
  validate_claritas_77.py \
  validate_claritas_77_reference_parity.py \
  validate_claritas_measured_comparison.py \
  test_claritas_production_diagnostics.py

python3 test_claritas_production_diagnostics.py
python3 validate_claritas_measured_comparison.py
python3 -m claritas_reference.validate \
  --output "$CLARITAS_VALIDATION_ROOT/cpu_reference_validation"
```

Expected results include:

```text
Ran 4 tests
OK

PASS: shape regions
PASS: end-to-end accounting
PASS: project measured data
PASS: 3 comparison validation checks

Ran 10 tests
OK
Validation metrics: 19/19 passed
```

These validate terminal/event accounting, explicit legacy provenance,
diagnostic invariants, HDF5 schema handling, measured-column parsing, the
0–50°/60–110°/120–170° shape regions, canonical absolute accounting, and the
rule that the measured curves must not be treated as absolutely calibrated.

## 3. Primary-only CPU-reference parity run

First exercise the transferred transport without the unresolved floc model.
`--reference-parity` uses the existing CUDA production code but replaces the
source with the CPU reference's centred/collimated launch and sets
`FLOC_ENABLED=False` in the generated case:

```bash
cd "$CLARITAS_PROJECT"
mkdir -p "$CLARITAS_VALIDATION_ROOT/controller_logs"

python3 run_claritas_parameter_screen.py \
  --base "$CLARITAS_PROJECT/CLARITAS_77.py" \
  --measured "$CLARITAS_PROJECT/sediment_data.csv" \
  --output "$CLARITAS_VALIDATION_ROOT/reference_parity" \
  --rays 100000 \
  --case baseline \
  --dataset loess_0p5gL \
  --reference-parity \
  2>&1 | tee "$CLARITAS_VALIDATION_ROOT/controller_logs/reference_parity.log"

export CLARITAS_PARITY_RUN="$CLARITAS_VALIDATION_ROOT/reference_parity/baseline/loess_0p5gL"

python3 validate_claritas_77.py \
  "$CLARITAS_PARITY_RUN/ray_exits_622nm.h5" \
  --sample-radius-m 0.049 \
  --output "$CLARITAS_PARITY_RUN/claritas_77_acceptance.csv"

python3 - "$CLARITAS_PARITY_RUN/ray_exits_622nm.h5" <<'PY'
import sys
import h5py

with h5py.File(sys.argv[1], "r") as handle:
    assert handle.attrs["claritas_version"] == "77"
    assert handle.attrs["source_model"] == "reference_collimated"
    assert not bool(handle.attrs["floc_enabled"])
    assert handle.attrs["transport_geometry"] == "3d_sphere"
print("PASS: primary-only reference-parity production configuration")
PY

python3 validate_claritas_77_reference_parity.py \
  "$CLARITAS_PARITY_RUN/ray_exits_622nm.h5" \
  --cpu-rays 100000 \
  --output "$CLARITAS_PARITY_RUN/claritas_77_reference_parity.csv"
```

The final command reconstructs the existing CPU-reference medium from the
recorded material, concentration, wavelength, radius, event cap, and detector
metadata. It must exit with status zero and report every check passed. Its
default statistical gate is the absolute difference bounded by five times the
two-sample standard error plus a dimensionless `5e-4` numerical floor; the
floor is a CUDA-float allowance, not a fitted physics parameter. CPU and GPU
histories are not expected to be ray-for-ray identical because their random
generators differ.

## 4. One 100,000-ray baseline production run

Use the screening driver even for the first run. It creates an isolated run
directory, makes the material/concentration substitutions through the AST,
copies `sediment_data.csv` into the run directory, and exposes the project
helper modules through `PYTHONPATH`.

```bash
cd "$CLARITAS_PROJECT"
mkdir -p "$CLARITAS_VALIDATION_ROOT/controller_logs"

python3 run_claritas_parameter_screen.py \
  --base "$CLARITAS_PROJECT/CLARITAS_77.py" \
  --measured "$CLARITAS_PROJECT/sediment_data.csv" \
  --output "$CLARITAS_VALIDATION_ROOT/screen" \
  --rays 100000 \
  --case baseline \
  --dataset loess_0p5gL \
  2>&1 | tee "$CLARITAS_VALIDATION_ROOT/controller_logs/one_baseline.log"

export CLARITAS_BASELINE_RUN="$CLARITAS_VALIDATION_ROOT/screen/baseline/loess_0p5gL"
test -s "$CLARITAS_BASELINE_RUN/ray_exits_622nm.h5"
test -s "$CLARITAS_BASELINE_RUN/run_summary_622nm.csv"
test -s "$CLARITAS_BASELINE_RUN/diagnostic_integrity_622nm.csv"
test -s "$CLARITAS_BASELINE_RUN/detector_hits.csv"
test -s "$CLARITAS_BASELINE_RUN/measured_comparison_metrics_622nm.csv"
test -s "$CLARITAS_BASELINE_RUN/result.json"
cmp "$CLARITAS_PROJECT/sediment_data.csv" "$CLARITAS_BASELINE_RUN/sediment_data.csv"
```

The driver should finish with `Status: completed`. Inspect these first:

- `run.log`: CUDA compilation/runtime messages, seed, material, concentration,
  probability-table checks, and any OOM retry.
- `ray_exits_622nm.h5`: canonical per-ray records and provenance attributes.
- `run_summary_622nm.csv`: terminal-state, detector, scatter, path, floc, and
  extinction summary.
- `diagnostic_integrity_622nm.csv`: applicable invariant checks.
- `detector_hits.csv` and `detector_efficiency_622nm.csv`: unique nearest-band
  detector accounting.
- `measured_comparison_622nm.csv`: per-angle measured/model shape and absolute
  simulated detector fraction.
- `measured_comparison_metrics_622nm.csv`: shape errors plus absolute simulated
  transport fractions.
- `result.json` and `measured_vs_model.csv`: screening-driver record.

Inspect the canonical HDF5 provenance:

```bash
python3 - "$CLARITAS_BASELINE_RUN/ray_exits_622nm.h5" <<'PY'
import sys
import h5py

path = sys.argv[1]
with h5py.File(path, "r") as handle:
    for name in (
        "claritas_version",
        "simulation_seed",
        "transport_seed_uint32",
        "azimuth_seed_uint32",
        "rng_seed_mapping",
        "maximum_unique_rng_initial_states",
        "transport_rng",
        "source_rng",
        "transport_geometry",
        "source_model",
        "detector_geometry",
        "phase_function_measure",
        "material",
        "concentration_g_per_L",
        "wavelength_m",
        "primary_refractive_index_real",
        "primary_refractive_index_imag_k",
        "n_rays",
        "sample_radius_m",
        "max_extinctions",
        "python_source_sha256",
        "cuda_source_sha256",
        "initial_chunk_rays",
        "final_chunk_rays",
        "oom_retry_count",
    ):
        print(f"{name}: {handle.attrs[name]}")
PY
```

The expected identifying values include `claritas_version=77`,
`simulation_seed=20260727`, `transport_geometry=3d_sphere`,
`source_model=production_beta`,
`detector_geometry=ideal_annular_nearest_accepted_band`, `material=loess`,
`concentration_g_per_L=0.5`, and `n_rays=100000`.

## 5. HDF5 and diagnostic acceptance

Run the independent post-run checker:

```bash
cd "$CLARITAS_PROJECT"

python3 validate_claritas_77.py \
  "$CLARITAS_BASELINE_RUN/ray_exits_622nm.h5" \
  --sample-radius-m 0.049 \
  --output "$CLARITAS_BASELINE_RUN/claritas_77_acceptance.csv"
```

The command must exit with status zero and print that every reported metric
passed. It checks:

- every ray has a known terminal state;
- terminal probabilities sum to one;
- absorbed/truncated flags agree with terminal state;
- escaped positions lie on the 0.049 m sphere within tolerance;
- escaped direction vectors remain normalized;
- scatter/extinction/floc event-count invariants and absorption-event closure
  hold;
- escaped, absorbed, and truncated path lengths are finite and non-negative.

Then inspect the richer diagnostic and measured summaries:

```bash
python3 - "$CLARITAS_BASELINE_RUN" <<'PY'
import sys
from pathlib import Path
import numpy as np
import pandas as pd

run = Path(sys.argv[1])
summary = pd.read_csv(run / "run_summary_622nm.csv").iloc[0]
integrity = pd.read_csv(run / "diagnostic_integrity_622nm.csv")
measured = pd.read_csv(run / "measured_comparison_metrics_622nm.csv").iloc[0]

applicable = integrity["applicable"].astype(str).str.lower().eq("true")
passed = integrity["passed"].astype(str).str.lower().eq("true")
failed = integrity[applicable & ~passed]

print(summary[[
    "total_ray_count",
    "escaped_fraction",
    "absorbed_fraction",
    "truncated_fraction",
    "unclassified_fraction",
    "partition_sum_including_unclassified",
    "total_detected_fraction",
    "ballistic_fraction",
    "single_scattered_fraction",
    "multiply_scattered_fraction",
    "mean_scatter_count",
    "floc_event_fraction",
    "primary_event_fraction",
]])
print("\nMeasured/model summary:")
print(measured[[
    "rmse",
    "mae",
    "forward_rmse_0_50",
    "middle_rmse_60_110",
    "rear_rmse_120_170",
    "absolute_detector_efficiency",
    "total_detected_fraction",
    "total_escaped_fraction",
    "total_absorbed_fraction",
    "total_truncated_fraction",
    "total_unclassified_fraction",
    "transport_partition_sum_including_unclassified",
    "measured_absolute_scale_available",
]])

assert failed.empty, failed.to_string(index=False)
assert np.isclose(summary["partition_sum_including_unclassified"], 1.0)
assert np.isclose(
    measured["transport_partition_sum_including_unclassified"], 1.0
)
assert measured["total_detected_fraction"] <= measured["total_escaped_fraction"]
assert np.isclose(
    measured["absolute_detector_efficiency"],
    measured["total_detected_fraction"],
)
assert str(measured["measured_absolute_scale_available"]).lower() == "false"
print("\nPASS: canonical diagnostic and measured-comparison accounting")
PY
```

Investigate any nonzero `unclassified_fraction`. A nonzero
`truncated_fraction` is explicitly accounted for rather than mislabelled as
absorption, but it should also be investigated before treating the run as a
production result.

The detailed diagnostic files to inspect when a summary is surprising are:

- `scatter_count_histogram_622nm.csv`
- `path_length_histogram_622nm.csv`
- `absorption_probability_vs_scatter_count_622nm.csv`
- `detector_contribution_vs_scatter_order_622nm.csv`
- `scatter_statistics_622nm.csv`
- `floc_internal_scatter_statistics_622nm.csv`
- `extinction_statistics_622nm.csv`
- `exit_direction_distribution_622nm.csv`
- `exit_position_distribution_622nm.csv`
- `exit_position_azimuth_distribution_622nm.csv`
- `exit_position_cartesian_marginals_622nm.csv`

## 6. Same-seed repeatability and chunk invariance

Run the identical material, concentration, parameter case, ray count, and
hard-coded master seed in a second fresh output tree, but force 25,000-ray
chunks. The first 100,000-ray run uses one chunk; this repeat uses four and
therefore tests both replay and chunk-size invariance:

```bash
cd "$CLARITAS_PROJECT"

python3 run_claritas_parameter_screen.py \
  --base "$CLARITAS_PROJECT/CLARITAS_77.py" \
  --measured "$CLARITAS_PROJECT/sediment_data.csv" \
  --output "$CLARITAS_VALIDATION_ROOT/same_seed_repeat" \
  --rays 100000 \
  --gpu-min-chunk-rays 25000 \
  --gpu-max-chunk-rays 25000 \
  --case baseline \
  --dataset loess_0p5gL \
  2>&1 | tee "$CLARITAS_VALIDATION_ROOT/controller_logs/same_seed_repeat.log"

export CLARITAS_REPEAT_RUN="$CLARITAS_VALIDATION_ROOT/same_seed_repeat/baseline/loess_0p5gL"

python3 validate_claritas_77.py \
  "$CLARITAS_BASELINE_RUN/ray_exits_622nm.h5" \
  --compare "$CLARITAS_REPEAT_RUN/ray_exits_622nm.h5" \
  --sample-radius-m 0.049 \
  --output "$CLARITAS_VALIDATION_ROOT/same_seed_acceptance.csv"
```

Confirm `initial_chunk_rays` and `final_chunk_rays` are 25,000 in the repeat
HDF5 attributes. The comparison must report all metrics passed. It requires the same HDF5
dataset schema and exact equality for every recorded dataset, with NaNs treated
as equal. The floating atomic heatmap and rendered plots are intentionally
outside this per-ray reproducibility gate.

For an additional file-level check of the acceptance result:

```bash
python3 - "$CLARITAS_VALIDATION_ROOT/same_seed_acceptance.csv" <<'PY'
import sys
import pandas as pd

rows = pd.read_csv(sys.argv[1])
passed = rows["passed"].astype(str).str.lower().eq("true")
failed = rows[~passed]
assert failed.empty, failed.to_string(index=False)
print(f"PASS: {len(rows)} HDF5 and same-seed acceptance metrics")
PY
```

## 7. Four-dataset baseline screen

Extend the fresh baseline output tree to all measured datasets. `--resume`
will reuse the already completed Loess 0.5 run only because it contains
`result.json`, `detector_hits.csv`, one measured-comparison metrics file, and
one canonical run-summary file.

```bash
cd "$CLARITAS_PROJECT"

python3 run_claritas_parameter_screen.py \
  --base "$CLARITAS_PROJECT/CLARITAS_77.py" \
  --measured "$CLARITAS_PROJECT/sediment_data.csv" \
  --output "$CLARITAS_VALIDATION_ROOT/screen" \
  --rays 100000 \
  --case baseline \
  --resume \
  2>&1 | tee "$CLARITAS_VALIDATION_ROOT/controller_logs/four_dataset_baseline.log"
```

This must produce one baseline run for each slug:

```text
loess_0p5gL
loess_4gL
kaolin_0p5gL
kaolin_4gL
```

Audit the aggregate summary:

```bash
python3 - "$CLARITAS_VALIDATION_ROOT/screen/screening_runs.csv" <<'PY'
import sys
import numpy as np
import pandas as pd

path = sys.argv[1]
rows = pd.read_csv(path)
expected = {
    "loess_0p5gL",
    "loess_4gL",
    "kaolin_0p5gL",
    "kaolin_4gL",
}
required_numeric = [
    "rmse_all",
    "mae_all",
    "rmse_forward_0_50",
    "rmse_middle_60_110",
    "rmse_rear_120_170",
    "absolute_detector_efficiency",
    "total_detected_fraction",
    "detector_hit_assignment_fraction",
    "total_escaped_fraction",
    "total_absorbed_fraction",
    "total_truncated_fraction",
    "total_unclassified_fraction",
    "transport_accounting_sum",
    "transport_partition_sum_including_unclassified",
]

assert len(rows) == 4, rows
assert set(rows["dataset"]) == expected
assert set(rows["case"]) == {"baseline"}
assert set(rows["status"]).issubset({"completed", "cached"})
assert np.isfinite(rows[required_numeric].to_numpy(dtype=float)).all()
assert np.allclose(rows["transport_partition_sum_including_unclassified"], 1.0)
assert (
    rows["total_detected_fraction"] <= rows["total_escaped_fraction"]
).all()
assert rows["measured_absolute_scale_available"].astype(str).str.lower().eq(
    "false"
).all()

print(rows[[
    "dataset",
    "status",
    "rmse_all",
    "mae_all",
    "rmse_forward_0_50",
    "rmse_middle_60_110",
    "rmse_rear_120_170",
    "absolute_detector_efficiency",
    "total_detected_fraction",
    "total_escaped_fraction",
    "total_absorbed_fraction",
    "total_truncated_fraction",
    "total_unclassified_fraction",
]])
print("\nPASS: four-dataset baseline screen")
PY
```

Also inspect:

- `$CLARITAS_VALIDATION_ROOT/screen/screening_runs.csv`
- `$CLARITAS_VALIDATION_ROOT/screen/case_rankings.csv`
- `$CLARITAS_VALIDATION_ROOT/screen/best_parameter_set.txt`
- each `screen/baseline/<dataset>/result.json`
- each `screen/baseline/<dataset>/run_summary_622nm.csv`
- each `screen/baseline/<dataset>/measured_comparison_metrics_622nm.csv`
- each `screen/baseline/<dataset>/run.log`

Run HDF5 acceptance over all four baseline datasets:

```bash
cd "$CLARITAS_PROJECT"
for dataset in loess_0p5gL loess_4gL kaolin_0p5gL kaolin_4gL; do
  run_dir="$CLARITAS_VALIDATION_ROOT/screen/baseline/$dataset"
  python3 validate_claritas_77.py \
    "$run_dir/ray_exits_622nm.h5" \
    --sample-radius-m 0.049 \
    --output "$run_dir/claritas_77_acceptance.csv"
done
```

## 8. Optional full 28-run parameter screen

Only start this after the smoke, helper, HDF5, repeatability, and four-dataset
baseline gates pass. The current screen has seven parameter cases and four
datasets. Reusing the same fresh output root with `--resume` skips the four
validated baseline runs and executes the remaining 24 runs:

```bash
cd "$CLARITAS_PROJECT"

python3 run_claritas_parameter_screen.py \
  --base "$CLARITAS_PROJECT/CLARITAS_77.py" \
  --measured "$CLARITAS_PROJECT/sediment_data.csv" \
  --output "$CLARITAS_VALIDATION_ROOT/screen" \
  --rays 100000 \
  --resume \
  2>&1 | tee "$CLARITAS_VALIDATION_ROOT/controller_logs/full_parameter_screen.log"
```

If interrupted, rerun that exact command; the tightened resume gate only
accepts runs that include the new measured-comparison and canonical run-summary
outputs.

Check the completed screen:

```bash
python3 - "$CLARITAS_VALIDATION_ROOT/screen" <<'PY'
import sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
runs = pd.read_csv(root / "screening_runs.csv")
ranking = pd.read_csv(root / "case_rankings.csv")
numeric = [
    "rmse_all",
    "mae_all",
    "absolute_detector_efficiency",
    "total_detected_fraction",
    "total_escaped_fraction",
    "total_absorbed_fraction",
    "total_truncated_fraction",
    "total_unclassified_fraction",
]

assert len(runs) == 28, len(runs)
assert runs["case"].nunique() == 7
assert runs["dataset"].nunique() == 4
assert set(runs["status"]).issubset({"completed", "cached"})
assert np.isfinite(runs[numeric].to_numpy(dtype=float)).all()
assert np.allclose(runs["transport_partition_sum_including_unclassified"], 1.0)
assert len(ranking) == 7
assert ranking["complete_case"].astype(str).str.lower().eq("true").all()

print(ranking[[
    "rank",
    "case",
    "combined_rmse",
    "mean_rmse",
    "worst_dataset_rmse",
]])
print("\nPASS: full 28-run screen is complete")
PY
```

Inspect `case_rankings.csv`, `best_parameter_set.txt`, and the per-run
diagnostics before accepting the nominal best shape score. A low shape RMSE
does not by itself validate the absolute transport.

## Interpretation of measured versus absolute results

All four columns in `sediment_data.csv` sum to approximately one. The file has
no incident optical power, detector gain, collection area calibration, or
other radiometric scale. Therefore:

- `rmse`, `mae`, and the forward/middle/rear errors compare **unit-sum angular
  shape**;
- `absolute_detector_efficiency`, `total_detected_fraction`,
  `total_escaped_fraction`, `total_absorbed_fraction`, and
  `total_truncated_fraction` are **absolute model predictions**;
- no absolute measured residual or absolute goodness-of-fit can be calculated
  from the current CSV;
- the parameter-screen ranking remains based on unit-sum shape RMSE;
- absolute simulated fractions must be reported and inspected, but must not be
  fitted by inventing a measurement scale or detector multiplier.

The expected metrics output records
`measured_absolute_scale_available=False`. If that value is ever true, the
measurement file and its calibration provenance must be reviewed before using
an absolute fit metric.

## Acceptance summary

CLARITAS 77 is ready for broader production screening only when:

1. the CUDA smoke kernel launches successfully;
2. all CPU-only helper checks pass;
3. the primary-only centred/collimated reference-parity case passes HDF5
   acceptance and every CPU/GPU statistical parity gate;
4. the first floc-enabled 100,000-ray production run completes with canonical
   output files;
5. HDF5 terminal-state, boundary, direction, event, and path checks pass;
6. applicable diagnostic-integrity checks pass and the full partition sums to
   one;
7. the same-seed cross-chunk per-ray HDF5 comparison is exact;
8. all four baseline datasets complete and expose finite shape and absolute
   metrics.

These checks validate implementation and accounting. They do not validate the
unresolved floc model or establish absolute agreement with uncalibrated
measurements.
