#!/usr/bin/env python3
"""Post-run acceptance checks for the CLARITAS_77 CUDA production output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np


STATE_EXITED = 1
STATE_ABSORBED = 2
STATE_TRUNCATED = 3
STATE_MISSED_SAMPLE = 4


def _metric(
    rows: List[Dict[str, object]],
    name: str,
    observed: float,
    expected: float,
    tolerance: float,
) -> bool:
    error = abs(observed - expected)
    passed = bool(np.isfinite(error) and error <= tolerance)
    rows.append({
        "metric": name,
        "observed": observed,
        "expected": expected,
        "absolute_error": error,
        "tolerance": tolerance,
        "passed": passed,
    })
    return passed


def validate_hdf5(path: Path, sample_radius_m: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    required = {
        "exit_x", "exit_y", "exit_z", "exit_vx", "exit_vy", "exit_vz",
        "exit_rpl", "scatter_count", "floc_event_count", "extinction_count",
        "floc_extinction_count", "absorbed", "truncated", "terminal_state",
    }
    with h5py.File(path, "r") as handle:
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise RuntimeError(f"{path} is missing datasets: {missing}")
        arrays = {name: handle[name][:] for name in required}
        lengths = {len(values) for values in arrays.values()}
        if len(lengths) != 1:
            raise RuntimeError(f"Inconsistent per-ray dataset lengths: {lengths}")

    state = arrays["terminal_state"]
    n_rays = len(state)
    known = np.isin(
        state,
        [STATE_EXITED, STATE_ABSORBED, STATE_TRUNCATED, STATE_MISSED_SAMPLE],
    )
    _metric(rows, "classified_ray_fraction", float(np.mean(known)), 1.0, 0.0)

    exited = state == STATE_EXITED
    absorbed = state == STATE_ABSORBED
    truncated = state == STATE_TRUNCATED
    missed = state == STATE_MISSED_SAMPLE
    completed = exited | absorbed | truncated
    accounting = (
        np.mean(exited) + np.mean(absorbed) + np.mean(truncated) + np.mean(missed)
    )
    _metric(rows, "terminal_probability_sum", float(accounting), 1.0, 1e-12)
    _metric(
        rows,
        "absorbed_flag_consistency",
        float(np.mean(arrays["absorbed"] == absorbed.astype(np.int32))),
        1.0,
        0.0,
    )
    _metric(
        rows,
        "truncated_flag_consistency",
        float(np.mean(arrays["truncated"] == truncated.astype(np.int32))),
        1.0,
        0.0,
    )

    finite_exit_record = (
        np.isfinite(arrays["exit_x"])
        & np.isfinite(arrays["exit_y"])
        & np.isfinite(arrays["exit_z"])
        & np.isfinite(arrays["exit_vx"])
        & np.isfinite(arrays["exit_vy"])
        & np.isfinite(arrays["exit_vz"])
    )
    _metric(
        rows,
        "exit_record_terminal_state_consistency",
        float(np.mean(finite_exit_record == exited)),
        1.0,
        0.0,
    )

    if np.any(exited):
        position = np.column_stack((
            arrays["exit_x"][exited],
            arrays["exit_y"][exited],
            arrays["exit_z"][exited],
        )).astype(np.float64)
        direction = np.column_stack((
            arrays["exit_vx"][exited],
            arrays["exit_vy"][exited],
            arrays["exit_vz"][exited],
        )).astype(np.float64)
        radius_error = np.max(
            np.abs(np.linalg.norm(position, axis=1) - sample_radius_m)
        )
        direction_norm_error = np.max(
            np.abs(np.linalg.norm(direction, axis=1) - 1.0)
        )
        _metric(
            rows,
            "maximum_exit_boundary_error_m",
            float(radius_error),
            0.0,
            max(2e-6, sample_radius_m * 5e-5),
        )
        _metric(
            rows,
            "maximum_direction_norm_error",
            float(direction_norm_error),
            0.0,
            5e-5,
        )

    valid_counts = (
        (arrays["scatter_count"] >= 0)
        & (arrays["extinction_count"] >= 0)
        & (arrays["floc_event_count"] >= 0)
        & (arrays["floc_extinction_count"] >= 0)
        & (arrays["floc_event_count"] <= arrays["scatter_count"])
        & (arrays["floc_extinction_count"] <= arrays["extinction_count"])
        & (arrays["floc_event_count"] <= arrays["floc_extinction_count"])
        & (arrays["scatter_count"] <= arrays["extinction_count"])
    )
    _metric(
        rows,
        "event_count_invariant_fraction",
        float(np.mean(valid_counts)),
        1.0,
        0.0,
    )
    finite_paths = np.isfinite(arrays["exit_rpl"]) & (arrays["exit_rpl"] >= 0.0)
    _metric(
        rows,
        "completed_path_valid_fraction",
        float(np.mean(finite_paths[completed])) if np.any(completed) else 1.0,
        1.0,
        0.0,
    )
    extinction_minus_scatter = (
        arrays["extinction_count"] - arrays["scatter_count"]
    )
    outcome_event_accounting = (
        (absorbed & (extinction_minus_scatter == 1))
        | ((exited | truncated | missed) & (extinction_minus_scatter == 0))
    )
    _metric(
        rows,
        "extinction_outcome_accounting_fraction",
        float(np.mean(outcome_event_accounting)),
        1.0,
        0.0,
    )
    return rows


def compare_reproducibility(first: Path, second: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with h5py.File(first, "r") as a, h5py.File(second, "r") as b:
        fields_a = {
            name for name, value in a.items()
            if isinstance(value, h5py.Dataset)
        }
        fields_b = {
            name for name, value in b.items()
            if isinstance(value, h5py.Dataset)
        }
        _metric(
            rows,
            "reproducible_dataset_schema",
            float(fields_a == fields_b),
            1.0,
            0.0,
        )
        fields = sorted(fields_a | fields_b)
        for field in fields:
            identical = float(
                field in a
                and field in b
                and np.array_equal(
                    a[field][:],
                    b[field][:],
                    equal_nan=True,
                )
            )
            _metric(rows, f"reproducible_{field}", identical, 1.0, 0.0)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path, help="CLARITAS_77 ray_exits HDF5")
    parser.add_argument("--compare", type=Path, help="Second same-seed run")
    parser.add_argument("--sample-radius-m", type=float, default=0.049)
    parser.add_argument("--output", type=Path, default=Path("claritas_77_acceptance.csv"))
    args = parser.parse_args()

    rows = validate_hdf5(args.hdf5, args.sample_radius_m)
    if args.compare is not None:
        rows.extend(compare_reproducibility(args.hdf5, args.compare))

    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "metric", "observed", "expected", "absolute_error", "tolerance", "passed"
        ])
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(bool(row["passed"]) for row in rows)
    print(f"CLARITAS_77 acceptance: {passed}/{len(rows)} metrics passed")
    print(f"Saved {args.output}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
