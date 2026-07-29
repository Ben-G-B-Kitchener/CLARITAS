#!/usr/bin/env python3
"""CPU-only validation checks for claritas_measured_comparison."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import tempfile

import numpy as np

from claritas_measured_comparison import (
    EXPECTED_ANGLES_DEG,
    calculate_shape_metrics,
    read_measured_datasets,
    save_measured_comparison,
)


def _write_csv(path: Path, rows):
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def validate_shape_regions() -> None:
    measured = np.full(18, 1.0 / 18.0)
    model = measured.copy()
    metrics = calculate_shape_metrics(measured, model)
    for name in (
        "rmse",
        "mae",
        "forward_rmse_0_50",
        "middle_rmse_60_110",
        "rear_rmse_120_170",
    ):
        _assert_close(metrics[name], 0.0, name)


def validate_end_to_end() -> None:
    with tempfile.TemporaryDirectory(prefix="claritas_comparison_") as temp:
        root = Path(temp)
        measured_path = root / "sediment_data.csv"
        detector_path = root / "detector_hits.csv"
        summary_path = root / "run_summary_622nm.csv"

        measured_rows = []
        detector_rows = []
        for angle in EXPECTED_ANGLES_DEG.astype(int):
            measured_rows.append(
                {
                    "detector angle": angle,
                    "loess 0.5 measured": 1.0 / 18.0,
                }
            )
            detector_rows.append(
                {
                    "Detector_deg": angle,
                    "H_622nm": 10,
                }
            )

        _write_csv(measured_path, measured_rows)
        _write_csv(detector_path, detector_rows)
        _write_csv(
            summary_path,
            [
                {
                    "total_ray_count": 1000,
                    "escaped_ray_count": 700,
                    "escaped_fraction": 0.7,
                    "absorbed_ray_count": 200,
                    "absorbed_fraction": 0.2,
                    "truncated_ray_count": 50,
                    "truncated_fraction": 0.05,
                    "unclassified_ray_count": 50,
                    "unclassified_fraction": 0.05,
                    "detected_unique_ray_count": 150,
                    "total_detected_fraction": 0.15,
                    "detector_hit_assignment_count": 180,
                }
            ],
        )

        result = save_measured_comparison(
            measured_csv=measured_path,
            detector_csv=detector_path,
            output_dir=root,
            material="loess",
            concentration_g_per_L=0.5,
            wavelength_nm=622,
        )
        metrics = result["metrics"]

        _assert_close(float(metrics["rmse"]), 0.0, "end-to-end RMSE")
        _assert_close(
            float(metrics["absolute_detector_efficiency"]),
            0.15,
            "absolute detector efficiency",
        )
        _assert_close(
            float(metrics["detector_hit_assignment_fraction"]),
            0.18,
            "assignment fraction",
        )
        _assert_close(
            float(metrics["total_detected_fraction"]),
            0.15,
            "unique detected fraction",
        )
        _assert_close(
            float(metrics["total_escaped_fraction"]),
            0.7,
            "escaped fraction",
        )
        _assert_close(
            float(metrics["total_absorbed_fraction"]),
            0.2,
            "absorbed fraction",
        )
        _assert_close(
            float(metrics["transport_accounting_sum"]),
            0.95,
            "completed accounting sum",
        )
        _assert_close(
            float(metrics["transport_partition_sum_including_unclassified"]),
            1.0,
            "partition sum",
        )
        if not bool(metrics["total_detected_fraction_is_unique"]):
            raise AssertionError("Canonical unique detector count was not used")
        if bool(metrics["measured_absolute_scale_available"]):
            raise AssertionError(
                "Uncalibrated measured shape was treated as absolute"
            )

        for path_key in ("curve_path", "metrics_path", "json_path"):
            if not Path(result[path_key]).is_file():
                raise AssertionError(f"Missing output: {result[path_key]}")


def validate_project_measured_data() -> None:
    measured_path = Path(__file__).resolve().parent / "sediment_data.csv"
    if not measured_path.is_file():
        return
    datasets = read_measured_datasets(measured_path)
    expected = {
        "loess_0p5gL",
        "loess_4gL",
        "kaolin_0p5gL",
        "kaolin_4gL",
    }
    actual = {str(dataset["slug"]) for dataset in datasets}
    if actual != expected:
        raise AssertionError(
            f"Measured datasets differ: expected {expected}, found {actual}"
        )
    if any(
        bool(dataset["absolute_scale_available"])
        for dataset in datasets
    ):
        raise AssertionError("Measured shape was treated as absolute")


def main() -> int:
    checks = (
        ("shape regions", validate_shape_regions),
        ("end-to-end accounting", validate_end_to_end),
        ("project measured data", validate_project_measured_data),
    )
    for name, check in checks:
        check()
        print(f"PASS: {name}")
    print(f"PASS: {len(checks)} comparison validation checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
