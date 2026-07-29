#!/usr/bin/env python3
"""
Measured-data comparison and absolute detector accounting for CLARITAS.

The measured curves in ``sediment_data.csv`` are angular shapes: each of the
current four columns sums to one.  They therefore constrain angular shape but
do not contain an absolute radiometric calibration.  This module deliberately
keeps the two comparisons separate:

* shape metrics compare unit-sum measured and model detector curves;
* absolute metrics report simulated ray fractions without fitting a scale.

For legacy CLARITAS_76 output, detector bins can overlap because the 6.5 degree
acceptance half-angle is larger than half the 10 degree detector spacing.
``detector_hit_assignment_fraction`` preserves that legacy sum, while
``absolute_detector_efficiency`` and ``total_detected_fraction`` count unique
detected rays where a canonical run summary or ray-level exit file is
available.

The module has no CUDA dependency and is intended to be imported by the
production CUDA script after it has written its detector and transport files.
It can also post-process an existing output directory from the command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EXPECTED_ANGLES_DEG = np.arange(0.0, 180.0, 10.0, dtype=np.float64)
DEFAULT_DETECTOR_ACCEPTANCE_DEG = 6.5

SHAPE_REGIONS = (
    ("forward", 0.0, 50.0),
    ("middle", 60.0, 110.0),
    ("rear", 120.0, 170.0),
)


class ComparisonError(RuntimeError):
    """Raised when comparison inputs are present but internally inconsistent."""


def _normalise_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ComparisonError(f"No CSV header found in {path}")
    if not rows:
        raise ComparisonError(f"No data rows found in {path}")
    return fieldnames, rows


def _float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"Could not parse {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ComparisonError(f"{label} is not finite: {value!r}")
    return result


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _safe_fraction(numerator: Optional[float], denominator: Optional[float]) -> float:
    if numerator is None or denominator is None or denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def _dataset_slug(material: str, concentration_g_per_l: float) -> str:
    concentration = f"{concentration_g_per_l:g}".replace(".", "p")
    return f"{material.lower()}_{concentration}gL"


def read_measured_datasets(csv_path: Path) -> List[Dict[str, object]]:
    """Read all ``<material> <concentration> measured`` columns."""

    csv_path = Path(csv_path)
    fieldnames, rows = _read_csv(csv_path)
    normalised = {_normalise_header(name): name for name in fieldnames}

    angle_column = None
    for candidate in ("detector_angle", "detector_angle_deg"):
        if candidate in normalised:
            angle_column = normalised[candidate]
            break
    if angle_column is None:
        raise ComparisonError(
            f"No detector-angle column found in {csv_path}; headers={fieldnames}"
        )

    angles = np.asarray(
        [_float(row.get(angle_column), f"{angle_column} row") for row in rows],
        dtype=np.float64,
    )
    order = np.argsort(angles)
    angles = angles[order]

    if len(angles) != len(EXPECTED_ANGLES_DEG) or not np.allclose(
        angles,
        EXPECTED_ANGLES_DEG,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ComparisonError(
            "Measured detector angles must be "
            f"{EXPECTED_ANGLES_DEG.tolist()}, found {angles.tolist()}"
        )

    pattern = re.compile(
        r"^(?P<material>[A-Za-z0-9_-]+)[ _]+"
        r"(?P<concentration>[0-9]+(?:\.[0-9]+)?)"
        r"(?:[ _]+g(?:_?per_?l|/?l))?"
        r"[ _]+measured$",
        flags=re.IGNORECASE,
    )

    datasets: List[Dict[str, object]] = []
    for column in fieldnames:
        if column == angle_column:
            continue
        match = pattern.fullmatch(column.strip())
        if match is None:
            raise ComparisonError(
                f"Measured column {column!r} does not match "
                "'<material> <concentration> measured'"
            )

        material = match.group("material").lower().replace("-", "_")
        concentration = float(match.group("concentration"))
        values = np.asarray(
            [_float(row.get(column), f"{column} row") for row in rows],
            dtype=np.float64,
        )[order]

        if np.any(values < 0.0):
            raise ComparisonError(f"Measured column {column!r} is negative")
        total = float(np.sum(values))
        if not math.isfinite(total) or total <= 0.0:
            raise ComparisonError(
                f"Measured column {column!r} has invalid sum {total}"
            )

        datasets.append(
            {
                "column": column.strip(),
                "material": material,
                "concentration_g_per_L": concentration,
                "slug": _dataset_slug(material, concentration),
                "angles_deg": angles.copy(),
                "raw": values,
                "raw_sum": total,
                "shape": values / total,
                "is_unit_sum": bool(math.isclose(total, 1.0, abs_tol=1.0e-8)),
                # sediment_data.csv has no incident-power or detector-gain
                # metadata.  A non-unit sum must not be assumed to be absolute.
                "absolute_scale_available": False,
            }
        )

    if not datasets:
        raise ComparisonError(f"No measured datasets found in {csv_path}")
    return datasets


def select_measured_dataset(
    datasets: Sequence[Mapping[str, object]],
    material: Optional[str] = None,
    concentration_g_per_L: Optional[float] = None,
    dataset_column: Optional[str] = None,
) -> Mapping[str, object]:
    """Select one measured curve without relying on column order."""

    if dataset_column is not None:
        matches = [
            dataset
            for dataset in datasets
            if str(dataset["column"]).lower() == dataset_column.strip().lower()
            or str(dataset["slug"]).lower() == dataset_column.strip().lower()
        ]
    else:
        if material is None or concentration_g_per_L is None:
            raise ComparisonError(
                "Specify dataset_column or both material and "
                "concentration_g_per_L"
            )
        material_key = material.lower().replace("-", "_")
        matches = [
            dataset
            for dataset in datasets
            if str(dataset["material"]) == material_key
            and math.isclose(
                float(dataset["concentration_g_per_L"]),
                float(concentration_g_per_L),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ]

    if len(matches) != 1:
        available = ", ".join(str(dataset["slug"]) for dataset in datasets)
        raise ComparisonError(
            f"Expected one measured dataset, found {len(matches)}. "
            f"Available: {available}"
        )
    return matches[0]


def read_detector_curve(
    detector_csv: Path,
    wavelength_nm: Optional[int] = None,
) -> Dict[str, object]:
    """Read a CLARITAS detector curve and select one wavelength column."""

    detector_csv = Path(detector_csv)
    fieldnames, rows = _read_csv(detector_csv)
    by_normalised = {_normalise_header(name): name for name in fieldnames}

    angle_column = None
    for candidate in ("detector_deg", "detector_angle_deg", "detector_angle"):
        if candidate in by_normalised:
            angle_column = by_normalised[candidate]
            break
    if angle_column is None:
        raise ComparisonError(
            f"No detector angle column found in {detector_csv}"
        )

    candidate_columns: List[Tuple[str, Optional[int]]] = []
    for column in fieldnames:
        if column == angle_column:
            continue
        normal = _normalise_header(column)
        match = re.fullmatch(r"h_(\d+)nm", normal)
        if match is None:
            match = re.fullmatch(r"counts?_(\d+)nm", normal)
        if match is not None:
            candidate_columns.append((column, int(match.group(1))))
        elif normal in {"hit_count", "model_raw_hits", "model_hit_count"}:
            candidate_columns.append((column, None))

    if wavelength_nm is not None:
        exact = [
            item
            for item in candidate_columns
            if item[1] == int(wavelength_nm)
        ]
        generic = [item for item in candidate_columns if item[1] is None]
        selected = exact or generic
    else:
        selected = candidate_columns

    if len(selected) != 1:
        raise ComparisonError(
            f"Could not select one detector-count column in {detector_csv}. "
            f"Candidates: {candidate_columns}; wavelength={wavelength_nm}"
        )

    count_column, parsed_wavelength = selected[0]
    angles = np.asarray(
        [_float(row.get(angle_column), f"{angle_column} row") for row in rows],
        dtype=np.float64,
    )
    counts = np.asarray(
        [_float(row.get(count_column), f"{count_column} row") for row in rows],
        dtype=np.float64,
    )
    order = np.argsort(angles)
    angles = angles[order]
    counts = counts[order]

    if len(angles) != len(EXPECTED_ANGLES_DEG) or not np.allclose(
        angles,
        EXPECTED_ANGLES_DEG,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ComparisonError(
            "Model detector angles must be "
            f"{EXPECTED_ANGLES_DEG.tolist()}, found {angles.tolist()}"
        )
    if np.any(counts < 0.0):
        raise ComparisonError(f"Negative detector count in {detector_csv}")

    total = float(np.sum(counts))
    if not math.isfinite(total) or total <= 0.0:
        raise ComparisonError(
            f"Detector curve in {detector_csv} has invalid sum {total}"
        )

    return {
        "angles_deg": angles,
        "counts": counts,
        "shape": counts / total,
        "hit_assignment_count": total,
        "wavelength_nm": (
            parsed_wavelength
            if parsed_wavelength is not None
            else wavelength_nm
        ),
        "count_column": count_column,
        "path": str(detector_csv),
    }


def calculate_shape_metrics(
    measured_shape: np.ndarray,
    model_shape: np.ndarray,
    angles_deg: np.ndarray = EXPECTED_ANGLES_DEG,
) -> Dict[str, float]:
    """Calculate global and requested regional unit-sum shape metrics."""

    measured_shape = np.asarray(measured_shape, dtype=np.float64)
    model_shape = np.asarray(model_shape, dtype=np.float64)
    angles_deg = np.asarray(angles_deg, dtype=np.float64)

    if measured_shape.shape != model_shape.shape:
        raise ComparisonError(
            f"Shape size mismatch: {measured_shape.shape} vs {model_shape.shape}"
        )
    if measured_shape.shape != angles_deg.shape:
        raise ComparisonError("Angle and response arrays have different sizes")
    if not np.all(np.isfinite(measured_shape)) or not np.all(
        np.isfinite(model_shape)
    ):
        raise ComparisonError("Shape arrays contain non-finite values")

    residual = model_shape - measured_shape
    metrics = {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "max_absolute_error": float(np.max(np.abs(residual))),
    }

    covered = np.zeros(len(angles_deg), dtype=bool)
    for label, lower, upper in SHAPE_REGIONS:
        mask = (angles_deg >= lower) & (angles_deg <= upper)
        if not np.any(mask):
            raise ComparisonError(
                f"No detector angles in requested {label} region"
            )
        covered |= mask
        region_residual = residual[mask]
        metrics[f"{label}_rmse_{int(lower)}_{int(upper)}"] = float(
            np.sqrt(np.mean(region_residual ** 2))
        )
        metrics[f"{label}_mae_{int(lower)}_{int(upper)}"] = float(
            np.mean(np.abs(region_residual))
        )

    if not np.all(covered):
        raise ComparisonError(
            "Forward/middle/rear metric regions do not cover all detectors"
        )
    return metrics


def _first_present(
    row: Mapping[str, object],
    names: Iterable[str],
) -> Optional[float]:
    normalised = {_normalise_header(key): value for key, value in row.items()}
    for name in names:
        value = _optional_float(normalised.get(_normalise_header(name)))
        if value is not None:
            return value
    return None


def _find_wavelength_file(
    output_dir: Path,
    prefix: str,
    wavelength_nm: Optional[int],
) -> Optional[Path]:
    if wavelength_nm is not None:
        exact = output_dir / f"{prefix}_{int(wavelength_nm)}nm.csv"
        if exact.exists():
            return exact
    candidates = sorted(output_dir.glob(f"{prefix}_*nm.csv"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _accounting_from_summary(
    summary_path: Optional[Path] = None,
    summary_row: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    if summary_row is None:
        if summary_path is None:
            raise ComparisonError("No canonical summary input was supplied")
        _, rows = _read_csv(summary_path)
        if len(rows) != 1:
            raise ComparisonError(
                f"Expected one row in run summary {summary_path}, "
                f"found {len(rows)}"
            )
        row: Mapping[str, object] = rows[0]
        source = str(summary_path)
    else:
        row = summary_row
        source = "in-memory canonical run summary"

    total = _first_present(
        row,
        ("total_ray_count", "total_rays", "n_rays", "N_RAYS"),
    )
    escaped_count = _first_present(
        row,
        ("escaped_ray_count", "valid_exit_rays", "escaped_rays"),
    )
    escaped_fraction = _first_present(
        row,
        ("escaped_fraction", "valid_exit_fraction", "total_escaped_fraction"),
    )
    absorbed_count = _first_present(
        row,
        ("absorbed_ray_count", "absorbed_rays"),
    )
    absorbed_fraction = _first_present(
        row,
        ("absorbed_fraction", "total_absorbed_fraction"),
    )
    unclassified_count = _first_present(
        row,
        (
            "unclassified_ray_count",
            "invalid_ray_count",
        ),
    )
    unclassified_fraction = _first_present(
        row,
        (
            "unclassified_fraction",
            "invalid_fraction",
        ),
    )
    truncated_count = _first_present(
        row,
        ("truncated_ray_count", "terminated_ray_count"),
    )
    truncated_fraction = _first_present(
        row,
        ("truncated_fraction", "terminated_fraction"),
    )
    unique_detected_count = _first_present(
        row,
        (
            "detected_unique_ray_count",
            "unique_detected_ray_count",
            "detected_ray_count",
        ),
    )
    total_detected_fraction = _first_present(
        row,
        ("total_detected_fraction", "detected_unique_fraction"),
    )
    hit_assignments = _first_present(
        row,
        ("detector_hit_assignment_count", "total_detector_hit_assignments"),
    )

    if total is None or total <= 0.0:
        raise ComparisonError(
            f"Canonical run summary has no valid total ray count: {source}"
        )

    if escaped_count is None and escaped_fraction is not None:
        escaped_count = escaped_fraction * total
    if escaped_fraction is None:
        escaped_fraction = _safe_fraction(escaped_count, total)

    if absorbed_count is None and absorbed_fraction is not None:
        absorbed_count = absorbed_fraction * total
    if absorbed_fraction is None:
        absorbed_fraction = _safe_fraction(absorbed_count, total)

    if unclassified_count is None and unclassified_fraction is not None:
        unclassified_count = unclassified_fraction * total
    if unclassified_fraction is None:
        unclassified_fraction = _safe_fraction(unclassified_count, total)

    if truncated_count is None and truncated_fraction is not None:
        truncated_count = truncated_fraction * total
    if truncated_fraction is None:
        truncated_fraction = _safe_fraction(truncated_count, total)

    if unique_detected_count is None and total_detected_fraction is not None:
        unique_detected_count = total_detected_fraction * total
    if total_detected_fraction is None:
        total_detected_fraction = _safe_fraction(unique_detected_count, total)

    return {
        "accounting_source": source,
        "accounting_quality": "canonical_run_summary",
        "total_ray_count": total,
        "escaped_ray_count": escaped_count,
        "total_escaped_fraction": escaped_fraction,
        "absorbed_ray_count": absorbed_count,
        "total_absorbed_fraction": absorbed_fraction,
        "unclassified_ray_count": unclassified_count,
        "total_unclassified_fraction": unclassified_fraction,
        "truncated_ray_count": truncated_count,
        "truncated_fraction": truncated_fraction,
        "detected_unique_ray_count": unique_detected_count,
        "total_detected_fraction": total_detected_fraction,
        "detector_hit_assignment_count_summary": hit_assignments,
        "total_detected_fraction_is_unique": unique_detected_count is not None,
        "absorption_definition": (
            "canonical explicit absorbed status"
            if absorbed_count is not None
            else "unavailable"
        ),
    }


def _nearest_unique_detector(
    exit_position_angle_deg: float,
    scatter_count: float,
    detector_angles_deg: np.ndarray,
    detector_acceptance_deg: float,
) -> Optional[int]:
    if not math.isfinite(exit_position_angle_deg):
        return None
    if exit_position_angle_deg < 0.0 or exit_position_angle_deg > 180.0:
        return None

    differences = np.abs(detector_angles_deg - exit_position_angle_deg)
    candidates = differences <= detector_acceptance_deg
    # Preserve the production rule that ballistic rays cannot contribute to
    # backscatter detector centres.
    if scatter_count <= 0.0:
        candidates &= detector_angles_deg < 90.0
    if not np.any(candidates):
        return None

    candidate_indices = np.flatnonzero(candidates)
    return int(candidate_indices[np.argmin(differences[candidate_indices])])


def _accounting_from_exit_points(
    exit_points_path: Path,
    detector_angles_deg: np.ndarray,
    detector_acceptance_deg: float,
) -> Dict[str, object]:
    fieldnames, rows = _read_csv(exit_points_path)
    names = {_normalise_header(name): name for name in fieldnames}

    valid_column = names.get("is_valid_exit")
    absorbed_column = names.get("absorbed_flag")
    absorbed_bool_column = names.get("is_absorbed")
    truncated_column = names.get("truncated_flag") or names.get("is_truncated")
    detector_index_column = names.get("detector_index")
    x_column = names.get("exit_x_m")
    y_column = names.get("exit_y_m")
    direction_column = names.get("exit_dir_deg")
    position_angle_column = names.get("exit_pos_angle_deg")
    scatter_column = names.get("scatter_count")

    total = len(rows)
    escaped = 0
    absorbed = 0
    truncated = 0
    unclassified = 0
    unique_counts = np.zeros(len(detector_angles_deg), dtype=np.int64)

    for row in rows:
        valid = (
            _parse_bool(row.get(valid_column))
            if valid_column is not None
            else None
        )
        if valid is None:
            x = _optional_float(row.get(x_column)) if x_column else None
            y = _optional_float(row.get(y_column)) if y_column else None
            direction = (
                _optional_float(row.get(direction_column))
                if direction_column
                else None
            )
            valid = x is not None and y is not None and direction is not None

        absorbed_flag = (
            _optional_float(row.get(absorbed_column))
            if absorbed_column is not None
            else None
        )
        absorbed_bool = (
            _parse_bool(row.get(absorbed_bool_column))
            if absorbed_bool_column is not None
            else None
        )
        is_absorbed = (
            absorbed_bool
            if absorbed_bool is not None
            else absorbed_flag is not None and absorbed_flag > 0.5
        )
        truncated_bool = (
            _parse_bool(row.get(truncated_column))
            if truncated_column is not None
            else None
        )
        truncated_value = (
            _optional_float(row.get(truncated_column))
            if truncated_column is not None and truncated_bool is None
            else None
        )
        is_truncated = bool(
            truncated_bool
            if truncated_bool is not None
            else truncated_value is not None and truncated_value > 0.5
        )

        if valid:
            escaped += 1
        elif is_absorbed:
            absorbed += 1
        elif is_truncated:
            truncated += 1
        else:
            unclassified += 1

        if not valid:
            continue

        if detector_index_column is not None:
            detector_index_value = _optional_float(
                row.get(detector_index_column)
            )
            if detector_index_value is not None:
                detector_index = int(detector_index_value)
                if 0 <= detector_index < len(detector_angles_deg):
                    unique_counts[detector_index] += 1
            continue

        if position_angle_column is None:
            continue
        position_angle = _optional_float(row.get(position_angle_column))
        scatter_count = (
            _optional_float(row.get(scatter_column))
            if scatter_column is not None
            else None
        )
        if position_angle is None or scatter_count is None:
            continue

        reconstructed_index = _nearest_unique_detector(
            position_angle,
            scatter_count,
            detector_angles_deg,
            detector_acceptance_deg,
        )
        if reconstructed_index is not None:
            unique_counts[reconstructed_index] += 1

    unique_detected = int(np.sum(unique_counts))
    # In CLARITAS_76, absorbed_flag also marks the extinction cap and certain
    # forced termination paths.  Preserve that limitation in the metadata.
    if truncated_column is not None and (
        absorbed_column is not None or absorbed_bool_column is not None
    ):
        absorption_definition = "explicit absorbed status; truncation separate"
        accounting_quality = "ray_level_explicit_status"
    elif absorbed_column is not None:
        absorption_definition = (
            "legacy absorbed_flag; may include forced termination"
        )
        accounting_quality = "legacy_ray_level_reconstruction"
    else:
        absorption_definition = (
            "unavailable; non-escaped rays are unclassified"
        )
        accounting_quality = "legacy_ray_level_reconstruction"

    return {
        "accounting_source": str(exit_points_path),
        "accounting_quality": accounting_quality,
        "total_ray_count": float(total),
        "escaped_ray_count": float(escaped),
        "total_escaped_fraction": escaped / total,
        "absorbed_ray_count": (
            float(absorbed)
            if absorbed_column is not None or absorbed_bool_column is not None
            else None
        ),
        "total_absorbed_fraction": (
            absorbed / total
            if absorbed_column is not None or absorbed_bool_column is not None
            else float("nan")
        ),
        "unclassified_ray_count": float(unclassified),
        "total_unclassified_fraction": unclassified / total,
        "truncated_ray_count": float(truncated),
        "truncated_fraction": truncated / total,
        "detected_unique_ray_count": float(unique_detected),
        "total_detected_fraction": unique_detected / total,
        "unique_detector_counts": unique_counts.astype(np.float64),
        "total_detected_fraction_is_unique": True,
        "absorption_definition": absorption_definition,
        "detector_acceptance_deg_for_unique_count": (
            None
            if detector_index_column is not None
            else detector_acceptance_deg
        ),
    }


def _accounting_from_legacy_overview(
    overview_path: Path,
) -> Dict[str, object]:
    _, rows = _read_csv(overview_path)
    if len(rows) != 1:
        raise ComparisonError(
            f"Expected one row in {overview_path}, found {len(rows)}"
        )
    row = rows[0]
    total = _first_present(row, ("total_rays", "total_ray_count"))
    escaped_count = _first_present(
        row,
        ("valid_exit_rays", "escaped_ray_count"),
    )
    escaped_fraction = _first_present(
        row,
        ("valid_exit_fraction", "escaped_fraction"),
    )
    unresolved_count = _first_present(
        row,
        ("absorbed_or_invalid_rays",),
    )
    unresolved_fraction = _first_present(
        row,
        ("absorbed_or_invalid_fraction",),
    )
    if total is None or total <= 0.0:
        raise ComparisonError(f"No total ray count in {overview_path}")
    if escaped_count is None and escaped_fraction is not None:
        escaped_count = escaped_fraction * total
    if escaped_fraction is None:
        escaped_fraction = _safe_fraction(escaped_count, total)
    if unresolved_count is None and unresolved_fraction is not None:
        unresolved_count = unresolved_fraction * total
    if unresolved_fraction is None:
        unresolved_fraction = _safe_fraction(unresolved_count, total)

    return {
        "accounting_source": str(overview_path),
        "accounting_quality": "legacy_aggregate_unresolved",
        "total_ray_count": total,
        "escaped_ray_count": escaped_count,
        "total_escaped_fraction": escaped_fraction,
        "absorbed_ray_count": None,
        "total_absorbed_fraction": float("nan"),
        "unclassified_ray_count": unresolved_count,
        "total_unclassified_fraction": unresolved_fraction,
        "truncated_ray_count": None,
        "truncated_fraction": float("nan"),
        "detected_unique_ray_count": None,
        "total_detected_fraction": float("nan"),
        "total_detected_fraction_is_unique": False,
        "absorption_definition": (
            "unavailable; legacy overview merges absorbed and invalid rays"
        ),
    }


def load_transport_accounting(
    output_dir: Path,
    wavelength_nm: Optional[int],
    detector_angles_deg: np.ndarray,
    n_rays: Optional[int] = None,
    detector_acceptance_deg: float = DEFAULT_DETECTOR_ACCEPTANCE_DEG,
    transport_summary: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """
    Load accounting in decreasing order of evidential quality.

    Canonical ``run_summary`` output is preferred.  Ray-level legacy output is
    next because it can distinguish the explicit flag and reconstruct unique
    detector hits.  The old aggregate overview is used only when necessary.
    """

    output_dir = Path(output_dir)
    if transport_summary is not None:
        return _accounting_from_summary(summary_row=transport_summary)

    summary_path = _find_wavelength_file(
        output_dir,
        "run_summary",
        wavelength_nm,
    )
    if summary_path is not None:
        return _accounting_from_summary(summary_path)

    exit_points_path = _find_wavelength_file(
        output_dir,
        "exit_points",
        wavelength_nm,
    )
    if exit_points_path is not None:
        return _accounting_from_exit_points(
            exit_points_path,
            detector_angles_deg,
            detector_acceptance_deg,
        )

    overview_path = _find_wavelength_file(
        output_dir,
        "detector_transport_overview",
        wavelength_nm,
    )
    if overview_path is not None:
        return _accounting_from_legacy_overview(overview_path)

    if n_rays is not None:
        if n_rays <= 0:
            raise ComparisonError("n_rays must be positive")
        return {
            "accounting_source": "explicit n_rays only",
            "accounting_quality": "detector_only",
            "total_ray_count": float(n_rays),
            "escaped_ray_count": None,
            "total_escaped_fraction": float("nan"),
            "absorbed_ray_count": None,
            "total_absorbed_fraction": float("nan"),
            "unclassified_ray_count": None,
            "total_unclassified_fraction": float("nan"),
            "truncated_ray_count": None,
            "truncated_fraction": float("nan"),
            "detected_unique_ray_count": None,
            "total_detected_fraction": float("nan"),
            "total_detected_fraction_is_unique": False,
            "absorption_definition": "unavailable",
        }

    return {
        "accounting_source": "none",
        "accounting_quality": "unavailable",
        "total_ray_count": None,
        "escaped_ray_count": None,
        "total_escaped_fraction": float("nan"),
        "absorbed_ray_count": None,
        "total_absorbed_fraction": float("nan"),
        "unclassified_ray_count": None,
        "total_unclassified_fraction": float("nan"),
        "truncated_ray_count": None,
        "truncated_fraction": float("nan"),
        "detected_unique_ray_count": None,
        "total_detected_fraction": float("nan"),
        "total_detected_fraction_is_unique": False,
        "absorption_definition": "unavailable",
    }


def compare_measured_to_output(
    measured_csv: Path,
    detector_csv: Path,
    output_dir: Path,
    material: Optional[str] = None,
    concentration_g_per_L: Optional[float] = None,
    dataset_column: Optional[str] = None,
    wavelength_nm: Optional[int] = None,
    n_rays: Optional[int] = None,
    detector_acceptance_deg: float = DEFAULT_DETECTOR_ACCEPTANCE_DEG,
    transport_summary: Optional[Mapping[str, object]] = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Compare one production run with one measured dataset."""

    datasets = read_measured_datasets(Path(measured_csv))
    dataset = select_measured_dataset(
        datasets,
        material=material,
        concentration_g_per_L=concentration_g_per_L,
        dataset_column=dataset_column,
    )
    model = read_detector_curve(
        Path(detector_csv),
        wavelength_nm=wavelength_nm,
    )
    resolved_wavelength = model["wavelength_nm"]
    accounting = load_transport_accounting(
        Path(output_dir),
        (
            int(resolved_wavelength)
            if resolved_wavelength is not None
            else wavelength_nm
        ),
        np.asarray(model["angles_deg"], dtype=np.float64),
        n_rays=n_rays,
        detector_acceptance_deg=detector_acceptance_deg,
        transport_summary=transport_summary,
    )

    measured_shape = np.asarray(dataset["shape"], dtype=np.float64)
    model_shape = np.asarray(model["shape"], dtype=np.float64)
    shape_metrics = calculate_shape_metrics(
        measured_shape,
        model_shape,
        np.asarray(model["angles_deg"], dtype=np.float64),
    )

    total_rays = accounting.get("total_ray_count")
    total_rays_float = (
        float(total_rays) if total_rays is not None else None
    )
    assignment_count = float(model["hit_assignment_count"])
    summary_assignment_value = accounting.get(
        "detector_hit_assignment_count_summary"
    )
    summary_assignment_count = (
        float(summary_assignment_value)
        if summary_assignment_value is not None
        else None
    )
    assignment_count_consistent = (
        summary_assignment_count is None
        or math.isclose(
            assignment_count,
            summary_assignment_count,
            rel_tol=0.0,
            abs_tol=0.5,
        )
    )
    assignment_count_difference = (
        assignment_count - summary_assignment_count
        if summary_assignment_count is not None
        else float("nan")
    )
    if (
        summary_assignment_count is not None
        and not assignment_count_consistent
    ):
        print(
            "[WARNING] Detector CSV and transport summary disagree: "
            f"{assignment_count} assignments in detector curve versus "
            f"{summary_assignment_count} in run summary. Shape scoring uses "
            "the detector CSV; unique-ray fractions use the run summary."
        )
    assignment_efficiency = _safe_fraction(
        assignment_count,
        total_rays_float,
    )

    detected_fraction = float(
        accounting.get("total_detected_fraction", float("nan"))
    )
    detected_unique_count = accounting.get("detected_unique_ray_count")
    if not math.isfinite(detected_fraction):
        # Preserve a usable legacy metric, but mark clearly that overlapping
        # assignments may make it differ from unique detected-ray fraction.
        detected_fraction = assignment_efficiency
        detected_unique_count = None

    escaped_count_value = accounting.get("escaped_ray_count")
    escaped_count = (
        float(escaped_count_value)
        if escaped_count_value is not None
        else None
    )
    detector_capture_fraction = _safe_fraction(
        (
            float(detected_unique_count)
            if detected_unique_count is not None
            else assignment_count
        ),
        escaped_count,
    )

    escaped_fraction = float(
        accounting.get("total_escaped_fraction", float("nan"))
    )
    absorbed_fraction = float(
        accounting.get("total_absorbed_fraction", float("nan"))
    )
    unclassified_fraction = float(
        accounting.get("total_unclassified_fraction", float("nan"))
    )
    truncated_fraction = float(
        accounting.get("truncated_fraction", float("nan"))
    )
    finite_terminal_accounting = [
        value
        for value in (
            escaped_fraction,
            absorbed_fraction,
            truncated_fraction,
        )
        if math.isfinite(value)
    ]
    accounting_sum = (
        float(sum(finite_terminal_accounting))
        if len(finite_terminal_accounting) == 3
        else float("nan")
    )
    partition_sum = (
        accounting_sum + unclassified_fraction
        if math.isfinite(accounting_sum)
        and math.isfinite(unclassified_fraction)
        else float("nan")
    )

    metrics: Dict[str, object] = {
        "dataset": dataset["slug"],
        "measured_column": dataset["column"],
        "material": dataset["material"],
        "concentration_g_per_L": dataset["concentration_g_per_L"],
        "wavelength_nm": resolved_wavelength,
        "measured_raw_sum": dataset["raw_sum"],
        "measured_curve_is_unit_sum": dataset["is_unit_sum"],
        "measured_absolute_scale_available": dataset[
            "absolute_scale_available"
        ],
        "model_detector_hit_assignment_count": assignment_count,
        "summary_detector_hit_assignment_count": summary_assignment_count,
        "detector_assignment_count_difference_vs_summary": (
            assignment_count_difference
        ),
        "detector_assignment_count_consistent_with_summary": (
            assignment_count_consistent
        ),
        # This is the sum over per-angle hit/launched efficiencies.  It can
        # exceed the unique detected fraction when angular acceptances overlap.
        "detector_hit_assignment_fraction": assignment_efficiency,
        "detector_hit_assignment_fraction_definition": (
            "detector hit assignments / launched rays"
        ),
        "absolute_detector_efficiency": detected_fraction,
        "absolute_detector_efficiency_definition": (
            "unique detected rays / launched rays"
            if accounting.get("total_detected_fraction_is_unique", False)
            else "detector hit assignments / launched rays (legacy fallback)"
        ),
        "detected_unique_ray_count": detected_unique_count,
        "total_detected_fraction": detected_fraction,
        "total_detected_fraction_is_unique": accounting.get(
            "total_detected_fraction_is_unique",
            False,
        ),
        "total_detected_fraction_definition": (
            "unique detected rays / launched rays"
            if accounting.get("total_detected_fraction_is_unique", False)
            else "detector hit assignments / launched rays (legacy fallback)"
        ),
        "detector_capture_fraction_of_escaped_rays": detector_capture_fraction,
        "total_escaped_fraction": escaped_fraction,
        "total_absorbed_fraction": absorbed_fraction,
        "total_truncated_fraction": truncated_fraction,
        "total_unclassified_fraction": unclassified_fraction,
        "transport_accounting_sum": accounting_sum,
        "transport_partition_sum_including_unclassified": partition_sum,
        "total_ray_count": total_rays,
        "escaped_ray_count": accounting.get("escaped_ray_count"),
        "absorbed_ray_count": accounting.get("absorbed_ray_count"),
        "truncated_ray_count": accounting.get("truncated_ray_count"),
        "unclassified_ray_count": accounting.get("unclassified_ray_count"),
        "accounting_source": accounting.get("accounting_source"),
        "accounting_quality": accounting.get("accounting_quality"),
        "absorption_definition": accounting.get("absorption_definition"),
        "detector_acceptance_deg_for_unique_count": accounting.get(
            "detector_acceptance_deg_for_unique_count"
        ),
        **shape_metrics,
    }

    measured_raw = np.asarray(dataset["raw"], dtype=np.float64)
    model_counts = np.asarray(model["counts"], dtype=np.float64)
    unique_counts_raw = accounting.get("unique_detector_counts")
    unique_counts = (
        np.asarray(unique_counts_raw, dtype=np.float64)
        if unique_counts_raw is not None
        else None
    )

    curve_rows: List[Dict[str, object]] = []
    for index, angle in enumerate(EXPECTED_ANGLES_DEG):
        absolute_per_angle = _safe_fraction(
            float(model_counts[index]),
            total_rays_float,
        )
        row: Dict[str, object] = {
            "detector_angle_deg": int(angle),
            "measured_raw": float(measured_raw[index]),
            "measured_shape_unit_sum": float(measured_shape[index]),
            "model_hit_assignments": float(model_counts[index]),
            "model_shape_unit_sum": float(model_shape[index]),
            "shape_residual_model_minus_measured": float(
                model_shape[index] - measured_shape[index]
            ),
            "model_absolute_detector_efficiency": absolute_per_angle,
        }
        if unique_counts is not None:
            row["model_unique_detector_hits"] = float(unique_counts[index])
            row["model_unique_detected_fraction"] = _safe_fraction(
                float(unique_counts[index]),
                total_rays_float,
            )
        curve_rows.append(row)

    return metrics, curve_rows


def _write_mapping_csv(path: Path, row: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_rows_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    if not rows:
        raise ComparisonError(f"Refusing to write empty CSV: {path}")
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_value(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_measured_comparison(
    measured_csv: Path,
    detector_csv: Path,
    output_dir: Path,
    material: Optional[str] = None,
    concentration_g_per_L: Optional[float] = None,
    dataset_column: Optional[str] = None,
    wavelength_nm: Optional[int] = None,
    n_rays: Optional[int] = None,
    detector_acceptance_deg: float = DEFAULT_DETECTOR_ACCEPTANCE_DEG,
    transport_summary: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Calculate and save per-angle and one-row summary comparison CSVs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, curve_rows = compare_measured_to_output(
        measured_csv=Path(measured_csv),
        detector_csv=Path(detector_csv),
        output_dir=output_dir,
        material=material,
        concentration_g_per_L=concentration_g_per_L,
        dataset_column=dataset_column,
        wavelength_nm=wavelength_nm,
        n_rays=n_rays,
        detector_acceptance_deg=detector_acceptance_deg,
        transport_summary=transport_summary,
    )

    resolved_wavelength = metrics.get("wavelength_nm")
    suffix = (
        f"_{int(resolved_wavelength)}nm"
        if resolved_wavelength is not None
        else ""
    )
    curve_path = output_dir / f"measured_comparison{suffix}.csv"
    metrics_path = output_dir / f"measured_comparison_metrics{suffix}.csv"
    json_path = output_dir / f"measured_comparison_metrics{suffix}.json"

    _write_rows_csv(curve_path, curve_rows)
    _write_mapping_csv(metrics_path, metrics)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {key: _json_value(value) for key, value in metrics.items()},
            handle,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")

    return {
        "metrics": metrics,
        "curve_rows": curve_rows,
        "curve_path": curve_path,
        "metrics_path": metrics_path,
        "json_path": json_path,
    }


def _resolve_measured_path(
    requested: Path,
    output_dir: Path,
) -> Optional[Path]:
    requested = Path(requested).expanduser()
    candidates: List[Path] = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.extend(
            [
                Path.cwd() / requested,
                Path(__file__).resolve().parent / requested,
                output_dir / requested,
            ]
        )
        for parent in (output_dir, *output_dir.parents):
            candidates.append(parent / requested.name)

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _infer_dataset_from_output_dir(
    output_dir: Path,
) -> Tuple[Optional[str], Optional[float]]:
    pattern = re.compile(
        r"(?P<material>[A-Za-z0-9_-]+)_"
        r"(?P<concentration>[0-9]+(?:p[0-9]+)?)gL$",
        flags=re.IGNORECASE,
    )
    for part in reversed(output_dir.parts):
        match = pattern.fullmatch(part)
        if match is None:
            continue
        return (
            match.group("material").lower(),
            float(match.group("concentration").replace("p", ".")),
        )
    return None, None


def save_measured_comparison_if_available(
    outdir: Path = Path("."),
    measured_csv: Path = Path("sediment_data.csv"),
    material: Optional[str] = None,
    concentration_g_per_L: Optional[float] = None,
    wavelength_nm: Optional[int] = None,
    detector_csv: Optional[Path] = None,
    n_rays: Optional[int] = None,
    detector_acceptance_deg: float = DEFAULT_DETECTOR_ACCEPTANCE_DEG,
    dataset_column: Optional[str] = None,
    output_dir: Optional[Path] = None,
    transport_summary: Optional[Mapping[str, object]] = None,
) -> Optional[Dict[str, object]]:
    """
    Production convenience wrapper.

    A missing measured-data file is non-fatal because CLARITAS can be run
    without local measurements.  If the file exists, malformed or ambiguous
    data raise ``ComparisonError`` rather than silently producing a bad score.
    """

    destination = Path(output_dir) if output_dir is not None else Path(outdir)
    destination = destination.expanduser().resolve()
    measured_path = _resolve_measured_path(Path(measured_csv), destination)
    if measured_path is None:
        print(
            "[INFO] Measured comparison skipped: could not find "
            f"{measured_csv}"
        )
        return None

    if detector_csv is None:
        detector_path = destination / "detector_hits.csv"
    else:
        requested_detector = Path(detector_csv).expanduser()
        if requested_detector.is_absolute():
            detector_path = requested_detector
        else:
            detector_candidates = (
                Path.cwd() / requested_detector,
                destination / requested_detector,
                destination / requested_detector.name,
            )
            detector_path = next(
                (
                    candidate.resolve()
                    for candidate in detector_candidates
                    if candidate.is_file()
                ),
                (destination / requested_detector.name).resolve(),
            )
    if not detector_path.is_file():
        raise ComparisonError(f"Detector CSV not found: {detector_path}")

    if dataset_column is None and (
        material is None or concentration_g_per_L is None
    ):
        inferred_material, inferred_concentration = (
            _infer_dataset_from_output_dir(destination)
        )
        material = material or inferred_material
        if concentration_g_per_L is None:
            concentration_g_per_L = inferred_concentration

    if dataset_column is None and (
        material is None
        or material.strip().lower() in {"", "unknown"}
        or concentration_g_per_L is None
    ):
        print(
            "[INFO] Measured comparison skipped: the active material and "
            "concentration do not identify a measured dataset"
        )
        return None

    result = save_measured_comparison(
        measured_csv=measured_path,
        detector_csv=detector_path,
        output_dir=destination,
        material=material,
        concentration_g_per_L=concentration_g_per_L,
        dataset_column=dataset_column,
        wavelength_nm=wavelength_nm,
        n_rays=n_rays,
        detector_acceptance_deg=detector_acceptance_deg,
        transport_summary=transport_summary,
    )
    metrics = result["metrics"]
    print(
        "[INFO] Measured comparison saved: "
        f"RMSE={float(metrics['rmse']):.8g}, "
        f"MAE={float(metrics['mae']):.8g}, "
        f"detected={float(metrics['total_detected_fraction']):.8g}, "
        f"escaped={float(metrics['total_escaped_fraction']):.8g}, "
        f"absorbed={float(metrics['total_absorbed_fraction']):.8g}"
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CLARITAS detector/output data with sediment_data.csv "
            "without discarding absolute simulated ray fractions."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="CLARITAS run output directory.",
    )
    parser.add_argument(
        "--measured",
        type=Path,
        default=Path("sediment_data.csv"),
        help="Measured CSV; default sediment_data.csv.",
    )
    parser.add_argument(
        "--detector-csv",
        type=Path,
        default=None,
        help="Detector CSV; default <output-dir>/detector_hits.csv.",
    )
    parser.add_argument("--material", default=None)
    parser.add_argument("--concentration", type=float, default=None)
    parser.add_argument(
        "--dataset-column",
        default=None,
        help="Measured column name or dataset slug; overrides material selection.",
    )
    parser.add_argument("--wavelength-nm", type=int, default=None)
    parser.add_argument(
        "--n-rays",
        type=int,
        default=None,
        help="Fallback launched-ray count when no transport output exists.",
    )
    parser.add_argument(
        "--detector-acceptance-deg",
        type=float,
        default=DEFAULT_DETECTOR_ACCEPTANCE_DEG,
        help=(
            "Existing production detector half-acceptance used only when "
            "reconstructing unique hits from legacy ray output; default 6.5."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.detector_acceptance_deg <= 0.0:
        parser.error("--detector-acceptance-deg must be positive")
    if args.n_rays is not None and args.n_rays <= 0:
        parser.error("--n-rays must be positive")

    try:
        result = save_measured_comparison_if_available(
            output_dir=args.output_dir,
            measured_csv=args.measured,
            material=args.material,
            concentration_g_per_L=args.concentration,
            wavelength_nm=args.wavelength_nm,
            detector_csv=args.detector_csv,
            n_rays=args.n_rays,
            detector_acceptance_deg=args.detector_acceptance_deg,
            dataset_column=args.dataset_column,
        )
    except (ComparisonError, OSError) as exc:
        parser.error(str(exc))

    if result is None:
        return 2

    print(f"Curve CSV  : {result['curve_path']}")
    print(f"Metrics CSV: {result['metrics_path']}")
    print(f"Metrics JSON: {result['json_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
