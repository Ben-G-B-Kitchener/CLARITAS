#!/usr/bin/env python3
"""
CLARITAS multi-material parameter screening driver.

Features
--------
- Reads measured detector curves from a CSV file.
- Automatically discovers columns named like:
      loess 0.5 measured
      kaolin 4.0 measured
- Runs every parameter case for every discovered material/concentration dataset.
- Uses 100,000 rays by default.
- Selects the newest CLARITAS_*.py source automatically unless --base is given.
- Changes only named top-level assignments using AST source positions.
- Preserves the original CLARITAS source, including CUDA strings.
- Supports --resume.
- Supports explicit GPU chunk bounds for reproducibility validation.
- Supports a floc-disabled, centred/collimated CPU-reference parity mode.
- Saves per-run curves, per-run metrics, a full summary, case-level combined
  scores, rankings, and the best parameter set.

Shape RMSE/MAE use unit-sum curves. Each completed CLARITAS_77 run also
contributes its unnormalised unique detector efficiency and escaped, absorbed,
truncated, and unclassified fractions to the screening summary.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_RAYS = 100_000
EXPECTED_ANGLES = np.arange(0, 180, 10, dtype=np.int32)

PARAMETER_CASES: List[Dict[str, object]] = [
    {
        "name": "baseline",
        "FLOC_COLLISION_LENGTH_M": 250.0e-6,
        "FLOC_FRACTAL_DIMENSION": 2.0,
        "PRIMARY_REFRACTIVE_INDEX_IMAG_K": 0.001,
    },
    {
        "name": "collision_length_100um",
        "FLOC_COLLISION_LENGTH_M": 100.0e-6,
        "FLOC_FRACTAL_DIMENSION": 2.0,
        "PRIMARY_REFRACTIVE_INDEX_IMAG_K": 0.001,
    },
    {
        "name": "collision_length_500um",
        "FLOC_COLLISION_LENGTH_M": 500.0e-6,
        "FLOC_FRACTAL_DIMENSION": 2.0,
        "PRIMARY_REFRACTIVE_INDEX_IMAG_K": 0.001,
    },
    {
        "name": "fractal_dimension_1p8",
        "FLOC_COLLISION_LENGTH_M": 250.0e-6,
        "FLOC_FRACTAL_DIMENSION": 1.8,
        "PRIMARY_REFRACTIVE_INDEX_IMAG_K": 0.001,
    },
    {
        "name": "fractal_dimension_2p4",
        "FLOC_COLLISION_LENGTH_M": 250.0e-6,
        "FLOC_FRACTAL_DIMENSION": 2.4,
        "PRIMARY_REFRACTIVE_INDEX_IMAG_K": 0.001,
    },
    {
        "name": "absorption_k_0p0003",
        "FLOC_COLLISION_LENGTH_M": 250.0e-6,
        "FLOC_FRACTAL_DIMENSION": 2.0,
        "PRIMARY_REFRACTIVE_INDEX_IMAG_K": 0.0003,
    },
    {
        "name": "absorption_k_0p002",
        "FLOC_COLLISION_LENGTH_M": 250.0e-6,
        "FLOC_FRACTAL_DIMENSION": 2.0,
        "PRIMARY_REFRACTIVE_INDEX_IMAG_K": 0.002,
    },
]

# Override here only if a material needs a density different from the CLARITAS
# default. Both current loess and kaolin runs use 2600 kg/m^3.
MATERIAL_DENSITY_KG_PER_M3: Dict[str, float] = {
    "loess": 2600.0,
    "kaolin": 2600.0,
}


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds_i = int(round(seconds))
    hours, remainder = divmod(seconds_i, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def newest_matching_file(
    directory: Path,
    pattern: str,
    excluded_substrings: Sequence[str] = (),
) -> Path:
    candidates = [
        path for path in directory.glob(pattern)
        if path.is_file()
        and not any(token in path.name for token in excluded_substrings)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No files matching {pattern!r} found in {directory}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def claritas_version_key(path: Path) -> Tuple[int, float]:
    match = re.match(r"CLARITAS_(\d+)", path.stem, flags=re.IGNORECASE)
    version = int(match.group(1)) if match else -1
    return version, path.stat().st_mtime


def newest_claritas_file(directory: Path) -> Path:
    candidates = [
        path for path in directory.glob("CLARITAS_*.py")
        if path.is_file()
        and "screen_case" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No CLARITAS_*.py files found in {directory}"
        )
    return max(candidates, key=claritas_version_key)


def read_measured_datasets(
    csv_path: Path,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    # Use csv.DictReader rather than np.genfromtxt(names=True). NumPy sanitises
    # headers such as "loess 0.5 measured" into "loess_05_measured", which
    # incorrectly turns 0.5 into 5.0 when the concentration is parsed.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(f"No CSV headers found in {csv_path}")
    if not rows:
        raise RuntimeError(f"No rows found in measured-data file: {csv_path}")

    angle_candidates = [
        name for name in fieldnames
        if re.fullmatch(
            r"detector[_ ]?angle(?:[_ ]?deg)?",
            name.strip(),
            flags=re.IGNORECASE,
        )
    ]
    if len(angle_candidates) != 1:
        raise RuntimeError(
            "Expected one detector-angle column. Found: "
            f"{angle_candidates or fieldnames}"
        )

    angle_column = angle_candidates[0]

    try:
        angles = np.asarray(
            [float(row[angle_column]) for row in rows],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not parse detector angles from {csv_path}"
        ) from exc

    order = np.argsort(angles)
    angles = angles[order]

    if len(angles) != len(EXPECTED_ANGLES):
        raise RuntimeError(
            f"Expected {len(EXPECTED_ANGLES)} measured angles, "
            f"found {len(angles)}"
        )

    if not np.array_equal(angles.astype(np.int32), EXPECTED_ANGLES):
        raise RuntimeError(
            f"Measured angles must be {EXPECTED_ANGLES.tolist()}, "
            f"found {angles.tolist()}"
        )

    datasets: List[Dict[str, object]] = []
    header_pattern = re.compile(
        r"^(?P<material>[A-Za-z0-9_-]+)[_ ]+"
        r"(?P<concentration>[0-9]+(?:\.[0-9]+)?)"
        r"(?:[_ ]+g(?:_?per_?l|/?l))?"
        r"[_ ]+measured$",
        flags=re.IGNORECASE,
    )

    for column in fieldnames:
        if column == angle_column:
            continue

        clean_column = column.strip()
        match = header_pattern.fullmatch(clean_column)
        if not match:
            raise RuntimeError(
                f"Measured-data column {column!r} does not match "
                "'<material> <concentration> measured'."
            )

        material = match.group("material").lower().replace("-", "_")
        concentration = float(match.group("concentration"))

        try:
            values = np.asarray(
                [float(row[column]) for row in rows],
                dtype=np.float64,
            )[order]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Could not parse measured values from column {column!r}"
            ) from exc

        if len(values) != len(EXPECTED_ANGLES):
            raise RuntimeError(
                f"Column {column!r} has {len(values)} rows; "
                f"expected {len(EXPECTED_ANGLES)}"
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Column {column!r} contains non-finite values"
            )
        if np.any(values < 0):
            raise RuntimeError(
                f"Column {column!r} contains negative values"
            )

        measured_sum = float(np.sum(values))
        if measured_sum <= 0:
            raise RuntimeError(
                f"Column {column!r} has a non-positive sum"
            )

        # Explicitly reject the exact decimal-point-loss failure that affected
        # the previous driver.
        if concentration not in (0.5, 4.0):
            raise RuntimeError(
                f"Unexpected concentration {concentration:g} g/L parsed from "
                f"{column!r}. Expected 0.5 or 4.0 g/L."
            )

        datasets.append({
            "column": clean_column,
            "material": material,
            "concentration_g_per_L": concentration,
            "measured_raw": values,
            "measured_normalised": values / measured_sum,
        })

    if not datasets:
        raise RuntimeError(
            f"No measured response columns were found in {csv_path}"
        )

    return angles, datasets


def source_expression_for_material(
    material: str,
) -> Tuple[str, str]:
    safe_name = material.lower().replace("-", "_")
    diameter_name = f"{safe_name}_diameter"
    weights_name = f"{safe_name}_weights"
    return diameter_name, weights_name


def python_literal(value: object) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return repr(value)
    raise TypeError(f"Unsupported replacement value: {value!r}")


def top_level_assignment_edits(
    source: str,
    constant_replacements: Mapping[str, object],
    expression_replacements: Mapping[str, str],
    filename: str,
) -> List[Tuple[int, int, str]]:
    tree = ast.parse(source, filename=filename)
    lines = source.splitlines(keepends=True)

    requested_names = set(constant_replacements) | set(expression_replacements)
    found = {name: 0 for name in requested_names}
    edits: List[Tuple[int, int, str]] = []

    for node in tree.body:
        target_name = None

        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                target_name = node.target.id

        if target_name not in requested_names:
            continue

        if getattr(node, "end_lineno", None) is None:
            raise RuntimeError(
                "This Python interpreter does not provide AST end-line "
                "positions. Python 3.8+ is required."
            )

        found[target_name] += 1
        source_line = lines[node.lineno - 1]
        indent = source_line[:len(source_line) - len(source_line.lstrip())]

        if target_name in expression_replacements:
            rhs = expression_replacements[target_name]
        else:
            rhs = python_literal(constant_replacements[target_name])

        replacement = f"{indent}{target_name} = {rhs}\n"
        edits.append((node.lineno - 1, node.end_lineno, replacement))

    bad_counts = {
        name: count for name, count in found.items()
        if count != 1
    }
    if bad_counts:
        raise RuntimeError(
            "Expected exactly one top-level assignment for each override. "
            f"Counts: {bad_counts}"
        )

    return edits


def build_case_script(
    base_path: Path,
    destination: Path,
    material: str,
    concentration: float,
    case: Mapping[str, object],
    rays: int,
    gpu_min_chunk_rays: Optional[int] = None,
    gpu_max_chunk_rays: Optional[int] = None,
    reference_parity: bool = False,
) -> None:
    source = base_path.read_text(encoding="utf-8", errors="strict")
    lines = source.splitlines(keepends=True)

    diameter_expression, weights_expression = (
        source_expression_for_material(material)
    )

    constant_replacements: Dict[str, object] = {
        "mass_concentration_g_per_L": float(concentration),
        "FLOC_COLLISION_LENGTH_M": float(
            case["FLOC_COLLISION_LENGTH_M"]
        ),
        "FLOC_FRACTAL_DIMENSION": float(
            case["FLOC_FRACTAL_DIMENSION"]
        ),
        "PRIMARY_REFRACTIVE_INDEX_IMAG_K": float(
            case["PRIMARY_REFRACTIVE_INDEX_IMAG_K"]
        ),
        "N_RAYS": int(rays),
        "OUTDIR": ".",
    }
    if gpu_min_chunk_rays is not None:
        constant_replacements["GPU_MIN_CHUNK_RAYS"] = int(
            gpu_min_chunk_rays
        )
    if gpu_max_chunk_rays is not None:
        constant_replacements["GPU_MAX_CHUNK_RAYS"] = int(
            gpu_max_chunk_rays
        )
    if reference_parity:
        constant_replacements["FLOC_ENABLED"] = False
        constant_replacements["SOURCE_MODE"] = "reference_collimated"

    if material in MATERIAL_DENSITY_KG_PER_M3:
        constant_replacements["particle_density_kg_per_m3"] = float(
            MATERIAL_DENSITY_KG_PER_M3[material]
        )

    expression_replacements = {
        "particle_diameter_m": diameter_expression,
        "particle_weights": weights_expression,
    }

    edits = top_level_assignment_edits(
        source=source,
        constant_replacements=constant_replacements,
        expression_replacements=expression_replacements,
        filename=str(base_path),
    )

    for start, end, replacement in sorted(
        edits,
        key=lambda item: item[0],
        reverse=True,
    ):
        lines[start:end] = [replacement]

    generated = "".join(lines)
    compile(generated, str(destination), "exec")
    destination.write_text(generated, encoding="utf-8")


def read_model_detector_curve(
    detector_csv: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(detector_csv, delimiter=",", names=True)
    names = list(data.dtype.names or [])

    if "Detector_deg" not in names:
        raise RuntimeError(
            f"Detector_deg column missing from {detector_csv}"
        )

    hit_columns = [
        name for name in names
        if name.startswith("H_")
    ]
    if not hit_columns:
        raise RuntimeError(
            f"No H_<wavelength>nm column found in {detector_csv}"
        )

    angles = np.atleast_1d(
        np.asarray(data["Detector_deg"], dtype=np.float64)
    )
    raw = np.atleast_1d(
        np.asarray(data[hit_columns[0]], dtype=np.float64)
    )

    order = np.argsort(angles)
    angles = angles[order]
    raw = raw[order]

    if len(angles) != len(EXPECTED_ANGLES):
        raise RuntimeError(
            f"Expected {len(EXPECTED_ANGLES)} detector bins, "
            f"found {len(angles)}"
        )
    if not np.array_equal(
        angles.astype(np.int32),
        EXPECTED_ANGLES,
    ):
        raise RuntimeError(
            f"Unexpected model angles: {angles.tolist()}"
        )

    total = float(np.sum(raw))
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError(
            f"Invalid model detector sum in {detector_csv}: {total}"
        )

    return angles, raw, raw / total


def calculate_fit_metrics(
    measured: np.ndarray,
    model: np.ndarray,
) -> Dict[str, float]:
    residual = model - measured

    result = {
        "rmse_all": float(np.sqrt(np.mean(residual ** 2))),
        "mae_all": float(np.mean(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "measured_sum": float(np.sum(measured)),
        "model_sum": float(np.sum(model)),
    }

    regions = {
        "forward_0_50": slice(0, 6),
        "middle_60_110": slice(6, 12),
        "rear_120_170": slice(12, 18),
    }

    for label, region in regions.items():
        region_residual = residual[region]
        result[f"rmse_{label}"] = float(
            np.sqrt(np.mean(region_residual ** 2))
        )
        result[f"mae_{label}"] = float(
            np.mean(np.abs(region_residual))
        )

    return result


def read_production_comparison_metrics(
    run_directory: Path,
) -> Dict[str, object]:
    """Read the one-row metrics CSV emitted by CLARITAS_77."""
    candidates = sorted(
        run_directory.glob("measured_comparison_metrics_*nm.csv")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one measured_comparison_metrics_*nm.csv in "
            f"{run_directory}; found {len(candidates)}"
        )
    with candidates[0].open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one metrics row in {candidates[0]}, found {len(rows)}"
        )

    row = rows[0]
    numeric_fields = (
        "absolute_detector_efficiency",
        "total_detected_fraction",
        "detector_hit_assignment_fraction",
        "total_escaped_fraction",
        "total_absorbed_fraction",
        "total_truncated_fraction",
        "total_unclassified_fraction",
        "transport_accounting_sum",
        "transport_partition_sum_including_unclassified",
    )
    parsed: Dict[str, object] = {
        "production_comparison_metrics": str(candidates[0]),
    }
    for field in numeric_fields:
        value = row.get(field, "")
        parsed[field] = (
            float(value) if value not in ("", None) else float("nan")
        )
    parsed["measured_absolute_scale_available"] = (
        str(row.get("measured_absolute_scale_available", "")).strip().lower()
        in {"1", "true", "yes"}
    )
    return parsed


def write_curve_comparison(
    path: Path,
    dataset: Mapping[str, object],
    model_raw: np.ndarray,
    model_normalised: np.ndarray,
) -> None:
    measured_raw = np.asarray(dataset["measured_raw"], dtype=np.float64)
    measured_normalised = np.asarray(
        dataset["measured_normalised"],
        dtype=np.float64,
    )

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "detector_deg",
            "measured_raw",
            "measured_normalised",
            "model_raw_hits",
            "model_normalised",
            "normalised_residual",
        ])

        for angle, mr, mn, rr, rn in zip(
            EXPECTED_ANGLES,
            measured_raw,
            measured_normalised,
            model_raw,
            model_normalised,
        ):
            writer.writerow([
                int(angle),
                float(mr),
                float(mn),
                float(rr),
                float(rn),
                float(rn - mn),
            ])


def write_dict_rows(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    preferred_columns: Sequence[str],
) -> None:
    rows = list(rows)
    if not rows:
        return

    columns: List[str] = []
    seen = set()

    for column in preferred_columns:
        if any(column in row for row in rows):
            columns.append(column)
            seen.add(column)

    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(column)
                seen.add(column)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def dataset_slug(dataset: Mapping[str, object]) -> str:
    material = str(dataset["material"])
    concentration = float(dataset["concentration_g_per_L"])
    concentration_text = (
        f"{concentration:g}".replace(".", "p")
    )
    return f"{material}_{concentration_text}gL"


def run_one_case(
    base_path: Path,
    measured_path: Path,
    python_executable: str,
    output_root: Path,
    dataset: Mapping[str, object],
    case: Mapping[str, object],
    rays: int,
    gpu_min_chunk_rays: Optional[int],
    gpu_max_chunk_rays: Optional[int],
    reference_parity: bool,
    timeout_seconds: int,
    resume: bool,
) -> Dict[str, object]:
    slug = dataset_slug(dataset)
    case_name = str(case["name"])
    run_directory = output_root / case_name / slug
    run_directory.mkdir(parents=True, exist_ok=True)

    result_json = run_directory / "result.json"
    detector_csv = run_directory / "detector_hits.csv"
    log_file = run_directory / "run.log"
    case_script = run_directory / "CLARITAS_screen_case.py"

    comparison_metric_files = list(
        run_directory.glob("measured_comparison_metrics_*nm.csv")
    )
    run_summary_files = list(run_directory.glob("run_summary_*nm.csv"))
    if (
        resume
        and result_json.exists()
        and detector_csv.exists()
        and len(comparison_metric_files) == 1
        and len(run_summary_files) == 1
    ):
        with result_json.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        resume_matches = (
            result.get("dataset") == slug
            and result.get("case") == case_name
            and int(result.get("N_RAYS", -1)) == int(rays)
            and result.get("GPU_MIN_CHUNK_RAYS") == gpu_min_chunk_rays
            and result.get("GPU_MAX_CHUNK_RAYS") == gpu_max_chunk_rays
            and bool(result.get("REFERENCE_PARITY", False))
            == bool(reference_parity)
        )
        if resume_matches:
            result["status"] = "cached"
            return result

    build_case_script(
        base_path=base_path,
        destination=case_script,
        material=str(dataset["material"]),
        concentration=float(dataset["concentration_g_per_L"]),
        case=case,
        rays=rays,
        gpu_min_chunk_rays=gpu_min_chunk_rays,
        gpu_max_chunk_rays=gpu_max_chunk_rays,
        reference_parity=reference_parity,
    )
    shutil.copy2(measured_path, run_directory / "sediment_data.csv")

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONUNBUFFERED"] = "1"
    source_directory = str(base_path.parent.resolve())
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        source_directory
        if not existing_pythonpath
        else source_directory + os.pathsep + existing_pythonpath
    )

    started = time.time()

    try:
        with log_file.open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                [python_executable, case_script.name],
                cwd=run_directory,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=(
                    timeout_seconds
                    if timeout_seconds > 0
                    else None
                ),
                check=False,
            )
    except subprocess.TimeoutExpired:
        result = {
            "status": "timeout",
            "dataset": slug,
            "material": dataset["material"],
            "concentration_g_per_L": dataset[
                "concentration_g_per_L"
            ],
            "case": case_name,
            "N_RAYS": rays,
            "GPU_MIN_CHUNK_RAYS": gpu_min_chunk_rays,
            "GPU_MAX_CHUNK_RAYS": gpu_max_chunk_rays,
            "REFERENCE_PARITY": reference_parity,
            "elapsed_s": time.time() - started,
            **case,
        }
        with result_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        return result

    elapsed = time.time() - started

    if completed.returncode != 0:
        result = {
            "status": "failed",
            "returncode": completed.returncode,
            "dataset": slug,
            "material": dataset["material"],
            "concentration_g_per_L": dataset[
                "concentration_g_per_L"
            ],
            "case": case_name,
            "N_RAYS": rays,
            "GPU_MIN_CHUNK_RAYS": gpu_min_chunk_rays,
            "GPU_MAX_CHUNK_RAYS": gpu_max_chunk_rays,
            "REFERENCE_PARITY": reference_parity,
            "elapsed_s": elapsed,
            "log": str(log_file),
            **case,
        }
        with result_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        return result

    _, model_raw, model_normalised = read_model_detector_curve(
        detector_csv
    )
    measured_normalised = np.asarray(
        dataset["measured_normalised"],
        dtype=np.float64,
    )
    production_metrics = read_production_comparison_metrics(run_directory)

    result = {
        "status": "completed",
        "dataset": slug,
        "measured_column": dataset["column"],
        "material": dataset["material"],
        "concentration_g_per_L": dataset[
            "concentration_g_per_L"
        ],
        "case": case_name,
        "N_RAYS": rays,
        "GPU_MIN_CHUNK_RAYS": gpu_min_chunk_rays,
        "GPU_MAX_CHUNK_RAYS": gpu_max_chunk_rays,
        "REFERENCE_PARITY": reference_parity,
        "elapsed_s": elapsed,
        **case,
        **calculate_fit_metrics(
            measured_normalised,
            model_normalised,
        ),
        **production_metrics,
    }

    write_curve_comparison(
        run_directory / "measured_vs_model.csv",
        dataset,
        model_raw,
        model_normalised,
    )

    with result_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return result


def aggregate_case_scores(
    run_rows: Sequence[Mapping[str, object]],
    case_names: Sequence[str],
    expected_dataset_count: int,
) -> List[Dict[str, object]]:
    aggregated: List[Dict[str, object]] = []

    for case_name in case_names:
        rows = [
            row for row in run_rows
            if row.get("case") == case_name
            and row.get("status") in {"completed", "cached"}
            and "rmse_all" in row
        ]

        if not rows:
            continue

        squared_rmse = np.array(
            [float(row["rmse_all"]) ** 2 for row in rows],
            dtype=np.float64,
        )

        combined_rmse = float(
            np.sqrt(np.mean(squared_rmse))
        )

        first = rows[0]
        aggregated.append({
            "case": case_name,
            "datasets_completed": len(rows),
            "datasets_expected": expected_dataset_count,
            "complete_case": len(rows) == expected_dataset_count,
            "combined_rmse": combined_rmse,
            "mean_rmse": float(np.mean([
                float(row["rmse_all"]) for row in rows
            ])),
            "worst_dataset_rmse": float(np.max([
                float(row["rmse_all"]) for row in rows
            ])),
            "mean_elapsed_s": float(np.mean([
                float(row.get("elapsed_s", 0.0)) for row in rows
            ])),
            "FLOC_COLLISION_LENGTH_M": first[
                "FLOC_COLLISION_LENGTH_M"
            ],
            "FLOC_FRACTAL_DIMENSION": first[
                "FLOC_FRACTAL_DIMENSION"
            ],
            "PRIMARY_REFRACTIVE_INDEX_IMAG_K": first[
                "PRIMARY_REFRACTIVE_INDEX_IMAG_K"
            ],
        })

    aggregated.sort(
        key=lambda row: (
            not bool(row["complete_case"]),
            float(row["combined_rmse"]),
        )
    )

    for rank, row in enumerate(aggregated, start=1):
        row["rank"] = rank

    return aggregated


def write_best_parameter_file(
    path: Path,
    case_scores: Sequence[Mapping[str, object]],
) -> None:
    complete = [
        row for row in case_scores
        if bool(row.get("complete_case"))
    ]
    if not complete:
        return

    best = min(
        complete,
        key=lambda row: float(row["combined_rmse"]),
    )

    content = (
        "BEST CLARITAS PARAMETER SCREEN RESULT\n"
        "=====================================\n\n"
        f"Case: {best['case']}\n"
        f"Combined RMSE: {float(best['combined_rmse']):.12g}\n"
        f"Worst dataset RMSE: "
        f"{float(best['worst_dataset_rmse']):.12g}\n\n"
        "Parameters:\n"
        f"FLOC_COLLISION_LENGTH_M = "
        f"{best['FLOC_COLLISION_LENGTH_M']!r}\n"
        f"FLOC_FRACTAL_DIMENSION = "
        f"{best['FLOC_FRACTAL_DIMENSION']!r}\n"
        f"PRIMARY_REFRACTIVE_INDEX_IMAG_K = "
        f"{best['PRIMARY_REFRACTIVE_INDEX_IMAG_K']!r}\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run CLARITAS parameter screening against all measured "
            "sediment datasets in a CSV file."
        )
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help=(
            "Base CLARITAS script. If omitted, the highest-version "
            "CLARITAS_*.py in the current directory is used."
        ),
    )
    parser.add_argument(
        "--measured",
        type=Path,
        default=None,
        help=(
            "Measured-data CSV. If omitted, the newest "
            "sediment_data*.csv in the current directory is used."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("claritas_parameter_screen"),
        help="Output root directory.",
    )
    parser.add_argument(
        "--rays",
        type=int,
        default=DEFAULT_RAYS,
        help=f"Rays per run; default {DEFAULT_RAYS:,}.",
    )
    parser.add_argument(
        "--gpu-min-chunk-rays",
        type=int,
        default=None,
        help=(
            "Override CLARITAS GPU_MIN_CHUNK_RAYS in generated cases. "
            "Supply together with --gpu-max-chunk-rays."
        ),
    )
    parser.add_argument(
        "--gpu-max-chunk-rays",
        type=int,
        default=None,
        help=(
            "Override CLARITAS GPU_MAX_CHUNK_RAYS in generated cases. "
            "Supply together with --gpu-min-chunk-rays."
        ),
    )
    parser.add_argument(
        "--reference-parity",
        action="store_true",
        help=(
            "Generate primary-only cases with the centred, collimated CPU "
            "reference source (FLOC_ENABLED=False)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-run timeout in seconds; zero disables timeout.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for each CLARITAS run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse runs with result, detector, measured-comparison, and "
            "canonical run-summary outputs."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only this case. May be supplied more than once.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Run only this dataset slug, e.g. loess_0p5gL. "
            "May be supplied more than once."
        ),
    )
    args = parser.parse_args()

    if args.rays < 1_000:
        parser.error("--rays must be at least 1,000")
    if (args.gpu_min_chunk_rays is None) != (
        args.gpu_max_chunk_rays is None
    ):
        parser.error(
            "--gpu-min-chunk-rays and --gpu-max-chunk-rays must be supplied "
            "together"
        )
    if args.gpu_min_chunk_rays is not None:
        if args.gpu_min_chunk_rays < 1:
            parser.error("--gpu-min-chunk-rays must be positive")
        if args.gpu_max_chunk_rays < args.gpu_min_chunk_rays:
            parser.error(
                "--gpu-max-chunk-rays must be greater than or equal to "
                "--gpu-min-chunk-rays"
            )
    current_directory = Path.cwd()

    try:
        base_path = (
            args.base.expanduser().resolve()
            if args.base is not None
            else newest_claritas_file(current_directory)
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    try:
        measured_path = (
            args.measured.expanduser().resolve()
            if args.measured is not None
            else newest_matching_file(
                current_directory,
                "sediment_data*.csv",
            )
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if not base_path.exists():
        parser.error(f"Base CLARITAS file not found: {base_path}")
    if not measured_path.exists():
        parser.error(f"Measured-data file not found: {measured_path}")

    _, datasets = read_measured_datasets(measured_path)

    if args.dataset:
        requested_datasets = set(args.dataset)
        datasets = [
            dataset for dataset in datasets
            if dataset_slug(dataset) in requested_datasets
        ]
        found = {dataset_slug(dataset) for dataset in datasets}
        missing = requested_datasets - found
        if missing:
            parser.error(
                "Unknown dataset slug(s): " + ", ".join(sorted(missing))
            )

    cases = PARAMETER_CASES
    if args.case:
        requested_cases = set(args.case)
        cases = [
            case for case in cases
            if str(case["name"]) in requested_cases
        ]
        found_cases = {str(case["name"]) for case in cases}
        missing_cases = requested_cases - found_cases
        if missing_cases:
            parser.error(
                "Unknown case name(s): "
                + ", ".join(sorted(missing_cases))
            )

    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total_runs = len(cases) * len(datasets)
    if total_runs == 0:
        parser.error("No runs selected")

    print(f"Base CLARITAS file : {base_path}")
    print(f"Measured-data file : {measured_path}")
    print(f"Output directory   : {output_root}")
    print(f"Rays per run       : {args.rays:,}")
    if args.gpu_min_chunk_rays is not None:
        print(
            "GPU chunk override : "
            f"{args.gpu_min_chunk_rays:,} to "
            f"{args.gpu_max_chunk_rays:,} rays"
        )
    if args.reference_parity:
        print(
            "Reference parity    : centred/collimated source, flocs disabled"
        )
    print(
        f"Runs               : {len(cases)} cases × "
        f"{len(datasets)} datasets = {total_runs}"
    )
    print("Datasets:")
    for dataset in datasets:
        print(
            f"  {dataset_slug(dataset):16s} <- "
            f"{dataset['column']}"
        )

    run_rows: List[Dict[str, object]] = []
    completed_durations: List[float] = []
    overall_started = time.time()
    run_number = 0

    preferred_run_columns = [
        "status",
        "dataset",
        "measured_column",
        "material",
        "concentration_g_per_L",
        "case",
        "N_RAYS",
        "GPU_MIN_CHUNK_RAYS",
        "GPU_MAX_CHUNK_RAYS",
        "REFERENCE_PARITY",
        "elapsed_s",
        "FLOC_COLLISION_LENGTH_M",
        "FLOC_FRACTAL_DIMENSION",
        "PRIMARY_REFRACTIVE_INDEX_IMAG_K",
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
        "measured_absolute_scale_available",
        "max_abs_error",
        "returncode",
        "error",
        "log",
    ]

    for case in cases:
        for dataset in datasets:
            run_number += 1
            remaining_before = total_runs - run_number + 1

            if completed_durations:
                estimated_remaining = (
                    float(np.mean(completed_durations))
                    * remaining_before
                )
            else:
                estimated_remaining = float("nan")

            print("\n" + "=" * 72)
            print(f"Run {run_number} / {total_runs}")
            print(f"Material      : {dataset['material']}")
            print(
                f"Concentration : "
                f"{float(dataset['concentration_g_per_L']):g} g/L"
            )
            print(f"Case          : {case['name']}")
            print(
                f"Elapsed total : "
                f"{format_duration(time.time() - overall_started)}"
            )
            print(
                f"Estimated left: "
                f"{format_duration(estimated_remaining)}"
            )
            print("=" * 72)

            try:
                result = run_one_case(
                    base_path=base_path,
                    measured_path=measured_path,
                    python_executable=args.python,
                    output_root=output_root,
                    dataset=dataset,
                    case=case,
                    rays=int(args.rays),
                    gpu_min_chunk_rays=args.gpu_min_chunk_rays,
                    gpu_max_chunk_rays=args.gpu_max_chunk_rays,
                    reference_parity=bool(args.reference_parity),
                    timeout_seconds=int(args.timeout),
                    resume=bool(args.resume),
                )
            except Exception as exc:
                result = {
                    "status": "error",
                    "dataset": dataset_slug(dataset),
                    "measured_column": dataset["column"],
                    "material": dataset["material"],
                    "concentration_g_per_L": dataset[
                        "concentration_g_per_L"
                    ],
                    "case": case["name"],
                    "N_RAYS": int(args.rays),
                    "GPU_MIN_CHUNK_RAYS": args.gpu_min_chunk_rays,
                    "GPU_MAX_CHUNK_RAYS": args.gpu_max_chunk_rays,
                    "REFERENCE_PARITY": bool(args.reference_parity),
                    "error": repr(exc),
                    **case,
                }

            run_rows.append(result)

            if "elapsed_s" in result:
                completed_durations.append(
                    float(result["elapsed_s"])
                )

            status = result.get("status")
            if status in {"completed", "cached"}:
                print(
                    f"Status        : {status}\n"
                    f"RMSE          : "
                    f"{float(result['rmse_all']):.8g}\n"
                    f"Run time      : "
                    f"{format_duration(float(result.get('elapsed_s', 0.0)))}"
                )
            else:
                print(f"Status        : {status}")
                if "log" in result:
                    print(f"Log           : {result['log']}")
                if "error" in result:
                    print(f"Error         : {result['error']}")

            write_dict_rows(
                output_root / "screening_runs.csv",
                run_rows,
                preferred_run_columns,
            )

            case_scores = aggregate_case_scores(
                run_rows,
                [str(case_item["name"]) for case_item in cases],
                expected_dataset_count=len(datasets),
            )
            write_dict_rows(
                output_root / "case_rankings.csv",
                case_scores,
                [
                    "rank",
                    "case",
                    "complete_case",
                    "datasets_completed",
                    "datasets_expected",
                    "combined_rmse",
                    "mean_rmse",
                    "worst_dataset_rmse",
                    "mean_elapsed_s",
                    "FLOC_COLLISION_LENGTH_M",
                    "FLOC_FRACTAL_DIMENSION",
                    "PRIMARY_REFRACTIVE_INDEX_IMAG_K",
                ],
            )
            write_best_parameter_file(
                output_root / "best_parameter_set.txt",
                case_scores,
            )

    final_scores = aggregate_case_scores(
        run_rows,
        [str(case["name"]) for case in cases],
        expected_dataset_count=len(datasets),
    )

    print("\n" + "=" * 72)
    print("SCREEN COMPLETE")
    print("=" * 72)
    print(
        f"Total elapsed: "
        f"{format_duration(time.time() - overall_started)}"
    )
    print(f"Run summary  : {output_root / 'screening_runs.csv'}")
    print(f"Case ranking : {output_root / 'case_rankings.csv'}")

    complete_scores = [
        row for row in final_scores
        if bool(row.get("complete_case"))
    ]
    if complete_scores:
        best = complete_scores[0]
        print(
            f"Best case    : {best['case']}\n"
            f"Combined RMSE: "
            f"{float(best['combined_rmse']):.8g}\n"
            f"Best params  : "
            f"{output_root / 'best_parameter_set.txt'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
