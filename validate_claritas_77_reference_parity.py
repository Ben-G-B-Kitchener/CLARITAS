#!/usr/bin/env python3
"""Statistical CPU-reference parity checks for a CLARITAS 77 CUDA run.

This is a validation-only program.  It deliberately delegates all optical
construction, CPU transport, direction rotation, and detector assignment to
the existing ``claritas_reference`` package.

Monte Carlo comparisons use

    tolerance = z_score * two_sample_standard_error + numerical_floor

The default five-standard-error interval is intentionally conservative.  The
default dimensionless numerical floor (5e-4) prevents zero estimated variance
and allows for small CPU-double versus CUDA-float effects; it is not a fitted
physics parameter.  Direction normalization is instead checked against its own
strict absolute tolerance.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np

from claritas_reference.config import (
    MATERIALS,
    DetectorConfig,
    SimulationConfig,
)
from claritas_reference.geometry import assign_detector_bins
from claritas_reference.physics import build_primary_medium
from claritas_reference.transport import SimulationResult, simulate


STATE_EXITED = 1
STATE_ABSORBED = 2
STATE_TRUNCATED = 3
STATE_MISSED_SAMPLE = 4

REQUIRED_DATASETS = (
    "exit_x",
    "exit_y",
    "exit_z",
    "exit_vx",
    "exit_vy",
    "exit_vz",
    "scatter_count",
    "absorbed",
    "truncated",
    "terminal_state",
)

CSV_FIELDS = (
    "section",
    "metric",
    "gpu_value",
    "cpu_value",
    "difference",
    "standard_error",
    "z_score",
    "numerical_floor",
    "tolerance",
    "passed",
    "method",
    "notes",
)


@dataclass
class RunningMoments:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        if not np.all(np.isfinite(values)):
            raise ValueError("RunningMoments received non-finite values")
        self.count += int(values.size)
        self.total += float(np.sum(values, dtype=np.float64))
        self.total_square += float(
            np.sum(values * values, dtype=np.float64)
        )

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else math.nan

    @property
    def sample_variance(self) -> float:
        if self.count < 2:
            return 0.0
        numerator = self.total_square - self.total * self.total / self.count
        return max(numerator / (self.count - 1), 0.0)


@dataclass
class TransportSummary:
    detector_count: int
    n_rays: int = 0
    terminal_counts: Dict[int, int] = field(
        default_factory=lambda: {
            STATE_EXITED: 0,
            STATE_ABSORBED: 0,
            STATE_TRUNCATED: 0,
            STATE_MISSED_SAMPLE: 0,
        }
    )
    unknown_terminal_count: int = 0
    ballistic_exit_count: int = 0
    single_exit_count: int = 0
    multiple_exit_count: int = 0
    invalid_scatter_count: int = 0
    absorbed_flag_mismatch_count: int = 0
    truncated_flag_mismatch_count: int = 0
    nonfinite_exit_position_count: int = 0
    nonfinite_exit_direction_count: int = 0
    detector_counts: np.ndarray = field(init=False)
    scatter_count_moments: RunningMoments = field(default_factory=RunningMoments)
    exit_vx_moments: RunningMoments = field(default_factory=RunningMoments)
    exit_vz_moments: RunningMoments = field(default_factory=RunningMoments)
    transverse_square_difference_moments: RunningMoments = field(
        default_factory=RunningMoments
    )
    direction_norm_error_moments: RunningMoments = field(
        default_factory=RunningMoments
    )
    maximum_direction_norm_error: float = 0.0

    def __post_init__(self) -> None:
        self.detector_counts = np.zeros(self.detector_count, dtype=np.int64)

    @property
    def exited_count(self) -> int:
        return self.terminal_counts[STATE_EXITED]

    @property
    def detected_count(self) -> int:
        return int(np.sum(self.detector_counts, dtype=np.int64))

    def fraction(self, count: int) -> float:
        return count / self.n_rays if self.n_rays else math.nan

    def terminal_fraction(self, state: int) -> float:
        return self.fraction(self.terminal_counts[state])


@dataclass(frozen=True)
class RunMetadata:
    n_rays: int
    seed: int
    material: str
    concentration_kg_m3: float
    wavelength_m: float
    sample_radius_m: float
    max_events: int
    mu_t_m_inv: float


def _attr_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _attr_bool(value: object, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = _attr_text(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"HDF5 attribute {name!r} is not boolean: {value!r}")


def _required_attr(handle: h5py.File, name: str) -> object:
    if name not in handle.attrs:
        raise ValueError(f"HDF5 file is missing required attribute {name!r}")
    return handle.attrs[name]


def _read_metadata(handle: h5py.File) -> RunMetadata:
    source_model = _attr_text(_required_attr(handle, "source_model"))
    if source_model != "reference_collimated":
        raise ValueError(
            "Reference parity requires source_model='reference_collimated'; "
            f"found {source_model!r}"
        )
    if _attr_bool(_required_attr(handle, "floc_enabled"), "floc_enabled"):
        raise ValueError("Reference parity requires floc_enabled=false")

    expected_text_attributes = {
        "transport_geometry": "3d_sphere",
        "detector_geometry": "ideal_annular_nearest_accepted_band",
        "phase_function_measure": "solid_angle_I_sin_theta",
        "psd_weight_mode": "mass_fraction",
    }
    for name, expected in expected_text_attributes.items():
        observed = _attr_text(_required_attr(handle, name))
        if observed != expected:
            raise ValueError(
                f"Reference parity requires {name}={expected!r}; "
                f"found {observed!r}"
            )

    material = _attr_text(_required_attr(handle, "material")).strip().lower()
    if material not in MATERIALS:
        raise ValueError(
            f"Unsupported material {material!r}; expected one of "
            f"{sorted(MATERIALS)}"
        )

    n_rays = int(_required_attr(handle, "n_rays"))
    seed = int(_required_attr(handle, "simulation_seed"))
    concentration = float(_required_attr(handle, "concentration_g_per_L"))
    wavelength = float(_required_attr(handle, "wavelength_m"))
    radius = float(_required_attr(handle, "sample_radius_m"))
    max_events = int(_required_attr(handle, "max_extinctions"))
    mu_t = float(_required_attr(handle, "mu_t_m_inv"))

    if n_rays <= 0:
        raise ValueError("n_rays must be positive")
    if not np.isfinite(concentration) or concentration < 0.0:
        raise ValueError("concentration_g_per_L must be finite and nonnegative")
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength_m must be finite and positive")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("sample_radius_m must be finite and positive")
    if max_events <= 0:
        raise ValueError("max_extinctions must be positive")
    if not np.isfinite(mu_t) or mu_t < 0.0:
        raise ValueError("mu_t_m_inv must be finite and nonnegative")

    missing = [name for name in REQUIRED_DATASETS if name not in handle]
    if missing:
        raise ValueError(f"HDF5 file is missing datasets: {missing}")
    lengths = {len(handle[name]) for name in REQUIRED_DATASETS}
    if lengths != {n_rays}:
        raise ValueError(
            "Per-ray dataset lengths must all equal the n_rays attribute; "
            f"found lengths {sorted(lengths)} and n_rays={n_rays}"
        )

    return RunMetadata(
        n_rays=n_rays,
        seed=seed,
        material=material,
        concentration_kg_m3=concentration,
        wavelength_m=wavelength,
        sample_radius_m=radius,
        max_events=max_events,
        mu_t_m_inv=mu_t,
    )


def _detector_config(
    handle: h5py.File,
    centres_override: Sequence[float] | None,
    acceptance_override: float | None,
) -> Tuple[DetectorConfig, str]:
    default = DetectorConfig()
    if centres_override is not None:
        centres = np.asarray(centres_override, dtype=np.float64)
        centres_source = "command line"
    elif "detector_centres_deg" in handle.attrs:
        centres = np.asarray(
            handle.attrs["detector_centres_deg"], dtype=np.float64
        )
        centres_source = "HDF5"
    else:
        centres = np.asarray(default.centres_deg, dtype=np.float64)
        centres_source = "claritas_reference default"

    if acceptance_override is not None:
        acceptance = float(acceptance_override)
        acceptance_source = "command line"
    elif "detector_acceptance_half_angle_deg" in handle.attrs:
        acceptance = float(
            handle.attrs["detector_acceptance_half_angle_deg"]
        )
        acceptance_source = "HDF5"
    else:
        acceptance = float(default.acceptance_half_angle_deg)
        acceptance_source = "claritas_reference default"

    detector = DetectorConfig(
        centres_deg=centres,
        acceptance_half_angle_deg=acceptance,
    )
    provenance = (
        f"centres from {centres_source}; acceptance from {acceptance_source}"
    )
    return detector, provenance


def _update_summary(
    summary: TransportSummary,
    detector: DetectorConfig,
    terminal_state: np.ndarray,
    scatter_count: np.ndarray,
    absorbed_flag: np.ndarray,
    truncated_flag: np.ndarray,
    exit_x: np.ndarray,
    exit_y: np.ndarray,
    exit_z: np.ndarray,
    exit_vx: np.ndarray,
    exit_vy: np.ndarray,
    exit_vz: np.ndarray,
) -> None:
    arrays = (
        terminal_state,
        scatter_count,
        absorbed_flag,
        truncated_flag,
        exit_x,
        exit_y,
        exit_z,
        exit_vx,
        exit_vy,
        exit_vz,
    )
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent per-ray chunk lengths: {lengths}")
    chunk_count = len(terminal_state)
    summary.n_rays += chunk_count

    state = np.asarray(terminal_state, dtype=np.int32)
    scatter = np.asarray(scatter_count, dtype=np.int64)
    absorbed = np.asarray(absorbed_flag, dtype=bool)
    truncated = np.asarray(truncated_flag, dtype=bool)

    known = np.zeros(chunk_count, dtype=bool)
    for code in (
        STATE_EXITED,
        STATE_ABSORBED,
        STATE_TRUNCATED,
        STATE_MISSED_SAMPLE,
    ):
        mask = state == code
        summary.terminal_counts[code] += int(np.count_nonzero(mask))
        known |= mask
    summary.unknown_terminal_count += int(np.count_nonzero(~known))
    summary.absorbed_flag_mismatch_count += int(
        np.count_nonzero(absorbed != (state == STATE_ABSORBED))
    )
    summary.truncated_flag_mismatch_count += int(
        np.count_nonzero(truncated != (state == STATE_TRUNCATED))
    )

    valid_scatter = scatter >= 0
    summary.invalid_scatter_count += int(np.count_nonzero(~valid_scatter))
    summary.scatter_count_moments.update(
        scatter[valid_scatter].astype(np.float64)
    )

    exited = state == STATE_EXITED
    summary.ballistic_exit_count += int(
        np.count_nonzero(exited & (scatter == 0))
    )
    summary.single_exit_count += int(
        np.count_nonzero(exited & (scatter == 1))
    )
    summary.multiple_exit_count += int(
        np.count_nonzero(exited & (scatter >= 2))
    )

    positions = np.column_stack((exit_x[exited], exit_y[exited], exit_z[exited]))
    finite_positions = np.all(np.isfinite(positions), axis=1)
    summary.nonfinite_exit_position_count += int(
        np.count_nonzero(~finite_positions)
    )
    if np.any(finite_positions):
        detector_index = assign_detector_bins(
            np.asarray(positions[finite_positions], dtype=np.float64),
            detector,
        )
        summary.detector_counts += np.bincount(
            detector_index[detector_index >= 0],
            minlength=summary.detector_count,
        ).astype(np.int64)

    directions = np.column_stack(
        (exit_vx[exited], exit_vy[exited], exit_vz[exited])
    ).astype(np.float64)
    finite_directions = np.all(np.isfinite(directions), axis=1)
    summary.nonfinite_exit_direction_count += int(
        np.count_nonzero(~finite_directions)
    )
    if np.any(finite_directions):
        valid_directions = directions[finite_directions]
        vx = valid_directions[:, 0]
        vz = valid_directions[:, 2]
        norms = np.linalg.norm(valid_directions, axis=1)
        norm_error = np.abs(norms - 1.0)
        summary.exit_vx_moments.update(vx)
        summary.exit_vz_moments.update(vz)
        summary.transverse_square_difference_moments.update(vx * vx - vz * vz)
        summary.direction_norm_error_moments.update(norm_error)
        summary.maximum_direction_norm_error = max(
            summary.maximum_direction_norm_error,
            float(np.max(norm_error)),
        )


def _summarize_gpu(
    handle: h5py.File,
    metadata: RunMetadata,
    detector: DetectorConfig,
    chunk_rays: int,
) -> TransportSummary:
    summary = TransportSummary(len(detector.centres_deg))
    for start in range(0, metadata.n_rays, chunk_rays):
        stop = min(start + chunk_rays, metadata.n_rays)
        values = {
            name: handle[name][start:stop] for name in REQUIRED_DATASETS
        }
        _update_summary(
            summary,
            detector,
            values["terminal_state"],
            values["scatter_count"],
            values["absorbed"],
            values["truncated"],
            values["exit_x"],
            values["exit_y"],
            values["exit_z"],
            values["exit_vx"],
            values["exit_vy"],
            values["exit_vz"],
        )
    return summary


def _summarize_cpu(
    result: SimulationResult,
    detector: DetectorConfig,
) -> TransportSummary:
    state = np.full(len(result.absorbed), STATE_EXITED, dtype=np.int32)
    state[result.absorbed] = STATE_ABSORBED
    state[result.truncated] = STATE_TRUNCATED
    summary = TransportSummary(len(detector.centres_deg))
    _update_summary(
        summary,
        detector,
        state,
        result.scatter_count,
        result.absorbed,
        result.truncated,
        result.exit_positions_m[:, 0],
        result.exit_positions_m[:, 1],
        result.exit_positions_m[:, 2],
        result.exit_directions[:, 0],
        result.exit_directions[:, 1],
        result.exit_directions[:, 2],
    )
    return summary


def _mean_standard_error(
    first: RunningMoments,
    second: RunningMoments,
) -> float:
    variance = 0.0
    if first.count:
        variance += first.sample_variance / first.count
    if second.count:
        variance += second.sample_variance / second.count
    return math.sqrt(max(variance, 0.0))


def _proportion_standard_error(
    first_count: int,
    first_n: int,
    second_count: int,
    second_n: int,
) -> float:
    def variance_of_mean(count: int, n: int) -> float:
        if n <= 0:
            return math.inf
        probability = count / n
        return probability * (1.0 - probability) / n

    return math.sqrt(
        variance_of_mean(first_count, first_n)
        + variance_of_mean(second_count, second_n)
    )


def _metric_row(
    section: str,
    metric: str,
    gpu_value: float,
    cpu_value: float,
    standard_error: float,
    z_score: float,
    numerical_floor: float,
    method: str,
    notes: str = "",
    tolerance_override: float | None = None,
    pass_override: bool | None = None,
) -> Dict[str, object]:
    difference = float(gpu_value - cpu_value)
    tolerance = (
        float(tolerance_override)
        if tolerance_override is not None
        else float(z_score * standard_error + numerical_floor)
    )
    passed = (
        bool(pass_override)
        if pass_override is not None
        else bool(
            np.isfinite(difference)
            and np.isfinite(tolerance)
            and abs(difference) <= tolerance
        )
    )
    return {
        "section": section,
        "metric": metric,
        "gpu_value": gpu_value,
        "cpu_value": cpu_value,
        "difference": difference,
        "standard_error": standard_error,
        "z_score": z_score,
        "numerical_floor": numerical_floor,
        "tolerance": tolerance,
        "passed": passed,
        "method": method,
        "notes": notes,
    }


def _proportion_row(
    section: str,
    metric: str,
    gpu_count: int,
    gpu_n: int,
    cpu_count: int,
    cpu_n: int,
    z_score: float,
    numerical_floor: float,
    notes: str = "",
) -> Dict[str, object]:
    gpu_value = gpu_count / gpu_n
    cpu_value = cpu_count / cpu_n
    standard_error = _proportion_standard_error(
        gpu_count, gpu_n, cpu_count, cpu_n
    )
    return _metric_row(
        section,
        metric,
        gpu_value,
        cpu_value,
        standard_error,
        z_score,
        numerical_floor,
        "two-sample Bernoulli standard error",
        notes,
    )


def _mean_row(
    section: str,
    metric: str,
    gpu: RunningMoments,
    cpu: RunningMoments,
    z_score: float,
    numerical_floor: float,
    notes: str = "",
) -> Dict[str, object]:
    if gpu.count == 0 or cpu.count == 0:
        return _metric_row(
            section,
            metric,
            gpu.mean,
            cpu.mean,
            math.inf,
            z_score,
            numerical_floor,
            "two-sample mean standard error",
            notes,
            pass_override=False,
        )
    return _metric_row(
        section,
        metric,
        gpu.mean,
        cpu.mean,
        _mean_standard_error(gpu, cpu),
        z_score,
        numerical_floor,
        "two-sample mean standard error",
        notes,
    )


def _symmetry_row(
    label: str,
    moments: RunningMoments,
    z_score: float,
    numerical_floor: float,
    implementation: str,
) -> Dict[str, object]:
    standard_error = (
        math.sqrt(moments.sample_variance / moments.count)
        if moments.count
        else math.inf
    )
    return _metric_row(
        "transverse_symmetry",
        f"{implementation}_{label}",
        moments.mean,
        0.0,
        standard_error,
        z_score,
        numerical_floor,
        "one-sample mean versus symmetry expectation zero",
        "Computed over exited rays with finite directions.",
        pass_override=False if moments.count == 0 else None,
    )


def _build_rows(
    gpu: TransportSummary,
    cpu: TransportSummary,
    metadata: RunMetadata,
    reconstructed_mu_t: float,
    detector: DetectorConfig,
    z_score: float,
    numerical_floor: float,
    direction_norm_tolerance: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    terminal_metrics = (
        ("exited_fraction", STATE_EXITED),
        ("absorbed_fraction", STATE_ABSORBED),
        ("truncated_fraction", STATE_TRUNCATED),
        ("missed_sample_fraction", STATE_MISSED_SAMPLE),
    )
    for name, state in terminal_metrics:
        rows.append(
            _proportion_row(
                "terminal_fractions",
                name,
                gpu.terminal_counts[state],
                gpu.n_rays,
                cpu.terminal_counts[state],
                cpu.n_rays,
                z_score,
                numerical_floor,
            )
        )

    scatter_metrics = (
        ("ballistic_exit_fraction", "ballistic_exit_count"),
        ("single_scattered_exit_fraction", "single_exit_count"),
        ("multiply_scattered_exit_fraction", "multiple_exit_count"),
    )
    for name, attribute in scatter_metrics:
        rows.append(
            _proportion_row(
                "scatter_order",
                name,
                int(getattr(gpu, attribute)),
                gpu.n_rays,
                int(getattr(cpu, attribute)),
                cpu.n_rays,
                z_score,
                numerical_floor,
                "Denominator is all launched rays; classes include exited rays only.",
            )
        )
    rows.append(
        _mean_row(
            "scatter_order",
            "mean_scatter_count_all_rays",
            gpu.scatter_count_moments,
            cpu.scatter_count_moments,
            z_score,
            numerical_floor,
        )
    )

    rows.append(
        _proportion_row(
            "detectors",
            "total_detected_fraction",
            gpu.detected_count,
            gpu.n_rays,
            cpu.detected_count,
            cpu.n_rays,
            z_score,
            numerical_floor,
            "Exclusive nearest accepted annular detector assignment.",
        )
    )
    for index, centre in enumerate(detector.centres_deg):
        rows.append(
            _proportion_row(
                "detectors",
                f"detector_{index:02d}_{float(centre):g}deg_fraction",
                int(gpu.detector_counts[index]),
                gpu.n_rays,
                int(cpu.detector_counts[index]),
                cpu.n_rays,
                z_score,
                numerical_floor,
                (
                    "Exclusive ideal-annular fraction of all launched rays; "
                    f"acceptance half-angle={detector.acceptance_half_angle_deg:g} deg."
                ),
            )
        )

    rows.extend(
        (
            _mean_row(
                "transverse_symmetry",
                "exit_direction_vx_mean_gpu_vs_cpu",
                gpu.exit_vx_moments,
                cpu.exit_vx_moments,
                z_score,
                numerical_floor,
            ),
            _mean_row(
                "transverse_symmetry",
                "exit_direction_vz_mean_gpu_vs_cpu",
                gpu.exit_vz_moments,
                cpu.exit_vz_moments,
                z_score,
                numerical_floor,
            ),
            _symmetry_row(
                "exit_direction_vx_mean",
                gpu.exit_vx_moments,
                z_score,
                numerical_floor,
                "gpu",
            ),
            _symmetry_row(
                "exit_direction_vz_mean",
                gpu.exit_vz_moments,
                z_score,
                numerical_floor,
                "gpu",
            ),
            _symmetry_row(
                "exit_direction_vx2_minus_vz2_mean",
                gpu.transverse_square_difference_moments,
                z_score,
                numerical_floor,
                "gpu",
            ),
            _symmetry_row(
                "exit_direction_vx_mean",
                cpu.exit_vx_moments,
                z_score,
                numerical_floor,
                "cpu",
            ),
            _symmetry_row(
                "exit_direction_vz_mean",
                cpu.exit_vz_moments,
                z_score,
                numerical_floor,
                "cpu",
            ),
            _symmetry_row(
                "exit_direction_vx2_minus_vz2_mean",
                cpu.transverse_square_difference_moments,
                z_score,
                numerical_floor,
                "cpu",
            ),
        )
    )

    for implementation, summary in (("gpu", gpu), ("cpu", cpu)):
        known_terminal = (
            sum(summary.terminal_counts.values()) / summary.n_rays
        )
        exit_class_sum = (
            summary.ballistic_exit_count
            + summary.single_exit_count
            + summary.multiple_exit_count
        ) / summary.n_rays
        exited_fraction = summary.terminal_fraction(STATE_EXITED)
        rows.extend(
            (
                _metric_row(
                    "accounting",
                    f"{implementation}_terminal_partition_sum",
                    known_terminal,
                    1.0,
                    0.0,
                    z_score,
                    0.0,
                    "exact terminal-state invariant",
                    tolerance_override=1.0e-12,
                ),
                _metric_row(
                    "accounting",
                    f"{implementation}_exit_scatter_class_sum",
                    exit_class_sum,
                    exited_fraction,
                    0.0,
                    z_score,
                    0.0,
                    "exact exited scatter-class invariant",
                    tolerance_override=1.0e-12,
                ),
                _metric_row(
                    "accounting",
                    f"{implementation}_invalid_scatter_fraction",
                    summary.invalid_scatter_count / summary.n_rays,
                    0.0,
                    0.0,
                    z_score,
                    0.0,
                    "exact record-integrity invariant",
                    tolerance_override=0.0,
                ),
                _metric_row(
                    "accounting",
                    f"{implementation}_absorbed_flag_mismatch_fraction",
                    summary.absorbed_flag_mismatch_count / summary.n_rays,
                    0.0,
                    0.0,
                    z_score,
                    0.0,
                    "exact record-integrity invariant",
                    tolerance_override=0.0,
                ),
                _metric_row(
                    "accounting",
                    f"{implementation}_truncated_flag_mismatch_fraction",
                    summary.truncated_flag_mismatch_count / summary.n_rays,
                    0.0,
                    0.0,
                    z_score,
                    0.0,
                    "exact record-integrity invariant",
                    tolerance_override=0.0,
                ),
                _metric_row(
                    "accounting",
                    f"{implementation}_nonfinite_exit_position_fraction",
                    summary.nonfinite_exit_position_count / summary.n_rays,
                    0.0,
                    0.0,
                    z_score,
                    0.0,
                    "exact record-integrity invariant",
                    "Count divided by all rays; only exited records are tested.",
                    tolerance_override=0.0,
                ),
                _metric_row(
                    "accounting",
                    f"{implementation}_nonfinite_exit_direction_fraction",
                    summary.nonfinite_exit_direction_count / summary.n_rays,
                    0.0,
                    0.0,
                    z_score,
                    0.0,
                    "exact record-integrity invariant",
                    "Count divided by all rays; only exited records are tested.",
                    tolerance_override=0.0,
                ),
                _metric_row(
                    "direction_norm",
                    f"{implementation}_maximum_exit_direction_norm_error",
                    summary.maximum_direction_norm_error,
                    0.0,
                    0.0,
                    z_score,
                    0.0,
                    "absolute numerical invariant",
                    tolerance_override=direction_norm_tolerance,
                ),
            )
        )

    mu_t_tolerance = max(1.0e-10, abs(reconstructed_mu_t) * 1.0e-8)
    rows.append(
        _metric_row(
            "configuration",
            "reconstructed_mu_t_m_inv",
            metadata.mu_t_m_inv,
            reconstructed_mu_t,
            0.0,
            z_score,
            0.0,
            "host/reference coefficient reconstruction",
            tolerance_override=mu_t_tolerance,
        )
    )
    return rows


def _write_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing CLARITAS CPU reference against a CLARITAS_77 "
            "reference-collimated, floc-disabled CUDA HDF5 file and save "
            "statistical parity checks to CSV."
        )
    )
    parser.add_argument("hdf5", type=Path, help="CLARITAS_77 ray-exit HDF5 file")
    parser.add_argument(
        "--cpu-rays",
        type=int,
        help="CPU reference ray count (default: same as HDF5 n_rays)",
    )
    parser.add_argument(
        "--cpu-seed",
        type=int,
        help="CPU reference seed (default: HDF5 simulation_seed)",
    )
    parser.add_argument(
        "--phase-grid-size",
        type=int,
        default=20_001,
        help="CPU reference Mie phase grid size (default: 20001)",
    )
    parser.add_argument(
        "--z-score",
        type=float,
        default=5.0,
        help="Monte Carlo standard-error multiplier (default: 5.0)",
    )
    parser.add_argument(
        "--numerical-floor",
        type=float,
        default=5.0e-4,
        help=(
            "Additive dimensionless tolerance for statistical comparisons "
            "(default: 5e-4; covers small CUDA-float numerical effects)"
        ),
    )
    parser.add_argument(
        "--direction-norm-tolerance",
        type=float,
        default=5.0e-5,
        help="Maximum absolute exited-direction norm error (default: 5e-5)",
    )
    parser.add_argument(
        "--detector-centres-deg",
        type=float,
        nargs="+",
        help=(
            "Detector centres if absent from HDF5 "
            "(default: reference centres 0,10,...,170)"
        ),
    )
    parser.add_argument(
        "--detector-acceptance-deg",
        type=float,
        help=(
            "Detector acceptance half-angle if absent from HDF5 "
            "(default: reference value 6.5)"
        ),
    )
    parser.add_argument(
        "--hdf-chunk-rays",
        type=int,
        default=1_000_000,
        help="HDF5 streaming chunk size (default: 1000000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output CSV path (default: <hdf5 stem>_reference_parity.csv)"
        ),
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.cpu_rays is not None and args.cpu_rays <= 0:
        raise ValueError("--cpu-rays must be positive")
    if args.phase_grid_size < 1001:
        raise ValueError("--phase-grid-size must be at least 1001")
    if not np.isfinite(args.z_score) or args.z_score <= 0.0:
        raise ValueError("--z-score must be finite and positive")
    if not np.isfinite(args.numerical_floor) or args.numerical_floor < 0.0:
        raise ValueError("--numerical-floor must be finite and nonnegative")
    if (
        not np.isfinite(args.direction_norm_tolerance)
        or args.direction_norm_tolerance < 0.0
    ):
        raise ValueError(
            "--direction-norm-tolerance must be finite and nonnegative"
        )
    if args.hdf_chunk_rays <= 0:
        raise ValueError("--hdf-chunk-rays must be positive")
    if not args.hdf5.is_file():
        raise ValueError(f"HDF5 file does not exist: {args.hdf5}")

    with h5py.File(args.hdf5, "r") as handle:
        metadata = _read_metadata(handle)
        detector, detector_provenance = _detector_config(
            handle,
            args.detector_centres_deg,
            args.detector_acceptance_deg,
        )
        gpu_summary = _summarize_gpu(
            handle,
            metadata,
            detector,
            args.hdf_chunk_rays,
        )

    cpu_rays = (
        metadata.n_rays if args.cpu_rays is None else int(args.cpu_rays)
    )
    cpu_seed = metadata.seed if args.cpu_seed is None else int(args.cpu_seed)
    print(
        "Building existing CPU reference medium for "
        f"{metadata.material}, {metadata.concentration_kg_m3:g} kg/m^3, "
        f"{metadata.wavelength_m * 1e9:g} nm..."
    )
    medium = build_primary_medium(
        MATERIALS[metadata.material],
        metadata.concentration_kg_m3,
        metadata.wavelength_m,
        phase_grid_size=args.phase_grid_size,
    )
    config = SimulationConfig(
        n_rays=cpu_rays,
        seed=cpu_seed,
        wavelength_m=metadata.wavelength_m,
        concentration_kg_m3=metadata.concentration_kg_m3,
        sample_radius_m=metadata.sample_radius_m,
        max_events=metadata.max_events,
        phase_grid_size=args.phase_grid_size,
        detector=detector,
    )
    print(
        f"Running existing CPU reference with {cpu_rays:,} rays "
        f"(seed={cpu_seed}, max_events={metadata.max_events})..."
    )
    cpu_result = simulate(config, medium)
    cpu_summary = _summarize_cpu(cpu_result, detector)

    rows = _build_rows(
        gpu_summary,
        cpu_summary,
        metadata,
        medium.mu_t_m_inv,
        detector,
        args.z_score,
        args.numerical_floor,
        args.direction_norm_tolerance,
    )
    output = args.output or args.hdf5.with_name(
        f"{args.hdf5.stem}_reference_parity.csv"
    )
    _write_rows(output, rows)

    failed = [row for row in rows if not bool(row["passed"])]
    print(f"Detector configuration: {detector_provenance}")
    print(
        f"CLARITAS_77 CPU-reference parity: "
        f"{len(rows) - len(failed)}/{len(rows)} checks passed"
    )
    print(f"Saved {output}")
    if failed:
        print("Failed checks:")
        for row in failed:
            print(
                f"  {row['section']}/{row['metric']}: "
                f"|difference|={abs(float(row['difference'])):.6g}, "
                f"tolerance={float(row['tolerance']):.6g}"
            )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
