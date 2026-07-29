"""Post-process CLARITAS ray records into reproducible transport diagnostics.

This module deliberately contains no transport physics.  It accepts the
per-ray arrays produced by the CUDA kernel and writes diagnostic CSV files with
explicit denominators and mutually exclusive terminal-state accounting.

The main entry point, :func:`save_comprehensive_transport_diagnostics`, is
designed for direct use by CLARITAS.  The HDF5 wrapper and command-line entry
point also support the dataset names written by CLARITAS_76.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


DIAGNOSTICS_SCHEMA_VERSION = "1.1"


def _as_1d(name: str, values: Any, *, length: Optional[int] = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if length is not None and len(array) != length:
        raise ValueError(
            f"{name} has length {len(array)}, expected the common ray count {length}"
        )
    return array


def _optional_1d(
    name: str,
    values: Any,
    *,
    length: int,
    fill_value: float,
    dtype: Any,
) -> np.ndarray:
    if values is None:
        return np.full(length, fill_value, dtype=dtype)
    return _as_1d(name, values, length=length).astype(dtype, copy=False)


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _finite_stats(values: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {
            "finite_count": 0,
            "mean": np.nan,
            "std": np.nan,
            "minimum": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "maximum": np.nan,
            "sum": 0.0,
        }

    return {
        "finite_count": int(selected.size),
        "mean": float(np.mean(selected)),
        "std": float(np.std(selected)),
        "minimum": float(np.min(selected)),
        "p25": float(np.percentile(selected, 25.0)),
        "median": float(np.median(selected)),
        "p75": float(np.percentile(selected, 75.0)),
        "p95": float(np.percentile(selected, 95.0)),
        "p99": float(np.percentile(selected, 99.0)),
        "maximum": float(np.max(selected)),
        "sum": float(np.sum(selected, dtype=np.float64)),
    }


def _statistics_table(
    values: np.ndarray,
    populations: Sequence[Tuple[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for population, mask in populations:
        mask = np.asarray(mask, dtype=bool)
        row = {
            "population": population,
            "ray_count": int(np.sum(mask)),
        }
        row.update(_finite_stats(values, mask))
        rows.append(row)
    return pd.DataFrame(rows)


def _path_histogram_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if bins < 1:
        raise ValueError("path_histogram_bins must be at least 1")
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite >= 0.0)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, bins + 1)

    maximum = float(np.max(finite))
    if maximum <= 0.0:
        maximum = float(np.nextafter(0.0, 1.0)) * bins
    return np.linspace(0.0, maximum, bins + 1)


def _angular_edges(stop_deg: float, width_deg: float) -> np.ndarray:
    if not np.isfinite(width_deg) or width_deg <= 0.0:
        raise ValueError("angular_bin_width_deg must be finite and positive")
    count = int(np.ceil(stop_deg / width_deg))
    return np.linspace(0.0, stop_deg, count + 1)


def _angular_histogram(
    angle_deg: np.ndarray,
    base_mask: np.ndarray,
    detected_mask: np.ndarray,
    edges: np.ndarray,
    *,
    polar: bool,
) -> pd.DataFrame:
    valid = (
        np.asarray(base_mask, dtype=bool)
        & np.isfinite(angle_deg)
        & (angle_deg >= edges[0])
        & (angle_deg <= edges[-1])
    )
    detected_valid = valid & np.asarray(detected_mask, dtype=bool)

    counts, _ = np.histogram(angle_deg[valid], bins=edges)
    detected_counts, _ = np.histogram(angle_deg[detected_valid], bins=edges)
    widths = np.diff(edges)
    base_total = int(np.sum(base_mask))
    detected_total = int(np.sum(np.asarray(base_mask, dtype=bool) & detected_mask))

    frame = pd.DataFrame(
        {
            "angle_bin_lower_deg": edges[:-1],
            "angle_bin_upper_deg": edges[1:],
            "angle_bin_centre_deg": 0.5 * (edges[:-1] + edges[1:]),
            "exit_count": counts.astype(np.int64),
            "fraction_of_all_escaped_rays": (
                counts / float(base_total)
                if base_total > 0
                else np.zeros_like(counts, dtype=float)
            ),
            "probability_density_per_degree": (
                counts / (float(base_total) * widths)
                if base_total > 0
                else np.zeros_like(counts, dtype=float)
            ),
            "detected_exit_count": detected_counts.astype(np.int64),
            "fraction_of_all_detected_rays": (
                detected_counts / float(detected_total)
                if detected_total > 0
                else np.zeros_like(detected_counts, dtype=float)
            ),
        }
    )

    if polar:
        lower_rad = np.deg2rad(edges[:-1])
        upper_rad = np.deg2rad(edges[1:])
        solid_angle = 2.0 * np.pi * (np.cos(lower_rad) - np.cos(upper_rad))
        frame["bin_solid_angle_sr"] = solid_angle
        frame["probability_density_per_sr"] = np.divide(
            counts,
            float(base_total) * solid_angle,
            out=np.zeros_like(solid_angle),
            where=(base_total > 0) & (solid_angle > 0.0),
        )

    return frame


def _write_csv(frame: pd.DataFrame, outdir: Path, stem: str, wl_nm: int) -> str:
    path = outdir / f"{stem}_{int(wl_nm)}nm.csv"
    frame.to_csv(path, index=False)
    return str(path)


def save_comprehensive_transport_diagnostics(
    *,
    wl_nm: int,
    outdir: Any,
    exit_x: Any,
    exit_y: Any,
    exit_z: Any,
    exit_vx: Any,
    exit_vy: Any,
    exit_vz: Any,
    path_length: Any,
    scatter_count: Any,
    floc_event_count: Any,
    floc_extinction_count: Any = None,
    extinction_count: Any,
    absorbed: Any,
    truncated: Any,
    detector_index: Any,
    detector_angles_deg: Any,
    floc_internal_scatter_count: Any = None,
    path_histogram_bins: int = 100,
    angular_bin_width_deg: float = 1.0,
    position_histogram_bins: int = 100,
    truncated_status_available: bool = True,
) -> Dict[str, Any]:
    """Save comprehensive diagnostics for one completed wavelength run.

    Fractions in ``run_summary`` use the number of launched ray records as
    their denominator.  ``ballistic_fraction``, ``single_scattered_fraction``
    and ``multiply_scattered_fraction`` describe escaped rays with respectively
    0, 1 and at least 2 recorded outer-transport scatters.  Their sum therefore
    equals ``escaped_fraction`` when every escaped ray has valid bookkeeping.

    ``floc_event_count`` is incremented only after an extinction event survives
    the albedo test, so it measures successful outer floc scatters.
    ``floc_extinction_count``, when supplied, measures floc selections before
    that albedo decision.  Canonical ``floc_event_fraction`` and
    ``primary_event_fraction`` then use all outer extinction events, while the
    successful-scatter fractions remain available under separate names.

    Legacy CLARITAS_76 files have no floc-extinction counter.  For those files
    the canonical fields retain the historical scatter-event calculation, but
    ``event_fraction_basis``, ``event_fraction_provenance``, and
    ``floc_extinction_count_available`` make that fallback explicit.
    """

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    scatter = _as_1d("scatter_count", scatter_count).astype(np.int64, copy=False)
    n_rays = len(scatter)

    extinction = _as_1d(
        "extinction_count", extinction_count, length=n_rays
    ).astype(np.int64, copy=False)
    floc_scatter = _as_1d(
        "floc_event_count", floc_event_count, length=n_rays
    ).astype(np.int64, copy=False)
    floc_extinction_available = floc_extinction_count is not None
    floc_extinction = _optional_1d(
        "floc_extinction_count",
        floc_extinction_count,
        length=n_rays,
        fill_value=-1,
        dtype=np.int64,
    )
    absorbed_flag = _as_1d(
        "absorbed", absorbed, length=n_rays
    ).astype(np.int64, copy=False)
    truncated_flag = _optional_1d(
        "truncated",
        truncated,
        length=n_rays,
        fill_value=0,
        dtype=np.int64,
    )

    x = _as_1d("exit_x", exit_x, length=n_rays).astype(np.float64, copy=False)
    y = _as_1d("exit_y", exit_y, length=n_rays).astype(np.float64, copy=False)
    z = _optional_1d(
        "exit_z", exit_z, length=n_rays, fill_value=0.0, dtype=np.float64
    )
    vx = _optional_1d(
        "exit_vx", exit_vx, length=n_rays, fill_value=np.nan, dtype=np.float64
    )
    vy = _optional_1d(
        "exit_vy", exit_vy, length=n_rays, fill_value=np.nan, dtype=np.float64
    )
    vz = _optional_1d(
        "exit_vz", exit_vz, length=n_rays, fill_value=np.nan, dtype=np.float64
    )
    paths = _as_1d(
        "path_length", path_length, length=n_rays
    ).astype(np.float64, copy=False)
    detector = _optional_1d(
        "detector_index",
        detector_index,
        length=n_rays,
        fill_value=-1,
        dtype=np.int64,
    )
    detector_angles = _as_1d(
        "detector_angles_deg", detector_angles_deg
    ).astype(np.float64, copy=False)
    internal_scatter = _optional_1d(
        "floc_internal_scatter_count",
        floc_internal_scatter_count,
        length=n_rays,
        fill_value=0,
        dtype=np.int64,
    )

    valid_flags = (
        np.isin(absorbed_flag, (0, 1))
        & np.isin(truncated_flag, (0, 1))
    )
    base_bookkeeping = (
        (scatter >= 0)
        & (extinction >= 0)
        & (floc_scatter >= 0)
        & (internal_scatter >= 0)
        & valid_flags
    )
    valid_bookkeeping = (
        base_bookkeeping
        & ((floc_extinction >= 0) if floc_extinction_available else True)
    )
    finite_exit_position = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    status_conflict = (absorbed_flag == 1) & (truncated_flag == 1)

    absorbed_mask = (
        valid_bookkeeping
        & (absorbed_flag == 1)
        & (truncated_flag == 0)
    )
    truncated_mask = (
        valid_bookkeeping
        & (truncated_flag == 1)
        & (absorbed_flag == 0)
    )
    escaped_mask = (
        valid_bookkeeping
        & (absorbed_flag == 0)
        & (truncated_flag == 0)
        & finite_exit_position
    )
    terminal_mask = absorbed_mask | truncated_mask | escaped_mask
    unclassified_mask = ~terminal_mask

    valid_detector_index = (detector >= 0) & (detector < len(detector_angles))
    invalid_positive_detector_index = detector >= len(detector_angles)
    detected_mask = escaped_mask & valid_detector_index

    ballistic_mask = escaped_mask & (scatter == 0)
    single_mask = escaped_mask & (scatter == 1)
    multiple_mask = escaped_mask & (scatter >= 2)

    n_completed = int(np.sum(terminal_mask))
    n_escaped = int(np.sum(escaped_mask))
    n_absorbed = int(np.sum(absorbed_mask))
    n_truncated = int(np.sum(truncated_mask))
    n_unclassified = int(np.sum(unclassified_mask))
    n_detected = int(np.sum(detected_mask))
    n_ballistic = int(np.sum(ballistic_mask))
    n_single = int(np.sum(single_mask))
    n_multiple = int(np.sum(multiple_mask))

    total_extinction_events = int(
        np.sum(extinction[terminal_mask], dtype=np.int64)
    )
    total_scatter_events = int(np.sum(scatter[terminal_mask], dtype=np.int64))
    total_floc_scatter_events = int(
        np.sum(floc_scatter[terminal_mask], dtype=np.int64)
    )
    total_primary_scatter_events = int(
        total_scatter_events - total_floc_scatter_events
    )
    successful_floc_scatter_fraction = _fraction(
        total_floc_scatter_events, total_scatter_events
    )
    successful_primary_scatter_fraction = _fraction(
        total_primary_scatter_events, total_scatter_events
    )

    if floc_extinction_available:
        total_floc_extinction_events = int(
            np.sum(floc_extinction[terminal_mask], dtype=np.int64)
        )
        total_primary_extinction_events = int(
            total_extinction_events - total_floc_extinction_events
        )
        floc_event_fraction = _fraction(
            total_floc_extinction_events, total_extinction_events
        )
        primary_event_fraction = _fraction(
            total_primary_extinction_events, total_extinction_events
        )
        event_fraction_basis = "all_outer_extinction_events"
        event_fraction_provenance = "per_ray_floc_extinction_count"
        event_fraction_is_all_extinction_events = True
    else:
        total_floc_extinction_events = None
        total_primary_extinction_events = None
        floc_event_fraction = successful_floc_scatter_fraction
        primary_event_fraction = successful_primary_scatter_fraction
        event_fraction_basis = "successful_outer_scattering_events_fallback"
        event_fraction_provenance = (
            "legacy_fallback_from_floc_event_count;"
            "floc_extinction_count_unavailable"
        )
        event_fraction_is_all_extinction_events = False
    total_internal_floc_scatter_events = int(
        np.sum(internal_scatter[terminal_mask], dtype=np.int64)
    )
    inferred_absorption_events = int(
        np.sum(extinction[terminal_mask] - scatter[terminal_mask], dtype=np.int64)
    )

    path_available = terminal_mask & np.isfinite(paths) & (paths >= 0.0)
    absorbed_path_available = absorbed_mask & path_available
    truncated_path_available = truncated_mask & path_available
    escaped_path_available = escaped_mask & path_available

    summary: Dict[str, Any] = {
        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "wavelength_nm": int(wl_nm),
        "total_ray_count": int(n_rays),
        "completed_ray_count": n_completed,
        "completed_fraction": _fraction(n_completed, n_rays),
        "escaped_ray_count": n_escaped,
        "escaped_fraction": _fraction(n_escaped, n_rays),
        "total_escaped_fraction": _fraction(n_escaped, n_rays),
        "absorbed_ray_count": n_absorbed,
        "absorbed_fraction": _fraction(n_absorbed, n_rays),
        "total_absorbed_fraction": _fraction(n_absorbed, n_rays),
        "truncated_ray_count": n_truncated,
        "truncated_fraction": _fraction(n_truncated, n_rays),
        "total_truncated_fraction": _fraction(n_truncated, n_rays),
        "unclassified_ray_count": n_unclassified,
        "unclassified_fraction": _fraction(n_unclassified, n_rays),
        "accounting_sum": _fraction(
            n_escaped + n_absorbed + n_truncated, n_rays
        ),
        "partition_sum_including_unclassified": _fraction(
            n_escaped + n_absorbed + n_truncated + n_unclassified, n_rays
        ),
        "detected_unique_ray_count": n_detected,
        "detector_hit_assignment_count": n_detected,
        "total_detected_fraction": _fraction(n_detected, n_rays),
        "absolute_detector_efficiency": _fraction(n_detected, n_rays),
        "detected_fraction_of_escaped": _fraction(n_detected, n_escaped),
        "ballistic_ray_count": n_ballistic,
        "ballistic_fraction": _fraction(n_ballistic, n_rays),
        "ballistic_fraction_of_escaped": _fraction(n_ballistic, n_escaped),
        "single_scattered_ray_count": n_single,
        "single_scattered_fraction": _fraction(n_single, n_rays),
        "single_scattered_fraction_of_escaped": _fraction(n_single, n_escaped),
        "multiply_scattered_ray_count": n_multiple,
        "multiply_scattered_fraction": _fraction(n_multiple, n_rays),
        "multiply_scattered_fraction_of_escaped": _fraction(n_multiple, n_escaped),
        "mean_scatter_count": (
            float(np.mean(scatter[terminal_mask])) if n_completed > 0 else np.nan
        ),
        "mean_scatter_count_escaped": (
            float(np.mean(scatter[escaped_mask])) if n_escaped > 0 else np.nan
        ),
        "mean_scatter_count_absorbed": (
            float(np.mean(scatter[absorbed_mask])) if n_absorbed > 0 else np.nan
        ),
        "mean_extinction_count": (
            float(np.mean(extinction[terminal_mask])) if n_completed > 0 else np.nan
        ),
        "total_extinction_event_count": total_extinction_events,
        "total_outer_scatter_event_count": total_scatter_events,
        "total_floc_outer_scatter_event_count": total_floc_scatter_events,
        "total_primary_outer_scatter_event_count": total_primary_scatter_events,
        "successful_floc_scatter_fraction": successful_floc_scatter_fraction,
        "successful_primary_scatter_fraction": successful_primary_scatter_fraction,
        "floc_scatter_event_fraction": successful_floc_scatter_fraction,
        "primary_scatter_event_fraction": successful_primary_scatter_fraction,
        "floc_extinction_count_available": bool(floc_extinction_available),
        "total_floc_outer_extinction_event_count": (
            total_floc_extinction_events
            if total_floc_extinction_events is not None
            else np.nan
        ),
        "total_primary_outer_extinction_event_count": (
            total_primary_extinction_events
            if total_primary_extinction_events is not None
            else np.nan
        ),
        "total_internal_floc_scatter_event_count": total_internal_floc_scatter_events,
        "mean_internal_floc_scatter_count": (
            float(np.mean(internal_scatter[terminal_mask]))
            if n_completed > 0
            else np.nan
        ),
        "floc_event_fraction": floc_event_fraction,
        "primary_event_fraction": primary_event_fraction,
        "event_fraction_basis": event_fraction_basis,
        "event_fraction_provenance": event_fraction_provenance,
        "event_fraction_is_all_extinction_events": bool(
            event_fraction_is_all_extinction_events
        ),
        "inferred_absorption_extinction_event_count": inferred_absorption_events,
        "realised_scattering_fraction_of_extinction_events": _fraction(
            total_scatter_events, total_extinction_events
        ),
        "realised_absorption_fraction_of_extinction_events": _fraction(
            inferred_absorption_events, total_extinction_events
        ),
        "path_length_recorded_ray_count": int(np.sum(path_available)),
        "path_length_recorded_fraction_of_completed": _fraction(
            int(np.sum(path_available)), n_completed
        ),
        "absorbed_path_length_recorded_fraction": _fraction(
            int(np.sum(absorbed_path_available)), n_absorbed
        ),
        "truncated_path_length_recorded_fraction": _fraction(
            int(np.sum(truncated_path_available)), n_truncated
        ),
        "escaped_path_length_recorded_fraction": _fraction(
            int(np.sum(escaped_path_available)), n_escaped
        ),
        "mean_path_length_m": (
            float(np.mean(paths[path_available])) if np.any(path_available) else np.nan
        ),
        "mean_escaped_path_length_m": (
            float(np.mean(paths[escaped_path_available]))
            if np.any(escaped_path_available)
            else np.nan
        ),
        "truncated_status_available": bool(truncated_status_available),
    }

    populations = (
        ("completed", terminal_mask),
        ("escaped", escaped_mask),
        ("absorbed", absorbed_mask),
        ("truncated", truncated_mask),
        ("detected", detected_mask),
    )
    scatter_statistics = _statistics_table(scatter, populations)
    floc_internal_scatter_statistics = _statistics_table(
        internal_scatter, populations
    )
    extinction_statistics = _statistics_table(extinction, populations)

    max_scatter = (
        int(np.max(scatter[terminal_mask])) if np.any(terminal_mask) else 0
    )
    scatter_orders = np.arange(max_scatter + 1, dtype=np.int64)
    completed_scatter_hist = np.bincount(
        scatter[terminal_mask], minlength=max_scatter + 1
    )
    escaped_scatter_hist = np.bincount(
        scatter[escaped_mask], minlength=max_scatter + 1
    )
    absorbed_scatter_hist = np.bincount(
        scatter[absorbed_mask], minlength=max_scatter + 1
    )
    truncated_scatter_hist = np.bincount(
        scatter[truncated_mask], minlength=max_scatter + 1
    )
    detected_scatter_hist = np.bincount(
        scatter[detected_mask], minlength=max_scatter + 1
    )

    scatter_histogram = pd.DataFrame(
        {
            "scatter_count": scatter_orders,
            "completed_ray_count": completed_scatter_hist,
            "escaped_ray_count": escaped_scatter_hist,
            "absorbed_ray_count": absorbed_scatter_hist,
            "truncated_ray_count": truncated_scatter_hist,
            "detected_ray_count": detected_scatter_hist,
            "fraction_of_all_rays": (
                completed_scatter_hist / float(n_rays)
                if n_rays > 0
                else np.zeros_like(completed_scatter_hist, dtype=float)
            ),
            "fraction_of_completed_rays": (
                completed_scatter_hist / float(n_completed)
                if n_completed > 0
                else np.zeros_like(completed_scatter_hist, dtype=float)
            ),
            "fraction_of_escaped_rays": (
                escaped_scatter_hist / float(n_escaped)
                if n_escaped > 0
                else np.zeros_like(escaped_scatter_hist, dtype=float)
            ),
        }
    )

    absorption_vs_scatter = pd.DataFrame(
        {
            "scatter_count": scatter_orders,
            "terminal_ray_count": completed_scatter_hist,
            "absorbed_ray_count": absorbed_scatter_hist,
            "escaped_ray_count": escaped_scatter_hist,
            "truncated_ray_count": truncated_scatter_hist,
            "ultimate_absorption_probability_given_scatter_count": np.divide(
                absorbed_scatter_hist,
                completed_scatter_hist,
                out=np.zeros_like(absorbed_scatter_hist, dtype=float),
                where=completed_scatter_hist > 0,
            ),
            "absorption_probability_excluding_truncated": np.divide(
                absorbed_scatter_hist,
                absorbed_scatter_hist + escaped_scatter_hist,
                out=np.zeros_like(absorbed_scatter_hist, dtype=float),
                where=(absorbed_scatter_hist + escaped_scatter_hist) > 0,
            ),
            "escape_probability_given_scatter_count": np.divide(
                escaped_scatter_hist,
                completed_scatter_hist,
                out=np.zeros_like(escaped_scatter_hist, dtype=float),
                where=completed_scatter_hist > 0,
            ),
            "truncation_probability_given_scatter_count": np.divide(
                truncated_scatter_hist,
                completed_scatter_hist,
                out=np.zeros_like(truncated_scatter_hist, dtype=float),
                where=completed_scatter_hist > 0,
            ),
        }
    )

    path_edges = _path_histogram_edges(paths[path_available], path_histogram_bins)
    path_rows: Dict[str, Any] = {
        "path_length_bin_lower_m": path_edges[:-1],
        "path_length_bin_upper_m": path_edges[1:],
        "path_length_bin_centre_m": 0.5 * (path_edges[:-1] + path_edges[1:]),
    }
    path_masks = (
        ("recorded", path_available),
        ("escaped", escaped_path_available),
        ("absorbed", absorbed_path_available),
        ("truncated", truncated_path_available),
        ("detected", detected_mask & path_available),
    )
    for label, mask in path_masks:
        counts, _ = np.histogram(paths[mask], bins=path_edges)
        path_rows[f"{label}_ray_count"] = counts.astype(np.int64)
        denominator = int(np.sum(mask))
        path_rows[f"fraction_of_{label}_rays"] = (
            counts / float(denominator)
            if denominator > 0
            else np.zeros_like(counts, dtype=float)
        )
    path_histogram = pd.DataFrame(path_rows)

    detector_rows = []
    detector_order_rows = []
    for det_idx, det_angle in enumerate(detector_angles):
        this_detector = detected_mask & (detector == det_idx)
        detector_count = int(np.sum(this_detector))
        detector_rows.append(
            {
                "detector_index": int(det_idx),
                "detector_angle_deg": float(det_angle),
                "detected_ray_count": detector_count,
                "fraction_of_all_rays": _fraction(detector_count, n_rays),
                "fraction_of_escaped_rays": _fraction(detector_count, n_escaped),
                "fraction_of_all_detected_rays": _fraction(detector_count, n_detected),
                "mean_scatter_count": (
                    float(np.mean(scatter[this_detector]))
                    if detector_count > 0
                    else np.nan
                ),
                "mean_extinction_count": (
                    float(np.mean(extinction[this_detector]))
                    if detector_count > 0
                    else np.nan
                ),
                "mean_path_length_m": (
                    float(np.mean(paths[this_detector & path_available]))
                    if np.any(this_detector & path_available)
                    else np.nan
                ),
            }
        )

        det_order_counts = np.bincount(
            scatter[this_detector], minlength=max_scatter + 1
        )
        for order, count in enumerate(det_order_counts):
            all_detected_at_order = int(detected_scatter_hist[order])
            detector_order_rows.append(
                {
                    "detector_index": int(det_idx),
                    "detector_angle_deg": float(det_angle),
                    "scatter_order": int(order),
                    "detected_ray_count": int(count),
                    "fraction_of_this_detector_hits": _fraction(
                        int(count), detector_count
                    ),
                    "fraction_of_all_hits_at_this_scatter_order": _fraction(
                        int(count), all_detected_at_order
                    ),
                    "fraction_of_all_detected_rays": _fraction(
                        int(count), n_detected
                    ),
                    "fraction_of_all_rays": _fraction(int(count), n_rays),
                }
            )

    detector_efficiency = pd.DataFrame(detector_rows)
    detector_by_order = pd.DataFrame(detector_order_rows)

    direction_norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    valid_direction = (
        escaped_mask
        & np.isfinite(direction_norm)
        & (direction_norm > 0.0)
    )
    exit_direction_polar_deg = np.full(n_rays, np.nan, dtype=np.float64)
    exit_direction_polar_deg[valid_direction] = np.rad2deg(
        np.arccos(
            np.clip(vy[valid_direction] / direction_norm[valid_direction], -1.0, 1.0)
        )
    )

    position_radius = np.sqrt(x * x + y * y + z * z)
    valid_position_angle = (
        escaped_mask
        & np.isfinite(position_radius)
        & (position_radius > 0.0)
    )
    exit_position_polar_deg = np.full(n_rays, np.nan, dtype=np.float64)
    exit_position_polar_deg[valid_position_angle] = np.rad2deg(
        np.arccos(
            np.clip(y[valid_position_angle] / position_radius[valid_position_angle], -1.0, 1.0)
        )
    )
    exit_position_azimuth_deg = np.full(n_rays, np.nan, dtype=np.float64)
    exit_position_azimuth_deg[valid_position_angle] = (
        np.rad2deg(np.arctan2(z[valid_position_angle], x[valid_position_angle]))
        + 360.0
    ) % 360.0
    summary["exit_direction_recorded_ray_count"] = int(np.sum(valid_direction))
    summary["exit_direction_recorded_fraction_of_escaped"] = _fraction(
        int(np.sum(valid_direction)), n_escaped
    )
    summary["exit_position_angle_recorded_ray_count"] = int(
        np.sum(valid_position_angle)
    )
    summary["exit_position_angle_recorded_fraction_of_escaped"] = _fraction(
        int(np.sum(valid_position_angle)), n_escaped
    )

    polar_edges = _angular_edges(180.0, angular_bin_width_deg)
    azimuth_edges = _angular_edges(360.0, angular_bin_width_deg)
    exit_direction_distribution = _angular_histogram(
        exit_direction_polar_deg,
        escaped_mask,
        detected_mask,
        polar_edges,
        polar=True,
    )
    exit_position_distribution = _angular_histogram(
        exit_position_polar_deg,
        escaped_mask,
        detected_mask,
        polar_edges,
        polar=True,
    )
    exit_position_azimuth_distribution = _angular_histogram(
        exit_position_azimuth_deg,
        escaped_mask,
        detected_mask,
        azimuth_edges,
        polar=False,
    )

    if position_histogram_bins < 1:
        raise ValueError("position_histogram_bins must be at least 1")
    position_values = np.concatenate(
        (x[escaped_mask], y[escaped_mask], z[escaped_mask])
    )
    position_values = position_values[np.isfinite(position_values)]
    coordinate_extent = (
        float(np.max(np.abs(position_values))) if position_values.size else 1.0
    )
    if coordinate_extent <= 0.0:
        coordinate_extent = 1.0
    coordinate_edges = np.linspace(
        -coordinate_extent, coordinate_extent, position_histogram_bins + 1
    )
    coordinate_rows = {
        "coordinate_bin_lower_m": coordinate_edges[:-1],
        "coordinate_bin_upper_m": coordinate_edges[1:],
        "coordinate_bin_centre_m": 0.5
        * (coordinate_edges[:-1] + coordinate_edges[1:]),
    }
    for name, values in (("x", x), ("y", y), ("z", z)):
        counts, _ = np.histogram(values[escaped_mask], bins=coordinate_edges)
        coordinate_rows[f"exit_{name}_count"] = counts.astype(np.int64)
        coordinate_rows[f"exit_{name}_fraction_of_escaped"] = (
            counts / float(n_escaped)
            if n_escaped > 0
            else np.zeros_like(counts, dtype=float)
        )
    exit_position_cartesian = pd.DataFrame(coordinate_rows)

    extinction_minus_scatter = extinction - scatter
    direction_norm_error = np.abs(direction_norm[valid_direction] - 1.0)
    absorption_accounting_ok = (
        np.all(extinction_minus_scatter[absorbed_mask] == 1)
        and np.all(extinction_minus_scatter[escaped_mask] == 0)
        and np.all(extinction_minus_scatter[truncated_mask] == 0)
    )
    if floc_extinction_available:
        floc_extinction_range_violations = int(
            np.sum(
                base_bookkeeping
                & (
                    (floc_extinction < 0)
                    | (floc_extinction > extinction)
                )
            )
        )
        floc_scatter_extinction_violations = int(
            np.sum(
                base_bookkeeping
                & (floc_extinction >= 0)
                & (floc_scatter > floc_extinction)
            )
        )
    else:
        floc_extinction_range_violations = 0
        floc_scatter_extinction_violations = 0

    integrity_rows = [
        {
            "check": "terminal_state_partition",
            "passed": bool(n_completed + n_unclassified == n_rays),
            "observed": float(n_completed + n_unclassified),
            "expected": float(n_rays),
            "tolerance": 0.0,
            "detail": "escaped + absorbed + truncated + unclassified must equal total",
        },
        {
            "check": "all_rays_classified",
            "passed": bool(n_unclassified == 0),
            "observed": float(n_unclassified),
            "expected": 0.0,
            "tolerance": 0.0,
            "detail": (
                "non-zero means invalid bookkeeping, conflicting flags, "
                "or missing exit data"
            ),
        },
        {
            "check": "no_absorbed_truncated_conflict",
            "passed": bool(not np.any(status_conflict)),
            "observed": float(np.sum(status_conflict)),
            "expected": 0.0,
            "tolerance": 0.0,
            "detail": "absorbed and truncated are mutually exclusive terminal states",
        },
        {
            "check": "scatter_not_greater_than_extinction",
            "passed": bool(np.all(scatter[terminal_mask] <= extinction[terminal_mask])),
            "observed": float(
                np.max(scatter[terminal_mask] - extinction[terminal_mask])
                if np.any(terminal_mask)
                else 0.0
            ),
            "expected": 0.0,
            "tolerance": 0.0,
            "detail": "each recorded scatter must originate from an extinction event",
        },
        {
            "check": "floc_scatter_not_greater_than_scatter",
            "passed": bool(np.all(floc_scatter[terminal_mask] <= scatter[terminal_mask])),
            "observed": float(
                np.max(floc_scatter[terminal_mask] - scatter[terminal_mask])
                if np.any(terminal_mask)
                else 0.0
            ),
            "expected": 0.0,
            "tolerance": 0.0,
            "detail": "floc_event_count is a subset of successful outer scatters",
        },
        {
            "check": "floc_extinction_count_in_range",
            "applicable": bool(floc_extinction_available),
            "passed": bool(
                not floc_extinction_available
                or floc_extinction_range_violations == 0
            ),
            "observed": (
                float(floc_extinction_range_violations)
                if floc_extinction_available
                else np.nan
            ),
            "expected": 0.0,
            "tolerance": 0.0,
            "detail": (
                "requires 0 <= floc_extinction_count <= extinction_count; "
                "not applicable to legacy files without the counter"
            ),
        },
        {
            "check": "floc_scatter_not_greater_than_floc_extinction",
            "applicable": bool(floc_extinction_available),
            "passed": bool(
                not floc_extinction_available
                or floc_scatter_extinction_violations == 0
            ),
            "observed": (
                float(floc_scatter_extinction_violations)
                if floc_extinction_available
                else np.nan
            ),
            "expected": 0.0,
            "tolerance": 0.0,
            "detail": (
                "successful floc scatters must be a subset of floc extinction "
                "selections; not applicable to legacy files"
            ),
        },
        {
            "check": "extinction_outcome_accounting",
            "passed": bool(absorption_accounting_ok),
            "observed": float(inferred_absorption_events),
            "expected": float(n_absorbed),
            "tolerance": 0.0,
            "detail": "extinction_count - scatter_count is one only for absorbed rays",
        },
        {
            "check": "detectors_only_receive_escaped_rays",
            "passed": bool(not np.any(valid_detector_index & ~escaped_mask)),
            "observed": float(np.sum(valid_detector_index & ~escaped_mask)),
            "expected": 0.0,
            "tolerance": 0.0,
            "detail": "detector_index must remain -1 for non-escaped rays",
        },
        {
            "check": "detector_indices_in_range",
            "passed": bool(not np.any(invalid_positive_detector_index)),
            "observed": float(np.sum(invalid_positive_detector_index)),
            "expected": 0.0,
            "tolerance": 0.0,
            "detail": "assigned detector indices must index detector_angles_deg",
        },
        {
            "check": "escaped_direction_available",
            "passed": bool(np.sum(valid_direction) == n_escaped),
            "observed": float(np.sum(valid_direction)),
            "expected": float(n_escaped),
            "tolerance": 0.0,
            "detail": "every escaped ray needs a finite, non-zero direction vector",
        },
        {
            "check": "escaped_direction_unit_norm",
            "passed": bool(
                direction_norm_error.size == 0
                or float(np.max(direction_norm_error)) <= 5.0e-5
            ),
            "observed": float(
                np.max(direction_norm_error) if direction_norm_error.size else 0.0
            ),
            "expected": 0.0,
            "tolerance": 5.0e-5,
            "detail": "CUDA direction vectors should remain normalized",
        },
        {
            "check": "completed_path_length_available",
            "passed": bool(np.sum(path_available) == n_completed),
            "observed": float(np.sum(path_available)),
            "expected": float(n_completed),
            "tolerance": 0.0,
            "detail": "path length must be written for escaped, absorbed, and truncated rays",
        },
    ]
    for row in integrity_rows:
        row.setdefault("applicable", True)
    integrity = pd.DataFrame(integrity_rows)
    applicable_integrity = integrity["applicable"].astype(bool)
    summary["integrity_checks_passed"] = int(
        np.sum(integrity["passed"].astype(bool) & applicable_integrity)
    )
    summary["integrity_checks_total"] = int(np.sum(applicable_integrity))
    summary["integrity_checks_catalogued"] = int(len(integrity))
    summary["integrity_checks_applicable"] = int(np.sum(applicable_integrity))
    summary["integrity_checks_not_applicable"] = int(
        len(integrity) - np.sum(applicable_integrity)
    )

    output_files = {
        "run_summary": _write_csv(
            pd.DataFrame([summary]), outdir_path, "run_summary", wl_nm
        ),
        "scatter_count_histogram": _write_csv(
            scatter_histogram, outdir_path, "scatter_count_histogram", wl_nm
        ),
        "path_length_histogram": _write_csv(
            path_histogram, outdir_path, "path_length_histogram", wl_nm
        ),
        "absorption_probability_vs_scatter_count": _write_csv(
            absorption_vs_scatter,
            outdir_path,
            "absorption_probability_vs_scatter_count",
            wl_nm,
        ),
        "detector_contribution_vs_scatter_order": _write_csv(
            detector_by_order,
            outdir_path,
            "detector_contribution_vs_scatter_order",
            wl_nm,
        ),
        "detector_efficiency": _write_csv(
            detector_efficiency, outdir_path, "detector_efficiency", wl_nm
        ),
        "scatter_statistics": _write_csv(
            scatter_statistics, outdir_path, "scatter_statistics", wl_nm
        ),
        "floc_internal_scatter_statistics": _write_csv(
            floc_internal_scatter_statistics,
            outdir_path,
            "floc_internal_scatter_statistics",
            wl_nm,
        ),
        "extinction_statistics": _write_csv(
            extinction_statistics, outdir_path, "extinction_statistics", wl_nm
        ),
        "exit_direction_distribution": _write_csv(
            exit_direction_distribution,
            outdir_path,
            "exit_direction_distribution",
            wl_nm,
        ),
        "exit_position_distribution": _write_csv(
            exit_position_distribution,
            outdir_path,
            "exit_position_distribution",
            wl_nm,
        ),
        "exit_position_azimuth_distribution": _write_csv(
            exit_position_azimuth_distribution,
            outdir_path,
            "exit_position_azimuth_distribution",
            wl_nm,
        ),
        "exit_position_cartesian_marginals": _write_csv(
            exit_position_cartesian,
            outdir_path,
            "exit_position_cartesian_marginals",
            wl_nm,
        ),
        "diagnostic_integrity": _write_csv(
            integrity, outdir_path, "diagnostic_integrity", wl_nm
        ),
    }

    result = dict(summary)
    result["output_files"] = output_files
    return result


def _read_first(
    hdf: h5py.File,
    names: Sequence[str],
    *,
    required: bool,
) -> Optional[np.ndarray]:
    for name in names:
        if name in hdf:
            return hdf[name][:]
    if required:
        raise KeyError(
            f"HDF5 file is missing required dataset; tried {', '.join(names)}"
        )
    return None


def _assign_nearest_detector(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    escaped: np.ndarray,
    detector_angles_deg: np.ndarray,
    detector_acceptance_deg: float,
) -> np.ndarray:
    detector_index = np.full(len(x), -1, dtype=np.int64)
    radius = np.sqrt(x * x + y * y + z * z)
    valid = escaped & np.isfinite(radius) & (radius > 0.0)
    if not np.any(valid) or detector_angles_deg.size == 0:
        return detector_index

    polar_deg = np.rad2deg(
        np.arccos(np.clip(y[valid] / radius[valid], -1.0, 1.0))
    )
    differences = np.abs(
        polar_deg[:, None] - detector_angles_deg[None, :]
    )
    nearest = np.argmin(differences, axis=1)
    accepted = (
        differences[np.arange(len(polar_deg)), nearest]
        <= detector_acceptance_deg
    )
    valid_indices = np.flatnonzero(valid)
    detector_index[valid_indices[accepted]] = nearest[accepted]
    return detector_index


def save_comprehensive_transport_diagnostics_from_hdf5(
    *,
    hdf5_path: Any,
    wl_nm: int,
    outdir: Any,
    detector_angles_deg: Any,
    detector_acceptance_deg: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the diagnostics on a CLARITAS_76/77 HDF5 ray file.

    CLARITAS_76 has no ``exit_z``, direction-vector, detector-index, truncation,
    or floc-extinction datasets. Missing z coordinates are treated as zero,
    legacy ``exit_dir = atan2(vy, vx)`` is converted back to a 2-D unit vector,
    and a nearest-detector assignment can be derived when an acceptance is
    supplied. The summary records unavailable fields and explicitly labels the
    successful-scatter fallback used for legacy floc/primary event fractions.
    The wrapper does not reinterpret invalid/absorbed rays as escaped.
    """

    hdf5_path = Path(hdf5_path)
    detector_angles = _as_1d(
        "detector_angles_deg", detector_angles_deg
    ).astype(np.float64, copy=False)

    with h5py.File(hdf5_path, "r") as hdf:
        scatter = _read_first(hdf, ("scatter_count",), required=True)
        extinction = _read_first(hdf, ("extinction_count",), required=True)
        floc = _read_first(hdf, ("floc_event_count",), required=True)
        floc_extinction = _read_first(
            hdf, ("floc_extinction_count",), required=False
        )
        absorbed = _read_first(hdf, ("absorbed", "absorbed_flag"), required=True)
        truncated = _read_first(hdf, ("truncated", "truncated_flag"), required=False)
        truncated_status_available = (
            "truncated" in hdf or "truncated_flag" in hdf
        )

        x = _read_first(hdf, ("exit_x",), required=True)
        y = _read_first(hdf, ("exit_y",), required=True)
        z = _read_first(hdf, ("exit_z",), required=False)
        path = _read_first(
            hdf, ("path_length", "ray_path_length", "exit_rpl"), required=True
        )
        vx = _read_first(hdf, ("exit_vx",), required=False)
        vy = _read_first(hdf, ("exit_vy",), required=False)
        vz = _read_first(hdf, ("exit_vz",), required=False)
        legacy_exit_dir = _read_first(hdf, ("exit_dir",), required=False)
        detector_index = _read_first(hdf, ("detector_index",), required=False)
        internal_scatter = _read_first(
            hdf, ("floc_internal_scatter_count",), required=False
        )

    n_rays = len(scatter)
    if z is None:
        z = np.zeros(n_rays, dtype=np.float64)
    if truncated is None:
        truncated = np.zeros(n_rays, dtype=np.int32)

    if vx is None or vy is None or vz is None:
        if legacy_exit_dir is None:
            vx = np.full(n_rays, np.nan, dtype=np.float64)
            vy = np.full(n_rays, np.nan, dtype=np.float64)
            vz = np.full(n_rays, np.nan, dtype=np.float64)
        else:
            legacy_exit_dir = np.asarray(legacy_exit_dir, dtype=np.float64)
            vx = np.cos(legacy_exit_dir)
            vy = np.sin(legacy_exit_dir)
            vz = np.zeros(n_rays, dtype=np.float64)
            invalid = ~np.isfinite(legacy_exit_dir)
            vx[invalid] = np.nan
            vy[invalid] = np.nan
            vz[invalid] = np.nan

    if detector_index is None:
        if detector_acceptance_deg is None:
            detector_index = np.full(n_rays, -1, dtype=np.int64)
        else:
            escaped = (
                (np.asarray(absorbed) == 0)
                & (np.asarray(truncated) == 0)
                & np.isfinite(x)
                & np.isfinite(y)
                & np.isfinite(z)
            )
            detector_index = _assign_nearest_detector(
                np.asarray(x, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
                np.asarray(z, dtype=np.float64),
                escaped,
                detector_angles,
                float(detector_acceptance_deg),
            )

    return save_comprehensive_transport_diagnostics(
        wl_nm=wl_nm,
        outdir=outdir,
        exit_x=x,
        exit_y=y,
        exit_z=z,
        exit_vx=vx,
        exit_vy=vy,
        exit_vz=vz,
        path_length=path,
        scatter_count=scatter,
        floc_event_count=floc,
        floc_extinction_count=floc_extinction,
        extinction_count=extinction,
        absorbed=absorbed,
        truncated=truncated,
        detector_index=detector_index,
        detector_angles_deg=detector_angles,
        floc_internal_scatter_count=internal_scatter,
        truncated_status_available=truncated_status_available,
        **kwargs,
    )


def _parse_detector_angles(value: str) -> np.ndarray:
    if ":" in value:
        fields = value.split(":")
        if len(fields) != 3:
            raise argparse.ArgumentTypeError(
                "detector range must use start:stop:step"
            )
        start, stop, step = (float(field) for field in fields)
        return np.arange(start, stop, step, dtype=np.float64)
    return np.asarray([float(field) for field in value.split(",")], dtype=np.float64)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate CLARITAS transport diagnostics from an HDF5 ray file"
    )
    parser.add_argument("hdf5_path", type=Path)
    parser.add_argument("--wl-nm", type=int, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument(
        "--detector-angles",
        type=_parse_detector_angles,
        default=np.arange(0.0, 180.0, 10.0),
        help="comma list or start:stop:step (default: 0:180:10)",
    )
    parser.add_argument(
        "--detector-acceptance-deg",
        type=float,
        default=6.5,
    )
    args = parser.parse_args(argv)

    result = save_comprehensive_transport_diagnostics_from_hdf5(
        hdf5_path=args.hdf5_path,
        wl_nm=args.wl_nm,
        outdir=args.outdir,
        detector_angles_deg=args.detector_angles,
        detector_acceptance_deg=args.detector_acceptance_deg,
    )
    print(
        f"Saved {len(result['output_files'])} diagnostic CSV files; "
        f"{result['integrity_checks_passed']}/{result['integrity_checks_total']} "
        "integrity checks passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
