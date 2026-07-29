from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import DATASETS, MATERIALS, SimulationConfig
from .physics import build_primary_medium
from .transport import simulate


def region_rmse(residual: np.ndarray, angles_deg: np.ndarray, low: float, high: float) -> float:
    mask = (angles_deg >= low) & (angles_deg <= high)
    return float(np.sqrt(np.mean(residual[mask] ** 2)))


def read_measured(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"detector angle"}
    if not required.issubset(frame.columns) or len(frame) != 18:
        raise ValueError("Measured CSV must contain 18 rows and a 'detector angle' column")
    return frame


def compare_curves(
    angles_deg: np.ndarray,
    measured_raw: np.ndarray,
    model_counts: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    measured_sum = float(np.sum(measured_raw))
    model_sum = float(np.sum(model_counts))
    measured_norm = measured_raw / measured_sum if measured_sum > 0.0 else np.zeros_like(measured_raw)
    model_norm = model_counts / model_sum if model_sum > 0.0 else np.zeros_like(model_counts, dtype=float)
    residual = model_norm - measured_norm
    table = pd.DataFrame({
        "detector_deg": angles_deg,
        "measured_raw": measured_raw,
        "measured_normalised": measured_norm,
        "model_raw_hits": model_counts,
        "model_normalised": model_norm,
        "normalised_residual": residual,
    })
    metrics = {
        "shape_rmse": float(np.sqrt(np.mean(residual**2))),
        "shape_mae": float(np.mean(np.abs(residual))),
        "forward_rmse_0_50": region_rmse(residual, angles_deg, 0.0, 50.0),
        "middle_rmse_60_110": region_rmse(residual, angles_deg, 60.0, 110.0),
        "rear_rmse_120_170": region_rmse(residual, angles_deg, 120.0, 170.0),
    }
    return table, metrics


def run_dataset(
    material_name: str,
    concentration: float,
    measured: pd.DataFrame,
    n_rays: int,
    seed: int,
    phase_grid_size: int,
    output_root: Path,
) -> Dict[str, object]:
    dataset_name = f"{material_name}_{str(concentration).replace('.', 'p')}gL"
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    measured_column = f"{material_name} {concentration:.1f} measured"
    if measured_column not in measured.columns:
        raise KeyError(f"Missing measured column {measured_column!r}")

    config = SimulationConfig(
        n_rays=n_rays,
        seed=seed,
        concentration_kg_m3=concentration,
        phase_grid_size=phase_grid_size,
    )
    t0 = time.time()
    medium = build_primary_medium(
        MATERIALS[material_name],
        concentration,
        config.wavelength_m,
        phase_grid_size=config.phase_grid_size,
    )
    build_seconds = time.time() - t0
    result = simulate(config, medium)
    elapsed_seconds = time.time() - t0

    angles = measured["detector angle"].to_numpy(dtype=np.float64)
    comparison, shape_metrics = compare_curves(
        angles,
        measured[measured_column].to_numpy(dtype=np.float64),
        result.detector_counts,
    )
    comparison.to_csv(output_dir / "measured_vs_model.csv", index=False)

    summary = result.summary()
    metrics: Dict[str, object] = {
        "dataset": dataset_name,
        "material": material_name,
        "concentration_kg_m3": concentration,
        "n_rays": n_rays,
        "seed": seed,
        "wavelength_m": config.wavelength_m,
        "sample_geometry": "sphere",
        "sample_radius_m": config.sample_radius_m,
        "phase_measure": "Mie intensity times sin(theta)",
        "mu_s_m_inv": medium.mu_s_m_inv,
        "mu_a_m_inv": medium.mu_a_m_inv,
        "mu_t_m_inv": medium.mu_t_m_inv,
        "phase_build_seconds": build_seconds,
        "elapsed_seconds": elapsed_seconds,
        **shape_metrics,
        **summary,
    }
    with (output_dir / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)

    path_frame = pd.DataFrame({
        "ray_index": np.arange(n_rays),
        "path_length_m": result.path_length_m,
        "scatter_count": result.scatter_count,
        "extinction_count": result.extinction_count,
        "absorbed": result.absorbed,
        "truncated": result.truncated,
        "exited": result.exited,
        "detector_index": result.detector_index,
    })
    path_frame.to_csv(output_dir / "ray_transport.csv", index=False)
    print(
        f"{dataset_name}: RMSE={metrics['shape_rmse']:.6f}, "
        f"detected={metrics['total_detected_fraction']:.6f}, "
        f"absorbed={metrics['absorbed_fraction']:.6f}, "
        f"elapsed={elapsed_seconds:.1f}s"
    )
    return metrics


def write_report(metrics: pd.DataFrame, output_path: Path, rays: int) -> None:
    lines = [
        "# Primary-only 3-D benchmark comparison",
        "",
        f"Each dataset used {rays:,} rays. Curves were scored for shape after unit-sum",
        "normalisation, while exit, absorption, and detection fractions were retained",
        "as separate absolute transport observables.",
        "",
        "| Dataset | Shape RMSE | Shape MAE | Forward RMSE | Middle RMSE | Rear RMSE | Detected fraction | Exit fraction | Absorbed fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['shape_rmse']:.6f} | {row['shape_mae']:.6f} | "
            f"{row['forward_rmse_0_50']:.6f} | {row['middle_rmse_60_110']:.6f} | "
            f"{row['rear_rmse_120_170']:.6f} | {row['total_detected_fraction']:.6f} | "
            f"{row['total_exit_fraction']:.6f} | {row['absorbed_fraction']:.6f} |"
        )
    lines.extend([
        "",
        "These are primary-only predictions, not fitted curves. Absolute detector",
        "fractions are model observables for the declared ideal spherical detector;",
        "they are not yet comparable to instrument power without source and detector",
        "calibration.",
    ])
    output_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run four CLARITAS primary-only 3-D benchmarks")
    parser.add_argument("--rays", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--phase-grid-size", type=int, default=5001)
    parser.add_argument("--measured", type=Path, default=Path("sediment_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("claritas_reference/benchmark_outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    measured = read_measured(args.measured)

    rows = []
    for index, (material, concentration) in enumerate(DATASETS):
        rows.append(run_dataset(
            material,
            concentration,
            measured,
            args.rays,
            args.seed + index,
            args.phase_grid_size,
            args.output,
        ))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "benchmark_summary.csv", index=False)
    write_report(summary, args.output / "BENCHMARK_REPORT.md", args.rays)
    print(f"Saved {args.output / 'benchmark_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

