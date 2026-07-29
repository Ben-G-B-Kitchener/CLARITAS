from __future__ import annotations

import argparse
import csv
from pathlib import Path
import unittest

import numpy as np

from .config import DetectorConfig, MATERIALS, SimulationConfig
from .geometry import assign_detector_bins, rotate_directions
from .physics import build_primary_medium, isotropic_phase_cdf, make_test_medium
from .transport import _sample_phase_angles, sample_event_bins, simulate


VALIDATION_ROWS = []


def record(name: str, metric: str, observed: float, expected: float, tolerance: float) -> None:
    error = abs(observed - expected)
    VALIDATION_ROWS.append({
        "test": name,
        "metric": metric,
        "observed": observed,
        "expected": expected,
        "absolute_error": error,
        "tolerance": tolerance,
        "passed": error <= tolerance,
    })


class ReferenceValidation(unittest.TestCase):
    def test_zero_concentration_ballistic(self) -> None:
        medium = make_test_medium(0.0, 0.0)
        result = simulate(SimulationConfig(n_rays=5000, seed=10), medium)
        summary = result.summary()
        record("zero_concentration", "ballistic_exit_fraction", summary["ballistic_exit_fraction"], 1.0, 0.0)
        record("zero_concentration", "absorbed_fraction", summary["absorbed_fraction"], 0.0, 0.0)
        self.assertEqual(summary["ballistic_exit_fraction"], 1.0)
        self.assertEqual(summary["absorbed_fraction"], 0.0)

    def test_pure_absorption_beer_lambert(self) -> None:
        mu_a = 8.0
        config = SimulationConfig(n_rays=100_000, seed=11)
        result = simulate(config, make_test_medium(0.0, mu_a))
        observed = result.summary()["total_exit_fraction"]
        expected = np.exp(-mu_a * 2.0 * config.sample_radius_m)
        tolerance = 4.0 * np.sqrt(expected * (1.0 - expected) / config.n_rays)
        record("pure_absorption", "transmission", observed, expected, tolerance)
        self.assertLessEqual(abs(observed - expected), tolerance)

    def test_isotropic_scattering_symmetry(self) -> None:
        config = SimulationConfig(n_rays=80_000, seed=12, sample_radius_m=0.01)
        result = simulate(config, make_test_medium(40.0, 0.0))
        scattered = result.exited & (result.scatter_count > 0)
        directions = result.exit_directions[scattered]
        mean_x = float(np.mean(directions[:, 0]))
        mean_z = float(np.mean(directions[:, 2]))
        tolerance = 0.02
        record("isotropic_scattering", "mean_exit_direction_x", mean_x, 0.0, tolerance)
        record("isotropic_scattering", "mean_exit_direction_z", mean_z, 0.0, tolerance)
        self.assertLess(abs(mean_x), tolerance)
        self.assertLess(abs(mean_z), tolerance)

    def test_mie_phase_sampling(self) -> None:
        material = MATERIALS["kaolin"]
        one_bin = type(material)(
            name="kaolin_single",
            diameters_m=np.array([1.005e-6]),
            mass_fraction_weights=np.array([1.0]),
            density_kg_m3=material.density_kg_m3,
            refractive_index_real=material.refractive_index_real,
            refractive_index_imag_k=material.refractive_index_imag_k,
        )
        medium = build_primary_medium(one_bin, 0.5, 622e-9, phase_grid_size=8001)
        rng = np.random.default_rng(13)
        sampled = _sample_phase_angles(rng, medium, np.zeros(250_000, dtype=np.int32))
        bins = np.linspace(0.0, np.pi, 181)
        empirical, _ = np.histogram(sampled, bins=bins)
        expected = np.interp(bins[1:], medium.theta_rad, medium.phase_cdf_by_bin[0])
        expected -= np.interp(bins[:-1], medium.theta_rad, medium.phase_cdf_by_bin[0])
        empirical = empirical / np.sum(empirical)
        residual = empirical - expected
        l1 = float(np.sum(np.abs(residual)))
        rms = float(np.sqrt(np.mean(residual**2)))
        maximum = float(np.max(np.abs(residual)))
        record("mie_phase_sampling", "L1", l1, 0.0, 0.025)
        record("mie_phase_sampling", "RMS", rms, 0.0, 0.001)
        record("mie_phase_sampling", "maximum", maximum, 0.0, 0.004)
        self.assertLess(l1, 0.025)
        self.assertLess(rms, 0.001)
        self.assertLess(maximum, 0.004)

    def test_low_optical_depth_ballistic_probability(self) -> None:
        config = SimulationConfig(n_rays=150_000, seed=14, sample_radius_m=0.01)
        mu_t = 2.0
        result = simulate(config, make_test_medium(mu_t, 0.0))
        observed = result.summary()["ballistic_exit_fraction"]
        expected = np.exp(-mu_t * 2.0 * config.sample_radius_m)
        tolerance = 4.0 * np.sqrt(expected * (1.0 - expected) / config.n_rays)
        record("low_optical_depth", "ballistic_probability", observed, expected, tolerance)
        self.assertLessEqual(abs(observed - expected), tolerance)

    def test_concentration_scaling_and_phase_invariance(self) -> None:
        material = MATERIALS["kaolin"]
        low = build_primary_medium(material, 0.5, 622e-9, phase_grid_size=2001)
        high = build_primary_medium(material, 4.0, 622e-9, phase_grid_size=2001)
        for name, low_value, high_value in (
            ("mu_s_ratio", low.mu_s_m_inv, high.mu_s_m_inv),
            ("mu_a_ratio", low.mu_a_m_inv, high.mu_a_m_inv),
            ("mu_t_ratio", low.mu_t_m_inv, high.mu_t_m_inv),
        ):
            ratio = high_value / low_value
            record("concentration_scaling", name, ratio, 8.0, 1e-12)
            self.assertAlmostEqual(ratio, 8.0, places=12)
        phase_difference = float(np.max(np.abs(low.phase_cdf_by_bin - high.phase_cdf_by_bin)))
        record("concentration_scaling", "phase_max_difference", phase_difference, 0.0, 0.0)
        self.assertEqual(phase_difference, 0.0)

    def test_detector_geometry_known_positions(self) -> None:
        detector = DetectorConfig()
        angles = np.deg2rad(np.array([0.0, 10.0, 90.0, 170.0, 180.0]))
        positions = np.column_stack((np.sin(angles), np.cos(angles), np.zeros_like(angles)))
        assigned = assign_detector_bins(positions, detector)
        expected = np.array([0, 1, 9, 17, -1], dtype=np.int32)
        matches = float(np.mean(assigned == expected))
        record("detector_geometry", "known_assignment_fraction", matches, 1.0, 0.0)
        np.testing.assert_array_equal(assigned, expected)

    def test_energy_accounting_and_reproducibility(self) -> None:
        config = SimulationConfig(n_rays=20_000, seed=15, sample_radius_m=0.01)
        medium = make_test_medium(20.0, 5.0)
        first = simulate(config, medium)
        second = simulate(config, medium)
        accounting = first.summary()["accounting_sum"]
        identical = float(
            np.array_equal(first.detector_counts, second.detector_counts) and
            np.array_equal(first.scatter_count, second.scatter_count) and
            np.array_equal(first.absorbed, second.absorbed)
        )
        record("energy_accounting", "outcome_sum", accounting, 1.0, 1e-15)
        record("reproducibility", "identical_outputs", identical, 1.0, 0.0)
        self.assertAlmostEqual(accounting, 1.0)
        self.assertEqual(identical, 1.0)

    def test_extinction_event_bin_sampling(self) -> None:
        rng = np.random.default_rng(17)
        probabilities = np.array([0.1, 0.3, 0.6])
        sampled = sample_event_bins(rng, np.cumsum(probabilities), 500_000)
        observed = np.bincount(sampled, minlength=3) / len(sampled)
        maximum_error = float(np.max(np.abs(observed - probabilities)))
        record("extinction_event_selection", "maximum_probability_error", maximum_error, 0.0, 0.0025)
        self.assertLess(maximum_error, 0.0025)

    def test_rotation_preserves_norm_and_isotropic_mean(self) -> None:
        rng = np.random.default_rng(16)
        n = 100_000
        directions = np.tile(np.array([[0.0, 1.0, 0.0]]), (n, 1))
        theta_grid = np.linspace(0.0, np.pi, 4001)
        theta = np.interp(rng.random(n), isotropic_phase_cdf(theta_grid), theta_grid)
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
        rotated = rotate_directions(directions, theta, phi)
        norm_error = float(np.max(np.abs(np.linalg.norm(rotated, axis=1) - 1.0)))
        mean_magnitude = float(np.linalg.norm(np.mean(rotated, axis=0)))
        record("direction_rotation", "max_norm_error", norm_error, 0.0, 1e-12)
        record("direction_rotation", "isotropic_mean_magnitude", mean_magnitude, 0.0, 0.01)
        self.assertLess(norm_error, 1e-12)
        self.assertLess(mean_magnitude, 0.01)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CLARITAS CPU 3-D reference transport")
    parser.add_argument("--output", type=Path, default=Path("claritas_reference/validation_outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceValidation)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    csv_path = args.output / "validation_results.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "test", "metric", "observed", "expected", "absolute_error", "tolerance", "passed"
        ])
        writer.writeheader()
        writer.writerows(VALIDATION_ROWS)
    passed = sum(bool(row["passed"]) for row in VALIDATION_ROWS)
    print(f"Validation metrics: {passed}/{len(VALIDATION_ROWS)} passed")
    print(f"Saved {csv_path}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
