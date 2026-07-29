import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from claritas_production_diagnostics import (
    save_comprehensive_transport_diagnostics,
    save_comprehensive_transport_diagnostics_from_hdf5,
)


class ComprehensiveDiagnosticsTests(unittest.TestCase):
    def test_terminal_and_event_accounting(self):
        nan = np.nan
        with tempfile.TemporaryDirectory() as directory:
            summary = save_comprehensive_transport_diagnostics(
                wl_nm=622,
                outdir=directory,
                exit_x=np.array([0.0, 1.0, 0.0, nan, nan, nan]),
                exit_y=np.array([1.0, 0.0, -1.0, nan, nan, nan]),
                exit_z=np.array([0.0, 0.0, 0.0, nan, nan, nan]),
                exit_vx=np.array([0.0, 1.0, 0.0, nan, nan, nan]),
                exit_vy=np.array([1.0, 0.0, -1.0, nan, nan, nan]),
                exit_vz=np.array([0.0, 0.0, 0.0, nan, nan, nan]),
                path_length=np.array([1.0, 1.1, 1.2, 0.5, 0.8, 0.9]),
                scatter_count=np.array([0, 1, 2, 0, 2, 3]),
                floc_event_count=np.array([0, 0, 1, 0, 1, 0]),
                floc_extinction_count=np.array([0, 0, 1, 0, 2, 1]),
                extinction_count=np.array([0, 1, 2, 1, 3, 3]),
                absorbed=np.array([0, 0, 0, 1, 1, 0]),
                truncated=np.array([0, 0, 0, 0, 0, 1]),
                detector_index=np.array([0, 1, 2, -1, -1, -1]),
                detector_angles_deg=np.array([0.0, 90.0, 180.0]),
                path_histogram_bins=4,
                angular_bin_width_deg=10.0,
                position_histogram_bins=4,
            )

            self.assertEqual(summary["total_ray_count"], 6)
            self.assertEqual(summary["escaped_ray_count"], 3)
            self.assertEqual(summary["absorbed_ray_count"], 2)
            self.assertEqual(summary["truncated_ray_count"], 1)
            self.assertEqual(summary["unclassified_ray_count"], 0)
            self.assertAlmostEqual(summary["escaped_fraction"], 0.5)
            self.assertAlmostEqual(summary["absorbed_fraction"], 2.0 / 6.0)
            self.assertAlmostEqual(summary["truncated_fraction"], 1.0 / 6.0)
            self.assertAlmostEqual(summary["ballistic_fraction"], 1.0 / 6.0)
            self.assertAlmostEqual(summary["single_scattered_fraction"], 1.0 / 6.0)
            self.assertAlmostEqual(summary["multiply_scattered_fraction"], 1.0 / 6.0)
            self.assertAlmostEqual(summary["floc_event_fraction"], 0.4)
            self.assertAlmostEqual(summary["primary_event_fraction"], 0.6)
            self.assertAlmostEqual(
                summary["successful_floc_scatter_fraction"], 0.25
            )
            self.assertAlmostEqual(
                summary["successful_primary_scatter_fraction"], 0.75
            )
            self.assertTrue(summary["floc_extinction_count_available"])
            self.assertTrue(
                summary["event_fraction_is_all_extinction_events"]
            )
            self.assertEqual(
                summary["event_fraction_basis"],
                "all_outer_extinction_events",
            )
            self.assertEqual(summary["inferred_absorption_extinction_event_count"], 2)
            self.assertEqual(
                summary["integrity_checks_passed"],
                summary["integrity_checks_total"],
            )

            output = Path(directory)
            run_summary = pd.read_csv(output / "run_summary_622nm.csv")
            self.assertEqual(int(run_summary.loc[0, "detected_unique_ray_count"]), 3)

            histogram = pd.read_csv(output / "scatter_count_histogram_622nm.csv")
            zero_scatter = histogram.loc[histogram["scatter_count"] == 0].iloc[0]
            self.assertEqual(int(zero_scatter["completed_ray_count"]), 2)
            self.assertEqual(int(zero_scatter["escaped_ray_count"]), 1)
            self.assertEqual(int(zero_scatter["absorbed_ray_count"]), 1)

            absorption = pd.read_csv(
                output / "absorption_probability_vs_scatter_count_622nm.csv"
            )
            zero_scatter = absorption.loc[absorption["scatter_count"] == 0].iloc[0]
            self.assertAlmostEqual(
                zero_scatter[
                    "ultimate_absorption_probability_given_scatter_count"
                ],
                0.5,
            )

            detector_order = pd.read_csv(
                output / "detector_contribution_vs_scatter_order_622nm.csv"
            )
            hit = detector_order.loc[
                (detector_order["detector_index"] == 2)
                & (detector_order["scatter_order"] == 2)
            ].iloc[0]
            self.assertEqual(int(hit["detected_ray_count"]), 1)

    def test_legacy_event_fraction_fallback_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = save_comprehensive_transport_diagnostics(
                wl_nm=622,
                outdir=directory,
                exit_x=np.array([0.0]),
                exit_y=np.array([1.0]),
                exit_z=np.array([0.0]),
                exit_vx=np.array([0.0]),
                exit_vy=np.array([1.0]),
                exit_vz=np.array([0.0]),
                path_length=np.array([1.0]),
                scatter_count=np.array([1]),
                floc_event_count=np.array([1]),
                extinction_count=np.array([1]),
                absorbed=np.array([0]),
                truncated=np.array([0]),
                detector_index=np.array([0]),
                detector_angles_deg=np.array([0.0]),
                path_histogram_bins=1,
                angular_bin_width_deg=90.0,
                position_histogram_bins=1,
            )

            self.assertFalse(summary["floc_extinction_count_available"])
            self.assertFalse(
                summary["event_fraction_is_all_extinction_events"]
            )
            self.assertEqual(
                summary["event_fraction_basis"],
                "successful_outer_scattering_events_fallback",
            )
            self.assertIn(
                "floc_extinction_count_unavailable",
                summary["event_fraction_provenance"],
            )
            self.assertEqual(summary["floc_event_fraction"], 1.0)
            self.assertEqual(summary["successful_floc_scatter_fraction"], 1.0)
            self.assertEqual(summary["integrity_checks_not_applicable"], 2)

    def test_floc_extinction_invariants_detect_invalid_records(self):
        with tempfile.TemporaryDirectory() as directory:
            save_comprehensive_transport_diagnostics(
                wl_nm=622,
                outdir=directory,
                exit_x=np.array([0.0, np.nan]),
                exit_y=np.array([1.0, np.nan]),
                exit_z=np.array([0.0, np.nan]),
                exit_vx=np.array([0.0, np.nan]),
                exit_vy=np.array([1.0, np.nan]),
                exit_vz=np.array([0.0, np.nan]),
                path_length=np.array([1.0, 0.5]),
                scatter_count=np.array([1, 0]),
                floc_event_count=np.array([1, 0]),
                floc_extinction_count=np.array([0, 2]),
                extinction_count=np.array([1, 1]),
                absorbed=np.array([0, 1]),
                truncated=np.array([0, 0]),
                detector_index=np.array([0, -1]),
                detector_angles_deg=np.array([0.0]),
                path_histogram_bins=1,
                angular_bin_width_deg=90.0,
                position_histogram_bins=1,
            )

            integrity = pd.read_csv(
                Path(directory) / "diagnostic_integrity_622nm.csv"
            ).set_index("check")
            self.assertFalse(
                bool(integrity.loc["floc_extinction_count_in_range", "passed"])
            )
            self.assertFalse(
                bool(
                    integrity.loc[
                        "floc_scatter_not_greater_than_floc_extinction",
                        "passed",
                    ]
                )
            )

    def test_hdf5_wrapper_detects_new_and_legacy_event_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            def write_ray_file(path, include_floc_extinction):
                with h5py.File(path, "w") as hdf:
                    arrays = {
                        "exit_x": np.array([0.0, np.nan], dtype=np.float32),
                        "exit_y": np.array([1.0, np.nan], dtype=np.float32),
                        "exit_z": np.array([0.0, np.nan], dtype=np.float32),
                        "exit_vx": np.array([0.0, np.nan], dtype=np.float32),
                        "exit_vy": np.array([1.0, np.nan], dtype=np.float32),
                        "exit_vz": np.array([0.0, np.nan], dtype=np.float32),
                        "exit_rpl": np.array([1.0, 0.5], dtype=np.float32),
                        "scatter_count": np.array([1, 0], dtype=np.int32),
                        "floc_event_count": np.array([1, 0], dtype=np.int32),
                        "extinction_count": np.array([1, 1], dtype=np.int32),
                        "absorbed": np.array([0, 1], dtype=np.int32),
                        "truncated": np.array([0, 0], dtype=np.int32),
                        "detector_index": np.array([0, -1], dtype=np.int32),
                    }
                    if include_floc_extinction:
                        arrays["floc_extinction_count"] = np.array(
                            [1, 1], dtype=np.int32
                        )
                    for name, values in arrays.items():
                        hdf.create_dataset(name, data=values)

            new_path = directory / "new.h5"
            write_ray_file(new_path, include_floc_extinction=True)
            new_summary = save_comprehensive_transport_diagnostics_from_hdf5(
                hdf5_path=new_path,
                wl_nm=622,
                outdir=directory / "new_outputs",
                detector_angles_deg=np.array([0.0]),
                detector_acceptance_deg=6.5,
                path_histogram_bins=1,
                angular_bin_width_deg=90.0,
                position_histogram_bins=1,
            )
            self.assertTrue(new_summary["floc_extinction_count_available"])
            self.assertEqual(new_summary["event_fraction_basis"], "all_outer_extinction_events")
            self.assertEqual(new_summary["floc_event_fraction"], 1.0)

            legacy_path = directory / "legacy.h5"
            write_ray_file(legacy_path, include_floc_extinction=False)
            legacy_summary = save_comprehensive_transport_diagnostics_from_hdf5(
                hdf5_path=legacy_path,
                wl_nm=622,
                outdir=directory / "legacy_outputs",
                detector_angles_deg=np.array([0.0]),
                detector_acceptance_deg=6.5,
                path_histogram_bins=1,
                angular_bin_width_deg=90.0,
                position_histogram_bins=1,
            )
            self.assertFalse(legacy_summary["floc_extinction_count_available"])
            self.assertEqual(
                legacy_summary["event_fraction_basis"],
                "successful_outer_scattering_events_fallback",
            )


if __name__ == "__main__":
    unittest.main()
