# CLARITAS_77: production CUDA transport with validated 3-D direction handling,
# solid-angle Mie sampling, deterministic per-ray RNG, and comprehensive diagnostics.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy.ma as ma
import miepython
import os
import cupy as cp
import time
import h5py
import hashlib

from claritas_production_diagnostics import save_comprehensive_transport_diagnostics
from claritas_measured_comparison import save_measured_comparison_if_available

SIMULATION_SEED = 20260727
GPU_MIN_CHUNK_RAYS = 100_000
GPU_MAX_CHUNK_RAYS = 1_000_000
SOURCE_MODE = "production_beta"  # "production_beta" or "reference_collimated"
PRODUCTION_BEAM_SIGMA_M = 10.0e-6
if SOURCE_MODE not in {"production_beta", "reference_collimated"}:
    raise ValueError(
        "SOURCE_MODE must be 'production_beta' or 'reference_collimated'"
    )

loess_diameter = np.array([1.729e-6, 1.981e-6, 2.269e-6, 2.599e-6, 2.976e-6, 3.409e-6, 3.905e-6,
                            4.472e-6, 5.122e-6, 5.867e-6, 6.72e-6, 7.697e-6, 8.816e-6, 10.097e-6,
                            11.565e-6, 13.246e-6, 15.172e-6, 17.377e-6, 19.904e-6, 22.797e-6,
                            26.111e-6, 29.907e-6, 34.255e-6, 39.234e-6, 44.938e-6, 51.471e-6,
                            58.953e-6, 67.523e-6, 77.34e-6, 88.583e-6, 101.46e-6, 116.21e-6,
                            133.103e-6, 152.453e-6, 174.616e-6, 200.000e-6, 229.075e-6, 262.376e-6])  # in meters
loess_weights = np.array([157, 227, 294, 354, 414, 487, 592, 747, 975, 1291, 1704, 2197, 2736,
                           3288, 3822, 4196, 4372, 4391, 4352, 4362, 4508, 4826, 5279, 5758,
                           6080, 6106, 5786, 5149, 4342, 3404, 2456, 1662, 1175, 858, 631, 463, 333, 230])  # relative weights

kaolin_diameter = np.array([0.172e-6, 0.197e-6, 0.226e-6, 0.259e-6, 0.296e-6, 0.339e-6, 0.389e-6,
                            0.445e-6, 0.51e-6, 0.584e-6, 0.669e-6, 0.766e-6, 0.877e-6, 1.005e-6,
                            1.151e-6, 1.318e-6, 1.51e-6, 1.729e-6, 1.981e-6, 2.269e-6,
                            2.599e-6, 2.976e-6, 3.409e-6, 3.905e-6, 4.472e-6, 5.122e-6,
                            5.867e-6, 6.72e-6, 7.697e-6, 8.816e-6, 10.097e-6, 11.565e-6,
                            13.246e-6, 15.172e-6, 17.377e-6, 19.904e-6, 22.797e-6])  # in meters
kaolin_weights = np.array([217, 547, 1112, 2032, 2985, 3492, 3308, 2644, 1893, 1300, 916, 700, 601,
                           584, 637, 757, 948, 1208, 1530, 1899, 2309, 2770, 3312, 3973,
                           4772, 5681, 6583, 7267, 7478, 7042, 6113, 5057, 3680, 2330, 1287, 631, 284])  # relative weights


null_diameter = np.array([0.0])
null_weights = np.array([0.0])

mass_concentration_g_per_L = 0.5



FLOC_ENABLED = True

FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M = np.array([
    2.0e-6,
    3.0e-6,
    4.0e-6,
    5.0e-6,
    6.5e-6,
    8.0e-6,
    10.0e-6,
    12.5e-6,
    15.0e-6,
    20.0e-6,
    25.0e-6,
    35.0e-6,
    50.0e-6
], dtype=np.float64)

FLOC_POOL_EFFECTIVE_DIAMETER_M = np.array([
    40.0e-6,
    50.0e-6,
    60.0e-6,
    70.0e-6,
    80.0e-6,
    90.0e-6,
    100.0e-6,
    110.0e-6,
    120.0e-6,
    150.0e-6,
    200.0e-6,
    250.0e-6
], dtype=np.float64)


if len(FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M) < 1:
    raise ValueError("At least one primary pooling band is required")
if len(FLOC_POOL_EFFECTIVE_DIAMETER_M) < 1:
    raise ValueError("At least one effective floc diameter bin is required")

FLOC_POOL_KERNEL_LOG_SIGMA = 0.35

FLOC_POOL_KERNEL_MIN_PROBABILITY = 0.0

FLOC_POOL_KERNEL_ENFORCE_FLOC_NOT_SMALLER_THAN_SOURCE = True

FLOC_FRACTAL_DIMENSION = 2.0

FLOC_COLLISION_LENGTH_M = 250.0e-6

#   sigma_floc = FLOC_SCATTER_EFFICIENCY * pi * r_floc^2
FLOC_SCATTER_EFFICIENCY = 1.0


PRIMARY_ROUGHNESS_STD_DEG = 0.0
FLOC_ROUGHNESS_STD_DEG = 0.0
PRIMARY_ROUGHNESS_STD_RAD = np.deg2rad(PRIMARY_ROUGHNESS_STD_DEG)
FLOC_ROUGHNESS_STD_RAD = np.deg2rad(FLOC_ROUGHNESS_STD_DEG)

particle_diameter_m = loess_diameter
particle_weights = loess_weights
particle_density_kg_per_m3 = 2600.0  # loess density


primary_particle_diameter_m = np.asarray(particle_diameter_m, dtype=np.float64)
primary_particle_weights = np.asarray(particle_weights, dtype=np.float64)
if (
    primary_particle_diameter_m.ndim != 1
    or primary_particle_weights.shape != primary_particle_diameter_m.shape
    or primary_particle_diameter_m.size == 0
    or not np.all(np.isfinite(primary_particle_diameter_m))
    or np.any(primary_particle_diameter_m <= 0.0)
):
    raise ValueError("Primary diameters must be a finite, positive 1-D table")
if (
    not np.all(np.isfinite(primary_particle_weights))
    or np.any(primary_particle_weights < 0.0)
    or np.sum(primary_particle_weights, dtype=np.float64) <= 0.0
):
    raise ValueError("Primary PSD weights must be finite and nonnegative")
if (
    not np.isfinite(mass_concentration_g_per_L)
    or mass_concentration_g_per_L < 0.0
):
    raise ValueError("Mass concentration must be finite and nonnegative")
if (
    not np.isfinite(particle_density_kg_per_m3)
    or particle_density_kg_per_m3 <= 0.0
):
    raise ValueError("Particle density must be finite and positive")
primary_particle_weights /= np.sum(primary_particle_weights)

primary_particle_density_by_bin_kg_per_m3 = np.full_like(
    primary_particle_diameter_m,
    particle_density_kg_per_m3,
    dtype=np.float64
)

primary_particle_radius_m = primary_particle_diameter_m / 2.0
primary_particle_volumes_m3 = (4.0 / 3.0) * np.pi * primary_particle_radius_m**3
primary_particle_masses_kg = primary_particle_volumes_m3 * particle_density_kg_per_m3

mass_concentration_kg_per_m3_prefloc = mass_concentration_g_per_L

PSD_WEIGHT_MODE = "mass_fraction"  # options: "mass_fraction", "number_fraction"
#PSD_WEIGHT_MODE = "number_fraction"  # options: "mass_fraction", "number_fraction"


if PSD_WEIGHT_MODE == "mass_fraction":
    primary_particle_number_density_by_bin = (
        mass_concentration_kg_per_m3_prefloc *
        primary_particle_weights /
        primary_particle_masses_kg
    )
elif PSD_WEIGHT_MODE == "number_fraction":
    primary_number_weights = primary_particle_weights / np.sum(primary_particle_weights)
    primary_average_particle_mass = np.sum(
        primary_number_weights * primary_particle_masses_kg
    )
    primary_total_number_density = (
        mass_concentration_kg_per_m3_prefloc /
        primary_average_particle_mass
    )
    primary_particle_number_density_by_bin = (
        primary_total_number_density *
        primary_number_weights
    )
else:
    raise ValueError(
        "PSD_WEIGHT_MODE must be either 'mass_fraction' or 'number_fraction'"
    )

if FLOC_ENABLED:
    primary_bin_floc_band_index = np.full(
        primary_particle_diameter_m.shape,
        -1,
        dtype=np.int32
    )

    previous_edge = 0.0
    for band_idx, band_edge in enumerate(FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M):
        band_mask = (
            (primary_particle_diameter_m <= band_edge) &
            (primary_particle_diameter_m > previous_edge)
        )
        primary_bin_floc_band_index[band_mask] = band_idx
        previous_edge = band_edge

    primary_bin_is_pooled_into_floc = primary_bin_floc_band_index >= 0

    eligible_primary_number_density_per_m3 = np.sum(
        primary_particle_number_density_by_bin[primary_bin_is_pooled_into_floc]
    )

    if eligible_primary_number_density_per_m3 > 0.0:
        eligible_primary_spacing_m = (
            eligible_primary_number_density_per_m3 ** (-1.0 / 3.0)
        )

        floc_encounter_probability = (
            FLOC_COLLISION_LENGTH_M /
            (FLOC_COLLISION_LENGTH_M + eligible_primary_spacing_m)
        )

        floc_mass_fraction = floc_encounter_probability
    else:
        eligible_primary_spacing_m = np.inf
        floc_encounter_probability = 0.0
        floc_mass_fraction = 0.0

    floc_encounter_probability = np.clip(floc_encounter_probability, 0.0, 1.0)
    floc_mass_fraction = np.clip(floc_mass_fraction, 0.0, 1.0)

    floc_property_state = 0.0

    floc_reference_primary_diameter_m = FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M.copy()
    floc_effective_diameter_by_band_m = FLOC_POOL_EFFECTIVE_DIAMETER_M.copy()

    n_source_bands = len(FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M)
    n_floc_bins = len(FLOC_POOL_EFFECTIVE_DIAMETER_M)

    if n_source_bands == n_floc_bins:
        preferred_floc_diameter_by_source_band_m = FLOC_POOL_EFFECTIVE_DIAMETER_M.copy()
    else:
        src_pos = np.linspace(0.0, 1.0, n_source_bands)
        floc_pos = np.linspace(0.0, 1.0, n_floc_bins)
        preferred_floc_diameter_by_source_band_m = np.exp(
            np.interp(
                src_pos,
                floc_pos,
                np.log(FLOC_POOL_EFFECTIVE_DIAMETER_M)
            )
        )

    sigma_log = max(float(FLOC_POOL_KERNEL_LOG_SIGMA), 1.0e-12)
    floc_pooling_kernel = np.zeros((n_source_bands, n_floc_bins), dtype=np.float64)

    for source_band_idx in range(n_source_bands):
        log_ratio = np.log(
            FLOC_POOL_EFFECTIVE_DIAMETER_M /
            max(preferred_floc_diameter_by_source_band_m[source_band_idx], 1.0e-30)
        )
        row = np.exp(-0.5 * (log_ratio / sigma_log) ** 2)
        row[~np.isfinite(row)] = 0.0
        row = np.maximum(row, 0.0)

        if FLOC_POOL_KERNEL_ENFORCE_FLOC_NOT_SMALLER_THAN_SOURCE:
            source_diameter_limit_m = FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M[source_band_idx]
            row[FLOC_POOL_EFFECTIVE_DIAMETER_M < source_diameter_limit_m] = 0.0

        if FLOC_POOL_KERNEL_MIN_PROBABILITY > 0.0:
            row[row < FLOC_POOL_KERNEL_MIN_PROBABILITY] = 0.0

        if np.sum(row) <= 0.0:
            if FLOC_POOL_KERNEL_ENFORCE_FLOC_NOT_SMALLER_THAN_SOURCE:
                valid_floc_bins = np.where(
                    FLOC_POOL_EFFECTIVE_DIAMETER_M >=
                    FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M[source_band_idx]
                )[0]
            else:
                valid_floc_bins = np.arange(n_floc_bins)

            if valid_floc_bins.size <= 0:
                valid_floc_bins = np.array([n_floc_bins - 1], dtype=np.int64)

            nearest_local = int(np.argmin(np.abs(log_ratio[valid_floc_bins])))
            nearest = int(valid_floc_bins[nearest_local])
            row[nearest] = 1.0

        floc_pooling_kernel[source_band_idx, :] = row / np.sum(row)

    floc_reference_primary_mass_kg = (
        particle_density_kg_per_m3 *
        (np.pi / 6.0) *
        floc_reference_primary_diameter_m**3
    )

    floc_mass_by_source_band_and_floc_bin_kg = np.zeros(
        (n_source_bands, n_floc_bins),
        dtype=np.float64
    )
    floc_effective_density_by_source_band_and_floc_bin_kg_per_m3 = np.zeros(
        (n_source_bands, n_floc_bins),
        dtype=np.float64
    )

    for source_band_idx in range(n_source_bands):
        d0 = max(float(floc_reference_primary_diameter_m[source_band_idx]), 1.0e-30)
        m0 = float(floc_reference_primary_mass_kg[source_band_idx])

        floc_mass_by_source_band_and_floc_bin_kg[source_band_idx, :] = (
            m0 *
            (FLOC_POOL_EFFECTIVE_DIAMETER_M / d0) ** FLOC_FRACTAL_DIMENSION
        )

        floc_volume_by_bin_m3 = (
            (np.pi / 6.0) *
            FLOC_POOL_EFFECTIVE_DIAMETER_M**3
        )

        floc_effective_density_by_source_band_and_floc_bin_kg_per_m3[source_band_idx, :] = (
            floc_mass_by_source_band_and_floc_bin_kg[source_band_idx, :] /
            floc_volume_by_bin_m3
        )

    floc_reference_source_index_by_floc_bin = np.zeros(n_floc_bins, dtype=np.int32)
    if n_source_bands == n_floc_bins:
        floc_reference_source_index_by_floc_bin = np.arange(n_floc_bins, dtype=np.int32)
    else:
        src_pos = np.linspace(0.0, 1.0, n_source_bands)
        floc_pos = np.linspace(0.0, 1.0, n_floc_bins)
        for floc_bin_idx, pos in enumerate(floc_pos):
            floc_reference_source_index_by_floc_bin[floc_bin_idx] = int(
                np.argmin(np.abs(src_pos - pos))
            )

    floc_mass_by_band_kg = np.asarray([
        floc_mass_by_source_band_and_floc_bin_kg[
            floc_reference_source_index_by_floc_bin[j],
            j
        ]
        for j in range(n_floc_bins)
    ], dtype=np.float64)

    floc_effective_density_by_band_kg_per_m3 = np.asarray([
        floc_effective_density_by_source_band_and_floc_bin_kg_per_m3[
            floc_reference_source_index_by_floc_bin[j],
            j
        ]
        for j in range(n_floc_bins)
    ], dtype=np.float64)

    effective_diameters = []
    effective_weights = []
    effective_particle_masses_kg = []
    effective_densities = []
    effective_is_floc = []
    effective_source_primary_min = []
    effective_source_primary_max = []
    effective_source_primary_mass_fraction = []
    effective_floc_band_index = []
    effective_bin_kind = []

    for source_band_idx in range(n_source_bands):
        band_mask = primary_bin_floc_band_index == source_band_idx
        band_mass_fraction = np.sum(primary_particle_weights[band_mask])

        if band_mass_fraction <= 0.0:
            continue

        pooled_band_mass_fraction = band_mass_fraction * floc_mass_fraction
        residual_band_mass_fraction = band_mass_fraction * (1.0 - floc_mass_fraction)

        if pooled_band_mass_fraction > 0.0:
            source_min = np.min(primary_particle_diameter_m[band_mask])
            source_max = np.max(primary_particle_diameter_m[band_mask])

            for floc_bin_idx in range(n_floc_bins):
                kernel_probability = floc_pooling_kernel[source_band_idx, floc_bin_idx]
                pooled_cell_mass_fraction = pooled_band_mass_fraction * kernel_probability

                if pooled_cell_mass_fraction <= 0.0:
                    continue

                effective_diameters.append(FLOC_POOL_EFFECTIVE_DIAMETER_M[floc_bin_idx])
                effective_weights.append(pooled_cell_mass_fraction)
                effective_particle_masses_kg.append(
                    floc_mass_by_source_band_and_floc_bin_kg[
                        source_band_idx,
                        floc_bin_idx
                    ]
                )
                effective_densities.append(
                    floc_effective_density_by_source_band_and_floc_bin_kg_per_m3[
                        source_band_idx,
                        floc_bin_idx
                    ]
                )
                effective_is_floc.append(True)
                effective_source_primary_min.append(source_min)
                effective_source_primary_max.append(source_max)
                effective_source_primary_mass_fraction.append(pooled_cell_mass_fraction)
                effective_floc_band_index.append(floc_bin_idx)
                effective_bin_kind.append("pooled_kernel_floc")

        if residual_band_mass_fraction > 0.0:
            band_weights = primary_particle_weights[band_mask]
            band_weight_sum = np.sum(band_weights)

            if band_weight_sum > 0.0:
                residual_weights_by_primary_bin = (
                    residual_band_mass_fraction *
                    band_weights /
                    band_weight_sum
                )

                for d_primary, w_residual in zip(
                    primary_particle_diameter_m[band_mask],
                    residual_weights_by_primary_bin
                ):
                    if w_residual <= 0.0:
                        continue

                    primary_mass_kg = (
                        particle_density_kg_per_m3 *
                        (np.pi / 6.0) *
                        d_primary**3
                    )

                    effective_diameters.append(d_primary)
                    effective_weights.append(w_residual)
                    effective_particle_masses_kg.append(primary_mass_kg)
                    effective_densities.append(particle_density_kg_per_m3)
                    effective_is_floc.append(False)
                    effective_source_primary_min.append(d_primary)
                    effective_source_primary_max.append(d_primary)
                    effective_source_primary_mass_fraction.append(w_residual)
                    effective_floc_band_index.append(-1)
                    effective_bin_kind.append("residual_primary")

    nonfloc_mask = ~primary_bin_is_pooled_into_floc
    for d_primary, w_primary in zip(
        primary_particle_diameter_m[nonfloc_mask],
        primary_particle_weights[nonfloc_mask]
    ):
        if w_primary <= 0.0:
            continue

        primary_mass_kg = (
            particle_density_kg_per_m3 *
            (np.pi / 6.0) *
            d_primary**3
        )

        effective_diameters.append(d_primary)
        effective_weights.append(w_primary)
        effective_particle_masses_kg.append(primary_mass_kg)
        effective_densities.append(particle_density_kg_per_m3)
        effective_is_floc.append(False)
        effective_source_primary_min.append(d_primary)
        effective_source_primary_max.append(d_primary)
        effective_source_primary_mass_fraction.append(w_primary)
        effective_floc_band_index.append(-1)
        effective_bin_kind.append("unchanged_primary")

    particle_diameter_m = np.asarray(effective_diameters, dtype=np.float64)
    particle_weights = np.asarray(effective_weights, dtype=np.float64)
    particle_weights /= np.sum(particle_weights)

    particle_mass_by_bin_kg = np.asarray(
        effective_particle_masses_kg,
        dtype=np.float64
    )

    particle_density_by_bin_kg_per_m3 = np.asarray(
        effective_densities,
        dtype=np.float64
    )

    particle_is_floc = np.asarray(effective_is_floc, dtype=bool)

    source_primary_min_diameter_m = np.asarray(
        effective_source_primary_min,
        dtype=np.float64
    )
    source_primary_max_diameter_m = np.asarray(
        effective_source_primary_max,
        dtype=np.float64
    )
    source_primary_mass_fraction = np.asarray(
        effective_source_primary_mass_fraction,
        dtype=np.float64
    )
    floc_band_index_by_effective_bin = np.asarray(
        effective_floc_band_index,
        dtype=np.int32
    )
    effective_bin_kind = np.asarray(effective_bin_kind, dtype=object)

else:
    primary_bin_floc_band_index = np.full(
        primary_particle_diameter_m.shape,
        -1,
        dtype=np.int32
    )
    primary_bin_is_pooled_into_floc = np.zeros_like(
        primary_particle_diameter_m,
        dtype=bool
    )
    eligible_primary_number_density_per_m3 = 0.0
    eligible_primary_spacing_m = np.inf
    floc_encounter_probability = 0.0
    floc_property_state = 0.0
    floc_mass_fraction = 0.0

    floc_reference_primary_diameter_m = np.array([], dtype=np.float64)
    floc_reference_primary_mass_kg = np.array([], dtype=np.float64)
    floc_mass_by_band_kg = np.array([], dtype=np.float64)
    floc_effective_density_by_band_kg_per_m3 = np.array([], dtype=np.float64)
    floc_pooling_kernel = np.zeros((0, 0), dtype=np.float64)
    floc_mass_by_source_band_and_floc_bin_kg = np.zeros((0, 0), dtype=np.float64)
    floc_effective_density_by_source_band_and_floc_bin_kg_per_m3 = np.zeros((0, 0), dtype=np.float64)

    particle_diameter_m = primary_particle_diameter_m.copy()
    particle_weights = primary_particle_weights.copy()

    particle_mass_by_bin_kg = (
        particle_density_kg_per_m3 *
        (np.pi / 6.0) *
        particle_diameter_m**3
    )

    particle_density_by_bin_kg_per_m3 = np.full_like(
        particle_diameter_m,
        particle_density_kg_per_m3,
        dtype=np.float64
    )

    particle_is_floc = np.zeros_like(particle_diameter_m, dtype=bool)

    source_primary_min_diameter_m = primary_particle_diameter_m.copy()
    source_primary_max_diameter_m = primary_particle_diameter_m.copy()
    source_primary_mass_fraction = primary_particle_weights.copy()
    floc_band_index_by_effective_bin = np.full_like(
        particle_diameter_m,
        -1,
        dtype=np.int32
    )
    effective_bin_kind = np.asarray(
        ["unchanged_primary"] * len(particle_diameter_m),
        dtype=object
    )
    
particle_radius_m = particle_diameter_m / 2

source_primary_geometric_mid_diameter_m = np.sqrt(
    source_primary_min_diameter_m * source_primary_max_diameter_m
)

floc_diameter_multiplier_by_bin = np.ones_like(
    particle_diameter_m,
    dtype=np.float64
)

valid_source = source_primary_geometric_mid_diameter_m > 0.0
floc_diameter_multiplier_by_bin[valid_source] = (
    particle_diameter_m[valid_source] /
    source_primary_geometric_mid_diameter_m[valid_source]
)

floc_effective_density_by_bin_kg_per_m3 = particle_density_by_bin_kg_per_m3.copy()
target_floc_density_by_bin = particle_density_by_bin_kg_per_m3.copy()

effective_particle_density_kg_per_m3 = np.sum(
    particle_weights * particle_density_by_bin_kg_per_m3
)

n_particle = 1.59  # real refractive index of solid primary particle material

PRIMARY_REFRACTIVE_INDEX_IMAG_K = 0.001

FLOC_ABSORPTION_K_MULTIPLIER = 1.0
FLOC_ABSORPTION_PATH_FACTOR = 1.0

n_medium = 1.33
n_external = 1.0  # metadata only; boundary Fresnel/refraction is not implemented


def maxwell_garnett_effective_index(
    matrix_index_complex,
    inclusion_index_complex,
    inclusion_volume_fraction
):
    phi = np.clip(inclusion_volume_fraction, 0.0, 1.0)

    eps_m = matrix_index_complex ** 2
    eps_i = inclusion_index_complex ** 2

    numerator = eps_i + 2.0 * eps_m + 2.0 * phi * (eps_i - eps_m)
    denominator = eps_i + 2.0 * eps_m - phi * (eps_i - eps_m)

    eps_eff = eps_m * numerator / denominator
    n_eff = np.sqrt(eps_eff)

    n_eff = np.where(
        np.imag(n_eff) > 0.0,
        np.conjugate(n_eff),
        n_eff
    )

    return n_eff



solid_primary_complex_index = complex(
    n_particle,
    -PRIMARY_REFRACTIVE_INDEX_IMAG_K
)

medium_complex_index = complex(n_medium, 0.0)

solid_volume_fraction_by_bin = np.clip(
    particle_density_by_bin_kg_per_m3 / particle_density_kg_per_m3,
    0.0,
    1.0
)

particle_complex_refractive_index_by_bin = np.full(
    particle_diameter_m.shape,
    solid_primary_complex_index,
    dtype=np.complex128
)

if np.any(particle_is_floc):
    particle_complex_refractive_index_by_bin[particle_is_floc] = (
        maxwell_garnett_effective_index(
            medium_complex_index,
            solid_primary_complex_index,
            solid_volume_fraction_by_bin[particle_is_floc]
        )
    )

particle_refractive_index_by_bin = np.real(
    particle_complex_refractive_index_by_bin
).astype(np.float64)

particle_refractive_index_imag_k_by_bin = np.maximum(
    -np.imag(particle_complex_refractive_index_by_bin),
    0.0
).astype(np.float64)

        
detector_angles = np.arange(0, 180, 10)   # centres in degrees 0 to 170 - matches TARDIIS & CLARITAS outputs
#detector_angles = np.arange(0, 190, 10)   # centres in degrees 0 to 180
detector_acceptance_deg = 6.5  # degrees
#detector_acceptance_deg = 10.0  # degrees
theta_deg = np.linspace(0.0, 180.0, 180_001, dtype=np.float64)


alpha1 = 1.0
alpha2 = 100.0

#wavelengths = [200e-9, 622e-9, 850e-9]  # in meters
#wavelengths = [200e-9, 622e-9, 950e-9]  # in meters
wavelengths = [622e-9]  # in meters


#R_REAL = 0.049    # Sample radius (m) TARDIIS
R_REAL = 0.049    # Sample radius (m)
RAY_OFFSET = 0.005  # Ray initial y-offset (m)
VISUAL_SCALE = 1.0
VIS_SIZE = 4096      # Heatmap resolution
N_RAYS = 1_000_00  # number of rays to simulate
MAX_EXTINCTIONS = 10000

OUTDIR = "."
os.makedirs(OUTDIR, exist_ok=True)

mass_concentration_kg_per_m3 = mass_concentration_g_per_L

if PSD_WEIGHT_MODE == "mass_fraction":
    particle_number_density_by_bin = (
        mass_concentration_kg_per_m3 *
        particle_weights /
        particle_mass_by_bin_kg
    )

elif PSD_WEIGHT_MODE == "number_fraction":
    number_weights = particle_weights / np.sum(particle_weights)

    average_particle_mass_from_number_distribution = np.sum(
        number_weights * particle_mass_by_bin_kg
    )

    total_number_density = (
        mass_concentration_kg_per_m3 /
        average_particle_mass_from_number_distribution
    )

    particle_number_density_by_bin = total_number_density * number_weights

else:
    raise ValueError(
        "PSD_WEIGHT_MODE must be either 'mass_fraction' or 'number_fraction'"
    )

n_particles_per_m3 = np.sum(particle_number_density_by_bin)
average_particle_separation_m = n_particles_per_m3 ** (-1.0 / 3.0)
average_particle_mass = (
    mass_concentration_kg_per_m3 / n_particles_per_m3
    if n_particles_per_m3 > 0.0
    else 0.0
)

pm = particle_mass_by_bin_kg
particle_mass = average_particle_mass

# Use the first configured wavelength for the scalar transport setup.
scatter_probability_wavelength = wavelengths[0]

#   sigma_s = FLOC_SCATTER_EFFICIENCY * pi*r_floc^2
sigma_s = []
sigma_a = []
sigma_t = []
single_scattering_albedo_by_bin = []
g_by_bin = []
floc_absorption_tau_by_bin = np.zeros_like(particle_diameter_m, dtype=np.float64)
floc_absorption_coefficient_by_bin_per_m = np.zeros_like(particle_diameter_m, dtype=np.float64)
floc_mean_chord_length_by_bin_m = np.zeros_like(particle_diameter_m, dtype=np.float64)

for idx, (r, refr_index, is_floc) in enumerate(zip(
    particle_radius_m,
    particle_refractive_index_by_bin,
    particle_is_floc
)):
    area = np.pi * r**2

    if is_floc:
        sigma_s_this = FLOC_SCATTER_EFFICIENCY * area

        solid_volume_fraction = solid_volume_fraction_by_bin[idx]

        # detector response or on a material-specific scatter multiplier.
        k_eff = (
            particle_refractive_index_imag_k_by_bin[idx] *
            FLOC_ABSORPTION_K_MULTIPLIER
        )

        alpha_abs_per_m = (
            4.0 * np.pi * k_eff / scatter_probability_wavelength
            if scatter_probability_wavelength > 0.0 else 0.0
        )
        alpha_abs_per_m = max(float(alpha_abs_per_m), 0.0)

        mean_chord_m = (2.0 / 3.0) * particle_diameter_m[idx]
        internal_path_m = FLOC_ABSORPTION_PATH_FACTOR * mean_chord_m

        tau_abs = alpha_abs_per_m * internal_path_m
        tau_abs = max(float(tau_abs), 0.0)

        floc_absorption_coefficient_by_bin_per_m[idx] = alpha_abs_per_m
        floc_mean_chord_length_by_bin_m[idx] = internal_path_m
        floc_absorption_tau_by_bin[idx] = tau_abs

        absorption_probability_this = 1.0 - np.exp(-tau_abs)

        sigma_a_this = area * absorption_probability_this
        sigma_t_this = sigma_s_this + sigma_a_this

        sigma_s.append(sigma_s_this)
        sigma_a.append(sigma_a_this)
        sigma_t.append(sigma_t_this)
        single_scattering_albedo_by_bin.append(
            sigma_s_this / sigma_t_this if sigma_t_this > 0.0 else 1.0
        )

        g_by_bin.append(0.0)

    else:
        m_rel = particle_complex_refractive_index_by_bin[idx] / n_medium
        x_mie = 2.0 * np.pi * n_medium * r / scatter_probability_wavelength

        qext, qsca, qback, g = miepython.efficiencies_mx(
            m_rel,
            x_mie
        )

        sigma_t_this = max(float(np.real(qext)) * area, 0.0)
        sigma_s_this = min(
            max(float(np.real(qsca)) * area, 0.0),
            sigma_t_this
        )
        sigma_a_this = sigma_t_this - sigma_s_this

        sigma_s.append(sigma_s_this)
        sigma_a.append(sigma_a_this)
        sigma_t.append(sigma_t_this)
        single_scattering_albedo_by_bin.append(
            sigma_s_this / sigma_t_this if sigma_t_this > 0.0 else 1.0
        )
        g_by_bin.append(float(np.real(g)))

sigma_s = np.asarray(sigma_s, dtype=np.float64)
sigma_a = np.asarray(sigma_a, dtype=np.float64)
sigma_t = np.asarray(sigma_t, dtype=np.float64)
single_scattering_albedo_by_bin = np.asarray(
    single_scattering_albedo_by_bin,
    dtype=np.float64
)
g_by_bin = np.asarray(g_by_bin, dtype=np.float64)

mu_s_by_bin = particle_number_density_by_bin * sigma_s
mu_a_by_bin = particle_number_density_by_bin * sigma_a
mu_t_by_bin = particle_number_density_by_bin * sigma_t

mu_s = np.sum(mu_s_by_bin)
mu_a = np.sum(mu_a_by_bin)
mu_t = np.sum(mu_t_by_bin)

mu_s_prime_by_bin = mu_s_by_bin * (1.0 - g_by_bin)
mu_s_prime = np.sum(mu_s_prime_by_bin)

if mu_s > 0.0:
    g_eff = np.sum(mu_s_by_bin * g_by_bin) / mu_s
else:
    g_eff = 0.0

medium_single_scattering_albedo = mu_s / mu_t if mu_t > 0.0 else 1.0

MEAN_FREE_PATH_M = 1.0 / mu_t if mu_t > 0.0 else np.inf
MEAN_SCATTERING_PATH_M = 1.0 / mu_s if mu_s > 0.0 else np.inf
MEAN_ABSORPTION_PATH_M = 1.0 / mu_a if mu_a > 0.0 else np.inf
TRANSPORT_MEAN_FREE_PATH_M = (
    1.0 / mu_s_prime if mu_s_prime > 0.0 else np.inf
)

extinction_event_strength_by_bin = particle_number_density_by_bin * sigma_t

if not np.allclose(
    extinction_event_strength_by_bin,
    mu_t_by_bin,
    rtol=1.0e-12,
    atol=0.0
):
    raise RuntimeError(
        "Extinction event strengths and mu_t_by_bin disagree. "
        "Check sigma_t / number-density construction."
    )

particle_event_weights = np.zeros_like(mu_t_by_bin, dtype=np.float64)

if mu_t > 0.0:
    particle_event_weights = extinction_event_strength_by_bin / mu_t
    particle_event_cdf = np.cumsum(particle_event_weights)
    particle_event_cdf /= particle_event_cdf[-1]
else:
    # The kernel bypasses event selection when mu_t is zero, but retain a
    # formally valid placeholder table so the ballistic case passes the same
    # host-side CDF validation and device interface.
    particle_event_cdf = np.linspace(
        1.0 / len(particle_event_weights),
        1.0,
        len(particle_event_weights),
        dtype=np.float64
    )

if (
    not np.all(np.isfinite(particle_event_cdf))
    or np.any(np.diff(particle_event_cdf) < -1.0e-14)
    or not np.isclose(particle_event_cdf[-1], 1.0, rtol=0.0, atol=1.0e-12)
):
    raise RuntimeError("Extinction-event CDF is invalid")
if (
    not np.all(np.isfinite(single_scattering_albedo_by_bin))
    or np.any(single_scattering_albedo_by_bin < 0.0)
    or np.any(single_scattering_albedo_by_bin > 1.0)
):
    raise RuntimeError("Single-scattering albedo table is invalid")

floc_event_probability = (
    np.sum(particle_event_weights[particle_is_floc])
    if np.any(particle_is_floc)
    else 0.0
)

primary_event_probability = (
    np.sum(particle_event_weights[~particle_is_floc])
    if np.any(~particle_is_floc)
    else 0.0
)

print(f"floc_event_probability: {floc_event_probability:.6f}")
print(f"primary_event_probability: {primary_event_probability:.6f}")

print(f"FLOC_ENABLED: {FLOC_ENABLED}")
print("CLARITAS_75: extended primary-size eligibility for floc pooling enabled")
print(
    "FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_um:",
    np.round(FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M * 1e6, 3).tolist()
)
print(
    "FLOC_POOL_EFFECTIVE_DIAMETER_um:",
    np.round(FLOC_POOL_EFFECTIVE_DIAMETER_M * 1e6, 3).tolist()
)
print(f"FLOC_FRACTAL_DIMENSION: {FLOC_FRACTAL_DIMENSION:.3f}")
print(f"FLOC_SCATTER_EFFICIENCY: {FLOC_SCATTER_EFFICIENCY:.3f}")
print(f"FLOC_COLLISION_LENGTH_um: {FLOC_COLLISION_LENGTH_M*1e6:.3f}")
print(f"eligible_primary_number_density_per_m3: {eligible_primary_number_density_per_m3:.3e}")
print(f"eligible_primary_spacing_m: {eligible_primary_spacing_m:.3e}")
print(f"floc_encounter_probability: {floc_encounter_probability:.6f}")
print(f"floc_property_state: {floc_property_state:.6f}")
print(f"floc_mass_fraction: {floc_mass_fraction:.6f}")
if FLOC_ENABLED:
    print(f"FLOC_POOL_KERNEL_LOG_SIGMA: {FLOC_POOL_KERNEL_LOG_SIGMA:.6g}")
    print(f"FLOC_POOL_KERNEL_MIN_PROBABILITY: {FLOC_POOL_KERNEL_MIN_PROBABILITY:.6g}")
    if 'floc_pooling_kernel' in globals():
        print(
            "floc_pooling_kernel_shape:",
            list(floc_pooling_kernel.shape)
        )
        print(
            "floc_pooling_kernel_row_sum_range:",
            f"{np.min(np.sum(floc_pooling_kernel, axis=1)):.6f} - "
            f"{np.max(np.sum(floc_pooling_kernel, axis=1)):.6f}"
        )

if np.any(particle_is_floc):
    print(
        "fractal_floc_mass_by_band_kg:",
        np.round(floc_mass_by_band_kg, 18).tolist()
    )
    print(
        "fractal_floc_effective_density_by_band_kg_per_m3:",
        np.round(floc_effective_density_by_band_kg_per_m3, 3).tolist()
    )

print(f"effective_floc_bins: {np.sum(particle_is_floc)}")
print(
    f"primary_bins_eligible_for_flocs: "
    f"{np.sum(primary_bin_is_pooled_into_floc)} / {len(primary_bin_is_pooled_into_floc)}"
)
print(f"effective_total_bins: {len(particle_diameter_m)}")
print(
    f"pooled_floc_mass_fraction_effective_psd: "
    f"{np.sum(particle_weights[particle_is_floc]) if np.any(particle_is_floc) else 0.0:.6f}"
)
print(
    f"residual_or_unchanged_primary_mass_fraction_effective_psd: "
    f"{np.sum(particle_weights[~particle_is_floc]) if np.any(~particle_is_floc) else 0.0:.6f}"
)
print(
    f"effective_floc_multiplier_range: "
    f"{np.min(floc_diameter_multiplier_by_bin[particle_is_floc]) if np.any(particle_is_floc) else 1.0:.3f} - "
    f"{np.max(floc_diameter_multiplier_by_bin[particle_is_floc]) if np.any(particle_is_floc) else 1.0:.3f}"
)
print(
    f"effective_floc_density_range_kg_per_m3: "
    f"{np.min(particle_density_by_bin_kg_per_m3[particle_is_floc]) if np.any(particle_is_floc) else 0.0:.3e} - "
    f"{np.max(particle_density_by_bin_kg_per_m3[particle_is_floc]) if np.any(particle_is_floc) else 0.0:.3e}"
)
print(
    f"primary_diameter_range_um: "
    f"{np.min(primary_particle_diameter_m)*1e6:.3f} - "
    f"{np.max(primary_particle_diameter_m)*1e6:.3f}"
)
print(
    f"effective_diameter_range_um: "
    f"{np.min(particle_diameter_m)*1e6:.3f} - "
    f"{np.max(particle_diameter_m)*1e6:.3f}"
)
print(
    f"effective_floc_diameter_range_um: "
    f"{np.min(particle_diameter_m[particle_is_floc])*1e6 if np.any(particle_is_floc) else 0.0:.3f} - "
    f"{np.max(particle_diameter_m[particle_is_floc])*1e6 if np.any(particle_is_floc) else 0.0:.3f}"
)
print(f"nonfloc_density_kg_per_m3: {particle_density_kg_per_m3:.3e}")
print(f"weighted_effective_density_kg_per_m3: {effective_particle_density_kg_per_m3:.3e}")
print(
    f"primary_refractive_index: {n_particle:.4f}; "
    f"floc_refractive_index: not used"
)
print("Primary reflection fudge: disabled/removed from CLARITAS_43 transport")
print(f"n_particles_per_m3: {n_particles_per_m3:.3e}")
print(f"particle_mass: {particle_mass:.3e}")
print(f"average_particle_mass: {average_particle_mass:.3e}")
print(f"average_particle_separation_m: {average_particle_separation_m:.3e}")
print(f"mu_s: {mu_s:.3e}")
print(f"mu_a: {mu_a:.3e}")
print(f"mu_t: {mu_t:.3e}")
print(f"medium_single_scattering_albedo: {medium_single_scattering_albedo:.6f}")
print(f"g_eff: {g_eff:.6f}")
print(f"mu_s_prime: {mu_s_prime:.3e}")
print(f"MEAN_FREE_PATH_M: {MEAN_FREE_PATH_M:.3e}")
print(f"PSD_WEIGHT_MODE: {PSD_WEIGHT_MODE}")
print(f"particle_event_weights_sum: {np.sum(particle_event_weights):.6f}")
print(
    "extinction_event_weighting: "
    "P_i = number_density_i * sigma_t_i / sum(number_density * sigma_t)"
)
print(
    f"extinction_weight_check_sum_mu_t_by_bin: "
    f"{np.sum(extinction_event_strength_by_bin):.6e}"
)
print(f"dominant_event_diameter_um: {particle_diameter_m[np.argmax(particle_event_weights)]*1e6:.3f}")
print(f"MEAN_SCATTERING_PATH_M: {MEAN_SCATTERING_PATH_M:.3e}")
print(f"MEAN_ABSORPTION_PATH_M: {MEAN_ABSORPTION_PATH_M:.3e}")
print(f"TRANSPORT_MEAN_FREE_PATH_M: {TRANSPORT_MEAN_FREE_PATH_M:.3e}")
print("Transport mode: event-driven free-path sampling from mu_t")
print("Transport geometry: three-dimensional sphere")
print(f"SIMULATION_SEED: {SIMULATION_SEED}")
print(f"SOURCE_MODE: {SOURCE_MODE}")
print(f"Primary roughness std: {PRIMARY_ROUGHNESS_STD_DEG:.3f} deg")
print(f"Floc roughness std: {FLOC_ROUGHNESS_STD_DEG:.3f} deg")
print(f"PRIMARY_REFRACTIVE_INDEX_IMAG_K: {PRIMARY_REFRACTIVE_INDEX_IMAG_K:.6g}")
print(f"FLOC_ABSORPTION_K_MULTIPLIER: {FLOC_ABSORPTION_K_MULTIPLIER:.6g}")
print(f"FLOC_ABSORPTION_PATH_FACTOR: {FLOC_ABSORPTION_PATH_FACTOR:.6g}")
print(f"single_scattering_albedo_range: {np.min(single_scattering_albedo_by_bin):.6f} - {np.max(single_scattering_albedo_by_bin):.6f}")
if np.any(particle_is_floc):
    print(
        f"floc_mean_chord_length_range_um: "
        f"{np.min(floc_mean_chord_length_by_bin_m[particle_is_floc]) * 1.0e6:.3f} - "
        f"{np.max(floc_mean_chord_length_by_bin_m[particle_is_floc]) * 1.0e6:.3f}"
    )
    print(
        f"floc_absorption_tau_range: "
        f"{np.min(floc_absorption_tau_by_bin[particle_is_floc]):.6e} - "
        f"{np.max(floc_absorption_tau_by_bin[particle_is_floc]):.6e}"
    )
print(
    "Boundary Fresnel/refraction disabled; index metadata only: "
    f"n_medium={n_medium:.3f}, n_external={n_external:.3f}"
)
if 'FLOC_CUDA_INTERNAL_DOMAIN_SPATIAL_DISPLACEMENT_ENABLED' in globals():
    print(f"FLOC_CUDA_INTERNAL_DOMAIN_SPATIAL_DISPLACEMENT_ENABLED: {FLOC_CUDA_INTERNAL_DOMAIN_SPATIAL_DISPLACEMENT_ENABLED}")

def closest_index(arr, value):
    i = np.searchsorted(arr, value)
    if i == 0:
        return 0
    if i == len(arr):
        return len(arr) - 1
    left = i - 1
    right = i
    return left if abs(arr[left] - value) <= abs(arr[right] - value) else right

theta_rad = np.deg2rad(theta_deg)

# Coarser grid for Mie computation — CDF integration converges quickly
# with trapezoidal rule; we interpolate the final CDF back to full resolution.
MIE_COARSENING_FACTOR = 10  # 180,001 → 18,001 points
_mie_theta_deg_coarse = np.linspace(
    0.0, 180.0,
    max(2, (len(theta_deg) - 1) // MIE_COARSENING_FACTOR + 1),
    dtype=np.float64
)
# Ensure the coarse grid exactly spans 0 to pi and includes both endpoints.
if not np.isclose(_mie_theta_deg_coarse[0], 0.0) or not np.isclose(_mie_theta_deg_coarse[-1], 180.0):
    _mie_theta_deg_coarse = np.linspace(0.0, 180.0, 18001, dtype=np.float64)
mie_theta_rad_coarse = np.deg2rad(_mie_theta_deg_coarse)
_mie_mu_coarse = np.cos(mie_theta_rad_coarse)

# Mie S1_S2 cache: maps (real_n, imag_k, size_param, wavelength_m) → (S1, S2)
# S1, S2 are complex arrays on the COARSE grid; CDFs are interpolated to full grid.
_mie_cache = {}

def _cached_mie_S1_S2(m_rel_complex, size_param, wavelength_m):
    """Compute Mie S1_S2 on the coarse grid, caching results."""
    key = (
        round(float(np.real(m_rel_complex)), 10),
        round(float(np.imag(m_rel_complex)), 10),
        round(float(size_param), 10),
        round(float(wavelength_m), 15),
    )
    cached = _mie_cache.get(key)
    if cached is not None:
        return cached
    S1, S2 = miepython.S1_S2(m_rel_complex, size_param, _mie_mu_coarse)
    S1 = np.asarray(S1, dtype=np.complex128)
    S2 = np.asarray(S2, dtype=np.complex128)
    _mie_cache[key] = (S1, S2)
    return S1, S2

def _interpolate_cdf_to_full_grid(cdf_coarse, theta_rad_coarse, theta_rad_full):
    """Linearly interpolate a CDF from coarse to full theta grid.
    
    A CDF is monotonic and defined on [0, pi]; linear interpolation
    between grid points preserves monotonicity and the [0, 1] range.
    The endpoints are pinned exactly: CDF(0)=0, CDF(pi)=1.
    """
    cdf_coarse = np.asarray(cdf_coarse, dtype=np.float64)
    if len(cdf_coarse) == len(theta_rad_full):
        return cdf_coarse
    cdf_full = np.interp(theta_rad_full, theta_rad_coarse, cdf_coarse)
    cdf_full[0] = 0.0
    cdf_full[-1] = 1.0
    # Enforce monotonicity and bounds.
    cdf_full = np.clip(cdf_full, 0.0, 1.0)
    cdf_full = np.maximum.accumulate(cdf_full)
    cdf_full[-1] = 1.0
    return cdf_full


MONOMER_PHASE_COMPONENT_SPLIT_ENABLED = True
MONOMER_DIFFRACTION_WIDTH_FACTOR = 1.0

# it derives from wavelength and aggregate size, not detector response.
AGGREGATE_COHERENCE_LIMIT_ENABLED = True
AGGREGATE_COHERENCE_LENGTH_MODE = "fresnel_patch"  # "fresnel_patch", "radius_gyration", "infinite"
AGGREGATE_COHERENCE_LENGTH_FACTOR = 1.0

# This is not a detector backscatter boost.  It is an optical-depth-derived
FLOC_INTERNAL_TRANSPORT_ENABLED = True
FLOC_INTERNAL_TRANSPORT_PATH_FACTOR = 1.0
FLOC_INTERNAL_TRANSPORT_USE_REDUCED_OD = True
FLOC_INTERNAL_TRANSPORT_MIN_TAU = 0.0
FLOC_INTERNAL_TRANSPORT_MAX_TAU = 50.0

FLOC_INTERNAL_DOMAIN_MC_ENABLED = True
FLOC_INTERNAL_DOMAIN_MC_RAYS = 12000
FLOC_INTERNAL_DOMAIN_MAX_SCATTERS = 64
FLOC_INTERNAL_DOMAIN_COARSE_STEP_DEG = 0.25
FLOC_INTERNAL_DOMAIN_MIN_PROFILE_FLOOR = 1.0e-18

FLOC_CUDA_TRUE_INTERNAL_DOMAIN_ENABLED = True

FLOC_CUDA_INTERNAL_DOMAIN_SPATIAL_DISPLACEMENT_ENABLED = True

# CLARITAS_74: when true CUDA floc-domain transport is enabled, the floc
# no longer uses a synthetic outer floc phase function. Internal scatters
# use the representative primary-particle Mie CDF and the exit direction is
# the direction produced by the internal random walk.
FLOC_DISABLE_SYNTHETIC_OUTER_PHASE_WHEN_TRUE_DOMAIN = True

floc_internal_transport_diagnostics = {}

PHASE_FUNCTION_MEASURE = "solid_angle_I_sin_theta"
mie_cache_version = "claritas_77_3d_solid_angle_cache_v2_exact_0_180"

mie_cache_key = (
    mie_cache_version +
    str(wavelengths) +
    str(n_particle) +
    str(n_medium) +
    str(float(theta_deg[1] - theta_deg[0])) +
    str(primary_particle_diameter_m.tolist()) +
    str(primary_particle_weights.tolist()) +
    str(particle_diameter_m.tolist()) +
    str(particle_weights.tolist()) +
    str(particle_is_floc.astype(int).tolist()) +
    str(FLOC_ENABLED) +
    str(FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M.tolist()) +
    str(FLOC_POOL_EFFECTIVE_DIAMETER_M.tolist()) +
    str(FLOC_FRACTAL_DIMENSION) +
    str(FLOC_COLLISION_LENGTH_M) +
    str(FLOC_SCATTER_EFFICIENCY) +
    str(FLOC_POOL_KERNEL_LOG_SIGMA) +
    str(FLOC_POOL_KERNEL_MIN_PROBABILITY) +
    str(PRIMARY_REFRACTIVE_INDEX_IMAG_K) +
    str(particle_complex_refractive_index_by_bin.real.tolist()) +
    str(particle_refractive_index_imag_k_by_bin.tolist()) +
    str(FLOC_ABSORPTION_K_MULTIPLIER) +
    str(FLOC_ABSORPTION_PATH_FACTOR) +
    str(floc_mass_fraction) +
    str(eligible_primary_spacing_m) +
    str(floc_mass_by_band_kg.tolist() if len(floc_mass_by_band_kg) else []) +
    str(floc_effective_density_by_band_kg_per_m3.tolist() if len(floc_effective_density_by_band_kg_per_m3) else []) +
    str(floc_band_index_by_effective_bin.tolist()) +
    str(effective_bin_kind.tolist()) +
    str(floc_pooling_kernel.tolist() if FLOC_ENABLED and 'floc_pooling_kernel' in globals() else []) +
    str(MONOMER_PHASE_COMPONENT_SPLIT_ENABLED) +
    str(MONOMER_DIFFRACTION_WIDTH_FACTOR) +
    str(AGGREGATE_COHERENCE_LIMIT_ENABLED) +
    str(AGGREGATE_COHERENCE_LENGTH_MODE) +
    str(AGGREGATE_COHERENCE_LENGTH_FACTOR) +
    PHASE_FUNCTION_MEASURE +
    str(FLOC_INTERNAL_TRANSPORT_ENABLED) +
    str(FLOC_INTERNAL_TRANSPORT_PATH_FACTOR) +
    str(FLOC_INTERNAL_TRANSPORT_USE_REDUCED_OD) +
    str(FLOC_INTERNAL_TRANSPORT_MIN_TAU) +
    str(FLOC_INTERNAL_TRANSPORT_MAX_TAU) +
    str(FLOC_INTERNAL_DOMAIN_MC_ENABLED) +
    str(FLOC_INTERNAL_DOMAIN_MC_RAYS) +
    str(FLOC_INTERNAL_DOMAIN_MAX_SCATTERS) +
    str(FLOC_INTERNAL_DOMAIN_COARSE_STEP_DEG) +
    str(FLOC_INTERNAL_DOMAIN_MIN_PROFILE_FLOOR) +
    str(FLOC_CUDA_TRUE_INTERNAL_DOMAIN_ENABLED) +
    str(FLOC_CUDA_INTERNAL_DOMAIN_SPATIAL_DISPLACEMENT_ENABLED) +
    str(FLOC_DISABLE_SYNTHETIC_OUTER_PHASE_WHEN_TRUE_DOMAIN) +
    str(FLOC_POOL_KERNEL_ENFORCE_FLOC_NOT_SMALLER_THAN_SOURCE)
)

mie_cache_hash = hashlib.md5(mie_cache_key.encode()).hexdigest()
mie_cache_file = os.path.join(OUTDIR, f"angular_cache_{mie_cache_hash}.npz")



def _normalise_profile_for_cdf(I):
    I = np.asarray(I, dtype=np.float64)
    I = np.real(I)
    if I.ndim != 1 or I.size < 2:
        raise ValueError("Phase-function intensity must be a non-empty 1-D grid")
    if not np.all(np.isfinite(I)):
        raise ValueError("Phase-function intensity contains non-finite values")
    if np.any(I < 0.0):
        raise ValueError("Phase-function intensity contains negative values")
    if np.sum(I, dtype=np.float64) <= 0.0:
        raise ValueError("Phase-function intensity has zero integral")
    return I


def build_angular_pdf_and_cdf(I, theta_rad_values):
    I = _normalise_profile_for_cdf(I)
    theta_rad_values = np.asarray(theta_rad_values, dtype=np.float64)
    if theta_rad_values.shape != I.shape:
        raise ValueError("Phase intensity and theta grids must have equal shape")
    if (
        not np.all(np.isfinite(theta_rad_values))
        or np.any(np.diff(theta_rad_values) <= 0.0)
        or not np.isclose(
            theta_rad_values[0], 0.0, rtol=0.0, atol=1.0e-14
        )
        or not np.isclose(
            theta_rad_values[-1], np.pi, rtol=0.0, atol=1.0e-14
        )
    ):
        raise ValueError(
            "Theta grid must be finite, strictly increasing, and span "
            "exactly 0 to pi radians"
        )

    # Mie intensity is per unit solid angle.  For an axisymmetric phase
    # function, the physical polar density is I(theta) sin(theta).
    angular_measure = np.sin(theta_rad_values).astype(np.float64)
    angular_measure[~np.isfinite(angular_measure)] = 0.0
    angular_measure = np.maximum(angular_measure, 0.0)
    density_theta = I * angular_measure
    density_theta[~np.isfinite(density_theta)] = 0.0
    density_theta = np.maximum(density_theta, 0.0)
    if np.sum(density_theta) <= 0.0:
        raise ValueError("Phase function has zero physical solid-angle integral")

    increments = (
        0.5
        * (density_theta[1:] + density_theta[:-1])
        * np.diff(theta_rad_values)
    )
    angle_cdf = np.concatenate((
        np.array([0.0], dtype=np.float64),
        np.cumsum(increments, dtype=np.float64)
    ))
    if angle_cdf[-1] <= 0.0 or not np.isfinite(angle_cdf[-1]):
        raise ValueError("Phase-function CDF integral is invalid")
    angle_cdf /= angle_cdf[-1]
    angle_cdf[-1] = 1.0

    # Probability mass associated with each tabulated interval, retained for
    # detector-proxy diagnostics.  The transport samples the CDF itself.
    pdf_theta = np.diff(angle_cdf, prepend=0.0)
    return pdf_theta, angle_cdf


def rotate_direction_3d_numpy(direction, theta_rad_value, phi_rad_value):
    """Reference 3-D direction rotation used by CPU floc diagnostics."""
    w = np.asarray(direction, dtype=np.float64)
    w = w / np.linalg.norm(w)
    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(w[2]) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )
    u = np.cross(reference, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    rotated = (
        np.cos(theta_rad_value) * w
        + np.sin(theta_rad_value)
        * (np.cos(phi_rad_value) * u + np.sin(phi_rad_value) * v)
    )
    return rotated / np.linalg.norm(rotated)


# This is not a detector-fitted backscatter boost.  It is a process-based
def split_monomer_phase_into_diffraction_and_internal(
    primary_form_factor,
    rep_diameter_m,
    wavelength_m,
    theta_rad_values
):
    P = _normalise_profile_for_cdf(primary_form_factor)

    d = max(float(rep_diameter_m), 1.0e-12)
    wl = max(float(wavelength_m), 1.0e-12)

    x = (
        np.pi * n_medium * d * np.sin(theta_rad_values) /
        (wl * max(float(MONOMER_DIFFRACTION_WIDTH_FACTOR), 1.0e-12))
    )

    diffraction_envelope = np.sinc(x / np.pi) ** 2
    diffraction_envelope = np.asarray(diffraction_envelope, dtype=np.float64)
    diffraction_envelope[~np.isfinite(diffraction_envelope)] = 0.0
    diffraction_envelope = np.maximum(diffraction_envelope, 0.0)

    primary_diffraction = P[0] * diffraction_envelope

    primary_diffraction = np.minimum(primary_diffraction, P)
    primary_internal = np.maximum(P - primary_diffraction, 0.0)

    if np.sum(primary_internal) <= 0.0:
        primary_internal = np.zeros_like(P)

    return primary_diffraction, primary_internal


def _floc_internal_transport_component(
    bin_idx,
    primary_internal_component,
    rep_diameter_m,
    wavelength_m,
    theta_rad_values
):
    P_internal = _normalise_profile_for_cdf(primary_internal_component)

    internal_sum = float(np.sum(P_internal))
    if internal_sum <= 0.0:
        return np.zeros_like(P_internal), 0.0, 0.0, 0.0

    if (not FLOC_INTERNAL_TRANSPORT_ENABLED) or bin_idx is None:
        return P_internal.copy(), 0.0, 0.0, 0.0

    n_synth, n_physical, rep_idx = _representative_monomer_count_for_floc_bin(bin_idx)

    floc_diameter_m = max(float(particle_diameter_m[bin_idx]), 1.0e-30)
    floc_volume_m3 = (np.pi / 6.0) * floc_diameter_m**3
    monomer_number_density_inside_floc = (
        float(n_physical) / floc_volume_m3 if floc_volume_m3 > 0.0 else 0.0
    )

    rep_radius_m = max(float(rep_diameter_m) / 2.0, 1.0e-30)
    rep_m_rel = solid_primary_complex_index / n_medium
    rep_x = 2.0 * np.pi * n_medium * rep_radius_m / max(float(wavelength_m), 1.0e-30)

    qext, qsca, qback, g_rep = miepython.efficiencies_mx(rep_m_rel, rep_x)
    qsca = max(float(np.real(qsca)), 0.0)
    g_rep = float(np.real(g_rep))
    g_rep = float(np.clip(g_rep, -1.0, 1.0))

    sigma_s_monomer_m2 = qsca * np.pi * rep_radius_m**2
    mu_s_inside_floc_per_m = monomer_number_density_inside_floc * sigma_s_monomer_m2

    mean_chord_m = (2.0 / 3.0) * floc_diameter_m
    internal_path_m = float(FLOC_INTERNAL_TRANSPORT_PATH_FACTOR) * mean_chord_m

    tau_s = max(mu_s_inside_floc_per_m * internal_path_m, 0.0)

    if FLOC_INTERNAL_TRANSPORT_USE_REDUCED_OD:
        tau_transport = tau_s * max(1.0 - g_rep, 0.0)
    else:
        tau_transport = tau_s

    tau_transport = float(np.clip(
        tau_transport,
        float(FLOC_INTERNAL_TRANSPORT_MIN_TAU),
        float(FLOC_INTERNAL_TRANSPORT_MAX_TAU)
    ))

    transport_fraction = 1.0 - np.exp(-tau_transport)
    transport_fraction = float(np.clip(transport_fraction, 0.0, 1.0))

    # to the current detector/transport geometry.  Preserve the raw angular
    internally_transport_broadened = np.ones_like(P_internal, dtype=np.float64)
    internally_transport_broadened *= internal_sum / max(len(P_internal), 1)

    floc_internal_transport_diagnostics[int(bin_idx)] = {
        "synthetic_monomer_count": float(n_synth),
        "physical_monomer_count": float(n_physical),
        "representative_primary_diameter_um": float(rep_diameter_m * 1.0e6),
        "monomer_number_density_inside_floc_per_m3": float(monomer_number_density_inside_floc),
        "sigma_s_monomer_m2": float(sigma_s_monomer_m2),
        "g_rep": float(g_rep),
        "mu_s_inside_floc_per_m": float(mu_s_inside_floc_per_m),
        "internal_path_um": float(internal_path_m * 1.0e6),
        "tau_s_internal": float(tau_s),
        "tau_transport_internal": float(tau_transport),
        "transport_fraction": float(transport_fraction),
    }

    return internally_transport_broadened, tau_s, tau_transport, transport_fraction


def build_floc_phase_from_split_monomer_and_structure(
    primary_form_factor,
    structure_factor,
    rep_diameter_m,
    wavelength_m,
    theta_rad_values,
    bin_idx=None
):
    P = _normalise_profile_for_cdf(primary_form_factor)
    S = _normalise_profile_for_cdf(structure_factor)

    if not MONOMER_PHASE_COMPONENT_SPLIT_ENABLED:
        return _normalise_profile_for_cdf(P * S), P, np.zeros_like(P)

    P_diff, P_internal = split_monomer_phase_into_diffraction_and_internal(
        P,
        rep_diameter_m,
        wavelength_m,
        theta_rad_values
    )

    coherent_internal = P_internal * S

    internal_transport_profile, tau_s, tau_transport, transport_fraction = (
        _floc_internal_transport_component(
            bin_idx,
            P_internal,
            rep_diameter_m,
            wavelength_m,
            theta_rad_values
        )
    )

    I = (
        P_diff +
        (1.0 - transport_fraction) * coherent_internal +
        transport_fraction * internal_transport_profile
    )

    I = _normalise_profile_for_cdf(I)
    return I, P_diff, P_internal

def _representative_primary_index_for_effective_bin(bin_idx):
    source_min = source_primary_min_diameter_m[bin_idx]
    source_max = source_primary_max_diameter_m[bin_idx]

    if source_min > 0.0 and source_max > 0.0:
        source_mid = np.sqrt(source_min * source_max)
    else:
        source_mid = np.nan

    if not np.isfinite(source_mid) or source_mid <= 0.0:
        source_mid = np.nanmedian(primary_particle_diameter_m)

    return int(np.argmin(np.abs(primary_particle_diameter_m - source_mid)))


def _deterministic_seed_from_bin(bin_idx, wavelength_m):
    key = (
        "synthetic_floc_structure_" +
        str(int(bin_idx)) + "_" +
        str(float(wavelength_m)) + "_" +
        str(float(particle_diameter_m[bin_idx])) + "_" +
        str(float(source_primary_min_diameter_m[bin_idx])) + "_" +
        str(float(source_primary_max_diameter_m[bin_idx])) + "_" +
        str(float(FLOC_FRACTAL_DIMENSION))
    )
    return int(hashlib.md5(key.encode("ascii")).hexdigest()[:8], 16)


def _representative_monomer_count_for_floc_bin(bin_idx):
    rep_idx = _representative_primary_index_for_effective_bin(bin_idx)
    d_rep = primary_particle_diameter_m[rep_idx]
    m_rep = particle_density_kg_per_m3 * (np.pi / 6.0) * d_rep**3

    physical_count = (
        particle_mass_by_bin_kg[bin_idx] / m_rep
        if m_rep > 0.0 else 1.0
    )
    physical_count = max(float(physical_count), 1.0)

    synthetic_count = int(np.clip(round(physical_count), 8, 384))

    return synthetic_count, physical_count, rep_idx


def _generate_synthetic_fractal_points(bin_idx, n_points, wavelength_m):
    rng = np.random.default_rng(_deterministic_seed_from_bin(bin_idx, wavelength_m))

    df = float(np.clip(FLOC_FRACTAL_DIMENSION, 1.1, 3.0))
    r_outer = max(float(particle_diameter_m[bin_idx]) / 2.0, 1.0e-12)

    u = rng.random(n_points)
    radii = r_outer * np.power(u, 1.0 / df)

    cos_t = 2.0 * rng.random(n_points) - 1.0
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t*cos_t))
    phi = 2.0 * np.pi * rng.random(n_points)

    points = np.column_stack((
        radii * sin_t * np.cos(phi),
        radii * sin_t * np.sin(phi),
        radii * cos_t
    ))

    points -= np.mean(points, axis=0)

    return points


def _aggregate_coherence_length_from_rg(rg_m, wavelength_m):
    rg = max(float(rg_m), 1.0e-12)
    wl = max(float(wavelength_m), 1.0e-12)

    if not AGGREGATE_COHERENCE_LIMIT_ENABLED:
        return np.inf

    mode = str(AGGREGATE_COHERENCE_LENGTH_MODE).lower()

    if mode == "infinite":
        return np.inf

    if mode == "radius_gyration":
        base_length = rg

    elif mode == "fresnel_patch":
        base_length = np.sqrt(wl * rg / max(float(n_medium), 1.0e-12))

    else:
        raise ValueError(
            "AGGREGATE_COHERENCE_LENGTH_MODE must be 'fresnel_patch', "
            "'radius_gyration', or 'infinite'"
        )

    return max(float(AGGREGATE_COHERENCE_LENGTH_FACTOR) * float(base_length), 1.0e-12)


def _pair_distance_structure_factor(theta_rad_values, wavelength_m, points):
    points = np.asarray(points, dtype=np.float64)
    n = int(points.shape[0])

    if n < 2:
        return np.ones_like(theta_rad_values, dtype=np.float64), 1.0e-12

    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    rij = dist[iu]
    rij = rij[np.isfinite(rij) & (rij > 0.0)].astype(np.float64)

    if rij.size == 0:
        return np.ones_like(theta_rad_values, dtype=np.float64), 1.0e-12

    centred = points - np.mean(points, axis=0)
    rg = np.sqrt(np.mean(np.sum(centred * centred, axis=1)))
    rg = max(float(rg), 1.0e-12)

    pair_histogram_bins = int(min(4096, max(256, np.sqrt(rij.size) * 4)))

    hist_counts, hist_edges = np.histogram(
        rij,
        bins=pair_histogram_bins,
        range=(0.0, float(np.max(rij)))
    )

    r_centres = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    valid = (hist_counts > 0) & (r_centres > 0.0)
    r_centres = r_centres[valid].astype(np.float64)
    hist_counts = hist_counts[valid].astype(np.float64)

    if r_centres.size == 0:
        return np.ones_like(theta_rad_values, dtype=np.float64), rg

    coherence_length_m = _aggregate_coherence_length_from_rg(rg, wavelength_m)

    if np.isfinite(coherence_length_m):
        coherence_weights = np.exp(-((r_centres / coherence_length_m) ** 2))
    else:
        coherence_weights = np.ones_like(r_centres, dtype=np.float64)

    q = (
        (4.0 * np.pi * n_medium / wavelength_m) *
        np.sin(theta_rad_values / 2.0)
    ).astype(np.float64)

    S = np.empty_like(q, dtype=np.float64)

    q_chunk = 2048
    for q_start in range(0, len(q), q_chunk):
        q_end = min(len(q), q_start + q_chunk)
        qr = q[q_start:q_end, None] * r_centres[None, :]

        sinc = np.ones_like(qr, dtype=np.float64)
        nz = np.abs(qr) > 1.0e-12
        sinc[nz] = np.sin(qr[nz]) / qr[nz]

        pair_sum = np.sum(
            sinc * hist_counts[None, :] * coherence_weights[None, :],
            axis=1
        )
        S[q_start:q_end] = 1.0 + (2.0 / float(n)) * pair_sum

    S = np.maximum(S, 0.0)

    if np.max(S) <= 0.0:
        S = np.ones_like(theta_rad_values, dtype=np.float64)

    return S, rg


def _floc_internal_domain_transport_profile(
    bin_idx,
    primary_internal_component,
    rep_diameter_m,
    wavelength_m,
    theta_rad_values
):
    P_internal = _normalise_profile_for_cdf(primary_internal_component)
    internal_sum = float(np.sum(P_internal))

    if internal_sum <= 0.0:
        return np.zeros_like(theta_rad_values, dtype=np.float64), {
            "domain_enabled": float(False),
            "domain_tau_s": 0.0,
            "domain_mean_internal_scatter_count": 0.0,
            "domain_exit_without_internal_scatter_fraction": 1.0,
            "domain_absorbed_or_truncated_fraction": 0.0,
        }

    if not FLOC_INTERNAL_DOMAIN_MC_ENABLED:
        return P_internal.copy(), {
            "domain_enabled": float(False),
            "domain_tau_s": 0.0,
            "domain_mean_internal_scatter_count": 0.0,
            "domain_exit_without_internal_scatter_fraction": 1.0,
            "domain_absorbed_or_truncated_fraction": 0.0,
        }

    n_synth, n_physical, rep_idx = _representative_monomer_count_for_floc_bin(bin_idx)

    floc_diameter_m = max(float(particle_diameter_m[bin_idx]), 1.0e-30)
    floc_radius_m = 0.5 * floc_diameter_m
    floc_volume_m3 = (np.pi / 6.0) * floc_diameter_m**3

    monomer_number_density_inside_floc = (
        float(n_physical) / floc_volume_m3 if floc_volume_m3 > 0.0 else 0.0
    )

    rep_radius_m = max(float(rep_diameter_m) / 2.0, 1.0e-30)
    rep_m_rel = solid_primary_complex_index / n_medium
    rep_x = 2.0 * np.pi * n_medium * rep_radius_m / max(float(wavelength_m), 1.0e-30)

    qext, qsca, qback, g_rep = miepython.efficiencies_mx(rep_m_rel, rep_x)
    qsca = max(float(np.real(qsca)), 0.0)
    g_rep = float(np.clip(float(np.real(g_rep)), -1.0, 1.0))

    sigma_s_monomer_m2 = qsca * np.pi * rep_radius_m**2
    mu_s_inside_floc_per_m = monomer_number_density_inside_floc * sigma_s_monomer_m2

    mean_chord_m = (2.0 / 3.0) * floc_diameter_m
    tau_s = max(mu_s_inside_floc_per_m * mean_chord_m, 0.0)
    tau_transport = tau_s * max(1.0 - g_rep, 0.0)

    if mu_s_inside_floc_per_m <= 0.0:
        profile = np.zeros_like(theta_rad_values, dtype=np.float64)
        profile[0] = internal_sum
        return profile, {
            "domain_enabled": float(True),
            "domain_tau_s": float(tau_s),
            "domain_tau_transport": float(tau_transport),
            "domain_mean_internal_scatter_count": 0.0,
            "domain_exit_without_internal_scatter_fraction": 1.0,
            "domain_absorbed_or_truncated_fraction": 0.0,
            "domain_mu_s_inside_floc_per_m": float(mu_s_inside_floc_per_m),
        }

    _, internal_angle_cdf = build_angular_pdf_and_cdf(P_internal, theta_rad_values)

    rng = np.random.default_rng(
        _deterministic_seed_from_bin(bin_idx, wavelength_m) ^ 0x63A5C0DE
    )

    n_rays = int(max(1, FLOC_INTERNAL_DOMAIN_MC_RAYS))
    max_scatters = int(max(1, FLOC_INTERNAL_DOMAIN_MAX_SCATTERS))

    coarse_step = max(float(FLOC_INTERNAL_DOMAIN_COARSE_STEP_DEG), 0.01)
    coarse_edges_deg = np.arange(0.0, 180.0 + coarse_step, coarse_step)
    if coarse_edges_deg[-1] < 180.0:
        coarse_edges_deg = np.append(coarse_edges_deg, 180.0)
    coarse_centres_deg = 0.5 * (coarse_edges_deg[:-1] + coarse_edges_deg[1:])
    coarse_hist = np.zeros_like(coarse_centres_deg, dtype=np.float64)

    internal_scatter_counts = []
    exited_without_scatter = 0
    truncated = 0

    eps = max(floc_radius_m * 1.0e-6, 1.0e-15)

    for _ in range(n_rays):
        # Uniform illumination over the projected disk of a spherical floc.
        impact_radius = floc_radius_m * np.sqrt(rng.random())
        impact_azimuth = 2.0 * np.pi * rng.random()
        y = impact_radius * np.cos(impact_azimuth)
        z = impact_radius * np.sin(impact_azimuth)
        x = -np.sqrt(max(
            floc_radius_m*floc_radius_m - y*y - z*z,
            0.0
        )) + eps

        vx = 1.0
        vy = 0.0
        vz = 0.0
        scatter_count = 0
        did_exit = False

        for _step in range(max_scatters):
            u_path = max(float(rng.random()), 1.0e-12)
            free_path = -np.log(u_path) / mu_s_inside_floc_per_m

            b = x * vx + y * vy + z * vz
            c = x*x + y*y + z*z - floc_radius_m*floc_radius_m
            disc = b*b - c

            if disc <= 0.0:
                did_exit = True
                break

            t_exit = -b + np.sqrt(disc)
            if t_exit <= 0.0:
                did_exit = True
                break

            if free_path >= t_exit:
                x += vx * t_exit
                y += vy * t_exit
                z += vz * t_exit
                did_exit = True
                break

            x += vx * free_path
            y += vy * free_path
            z += vz * free_path
            scatter_count += 1

            u_angle = float(rng.random())
            idx_angle = int(np.searchsorted(internal_angle_cdf, u_angle, side="left"))
            if idx_angle < 0:
                idx_angle = 0
            elif idx_angle >= len(theta_rad_values):
                idx_angle = len(theta_rad_values) - 1

            theta = float(theta_rad_values[idx_angle])
            phi = 2.0 * np.pi * rng.random()
            direction = rotate_direction_3d_numpy(
                np.array([vx, vy, vz]), theta, phi
            )
            vx, vy, vz = map(float, direction)

        if not did_exit:
            truncated += 1

        if scatter_count == 0:
            exited_without_scatter += 1

        internal_scatter_counts.append(scatter_count)

        exit_theta = float(np.arccos(np.clip(vx, -1.0, 1.0)))
        exit_theta_deg = np.rad2deg(exit_theta)
        bin_pos = int(np.searchsorted(coarse_edges_deg, exit_theta_deg, side="right") - 1)
        bin_pos = max(0, min(len(coarse_hist) - 1, bin_pos))
        coarse_hist[bin_pos] += 1.0

    if np.sum(coarse_hist) <= 0.0:
        profile = P_internal.copy()
    else:
        floor = float(FLOC_INTERNAL_DOMAIN_MIN_PROFILE_FLOOR)
        coarse_hist = np.maximum(coarse_hist, floor)
        profile = np.interp(
            np.rad2deg(theta_rad_values),
            coarse_centres_deg,
            coarse_hist,
            left=coarse_hist[0],
            right=coarse_hist[-1]
        ).astype(np.float64)
        profile[~np.isfinite(profile)] = 0.0
        profile = np.maximum(profile, floor)
        profile *= internal_sum / max(float(np.sum(profile)), 1.0e-300)

    internal_scatter_counts = np.asarray(internal_scatter_counts, dtype=np.float64)
    diagnostics = {
        "domain_enabled": float(True),
        "domain_rays": float(n_rays),
        "domain_max_scatters": float(max_scatters),
        "domain_tau_s": float(tau_s),
        "domain_tau_transport": float(tau_transport),
        "domain_mu_s_inside_floc_per_m": float(mu_s_inside_floc_per_m),
        "domain_mean_internal_scatter_count": float(np.mean(internal_scatter_counts)),
        "domain_median_internal_scatter_count": float(np.median(internal_scatter_counts)),
        "domain_exit_without_internal_scatter_fraction": float(exited_without_scatter / n_rays),
        "domain_absorbed_or_truncated_fraction": float(truncated / n_rays),
        "domain_profile_raw_sum": float(np.sum(profile)),
    }

    return profile, diagnostics

def floc_synthetic_structure_profile(bin_idx, wavelength_m, mu_values, theta_rad_values):
    n_synth, n_physical, rep_idx = _representative_monomer_count_for_floc_bin(bin_idx)

    points = _generate_synthetic_fractal_points(bin_idx, n_synth, wavelength_m)
    structure_factor, rg = _pair_distance_structure_factor(
        theta_rad_values,
        wavelength_m,
        points
    )

    rep_diameter = primary_particle_diameter_m[rep_idx]
    rep_radius = rep_diameter / 2.0
    rep_m_rel = solid_primary_complex_index / n_medium
    rep_x = 2.0 * np.pi * n_medium * rep_radius / wavelength_m

    S1_rep, S2_rep = miepython.S1_S2(rep_m_rel, rep_x, mu_values)
    primary_form_factor = 0.5 * (np.abs(S1_rep)**2 + np.abs(S2_rep)**2)
    primary_form_factor = np.real(primary_form_factor).astype(np.float64)
    primary_form_factor = _normalise_profile_for_cdf(primary_form_factor)

    primary_diffraction, primary_internal = split_monomer_phase_into_diffraction_and_internal(
        primary_form_factor,
        rep_diameter,
        wavelength_m,
        theta_rad_values
    )

    floc_diameter = max(float(particle_diameter_m[bin_idx]), float(wavelength_m))
    x_agg = (
        np.pi * n_medium * floc_diameter * np.sin(theta_rad_values) /
        max(float(wavelength_m), 1.0e-30)
    )

    aggregate_diffraction = np.sinc(x_agg / np.pi) ** 2
    aggregate_diffraction = np.asarray(aggregate_diffraction, dtype=np.float64)
    aggregate_diffraction[~np.isfinite(aggregate_diffraction)] = 0.0
    aggregate_diffraction = np.maximum(aggregate_diffraction, 0.0)

    diff_sum = float(np.sum(primary_diffraction))
    agg_sum = float(np.sum(aggregate_diffraction))
    if agg_sum > 0.0 and diff_sum > 0.0:
        aggregate_diffraction *= diff_sum / agg_sum
    else:
        aggregate_diffraction = primary_diffraction.copy()

    structure_factor = np.asarray(structure_factor, dtype=np.float64)
    structure_factor[~np.isfinite(structure_factor)] = 1.0
    structure_factor = np.maximum(structure_factor, 0.0)

    coherent_internal = primary_internal * structure_factor

    if FLOC_INTERNAL_DOMAIN_MC_ENABLED:
        internal_component, domain_diag = _floc_internal_domain_transport_profile(
            bin_idx,
            primary_internal,
            rep_diameter,
            wavelength_m,
            theta_rad_values
        )
        phase_model = "aggregate_diffraction_plus_miniature_internal_domain_transport"
    else:
        internal_transport_profile, tau_s, tau_transport, transport_fraction = (
            _floc_internal_transport_component(
                bin_idx,
                primary_internal,
                rep_diameter,
                wavelength_m,
                theta_rad_values
            )
        )
        internal_component = (
            (1.0 - transport_fraction) * coherent_internal +
            transport_fraction * internal_transport_profile
        )
        domain_diag = {
            "domain_enabled": float(False),
            "domain_tau_s": float(tau_s),
            "domain_tau_transport": float(tau_transport),
            "domain_transport_fraction": float(transport_fraction),
        }
        phase_model = "aggregate_diffraction_plus_lumped_internal_transport"

    I = aggregate_diffraction + internal_component
    I = _normalise_profile_for_cdf(I)

    diag = floc_internal_transport_diagnostics.get(int(bin_idx), {})
    diag.update(domain_diag)
    diag.update({
        "aggregate_phase_model": phase_model,
        "radius_of_gyration_um": float(rg * 1.0e6),
        "synthetic_monomer_count": float(n_synth),
        "physical_monomer_count": float(n_physical),
        "representative_primary_index": float(rep_idx),
        "representative_primary_diameter_um": float(rep_diameter * 1.0e6),
        "floc_diameter_um": float(floc_diameter * 1.0e6),
        "aggregate_diffraction_raw_sum": float(np.sum(aggregate_diffraction)),
        "primary_internal_raw_sum": float(np.sum(primary_internal)),
        "coherent_internal_raw_sum": float(np.sum(coherent_internal)),
        "domain_internal_raw_sum": float(np.sum(internal_component)),
        "final_phase_raw_sum": float(np.sum(I)),
    })
    floc_internal_transport_diagnostics[int(bin_idx)] = diag

    return I

def trace_phase_array(bin_idx, wavelength_m):
    mu = np.cos(theta_rad)
    n_synth, n_physical, rep_idx = _representative_monomer_count_for_floc_bin(bin_idx)
    pts = _generate_synthetic_fractal_points(bin_idx, n_synth, wavelength_m)
    S, rg = _pair_distance_structure_factor(theta_rad, wavelength_m, pts)
    coherence_length_m = _aggregate_coherence_length_from_rg(rg, wavelength_m)
    rep_radius = primary_particle_diameter_m[rep_idx] / 2.0
    rep_diameter = primary_particle_diameter_m[rep_idx]
    rep_m_rel = solid_primary_complex_index / n_medium
    rep_x = 2.0 * np.pi * n_medium * rep_radius / wavelength_m
    S1, S2 = miepython.S1_S2(rep_m_rel, rep_x, mu)
    P = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2)
    P = _normalise_profile_for_cdf(P)
    I = floc_synthetic_structure_profile(bin_idx, wavelength_m, mu, theta_rad)
    P_diff, P_internal = split_monomer_phase_into_diffraction_and_internal(
        P, rep_diameter, wavelength_m, theta_rad
    )
    pd.DataFrame({
        "theta_deg": theta_deg,
        "primary_form_factor": P,
        "primary_diffraction_component": P_diff,
        "primary_internal_component": P_internal,
        "structure_factor": S,
        "coherence_length_m": np.full_like(theta_deg, coherence_length_m, dtype=np.float64),
        "final_phase": I
    }).to_csv(os.path.join(OUTDIR, "phase_array_trace.csv"), index=False)
    print("TRACE ids:", id(P), id(S), id(I))
    print("TRACE primary diffraction sum fraction:", np.sum(P_diff) / max(np.sum(P), 1.0e-300))
    print("TRACE primary internal sum fraction:", np.sum(P_internal) / max(np.sum(P), 1.0e-300))
    print("TRACE aggregate coherence mode:", AGGREGATE_COHERENCE_LENGTH_MODE)
    print("TRACE aggregate coherence length um:", coherence_length_m * 1.0e6 if np.isfinite(coherence_length_m) else np.inf)
    diag = floc_internal_transport_diagnostics.get(int(bin_idx), {})
    if diag:
        print("TRACE internal transport tau_s:", diag.get("tau_s_internal", diag.get("domain_tau_s", np.nan)))
        print("TRACE internal transport tau_transport:", diag.get("tau_transport_internal", diag.get("domain_tau_transport", np.nan)))
        print("TRACE internal transport fraction:", diag.get("transport_fraction", diag.get("domain_transport_fraction", np.nan)))
        print("TRACE floc domain mean internal scatters:", diag.get("domain_mean_internal_scatter_count", np.nan))
        print("TRACE floc domain unscattered exit fraction:", diag.get("domain_exit_without_internal_scatter_fraction", np.nan))
    print("TRACE first10:", I[:10])
    print("TRACE last10:", I[-10:])


def export_structure_factor_audit(bin_idx, wavelength_m):
    n_synth, n_physical, rep_idx = _representative_monomer_count_for_floc_bin(bin_idx)
    points = _generate_synthetic_fractal_points(bin_idx, n_synth, wavelength_m)
    points = np.asarray(points, dtype=np.float64)
    n = int(points.shape[0])

    if n < 2:
        print("STRUCTURE AUDIT: not enough points for pair-distance diagnostics")
        return

    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    rij = dist[iu]
    rij = rij[np.isfinite(rij) & (rij > 0.0)].astype(np.float64)

    centred = points - np.mean(points, axis=0)
    rg = np.sqrt(np.mean(np.sum(centred * centred, axis=1)))
    rg = max(float(rg), 1.0e-12)
    coherence_length_m = _aggregate_coherence_length_from_rg(rg, wavelength_m)

    pair_histogram_bins = int(min(4096, max(256, np.sqrt(rij.size) * 4)))
    hist_counts, hist_edges = np.histogram(
        rij,
        bins=pair_histogram_bins,
        range=(0.0, float(np.max(rij)))
    )

    r_centres = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    valid = (hist_counts > 0) & (r_centres > 0.0)
    r_centres = r_centres[valid].astype(np.float64)
    hist_counts = hist_counts[valid].astype(np.float64)

    if r_centres.size == 0:
        print("STRUCTURE AUDIT: no valid pair-distance bins")
        return

    if np.isfinite(coherence_length_m):
        coherence_weights = np.exp(-((r_centres / coherence_length_m) ** 2))
    else:
        coherence_weights = np.ones_like(r_centres, dtype=np.float64)

    q = (
        (4.0 * np.pi * n_medium / wavelength_m) *
        np.sin(theta_rad / 2.0)
    ).astype(np.float64)

    S_unweighted = np.empty_like(q, dtype=np.float64)
    S_weighted = np.empty_like(q, dtype=np.float64)

    q_chunk = 2048
    for q_start in range(0, len(q), q_chunk):
        q_end = min(len(q), q_start + q_chunk)
        qr = q[q_start:q_end, None] * r_centres[None, :]

        sinc = np.ones_like(qr, dtype=np.float64)
        nz = np.abs(qr) > 1.0e-12
        sinc[nz] = np.sin(qr[nz]) / qr[nz]

        pair_sum_unweighted = np.sum(
            sinc * hist_counts[None, :],
            axis=1
        )
        pair_sum_weighted = np.sum(
            sinc * hist_counts[None, :] * coherence_weights[None, :],
            axis=1
        )

        S_unweighted[q_start:q_end] = 1.0 + (2.0 / float(n)) * pair_sum_unweighted
        S_weighted[q_start:q_end] = 1.0 + (2.0 / float(n)) * pair_sum_weighted

    S_unweighted_clamped = np.maximum(S_unweighted, 0.0)
    S_weighted_clamped = np.maximum(S_weighted, 0.0)

    structure_df = pd.DataFrame({
        "theta_deg": theta_deg,
        "q_per_m": q,
        "S_unweighted_raw": S_unweighted,
        "S_weighted_raw": S_weighted,
        "S_unweighted_clamped": S_unweighted_clamped,
        "S_weighted_clamped": S_weighted_clamped,
        "weighted_over_unweighted_clamped": (
            S_weighted_clamped / np.maximum(S_unweighted_clamped, 1.0e-300)
        ),
        "coherence_length_um": np.full_like(theta_deg, coherence_length_m * 1.0e6 if np.isfinite(coherence_length_m) else np.inf, dtype=np.float64),
        "radius_of_gyration_um": np.full_like(theta_deg, rg * 1.0e6, dtype=np.float64),
        "synthetic_monomer_count": np.full_like(theta_deg, n_synth, dtype=np.float64),
        "physical_monomer_count_estimate": np.full_like(theta_deg, n_physical, dtype=np.float64),
    })

    structure_path = os.path.join(
        OUTDIR,
        f"structure_factor_theta_audit_bin_{bin_idx}_conc_{mass_concentration_g_per_L}.csv"
    )
    structure_df.to_csv(structure_path, index=False)

    pair_df = pd.DataFrame({
        "r_um": r_centres * 1.0e6,
        "pair_hist_count": hist_counts,
        "coherence_weight": coherence_weights,
        "weighted_pair_hist_count": hist_counts * coherence_weights,
        "cumulative_pair_fraction": np.cumsum(hist_counts) / np.sum(hist_counts),
        "cumulative_weighted_pair_fraction": np.cumsum(hist_counts * coherence_weights) / max(np.sum(hist_counts * coherence_weights), 1.0e-300),
    })

    pair_path = os.path.join(
        OUTDIR,
        f"structure_pair_distance_audit_bin_{bin_idx}_conc_{mass_concentration_g_per_L}.csv"
    )
    pair_df.to_csv(pair_path, index=False)

    selected_angles = [0.0, 0.1, 1.0, 2.0, 5.0, 10.0, 30.0, 90.0, 170.0]
    contrib_rows = []

    for angle in selected_angles:
        tidx = closest_index(theta_deg, angle)
        qr = q[tidx] * r_centres
        sinc = np.ones_like(qr, dtype=np.float64)
        nz = np.abs(qr) > 1.0e-12
        sinc[nz] = np.sin(qr[nz]) / qr[nz]

        contribution_unweighted = (2.0 / float(n)) * hist_counts * sinc
        contribution_weighted = (2.0 / float(n)) * hist_counts * coherence_weights * sinc

        for r_um, hc, cw, sinc_val, cu, cwc in zip(
            r_centres * 1.0e6,
            hist_counts,
            coherence_weights,
            sinc,
            contribution_unweighted,
            contribution_weighted
        ):
            contrib_rows.append({
                "angle_deg": theta_deg[tidx],
                "r_um": r_um,
                "pair_hist_count": hc,
                "coherence_weight": cw,
                "sinc_qr": sinc_val,
                "contribution_to_Sminus1_unweighted": cu,
                "contribution_to_Sminus1_weighted": cwc,
            })

    contrib_df = pd.DataFrame(contrib_rows)
    contrib_path = os.path.join(
        OUTDIR,
        f"structure_pair_contribution_audit_bin_{bin_idx}_conc_{mass_concentration_g_per_L}.csv"
    )
    contrib_df.to_csv(contrib_path, index=False)

    print("=========== CLARITAS_52 STRUCTURE-FACTOR AUDIT ===========")
    print(f"Audit floc bin index: {bin_idx}")
    print(f"Audit floc effective diameter: {particle_diameter_m[bin_idx] * 1.0e6:.3f} um")
    print(f"Representative primary diameter: {primary_particle_diameter_m[rep_idx] * 1.0e6:.3f} um")
    print(f"Synthetic monomer count: {n_synth}; physical count estimate: {n_physical:.3f}")
    print(f"Pair count: {rij.size}")
    print(f"Pair histogram bins used: {len(r_centres)}")
    print(f"Radius of gyration: {rg * 1.0e6:.3f} um")
    print(f"Coherence mode: {AGGREGATE_COHERENCE_LENGTH_MODE}")
    print(f"Coherence length: {coherence_length_m * 1.0e6 if np.isfinite(coherence_length_m) else np.inf:.6f} um")
    print(f"Raw pair-count sum: {np.sum(hist_counts):.6e}")
    print(f"Weighted pair-count sum: {np.sum(hist_counts * coherence_weights):.6e}")
    print(f"Weighted/raw pair-count ratio: {np.sum(hist_counts * coherence_weights) / max(np.sum(hist_counts), 1.0e-300):.6e}")
    for angle in [0.0, 1.0, 10.0, 90.0, 170.0]:
        tidx = closest_index(theta_deg, angle)
        print(
            f"S audit theta {theta_deg[tidx]:.3f} deg: "
            f"S_unweighted={S_unweighted_clamped[tidx]:.6e}, "
            f"S_weighted={S_weighted_clamped[tidx]:.6e}, "
            f"ratio={S_weighted_clamped[tidx] / max(S_unweighted_clamped[tidx], 1.0e-300):.6e}"
        )
    print(f"Saved {structure_path}")
    print(f"Saved {pair_path}")
    print(f"Saved {contrib_path}")
    print("==========================================================")


if os.path.exists(mie_cache_file):
    print(f"Loading cached angular tables: {mie_cache_file}")

    cache = np.load(mie_cache_file, allow_pickle=True)

    all_profiles = [
        np.asarray(profile, dtype=np.float64)
        for profile in cache["all_profiles"]
    ]

    cdf_profiles = [
        np.asarray(profile, dtype=np.float64)
        for profile in cache["cdf_profiles"]
    ]

else:
    print("Building angular tables: primary Mie + split-monomer floc aggregate diagnostics...")

    all_profiles = []
    cdf_profiles = []

    for wl in wavelengths:
        mu = np.cos(theta_rad)

        psd_weighted_profile = np.zeros_like(theta_rad, dtype=np.float64)
        per_particle_cdfs = []

        for bin_idx, (radius, weight, refr_index, is_floc) in enumerate(zip(
            particle_radius_m,
            particle_weights,
            particle_refractive_index_by_bin,
            particle_is_floc
        )):
            if is_floc and FLOC_CUDA_TRUE_INTERNAL_DOMAIN_ENABLED and FLOC_DISABLE_SYNTHETIC_OUTER_PHASE_WHEN_TRUE_DOMAIN:
                # True floc-domain transport does not use an outer floc phase CDF.
                # Keep this table as a representative monomer-Mie fallback/diagnostic only.
                _, _, rep_idx = _representative_monomer_count_for_floc_bin(bin_idx)
                rep_radius = primary_particle_diameter_m[rep_idx] / 2.0
                rep_m_rel = solid_primary_complex_index / n_medium
                rep_x = 2.0 * np.pi * n_medium * rep_radius / wl

                S1, S2 = _cached_mie_S1_S2(rep_m_rel, rep_x, wl)
                I_coarse = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2)
                I_coarse = np.real(I_coarse).astype(np.float64)

                I = np.interp(theta_rad, mie_theta_rad_coarse, I_coarse)

            elif is_floc:
                I = floc_synthetic_structure_profile(
                    bin_idx,
                    wl,
                    mu,
                    theta_rad
                )

            else:
                m_rel = particle_complex_refractive_index_by_bin[bin_idx] / n_medium
                x = 2.0 * np.pi * n_medium * radius / wl

                S1, S2 = _cached_mie_S1_S2(m_rel, x, wl)
                I_coarse = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2)
                I_coarse = np.real(I_coarse).astype(np.float64)

                I = np.interp(theta_rad, mie_theta_rad_coarse, I_coarse)

            if is_floc and not (FLOC_CUDA_TRUE_INTERNAL_DOMAIN_ENABLED and FLOC_DISABLE_SYNTHETIC_OUTER_PHASE_WHEN_TRUE_DOMAIN):
                # floc_synthetic_structure_profile already returns I on full grid
                I_norm = _normalise_profile_for_cdf(I)
                psd_weighted_profile += weight * I_norm
                _, angle_cdf = build_angular_pdf_and_cdf(I_norm, theta_rad)
                per_particle_cdfs.append(angle_cdf)
            else:
                # Build CDF on coarse grid, then interpolate to full resolution
                I_coarse_norm = _normalise_profile_for_cdf(I_coarse)
                I_norm_full = np.interp(theta_rad, mie_theta_rad_coarse, I_coarse_norm)
                psd_weighted_profile += weight * I_norm_full
                _, angle_cdf_coarse = build_angular_pdf_and_cdf(I_coarse_norm, mie_theta_rad_coarse)
                angle_cdf = _interpolate_cdf_to_full_grid(angle_cdf_coarse, mie_theta_rad_coarse, theta_rad)
                per_particle_cdfs.append(angle_cdf)

        all_profiles.append(psd_weighted_profile.astype(np.float64))
        cdf_profiles.append(np.asarray(per_particle_cdfs, dtype=np.float64))

    np.savez_compressed(
        mie_cache_file,
        all_profiles=np.asarray(all_profiles, dtype=np.float64),
        cdf_profiles=np.asarray(cdf_profiles, dtype=np.float64)
    )

    print(f"Saved angular cache: {mie_cache_file}")



def build_or_load_cuda_internal_floc_domain_tables():
    internal_cache_key = (
        mie_cache_version +
        "_internal_domain_" +
        str(wavelengths) +
        str(primary_particle_diameter_m.tolist()) +
        str(source_primary_min_diameter_m.tolist()) +
        str(source_primary_max_diameter_m.tolist()) +
        str(particle_diameter_m.tolist()) +
        str(particle_mass_by_bin_kg.tolist()) +
        str(particle_is_floc.astype(np.int8).tolist()) +
        str(FLOC_INTERNAL_TRANSPORT_PATH_FACTOR) +
        str(FLOC_INTERNAL_TRANSPORT_USE_REDUCED_OD) +
        str(FLOC_INTERNAL_TRANSPORT_MIN_TAU) +
        str(FLOC_INTERNAL_TRANSPORT_MAX_TAU) +
        PHASE_FUNCTION_MEASURE
    )
    internal_cache_hash = hashlib.md5(internal_cache_key.encode()).hexdigest()
    internal_cache_file = os.path.join(
        OUTDIR,
        f"internal_floc_cache_{internal_cache_hash}.npz"
    )

    if os.path.exists(internal_cache_file):
        t0 = time.perf_counter()
        cache = np.load(internal_cache_file, allow_pickle=False)

        unique_cdfs_by_wavelength = np.asarray(
            cache["unique_cdfs_by_wavelength"],
            dtype=np.float64
        )
        floc_rep_slot_by_bin = np.asarray(
            cache["floc_rep_slot_by_bin"],
            dtype=np.int32
        )
        floc_internal_tau_s_profiles = np.asarray(
            cache["floc_internal_tau_s_profiles"],
            dtype=np.float64
        )

        internal_cdf_profiles = []
        for wl_idx in range(len(wavelengths)):
            expanded = np.asarray(cdf_profiles[wl_idx], dtype=np.float64).copy()
            floc_bins = np.where(particle_is_floc)[0]
            if floc_bins.size:
                expanded[floc_bins, :] = unique_cdfs_by_wavelength[
                    wl_idx,
                    floc_rep_slot_by_bin[floc_bins],
                    :
                ]
            internal_cdf_profiles.append(expanded)

        print(
            f"✅ Loaded cached CUDA internal-floc tables in "
            f"{time.perf_counter() - t0:.2f} s: {internal_cache_file}"
        )
        return internal_cdf_profiles, [
            np.asarray(row, dtype=np.float64)
            for row in floc_internal_tau_s_profiles
        ]

    t0 = time.perf_counter()

    floc_bins = np.where(particle_is_floc)[0]
    rep_idx_by_bin = np.full(len(particle_is_floc), -1, dtype=np.int32)

    for bin_idx in floc_bins:
        rep_idx_by_bin[bin_idx] = _representative_primary_index_for_effective_bin(
            int(bin_idx)
        )

    unique_rep_indices = np.unique(rep_idx_by_bin[floc_bins])
    rep_slot_lookup = {
        int(rep_idx): slot
        for slot, rep_idx in enumerate(unique_rep_indices)
    }

    floc_rep_slot_by_bin = np.full(len(particle_is_floc), -1, dtype=np.int32)
    for bin_idx in floc_bins:
        floc_rep_slot_by_bin[bin_idx] = rep_slot_lookup[
            int(rep_idx_by_bin[bin_idx])
        ]

    unique_cdfs_by_wavelength = np.empty(
        (len(wavelengths), len(unique_rep_indices), len(theta_rad)),
        dtype=np.float64
    )
    floc_internal_tau_s_profiles = []

    mu = np.cos(theta_rad)

    for wl_idx, wl in enumerate(wavelengths):
        for slot, rep_idx in enumerate(unique_rep_indices):
            rep_diameter = primary_particle_diameter_m[int(rep_idx)]
            rep_radius = rep_diameter / 2.0
            rep_m_rel = solid_primary_complex_index / n_medium
            rep_x = 2.0 * np.pi * n_medium * rep_radius / wl

            S1_rep, S2_rep = _cached_mie_S1_S2(rep_m_rel, rep_x, wl)
            I_rep = 0.5 * (np.abs(S1_rep)**2 + np.abs(S2_rep)**2)
            I_rep = np.real(I_rep).astype(np.float64)
            I_rep = _normalise_profile_for_cdf(I_rep)
            _, internal_cdf_coarse = build_angular_pdf_and_cdf(I_rep, mie_theta_rad_coarse)
            internal_cdf = _interpolate_cdf_to_full_grid(internal_cdf_coarse, mie_theta_rad_coarse, theta_rad)

            unique_cdfs_by_wavelength[wl_idx, slot, :] = internal_cdf

        per_bin_tau_s = np.zeros(len(particle_is_floc), dtype=np.float64)

        for bin_idx in floc_bins:
            rep_idx = int(rep_idx_by_bin[bin_idx])
            rep_diameter = primary_particle_diameter_m[rep_idx]

            _, tau_s, _, _ = _floc_internal_transport_component(
                int(bin_idx),
                np.ones_like(theta_rad, dtype=np.float64),
                rep_diameter,
                wl,
                theta_rad
            )
            per_bin_tau_s[bin_idx] = float(tau_s)

        floc_internal_tau_s_profiles.append(per_bin_tau_s)

    np.savez(
        internal_cache_file,
        unique_cdfs_by_wavelength=unique_cdfs_by_wavelength.astype(np.float32),
        floc_rep_slot_by_bin=floc_rep_slot_by_bin,
        floc_internal_tau_s_profiles=np.asarray(
            floc_internal_tau_s_profiles,
            dtype=np.float32
        )
    )

    internal_cdf_profiles = []
    for wl_idx in range(len(wavelengths)):
        expanded = np.asarray(cdf_profiles[wl_idx], dtype=np.float64).copy()
        if floc_bins.size:
            expanded[floc_bins, :] = unique_cdfs_by_wavelength[
                wl_idx,
                floc_rep_slot_by_bin[floc_bins],
                :
            ]
        internal_cdf_profiles.append(expanded)

    print(
        f"✅ Built and cached CUDA internal-floc tables in "
        f"{time.perf_counter() - t0:.2f} s using "
        f"{len(unique_rep_indices)} unique representative-primary CDFs "
        f"for {len(floc_bins)} floc bins"
    )

    return internal_cdf_profiles, floc_internal_tau_s_profiles


internal_floc_angle_cdf_profiles, floc_internal_tau_s_profiles = (
    build_or_load_cuda_internal_floc_domain_tables()
)

if np.any(particle_is_floc):
    tau0 = np.asarray(
        floc_internal_tau_s_profiles[0],
        dtype=np.float64
    )[particle_is_floc]
    print(
        "CLARITAS_76 floc raw internal tau_s range: "
        f"{np.min(tau0):.6e} - {np.max(tau0):.6e}"
    )


df_angles = pd.DataFrame({"Angle_deg": theta_deg})

for wl_idx, wl in enumerate(wavelengths):
    df_angles[f"I_{int(wl*1e9)}nm"] = all_profiles[wl_idx]

csv_angles_path = os.path.join(OUTDIR, "angular_scattering_profiles.csv")
df_angles.to_csv(csv_angles_path, index=False)
print(f"✅ Saved {csv_angles_path}")

plt.figure(figsize=(8, 5))

for wl_idx, wl in enumerate(wavelengths):
    plt.plot(theta_deg, all_profiles[wl_idx], label=f"{int(wl*1e9)} nm")

plt.xlabel("Scattering angle (deg)")
plt.ylabel("Intensity (a.u.)")
plt.title("Angular scattering profiles - primary Mie + floc internal transport phase")
plt.legend(title="Wavelength")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "angular_scattering_profiles.png"), dpi=200)
plt.close()

print("✅ Saved angular_scattering_profiles.png")

if floc_internal_transport_diagnostics:
    floc_internal_transport_df = pd.DataFrame([
        {"effective_bin_index": int(k), **v}
        for k, v in sorted(floc_internal_transport_diagnostics.items())
    ])
    floc_internal_transport_path = os.path.join(
        OUTDIR,
        f"floc_internal_transport_diagnostics_conc_{mass_concentration_g_per_L}_{int(wavelengths[0]*1e9)}nm.csv"
    )
    floc_internal_transport_df.to_csv(floc_internal_transport_path, index=False)
    print(f"✅ Saved {floc_internal_transport_path}")


def export_phase_function_audit_for_representative_floc(wl_idx=0, target_floc_diameter_um=120.0):
    if not np.any(particle_is_floc):
        print("No floc bins present; skipped CLARITAS_48 phase-function audit.")
        return

    wl = wavelengths[wl_idx]
    mu = np.cos(theta_rad)

    floc_indices = np.where(particle_is_floc)[0]
    floc_diam_um = particle_diameter_m[floc_indices] * 1.0e6
    audit_bin = int(floc_indices[np.argmin(np.abs(floc_diam_um - target_floc_diameter_um))])

    n_synth, n_physical, rep_idx = _representative_monomer_count_for_floc_bin(audit_bin)
    points = _generate_synthetic_fractal_points(audit_bin, n_synth, wl)

    structure_factor, rg = _pair_distance_structure_factor(
        theta_rad,
        wl,
        points
    )

    rep_radius = primary_particle_diameter_m[rep_idx] / 2.0
    rep_m_rel = solid_primary_complex_index / n_medium
    rep_x = 2.0 * np.pi * n_medium * rep_radius / wl

    S1_rep, S2_rep = miepython.S1_S2(rep_m_rel, rep_x, mu)
    primary_form_factor = 0.5 * (np.abs(S1_rep)**2 + np.abs(S2_rep)**2)
    primary_form_factor = np.real(primary_form_factor).astype(np.float64)
    primary_form_factor = _normalise_profile_for_cdf(primary_form_factor)

    if FLOC_CUDA_TRUE_INTERNAL_DOMAIN_ENABLED and FLOC_DISABLE_SYNTHETIC_OUTER_PHASE_WHEN_TRUE_DOMAIN:
        final_intensity = primary_form_factor.copy()
        primary_diffraction_component = np.zeros_like(primary_form_factor)
        primary_internal_component = primary_form_factor.copy()
    else:
        final_intensity, primary_diffraction_component, primary_internal_component = (
            build_floc_phase_from_split_monomer_and_structure(
                primary_form_factor,
                structure_factor,
                primary_particle_diameter_m[rep_idx],
                wl,
                theta_rad,
                bin_idx=audit_bin
            )
        )

    final_pdf, cpu_cdf_recomputed = build_angular_pdf_and_cdf(
        final_intensity,
        theta_rad
    )

    cpu_cdf_table = np.asarray(cdf_profiles[wl_idx][audit_bin], dtype=np.float64)

    gpu_cdf_copyback = cp.asnumpy(cp.asarray(cpu_cdf_table, dtype=cp.float64))

    cdf_recompute_vs_table = cpu_cdf_recomputed - cpu_cdf_table
    cdf_table_vs_gpu = cpu_cdf_table - gpu_cdf_copyback

    cpu_gpu_max_abs = float(np.max(np.abs(cdf_table_vs_gpu)))
    cpu_gpu_rms = float(np.sqrt(np.mean(cdf_table_vs_gpu**2)))
    recompute_table_max_abs = float(np.max(np.abs(cdf_recompute_vs_table)))
    recompute_table_rms = float(np.sqrt(np.mean(cdf_recompute_vs_table**2)))

    def audit_idx(angle_deg):
        return closest_index(theta_deg, angle_deg)

    i0 = audit_idx(0.0)
    i10 = audit_idx(10.0)
    i90 = audit_idx(90.0)
    i120 = audit_idx(120.0)
    i150 = audit_idx(150.0)
    i170 = audit_idx(170.0)

    phase_audit_df = pd.DataFrame({
        "theta_deg": theta_deg,
        "structure_factor_raw": structure_factor,
        "primary_form_factor_raw": primary_form_factor,
        "primary_diffraction_component": primary_diffraction_component,
        "primary_internal_component": primary_internal_component,
        "final_intensity_raw": final_intensity,
        "floc_internal_transport_fraction": np.full_like(theta_deg, floc_internal_transport_diagnostics.get(int(audit_bin), {}).get("transport_fraction", np.nan), dtype=np.float64),
        "floc_internal_tau_s": np.full_like(theta_deg, floc_internal_transport_diagnostics.get(int(audit_bin), {}).get("tau_s_internal", np.nan), dtype=np.float64),
        "floc_internal_tau_transport": np.full_like(theta_deg, floc_internal_transport_diagnostics.get(int(audit_bin), {}).get("tau_transport_internal", np.nan), dtype=np.float64),
        "phase_function_measure": np.full(
            theta_deg.shape,
            PHASE_FUNCTION_MEASURE,
            dtype=object
        ),
        "final_pdf_transport_measure": final_pdf,
        "cpu_cdf_recomputed": cpu_cdf_recomputed,
        "cpu_cdf_table_used_by_transport": cpu_cdf_table,
        "gpu_cdf_copyback": gpu_cdf_copyback,
        "cdf_recompute_minus_table": cdf_recompute_vs_table,
        "cdf_table_minus_gpu_copyback": cdf_table_vs_gpu,
    })

    phase_audit_path = os.path.join(OUTDIR, "phase_function_cpu_vs_gpu.csv")
    phase_audit_df.to_csv(phase_audit_path, index=False)

    print("=========== CLARITAS_74 FLOC DOMAIN / FALLBACK CDF AUDIT ===========")
    print(f"Audit floc bin index: {audit_bin}")
    print(f"Audit floc effective diameter: {particle_diameter_m[audit_bin]*1.0e6:.3f} um")
    print(f"Representative primary diameter: {primary_particle_diameter_m[rep_idx]*1.0e6:.3f} um")
    print(f"Synthetic monomer count: {n_synth}; physical count estimate: {n_physical:.3f}")
    print(f"Radius of gyration: {rg*1.0e6:.3f} um")
    print(f"Synthetic outer floc phase disabled for true domain: {FLOC_DISABLE_SYNTHETIC_OUTER_PHASE_WHEN_TRUE_DOMAIN}")
    print(f"Monomer component split enabled: {MONOMER_PHASE_COMPONENT_SPLIT_ENABLED}")
    print(f"Monomer diffraction width factor: {MONOMER_DIFFRACTION_WIDTH_FACTOR:.6g}")
    print(f"Phase-function measure: {PHASE_FUNCTION_MEASURE}")
    diag = floc_internal_transport_diagnostics.get(int(audit_bin), {})
    print(f"Floc internal transport enabled: {FLOC_INTERNAL_TRANSPORT_ENABLED}")
    if diag:
        print(f"Floc internal tau_s: {diag.get('tau_s_internal', np.nan):.6e}")
        print(f"Floc internal tau_transport: {diag.get('tau_transport_internal', np.nan):.6e}")
        print(f"Floc internal transport fraction: {diag.get('transport_fraction', np.nan):.6e}")
    print(f"Primary diffraction sum fraction: {np.sum(primary_diffraction_component)/max(np.sum(primary_form_factor),1.0e-300):.6e}")
    print(f"Primary internal sum fraction: {np.sum(primary_internal_component)/max(np.sum(primary_form_factor),1.0e-300):.6e}")
    print(f"CPU recomputed CDF vs transport table max abs: {recompute_table_max_abs:.3e}")
    print(f"CPU recomputed CDF vs transport table RMS: {recompute_table_rms:.3e}")
    print(f"CPU transport table vs GPU copyback max abs: {cpu_gpu_max_abs:.3e}")
    print(f"CPU transport table vs GPU copyback RMS: {cpu_gpu_rms:.3e}")
    print(
        "Final phase ratios: "
        f"I10/I0={final_intensity[i10]/max(final_intensity[i0],1e-300):.6e}, "
        f"I90/I0={final_intensity[i90]/max(final_intensity[i0],1e-300):.6e}, "
        f"I120/I0={final_intensity[i120]/max(final_intensity[i0],1e-300):.6e}, "
        f"I150/I0={final_intensity[i150]/max(final_intensity[i0],1e-300):.6e}, "
        f"I170/I0={final_intensity[i170]/max(final_intensity[i0],1e-300):.6e}"
    )
    print(
        "CDF checkpoints: "
        f"CDF10={cpu_cdf_table[i10]:.6e}, "
        f"CDF90={cpu_cdf_table[i90]:.6e}, "
        f"CDF170={cpu_cdf_table[i170]:.6e}, "
        f"CDFend={cpu_cdf_table[-1]:.6e}"
    )
    print(f"Saved {phase_audit_path}")
    print("=========================================================")

    n_audit_samples = 1_000_000
    rng = np.random.default_rng(123456789)
    u = rng.random(n_audit_samples)
    sampled_idx = np.searchsorted(cpu_cdf_table, u, side="left")
    sampled_idx = np.clip(sampled_idx, 0, len(theta_deg) - 1)
    sampled_counts = np.bincount(sampled_idx, minlength=len(theta_deg)).astype(np.float64)
    sampled_pdf = sampled_counts / np.sum(sampled_counts)

    sampling_error = sampled_pdf - final_pdf
    sampling_validation_df = pd.DataFrame({
        "theta_deg": theta_deg,
        "expected_pdf": final_pdf,
        "sampled_pdf": sampled_pdf,
        "sampled_counts": sampled_counts,
        "sampled_minus_expected_pdf": sampling_error,
    })

    sampling_validation_path = os.path.join(OUTDIR, "phase_sampling_validation.csv")
    sampling_validation_df.to_csv(sampling_validation_path, index=False)

    sampling_l1 = float(np.sum(np.abs(sampling_error)))
    sampling_rms = float(np.sqrt(np.mean(sampling_error**2)))
    sampling_max_abs = float(np.max(np.abs(sampling_error)))

    plt.figure(figsize=(9, 5))
    plt.plot(theta_deg, final_pdf, label="Expected PDF from CDF table")
    plt.plot(theta_deg, sampled_pdf, alpha=0.7, label="Sampled inverse-CDF PDF")
    plt.xlabel("Scattering angle (deg)")
    plt.ylabel("Probability per angular grid bin")
    plt.title(
        "CLARITAS_48 phase sampling validation - "
        f"floc {particle_diameter_m[audit_bin]*1.0e6:.1f} um"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    sampling_plot_path = os.path.join(OUTDIR, "phase_sampling_validation.png")
    plt.savefig(sampling_plot_path, dpi=200)
    plt.close()

    print("=========== CLARITAS_48 SAMPLING VALIDATION ===========")
    print(f"Audit samples: {n_audit_samples}")
    print(f"Sampling L1 error: {sampling_l1:.3e}")
    print(f"Sampling RMS error: {sampling_rms:.3e}")
    print(f"Sampling max abs error: {sampling_max_abs:.3e}")
    print(f"Saved {sampling_validation_path}")
    print(f"Saved {sampling_plot_path}")
    print("=======================================================")


export_phase_function_audit_for_representative_floc(
    wl_idx=0,
    target_floc_diameter_um=120.0
)
def sample_beta_angles(N, a1, a2, rng):
    N_half = N // 2
    u_left = rng.beta(a1, a2, N_half)
    angles_left = (1-u_left) * (np.pi/2) - (np.pi/2)
    angles_right = -angles_left
    angles = np.concatenate([angles_left, angles_right])
    if len(angles) < N:
        angles = np.append(angles, 0.0)
    return angles.astype(np.float64)

cuda_src = r"""
extern "C" {

__device__ unsigned int xorshift32_state(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

__device__ double rnd_uniform_double(unsigned int* state) {
    unsigned int r = xorshift32_state(state);
    return ((double)r + 0.5) * 2.3283064365386962890625e-10;
}

__device__ float rnd_uniform(unsigned int* state) {
    float u = (float)rnd_uniform_double(state);
    return fminf(
        0.9999999403953552f,
        fmaxf(1.1641532182693481e-10f, u)
    );
}

__device__ float gaussian_jitter(unsigned int* state, float std_rad) {
    if (std_rad <= 0.0f) return 0.0f;

    float u1 = fmaxf(rnd_uniform(state), 1.0e-12f);
    float u2 = rnd_uniform(state);

    return std_rad *
        sqrtf(-2.0f * logf(u1)) *
        cosf(2.0f * 3.1415927f * u2);
}

__device__ int cdf_binary_search(
    const double* cdf,
    int n,
    double u)
{
    int lo = 0;
    int hi = n - 1;

    while (lo < hi) {
        int mid = (lo + hi) >> 1;

        if (u <= cdf[mid]) {
            hi = mid;
        }
        else {
            lo = mid + 1;
        }
    }

    return lo;
}

__device__ float sample_theta_from_cdf(
    const double* cdf,
    const double* theta,
    int n,
    double u)
{
    int hi = cdf_binary_search(cdf, n, u);
    if (hi <= 0) return (float)theta[0];

    int lo = hi - 1;
    double c0 = cdf[lo];
    double c1 = cdf[hi];
    double fraction = 0.0;
    if (c1 > c0) {
        fraction = (u - c0) / (c1 - c0);
    }
    fraction = fmin(1.0, fmax(0.0, fraction));
    return (float)(theta[lo] + fraction * (theta[hi] - theta[lo]));
}

__device__ void scatter_direction_3d(
    float* vx,
    float* vy,
    float* vz,
    float theta,
    float phi)
{
    float wx = *vx;
    float wy = *vy;
    float wz = *vz;

    // Construct a stable orthonormal basis (u, v, w) around the incident ray.
    float ax = (fabsf(wz) < 0.9f) ? 0.0f : 1.0f;
    float ay = 0.0f;
    float az = (fabsf(wz) < 0.9f) ? 1.0f : 0.0f;

    float ux = ay * wz - az * wy;
    float uy = az * wx - ax * wz;
    float uz = ax * wy - ay * wx;
    float unorm = sqrtf(fmaxf(ux*ux + uy*uy + uz*uz, 1.0e-30f));
    ux /= unorm;
    uy /= unorm;
    uz /= unorm;

    float qx = wy * uz - wz * uy;
    float qy = wz * ux - wx * uz;
    float qz = wx * uy - wy * ux;

    float ct = cosf(theta);
    float st = sinf(theta);
    float cp = cosf(phi);
    float sp = sinf(phi);

    float nx = ct * wx + st * (cp * ux + sp * qx);
    float ny = ct * wy + st * (cp * uy + sp * qy);
    float nz = ct * wz + st * (cp * uz + sp * qz);
    float nnorm = sqrtf(fmaxf(nx*nx + ny*ny + nz*nz, 1.0e-30f));

    *vx = nx / nnorm;
    *vy = ny / nnorm;
    *vz = nz / nnorm;
}

__global__ void trace_kernel(
    const int MAX_EXTINCTIONS,
    const float MU_T,
    const float PRIMARY_ROUGHNESS_STD_RAD,
    const float FLOC_ROUGHNESS_STD_RAD,
    const float R_REAL,
    const float R_OFF,
    const int VIS_SIZE,
    const float VISUAL_SCALE,
    const int REFERENCE_COLLIMATED_SOURCE,
    const float PRODUCTION_BEAM_SIGMA_M,
    const double* angles_init,
    const int N_rays,
    const double* extinction_cdf_table,
    const double* particle_is_floc_table,
    const double* single_scattering_albedo_table,
    const double* angle_cdf_table,
    const double* internal_angle_cdf_table,
    const double* floc_internal_tau_s_table,
    const double* particle_diameter_table,
    const double* theta_table,
    const int n_particles,
    const int n_theta,
    const int FLOC_CUDA_TRUE_INTERNAL_DOMAIN_ENABLED,
    const int FLOC_INTERNAL_DOMAIN_MAX_SCATTERS,
    float* heatmap_flat,
    float* exit_dir_out,
    float* exit_x_out,
    float* exit_y_out,
    float* exit_z_out,
    float* exit_vx_out,
    float* exit_vy_out,
    float* exit_vz_out,
    float* ray_path_length_out,
    int* scatter_count_out,
    int* floc_event_count_out,
    int* floc_extinction_count_out,
    int* last_event_was_floc_out,
    int* last_scatter_bin_out,
    int* extinction_count_out,
    int* absorbed_out,
    int* truncated_out,
    int* terminal_state_out,
    float* floc_domain_dx_out,
    float* floc_domain_dy_out,
    float* floc_domain_dz_out,
    float* floc_domain_path_out,
    int* floc_internal_scatter_count_out,
    unsigned int transport_seed,
    unsigned int azimuth_seed,
    const unsigned long long ray_index_offset)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N_rays) return;
    terminal_state_out[tid] = 0;
    scatter_count_out[tid] = 0;
    floc_event_count_out[tid] = 0;
    floc_extinction_count_out[tid] = 0;
    last_event_was_floc_out[tid] = 0;
    last_scatter_bin_out[tid] = -1;
    extinction_count_out[tid] = 0;
    absorbed_out[tid] = 0;
    truncated_out[tid] = 0;
    floc_domain_dx_out[tid] = 0.0f;
    floc_domain_dy_out[tid] = 0.0f;
    floc_domain_dz_out[tid] = 0.0f;
    floc_domain_path_out[tid] = 0.0f;
    floc_internal_scatter_count_out[tid] = 0;
    ray_path_length_out[tid] = 0.0f;

    // Map each supported global ray ID bijectively onto a nonzero xorshift32
    // state.  Both odd multipliers are coprime to 2^32-1, so no stream seed is
    // repeated while global_tid is in the host-enforced range [0, 2^32-2].
    const unsigned long long rng_state_modulus = 0xffffffffULL;
    const unsigned long long global_tid =
        ray_index_offset + (unsigned long long)tid;
    unsigned int state = (unsigned int)(
        (
            (unsigned long long)transport_seed
            + global_tid * 74729ULL
            + 13ULL
        ) % rng_state_modulus
    ) + 1u;
    unsigned int stateJITTER = (unsigned int)(
        (
            (unsigned long long)azimuth_seed
            + global_tid * 31337ULL
            + 29ULL
        ) % rng_state_modulus
    ) + 1u;

    float x0;
    float y0;
    float z0 = 0.0f;
    float angle_init;
    if (REFERENCE_COLLIMATED_SOURCE) {
        x0 = 0.0f;
        y0 = -R_REAL;
        angle_init = 0.0f;
    }
    else {
        float u1 = rnd_uniform(&state);
        float u2 = rnd_uniform(&state);
        float gaussian =
            sqrtf(-2.0f * logf(u1)) *
            cosf(2.0f * 3.1415927f * u2);
        x0 = PRODUCTION_BEAM_SIGMA_M * gaussian;
        y0 = -(R_REAL + R_OFF);
        angle_init = (float)angles_init[tid];
    }

    float vx = sinf(angle_init);
    float vy = cosf(angle_init);
    float vz = 0.0f;

    float b = x0 * vx + y0 * vy + z0 * vz;
    float c = x0 * x0 + y0 * y0 + z0 * z0 - R_REAL * R_REAL;
    float disc = b * b - c;

    if (disc < 0.0f) {
        terminal_state_out[tid] = 4;
        return;
    }

    float t = -b - sqrtf(disc);
    if (t < 0.0f) {
        terminal_state_out[tid] = 4;
        return;
    }

    float x = x0 + t * vx;
    float y = y0 + t * vy;
    float z = z0 + t * vz;

    int absorbed = 0;
    int truncated = 0;
    int scatter_count = 0;
    int extinction_count = 0;
    int floc_event_count = 0;
    int floc_extinction_count = 0;
    int last_event_was_floc = 0;
    int last_scatter_bin = -1;
    float rpl = 0.0f;
    float floc_domain_dx_sum = 0.0f;
    float floc_domain_dy_sum = 0.0f;
    float floc_domain_dz_sum = 0.0f;
    float floc_domain_path_sum = 0.0f;
    int floc_internal_scatter_count_sum = 0;

    const float HEATMAP_SAMPLE_SPACING = 1.0e-6f;

    while (x * x + y * y + z * z <= R_REAL * R_REAL) {
        float boundary_b = x * vx + y * vy + z * vz;
        float boundary_c = x*x + y*y + z*z - R_REAL*R_REAL;
        float boundary_disc = fmaxf(boundary_b*boundary_b - boundary_c, 0.0f);
        float distance_to_boundary = -boundary_b + sqrtf(boundary_disc);

        float free_path = distance_to_boundary;
        if (MU_T > 0.0f) {
            double u_path = rnd_uniform_double(&state);
            free_path = (float)(-log(u_path) / (double)MU_T);
        }

        int exited = (MU_T <= 0.0f || free_path >= distance_to_boundary) ? 1 : 0;
        if (!exited && extinction_count >= MAX_EXTINCTIONS) {
            // The next event lies inside the sample but processing it would
            // exceed the configured interaction cap.  A ray whose next free
            // path reaches the boundary is still allowed to exit at the cap.
            truncated = 1;
            break;
        }
        float travel_distance = exited ? distance_to_boundary : free_path;

        int heatmap_steps = (int)ceilf(travel_distance / HEATMAP_SAMPLE_SPACING);
        if (heatmap_steps < 1) heatmap_steps = 1;

        const float segment_x = x;
        const float segment_y = y;
        const float segment_z = z;

        for (int hs = 0; hs < heatmap_steps; hs++) {
            // Heatmap sampling is diagnostic only.  Derive each sample from
            // the segment origin so visualization resolution cannot alter the
            // transport endpoint through accumulated stepping error.
            float fraction = (float)(hs + 1) / (float)heatmap_steps;
            float heat_x = segment_x + vx * travel_distance * fraction;
            float heat_y = segment_y + vy * travel_distance * fraction;

            int ix = (int)(((heat_x + R_REAL) / (2.0f * R_REAL)) * (float)VIS_SIZE);
            if (ix < 0) ix = 0;
            if (ix > VIS_SIZE - 1) ix = VIS_SIZE - 1;

            int iy = VIS_SIZE - 1 - (int)(((heat_y + R_REAL) / (2.0f * R_REAL)) * (float)VIS_SIZE);
            if (iy < 0) iy = 0;
            if (iy > VIS_SIZE - 1) iy = VIS_SIZE - 1;

            int pix_idx = iy * VIS_SIZE + ix;
            atomicAdd(&heatmap_flat[pix_idx], VISUAL_SCALE);
        }

        x = segment_x + vx * travel_distance;
        y = segment_y + vy * travel_distance;
        z = segment_z + vz * travel_distance;
        rpl += travel_distance;

        if (exited) {
            break;
        }

        extinction_count++;

        double u_particle = rnd_uniform_double(&state);
        int pidx = cdf_binary_search(
            extinction_cdf_table,
            n_particles,
            u_particle
        );

        double albedo_this = single_scattering_albedo_table[pidx];
        albedo_this = fmin(1.0, fmax(0.0, albedo_this));

        bool is_floc_event =
            particle_is_floc_table[pidx] > 0.5;
        if (is_floc_event) {
            floc_extinction_count++;
        }

        if (rnd_uniform_double(&state) >= albedo_this) {
            absorbed = 1;
            break;
        }

        scatter_count++;
        last_scatter_bin = pidx;
        if (is_floc_event) {
            floc_event_count++;
            last_event_was_floc = 1;
        }
        else {
            last_event_was_floc = 0;
        }

        int angle_offset = pidx * n_theta;

        if (is_floc_event && FLOC_CUDA_TRUE_INTERNAL_DOMAIN_ENABLED) {
            // CLARITAS_74 true internal floc-domain transport; no outer floc phase sample.
            // entry-to-exit displacement.  The outer extinction event is treated
            // as an encounter with a finite spherical floc domain.  The ray enters
            // the near surface, random-walks through representative-monomer
            // scatters, exits the domain, and the bulk ray resumes from the exit
            // surface rather than from the original point.
            float tau_s = (float)floc_internal_tau_s_table[pidx];
            if (!isfinite(tau_s) || tau_s < 0.0f) tau_s = 0.0f;

            float floc_diameter = (float)particle_diameter_table[pidx];
            if (!isfinite(floc_diameter) || floc_diameter <= 0.0f) {
                floc_diameter = 0.0f;
            }

            float internal_path_m = 0.0f;

            if (tau_s > 0.0f && floc_diameter > 0.0f) {
                float rf = 0.5f * floc_diameter;

                // Start at the upstream surface of a spherical floc whose centre
                // is one radius ahead of the bulk extinction point.  The local
                // entry point is therefore -v*R in centre coordinates.
                float entry_lx = -vx * rf;
                float entry_ly = -vy * rf;
                float entry_lz = -vz * rf;
                float lx = entry_lx;
                float ly = entry_ly;
                float lz = entry_lz;

                // The printed tau_s diagnostics are based on mean chord optical
                // depth.  Convert back to a local physical mean free path.
                float mean_chord = (2.0f / 3.0f) * floc_diameter;
                float internal_mfp = mean_chord / fmaxf(tau_s, 1.0e-12f);
                internal_mfp = fmaxf(internal_mfp, 1.0e-12f);

                int internal_scatter_count = 0;
                int exited_floc = 0;

                while (internal_scatter_count < FLOC_INTERNAL_DOMAIN_MAX_SCATTERS) {
                    double u_step = rnd_uniform_double(&state);
                    float step_m = (float)(
                        -log(u_step) * (double)internal_mfp
                    );

                    float nx = lx + vx * step_m;
                    float ny = ly + vy * step_m;
                    float nz = lz + vz * step_m;

                    if (nx * nx + ny * ny + nz * nz >= rf * rf) {
                        // The sampled free path leaves the spherical floc before
                        // another internal scatter.  Move exactly to the exit
                        // surface along the current direction.
                        float bq = 2.0f * (lx * vx + ly * vy + lz * vz);
                        float cq = lx * lx + ly * ly + lz * lz - rf * rf;
                        float discq = bq * bq - 4.0f * cq;
                        float t_exit = step_m;

                        if (discq >= 0.0f) {
                            float sqrt_discq = sqrtf(discq);
                            float t1 = (-bq + sqrt_discq) * 0.5f;
                            float t2 = (-bq - sqrt_discq) * 0.5f;

                            if (t1 > 0.0f && t2 > 0.0f) {
                                t_exit = fminf(t1, t2);
                            }
                            else if (t1 > 0.0f) {
                                t_exit = t1;
                            }
                            else if (t2 > 0.0f) {
                                t_exit = t2;
                            }
                        }

                        lx += vx * t_exit;
                        ly += vy * t_exit;
                        lz += vz * t_exit;
                        internal_path_m += t_exit;
                        exited_floc = 1;
                        break;
                    }

                    lx = nx;
                    ly = ny;
                    lz = nz;
                    internal_path_m += step_m;

                    double u_angle_internal = rnd_uniform_double(&state);
                    float theta_internal = sample_theta_from_cdf(
                        &internal_angle_cdf_table[angle_offset],
                        theta_table,
                        n_theta,
                        u_angle_internal
                    );
                    float phi_internal =
                        2.0f * 3.1415927f * rnd_uniform(&stateJITTER);
                    scatter_direction_3d(
                        &vx, &vy, &vz, theta_internal, phi_internal
                    );
                    internal_scatter_count++;
                }

                if (!exited_floc) {
                    // If the scatter cap is reached, project the photon to the
                    // floc boundary along its final direction so the outer
                    // transport still resumes from a physical exit point.
                    float bq = 2.0f * (lx * vx + ly * vy + lz * vz);
                    float cq = lx * lx + ly * ly + lz * lz - rf * rf;
                    float discq = bq * bq - 4.0f * cq;

                    if (discq >= 0.0f) {
                        float sqrt_discq = sqrtf(discq);
                        float t1 = (-bq + sqrt_discq) * 0.5f;
                        float t2 = (-bq - sqrt_discq) * 0.5f;
                        float t_exit = 0.0f;

                        if (t1 > 0.0f && t2 > 0.0f) {
                            t_exit = fminf(t1, t2);
                        }
                        else if (t1 > 0.0f) {
                            t_exit = t1;
                        }
                        else if (t2 > 0.0f) {
                            t_exit = t2;
                        }

                        lx += vx * t_exit;
                        ly += vy * t_exit;
                        lz += vz * t_exit;
                        internal_path_m += t_exit;
                    }
                }

                // Convert local entry-to-exit offset into a bulk-domain position
                // displacement.  This is the key CLARITAS_68 change: flocs are
                // no longer point scatterers.
                float floc_dx = lx - entry_lx;
                float floc_dy = ly - entry_ly;
                float floc_dz = lz - entry_lz;
                x += floc_dx;
                y += floc_dy;
                z += floc_dz;
                rpl += internal_path_m;
                floc_domain_dx_sum += floc_dx;
                floc_domain_dy_sum += floc_dy;
                floc_domain_dz_sum += floc_dz;
                floc_domain_path_sum += internal_path_m;
                floc_internal_scatter_count_sum += internal_scatter_count;
            }
            else {
                // Degenerate fallback: no internal optical depth or no diameter.
                // Keep the old one-angle floc CDF behaviour for numerical safety.
                double u_angle = rnd_uniform_double(&state);
                float theta_3d = sample_theta_from_cdf(
                    &angle_cdf_table[angle_offset],
                    theta_table,
                    n_theta,
                    u_angle
                );
                float phi =
                    2.0f * 3.1415927f * rnd_uniform(&stateJITTER);
                scatter_direction_3d(&vx, &vy, &vz, theta_3d, phi);
            }

            float roughness_jitter =
                gaussian_jitter(&stateJITTER, FLOC_ROUGHNESS_STD_RAD);

            if (roughness_jitter != 0.0f) {
                float phi_rough =
                    2.0f * 3.1415927f * rnd_uniform(&stateJITTER);
                scatter_direction_3d(
                    &vx, &vy, &vz, fabsf(roughness_jitter), phi_rough
                );
            }

            // The finite floc can carry the ray outside the main sample domain.
            // If so, this is a valid exit, not an absorption.
            if (x * x + y * y + z * z > R_REAL * R_REAL) {
                break;
            }
        }
        else {
            double u_angle = rnd_uniform_double(&state);

            float theta_3d = sample_theta_from_cdf(
                &angle_cdf_table[angle_offset],
                theta_table,
                n_theta,
                u_angle
            );
            float phi =
                2.0f * 3.1415927f * rnd_uniform(&stateJITTER);

            float roughness_std_this =
                is_floc_event ? FLOC_ROUGHNESS_STD_RAD : PRIMARY_ROUGHNESS_STD_RAD;

            float roughness_jitter =
                gaussian_jitter(&stateJITTER, roughness_std_this);

            scatter_direction_3d(&vx, &vy, &vz, theta_3d, phi);
            if (roughness_jitter != 0.0f) {
                float phi_rough =
                    2.0f * 3.1415927f * rnd_uniform(&stateJITTER);
                scatter_direction_3d(
                    &vx, &vy, &vz, fabsf(roughness_jitter), phi_rough
                );
            }
        }
    }

    scatter_count_out[tid] = scatter_count;
    floc_event_count_out[tid] = floc_event_count;
    floc_extinction_count_out[tid] = floc_extinction_count;
    last_event_was_floc_out[tid] = last_event_was_floc;
    last_scatter_bin_out[tid] = last_scatter_bin;
    extinction_count_out[tid] = extinction_count;
    absorbed_out[tid] = absorbed;
    truncated_out[tid] = truncated;
    terminal_state_out[tid] = absorbed ? 2 : (truncated ? 3 : 1);
    floc_domain_dx_out[tid] = floc_domain_dx_sum;
    floc_domain_dy_out[tid] = floc_domain_dy_sum;
    floc_domain_dz_out[tid] = floc_domain_dz_sum;
    floc_domain_path_out[tid] = floc_domain_path_sum;
    floc_internal_scatter_count_out[tid] = floc_internal_scatter_count_sum;
    ray_path_length_out[tid] = rpl;

    if (absorbed == 0 && truncated == 0) {
        exit_x_out[tid] = x;
        exit_y_out[tid] = y;
        exit_z_out[tid] = z;
        exit_vx_out[tid] = vx;
        exit_vy_out[tid] = vy;
        exit_vz_out[tid] = vz;
        exit_dir_out[tid] = acosf(fminf(1.0f, fmaxf(-1.0f, vy)));
    }
}
}
"""

module = cp.RawModule(code=cuda_src, options=('-std=c++11',))
trace_kernel = module.get_function('trace_kernel')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_gpu_free_bytes():
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        return int(free), int(total)
    except Exception:
        try:
            dev = cp.cuda.Device()
            mem = dev.mem_info  # may be tuple (free, total)
            if isinstance(mem, tuple) and len(mem) == 2:
                return int(mem[0]), int(mem[1])
        except Exception:
            pass
    return None, None

def estimate_chunk_size_bytes(free_bytes, safety_fraction=0.2, overhead_bytes=256*1024*1024):
    # Current chunk buffers require about 96 bytes/ray.  Use headroom for
    # allocator alignment and temporary conversion buffers.
    per_ray_bytes = 128

    usable = int(free_bytes * safety_fraction) - overhead_bytes
    if usable <= 0:
        return 0
    return max(1, usable // per_ray_bytes)


def validate_transport_tables(
    angles_init_np,
    particle_cdf_table_np,
    particle_is_floc_table_np,
    single_scattering_albedo_table_np,
    angle_cdf_table_np,
    internal_angle_cdf_table_np,
    floc_internal_tau_s_table_np,
    particle_diameter_table_np,
    theta_table_np,
):
    """Fail before CUDA launch when probability or geometry tables are invalid."""
    launch_angles = np.asarray(angles_init_np)
    event_cdf = np.asarray(particle_cdf_table_np, dtype=np.float64)
    particle_is_floc_values = np.asarray(
        particle_is_floc_table_np, dtype=np.float64
    )
    albedo = np.asarray(
        single_scattering_albedo_table_np, dtype=np.float64
    )
    phase_cdf = np.asarray(angle_cdf_table_np, dtype=np.float64)
    internal_phase_cdf = np.asarray(
        internal_angle_cdf_table_np, dtype=np.float64
    )
    internal_tau = np.asarray(
        floc_internal_tau_s_table_np, dtype=np.float64
    )
    particle_diameter = np.asarray(
        particle_diameter_table_np, dtype=np.float64
    )
    theta_values = np.asarray(theta_table_np, dtype=np.float64)

    if launch_angles.ndim != 1 or launch_angles.size < 1:
        raise ValueError("At least one one-dimensional launch angle is required")
    if not np.all(np.isfinite(launch_angles)):
        raise ValueError("Launch angles contain non-finite values")
    if (
        theta_values.ndim != 1
        or theta_values.size < 2
        or not np.all(np.isfinite(theta_values))
        or np.any(np.diff(theta_values) <= 0.0)
        or not np.isclose(
            theta_values[0], 0.0, rtol=0.0, atol=1.0e-14
        )
        or not np.isclose(
            theta_values[-1], np.pi, rtol=0.0, atol=1.0e-14
        )
    ):
        raise ValueError(
            "Theta table must be finite, strictly increasing, and span "
            "exactly 0 to pi radians"
        )

    n_particles = event_cdf.size
    expected_particle_shape = (n_particles,)
    for name, values in (
        ("particle event CDF", event_cdf),
        ("particle-is-floc table", particle_is_floc_values),
        ("single-scattering albedo", albedo),
        ("internal floc optical depth", internal_tau),
        ("particle diameter", particle_diameter),
    ):
        if values.shape != expected_particle_shape:
            raise ValueError(
                f"{name} has shape {values.shape}; expected "
                f"{expected_particle_shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")

    if (
        n_particles < 1
        or np.any(event_cdf < 0.0)
        or np.any(np.diff(event_cdf) < -1.0e-14)
        or not np.isclose(event_cdf[-1], 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(
            "Particle event CDF must be nonnegative, monotone, and end at one"
        )
    if np.any((particle_is_floc_values != 0.0) & (particle_is_floc_values != 1.0)):
        raise ValueError("particle-is-floc table must contain only zero or one")
    if np.any((albedo < 0.0) | (albedo > 1.0)):
        raise ValueError("Single-scattering albedo must lie in [0, 1]")
    if np.any(internal_tau < 0.0):
        raise ValueError("Internal floc optical depths cannot be negative")
    if np.any(particle_diameter < 0.0):
        raise ValueError("Particle diameters cannot be negative")

    expected_phase_shape = (n_particles, theta_values.size)
    for name, values in (
        ("outer phase CDF", phase_cdf),
        ("internal phase CDF", internal_phase_cdf),
    ):
        if values.shape != expected_phase_shape:
            raise ValueError(
                f"{name} has shape {values.shape}; expected "
                f"{expected_phase_shape}"
            )
        if (
            not np.all(np.isfinite(values))
            or np.any(np.diff(values, axis=1) < -1.0e-12)
            or not np.allclose(values[:, 0], 0.0, rtol=0.0, atol=1.0e-7)
            or not np.allclose(values[:, -1], 1.0, rtol=0.0, atol=1.0e-7)
        ):
            raise ValueError(
                f"{name} must be finite, monotone, start at zero, and end at one"
            )

    if not np.isfinite(mu_t) or mu_t < 0.0:
        raise ValueError("Bulk extinction coefficient mu_t must be finite and nonnegative")
    if MAX_EXTINCTIONS < 0:
        raise ValueError("MAX_EXTINCTIONS cannot be negative")


from tqdm import tqdm

def trace_rays_gpu(angles_init_np, primary_roughness_std_rad, floc_roughness_std_rad,
                   R_REAL, RAY_OFFSET, VIS_SIZE,
                   VISUAL_SCALE, particle_cdf_table_np,
                   particle_is_floc_table_np, single_scattering_albedo_table_np, angle_cdf_table_np,
                   internal_angle_cdf_table_np, floc_internal_tau_s_table_np, particle_diameter_table_np,
                   theta_table_np, wavelength_m, material_name,
                   concentration_g_per_L,
                   hdf5_file="ray_exits.h5",
                   safety_fraction=0.01,
                   min_chunk=GPU_MIN_CHUNK_RAYS,
                   max_chunk=GPU_MAX_CHUNK_RAYS):
    validate_transport_tables(
        angles_init_np,
        particle_cdf_table_np,
        particle_is_floc_table_np,
        single_scattering_albedo_table_np,
        angle_cdf_table_np,
        internal_angle_cdf_table_np,
        floc_internal_tau_s_table_np,
        particle_diameter_table_np,
        theta_table_np,
    )
    N = angles_init_np.shape[0]
    max_unique_xorshift32_initial_states = int(np.iinfo(np.uint32).max)
    if N > max_unique_xorshift32_initial_states:
        raise ValueError(
            "This xorshift32 transport supports at most "
            f"{max_unique_xorshift32_initial_states:,} rays without repeating a "
            "nonzero initial RNG state"
        )

    particle_cdf_dev = cp.asarray(particle_cdf_table_np, dtype=cp.float64)
    particle_is_floc_dev = cp.asarray(particle_is_floc_table_np, dtype=cp.float64)
    single_scattering_albedo_dev = cp.asarray(single_scattering_albedo_table_np, dtype=cp.float64)
    angle_cdf_dev = cp.asarray(angle_cdf_table_np, dtype=cp.float64)
    internal_angle_cdf_dev = cp.asarray(internal_angle_cdf_table_np, dtype=cp.float64)
    floc_internal_tau_s_dev = cp.asarray(floc_internal_tau_s_table_np, dtype=cp.float64)
    particle_diameter_dev = cp.asarray(particle_diameter_table_np, dtype=cp.float64)
    theta_dev = cp.asarray(theta_table_np, dtype=cp.float64)
    heatmap_dev = cp.zeros((VIS_SIZE*VIS_SIZE,), dtype=cp.float32)
    threads_per_block = 256

    free_bytes, total_bytes = get_gpu_free_bytes()
    if free_bytes is None:
        estimated_chunk = 2_000_000
    else:
        est = estimate_chunk_size_bytes(free_bytes, safety_fraction=safety_fraction)
        estimated_chunk = int(max(min_chunk, min(est, max_chunk)))
    estimated_chunk = max(1, min(int(estimated_chunk), int(N)))

    with h5py.File(hdf5_file, "w") as f:
        f.attrs["claritas_version"] = "77"
        f.attrs["simulation_seed"] = np.uint64(SIMULATION_SEED)
        f.attrs["transport_rng"] = "per_ray_xorshift32_global_ray_id"
        f.attrs["rng_seed_mapping"] = (
            "nonzero_affine_mod_2^32_minus_1"
        )
        f.attrs["maximum_unique_rng_initial_states"] = np.uint64(
            max_unique_xorshift32_initial_states
        )
        f.attrs["source_rng"] = (
            "numpy_pcg64_angles+xorshift32_beam_offset"
            if SOURCE_MODE == "production_beta"
            else "none_collimated"
        )
        f.attrs["transport_geometry"] = "3d_sphere"
        f.attrs["source_model"] = SOURCE_MODE
        f.attrs["detector_geometry"] = "ideal_annular_nearest_accepted_band"
        f.attrs["phase_function_measure"] = PHASE_FUNCTION_MEASURE
        f.attrs["material"] = str(material_name)
        f.attrs["concentration_g_per_L"] = float(concentration_g_per_L)
        f.attrs["wavelength_m"] = float(wavelength_m)
        f.attrs["psd_weight_mode"] = PSD_WEIGHT_MODE
        f.attrs["particle_density_kg_per_m3"] = float(
            particle_density_kg_per_m3
        )
        f.attrs["primary_refractive_index_real"] = float(n_particle)
        f.attrs["primary_refractive_index_imag_k"] = float(
            PRIMARY_REFRACTIVE_INDEX_IMAG_K
        )
        f.attrs["floc_enabled"] = bool(FLOC_ENABLED)
        f.attrs["floc_internal_domain_enabled"] = bool(
            FLOC_ENABLED and FLOC_CUDA_TRUE_INTERNAL_DOMAIN_ENABLED
        )
        f.attrs["phase_cache_hash"] = mie_cache_hash
        f.attrs["python_source_sha256"] = sha256_file(__file__)
        f.attrs["cuda_source_sha256"] = hashlib.sha256(
            cuda_src.encode("utf-8")
        ).hexdigest()
        f.attrs["terminal_state_codes"] = (
            "1=exited,2=absorbed,3=truncated,4=missed_sample"
        )
        f.attrs["n_rays"] = np.int64(N)
        f.attrs["mu_s_m_inv"] = float(mu_s)
        f.attrs["mu_a_m_inv"] = float(mu_a)
        f.attrs["mu_t_m_inv"] = float(mu_t)
        f.attrs["sample_radius_m"] = float(R_REAL)
        f.attrs["ray_offset_m"] = float(RAY_OFFSET)
        f.attrs["production_beam_sigma_m"] = float(
            PRODUCTION_BEAM_SIGMA_M
        )
        f.attrs["production_source_beta_alpha1"] = float(alpha1)
        f.attrs["production_source_beta_alpha2"] = float(alpha2)
        f.attrs["phase_theta_grid_size"] = np.int64(len(theta_table_np))
        f.attrs["phase_theta_min_rad"] = float(theta_table_np[0])
        f.attrs["phase_theta_max_rad"] = float(theta_table_np[-1])
        f.attrs["phase_theta_step_rad"] = float(
            theta_table_np[1] - theta_table_np[0]
        )
        f.attrs["detector_centres_deg"] = np.asarray(
            detector_angles, dtype=np.float64
        )
        f.attrs["detector_acceptance_half_angle_deg"] = float(
            detector_acceptance_deg
        )
        f.attrs["max_extinctions"] = np.int64(MAX_EXTINCTIONS)
        f.attrs["initial_chunk_rays"] = np.int64(estimated_chunk)
        dset_exit_x = f.create_dataset("exit_x", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_exit_y = f.create_dataset("exit_y", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_exit_z = f.create_dataset("exit_z", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_exit_vx = f.create_dataset("exit_vx", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_exit_vy = f.create_dataset("exit_vy", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_exit_vz = f.create_dataset("exit_vz", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_exit_dir = f.create_dataset("exit_dir", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_rpl = f.create_dataset("exit_rpl", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_scatter_count = f.create_dataset("scatter_count", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_floc_event_count = f.create_dataset("floc_event_count", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_floc_extinction_count = f.create_dataset("floc_extinction_count", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_last_event_was_floc = f.create_dataset("last_event_was_floc", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_last_scatter_bin = f.create_dataset("last_scatter_bin", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_extinction_count = f.create_dataset("extinction_count", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_absorbed = f.create_dataset("absorbed", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_truncated = f.create_dataset("truncated", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_terminal_state = f.create_dataset("terminal_state", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_floc_domain_dx = f.create_dataset("floc_domain_dx", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_floc_domain_dy = f.create_dataset("floc_domain_dy", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_floc_domain_dz = f.create_dataset("floc_domain_dz", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_floc_domain_path = f.create_dataset("floc_domain_path", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_floc_internal_scatter_count = f.create_dataset("floc_internal_scatter_count", shape=(N,), dtype='i4', chunks=(estimated_chunk,))

        seed_sequence = np.random.SeedSequence(SIMULATION_SEED)
        seed_values = seed_sequence.generate_state(2, dtype=np.uint32)
        transport_seed, azimuth_seed = (
            np.uint32(value) for value in seed_values
        )
        f.attrs["transport_seed_uint32"] = transport_seed
        f.attrs["azimuth_seed_uint32"] = azimuth_seed

        def allocate_chunk_buffers(start_index, end_index):
            chunk_size = end_index - start_index
            return {
                "angles": cp.asarray(
                    angles_init_np[start_index:end_index],
                    dtype=cp.float64
                ),
                "exit_dir": cp.full((chunk_size,), cp.nan, dtype=cp.float32),
                "exit_x": cp.full((chunk_size,), cp.nan, dtype=cp.float32),
                "exit_y": cp.full((chunk_size,), cp.nan, dtype=cp.float32),
                "exit_z": cp.full((chunk_size,), cp.nan, dtype=cp.float32),
                "exit_vx": cp.full((chunk_size,), cp.nan, dtype=cp.float32),
                "exit_vy": cp.full((chunk_size,), cp.nan, dtype=cp.float32),
                "exit_vz": cp.full((chunk_size,), cp.nan, dtype=cp.float32),
                "rpl": cp.full((chunk_size,), cp.nan, dtype=cp.float32),
                "scatter_count": cp.full((chunk_size,), -1, dtype=cp.int32),
                "floc_event_count": cp.full((chunk_size,), -1, dtype=cp.int32),
                "floc_extinction_count": cp.full(
                    (chunk_size,), -1, dtype=cp.int32
                ),
                "last_event_was_floc": cp.full(
                    (chunk_size,), -1, dtype=cp.int32
                ),
                "last_scatter_bin": cp.full(
                    (chunk_size,), -1, dtype=cp.int32
                ),
                "extinction_count": cp.full(
                    (chunk_size,), -1, dtype=cp.int32
                ),
                "absorbed": cp.full((chunk_size,), -1, dtype=cp.int32),
                "truncated": cp.full((chunk_size,), -1, dtype=cp.int32),
                "terminal_state": cp.zeros((chunk_size,), dtype=cp.int32),
                "floc_domain_dx": cp.zeros((chunk_size,), dtype=cp.float32),
                "floc_domain_dy": cp.zeros((chunk_size,), dtype=cp.float32),
                "floc_domain_dz": cp.zeros((chunk_size,), dtype=cp.float32),
                "floc_domain_path": cp.zeros((chunk_size,), dtype=cp.float32),
                "floc_internal_scatter_count": cp.zeros(
                    (chunk_size,), dtype=cp.int32
                ),
            }

        start = 0
        active_chunk = estimated_chunk
        oom_retry_count = 0
        with tqdm(total=N, unit="ray", desc="Tracing rays") as progress:
            while start < N:
                end = min(N, start + active_chunk)
                sz = end - start
                buffers = None
                try:
                    buffers = allocate_chunk_buffers(start, end)
                    blocks = (
                        sz + threads_per_block - 1
                    ) // threads_per_block
                    trace_kernel((blocks,), (threads_per_block,),
                        (
                            np.int32(MAX_EXTINCTIONS),
                            np.float32(mu_t),
                            np.float32(primary_roughness_std_rad),
                            np.float32(floc_roughness_std_rad),
                            np.float32(R_REAL),
                            np.float32(RAY_OFFSET),
                            np.int32(VIS_SIZE),
                            np.float32(VISUAL_SCALE),
                            np.int32(
                                1 if SOURCE_MODE == "reference_collimated" else 0
                            ),
                            np.float32(PRODUCTION_BEAM_SIGMA_M),
                            buffers["angles"],
                            np.int32(sz),
                            particle_cdf_dev,
                            particle_is_floc_dev,
                            single_scattering_albedo_dev,
                            angle_cdf_dev,
                            internal_angle_cdf_dev,
                            floc_internal_tau_s_dev,
                            particle_diameter_dev,
                            theta_dev,
                            np.int32(len(particle_cdf_table_np)),
                            np.int32(len(theta_table_np)),
                            np.int32(
                                1
                                if (
                                    FLOC_ENABLED
                                    and FLOC_CUDA_TRUE_INTERNAL_DOMAIN_ENABLED
                                )
                                else 0
                            ),
                            np.int32(FLOC_INTERNAL_DOMAIN_MAX_SCATTERS),
                            heatmap_dev,
                            buffers["exit_dir"],
                            buffers["exit_x"],
                            buffers["exit_y"],
                            buffers["exit_z"],
                            buffers["exit_vx"],
                            buffers["exit_vy"],
                            buffers["exit_vz"],
                            buffers["rpl"],
                            buffers["scatter_count"],
                            buffers["floc_event_count"],
                            buffers["floc_extinction_count"],
                            buffers["last_event_was_floc"],
                            buffers["last_scatter_bin"],
                            buffers["extinction_count"],
                            buffers["absorbed"],
                            buffers["truncated"],
                            buffers["terminal_state"],
                            buffers["floc_domain_dx"],
                            buffers["floc_domain_dy"],
                            buffers["floc_domain_dz"],
                            buffers["floc_domain_path"],
                            buffers["floc_internal_scatter_count"],
                            np.uint32(transport_seed),
                            np.uint32(azimuth_seed),
                            np.uint64(start)
                        ))
                    cp.cuda.Stream.null.synchronize()
                except cp.cuda.memory.OutOfMemoryError as error:
                    buffers = None
                    cp.get_default_memory_pool().free_all_blocks()
                    if sz <= 1:
                        raise RuntimeError(
                            "CUDA out of memory even for a one-ray chunk"
                        ) from error
                    active_chunk = max(1, sz // 2)
                    oom_retry_count += 1
                    progress.write(
                        "CUDA OOM: retrying the same global ray index "
                        f"{start} with chunk size {active_chunk}"
                    )
                    continue

                dset_exit_dir[start:end] = cp.asnumpy(buffers["exit_dir"])
                dset_exit_x[start:end] = cp.asnumpy(buffers["exit_x"])
                dset_exit_y[start:end] = cp.asnumpy(buffers["exit_y"])
                dset_exit_z[start:end] = cp.asnumpy(buffers["exit_z"])
                dset_exit_vx[start:end] = cp.asnumpy(buffers["exit_vx"])
                dset_exit_vy[start:end] = cp.asnumpy(buffers["exit_vy"])
                dset_exit_vz[start:end] = cp.asnumpy(buffers["exit_vz"])
                dset_rpl[start:end] = cp.asnumpy(buffers["rpl"])
                dset_scatter_count[start:end] = cp.asnumpy(
                    buffers["scatter_count"]
                )
                dset_floc_event_count[start:end] = cp.asnumpy(
                    buffers["floc_event_count"]
                )
                dset_floc_extinction_count[start:end] = cp.asnumpy(
                    buffers["floc_extinction_count"]
                )
                dset_last_event_was_floc[start:end] = cp.asnumpy(
                    buffers["last_event_was_floc"]
                )
                dset_last_scatter_bin[start:end] = cp.asnumpy(
                    buffers["last_scatter_bin"]
                )
                dset_extinction_count[start:end] = cp.asnumpy(
                    buffers["extinction_count"]
                )
                dset_absorbed[start:end] = cp.asnumpy(buffers["absorbed"])
                dset_truncated[start:end] = cp.asnumpy(buffers["truncated"])
                dset_terminal_state[start:end] = cp.asnumpy(
                    buffers["terminal_state"]
                )
                dset_floc_domain_dx[start:end] = cp.asnumpy(
                    buffers["floc_domain_dx"]
                )
                dset_floc_domain_dy[start:end] = cp.asnumpy(
                    buffers["floc_domain_dy"]
                )
                dset_floc_domain_dz[start:end] = cp.asnumpy(
                    buffers["floc_domain_dz"]
                )
                dset_floc_domain_path[start:end] = cp.asnumpy(
                    buffers["floc_domain_path"]
                )
                dset_floc_internal_scatter_count[start:end] = cp.asnumpy(
                    buffers["floc_internal_scatter_count"]
                )

                buffers = None
                cp.get_default_memory_pool().free_all_blocks()
                start = end
                progress.update(sz)
        f.attrs["final_chunk_rays"] = np.int64(active_chunk)
        f.attrs["oom_retry_count"] = np.int64(oom_retry_count)

    heatmap = cp.asnumpy(heatmap_dev).reshape((VIS_SIZE, VIS_SIZE))
    return heatmap



def spherical_polar_angle_deg(exit_x, exit_y, exit_z):
    """Polar boundary angle measured from the forward +y axis."""
    radius = np.sqrt(exit_x**2 + exit_y**2 + exit_z**2)
    cos_angle = np.divide(
        exit_y,
        radius,
        out=np.zeros_like(radius, dtype=np.float64),
        where=radius > 0.0
    )
    return np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def save_detector_transport_breakdown(
    wl_nm,
    exit_x,
    exit_y,
    exit_z,
    exit_dirs,
    exit_rpl,
    scatter_count,
    floc_event_count,
    last_event_was_floc,
    extinction_count,
    absorbed_flag,
    detector_angles_deg,
    detector_acceptance_deg,
    outdir
):
    exit_x = np.asarray(exit_x)
    exit_y = np.asarray(exit_y)
    exit_z = np.asarray(exit_z)
    exit_dirs = np.asarray(exit_dirs)
    exit_rpl = np.asarray(exit_rpl)
    scatter_count = np.asarray(scatter_count)
    floc_event_count = np.asarray(floc_event_count)
    last_event_was_floc = np.asarray(last_event_was_floc)
    extinction_count = np.asarray(extinction_count)
    absorbed_flag = np.asarray(absorbed_flag)

    valid_exit_mask = (
        np.isfinite(exit_x) &
        np.isfinite(exit_y) &
        np.isfinite(exit_z) &
        np.isfinite(exit_dirs) &
        (scatter_count >= 0)
    )

    exit_position_angle_deg = np.full_like(exit_x, np.nan, dtype=np.float64)
    exit_position_angle_deg[valid_exit_mask] = spherical_polar_angle_deg(
        exit_x[valid_exit_mask],
        exit_y[valid_exit_mask],
        exit_z[valid_exit_mask]
    )

    rows = []
    n_total_rays = int(len(scatter_count))
    n_valid_exit = int(np.sum(valid_exit_mask))
    n_absorbed_or_invalid = n_total_rays - n_valid_exit

    assigned_detector = assign_detector_indices_from_angles(
        exit_position_angle_deg,
        detector_angles_deg,
        detector_acceptance_deg,
        valid_exit_mask
    )
    for detector_idx, centre in enumerate(detector_angles_deg):
        hit_mask = assigned_detector == detector_idx

        n_hit = int(np.sum(hit_mask))

        if n_hit > 0:
            sc = scatter_count[hit_mask]
            fc = floc_event_count[hit_mask]
            le = last_event_was_floc[hit_mask]
            ex = extinction_count[hit_mask]
            rpl = exit_rpl[hit_mask]

            mean_scatter = float(np.mean(sc))
            median_scatter = float(np.median(sc))
            mean_floc = float(np.mean(fc))
            frac_any_floc = float(np.mean(fc > 0))
            frac_last_floc = float(np.mean(le > 0))
            mean_extinction = float(np.mean(ex))
            mean_rpl = float(np.nanmean(rpl))
            frac_0 = float(np.mean(sc == 0))
            frac_1 = float(np.mean(sc == 1))
            frac_2 = float(np.mean(sc == 2))
            frac_3 = float(np.mean(sc == 3))
            frac_4 = float(np.mean(sc == 4))
            frac_5plus = float(np.mean(sc >= 5))
        else:
            mean_scatter = np.nan
            median_scatter = np.nan
            mean_floc = np.nan
            frac_any_floc = np.nan
            frac_last_floc = np.nan
            mean_extinction = np.nan
            mean_rpl = np.nan
            frac_0 = np.nan
            frac_1 = np.nan
            frac_2 = np.nan
            frac_3 = np.nan
            frac_4 = np.nan
            frac_5plus = np.nan

        rows.append({
            "detector_angle_deg": float(centre),
            "hit_count": n_hit,
            "hit_fraction_of_all_rays": float(n_hit / n_total_rays) if n_total_rays > 0 else 0.0,
            "hit_fraction_of_valid_exits": float(n_hit / n_valid_exit) if n_valid_exit > 0 else 0.0,
            "mean_scatter_count": mean_scatter,
            "median_scatter_count": median_scatter,
            "mean_floc_event_count": mean_floc,
            "fraction_any_floc_event": frac_any_floc,
            "fraction_last_event_floc": frac_last_floc,
            "fraction_0_scatter": frac_0,
            "fraction_1_scatter": frac_1,
            "fraction_2_scatter": frac_2,
            "fraction_3_scatter": frac_3,
            "fraction_4_scatter": frac_4,
            "fraction_5plus_scatter": frac_5plus,
            "mean_extinction_count": mean_extinction,
            "mean_exit_path_length_m": mean_rpl,
        })

    df = pd.DataFrame(rows)
    total_detector_hits = float(df["hit_count"].sum()) if not df.empty else 0.0
    if total_detector_hits > 0.0:
        df["hit_fraction_of_detector_total"] = df["hit_count"] / total_detector_hits
    else:
        df["hit_fraction_of_detector_total"] = 0.0

    overview = pd.DataFrame([{
        "total_rays": n_total_rays,
        "valid_exit_rays": n_valid_exit,
        "absorbed_or_invalid_rays": n_absorbed_or_invalid,
        "valid_exit_fraction": float(n_valid_exit / n_total_rays) if n_total_rays > 0 else 0.0,
        "absorbed_or_invalid_fraction": float(n_absorbed_or_invalid / n_total_rays) if n_total_rays > 0 else 0.0,
        "mean_scatter_count_valid_exits": float(np.mean(scatter_count[valid_exit_mask])) if n_valid_exit > 0 else np.nan,
        "mean_floc_event_count_valid_exits": float(np.mean(floc_event_count[valid_exit_mask])) if n_valid_exit > 0 else np.nan,
        "fraction_valid_exits_with_any_floc": float(np.mean(floc_event_count[valid_exit_mask] > 0)) if n_valid_exit > 0 else np.nan,
        "fraction_valid_exits_last_event_floc": float(np.mean(last_event_was_floc[valid_exit_mask] > 0)) if n_valid_exit > 0 else np.nan,
        "mean_scatter_count_absorbed_or_invalid": float(np.mean(scatter_count[~valid_exit_mask & (scatter_count >= 0)])) if np.any(~valid_exit_mask & (scatter_count >= 0)) else np.nan,
        "mean_floc_event_count_absorbed_or_invalid": float(np.mean(floc_event_count[~valid_exit_mask & (floc_event_count >= 0)])) if np.any(~valid_exit_mask & (floc_event_count >= 0)) else np.nan,
    }])

    breakdown_path = os.path.join(
        outdir,
        f"detector_transport_breakdown_{wl_nm}nm.csv"
    )
    overview_path = os.path.join(
        outdir,
        f"detector_transport_overview_{wl_nm}nm.csv"
    )
    df.to_csv(breakdown_path, index=False)
    overview.to_csv(overview_path, index=False)

    print(f"✅ Saved {breakdown_path}")
    print(f"✅ Saved {overview_path}")

    if not df.empty:
        rear = df[df["detector_angle_deg"] >= 90.0]
        if not rear.empty:
            print("=========== CLARITAS_57 DETECTOR TRANSPORT BREAKDOWN ===========")
            print(
                "Rear detectors >=90 deg: "
                f"mean hit fraction={rear['hit_fraction_of_detector_total'].sum():.6f}, "
                f"mean scatter={rear['mean_scatter_count'].mean():.3f}, "
                f"any floc={rear['fraction_any_floc_event'].mean():.3f}, "
                f"last floc={rear['fraction_last_event_floc'].mean():.3f}"
            )
            print("===============================================================")

    return df, overview



def save_detector_response_by_scatter_class(
    wl_nm,
    exit_x,
    exit_y,
    exit_z,
    scatter_count,
    floc_event_count,
    last_event_was_floc,
    detector_angles_deg,
    detector_acceptance_deg,
    outdir
):
    exit_x = np.asarray(exit_x)
    exit_y = np.asarray(exit_y)
    exit_z = np.asarray(exit_z)
    scatter_count = np.asarray(scatter_count)
    floc_event_count = np.asarray(floc_event_count)
    last_event_was_floc = np.asarray(last_event_was_floc)

    valid_exit_mask = (
        np.isfinite(exit_x) &
        np.isfinite(exit_y) &
        np.isfinite(exit_z) &
        (scatter_count >= 0)
    )

    exit_position_angle_deg = np.full_like(exit_x, np.nan, dtype=np.float64)
    exit_position_angle_deg[valid_exit_mask] = spherical_polar_angle_deg(
        exit_x[valid_exit_mask],
        exit_y[valid_exit_mask],
        exit_z[valid_exit_mask]
    )

    class_defs = [
        ("ballistic_0", scatter_count == 0),
        ("single_1", scatter_count == 1),
        ("low_multiple_2_to_5", (scatter_count >= 2) & (scatter_count <= 5)),
        ("high_multiple_gt_5", scatter_count > 5),
    ]

    rows = []
    total_hits_all_classes = 0
    total_hits_by_class = {name: 0 for name, _ in class_defs}

    assigned_detector = assign_detector_indices_from_angles(
        exit_position_angle_deg,
        detector_angles_deg,
        detector_acceptance_deg,
        valid_exit_mask
    )
    # First pass: exclusive raw counts by detector and class.
    for detector_idx, centre in enumerate(detector_angles_deg):
        hit_mask_base = assigned_detector == detector_idx

        total_detector_hits = int(np.sum(hit_mask_base))
        total_hits_all_classes += total_detector_hits

        for class_name, class_mask in class_defs:
            hit_mask = hit_mask_base & class_mask
            n_hit = int(np.sum(hit_mask))
            total_hits_by_class[class_name] += n_hit

            if n_hit > 0:
                mean_scatter = float(np.mean(scatter_count[hit_mask]))
                mean_floc_events = float(np.mean(floc_event_count[hit_mask]))
                frac_any_floc = float(np.mean(floc_event_count[hit_mask] > 0))
                frac_last_floc = float(np.mean(last_event_was_floc[hit_mask] > 0))
            else:
                mean_scatter = np.nan
                mean_floc_events = np.nan
                frac_any_floc = np.nan
                frac_last_floc = np.nan

            rows.append({
                "detector_angle_deg": float(centre),
                "scatter_class": class_name,
                "hit_count": n_hit,
                "detector_total_hit_count": total_detector_hits,
                "fraction_of_this_detector_hits": (
                    float(n_hit / total_detector_hits) if total_detector_hits > 0 else 0.0
                ),
                "mean_scatter_count": mean_scatter,
                "mean_floc_event_count": mean_floc_events,
                "fraction_any_floc_event": frac_any_floc,
                "fraction_last_event_floc": frac_last_floc,
            })

    df = pd.DataFrame(rows)

    if total_hits_all_classes > 0:
        df["fraction_of_all_detector_hits"] = df["hit_count"] / float(total_hits_all_classes)
    else:
        df["fraction_of_all_detector_hits"] = 0.0

    # Add class-normalised detector profiles.  These are useful because they
    df["fraction_of_class_hits"] = 0.0
    for class_name, total_class_hits in total_hits_by_class.items():
        class_sel = df["scatter_class"] == class_name
        if total_class_hits > 0:
            df.loc[class_sel, "fraction_of_class_hits"] = (
                df.loc[class_sel, "hit_count"] / float(total_class_hits)
            )

    detector_totals = (
        df.groupby("detector_angle_deg", as_index=False)["hit_count"]
        .sum()
        .rename(columns={"hit_count": "total_hit_count"})
    )
    total_detector_hits_all = float(detector_totals["total_hit_count"].sum())
    detector_totals["normalised_detector_response"] = (
        detector_totals["total_hit_count"] / total_detector_hits_all
        if total_detector_hits_all > 0 else 0.0
    )

    # Wide matrix: one row per detector, one column per scatter class containing
    # fraction of all detector hits.  This is quick to compare against measured
    # detector tables.
    matrix = df.pivot_table(
        index="detector_angle_deg",
        columns="scatter_class",
        values="fraction_of_all_detector_hits",
        aggfunc="sum",
        fill_value=0.0
    ).reset_index()
    matrix = matrix.merge(detector_totals, on="detector_angle_deg", how="left")

    overview_rows = []
    n_total_rays = int(len(scatter_count))
    n_valid_exit = int(np.sum(valid_exit_mask))
    rear_mask_all = df["detector_angle_deg"] >= 90.0

    for class_name, class_mask in class_defs:
        valid_class = valid_exit_mask & class_mask
        class_rows = df[df["scatter_class"] == class_name]
        rear_rows = class_rows[class_rows["detector_angle_deg"] >= 90.0]

        class_detector_hits = int(total_hits_by_class[class_name])
        class_rear_hits = int(rear_rows["hit_count"].sum()) if not rear_rows.empty else 0

        overview_rows.append({
            "scatter_class": class_name,
            "valid_exit_ray_count": int(np.sum(valid_class)),
            "valid_exit_ray_fraction": float(np.sum(valid_class) / n_valid_exit) if n_valid_exit > 0 else 0.0,
            "detector_hit_count": class_detector_hits,
            "detector_hit_fraction_of_all_detector_hits": (
                float(class_detector_hits / total_hits_all_classes)
                if total_hits_all_classes > 0 else 0.0
            ),
            "rear_detector_hit_count": class_rear_hits,
            "rear_fraction_within_class_detector_hits": (
                float(class_rear_hits / class_detector_hits)
                if class_detector_hits > 0 else 0.0
            ),
            "mean_floc_event_count_valid_exits": (
                float(np.mean(floc_event_count[valid_class])) if np.any(valid_class) else np.nan
            ),
            "fraction_any_floc_valid_exits": (
                float(np.mean(floc_event_count[valid_class] > 0)) if np.any(valid_class) else np.nan
            ),
            "fraction_last_floc_valid_exits": (
                float(np.mean(last_event_was_floc[valid_class] > 0)) if np.any(valid_class) else np.nan
            ),
        })

    overview = pd.DataFrame(overview_rows)

    detail_path = os.path.join(outdir, f"detector_response_by_scatter_class_{wl_nm}nm.csv")
    matrix_path = os.path.join(outdir, f"detector_response_by_scatter_class_matrix_{wl_nm}nm.csv")
    overview_path = os.path.join(outdir, f"detector_response_by_scatter_class_overview_{wl_nm}nm.csv")

    df.to_csv(detail_path, index=False)
    matrix.to_csv(matrix_path, index=False)
    overview.to_csv(overview_path, index=False)

    print(f"✅ Saved {detail_path}")
    print(f"✅ Saved {matrix_path}")
    print(f"✅ Saved {overview_path}")

    print("=========== CLARITAS_67 SCATTER-CLASS DETECTOR RESPONSE ===========")
    if not overview.empty:
        for _, row in overview.iterrows():
            print(
                f"{row['scatter_class']}: "
                f"detector={row['detector_hit_fraction_of_all_detector_hits']:.3f}, "
                f"rear_within_class={row['rear_fraction_within_class_detector_hits']:.3f}, "
                f"valid_exit={row['valid_exit_ray_fraction']:.3f}, "
                f"any_floc={row['fraction_any_floc_valid_exits']:.3f}"
            )
    print("==================================================================")

    return df, matrix, overview


def save_detector_last_scatter_bin_contribution(
    wl_nm,
    exit_x,
    exit_y,
    exit_z,
    scatter_count,
    last_scatter_bin,
    detector_angles_deg,
    detector_acceptance_deg,
    outdir
):
    exit_x = np.asarray(exit_x)
    exit_y = np.asarray(exit_y)
    exit_z = np.asarray(exit_z)
    scatter_count = np.asarray(scatter_count)
    last_scatter_bin = np.asarray(last_scatter_bin)

    valid_exit_mask = (
        np.isfinite(exit_x) &
        np.isfinite(exit_y) &
        np.isfinite(exit_z) &
        (scatter_count >= 0)
    )

    exit_position_angle_deg = np.full_like(exit_x, np.nan, dtype=np.float64)
    exit_position_angle_deg[valid_exit_mask] = spherical_polar_angle_deg(
        exit_x[valid_exit_mask],
        exit_y[valid_exit_mask],
        exit_z[valid_exit_mask]
    )

    n_bins = len(particle_diameter_m)
    n_det = len(detector_angles_deg)

    count_matrix = np.zeros((n_bins, n_det), dtype=np.int64)
    detector_hit_counts = np.zeros(n_det, dtype=np.int64)

    assigned_detector = assign_detector_indices_from_angles(
        exit_position_angle_deg,
        detector_angles_deg,
        detector_acceptance_deg,
        valid_exit_mask
    )
    for j, centre in enumerate(detector_angles_deg):
        hit_mask = assigned_detector == j

        detector_hit_counts[j] = int(np.sum(hit_mask))

        bins_hit = last_scatter_bin[hit_mask]
        bins_hit = bins_hit[(bins_hit >= 0) & (bins_hit < n_bins)]
        if bins_hit.size > 0:
            binc = np.bincount(bins_hit.astype(np.int64), minlength=n_bins)
            count_matrix[:, j] = binc[:n_bins]

    total_assigned_hits = int(np.sum(count_matrix))
    rear_cols = np.where(np.asarray(detector_angles_deg) >= 90.0)[0]

    matrix_rows = []
    for i in range(n_bins):
        row = {
            "bin_index": i,
            "kind": "floc" if bool(particle_is_floc[i]) else "primary",
            "effective_diameter_um": float(particle_diameter_m[i] * 1.0e6),
            "source_primary_min_um": float(source_primary_min_diameter_m[i] * 1.0e6),
            "source_primary_max_um": float(source_primary_max_diameter_m[i] * 1.0e6),
            "mass_fraction_percent": float(particle_weights[i] * 100.0),
            "mu_t_fraction_percent": float(mu_t_by_bin[i] / mu_t * 100.0) if mu_t > 0.0 else 0.0,
            "event_probability_percent": float(particle_event_weights[i] * 100.0),
        }
        for j, centre in enumerate(detector_angles_deg):
            row[f"detector_{int(centre)}deg_last_scatter_count"] = int(count_matrix[i, j])
            row[f"detector_{int(centre)}deg_fraction_of_detector_hits"] = (
                float(count_matrix[i, j] / detector_hit_counts[j])
                if detector_hit_counts[j] > 0 else 0.0
            )
        row["total_last_scatter_detector_hits"] = int(np.sum(count_matrix[i, :]))
        row["rear_last_scatter_detector_hits"] = int(np.sum(count_matrix[i, rear_cols])) if rear_cols.size > 0 else 0
        row["fraction_of_all_assigned_detector_hits"] = (
            float(row["total_last_scatter_detector_hits"] / total_assigned_hits)
            if total_assigned_hits > 0 else 0.0
        )
        row["fraction_of_rear_assigned_detector_hits"] = (
            float(row["rear_last_scatter_detector_hits"] / np.sum(count_matrix[:, rear_cols]))
            if rear_cols.size > 0 and np.sum(count_matrix[:, rear_cols]) > 0 else 0.0
        )
        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows)
    summary_df = matrix_df[[
        "bin_index", "kind", "effective_diameter_um",
        "source_primary_min_um", "source_primary_max_um",
        "mass_fraction_percent", "mu_t_fraction_percent",
        "event_probability_percent", "total_last_scatter_detector_hits",
        "rear_last_scatter_detector_hits",
        "fraction_of_all_assigned_detector_hits",
        "fraction_of_rear_assigned_detector_hits"
    ]].copy()
    summary_df = summary_df.sort_values(
        ["rear_last_scatter_detector_hits", "total_last_scatter_detector_hits"],
        ascending=False
    )

    matrix_path = os.path.join(
        outdir,
        f"detector_last_scatter_bin_matrix_{wl_nm}nm.csv"
    )
    summary_path = os.path.join(
        outdir,
        f"detector_last_scatter_bin_summary_{wl_nm}nm.csv"
    )
    matrix_df.to_csv(matrix_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"✅ Saved {matrix_path}")
    print(f"✅ Saved {summary_path}")

    print("=========== CLARITAS_59 DETECTOR LAST-SCATTER BIN CONTRIBUTION ===========")
    print(f"Assigned detector hits with a last scatter bin: {total_assigned_hits}")
    if not summary_df.empty:
        top = summary_df.head(8)
        for _, r in top.iterrows():
            print(
                f"bin {int(r['bin_index'])}, {r['kind']}, "
                f"D={r['effective_diameter_um']:.3f} um, "
                f"mu_t={r['mu_t_fraction_percent']:.3f}%, "
                f"rear_last={r['fraction_of_rear_assigned_detector_hits']*100.0:.3f}%, "
                f"all_last={r['fraction_of_all_assigned_detector_hits']*100.0:.3f}%"
            )
    print("==========================================================================")

    return matrix_df, summary_df

from tqdm import tqdm
import h5py
import hashlib

bulk_profiles = []


detector_centers_rad = np.deg2rad(detector_angles)
detector_accept = detector_acceptance_deg
detector_hit_counts = {}

MULTIPLY_SCATTERED_MIN_COUNT = 6

def circular_angle_difference_deg(angle_deg, centre_deg):
    return ((angle_deg - centre_deg + 180.0) % 360.0) - 180.0


def assign_detector_indices_from_angles(
    ray_angle_deg,
    detector_angles_deg,
    detector_acceptance_deg,
    valid_mask=None
):
    ray_angle_deg = np.asarray(ray_angle_deg, dtype=np.float64)
    detector_angles_deg = np.asarray(detector_angles_deg, dtype=np.float64)
    if valid_mask is None:
        valid_mask = np.isfinite(ray_angle_deg)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(ray_angle_deg)

    assigned = np.full(ray_angle_deg.shape, -1, dtype=np.int32)
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return assigned

    differences = np.abs(
        ray_angle_deg[valid_indices, None] - detector_angles_deg[None, :]
    )
    nearest = np.argmin(differences, axis=1)
    accepted = (
        differences[np.arange(valid_indices.size), nearest]
        <= detector_acceptance_deg
    )
    assigned[valid_indices[accepted]] = nearest[accepted]
    return assigned


def detector_response_from_angles(
    ray_angle_deg,
    detector_angles_deg,
    detector_acceptance_deg,
    valid_mask=None
):
    if valid_mask is None:
        valid_mask = np.ones_like(ray_angle_deg, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    assigned = assign_detector_indices_from_angles(
        ray_angle_deg,
        detector_angles_deg,
        detector_acceptance_deg,
        valid_mask
    )
    counts = np.bincount(
        assigned[assigned >= 0],
        minlength=len(detector_angles_deg)
    ).astype(np.float64)
    total = np.sum(counts)
    response = counts / total if total > 0.0 else np.zeros_like(counts)
    return counts, response



def save_floc_bin_detector_proxy_diagnostics(
    wl_nm,
    angle_cdf_table_np,
    theta_rad_values,
    detector_angles_deg,
    detector_acceptance_deg,
    outdir
):
    theta_deg_values = np.rad2deg(theta_rad_values)
    n_bins = len(particle_diameter_m)

    cdf = np.asarray(angle_cdf_table_np, dtype=np.float64)
    pdf = np.diff(
        np.concatenate([
            np.zeros((cdf.shape[0], 1), dtype=np.float64),
            cdf
        ], axis=1),
        axis=1
    )
    pdf[~np.isfinite(pdf)] = 0.0
    pdf = np.maximum(pdf, 0.0)

    pdf_sums = np.sum(pdf, axis=1)
    valid_pdf = pdf_sums > 0.0
    pdf[valid_pdf] = pdf[valid_pdf] / pdf_sums[valid_pdf, None]

    total_mu_t = float(np.sum(mu_t_by_bin)) if np.sum(mu_t_by_bin) > 0.0 else 1.0
    total_mu_s = float(np.sum(mu_s_by_bin)) if np.sum(mu_s_by_bin) > 0.0 else 1.0
    total_mass_fraction = float(np.sum(particle_weights)) if np.sum(particle_weights) > 0.0 else 1.0

    summary_rows = []
    matrix_rows = []

    for bin_idx in range(n_bins):
        if not bool(particle_is_floc[bin_idx]):
            continue

        detector_probabilities = []
        detector_proxy_strengths = []

        for centre in detector_angles_deg:
            delta = circular_angle_difference_deg(theta_deg_values, centre)
            det_mask = np.abs(delta) <= detector_acceptance_deg
            p_det = float(np.sum(pdf[bin_idx, det_mask]))
            proxy_strength = float(mu_s_by_bin[bin_idx] * p_det)

            detector_probabilities.append(p_det)
            detector_proxy_strengths.append(proxy_strength)

            matrix_rows.append({
                "effective_bin_index": int(bin_idx),
                "effective_diameter_um": float(particle_diameter_m[bin_idx] * 1.0e6),
                "detector_angle_deg": float(centre),
                "phase_probability_in_detector": p_det,
                "mu_s_weighted_proxy": proxy_strength,
                "mu_t_weighted_proxy": float(mu_t_by_bin[bin_idx] * p_det),
                "mass_fraction_weighted_proxy": float(particle_weights[bin_idx] * p_det),
            })

        row = {
            "effective_bin_index": int(bin_idx),
            "effective_diameter_um": float(particle_diameter_m[bin_idx] * 1.0e6),
            "source_primary_min_um": float(source_primary_min_diameter_m[bin_idx] * 1.0e6),
            "source_primary_max_um": float(source_primary_max_diameter_m[bin_idx] * 1.0e6),
            "floc_band_index": int(floc_band_index_by_effective_bin[bin_idx]),
            "mass_fraction": float(particle_weights[bin_idx]),
            "mass_fraction_percent_of_total": float(100.0 * particle_weights[bin_idx] / total_mass_fraction),
            "number_density_per_m3": float(particle_number_density_by_bin[bin_idx]),
            "sigma_s_m2": float(sigma_s[bin_idx]),
            "sigma_a_m2": float(sigma_a[bin_idx]),
            "sigma_t_m2": float(sigma_t[bin_idx]),
            "mu_s_per_m": float(mu_s_by_bin[bin_idx]),
            "mu_a_per_m": float(mu_a_by_bin[bin_idx]),
            "mu_t_per_m": float(mu_t_by_bin[bin_idx]),
            "mu_s_fraction_percent_of_total": float(100.0 * mu_s_by_bin[bin_idx] / total_mu_s),
            "mu_t_fraction_percent_of_total": float(100.0 * mu_t_by_bin[bin_idx] / total_mu_t),
            "single_scattering_albedo": float(single_scattering_albedo_by_bin[bin_idx]),
            "phase_probability_sum_all_detectors": float(np.sum(detector_probabilities)),
            "mu_s_weighted_detector_proxy_sum": float(np.sum(detector_proxy_strengths)),
        }

        for centre, p_det, strength in zip(
            detector_angles_deg,
            detector_probabilities,
            detector_proxy_strengths
        ):
            row[f"phase_probability_det_{int(centre)}deg"] = p_det
            row[f"mu_s_proxy_det_{int(centre)}deg"] = strength

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    matrix_df = pd.DataFrame(matrix_rows)

    if not summary_df.empty:
        total_proxy = summary_df["mu_s_weighted_detector_proxy_sum"].sum()
        if total_proxy > 0.0:
            summary_df["detector_proxy_percent_of_floc_total"] = (
                100.0 * summary_df["mu_s_weighted_detector_proxy_sum"] / total_proxy
            )
        else:
            summary_df["detector_proxy_percent_of_floc_total"] = 0.0

    if not matrix_df.empty:
        total_matrix_proxy = matrix_df["mu_s_weighted_proxy"].sum()
        if total_matrix_proxy > 0.0:
            matrix_df["percent_of_all_floc_detector_proxy"] = (
                100.0 * matrix_df["mu_s_weighted_proxy"] / total_matrix_proxy
            )
        else:
            matrix_df["percent_of_all_floc_detector_proxy"] = 0.0

    conc_label = f"{mass_concentration_g_per_L:g}"
    summary_path = os.path.join(
        outdir,
        f"floc_bin_optical_detector_proxy_conc_{conc_label}_{wl_nm}nm.csv"
    )
    matrix_path = os.path.join(
        outdir,
        f"floc_detector_proxy_matrix_conc_{conc_label}_{wl_nm}nm.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    matrix_df.to_csv(matrix_path, index=False)

    print(f"✅ Saved {summary_path}")
    print(f"✅ Saved {matrix_path}")

    if not summary_df.empty:
        dominant = summary_df.sort_values(
            "detector_proxy_percent_of_floc_total",
            ascending=False
        ).head(5)
        print("=========== CLARITAS_53 FLOC BIN DETECTOR PROXY ===========")
        for _, r in dominant.iterrows():
            print(
                f"bin {int(r['effective_bin_index'])}, "
                f"D={r['effective_diameter_um']:.1f} um, "
                f"mass={r['mass_fraction_percent_of_total']:.3f}%, "
                f"mu_t={r['mu_t_fraction_percent_of_total']:.3f}%, "
                f"proxy={r['detector_proxy_percent_of_floc_total']:.3f}%"
            )
        print("============================================================")

    return summary_df, matrix_df


def save_optical_budget_diagnostics(wl_nm, floc_proxy_summary_df, outdir):
    total_mass = float(np.sum(particle_weights)) if np.sum(particle_weights) > 0.0 else 1.0
    total_mu_s = float(np.sum(mu_s_by_bin)) if np.sum(mu_s_by_bin) > 0.0 else 1.0
    total_mu_a = float(np.sum(mu_a_by_bin)) if np.sum(mu_a_by_bin) > 0.0 else 1.0
    total_mu_t = float(np.sum(mu_t_by_bin)) if np.sum(mu_t_by_bin) > 0.0 else 1.0
    total_event = float(np.sum(particle_event_weights)) if np.sum(particle_event_weights) > 0.0 else 1.0

    primary_mask = ~particle_is_floc
    floc_mask = particle_is_floc

    def _budget_row(label, mask):
        mass = float(np.sum(particle_weights[mask])) if np.any(mask) else 0.0
        mus = float(np.sum(mu_s_by_bin[mask])) if np.any(mask) else 0.0
        mua = float(np.sum(mu_a_by_bin[mask])) if np.any(mask) else 0.0
        mut = float(np.sum(mu_t_by_bin[mask])) if np.any(mask) else 0.0
        events = float(np.sum(particle_event_weights[mask])) if np.any(mask) else 0.0
        count = int(np.sum(mask))
        return {
            "population": label,
            "effective_bin_count": count,
            "mass_fraction": mass,
            "mass_percent_of_total": 100.0 * mass / total_mass,
            "mu_s_per_m": mus,
            "mu_s_percent_of_total": 100.0 * mus / total_mu_s,
            "mu_a_per_m": mua,
            "mu_a_percent_of_total": 100.0 * mua / total_mu_a,
            "mu_t_per_m": mut,
            "mu_t_percent_of_total": 100.0 * mut / total_mu_t,
            "event_probability": events,
            "event_probability_percent": 100.0 * events / total_event,
        }

    summary_df = pd.DataFrame([
        _budget_row("primary", primary_mask),
        _budget_row("floc", floc_mask),
    ])

    proxy_by_bin = {}
    if isinstance(floc_proxy_summary_df, pd.DataFrame) and not floc_proxy_summary_df.empty:
        for _, r in floc_proxy_summary_df.iterrows():
            proxy_by_bin[int(r["effective_bin_index"])] = {
                "mu_s_weighted_detector_proxy_sum": float(r.get("mu_s_weighted_detector_proxy_sum", np.nan)),
                "detector_proxy_percent_of_floc_total": float(r.get("detector_proxy_percent_of_floc_total", np.nan)),
                "phase_probability_sum_all_detectors": float(r.get("phase_probability_sum_all_detectors", np.nan)),
            }

    floc_rows = []
    for bin_idx in range(len(particle_diameter_m)):
        if not bool(particle_is_floc[bin_idx]):
            continue

        proxy = proxy_by_bin.get(int(bin_idx), {})
        floc_rows.append({
            "effective_bin_index": int(bin_idx),
            "effective_diameter_um": float(particle_diameter_m[bin_idx] * 1.0e6),
            "source_primary_min_um": float(source_primary_min_diameter_m[bin_idx] * 1.0e6),
            "source_primary_max_um": float(source_primary_max_diameter_m[bin_idx] * 1.0e6),
            "source_primary_geometric_mid_um": float(source_primary_geometric_mid_diameter_m[bin_idx] * 1.0e6),
            "floc_band_index": int(floc_band_index_by_effective_bin[bin_idx]),
            "mass_fraction": float(particle_weights[bin_idx]),
            "mass_percent_of_total": float(100.0 * particle_weights[bin_idx] / total_mass),
            "number_density_per_m3": float(particle_number_density_by_bin[bin_idx]),
            "sigma_s_m2": float(sigma_s[bin_idx]),
            "sigma_a_m2": float(sigma_a[bin_idx]),
            "sigma_t_m2": float(sigma_t[bin_idx]),
            "mu_s_per_m": float(mu_s_by_bin[bin_idx]),
            "mu_s_percent_of_total": float(100.0 * mu_s_by_bin[bin_idx] / total_mu_s),
            "mu_s_percent_of_floc": 0.0,
            "mu_a_per_m": float(mu_a_by_bin[bin_idx]),
            "mu_a_percent_of_total": float(100.0 * mu_a_by_bin[bin_idx] / total_mu_a),
            "mu_a_percent_of_floc": 0.0,
            "mu_t_per_m": float(mu_t_by_bin[bin_idx]),
            "mu_t_percent_of_total": float(100.0 * mu_t_by_bin[bin_idx] / total_mu_t),
            "mu_t_percent_of_floc": 0.0,
            "event_probability": float(particle_event_weights[bin_idx]),
            "event_probability_percent_of_total": float(100.0 * particle_event_weights[bin_idx] / total_event),
            "single_scattering_albedo": float(single_scattering_albedo_by_bin[bin_idx]),
            "mu_s_weighted_detector_proxy_sum": proxy.get("mu_s_weighted_detector_proxy_sum", np.nan),
            "detector_proxy_percent_of_floc_total": proxy.get("detector_proxy_percent_of_floc_total", np.nan),
            "phase_probability_sum_all_detectors": proxy.get("phase_probability_sum_all_detectors", np.nan),
        })

    floc_df = pd.DataFrame(floc_rows)
    if not floc_df.empty:
        floc_mu_s_total = float(np.sum(mu_s_by_bin[floc_mask])) if np.sum(mu_s_by_bin[floc_mask]) > 0.0 else 1.0
        floc_mu_a_total = float(np.sum(mu_a_by_bin[floc_mask])) if np.sum(mu_a_by_bin[floc_mask]) > 0.0 else 1.0
        floc_mu_t_total = float(np.sum(mu_t_by_bin[floc_mask])) if np.sum(mu_t_by_bin[floc_mask]) > 0.0 else 1.0

        floc_df["mu_s_percent_of_floc"] = 100.0 * floc_df["mu_s_per_m"] / floc_mu_s_total
        floc_df["mu_a_percent_of_floc"] = 100.0 * floc_df["mu_a_per_m"] / floc_mu_a_total
        floc_df["mu_t_percent_of_floc"] = 100.0 * floc_df["mu_t_per_m"] / floc_mu_t_total
        floc_df = floc_df.sort_values("mu_t_per_m", ascending=False).reset_index(drop=True)
        floc_df["cumulative_mu_t_percent_of_floc"] = np.cumsum(floc_df["mu_t_percent_of_floc"])
        floc_df["cumulative_mu_t_percent_of_total"] = np.cumsum(floc_df["mu_t_percent_of_total"])

    conc_label = f"{mass_concentration_g_per_L:g}"
    summary_path = os.path.join(outdir, f"optical_budget_summary_conc_{conc_label}_{wl_nm}nm.csv")
    floc_path = os.path.join(outdir, f"floc_optical_budget_conc_{conc_label}_{wl_nm}nm.csv")

    summary_df.to_csv(summary_path, index=False)
    floc_df.to_csv(floc_path, index=False)

    print(f"✅ Saved {summary_path}")
    print(f"✅ Saved {floc_path}")

    print("=========== CLARITAS_58 OPTICAL BUDGET ===========")
    for _, r in summary_df.iterrows():
        print(
            f"{r['population']}: mass={r['mass_percent_of_total']:.3f}%, "
            f"mu_s={r['mu_s_percent_of_total']:.3f}%, "
            f"mu_a={r['mu_a_percent_of_total']:.3f}%, "
            f"mu_t={r['mu_t_percent_of_total']:.3f}%, "
            f"events={r['event_probability_percent']:.3f}%"
        )

    if not floc_df.empty:
        print("Top floc bins by mu_t contribution:")
        for _, r in floc_df.head(5).iterrows():
            print(
                f"bin {int(r['effective_bin_index'])}, "
                f"D={r['effective_diameter_um']:.1f} um, "
                f"src={r['source_primary_geometric_mid_um']:.2f} um, "
                f"mass={r['mass_percent_of_total']:.3f}%, "
                f"mu_t_total={r['mu_t_percent_of_total']:.3f}%, "
                f"mu_t_floc={r['mu_t_percent_of_floc']:.3f}%, "
                f"event={r['event_probability_percent_of_total']:.3f}%"
            )

        for n_top in [1, 3, 5, 10, 20]:
            n = min(n_top, len(floc_df))
            if n <= 0:
                continue
            cum = float(floc_df.loc[n - 1, "cumulative_mu_t_percent_of_floc"])
            print(f"Top {n} floc bins cumulative floc mu_t: {cum:.3f}%")

    print("===================================================")
    return summary_df, floc_df


def save_detector_geometry_diagnostics(
    wl_nm,
    exit_x,
    exit_y,
    exit_z,
    exit_dirs,
    scatter_count,
    detector_angles_deg,
    detector_acceptance_deg,
    outdir,
    multiply_scattered_min_count=6
):
    exit_x = np.asarray(exit_x)
    exit_y = np.asarray(exit_y)
    exit_z = np.asarray(exit_z)
    exit_dirs = np.asarray(exit_dirs)
    scatter_count = np.asarray(scatter_count)

    valid_exit_mask = (
        np.isfinite(exit_x) &
        np.isfinite(exit_y) &
        np.isfinite(exit_z) &
        np.isfinite(exit_dirs) &
        (scatter_count >= 0)
    )

    exit_position_angle_deg = spherical_polar_angle_deg(
        exit_x, exit_y, exit_z
    )
    exit_direction_angle_deg = np.rad2deg(exit_dirs)

    ballistic_mask = valid_exit_mask & (scatter_count == 0)
    quasi_ballistic_mask = valid_exit_mask & (
        (scatter_count > 0) &
        (scatter_count < multiply_scattered_min_count)
    )
    multiply_scattered_mask = valid_exit_mask & (
        scatter_count >= multiply_scattered_min_count
    )

    pos_counts, pos_response = detector_response_from_angles(
        exit_position_angle_deg,
        detector_angles_deg,
        detector_acceptance_deg,
        valid_exit_mask
    )

    dir_counts, dir_response = detector_response_from_angles(
        exit_direction_angle_deg,
        detector_angles_deg,
        detector_acceptance_deg,
        valid_exit_mask
    )

    ms_pos_counts, ms_pos_response = detector_response_from_angles(
        exit_position_angle_deg,
        detector_angles_deg,
        detector_acceptance_deg,
        multiply_scattered_mask
    )

    ballistic_pos_counts, ballistic_pos_response = detector_response_from_angles(
        exit_position_angle_deg,
        detector_angles_deg,
        detector_acceptance_deg,
        ballistic_mask
    )

    quasi_pos_counts, quasi_pos_response = detector_response_from_angles(
        exit_position_angle_deg,
        detector_angles_deg,
        detector_acceptance_deg,
        quasi_ballistic_mask
    )

    diagnostics_df = pd.DataFrame({
        "detector_angle_deg": detector_angles_deg,
        "position_angle_counts_all": pos_counts,
        "position_angle_response_all": pos_response,
        "exit_direction_counts_all": dir_counts,
        "exit_direction_response_all": dir_response,
        "position_angle_counts_multiply_scattered": ms_pos_counts,
        "position_angle_response_multiply_scattered": ms_pos_response,
        "position_angle_counts_ballistic": ballistic_pos_counts,
        "position_angle_response_ballistic": ballistic_pos_response,
        "position_angle_counts_quasi_ballistic": quasi_pos_counts,
        "position_angle_response_quasi_ballistic": quasi_pos_response,
    })

    csv_path = os.path.join(outdir, f"detector_geometry_diagnostics_{wl_nm}nm.csv")
    diagnostics_df.to_csv(csv_path, index=False)

    png_path = os.path.join(outdir, f"detector_geometry_diagnostics_{wl_nm}nm.png")
    plt.figure(figsize=(10, 6))
    plt.plot(detector_angles_deg, pos_response, marker="o", label="Position angle, all rays")
    plt.plot(detector_angles_deg, dir_response, marker="s", label="Exit direction, all rays")
    plt.plot(
        detector_angles_deg,
        ms_pos_response,
        marker="^",
        label=f"Position angle, scatter_count >= {multiply_scattered_min_count}"
    )
    plt.xlabel("Detector angle (deg)")
    plt.ylabel("Normalised detector response")
    plt.title(f"Detector geometry diagnostics, {wl_nm} nm")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    valid_count = np.sum(valid_exit_mask)
    scatter_summary_df = pd.DataFrame({
        "category": [
            "valid_exit",
            "ballistic_scatter_count_eq_0",
            "quasi_ballistic_1_to_min_minus_1",
            "multiply_scattered_ge_min"
        ],
        "count": [
            int(valid_count),
            int(np.sum(ballistic_mask)),
            int(np.sum(quasi_ballistic_mask)),
            int(np.sum(multiply_scattered_mask))
        ],
        "fraction_of_valid_exits": [
            1.0 if valid_count > 0 else 0.0,
            float(np.sum(ballistic_mask) / valid_count) if valid_count > 0 else 0.0,
            float(np.sum(quasi_ballistic_mask) / valid_count) if valid_count > 0 else 0.0,
            float(np.sum(multiply_scattered_mask) / valid_count) if valid_count > 0 else 0.0,
        ],
        "multiply_scattered_min_count": [
            multiply_scattered_min_count,
            multiply_scattered_min_count,
            multiply_scattered_min_count,
            multiply_scattered_min_count,
        ]
    })

    summary_csv_path = os.path.join(outdir, f"detector_geometry_scatter_summary_{wl_nm}nm.csv")
    scatter_summary_df.to_csv(summary_csv_path, index=False)

    if np.any(valid_exit_mask):
        max_scatter = int(np.max(scatter_count[valid_exit_mask]))
        bins = np.arange(0, max_scatter + 2)
        scatter_values, scatter_bins = np.histogram(scatter_count[valid_exit_mask], bins=bins)
    else:
        scatter_values = np.array([0])
        scatter_bins = np.array([0, 1])

    scatter_hist_df = pd.DataFrame({
        "scatter_count": scatter_bins[:-1],
        "ray_count": scatter_values
    })

    scatter_hist_csv_path = os.path.join(outdir, f"scatter_count_histogram_{wl_nm}nm.csv")
    scatter_hist_df.to_csv(scatter_hist_csv_path, index=False)

    scatter_hist_png_path = os.path.join(outdir, f"scatter_count_histogram_{wl_nm}nm.png")
    plt.figure(figsize=(10, 5))
    plt.bar(scatter_hist_df["scatter_count"], scatter_hist_df["ray_count"])
    plt.yscale("log")
    plt.xlabel("Scatter count")
    plt.ylabel("Ray count, log scale")
    plt.title(f"Scatter-count histogram, {wl_nm} nm")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(scatter_hist_png_path, dpi=200)
    plt.close()

    raw_exit_csv_path = os.path.join(outdir, f"exit_position_direction_scatter_{wl_nm}nm.csv")
    pd.DataFrame({
        "exit_x_m": exit_x,
        "exit_y_m": exit_y,
        "exit_position_angle_deg": exit_position_angle_deg,
        "exit_direction_angle_deg": exit_direction_angle_deg,
        "scatter_count": scatter_count,
        "is_ballistic": ballistic_mask,
        "is_quasi_ballistic": quasi_ballistic_mask,
        "is_multiply_scattered": multiply_scattered_mask,
    }).to_csv(raw_exit_csv_path, index=False)

    print(f"✅ Saved {csv_path}")
    print(f"✅ Saved {png_path}")
    print(f"✅ Saved {summary_csv_path}")
    print(f"✅ Saved {scatter_hist_csv_path}")
    print(f"✅ Saved {scatter_hist_png_path}")
    print(f"✅ Saved {raw_exit_csv_path}")

    return diagnostics_df, scatter_summary_df

def sample_laser_angles(N, half_angle_deg=2.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(SIMULATION_SEED)
    angles = rng.uniform(
        -np.deg2rad(half_angle_deg),
        np.deg2rad(half_angle_deg),
        N
    )
    return angles.astype(np.float64)

host_rng = np.random.default_rng(SIMULATION_SEED)
configured_material_name = (
    "loess" if np.array_equal(primary_particle_diameter_m, loess_diameter)
    else "kaolin" if np.array_equal(primary_particle_diameter_m, kaolin_diameter)
    else "unknown"
)

for wl_idx, wl in enumerate(wavelengths):
    print(f"--- Wavelength {int(wl*1e9)} nm ---")
    if SOURCE_MODE == "production_beta":
        angles_init = sample_beta_angles(N_RAYS, alpha1, alpha2, host_rng)
    else:
        angles_init = np.zeros(N_RAYS, dtype=np.float64)
    #angles_init = sample_laser_angles(N_RAYS, half_angle_deg=2.0)
    particle_cdf_table_np = np.array(particle_event_cdf, dtype=np.float64)
    particle_diameter_table_np = np.array(particle_diameter_m, dtype=np.float64)
    particle_is_floc_table_np = np.array(particle_is_floc, dtype=np.float64)
    single_scattering_albedo_table_np = np.array(
        single_scattering_albedo_by_bin,
        dtype=np.float64
    )
    angle_cdf_table_np = np.array(cdf_profiles[wl_idx], dtype=np.float64)
    internal_angle_cdf_table_np = np.array(internal_floc_angle_cdf_profiles[wl_idx], dtype=np.float64)
    floc_internal_tau_s_table_np = np.array(floc_internal_tau_s_profiles[wl_idx], dtype=np.float64)
    theta_table_np = np.array(theta_rad, dtype=np.float64)

    # CLARITAS_53: pre-transport floc-bin optical/detector proxy diagnostics.
    floc_proxy_summary_df, floc_proxy_matrix_df = save_floc_bin_detector_proxy_diagnostics(
        int(wl * 1e9),
        angle_cdf_table_np,
        theta_table_np,
        detector_angles,
        detector_acceptance_deg,
        OUTDIR
    )

    save_optical_budget_diagnostics(
        int(wl * 1e9),
        floc_proxy_summary_df,
        OUTDIR
    )

    hdf5_file = os.path.join(OUTDIR, f"ray_exits_{int(wl*1e9)}nm.h5")

    t0 = time.time()

    print("DEBUG floc_event_probability:", floc_event_probability)
    print("DEBUG particle_is_floc count:", np.sum(particle_is_floc))

    heatmap = trace_rays_gpu(angles_init, 
                             PRIMARY_ROUGHNESS_STD_RAD,
                             FLOC_ROUGHNESS_STD_RAD,
                             R_REAL, RAY_OFFSET, VIS_SIZE, VISUAL_SCALE,
                             particle_cdf_table_np,
                             particle_is_floc_table_np, single_scattering_albedo_table_np, angle_cdf_table_np,
                             internal_angle_cdf_table_np, floc_internal_tau_s_table_np, particle_diameter_m,
                             theta_table_np,
                             wavelength_m=wl,
                             material_name=configured_material_name,
                             concentration_g_per_L=mass_concentration_g_per_L,
                             hdf5_file=hdf5_file)
    t1 = time.time()
    print(f"[INFO] trace_rays_gpu completed in {t1-t0:.2f} s")

    # Load exit data from HDF5 for plotting and detector binning
    with h5py.File(hdf5_file, "r") as f:
        exit_x = f["exit_x"][:]
        exit_y = f["exit_y"][:]
        exit_z = f["exit_z"][:]
        exit_vx = f["exit_vx"][:]
        exit_vy = f["exit_vy"][:]
        exit_vz = f["exit_vz"][:]
        exit_dirs = f["exit_dir"][:]
        exit_rpl = f["exit_rpl"][:]
        scatter_count = f["scatter_count"][:]
        floc_event_count = f["floc_event_count"][:] if "floc_event_count" in f else np.full_like(scatter_count, -1)
        floc_extinction_count = f["floc_extinction_count"][:] if "floc_extinction_count" in f else np.full_like(scatter_count, -1)
        last_event_was_floc = f["last_event_was_floc"][:] if "last_event_was_floc" in f else np.full_like(scatter_count, -1)
        last_scatter_bin = f["last_scatter_bin"][:] if "last_scatter_bin" in f else np.full_like(scatter_count, -1)
        extinction_count = f["extinction_count"][:] if "extinction_count" in f else np.full_like(scatter_count, -1)
        absorbed_flag = f["absorbed"][:] if "absorbed" in f else np.full_like(scatter_count, -1)
        truncated_flag = f["truncated"][:] if "truncated" in f else np.full_like(scatter_count, -1)
        terminal_state = f["terminal_state"][:] if "terminal_state" in f else np.zeros_like(scatter_count)
        floc_domain_dx = f["floc_domain_dx"][:] if "floc_domain_dx" in f else np.zeros_like(exit_x)
        floc_domain_dy = f["floc_domain_dy"][:] if "floc_domain_dy" in f else np.zeros_like(exit_y)
        floc_domain_dz = f["floc_domain_dz"][:] if "floc_domain_dz" in f else np.zeros_like(exit_z)
        floc_domain_path = f["floc_domain_path"][:] if "floc_domain_path" in f else np.zeros_like(exit_rpl)
        floc_internal_scatter_count = f["floc_internal_scatter_count"][:] if "floc_internal_scatter_count" in f else np.zeros_like(scatter_count)

    valid_exit_mask = (
        np.isfinite(exit_x) &
        np.isfinite(exit_y) &
        np.isfinite(exit_z) &
        np.isfinite(exit_vx) &
        np.isfinite(exit_vy) &
        np.isfinite(exit_vz) &
        np.isfinite(exit_dirs) &
        (scatter_count >= 0)
    )

    n_valid_exits = int(np.sum(valid_exit_mask))
    n_absorbed = int(np.sum(absorbed_flag == 1))
    n_truncated = int(np.sum(truncated_flag == 1))
    n_invalid = int(
        len(scatter_count) - n_valid_exits - n_absorbed - n_truncated
    )

    if n_valid_exits > 0:
        floc_displacement_m = np.sqrt(
            floc_domain_dx**2 + floc_domain_dy**2 + floc_domain_dz**2
        )
        floc_touched_mask = valid_exit_mask & (floc_event_count > 0)
        if np.any(floc_touched_mask):
            floc_exit_summary_df = pd.DataFrame([{
                "wavelength_nm": int(wavelengths[0] * 1.0e9),
                "rays_with_floc_events": int(np.sum(floc_touched_mask)),
                "mean_floc_exit_displacement_um": float(np.mean(floc_displacement_m[floc_touched_mask]) * 1.0e6),
                "median_floc_exit_displacement_um": float(np.median(floc_displacement_m[floc_touched_mask]) * 1.0e6),
                "p95_floc_exit_displacement_um": float(np.percentile(floc_displacement_m[floc_touched_mask], 95.0) * 1.0e6),
                "mean_floc_domain_path_um": float(np.mean(floc_domain_path[floc_touched_mask]) * 1.0e6),
                "mean_floc_internal_scatter_count": float(np.mean(floc_internal_scatter_count[floc_touched_mask])),
            }])
            floc_exit_summary_path = os.path.join(
                OUTDIR,
                f"floc_domain_exit_position_summary_{int(wl*1e9)}nm.csv"
            )
            floc_exit_summary_df.to_csv(floc_exit_summary_path, index=False)
            print(f"✅ Saved {floc_exit_summary_path}")
            print(
                "CLARITAS_72 floc exit displacement: "
                f"mean={np.mean(floc_displacement_m[floc_touched_mask])*1.0e6:.3f} um, "
                f"p95={np.percentile(floc_displacement_m[floc_touched_mask],95.0)*1.0e6:.3f} um"
            )
        print(f"Mean scatter count: {np.mean(scatter_count[valid_exit_mask]):.3e}")
        print(f"Max scatter count: {np.max(scatter_count[valid_exit_mask])}")
    else:
        print("Mean scatter count: nan (no valid exits)")
        print("Max scatter count: nan (no valid exits)")

    print(f"Valid exit rays: {n_valid_exits} / {len(scatter_count)}")
    print(f"Absorbed ray fraction: {n_absorbed / len(scatter_count):.6f}")
    print(f"Truncated ray fraction: {n_truncated / len(scatter_count):.6f}")
    print(f"Invalid launch fraction: {n_invalid / len(scatter_count):.6f}")

    detector_geometry_df, detector_geometry_summary_df = save_detector_geometry_diagnostics(
        int(wl * 1e9),
        exit_x,
        exit_y,
        exit_z,
        exit_dirs,
        scatter_count,
        detector_angles,
        detector_acceptance_deg,
        OUTDIR,
        multiply_scattered_min_count=MULTIPLY_SCATTERED_MIN_COUNT
    )

    detector_transport_breakdown_df, detector_transport_overview_df = save_detector_transport_breakdown(
        int(wl * 1e9),
        exit_x,
        exit_y,
        exit_z,
        exit_dirs,
        exit_rpl,
        scatter_count,
        floc_event_count,
        last_event_was_floc,
        extinction_count,
        absorbed_flag,
        detector_angles,
        detector_acceptance_deg,
        OUTDIR
    )

    detector_last_scatter_matrix_df, detector_last_scatter_summary_df = save_detector_last_scatter_bin_contribution(
        int(wl * 1e9),
        exit_x,
        exit_y,
        exit_z,
        scatter_count,
        last_scatter_bin,
        detector_angles,
        detector_acceptance_deg,
        OUTDIR
    )

    detector_scatter_class_detail_df, detector_scatter_class_matrix_df, detector_scatter_class_overview_df = save_detector_response_by_scatter_class(
        int(wl * 1e9),
        exit_x,
        exit_y,
        exit_z,
        scatter_count,
        floc_event_count,
        last_event_was_floc,
        detector_angles,
        detector_acceptance_deg,
        OUTDIR
    )

    masked = ma.masked_where(heatmap == 0, heatmap)

    if masked.count() > 0:
        vmin = masked.min()
        vmax = masked.max()
    else:
        vmin, vmax = 1e-12, R_REAL


    plt.figure(figsize=(6, 6))

    colors = [(0, 0, 0), (1, 1, 1)]
    cmap = LinearSegmentedColormap.from_list("black_white", colors, N=256)
    cmap.set_bad(color='black')

    extent_mm = [
        -R_REAL * 1000,  # xmin (mm)
        R_REAL * 1000,  # xmax (mm)
        -R_REAL * 1000,  # ymin (mm)
        R_REAL * 1000   # ymax (mm)
    ]

    plt.imshow(
        masked,
        cmap=cmap,
        aspect="equal",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=extent_mm,
        interpolation='gaussian'
    )

    circle = plt.Circle(
        (0, 0),                 # center (mm)
        R_REAL * 1000,          # radius (mm)
        color='red',
        linewidth=1.5,
        fill=False,
        linestyle='--'
    )
    plt.gca().add_patch(circle)

    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(f"{mass_concentration_g_per_L} g/L, {int(wl*1e9)} nm")

    start = -R_REAL * 1000
    end = R_REAL * 1000
    step = (end - start) / 5.0

    plt.xticks(np.arange(start, end + step, step))
    plt.yticks(np.arange(start, end + step, step))

    plt.colorbar(label="Counts")
    plt.tight_layout()

    heatmap_path = os.path.join(OUTDIR, f"conc_{mass_concentration_g_per_L}_{int(wl*1e9)}nm.png")
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    print(f"✅ Saved {heatmap_path}")

    exit_dirs_valid = exit_dirs[valid_exit_mask]

    if exit_dirs_valid.size > 0:
        exit_dirs_deg = np.rad2deg(exit_dirs_valid)
        hist_bulk, _ = np.histogram(exit_dirs_deg, bins=theta_deg, density=True)
    else:
        hist_bulk = np.zeros(len(theta_deg) - 1, dtype=np.float64)

    bulk_profiles.append(hist_bulk)

    exit_pos_angles = np.full_like(exit_x, np.nan, dtype=np.float64)
    exit_pos_angles[valid_exit_mask] = spherical_polar_angle_deg(
        exit_x[valid_exit_mask],
        exit_y[valid_exit_mask],
        exit_z[valid_exit_mask]
    )

    df_exits = pd.DataFrame({
        "exit_x_m": exit_x,
        "exit_y_m": exit_y,
        "exit_z_m": exit_z,
        "exit_vx": exit_vx,
        "exit_vy": exit_vy,
        "exit_vz": exit_vz,
        "exit_rpl_m": exit_rpl,
        "scatter_count": scatter_count,
        "floc_event_count": floc_event_count,
        "floc_extinction_count": floc_extinction_count,
        "last_event_was_floc": last_event_was_floc,
        "last_scatter_bin": last_scatter_bin,
        "extinction_count": extinction_count,
        "absorbed_flag": absorbed_flag,
        "truncated_flag": truncated_flag,
        "terminal_state": terminal_state,
        "is_valid_exit": valid_exit_mask,
        "is_absorbed": absorbed_flag == 1,
        "is_truncated": truncated_flag == 1,
        "is_ballistic": valid_exit_mask & (scatter_count == 0),
        "exit_pos_angle_deg": exit_pos_angles,
        "exit_dir_polar_deg": np.rad2deg(exit_dirs)
    })
    # Assign each valid exit to at most one nearest detector band.  This removes
    # overlap double-counting and applies no scatter-class-specific suppression.
    detector_index = np.full(len(exit_x), -1, dtype=np.int32)
    valid_angles = exit_pos_angles[valid_exit_mask]
    if valid_angles.size:
        differences = np.abs(
            valid_angles[:, None] - detector_angles[None, :]
        )
        nearest = np.argmin(differences, axis=1)
        accepted = (
            differences[np.arange(valid_angles.size), nearest]
            <= detector_accept
        )
        valid_indices = np.flatnonzero(valid_exit_mask)
        detector_index[valid_indices[accepted]] = nearest[accepted]

    counts = np.bincount(
        detector_index[detector_index >= 0],
        minlength=len(detector_angles)
    ).astype(int)
    df_exits["detector_index"] = detector_index
    exits_csv_path = os.path.join(
        OUTDIR, f"exit_points_{int(wl*1e9)}nm.csv"
    )
    df_exits.to_csv(exits_csv_path, index=False)
    print(f"✅ Saved {exits_csv_path}")

    comprehensive_summary = save_comprehensive_transport_diagnostics(
        wl_nm=int(wl * 1e9),
        outdir=OUTDIR,
        exit_x=exit_x,
        exit_y=exit_y,
        exit_z=exit_z,
        exit_vx=exit_vx,
        exit_vy=exit_vy,
        exit_vz=exit_vz,
        path_length=exit_rpl,
        scatter_count=scatter_count,
        floc_event_count=floc_event_count,
        floc_extinction_count=floc_extinction_count,
        extinction_count=extinction_count,
        absorbed=absorbed_flag,
        truncated=truncated_flag,
        detector_index=detector_index,
        detector_angles_deg=detector_angles,
        floc_internal_scatter_count=floc_internal_scatter_count
    )

    detector_hit_counts[int(wl*1e9)] = counts

# ---------------- Final outputs: detector CSV and plots ----------------
df_det = pd.DataFrame({"Detector_deg": detector_angles})
for wl_nm, counts in detector_hit_counts.items():
    df_det[f"H_{wl_nm}nm"] = counts
det_csv_path = os.path.join(OUTDIR, "detector_hits.csv")
df_det.to_csv(det_csv_path, index=False)
print(f"✅ Saved {det_csv_path}")

save_measured_comparison_if_available(
    outdir=OUTDIR,
    measured_csv="sediment_data.csv",
    material=configured_material_name,
    concentration_g_per_L=mass_concentration_g_per_L,
    wavelength_nm=int(wavelengths[0] * 1e9),
    detector_csv=det_csv_path,
    n_rays=N_RAYS,
    transport_summary=comprehensive_summary
)

plt.figure(figsize=(9,5))
for wl_nm, counts in detector_hit_counts.items():
    plt.plot(detector_angles, counts, '-o', label=f"{wl_nm} nm")
plt.xlabel("Detector angle (deg)")
plt.ylabel("Hit count")
plt.title("Detector Hit Counts vs Angle (Physical Boundary Position)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "detector_hits.png"), dpi=200)
plt.close()
print(f"✅ Saved detector_hits.png")

plt.figure(figsize=(6,6))
ax = plt.subplot(111, projection='polar')
for wl_nm, counts in detector_hit_counts.items():
    thetas = np.deg2rad(detector_angles)
    ax.plot(thetas, counts, '-o', label=f"{wl_nm} nm")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetalim(0, np.pi)
ax.set_title("Detector hits (polar)")
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "detector_hits_polar.png"), dpi=200)
plt.close()
print(f"✅ Saved detector_hits_polar.png")

df_bulk = pd.DataFrame({"Angle_deg": theta_deg[:-1]})
for wl_idx, wl in enumerate(wavelengths):
    df_bulk[f"I_bulk_{int(wl*1e9)}nm"] = bulk_profiles[wl_idx]
bulk_csv_path = os.path.join(OUTDIR, "bulk_angular_scattering_profiles.csv")
df_bulk.to_csv(bulk_csv_path, index=False)

plt.figure(figsize=(8,5))
for wl_idx, wl in enumerate(wavelengths):
    plt.plot(theta_deg[:-1], bulk_profiles[wl_idx], label=f"{int(wl*1e9)} nm")
plt.xlabel("Scattering angle (deg)")
plt.ylabel("Normalized intensity")
plt.title("Bulk Angular Scattering Profiles")
plt.legend(title="Wavelength")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "bulk_angular_scattering_profiles.png"), dpi=200)
plt.close()
print("✅ Simulation complete. Heatmaps + bulk angular scattering + detector hits + exit point plots saved.")

# Wavelength-dependent detector response (unchanged plotting semantics)
plt.figure(figsize=(9,5))
for wl_nm, counts in detector_hit_counts.items():
    counts_norm = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    plt.plot(detector_angles, counts_norm, '-', label=f"{wl_nm} nm")
plt.xlabel("Detector angle (deg)")
plt.ylabel("Normalized detector response")
plt.title("Wavelength-dependent Detector Response")
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 181, 10))
plt.legend()
plt.tight_layout()
wavelength_response_path = os.path.join(OUTDIR, "detector_response_vs_wavelength.png")
plt.savefig(wavelength_response_path, dpi=200)
plt.close()
print(f"✅ Saved {wavelength_response_path}")

plt.figure(figsize=(6,6))
ax = plt.subplot(111, projection='polar')
for wl_nm, counts in detector_hit_counts.items():
    counts_norm = counts / counts.max() if counts.max() > 0 else np.zeros_like(counts, dtype=float)
    thetas = np.deg2rad(detector_angles)
    ax.plot(thetas, counts_norm, '-o', label=f"{wl_nm} nm")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetalim(0, np.pi)
ax.set_title("Wavelength-dependent Detector Response (polar)")
ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.0))
plt.tight_layout()
polar_response_path = os.path.join(OUTDIR, "detector_response_vs_wavelength_polar.png")
plt.savefig(polar_response_path, dpi=200)
plt.close()
print(f"✅ Saved {polar_response_path}")


n_bins = 1000  # number of bins
hist_color = "skyblue"
hist_edge = "black"
hist_alpha = 0.7

exit_rpl_valid = exit_rpl[np.isfinite(exit_rpl)]

if len(exit_rpl_valid) > 0:
    hist_counts, bin_edges = np.histogram(exit_rpl_valid, bins=n_bins)
else:
    hist_counts = np.zeros(n_bins, dtype=np.int64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

hist_counts_safe = np.where(hist_counts == 0, 1, hist_counts)

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

plt.figure(figsize=(8,5))
plt.bar(bin_centers, hist_counts_safe, width=(bin_edges[1]-bin_edges[0]),
        color=hist_color, edgecolor=hist_edge, alpha=hist_alpha)
plt.xlim(0, 2.5)                     # set x-axis from 0 to 0.02 meters
plt.xticks(np.arange(0, 2.5, 0.25))  # set ticks every 0.002 meters
plt.yscale("log")  # log scale for counts
plt.xlabel("Exit Ray Path Length (m)")
plt.ylabel("Counts (log scale)")
plt.title("Histogram of Exit Ray Path Lengths (log scale)")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
hist_png_path = "./exit_rpl_histogram_log.png"
plt.savefig(hist_png_path, dpi=200)
plt.close()
print(f"✅ Saved log-scale histogram plot as {hist_png_path}")

df_hist = pd.DataFrame({
    "bin_center_m": bin_centers,
    "counts": hist_counts,          # raw counts
    "counts_for_log_plot": hist_counts_safe  # counts used for plotting
})
hist_csv_path = "./exit_rpl_histogram.csv"
df_hist.to_csv(hist_csv_path, index=False)
print(f"✅ Saved histogram data to {hist_csv_path}")


angles = detector_angles  # already defined
counts = detector_hit_counts[622]  # or loop over wavelengths if needed

angles = np.array(angles)
counts = np.array(counts)

dtheta = np.deg2rad(angles[1] - angles[0])  # radians

total = np.sum(counts * dtheta)

mask_back = angles >= 90
backscatter = np.sum(counts[mask_back] * dtheta)

backscatter_fraction = backscatter / total if total > 0 else 0.0

print("\n=========== DETECTOR-INTEGRATED BACKSCATTER ===========")
print(f"Backscatter fraction (integrated): {backscatter_fraction:.6f}")
print("======================================================")


optical_object_diagnostics_df = pd.DataFrame({
    "effective_bin_index": np.arange(len(particle_diameter_m)),
    "effective_bin_kind": effective_bin_kind,
    "is_floc": particle_is_floc,
    "diameter_um": particle_diameter_m * 1.0e6,
    "mass_fraction": particle_weights,
    "number_density_per_m3": particle_number_density_by_bin,
    "sigma_s_m2": sigma_s,
    "sigma_a_m2": sigma_a,
    "sigma_t_m2": sigma_t,
    "single_scattering_albedo": single_scattering_albedo_by_bin,
    "solid_volume_fraction": solid_volume_fraction_by_bin,
    "effective_refractive_index_real": particle_refractive_index_by_bin,
    "effective_refractive_index_imag_k": particle_refractive_index_imag_k_by_bin,
    "mu_s_by_bin_per_m": mu_s_by_bin,
    "mu_a_by_bin_per_m": mu_a_by_bin,
    "mu_t_by_bin_per_m": mu_t_by_bin,
    "extinction_event_strength_by_bin_per_m": extinction_event_strength_by_bin,
    "event_probability_extinction": particle_event_weights,
    "g_by_bin": g_by_bin,
    "floc_absorption_tau": floc_absorption_tau_by_bin,
    "floc_absorption_coefficient_per_m": floc_absorption_coefficient_by_bin_per_m,
    "floc_mean_chord_length_m": floc_mean_chord_length_by_bin_m,
    "source_primary_min_um": source_primary_min_diameter_m * 1.0e6,
    "source_primary_max_um": source_primary_max_diameter_m * 1.0e6,
    "floc_band_index": floc_band_index_by_effective_bin,
})

optical_object_diagnostics_path = os.path.join(
    OUTDIR,
    f"optical_object_diagnostics_conc_{mass_concentration_g_per_L}.csv"
)

optical_object_diagnostics_df.to_csv(
    optical_object_diagnostics_path,
    index=False
)

print(f"✅ Saved {optical_object_diagnostics_path}")


psd_compare_png = os.path.join(
    OUTDIR,
    "psd_original_vs_effective.png"
)

psd_compare_csv = os.path.join(
    OUTDIR,
    "psd_original_vs_effective.csv"
)

primary_um = primary_particle_diameter_m * 1.0e6
effective_um = particle_diameter_m * 1.0e6

df_primary_psd = pd.DataFrame({
    "psd_type": "original_primary",
    "diameter_m": primary_particle_diameter_m,
    "diameter_um": primary_um,
    "mass_fraction": primary_particle_weights,
    "particle_mass_by_bin_kg": primary_particle_masses_kg,
    "number_density_per_m3": primary_particle_number_density_by_bin,
    "scattering_cross_section_m2": np.nan,
    "mu_s_by_bin_per_m": np.nan,
    "event_probability": np.nan,
    "g_by_bin": np.nan,
    "is_floc": False,
    "effective_density_kg_per_m3": primary_particle_density_by_bin_kg_per_m3,
    "source_primary_min_diameter_um": primary_um,
    "source_primary_max_diameter_um": primary_um,
    "source_primary_mass_fraction": primary_particle_weights,
    "floc_band_index": primary_bin_floc_band_index,
    "effective_bin_kind": "original_primary",
    "primary_refractive_index_or_nan": n_particle
})

df_effective_psd = pd.DataFrame({
    "psd_type": "effective",
    "diameter_m": particle_diameter_m,
    "diameter_um": effective_um,
    "mass_fraction": particle_weights,
    "particle_mass_by_bin_kg": particle_mass_by_bin_kg,
    "number_density_per_m3": particle_number_density_by_bin,
    "scattering_cross_section_m2": sigma_s,
    "mu_s_by_bin_per_m": mu_s_by_bin,
    "event_probability": particle_event_weights,
    "g_by_bin": g_by_bin,
    "is_floc": particle_is_floc,
    "effective_density_kg_per_m3": particle_density_by_bin_kg_per_m3,
    "source_primary_min_diameter_um": source_primary_min_diameter_m * 1.0e6,
    "source_primary_max_diameter_um": source_primary_max_diameter_m * 1.0e6,
    "source_primary_mass_fraction": source_primary_mass_fraction,
    "floc_band_index": floc_band_index_by_effective_bin,
    "effective_bin_kind": effective_bin_kind,
    "primary_refractive_index_or_nan": particle_refractive_index_by_bin
})

if FLOC_ENABLED and len(FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M) > 0:
    floc_band_summary_df = pd.DataFrame({
        "floc_band_index": np.arange(len(FLOC_POOL_EFFECTIVE_DIAMETER_M)),
        "floc_effective_diameter_m": FLOC_POOL_EFFECTIVE_DIAMETER_M,
        "floc_effective_diameter_um": FLOC_POOL_EFFECTIVE_DIAMETER_M * 1.0e6,
        "reference_source_band_index": floc_reference_source_index_by_floc_bin,
        "reference_primary_band_max_diameter_m": (
            FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M[floc_reference_source_index_by_floc_bin]
        ),
        "reference_primary_band_max_diameter_um": (
            FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M[floc_reference_source_index_by_floc_bin] * 1.0e6
        ),
        "floc_mass_by_band_kg_reference_path": floc_mass_by_band_kg,
        "floc_effective_density_by_band_kg_per_m3_reference_path": floc_effective_density_by_band_kg_per_m3,
        "floc_fractal_dimension": np.full(
            len(FLOC_POOL_EFFECTIVE_DIAMETER_M),
            FLOC_FRACTAL_DIMENSION
        ),
        "floc_scatter_efficiency": np.full(
            len(FLOC_POOL_EFFECTIVE_DIAMETER_M),
            FLOC_SCATTER_EFFICIENCY
        ),
    })

    floc_band_summary_path = os.path.join(
        OUTDIR,
        "floc_band_summary.csv"
    )

    floc_band_summary_df.to_csv(
        floc_band_summary_path,
        index=False
    )

    print(f"✅ Saved {floc_band_summary_path}")

    source_idx_grid, floc_idx_grid = np.meshgrid(
        np.arange(len(FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M)),
        np.arange(len(FLOC_POOL_EFFECTIVE_DIAMETER_M)),
        indexing="ij"
    )

    floc_pooling_kernel_long_df = pd.DataFrame({
        "source_band_index": source_idx_grid.ravel(),
        "source_primary_band_max_diameter_m": (
            FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M[source_idx_grid.ravel()]
        ),
        "source_primary_band_max_diameter_um": (
            FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M[source_idx_grid.ravel()] * 1.0e6
        ),
        "floc_bin_index": floc_idx_grid.ravel(),
        "floc_effective_diameter_m": (
            FLOC_POOL_EFFECTIVE_DIAMETER_M[floc_idx_grid.ravel()]
        ),
        "floc_effective_diameter_um": (
            FLOC_POOL_EFFECTIVE_DIAMETER_M[floc_idx_grid.ravel()] * 1.0e6
        ),
        "kernel_probability": floc_pooling_kernel.ravel(),
        "floc_mass_kg": floc_mass_by_source_band_and_floc_bin_kg.ravel(),
        "floc_effective_density_kg_per_m3": (
            floc_effective_density_by_source_band_and_floc_bin_kg_per_m3.ravel()
        ),
    })

    floc_pooling_kernel_long_path = os.path.join(
        OUTDIR,
        "floc_pooling_kernel_long.csv"
    )

    floc_pooling_kernel_long_df.to_csv(
        floc_pooling_kernel_long_path,
        index=False
    )

    print(f"✅ Saved {floc_pooling_kernel_long_path}")

df_psd_compare = pd.concat(
    [df_primary_psd, df_effective_psd],
    ignore_index=True
)

df_psd_compare.to_csv(
    psd_compare_csv,
    index=False
)

print(f"✅ Saved {psd_compare_csv}")

plt.figure(figsize=(10, 6))

plt.bar(
    primary_um,
    primary_particle_weights,
    width=primary_um * 0.08,
    alpha=0.35,
    color="lightsteelblue",
    label=f"Original primary PSD ({particle_density_kg_per_m3:.0f} kg/m³)"
)

effective_bin_kind_array = np.asarray(effective_bin_kind)

unchanged_primary_mask = (
    effective_bin_kind_array == "unchanged_primary"
)

residual_primary_mask = (
    effective_bin_kind_array == "residual_primary"
)

pooled_floc_mask = np.asarray(particle_is_floc, dtype=bool)

plt.bar(
    effective_um[unchanged_primary_mask],
    particle_weights[unchanged_primary_mask],
    width=effective_um[unchanged_primary_mask] * 0.08,
    alpha=0.85,
    color="tab:blue",
    label="Effective PSD (unchanged coarse bins)"
)

plt.bar(
    effective_um[residual_primary_mask],
    particle_weights[residual_primary_mask],
    width=effective_um[residual_primary_mask] * 0.08,
    alpha=0.65,
    color="tab:green",
    label="Effective PSD (residual eligible primary bins)"
)

plt.bar(
    effective_um[pooled_floc_mask],
    particle_weights[pooled_floc_mask],
    width=effective_um[pooled_floc_mask] * 0.08,
    alpha=0.95,
    color="tab:red",
    edgecolor="black",
    linewidth=0.6,
    zorder=10,
    label="Effective PSD (pooled floc bins)"
)

plt.xscale("log")
plt.xlabel("Particle / floc diameter (um)")
plt.ylabel("PSD mass fraction")
plt.title("Original Primary PSD vs Fractal Pooled-Floc Effective PSD")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(psd_compare_png, dpi=200)
plt.close()

print(f"✅ Saved {psd_compare_png}")

floc_diag_csv = os.path.join(
    OUTDIR,
    "floc_model_diagnostics.csv"
)

floc_diag_png_1 = os.path.join(
    OUTDIR,
    "floc_primary_band_vs_effective_diameter.png"
)

floc_diag_png_2 = os.path.join(
    OUTDIR,
    "floc_density_vs_effective_diameter.png"
)

floc_diag_png_3 = os.path.join(
    OUTDIR,
    "floc_multiplier_vs_source_primary_mid_diameter.png"
)

primary_um = primary_particle_diameter_m * 1e6
effective_um = particle_diameter_m * 1e6
density_values = particle_density_by_bin_kg_per_m3
source_mid_um = source_primary_geometric_mid_diameter_m * 1e6

df_floc_diag = pd.DataFrame({
    "effective_diameter_um": effective_um,
    "source_primary_min_diameter_um": source_primary_min_diameter_m * 1e6,
    "source_primary_max_diameter_um": source_primary_max_diameter_m * 1e6,
    "source_primary_geometric_mid_diameter_um": source_mid_um,
    "source_primary_mass_fraction": source_primary_mass_fraction,
    "floc_diameter_multiplier": floc_diameter_multiplier_by_bin,
    "weight_fraction": particle_weights,
    "is_floc": particle_is_floc.astype(int),
    "floc_band_index": floc_band_index_by_effective_bin,
    "effective_bin_kind": effective_bin_kind,
    "density_kg_per_m3": density_values,
    "floc_effective_density_kg_per_m3": floc_effective_density_by_bin_kg_per_m3,
    "target_floc_density_kg_per_m3": target_floc_density_by_bin,
    "refractive_index": particle_refractive_index_by_bin,
    "number_density_per_m3": particle_number_density_by_bin,
    "sigma_s_m2": sigma_s,
    "sigma_a_m2": sigma_a,
    "sigma_t_m2": sigma_t,
    "single_scattering_albedo": single_scattering_albedo_by_bin,
    "solid_volume_fraction": solid_volume_fraction_by_bin,
    "effective_refractive_index_real": particle_refractive_index_by_bin,
    "effective_refractive_index_imag_k": particle_refractive_index_imag_k_by_bin,
    "mu_s_by_bin_per_m": mu_s_by_bin,
    "mu_a_by_bin_per_m": mu_a_by_bin,
    "mu_t_by_bin_per_m": mu_t_by_bin,
    "extinction_event_strength_by_bin_per_m": extinction_event_strength_by_bin,
    "particle_event_weight": particle_event_weights,
    "floc_mass_fraction_global": floc_mass_fraction,
    "eligible_primary_spacing_m": eligible_primary_spacing_m,
    "FLOC_COLLISION_LENGTH_M": FLOC_COLLISION_LENGTH_M
})

df_floc_diag.to_csv(
    floc_diag_csv,
    index=False
)

print(f"✅ Saved {floc_diag_csv}")

plt.figure(figsize=(8, 6))

plt.scatter(
    source_mid_um[~particle_is_floc],
    effective_um[~particle_is_floc],
    s=60,
    alpha=0.75,
    label="Primary/residual bins"
)

plt.scatter(
    source_mid_um[particle_is_floc],
    effective_um[particle_is_floc],
    s=80,
    alpha=0.85,
    label="Pooled floc bins"
)

plt.plot(
    [primary_um.min(), primary_um.max()],
    [primary_um.min(), primary_um.max()],
    linestyle="--",
    linewidth=1.5,
    label="1:1 unchanged"
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Source primary geometric-mid diameter (um)")
plt.ylabel("Effective particle/floc diameter (um)")
plt.title("Source Primary Band vs Effective Pooled-Floc Diameter")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(floc_diag_png_1, dpi=200)
plt.close()

print(f"✅ Saved {floc_diag_png_1}")

plt.figure(figsize=(8, 6))

plt.scatter(
    effective_um[~particle_is_floc],
    density_values[~particle_is_floc],
    s=60,
    alpha=0.75,
    label="Primary/residual bins"
)

plt.scatter(
    effective_um[particle_is_floc],
    density_values[particle_is_floc],
    s=80,
    alpha=0.85,
    label="Pooled floc bins"
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Effective diameter (um)")
plt.ylabel("Density used in model (kg/m³)")
plt.title("Effective Density vs Effective Diameter")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(floc_diag_png_2, dpi=200)
plt.close()

print(f"✅ Saved {floc_diag_png_2}")

plt.figure(figsize=(8, 6))

plt.scatter(
    source_mid_um[~particle_is_floc],
    floc_diameter_multiplier_by_bin[~particle_is_floc],
    s=60,
    alpha=0.75,
    label="Primary/residual bins"
)

plt.scatter(
    source_mid_um[particle_is_floc],
    floc_diameter_multiplier_by_bin[particle_is_floc],
    s=80,
    alpha=0.85,
    label="Pooled floc bins"
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Source primary geometric-mid diameter (um)")
plt.ylabel("Effective diameter / source-mid diameter")
plt.title("Pooled-Floc Diameter Multiplier vs Source Primary Band")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(floc_diag_png_3, dpi=200)
plt.close()

print(f"✅ Saved {floc_diag_png_3}")


ri_diag_csv = os.path.join(
    OUTDIR,
    "floc_effective_complex_index_diagnostics.csv"
)

ri_diag_png = os.path.join(
    OUTDIR,
    "floc_effective_complex_index_vs_effective_diameter.png"
)

df_ri_diag = pd.DataFrame({
    "effective_diameter_um": effective_um,
    "source_primary_min_diameter_um": source_primary_min_diameter_m * 1e6,
    "source_primary_max_diameter_um": source_primary_max_diameter_m * 1e6,
    "is_floc": particle_is_floc.astype(int),
    "density_kg_per_m3": particle_density_by_bin_kg_per_m3,
    "solid_volume_fraction": solid_volume_fraction_by_bin,
    "refractive_index_real": particle_refractive_index_by_bin,
    "refractive_index_imag_k": particle_refractive_index_imag_k_by_bin,
    "effective_complex_index_real": particle_refractive_index_by_bin,
    "effective_complex_index_imag_k": particle_refractive_index_imag_k_by_bin,
})

df_ri_diag.to_csv(
    ri_diag_csv,
    index=False
)

print(f"✅ Saved {ri_diag_csv}")

plt.figure(figsize=(8, 6))

plt.scatter(
    effective_um[~particle_is_floc],
    particle_refractive_index_by_bin[~particle_is_floc],
    s=60,
    alpha=0.75,
    label="Unchanged bins"
)

plt.scatter(
    effective_um[particle_is_floc],
    particle_refractive_index_by_bin[particle_is_floc],
    s=80,
    alpha=0.85,
    label="Pooled floc bins"
)

plt.axhline(
    n_medium,
    linestyle="--",
    linewidth=1.5,
    label=f"Medium n = {n_medium:.3f}"
)

plt.axhline(
    n_particle,
    linestyle=":",
    linewidth=1.5,
    label=f"Solid particle n = {n_particle:.3f}"
)

plt.xscale("log")
plt.xlabel("Effective diameter (um)")
plt.ylabel("Effective refractive index")
plt.title("Effective Effective Complex Index vs Effective Diameter")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(ri_diag_png, dpi=200)
plt.close()

print(f"✅ Saved {ri_diag_png}")

print("MEAN_FREE_PATH_M =", MEAN_FREE_PATH_M)
print("Transport mode = event-driven free-path sampling from mu_t")
print("R_REAL =", R_REAL)
print("CELL_DIAMETER =", 2*R_REAL)

event_diagnostics_df = pd.DataFrame({
    "effective_bin_index": np.arange(len(particle_diameter_m)),
    "diameter_um": particle_diameter_m * 1.0e6,
    "is_floc": particle_is_floc,
    "effective_bin_kind": effective_bin_kind,
    "mass_fraction": particle_weights,
    "particle_mass_by_bin_kg": particle_mass_by_bin_kg,
    "number_density_per_m3": particle_number_density_by_bin,
    "sigma_s_m2": sigma_s,
    "sigma_a_m2": sigma_a,
    "sigma_t_m2": sigma_t,
    "single_scattering_albedo": single_scattering_albedo_by_bin,
    "mu_s_by_bin_per_m": mu_s_by_bin,
    "mu_a_by_bin_per_m": mu_a_by_bin,
    "mu_t_by_bin_per_m": mu_t_by_bin,
    "extinction_event_strength_by_bin_per_m": extinction_event_strength_by_bin,
    "event_probability": particle_event_weights,
    "g_by_bin": g_by_bin,
    "source_primary_min_um": source_primary_min_diameter_m * 1.0e6,
    "source_primary_max_um": source_primary_max_diameter_m * 1.0e6,
    "floc_band_index": floc_band_index_by_effective_bin
})

event_diagnostics_path = os.path.join(
    OUTDIR,
    "event_probability_diagnostics.csv"
)

event_diagnostics_df.to_csv(
    event_diagnostics_path,
    index=False
)

print(f"✅ Saved {event_diagnostics_path}")


floc_bin_diagnostics_df = pd.DataFrame({
    "mass_concentration_g_per_L": np.full(
        len(particle_diameter_m),
        mass_concentration_g_per_L
    ),
    "effective_bin_index": np.arange(len(particle_diameter_m)),
    "is_floc": particle_is_floc,
    "effective_bin_kind": effective_bin_kind,
    "floc_band_index": floc_band_index_by_effective_bin,

    "effective_diameter_um": particle_diameter_m * 1.0e6,
    "source_primary_min_um": source_primary_min_diameter_m * 1.0e6,
    "source_primary_max_um": source_primary_max_diameter_m * 1.0e6,
    "source_primary_mid_um": (
        np.sqrt(source_primary_min_diameter_m * source_primary_max_diameter_m)
        * 1.0e6
    ),

    "diameter_multiplier": floc_diameter_multiplier_by_bin,
    "mass_fraction": particle_weights,
    "source_primary_mass_fraction": source_primary_mass_fraction,
    "particle_mass_kg": particle_mass_by_bin_kg,
    "effective_density_kg_per_m3": particle_density_by_bin_kg_per_m3,
    "number_density_per_m3": particle_number_density_by_bin,

    "sigma_s_m2": sigma_s,
    "sigma_a_m2": sigma_a,
    "sigma_t_m2": sigma_t,
    "single_scattering_albedo": single_scattering_albedo_by_bin,
    "mu_s_by_bin_per_m": mu_s_by_bin,
    "mu_a_by_bin_per_m": mu_a_by_bin,
    "mu_t_by_bin_per_m": mu_t_by_bin,
    "extinction_event_strength_by_bin_per_m": extinction_event_strength_by_bin,
    "event_probability": particle_event_weights,
    "g_by_bin": g_by_bin,

    "global_floc_mass_fraction": np.full(
        len(particle_diameter_m),
        floc_mass_fraction
    ),
    "global_floc_event_probability": np.full(
        len(particle_diameter_m),
        floc_event_probability
    ),
    "FLOC_FRACTAL_DIMENSION": np.full(
        len(particle_diameter_m),
        FLOC_FRACTAL_DIMENSION
    ),
    "FLOC_COLLISION_LENGTH_M": np.full(
        len(particle_diameter_m),
        FLOC_COLLISION_LENGTH_M
    ),
    "FLOC_SCATTER_EFFICIENCY": np.full(
        len(particle_diameter_m),
        FLOC_SCATTER_EFFICIENCY
    ),
})

floc_only_diagnostics_df = floc_bin_diagnostics_df[
    floc_bin_diagnostics_df["is_floc"]
].copy()

floc_diag_path = os.path.join(
    OUTDIR,
    f"floc_bin_diagnostics_conc_{mass_concentration_g_per_L}.csv"
)

floc_only_diagnostics_df.to_csv(
    floc_diag_path,
    index=False
)

print(f"✅ Saved {floc_diag_path}")


if np.any(particle_is_floc):
    floc_event_df = pd.DataFrame({
        "mass_concentration_g_per_L": mass_concentration_g_per_L,
        "effective_bin_index": np.arange(len(particle_diameter_m)),
        "floc_band_index": floc_band_index_by_effective_bin,
        "effective_diameter_um": particle_diameter_m * 1.0e6,
        "source_primary_min_um": source_primary_min_diameter_m * 1.0e6,
        "source_primary_max_um": source_primary_max_diameter_m * 1.0e6,
        "source_primary_mid_um": (
            np.sqrt(source_primary_min_diameter_m * source_primary_max_diameter_m)
            * 1.0e6
        ),
        "diameter_multiplier": floc_diameter_multiplier_by_bin,
        "mass_fraction": particle_weights,
        "particle_mass_kg": particle_mass_by_bin_kg,
        "effective_density_kg_per_m3": particle_density_by_bin_kg_per_m3,
        "number_density_per_m3": particle_number_density_by_bin,
        "sigma_s_m2": sigma_s,
        "mu_s_by_bin_per_m": mu_s_by_bin,
        "event_probability": particle_event_weights,
        "g_by_bin": g_by_bin,
        "global_floc_event_probability": floc_event_probability,
        "fraction_of_all_floc_events": np.where(
            floc_event_probability > 0.0,
            particle_event_weights / floc_event_probability,
            0.0
        )
    })

    floc_event_df = floc_event_df[
        floc_event_df["floc_band_index"] >= 0
    ].copy()

    floc_event_df = floc_event_df.sort_values(
        "effective_diameter_um"
    )

    floc_event_df["cumulative_fraction_of_floc_events"] = (
        floc_event_df["fraction_of_all_floc_events"].cumsum()
    )

    floc_event_path = os.path.join(
        OUTDIR,
        f"floc_event_weight_by_band_conc_{mass_concentration_g_per_L}.csv"
    )

    floc_event_df.to_csv(
        floc_event_path,
        index=False
    )

    print(f"✅ Saved {floc_event_path}")
else:
    print("No floc bins present; skipped floc_event_weight_by_band diagnostic.")
if np.any(particle_is_floc):
    _floc_indices_for_trace = np.where(particle_is_floc)[0]
    _floc_diam_um_for_trace = particle_diameter_m[_floc_indices_for_trace] * 1.0e6
    _audit_floc_bin_for_trace = int(_floc_indices_for_trace[np.argmin(np.abs(_floc_diam_um_for_trace - 120.0))])
    trace_phase_array(_audit_floc_bin_for_trace, wavelengths[0])
    export_structure_factor_audit(_audit_floc_bin_for_trace, wavelengths[0])
