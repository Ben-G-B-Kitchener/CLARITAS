#!/usr/bin/env python3
# CLARITAS_60.py
# Full simulation script with adaptive GPU chunk_size probing via CuPy.
# Process-based ray tracer with a deliberately constrained pooled-mass floc PSD model, explicit mu_s step factor, fixed floc phase-regime mixture, detector-geometry diagnostics, and adaptive GPU chunking.
#
# Simplification note:
#   The previous floc implementation had too many compensating fit knobs.
#   This version keeps flocculation as a process-based effective PSD transform,
#   but removes concentration-dependent floc size/density interpolation and
#   disables scatter-count-dependent floc angular tuning by default.

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

# ============================ SEDIMENT PARTICLE DISTRIBUTION ============================
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
# No fixed-step scatter probability is used in CLARITAS_42.

# ============================ FLOCCULATION MODEL ============================
# CLARITAS 4.2 floc model.
#
# Primary particles remain individual Mie scatterers.
#
# Flocs are treated as porous/fractal aggregate scattering centres:
#   - eligible primary mass pools into floc bands
#   - floc mass is calculated by fractal scaling
#   - floc scattering cross-section is geometric: Q_floc * pi * r^2
#   - floc angular scattering is sampled from a synthetic fractal aggregate structure-factor CDF
#
# No floc Mie scattering.
# No floc refractive-index interpolation.
# No forward/side/back floc phase ratios.
# No concentration-dependent floc diameter/density retuning.

FLOC_ENABLED = True

# Primary-particle diameter band edges eligible for pooling into floc bands.
# Each band pools primary material above the previous edge and up to this edge.
FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M = np.array([
    1.5e-6,
    2.5e-6,
    4.0e-6,
    6.0e-6,
    9.0e-6,
    13.0e-6,
    18.0e-6,
    25.0e-6
#    35.0e-6,
#    50.0e-6
], dtype=np.float64)

FLOC_POOL_EFFECTIVE_DIAMETER_M = np.array([
    8.0e-6,
    12.0e-6,
    18.0e-6,
    28.0e-6,
    40.0e-6,
    60.0e-6,
    120.0e-6,
    250.0e-6,
    500.0e-6,
    1000.0e-6
], dtype=np.float64)

# CLARITAS_55:
# Primary-size bands and floc-size bins no longer have to be a strict
# one-to-one mapping.  Eligible primary material is redistributed into the
# available floc-size bins with a smooth log-normal pooling kernel below.
if len(FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M) < 1:
    raise ValueError("At least one primary pooling band is required")
if len(FLOC_POOL_EFFECTIVE_DIAMETER_M) < 1:
    raise ValueError("At least one effective floc diameter bin is required")

# Kernel width for distributing each primary band into all floc diameter bins.
# Larger values make broader/more mixed aggregate populations.
# sigma is in natural-log diameter units, i.e. sigma=1 spans about a factor e.
FLOC_POOL_KERNEL_LOG_SIGMA = 0.50

# Computational threshold only: contributions below this fraction of a source
# band's pooled floc mass are omitted.  Set to 0.0 to keep every matrix element.
FLOC_POOL_KERNEL_MIN_PROBABILITY = 0.0

# Physical guard for the 2-D floc pooling kernel. A floc aggregate cannot
# have an effective diameter smaller than the source primary band that built it.
# This prevents unphysical source/floc cells with density above the solid material.
FLOC_POOL_KERNEL_ENFORCE_FLOC_NOT_SMALLER_THAN_SOURCE = True

# Fractal aggregate compactness.
#
# Df = 3.0  solid-sphere-like compact aggregate
# Df = 2.5  compact floc
# Df = 2.0  open mineral aggregate
# Df < 2.0  very loose/fluffy aggregate
FLOC_FRACTAL_DIMENSION = 2.0

# Aggregation opportunity length used later in:
#   floc_mass_fraction = L_collision / (L_collision + eligible_spacing)
FLOC_COLLISION_LENGTH_M = 500.0e-6

# Geometric scattering efficiency for aggregate flocs:
#   sigma_floc = FLOC_SCATTER_EFFICIENCY * pi * r_floc^2
FLOC_SCATTER_EFFICIENCY = 1.0

# Floc angular scattering is built from a per-bin Mie-structure-factor
# aggregate CDF below. There is no scalar floc anisotropy parameter here.

# Roughness / non-sphericity angular jitter.
# Keep fixed across concentrations.
PRIMARY_ROUGHNESS_STD_DEG = 0.0
FLOC_ROUGHNESS_STD_DEG = 0.0
PRIMARY_ROUGHNESS_STD_RAD = np.deg2rad(PRIMARY_ROUGHNESS_STD_DEG)
FLOC_ROUGHNESS_STD_RAD = np.deg2rad(FLOC_ROUGHNESS_STD_DEG)

# Backwards-compatible roughness diagnostic aliases only.
PARTICLE_ROUGHNESS_STD_DEG = PRIMARY_ROUGHNESS_STD_DEG
PARTICLE_ROUGHNESS_STD_RAD = PRIMARY_ROUGHNESS_STD_RAD
# ============================ PARTICLE / FLOC ANGULAR PHYSICS ============================
# Primary particles use Mie angular CDFs.
# Flocs use aggregate transport physics in the CUDA kernel through a per-bin
# synthetic fractal aggregate angular CDF, not forward/side/back fitted ratios.

# Legacy primary reflection branch retained for controlled tests only.
PRIMARY_REFLECT_PROB = 0.0
PRIMARY_REFLECT_SIZE_THRESHOLD = 70.0e-6

# Legacy names retained for backwards-compatible diagnostics only.
# Floc angular physics is controlled by the fixed phase-regime fractions above.
FLOC_REFLECT_PROB = 0.0
FLOC_REFLECT_SIZE_THRESHOLD = 0.0e-6

#particle_diameter_m = null_diameter
#particle_weights = null_weights
#particle_density_kg_per_m3 = 1.0  # loess density

particle_diameter_m = loess_diameter
particle_weights = loess_weights
particle_density_kg_per_m3 = 2600.0  # loess density

#particle_diameter_m = kaolin_diameter
#particle_weights = kaolin_weights
#particle_density_kg_per_m3 = 2600.0  # kaolin density

# Preserve the selected primary PSD, then build the effective PSD used by
# the transport and Mie calculations.
primary_particle_diameter_m = np.asarray(particle_diameter_m, dtype=np.float64)
primary_particle_weights = np.asarray(particle_weights, dtype=np.float64)
primary_particle_weights /= np.sum(primary_particle_weights)

# Original primary-bin density array, retained for diagnostics.
primary_particle_density_by_bin_kg_per_m3 = np.full_like(
    primary_particle_diameter_m,
    particle_density_kg_per_m3,
    dtype=np.float64
)

# Calculate primary number density first, using the unmodified primary PSD.
# This is only used to estimate the spacing of floc-eligible primary particles
# for the aggregation-state model. The final transport number density is
# recalculated later from the effective PSD.
primary_particle_radius_m = primary_particle_diameter_m / 2.0
primary_particle_volumes_m3 = (4.0 / 3.0) * np.pi * primary_particle_radius_m**3
primary_particle_masses_kg = primary_particle_volumes_m3 * particle_density_kg_per_m3

mass_concentration_kg_per_m3_prefloc = mass_concentration_g_per_L

# PSD weight interpretation.
# Keep this as "mass_fraction" if the PSD weights are mass/volume fractions.
# Use "number_fraction" if the PSD weights are particle-count/bin-number frequencies.
# This preserves the process-based kernel model: it only changes how the physical
# number density in each particle-size bin is inferred from the supplied PSD.
#
# Suggested testing:
#   - Loess may fit best as "mass_fraction" depending on how the source PSD was exported.
#   - Kaolin may need "number_fraction" if the listed weights are number/count frequencies.
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
    # Assign primary bins to floc-eligible bands.
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

    # This is no longer a tunable floc-property interpolation state.
    # Retained only as a scalar diagnostic showing that no concentration-dependent
    # floc size/density retuning is being applied.
    floc_property_state = 0.0

    # Fractal floc mass model with kernel-based primary-to-floc pooling.
    #
    # CLARITAS_55 removes the previous one-to-one mapping:
    #
    #     primary band k -> floc bin k
    #
    # and replaces it with a smooth pooling kernel:
    #
    #     P(D_floc_j | source primary band k)
    #
    # Each source-primary band still has a preferred floc diameter, but material
    # from that band can now contribute to every floc-size bin.  Each kernel row
    # is normalised to one, so pooled floc mass is conserved exactly.
    #
    # The fractal mass of an optical floc object remains process-based:
    #
    #     m_floc(k,j) = m0(k) * (D_floc_j / d0(k))^Df
    #
    # where d0(k) is the source-band reference primary diameter and D_floc_j is
    # the effective aggregate diameter.  This means two flocs with the same
    # effective diameter can have different masses if they were assembled from
    # different representative primary-size bands.
    floc_reference_primary_diameter_m = FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M.copy()
    floc_effective_diameter_by_band_m = FLOC_POOL_EFFECTIVE_DIAMETER_M.copy()

    n_source_bands = len(FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M)
    n_floc_bins = len(FLOC_POOL_EFFECTIVE_DIAMETER_M)

    # Preferred floc diameter for each source band.  If the two arrays have the
    # same length, this preserves the old centre locations while broadening them
    # with the kernel.  If they differ, use log-space interpolation across the
    # floc grid so the source bands are spread smoothly across the available
    # aggregate diameters.
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
            # Defensive fallback to the nearest physically allowed floc bin if an
            # over-aggressive threshold removes every contribution.
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

    # Legacy 1-D diagnostic arrays retained for existing plots/HDF5 keys.
    # These represent the old same-index reference path only; actual transport
    # uses the full source-band x floc-bin matrix above.
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

    # Build the effective PSD arrays.
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

    # For each eligible primary source band:
    #   - pool floc_mass_fraction of its mass into all floc bins through the
    #     kernel P(D_floc | source band)
    #   - leave the remaining mass as residual primary bins
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
            # Retain residual primary material as the original bins, scaled so
            # total residual mass from the band is conserved with the same
            # within-band PSD shape.
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

    # Add all non-eligible primary bins unchanged.
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

    # Important:
    # This is now the authoritative per-bin particle/floc mass.
    # Step 3 will use this directly for number-density calculation.
    particle_mass_by_bin_kg = np.asarray(
        effective_particle_masses_kg,
        dtype=np.float64
    )

    # Effective density is now diagnostic only.
    # For primaries this is the solid particle density.
    # For flocs this is derived from the fractal mass model.
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

# Diagnostic multiplier only. For pooled flocs this is relative to the
# mass-weighted geometric mean of the source primary band.
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

# Weighted scalar diagnostic only.
effective_particle_density_kg_per_m3 = np.sum(
    particle_weights * particle_density_by_bin_kg_per_m3
)

# Legacy raw/effective PSD CDF retained only for diagnostics/backwards compatibility.
cdf = np.cumsum(particle_weights)
cdf /= np.sum(cdf)

n_particle = 1.59  # real refractive index of solid primary particle material

# ============================ COMPLEX INDEX / ALBEDO MODEL ============================
# CLARITAS 4.5 optical-object model.
#
# Each effective bin, primary or floc, has:
#   sigma_s        scattering cross-section
#   sigma_a        absorption cross-section
#   sigma_t        extinction cross-section = sigma_s + sigma_a
#   omega          single-scattering albedo = sigma_s / sigma_t
#   phase CDF      angular scattering law used only if the event survives absorption
#
# Primary particles use complex-index Mie to separate scattering and absorption.
# Mie convention here uses m = n - i*k. Increase k to increase absorption.
PRIMARY_REFRACTIVE_INDEX_IMAG_K = 0.001

# Floc absorption/albedo is treated as a finite-optical-depth process through
# a porous aggregate.  The absorption coefficient comes from the Maxwell-Garnett
# effective imaginary index for each floc bin, and the interaction path is the
# mean chord length through an equivalent spherical aggregate:
#
#     mean_chord = 4V/A = 2D/3
#
# The path factor is a physical/process sensitivity parameter allowing the
# mean internal optical path to be lengthened or shortened without changing
# the angular phase function.
FLOC_ABSORPTION_K_MULTIPLIER = 1.0
FLOC_ABSORPTION_PATH_FACTOR = 1.0

n_medium = 1.33
n_external = 1.0  # refractive index outside circular sample boundary, e.g. air


def maxwell_garnett_effective_index(
    matrix_index_complex,
    inclusion_index_complex,
    inclusion_volume_fraction
):
    """
    Maxwell-Garnett effective complex refractive index.

    The floc is treated as solid primary material embedded in the
    surrounding aqueous medium. This is not used to make flocs into
    compact Mie spheres; it provides a physically derived effective
    complex refractive index for porous aggregate optical diagnostics
    and absorption/albedo calculations.

    Parameters
    ----------
    matrix_index_complex : complex
        Complex refractive index of the host medium.
    inclusion_index_complex : complex
        Complex refractive index of the solid primary material.
    inclusion_volume_fraction : float or ndarray
        Solid volume fraction inside the aggregate.

    Returns
    -------
    complex or ndarray
        Effective complex refractive index using the same convention
        as the rest of CLARITAS: n - i*k.
    """
    phi = np.clip(inclusion_volume_fraction, 0.0, 1.0)

    eps_m = matrix_index_complex ** 2
    eps_i = inclusion_index_complex ** 2

    numerator = eps_i + 2.0 * eps_m + 2.0 * phi * (eps_i - eps_m)
    denominator = eps_i + 2.0 * eps_m - phi * (eps_i - eps_m)

    eps_eff = eps_m * numerator / denominator
    n_eff = np.sqrt(eps_eff)

    # Keep the n - i*k sign convention. Numerical branch choices can
    # occasionally return the conjugate for very small k, so force
    # non-positive imaginary part.
    n_eff = np.where(
        np.imag(n_eff) > 0.0,
        np.conjugate(n_eff),
        n_eff
    )

    return n_eff


# ============================ PRIMARY / FLOC EFFECTIVE COMPLEX REFRACTIVE INDEX ============================
# Primary particles use complex-index Mie directly.
#
# Flocs are not treated as compact Mie spheres for angular scattering.
# However, each porous floc bin is assigned an effective complex refractive
# index using Maxwell-Garnett mixing:
#
#     host medium = water / surrounding medium
#     inclusion   = solid primary mineral material
#     phi         = solid volume fraction = rho_floc / rho_primary
#
# This gives each floc size bin a physically derived n_eff - i*k_eff for
# albedo/absorption diagnostics and optical-depth calculations.

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
#theta_deg = np.arange(0.0, 191.0, 0.001)
theta_deg = np.arange(0.0, 181.0, 0.001)


#### LED Beam parameters ####
alpha1 = 1.0
alpha2 = 100.0

#wavelengths = [200e-9, 622e-9, 850e-9]  # in meters
#wavelengths = [200e-9, 622e-9, 950e-9]  # in meters
wavelengths = [622e-9]  # in meters


#### Kernel parameters for TARDIIS####
#R_REAL = 0.049    # Sample radius (m) TARDIIS
R_REAL = 0.049    # Sample radius (m)
#reflection_path_length = R_REAL
#reflection_path_length = 1.0e4  ## steps
#reflection_path_length = 0.0
#RAY_OFFSET = 0.05  # Ray initial y-offset (m) TARDIIS
RAY_OFFSET = 0.005  # Ray initial y-offset (m)
#STEP_SIZE = 1.0e-6  # integration step size (m)
#STEP_SIZE = 1.0e-7  # integration step size (m)
#VISUAL_SCALE = 100.0 TARDIIS
#VIS_SIZE = 2048      # Heatmap resolution TARDIIS
VISUAL_SCALE = 1.0
VIS_SIZE = 4096      # Heatmap resolution
N_RAYS = 1_000_00  # number of rays to simulate
MAX_EXTINCTIONS = 10000

###### OUTPUT DIRECTORY #######
OUTDIR = "."
os.makedirs(OUTDIR, exist_ok=True)

# ============================ PHYSICAL PSD / SCATTERING SETUP ============================
# Unit conversion:
#   1 g/L == 1 kg/m^3
mass_concentration_kg_per_m3 = mass_concentration_g_per_L

# Convert the effective PSD into physical number density per bin.
#
# CLARITAS 4.5 important point:
#
#   particle_mass_by_bin_kg is authoritative.
#
# For primaries:
#   particle_mass_by_bin_kg = rho_solid * pi/6 * d^3
#
# For flocs:
#   particle_mass_by_bin_kg = m0 * (d_floc/d0)^Df
#
# This preserves the fractal aggregate model and avoids accidentally treating
# flocs as compact spheres during number-density calculation.
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

# Diagnostic quantities retained for compatibility with previous printed outputs.
pm = particle_mass_by_bin_kg
particle_mass = average_particle_mass

# Use the first configured wavelength for the scalar transport setup.
scatter_probability_wavelength = wavelengths[0]

# Optical cross-sections and anisotropy per effective bin.
#
# Primary particles:
#   sigma_t = Qext * pi*r^2 from complex-index Mie
#   sigma_s = Qsca * pi*r^2 from complex-index Mie
#   sigma_a = max(sigma_t - sigma_s, 0)
#   omega   = sigma_s / sigma_t
#   g       = Mie asymmetry parameter
#
# Flocs:
#   n_eff - i*k_eff = Maxwell-Garnett(medium, solid primary, solid volume fraction)
#   sigma_s = FLOC_SCATTER_EFFICIENCY * pi*r_floc^2
#   sigma_a = pi*r_floc^2 * (1 - exp(-tau_abs_floc))
#   tau_abs_floc = 4*pi*k_eff*path/lambda
#   phase CDF = synthetic fractal aggregate structure-factor CDF built below
#
# The GPU transport kernel is generic: it selects an extinction event from
# sigma_t, then uses omega to decide whether the event scatters or absorbs.
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

        # Maxwell-Garnett effective imaginary component for this porous floc.
        # The multiplier acts on the physically derived k_eff, not directly on
        # detector response or on a material-specific scatter multiplier.
        k_eff = (
            particle_refractive_index_imag_k_by_bin[idx] *
            FLOC_ABSORPTION_K_MULTIPLIER
        )

        # Absorption coefficient for intensity in a medium with complex index
        # n - i*k:
        #
        #     alpha_abs = 4*pi*k / lambda
        #
        # The finite-floc optical depth is then alpha_abs times the mean chord
        # length through an equivalent spherical aggregate.
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

        # Effective absorption cross-section for the finite porous aggregate.
        # The geometric area gives the encounter cross-section; the optical depth
        # gives the absorption probability conditional on such an encounter.
        sigma_a_this = area * absorption_probability_this
        sigma_t_this = sigma_s_this + sigma_a_this

        sigma_s.append(sigma_s_this)
        sigma_a.append(sigma_a_this)
        sigma_t.append(sigma_t_this)
        single_scattering_albedo_by_bin.append(
            sigma_s_this / sigma_t_this if sigma_t_this > 0.0 else 1.0
        )

        # Diagnostic placeholder; floc angular behaviour comes from the
        # synthetic fractal aggregate CDF built below.
        g_by_bin.append(0.0)

    else:
        # Complex refractive index for primary particles.
        # Absorption is represented by the imaginary component k.
        m_rel = particle_complex_refractive_index_by_bin[idx] / n_medium
        x_mie = 2.0 * np.pi * n_medium * r / scatter_probability_wavelength

        qext, qsca, qback, g = miepython.efficiencies_mx(
            m_rel,
            x_mie
        )

        sigma_t_this = max(float(np.real(qext)) * area, 0.0)
        sigma_s_this = max(float(np.real(qsca)) * area, 0.0)
        sigma_a_this = max(sigma_t_this - sigma_s_this, 0.0)

        # Numerical guard: for very small roundoff errors, preserve extinction.
        if sigma_s_this > sigma_t_this and sigma_t_this > 0.0:
            sigma_s_this = sigma_t_this
            sigma_a_this = 0.0

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

# Macroscopic scattering, absorption and extinction coefficients.
# Units: 1/m
mu_s_by_bin = particle_number_density_by_bin * sigma_s
mu_a_by_bin = particle_number_density_by_bin * sigma_a
mu_t_by_bin = particle_number_density_by_bin * sigma_t

mu_s = np.sum(mu_s_by_bin)
mu_a = np.sum(mu_a_by_bin)
mu_t = np.sum(mu_t_by_bin)

# Reduced scattering coefficient:
#   mu_s_prime = sum_i(mu_s_i * (1 - g_i))
#
# Primary g values come from Mie.
# Floc g values are diagnostic placeholders; floc angular behaviour is in the CDF.
mu_s_prime_by_bin = mu_s_by_bin * (1.0 - g_by_bin)
mu_s_prime = np.sum(mu_s_prime_by_bin)

if mu_s > 0.0:
    g_eff = np.sum(mu_s_by_bin * g_by_bin) / mu_s
else:
    g_eff = 0.0

medium_single_scattering_albedo = mu_s / mu_t if mu_t > 0.0 else 1.0

# Event-driven transport diagnostics.
# CLARITAS_42 does not use a fixed numerical STEP_SIZE or a per-step
# scatter probability. The CUDA transport kernel samples physical free paths
# directly from the macroscopic extinction coefficient:
#
#     s = -ln(U) / mu_t
#
# These values are printed only as physical references.
MEAN_FREE_PATH_M = 1.0 / mu_t if mu_t > 0.0 else np.inf
MEAN_SCATTERING_PATH_M = 1.0 / mu_s if mu_s > 0.0 else np.inf
MEAN_ABSORPTION_PATH_M = 1.0 / mu_a if mu_a > 0.0 else np.inf
TRANSPORT_MEAN_FREE_PATH_M = (
    1.0 / mu_s_prime if mu_s_prime > 0.0 else np.inf
)

# Particle/floc choice during an extinction event is weighted by the
# macroscopic extinction contribution of each effective optical object:
#
#     P_i = N_i * sigma_t_i / sum_j(N_j * sigma_t_j)
#
# where:
#     N_i       = number density of bin i [1/m^3]
#     sigma_t_i = extinction cross-section of bin i [m^2]
#     mu_t_i    = N_i * sigma_t_i [1/m]
#
# This is deliberately NOT PSD-mass weighting and NOT scattering-only
# weighting.  It is the physically correct event-selection distribution for
# event-driven Monte Carlo transport, because a photon reaches an extinction
# event before deciding whether that event scatters or absorbs.
extinction_event_strength_by_bin = particle_number_density_by_bin * sigma_t

# This should be numerically identical to mu_t_by_bin.  Keep both names:
#   - mu_t_by_bin is used in optical diagnostics
#   - extinction_event_strength_by_bin documents the transport meaning
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

if particle_event_cdf[-1] > 0.0:
    particle_event_cdf /= particle_event_cdf[-1]

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
print(f"Boundary refractive indices: n_medium={n_medium:.3f}, n_external={n_external:.3f}")

# Reflection path length disabled unless you deliberately re-enable the old empirical model.
reflection_path_length = 0.0

#reflection_path_length = 0.0
#scatter_prob_per_step = (1.0 / average_particle_separation_m) ** SCAT_PROB_EXPONENT
#scatter_prob_per_step = STEP_SIZE / average_particle_separation_m 
#scatter_prob_per_step = mu_s * STEP_SIZE / average_particle_separation_m

##############################################


def closest_index(arr, value):
    i = np.searchsorted(arr, value)
    if i == 0:
        return 0
    if i == len(arr):
        return len(arr) - 1
    left = i - 1
    right = i
    return left if abs(arr[left] - value) <= abs(arr[right] - value) else right

# ================= ANGULAR PROFILES (host side - primary Mie + floc Mie-structure aggregate CDF, cached) =================
theta_rad = np.deg2rad(theta_deg)

# CLARITAS 4.2 angular-profile cache.
#
# Primary bins:
#   - Mie angular intensity
#   - Mie CDF sampled by CUDA kernel
#
# Floc bins:
#   - synthetic fractal aggregate angular profile
#   - representative source-primary Mie form factor
#   - fractal structure factor from effective floc size and Df
#   - optical-depth blend toward structure-dominated scattering
#
# The CUDA kernel samples angle_cdf_table for both primary and floc bins.

#mie_cache_version = "claritas_3p0_primary_mie_floc_fractal_hg_v1"

# ================= AGGREGATE PHASE OPTIONS (CLARITAS_51) =================
# Monomer phase split inherited from CLARITAS_50.
MONOMER_PHASE_COMPONENT_SPLIT_ENABLED = True
MONOMER_DIFFRACTION_WIDTH_FACTOR = 1.0

# Coherence-limited structure factor.
# The fully coherent Debye S(q) can make large flocs act like unrealistically
# coherent optical apertures. CLARITAS_51 damps long-range pair interference
# while keeping short-range primary-particle correlations:
#
#     sinc(qr) -> exp[-(r/Lc)^2] * sinc(qr)
#
# The default Lc is a Fresnel-like transverse coherence patch length:
#
#     Lc = factor * sqrt(lambda * Rg / n_medium)
#
# where Rg is the generated aggregate radius of gyration. This is process-based:
# it derives from wavelength and aggregate size, not detector response.
AGGREGATE_COHERENCE_LIMIT_ENABLED = True
AGGREGATE_COHERENCE_LENGTH_MODE = "fresnel_patch"  # "fresnel_patch", "radius_gyration", "infinite"
AGGREGATE_COHERENCE_LENGTH_FACTOR = 1.0

# Angular sampling measure used to convert intensity I(theta) into the
# CDF sampled by the transport kernel.
#
#   0.0 -> planar/raw angular CDF:       dP proportional to I(theta) dtheta
#   1.0 -> solid-angle-weighted CDF:     dP proportional to I(theta) sin(theta) dtheta
#
# Keep this explicit because it has a large effect on detector response.
ANGULAR_CDF_POWER = 1.0

# ================= FLOC INTERNAL TRANSPORT COMPONENT (CLARITAS_60) =================
# The Debye structure-factor term is first-order aggregate scattering.  It
# captures coherent pair correlations but does not represent photons that
# enter a porous floc and undergo a short internal random walk before leaving.
#
# CLARITAS_60 adds a process-derived internal-transport component to the
# floc phase function:
#
#   I_floc = diffraction
#          + (1 - f_transport) * coherent_internal
#          + f_transport * internally_transport_broadened_internal
#
# where f_transport = 1 - exp(-tau_transport), and tau_transport is estimated
# from the representative monomer number density inside the floc, the Mie
# scattering cross-section of that representative monomer, the floc mean chord
# length, and (1-g) for transport rather than raw scattering optical depth.
#
# This is not a detector backscatter boost.  It is an optical-depth-derived
# aggregate-internal transport approximation.
FLOC_INTERNAL_TRANSPORT_ENABLED = True
FLOC_INTERNAL_TRANSPORT_PATH_FACTOR = 1.0
FLOC_INTERNAL_TRANSPORT_USE_REDUCED_OD = True
FLOC_INTERNAL_TRANSPORT_MIN_TAU = 0.0
FLOC_INTERNAL_TRANSPORT_MAX_TAU = 50.0

# Stores per-bin diagnostics from the most recent floc phase construction.
floc_internal_transport_diagnostics = {}

mie_cache_version = "claritas_6p0_internal_floc_transport_v1"

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
    str(ANGULAR_CDF_POWER) +
    str(FLOC_INTERNAL_TRANSPORT_ENABLED) +
    str(FLOC_INTERNAL_TRANSPORT_PATH_FACTOR) +
    str(FLOC_INTERNAL_TRANSPORT_USE_REDUCED_OD) +
    str(FLOC_INTERNAL_TRANSPORT_MIN_TAU) +
    str(FLOC_INTERNAL_TRANSPORT_MAX_TAU) +
    str(FLOC_POOL_KERNEL_ENFORCE_FLOC_NOT_SMALLER_THAN_SOURCE)
)

mie_cache_hash = hashlib.md5(mie_cache_key.encode()).hexdigest()
mie_cache_file = os.path.join(OUTDIR, f"angular_cache_{mie_cache_hash}.npz")



def _normalise_profile_for_cdf(I):
    """Return a positive finite angular profile suitable for CDF building."""
    I = np.asarray(I, dtype=np.float64)
    I = np.real(I)
    I[~np.isfinite(I)] = 0.0
    I = np.maximum(I, 0.0)
    if np.sum(I) <= 0.0:
        I = np.ones_like(theta_rad, dtype=np.float64)
    return I


def build_angular_pdf_and_cdf(I, theta_rad_values):
    """
    Convert an angular intensity profile into the exact PDF/CDF used by
    the transport kernel.

    This single helper is deliberately used by both:
      - angular-cache construction
      - phase-function audit / sampling validation

    so the audit cannot silently compare a raw-I CDF against an I*sin(theta)
    transport CDF again.
    """
    I = _normalise_profile_for_cdf(I)

    power = float(ANGULAR_CDF_POWER)
    if abs(power) < 1.0e-15:
        angular_measure = np.ones_like(theta_rad_values, dtype=np.float64)
    else:
        angular_measure = np.sin(theta_rad_values).astype(np.float64)
        angular_measure[~np.isfinite(angular_measure)] = 0.0
        angular_measure = np.maximum(angular_measure, 0.0) ** power

    pdf_theta = I * angular_measure
    pdf_theta[~np.isfinite(pdf_theta)] = 0.0
    pdf_theta = np.maximum(pdf_theta, 0.0)

    if np.sum(pdf_theta) <= 0.0:
        # Extremely defensive fallback; should not normally be reached.
        pdf_theta = I.copy()

    pdf_sum = np.sum(pdf_theta)
    if pdf_sum > 0.0:
        pdf_theta = pdf_theta / pdf_sum
    else:
        pdf_theta = np.ones_like(I, dtype=np.float64) / len(I)

    angle_cdf = np.cumsum(pdf_theta).astype(np.float64)
    if angle_cdf[-1] > 0.0:
        angle_cdf /= angle_cdf[-1]
    else:
        angle_cdf = np.linspace(
            1.0 / len(theta_rad_values),
            1.0,
            len(theta_rad_values),
            dtype=np.float64
        )

    angle_cdf[-1] = 1.0
    return pdf_theta, angle_cdf


# ================= MONOMER PHASE COMPONENT SPLIT (CLARITAS_50) =================
# The previous floc phase law used:
#
#     I_floc(theta) = P_primary(theta) * S_aggregate(q)
#
# That can over-concentrate the forward lobe because the monomer Mie phase
# contains a diffraction-dominated forward peak and the aggregate structure
# factor also contains a coherent forward peak.  CLARITAS_50 separates the
# representative monomer phase into:
#
#     P_primary(theta) = P_diffraction(theta) + P_internal(theta)
#
# and applies the aggregate structure factor only to the internal component:
#
#     I_floc(theta) = P_diffraction(theta) + P_internal(theta) * S_aggregate(q)
#
# This is not a detector-fitted backscatter boost.  It is a process-based
# attempt to avoid double-counting coherent forward diffraction while keeping
# the monomer angular physics and aggregate correlation physics explicit.
def split_monomer_phase_into_diffraction_and_internal(
    primary_form_factor,
    rep_diameter_m,
    wavelength_m,
    theta_rad_values
):
    """
    Split a representative monomer Mie phase function into an approximate
    forward-diffraction component and a residual internal/scattering component.

    The diffraction proxy is an Airy/sinc-like forward envelope with the same
    zero-angle height as the monomer phase.  It is clipped so it never exceeds
    the original monomer phase at any angle.  The residual is the remaining
    non-negative part of the monomer phase.

    Returns
    -------
    primary_diffraction : ndarray
        Approximate monomer forward-diffraction component.
    primary_internal : ndarray
        Residual monomer phase component after removing diffraction proxy.
    """
    P = _normalise_profile_for_cdf(primary_form_factor)

    d = max(float(rep_diameter_m), 1.0e-12)
    wl = max(float(wavelength_m), 1.0e-12)

    # Dimensionless circular-aperture diffraction variable.  The form is a
    # smooth proxy rather than a full Bessel/Airy implementation, avoiding new
    # dependencies while preserving the physically important width scaling:
    #
    #     theta_diff ~ lambda / (n_medium * d)
    #
    # Wider for smaller monomers, narrower for larger monomers.
    x = (
        np.pi * n_medium * d * np.sin(theta_rad_values) /
        (wl * max(float(MONOMER_DIFFRACTION_WIDTH_FACTOR), 1.0e-12))
    )

    # np.sinc(z) = sin(pi*z)/(pi*z), so np.sinc(x/pi) = sin(x)/x.
    diffraction_envelope = np.sinc(x / np.pi) ** 2
    diffraction_envelope = np.asarray(diffraction_envelope, dtype=np.float64)
    diffraction_envelope[~np.isfinite(diffraction_envelope)] = 0.0
    diffraction_envelope = np.maximum(diffraction_envelope, 0.0)

    primary_diffraction = P[0] * diffraction_envelope

    # The extracted diffraction component cannot be larger than the actual
    # Mie phase.  This keeps the split conservative and guarantees that the
    # residual internal component is non-negative.
    primary_diffraction = np.minimum(primary_diffraction, P)
    primary_internal = np.maximum(P - primary_diffraction, 0.0)

    # If the split degenerates because the diffraction proxy consumes almost
    # everything, keep a tiny residual copy so all non-forward angles remain
    # numerically represented.
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
    """
    Approximate the aggregate-internal transport component for a porous floc.

    This estimates the reduced optical depth across the floc from the number
    density of representative monomers inside the aggregate and the Mie
    scattering cross-section of those monomers.  The transport fraction is then:

        f_transport = 1 - exp(-tau_transport)

    The internally transported component is represented as an isotropised
    angular intensity with the same total raw angular weight as the original
    primary-internal component.  This broadens the internal component without
    changing the floc encounter cross-section or detector geometry.
    """
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

    # Internally transported light is approximated as directionally mixed after
    # a short random walk.  Use a constant angular intensity; the selected
    # angular measure in build_angular_pdf_and_cdf() then maps this consistently
    # to the current detector/transport geometry.  Preserve the raw angular
    # weight of the internal component so this changes shape, not total scale.
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
    """
    Build the CLARITAS_60 floc phase law from separated monomer components.

    CLARITAS_50 used:

        I = diffraction + internal * S(q)

    CLARITAS_60 keeps that coherent aggregate term but adds an optical-depth
    derived internal-transport term for photons that undergo a short random walk
    within the porous aggregate before leaving:

        I = diffraction
          + (1-f) * internal * S(q)
          + f     * internally_transport_broadened_internal

    where f = 1 - exp(-tau_transport).  tau_transport is derived from monomer
    number density inside the floc, representative monomer Mie sigma_s, chord
    length, and optionally (1-g).
    """
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
    """Pick the nearest primary PSD bin to the source band geometric midpoint."""
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
    """Stable integer seed for synthetic aggregate generation."""
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
    """
    Estimate the number of representative source-primary monomers in a floc.

    This uses existing fractal mass and source-primary quantities only. The
    returned count is used for the aggregate structure factor. It is capped for
    CPU cost, while the physical count is retained for diagnostics.
    """
    rep_idx = _representative_primary_index_for_effective_bin(bin_idx)
    d_rep = primary_particle_diameter_m[rep_idx]
    m_rep = particle_density_kg_per_m3 * (np.pi / 6.0) * d_rep**3

    physical_count = (
        particle_mass_by_bin_kg[bin_idx] / m_rep
        if m_rep > 0.0 else 1.0
    )
    physical_count = max(float(physical_count), 1.0)

    # Pair summation cost is O(N^2). This cap is computational, not a fit knob.
    # It preserves the aggregate length scales while keeping angular-table
    # generation practical.
    synthetic_count = int(np.clip(round(physical_count), 8, 384))

    return synthetic_count, physical_count, rep_idx


def _generate_synthetic_fractal_points(bin_idx, n_points, wavelength_m):
    """
    Generate a deterministic synthetic fractal-like 3D aggregate.

    The radial number scaling follows N(r) proportional to r^Df inside the
    aggregate envelope. This does not try to build a mechanically exact DLCA/RLCA
    cluster; it generates a pair-distance distribution with the requested Df,
    floc size, and source-primary scale for structure-factor evaluation.
    """
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

    # Recentre so the aggregate centre of mass is at the scattering event.
    points -= np.mean(points, axis=0)

    return points


def _aggregate_coherence_length_from_rg(rg_m, wavelength_m):
    """
    Physically derived aggregate coherence length for CLARITAS_51.

    The returned length controls how much distant primary-particle pairs
    interfere coherently in the Debye structure factor. It is not a detector
    boost: it is derived from wavelength, medium index and aggregate size.
    """
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
        # Fresnel-like transverse coherence patch across an extended aggregate.
        # Long-range pair interference beyond this scale is progressively
        # suppressed, while short-range aggregate correlations remain coherent.
        base_length = np.sqrt(wl * rg / max(float(n_medium), 1.0e-12))

    else:
        raise ValueError(
            "AGGREGATE_COHERENCE_LENGTH_MODE must be 'fresnel_patch', "
            "'radius_gyration', or 'infinite'"
        )

    return max(float(AGGREGATE_COHERENCE_LENGTH_FACTOR) * float(base_length), 1.0e-12)


def _pair_distance_structure_factor(theta_rad_values, wavelength_m, points):
    """
    Coherence-limited histogram-based Debye pair-distance structure factor.

    This returns the aggregate structure factor S(q), not a complete floc
    phase function.  In CLARITAS_51 the long-range Debye pair interference is
    damped by a physically derived coherence envelope. The floc phase function
    is still built as:

        I_floc(theta) = P_primary(theta) * S_aggregate(q)

    where P_primary(theta) is the representative source-primary Mie phase
    function and S_aggregate(q) is the pair-correlation structure factor.

    The pair distances are compressed into a histogram before evaluating
    sin(qr)/(qr), which is a numerical quadrature of the same Debye sum and
    avoids constructing a huge q-by-pair array. CLARITAS_51 applies
    exp[-(r/Lc)^2] to the pair histogram before the Debye sum, suppressing
    only long-range coherent interference.
    """
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

    # Numerical Debye sums can dip slightly below zero because of histogram
    # quadrature and oscillatory sinc terms.  Negative intensities are not
    # physical, so clamp at zero while preserving the angular structure.
    S = np.maximum(S, 0.0)

    if np.max(S) <= 0.0:
        S = np.ones_like(theta_rad_values, dtype=np.float64)

    return S, rg

def floc_synthetic_structure_profile(bin_idx, wavelength_m, mu_values, theta_rad_values):
    """
    CLARITAS_51 floc aggregate phase function.

    The floc is treated as a correlated collection of representative primary
    particles, but the monomer Mie phase is split before applying the
    aggregate structure factor:

        P_primary(theta) = P_diffraction(theta) + P_internal(theta)

        I_floc(theta) = P_diffraction(theta) + P_internal(theta) * S_aggregate(q)

    This avoids applying the aggregate coherent forward lobe to the monomer
    diffraction lobe a second time.  No HG, no retroreflection, no detector
    tuning.
    """
    n_synth, n_physical, rep_idx = _representative_monomer_count_for_floc_bin(bin_idx)
    points = _generate_synthetic_fractal_points(bin_idx, n_synth, wavelength_m)
    structure_factor, rg = _pair_distance_structure_factor(
        theta_rad_values,
        wavelength_m,
        points
    )

    rep_radius = primary_particle_diameter_m[rep_idx] / 2.0
    rep_diameter = primary_particle_diameter_m[rep_idx]
    rep_m_rel = solid_primary_complex_index / n_medium
    rep_x = 2.0 * np.pi * n_medium * rep_radius / wavelength_m

    S1_rep, S2_rep = miepython.S1_S2(rep_m_rel, rep_x, mu_values)
    primary_form_factor = 0.5 * (np.abs(S1_rep)**2 + np.abs(S2_rep)**2)
    primary_form_factor = np.real(primary_form_factor).astype(np.float64)

    I, primary_diffraction, primary_internal = (
        build_floc_phase_from_split_monomer_and_structure(
            primary_form_factor,
            structure_factor,
            rep_diameter,
            wavelength_m,
            theta_rad_values,
            bin_idx=bin_idx
        )
    )

    return I



# ================= PHASE ARRAY TRACE (CLARITAS_49) =================
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
    I, P_diff, P_internal = build_floc_phase_from_split_monomer_and_structure(
        P, S, rep_diameter, wavelength_m, theta_rad, bin_idx=bin_idx
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
        print("TRACE internal transport tau_s:", diag.get("tau_s_internal", np.nan))
        print("TRACE internal transport tau_transport:", diag.get("tau_transport_internal", np.nan))
        print("TRACE internal transport fraction:", diag.get("transport_fraction", np.nan))
    print("TRACE first10:", I[:10])
    print("TRACE last10:", I[-10:])


# ================= STRUCTURE FACTOR AUDIT (CLARITAS_52) =================
def export_structure_factor_audit(bin_idx, wavelength_m):
    """
    Export detailed diagnostics for the Debye / coherence-limited structure
    factor calculation for one representative floc bin.

    This does not change the model. It records whether the coherence envelope
    is actually suppressing pair-distance contributions before normalisation.
    """
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
            if is_floc:
                # Flocs use representative-primary Mie form factor multiplied by
                # the synthetic aggregate Debye structure factor. No HG, no
                # retroreflection, and no detector-fitted phase parameters.
                I = floc_synthetic_structure_profile(
                    bin_idx,
                    wl,
                    mu,
                    theta_rad
                )

            else:
                m_rel = particle_complex_refractive_index_by_bin[bin_idx] / n_medium
                x = 2.0 * np.pi * n_medium * radius / wl

                S1, S2 = miepython.S1_S2(m_rel, x, mu)

                # Unpolarised Mie intensity.
                I = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2)
                I = np.real(I).astype(np.float64)

            I = _normalise_profile_for_cdf(I)

            # Keep this diagnostic profile as raw angular intensity, not as the
            # probability-weighted transport PDF.
            psd_weighted_profile += weight * I

            # Build the exact PDF/CDF used by the transport kernel.  The same
            # helper is used by the audit routine below, preventing mismatches
            # between cache construction and diagnostics.
            _, angle_cdf = build_angular_pdf_and_cdf(I, theta_rad)

            per_particle_cdfs.append(angle_cdf)

        all_profiles.append(psd_weighted_profile.astype(np.float64))
        cdf_profiles.append(np.asarray(per_particle_cdfs, dtype=np.float64))

    np.savez_compressed(
        mie_cache_file,
        all_profiles=np.asarray(all_profiles, dtype=np.float64),
        cdf_profiles=np.asarray(cdf_profiles, dtype=np.float64)
    )

    print(f"Saved angular cache: {mie_cache_file}")


# Export angular scattering as before.
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

# Save CLARITAS_60 floc internal-transport diagnostics generated during angular table construction/audit.
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


# ================= PHASE-FUNCTION AUDIT (CLARITAS_48) =================
def export_phase_function_audit_for_representative_floc(wl_idx=0, target_floc_diameter_um=120.0):
    """
    Audit that the CPU-generated floc phase function and CDF are the same
    objects being uploaded for GPU sampling.

    This does not change transport or physics. It exports:
      - phase_function_cpu_vs_gpu.csv
      - phase_sampling_validation.csv
      - phase_sampling_validation.png
    """
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

    # Recompute the expected PDF/CDF using the exact same angular measure as
    # the cache builder and transport table.
    final_pdf, cpu_cdf_recomputed = build_angular_pdf_and_cdf(
        final_intensity,
        theta_rad
    )

    # This is the actual CDF table used by the main transport code.
    cpu_cdf_table = np.asarray(cdf_profiles[wl_idx][audit_bin], dtype=np.float64)

    # Upload-copyback audit: this validates that CuPy/GPU buffer contents match
    # the CPU-side CDF before CUDA sampling.
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
        "angular_cdf_power": np.full_like(theta_deg, ANGULAR_CDF_POWER, dtype=np.float64),
        "final_pdf_transport_measure": final_pdf,
        "cpu_cdf_recomputed": cpu_cdf_recomputed,
        "cpu_cdf_table_used_by_transport": cpu_cdf_table,
        "gpu_cdf_copyback": gpu_cdf_copyback,
        "cdf_recompute_minus_table": cdf_recompute_vs_table,
        "cdf_table_minus_gpu_copyback": cdf_table_vs_gpu,
    })

    phase_audit_path = os.path.join(OUTDIR, "phase_function_cpu_vs_gpu.csv")
    phase_audit_df.to_csv(phase_audit_path, index=False)

    print("=========== CLARITAS_48 PHASE-FUNCTION AUDIT ===========")
    print(f"Audit floc bin index: {audit_bin}")
    print(f"Audit floc effective diameter: {particle_diameter_m[audit_bin]*1.0e6:.3f} um")
    print(f"Representative primary diameter: {primary_particle_diameter_m[rep_idx]*1.0e6:.3f} um")
    print(f"Synthetic monomer count: {n_synth}; physical count estimate: {n_physical:.3f}")
    print(f"Radius of gyration: {rg*1.0e6:.3f} um")
    print(f"Monomer component split enabled: {MONOMER_PHASE_COMPONENT_SPLIT_ENABLED}")
    print(f"Monomer diffraction width factor: {MONOMER_DIFFRACTION_WIDTH_FACTOR:.6g}")
    print(f"Angular CDF power: {ANGULAR_CDF_POWER:.6g}")
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

    # Sampling validation. This uses the same inverse-CDF operation as the CUDA
    # kernel, but on CPU so the expected and sampled distributions can be
    # directly compared and plotted at full angular-grid resolution.
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
# ================= ANGLE SAMPLING (host-side) =================
def sample_beta_angles(N, a1, a2):
    N_half = N // 2
    u_left = np.random.beta(a1, a2, N_half)
    angles_left = (1-u_left) * (np.pi/2) - (np.pi/2)   # -pi/2 = straight up
    angles_right = -angles_left
    angles = np.concatenate([angles_left, angles_right])
    if len(angles) < N:
        angles = np.append(angles, 0.0)
    return angles.astype(np.float64)

# ================= CUDA KERNEL (CuPy RawKernel) =================
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

__device__ float rnd_uniform(unsigned int* state) {
    unsigned int r = xorshift32_state(state);
    return (float)(r * 2.3283064e-10f);
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
    float u)
{
    int lo = 0;
    int hi = n - 1;

    while (lo < hi) {
        int mid = (lo + hi) >> 1;

        if (u <= (float)cdf[mid]) {
            hi = mid;
        }
        else {
            lo = mid + 1;
        }
    }

    return lo;
}

__global__ void trace_kernel(
    const float MAX_EXTINCTIONS,
    const float MU_T,
    const float PRIMARY_ROUGHNESS_STD_RAD,
    const float FLOC_ROUGHNESS_STD_RAD,
    const float R_REAL,
    const float R_OFF,
    const int VIS_SIZE,
    const float VISUAL_SCALE,
    const double* angles_init,
    const int N_rays,
    const double* extinction_cdf_table,
    const double* particle_is_floc_table,
    const double* single_scattering_albedo_table,
    const double* angle_cdf_table,
    const double* theta_table,
    const int n_particles,
    const int n_theta,
    float* heatmap_flat,
    float* exit_dir_out,
    float* exit_x_out,
    float* exit_y_out,
    float* ray_path_length_out,
    int* scatter_count_out,
    int* floc_event_count_out,
    int* last_event_was_floc_out,
    int* last_scatter_bin_out,
    int* extinction_count_out,
    int* absorbed_out,
    unsigned int seed0,
    unsigned int seed1,
    unsigned int seed2)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N_rays) return;

    unsigned int state = seed0 + (unsigned int)tid * 74729u + 13u;
    unsigned int stateJITTER = seed2 + (unsigned int)tid * 74729u + 13u;

    float beam_sigma = 0.00001f;

    float u1 = fmaxf(rnd_uniform(&state), 1.0e-12f);
    float u2 = rnd_uniform(&state);

    float gaussian =
        sqrtf(-2.0f * logf(u1)) *
        cosf(2.0f * 3.1415927f * u2);

    float x0 = beam_sigma * gaussian;
    float y0 = -(R_REAL + R_OFF);

    float angle_init = (float)angles_init[tid];

    float vx = sinf(angle_init);
    float vy = cosf(angle_init);

    if (vy < 0.0f) {
        vx = -vx;
        vy = -vy;
    }

    float b = x0 * vx + y0 * vy;
    float c = x0 * x0 + y0 * y0 - R_REAL * R_REAL;
    float disc = b * b - c;

    if (disc <= 0.0f) return;

    float t = -b - sqrtf(disc);
    if (t < 0.0f) return;

    float x = x0 + t * vx;
    float y = y0 + t * vy;

    int absorbed = 0;
    int scatter_count = 0;
    int extinction_count = 0;
    int floc_event_count = 0;
    int last_event_was_floc = 0;
    int last_scatter_bin = -1;
    float rpl = 0.0f;

    const float HEATMAP_SAMPLE_SPACING = 1.0e-6f;

    while (x * x + y * y <= R_REAL * R_REAL) {

        if (extinction_count >= (int)MAX_EXTINCTIONS) {
            absorbed = 1;
            break;
        }

        if (MU_T <= 0.0f) {
            absorbed = 1;
            break;
        }

        float u_path = fmaxf(rnd_uniform(&state), 1.0e-12f);
        float free_path = -logf(u_path) / MU_T;

        int heatmap_steps = (int)ceilf(free_path / HEATMAP_SAMPLE_SPACING);
        if (heatmap_steps < 1) heatmap_steps = 1;

        float dx = vx * free_path / (float)heatmap_steps;
        float dy = vy * free_path / (float)heatmap_steps;

        int exited = 0;
        float travelled = 0.0f;

        for (int hs = 0; hs < heatmap_steps; hs++) {
            x += dx;
            y += dy;
            travelled += free_path / (float)heatmap_steps;

            if (x * x + y * y > R_REAL * R_REAL) {
                exited = 1;
                break;
            }

            int ix = (int)(((x + R_REAL) / (2.0f * R_REAL)) * (float)VIS_SIZE);
            if (ix < 0) ix = 0;
            if (ix > VIS_SIZE - 1) ix = VIS_SIZE - 1;

            int iy = VIS_SIZE - 1 - (int)(((y + R_REAL) / (2.0f * R_REAL)) * (float)VIS_SIZE);
            if (iy < 0) iy = 0;
            if (iy > VIS_SIZE - 1) iy = VIS_SIZE - 1;

            int pix_idx = iy * VIS_SIZE + ix;
            atomicAdd(&heatmap_flat[pix_idx], VISUAL_SCALE);
        }

        rpl += travelled;

        if (exited) {
            break;
        }

        extinction_count++;

        float u_particle = rnd_uniform(&state);
        int pidx = cdf_binary_search(
            extinction_cdf_table,
            n_particles,
            u_particle
        );

        float albedo_this = (float)single_scattering_albedo_table[pidx];
        albedo_this = fminf(1.0f, fmaxf(0.0f, albedo_this));

        bool is_floc_event =
            particle_is_floc_table[pidx] > 0.5;

        if (rnd_uniform(&state) > albedo_this) {
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

        float u_angle = rnd_uniform(&state);
        int angle_offset = pidx * n_theta;

        int idx = cdf_binary_search(
            &angle_cdf_table[angle_offset],
            n_theta,
            u_angle
        );

        float theta_3d = (float)theta_table[idx];

        float sign2 =
            (rnd_uniform(&stateJITTER) < 0.5f) ? -1.0f : 1.0f;

        float theta_projected = sign2 * theta_3d;

        float ray_angle = atan2f(vy, vx);

        float roughness_std_this =
            is_floc_event ? FLOC_ROUGHNESS_STD_RAD : PRIMARY_ROUGHNESS_STD_RAD;

        float roughness_jitter =
            gaussian_jitter(&stateJITTER, roughness_std_this);

        float new_angle =
            ray_angle + theta_projected + roughness_jitter;

        vx = cosf(new_angle);
        vy = sinf(new_angle);
    }

    scatter_count_out[tid] = scatter_count;
    floc_event_count_out[tid] = floc_event_count;
    last_event_was_floc_out[tid] = last_event_was_floc;
    last_scatter_bin_out[tid] = last_scatter_bin;
    extinction_count_out[tid] = extinction_count;
    absorbed_out[tid] = absorbed;

    if (absorbed == 0) {
        exit_x_out[tid] = x;
        exit_y_out[tid] = y;
        exit_dir_out[tid] = atan2f(vy, vx);
        ray_path_length_out[tid] = rpl;
    }
}
}
"""

# compile kernel
module = cp.RawModule(code=cuda_src, options=('-std=c++11',))
trace_kernel = module.get_function('trace_kernel')

# ================= Adaptive chunk size helper =================
def get_gpu_free_bytes():
    """Return (free_bytes, total_bytes) using CuPy runtime; robust fallback."""
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
    """
    Estimate a safe chunk size (in rays) given free GPU bytes.
    - safety_fraction: fraction of reported free memory to use for buffers
    - overhead_bytes: reserved bytes for other allocations / kernel overhead
    """
    # Conservative per-ray memory estimate (bytes):
    # angles (double): 8 bytes per ray (but angles in device_full already allocated once)
    # exit arrays per-chunk: 3 * 4 bytes = 12 bytes
    # kernel stack / temp: assume 40 bytes per ray (conservative)
    # So per-ray ~ 60 bytes. Use safety multiplier.
    per_ray_bytes = 64  # conservative

    usable = int(free_bytes * safety_fraction) - overhead_bytes
    if usable <= 0:
        return 0
    return max(1, usable // per_ray_bytes)

# ================= GPU wrapper function (adaptive chunking) =================
from tqdm import tqdm

# ================= GPU wrapper function (adaptive chunking) =================
def trace_rays_gpu(angles_init_np, primary_roughness_std_rad, floc_roughness_std_rad,
                   reflection_path_length, n_medium, n_external, R_REAL, RAY_OFFSET, VIS_SIZE,
                   VISUAL_SCALE, particle_cdf_table_np,
                   particle_is_floc_table_np, single_scattering_albedo_table_np, angle_cdf_table_np, theta_table_np,
                   hdf5_file="ray_exits.h5",
                   safety_fraction=0.01, min_chunk=100_000, max_chunk=1_000_000):
    """
    GPU ray tracing with adaptive chunking, streaming per-ray exit data to HDF5.
    Returns only heatmap; exit data is streamed to HDF5.
    """
    N = angles_init_np.shape[0]

    particle_cdf_dev = cp.asarray(particle_cdf_table_np, dtype=cp.float64)
    particle_is_floc_dev = cp.asarray(particle_is_floc_table_np, dtype=cp.float64)
    single_scattering_albedo_dev = cp.asarray(single_scattering_albedo_table_np, dtype=cp.float64)
    angle_cdf_dev = cp.asarray(angle_cdf_table_np, dtype=cp.float64)
    theta_dev = cp.asarray(theta_table_np, dtype=cp.float64)
    heatmap_dev = cp.zeros((VIS_SIZE*VIS_SIZE,), dtype=cp.float32)
    threads_per_block = 256

    free_bytes, total_bytes = get_gpu_free_bytes()
    if free_bytes is None:
        estimated_chunk = 2_000_000
    else:
        est = estimate_chunk_size_bytes(free_bytes, safety_fraction=safety_fraction)
        estimated_chunk = int(max(min_chunk, min(est, max_chunk)))

    # Open HDF5 for streaming output
    with h5py.File(hdf5_file, "w") as f:
        dset_exit_x = f.create_dataset("exit_x", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_exit_y = f.create_dataset("exit_y", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_exit_dir = f.create_dataset("exit_dir", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_rpl = f.create_dataset("exit_rpl", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_scatter_count = f.create_dataset("scatter_count", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_floc_event_count = f.create_dataset("floc_event_count", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_last_event_was_floc = f.create_dataset("last_event_was_floc", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_last_scatter_bin = f.create_dataset("last_scatter_bin", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_extinction_count = f.create_dataset("extinction_count", shape=(N,), dtype='i4', chunks=(estimated_chunk,))
        dset_absorbed = f.create_dataset("absorbed", shape=(N,), dtype='i4', chunks=(estimated_chunk,))

        # Wrap per-chunk loop with tqdm
        for start in tqdm(range(0, N, estimated_chunk), total=(N + estimated_chunk - 1)//estimated_chunk, desc="Tracing rays"):
            end = min(N, start + estimated_chunk)
            sz = end - start

            angles_chunk = cp.asarray(angles_init_np[start:end], dtype=cp.float64)
            # Initialise outputs to invalid sentinels.
            # Absorbed rays deliberately do not write exit values in the CUDA kernel,
            # so these sentinels prevent absorbed rays being mis-counted as real 0 deg exits.
            exit_dir_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
            exit_x_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
            exit_y_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
            rpl_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
            scatter_count_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)
            floc_event_count_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)
            last_event_was_floc_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)
            last_scatter_bin_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)
            extinction_count_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)
            absorbed_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)

            blocks = (sz + threads_per_block - 1) // threads_per_block
            seed0 = np.uint32(np.random.randint(1, 2**31 - 1))
            seed1 = np.uint32(np.random.randint(1, 2**31 - 1))
            seed2 = np.uint32(np.random.randint(1, 2**31 - 1))

            # launch kernel with OOM handling
            launched = False
            attempt_chunk = estimated_chunk
            while not launched:
                try:
                    trace_kernel((blocks,), (threads_per_block,),
                        (
                            np.float32(MAX_EXTINCTIONS),
                            np.float32(mu_t),
                            np.float32(primary_roughness_std_rad),
                            np.float32(floc_roughness_std_rad),
                            np.float32(R_REAL),
                            np.float32(RAY_OFFSET),
                            np.int32(VIS_SIZE),
                            np.float32(VISUAL_SCALE),
                            angles_chunk,
                            np.int32(sz),
                            particle_cdf_dev,
                            particle_is_floc_dev,
                            single_scattering_albedo_dev,
                            angle_cdf_dev,
                            theta_dev,
                            np.int32(len(particle_cdf_table_np)),
                            np.int32(len(theta_table_np)),
                            heatmap_dev,
                            exit_dir_chunk_dev,
                            exit_x_chunk_dev,
                            exit_y_chunk_dev,
                            rpl_chunk_dev,
                            scatter_count_chunk_dev,
                            floc_event_count_chunk_dev,
                            last_event_was_floc_chunk_dev,
                            last_scatter_bin_chunk_dev,
                            extinction_count_chunk_dev,
                            absorbed_chunk_dev,
                            np.uint32(seed0),
                            np.uint32(seed1),
                            np.uint32(seed2)
                        ))
                    cp.cuda.Stream.null.synchronize()
                    launched = True
                except cp.cuda.memory.OutOfMemoryError:
                    attempt_chunk = max(min_chunk, attempt_chunk // 2)
                    if attempt_chunk < 2: raise
                    end = min(N, start + attempt_chunk)
                    sz = end - start
                    cp._default_memory_pool.free_all_blocks()
                    angles_chunk = cp.asarray(angles_init_np[start:end], dtype=cp.float64)
                    # Reinitialise resized chunk outputs to invalid sentinels.
                    exit_dir_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
                    exit_x_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
                    exit_y_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
                    rpl_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
                    scatter_count_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)
                    floc_event_count_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)
                    last_event_was_floc_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)
                    last_scatter_bin_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)
                    extinction_count_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)
                    absorbed_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)
                    blocks = (sz + threads_per_block - 1) // threads_per_block
                    estimated_chunk = attempt_chunk

            # Write chunk to HDF5
            dset_exit_dir[start:end] = cp.asnumpy(exit_dir_chunk_dev)
            dset_exit_x[start:end] = cp.asnumpy(exit_x_chunk_dev)
            dset_exit_y[start:end] = cp.asnumpy(exit_y_chunk_dev)
            dset_rpl[start:end] = cp.asnumpy(rpl_chunk_dev)
            dset_scatter_count[start:end] = cp.asnumpy(scatter_count_chunk_dev)
            dset_floc_event_count[start:end] = cp.asnumpy(floc_event_count_chunk_dev)
            dset_last_event_was_floc[start:end] = cp.asnumpy(last_event_was_floc_chunk_dev)
            dset_last_scatter_bin[start:end] = cp.asnumpy(last_scatter_bin_chunk_dev)
            dset_extinction_count[start:end] = cp.asnumpy(extinction_count_chunk_dev)
            dset_absorbed[start:end] = cp.asnumpy(absorbed_chunk_dev)

            # Free GPU memory
            del angles_chunk, exit_dir_chunk_dev, exit_x_chunk_dev, exit_y_chunk_dev, rpl_chunk_dev, scatter_count_chunk_dev, floc_event_count_chunk_dev, last_event_was_floc_chunk_dev, last_scatter_bin_chunk_dev, extinction_count_chunk_dev, absorbed_chunk_dev
            cp._default_memory_pool.free_all_blocks()

    heatmap = cp.asnumpy(heatmap_dev).reshape((VIS_SIZE, VIS_SIZE))
    return heatmap



def save_detector_transport_breakdown(
    wl_nm,
    exit_x,
    exit_y,
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
    """
    CLARITAS_57 diagnostic only.

    Break down radial-detector hits by full-transport history. This is not a
    single-particle or single-floc backscatter diagnostic; it describes rays
    after propagation through the whole cell.
    """
    exit_x = np.asarray(exit_x)
    exit_y = np.asarray(exit_y)
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
        np.isfinite(exit_dirs) &
        (scatter_count >= 0)
    )

    exit_position_angle_deg = np.full_like(exit_x, np.nan, dtype=np.float64)
    exit_position_angle_deg[valid_exit_mask] = (
        np.rad2deg(np.arctan2(exit_x[valid_exit_mask], exit_y[valid_exit_mask])) + 360.0
    ) % 360.0

    rows = []
    n_total_rays = int(len(scatter_count))
    n_valid_exit = int(np.sum(valid_exit_mask))
    n_absorbed_or_invalid = n_total_rays - n_valid_exit

    for centre in detector_angles_deg:
        delta = circular_angle_difference_deg(exit_position_angle_deg, centre)
        hit_mask_raw = valid_exit_mask & (np.abs(delta) <= detector_acceptance_deg)

        # Match the production detector-hit rule exactly: ballistic rays are
        # suppressed only for the backscatter semicircle bins.
        if centre >= 90:
            hit_mask = hit_mask_raw & (scatter_count > 0)
        else:
            hit_mask = hit_mask_raw

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


def save_detector_last_scatter_bin_contribution(
    wl_nm,
    exit_x,
    exit_y,
    scatter_count,
    last_scatter_bin,
    detector_angles_deg,
    detector_acceptance_deg,
    outdir
):
    """
    CLARITAS_59 diagnostic only.

    Attribute detector hits to the final scattering bin before exit. This is a
    full-transport radial-detector diagnostic, not a single-event backscatter
    proxy: the ray may have undergone many earlier scatterings before the last
    event. Ballistic rays have last_scatter_bin = -1 and are not assigned to a
    particle/floc bin.
    """
    exit_x = np.asarray(exit_x)
    exit_y = np.asarray(exit_y)
    scatter_count = np.asarray(scatter_count)
    last_scatter_bin = np.asarray(last_scatter_bin)

    valid_exit_mask = (
        np.isfinite(exit_x) &
        np.isfinite(exit_y) &
        (scatter_count >= 0)
    )

    exit_position_angle_deg = np.full_like(exit_x, np.nan, dtype=np.float64)
    exit_position_angle_deg[valid_exit_mask] = (
        np.rad2deg(np.arctan2(exit_x[valid_exit_mask], exit_y[valid_exit_mask])) + 360.0
    ) % 360.0

    n_bins = len(particle_diameter_m)
    n_det = len(detector_angles_deg)

    count_matrix = np.zeros((n_bins, n_det), dtype=np.int64)
    detector_hit_counts = np.zeros(n_det, dtype=np.int64)

    for j, centre in enumerate(detector_angles_deg):
        delta = circular_angle_difference_deg(exit_position_angle_deg, centre)
        hit_mask_raw = valid_exit_mask & (np.abs(delta) <= detector_acceptance_deg)

        # Match production detector-hit rule: ballistic rays are suppressed only
        # for detector centres in the backscatter semicircle.
        if centre >= 90:
            hit_mask = hit_mask_raw & (scatter_count > 0)
        else:
            hit_mask = hit_mask_raw

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

# ================= SIMULATION (main loop) =================
from tqdm import tqdm
import h5py
import hashlib

bulk_profiles = []


detector_centers_rad = np.deg2rad(detector_angles)
detector_accept = detector_acceptance_deg
detector_hit_counts = {}

# ================= DETECTOR GEOMETRY DIAGNOSTICS =================
# These diagnostics do not change the ray tracing. They re-bin the same exit data
# using position angle, outgoing direction angle, and scatter-count categories.
MULTIPLY_SCATTERED_MIN_COUNT = 6

def circular_angle_difference_deg(angle_deg, centre_deg):
    return ((angle_deg - centre_deg + 180.0) % 360.0) - 180.0

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

    counts = []
    for centre in detector_angles_deg:
        delta = circular_angle_difference_deg(ray_angle_deg, centre)
        hit_mask = valid_mask & (np.abs(delta) <= detector_acceptance_deg)
        counts.append(np.sum(hit_mask))

    counts = np.asarray(counts, dtype=np.float64)
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
    """
    CLARITAS_53 diagnostic only.

    Estimate which floc bins are optically capable of contributing to each
    detector, using the current per-bin phase CDF and optical event weights.

    This is not a transport replacement and does not change ray tracing.
    It answers the narrower question:

        Which effective floc bins dominate mu_t, mu_s, and the single-event
        angular probability into each detector acceptance window?

    Outputs:
      - floc_bin_optical_detector_proxy_conc_<conc>_<wl>nm.csv
      - floc_detector_proxy_matrix_conc_<conc>_<wl>nm.csv
    """
    theta_deg_values = np.rad2deg(theta_rad_values)
    n_bins = len(particle_diameter_m)

    # Recover a discrete angular PDF from each CDF row. This deliberately uses
    # the actual transport CDF after whatever angular measure is currently active.
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
    """
    CLARITAS_58 diagnostic only.

    Audit where the optical extinction budget comes from.  This does not
    alter ray tracing or any phase function.  It summarises primary vs floc
    mass, scattering, absorption, extinction, and event-probability budgets,
    then writes a per-floc effective-bin table sorted by mu_t contribution.
    """
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

    # Detector proxy from CLARITAS_53, keyed by effective_bin_index.  This is
    # optional and diagnostic-only; leave NaN if the proxy table is unavailable.
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
    exit_dirs,
    scatter_count,
    detector_angles_deg,
    detector_acceptance_deg,
    outdir,
    multiply_scattered_min_count=6
):
    exit_x = np.asarray(exit_x)
    exit_y = np.asarray(exit_y)
    exit_dirs = np.asarray(exit_dirs)
    scatter_count = np.asarray(scatter_count)

    valid_exit_mask = (
        np.isfinite(exit_x) &
        np.isfinite(exit_y) &
        np.isfinite(exit_dirs) &
        (scatter_count >= 0)
    )

    exit_position_angle_deg = (
        np.rad2deg(np.arctan2(exit_x, exit_y)) + 360.0
    ) % 360.0

    # exit_dirs is atan2(vy, vx). Direction angle relative to +y/incident beam is atan2(vx, vy).
    exit_direction_angle_deg = (
        np.rad2deg(np.arctan2(np.cos(exit_dirs), np.sin(exit_dirs))) + 360.0
    ) % 360.0

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

def sample_laser_angles(N, half_angle_deg=2.0):
    angles = np.random.uniform(
        -np.deg2rad(half_angle_deg),
        np.deg2rad(half_angle_deg),
        N
    )
    return angles.astype(np.float64)

for wl_idx, wl in enumerate(wavelengths):
    print(f"--- Wavelength {int(wl*1e9)} nm ---")
    angles_init = sample_beta_angles(N_RAYS, alpha1, alpha2)  # host numpy float64
    #angles_init = sample_laser_angles(N_RAYS, half_angle_deg=2.0)
    particle_cdf_table_np = np.array(particle_event_cdf, dtype=np.float64)
    particle_diameter_table_np = np.array(particle_diameter_m, dtype=np.float64)
    particle_is_floc_table_np = np.array(particle_is_floc, dtype=np.float64)
    single_scattering_albedo_table_np = np.array(
        single_scattering_albedo_by_bin,
        dtype=np.float64
    )
    angle_cdf_table_np = np.array(cdf_profiles[wl_idx], dtype=np.float64)
    theta_table_np = np.array(theta_rad, dtype=np.float64)

    # CLARITAS_53: pre-transport floc-bin optical/detector proxy diagnostics.
    # This does not alter ray tracing; it uses the actual transport CDF table.
    floc_proxy_summary_df, floc_proxy_matrix_df = save_floc_bin_detector_proxy_diagnostics(
        int(wl * 1e9),
        angle_cdf_table_np,
        theta_table_np,
        detector_angles,
        detector_acceptance_deg,
        OUTDIR
    )

    # CLARITAS_58: optical budget audit.  This does not alter ray tracing.
    save_optical_budget_diagnostics(
        int(wl * 1e9),
        floc_proxy_summary_df,
        OUTDIR
    )

    hdf5_file = os.path.join(OUTDIR, f"ray_exits_{int(wl*1e9)}nm.h5")

    t0 = time.time()
    # Call GPU tracer (adaptive chunking, HDF5 streaming)

    print("DEBUG floc_event_probability:", floc_event_probability)
    print("DEBUG particle_is_floc count:", np.sum(particle_is_floc))

    heatmap = trace_rays_gpu(angles_init, 
                             PRIMARY_ROUGHNESS_STD_RAD,
                             FLOC_ROUGHNESS_STD_RAD,
                             reflection_path_length,
                             n_medium, n_external,
                             R_REAL, RAY_OFFSET, VIS_SIZE, VISUAL_SCALE,
                             particle_cdf_table_np,
                             particle_is_floc_table_np, single_scattering_albedo_table_np, angle_cdf_table_np, theta_table_np,
                             hdf5_file=hdf5_file)
    t1 = time.time()
    print(f"[INFO] trace_rays_gpu completed in {t1-t0:.2f} s")

    # Load exit data from HDF5 for plotting and detector binning
    with h5py.File(hdf5_file, "r") as f:
        exit_x = f["exit_x"][:]
        exit_y = f["exit_y"][:]
        exit_dirs = f["exit_dir"][:]
        exit_rpl = f["exit_rpl"][:]
        scatter_count = f["scatter_count"][:]
        floc_event_count = f["floc_event_count"][:] if "floc_event_count" in f else np.full_like(scatter_count, -1)
        last_event_was_floc = f["last_event_was_floc"][:] if "last_event_was_floc" in f else np.full_like(scatter_count, -1)
        last_scatter_bin = f["last_scatter_bin"][:] if "last_scatter_bin" in f else np.full_like(scatter_count, -1)
        extinction_count = f["extinction_count"][:] if "extinction_count" in f else np.full_like(scatter_count, -1)
        absorbed_flag = f["absorbed"][:] if "absorbed" in f else np.full_like(scatter_count, -1)

    valid_exit_mask = (
        np.isfinite(exit_x) &
        np.isfinite(exit_y) &
        np.isfinite(exit_dirs) &
        (scatter_count >= 0)
    )

    n_valid_exits = int(np.sum(valid_exit_mask))
    n_absorbed_or_invalid = int(len(scatter_count) - n_valid_exits)
    absorbed_or_invalid_fraction = (
        n_absorbed_or_invalid / len(scatter_count)
        if len(scatter_count) > 0 else 0.0
    )

    if n_valid_exits > 0:
        print(f"Mean scatter count: {np.mean(scatter_count[valid_exit_mask]):.3e}")
        print(f"Max scatter count: {np.max(scatter_count[valid_exit_mask])}")
    else:
        print("Mean scatter count: nan (no valid exits)")
        print("Max scatter count: nan (no valid exits)")

    print(f"Valid exit rays: {n_valid_exits} / {len(scatter_count)}")
    print(f"Absorbed/invalid ray fraction: {absorbed_or_invalid_fraction:.6f}")

    detector_geometry_df, detector_geometry_summary_df = save_detector_geometry_diagnostics(
        int(wl * 1e9),
        exit_x,
        exit_y,
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
        scatter_count,
        last_scatter_bin,
        detector_angles,
        detector_acceptance_deg,
        OUTDIR
    )

    # Save heatmap (physically scaled, with boundary overlay)
    masked = ma.masked_where(heatmap == 0, heatmap)

    # ---- Robust LogNorm limits ----
    if masked.count() > 0:
        vmin = masked.min()
        vmax = masked.max()
    else:
        # fallback to avoid crashing on fully empty heatmaps
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
#        norm=LogNorm(vmin=1, vmax=np.max(heatmap)),
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=extent_mm,
        interpolation='gaussian'
    )

    # ---- Physical boundary overlay ----
    circle = plt.Circle(
        (0, 0),                 # center (mm)
        R_REAL * 1000,          # radius (mm)
        color='red',
        linewidth=1.5,
        fill=False,
        linestyle='--'
    )
    plt.gca().add_patch(circle)

    # ---- Axis formatting ----
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

    # Bulk angular scattering histogram. Only real exits are included.
    exit_dirs_valid = exit_dirs[valid_exit_mask]

    if exit_dirs_valid.size > 0:
        exit_dirs_deg = np.rad2deg(exit_dirs_valid)
        hist_bulk, _ = np.histogram(exit_dirs_deg, bins=theta_deg, density=True)
    else:
        hist_bulk = np.zeros(len(theta_deg) - 1, dtype=np.float64)

    bulk_profiles.append(hist_bulk)

    # Compute exit position angles. Invalid/absorbed rays remain NaN.
    exit_pos_angles = np.full_like(exit_x, np.nan, dtype=np.float64)
    exit_pos_angles[valid_exit_mask] = (
        np.rad2deg(np.arctan2(exit_x[valid_exit_mask], exit_y[valid_exit_mask])) + 360.0
    ) % 360.0

    # Save exit positions CSV
    df_exits = pd.DataFrame({
        "exit_x_m": exit_x,
        "exit_y_m": exit_y,
        "exit_rpl_m": exit_rpl,
        "scatter_count": scatter_count,
        "floc_event_count": floc_event_count,
        "last_event_was_floc": last_event_was_floc,
        "last_scatter_bin": last_scatter_bin,
        "extinction_count": extinction_count,
        "absorbed_flag": absorbed_flag,
        "is_valid_exit": valid_exit_mask,
        "is_absorbed_or_invalid": ~valid_exit_mask,
        "is_ballistic": valid_exit_mask & (scatter_count == 0),
        "exit_pos_angle_deg": exit_pos_angles,
        "exit_dir_deg": (np.rad2deg(exit_dirs) + 360) % 360
    })
    exits_csv_path = os.path.join(OUTDIR, f"exit_points_{int(wl*1e9)}nm.csv")
    df_exits.to_csv(exits_csv_path, index=False)
    print(f"✅ Saved {exits_csv_path}")

    # Detector binning (vectorized)
    #
    # Physical detector rule:
    # - Forward/side detector hits from ballistic rays are allowed.
    # - Backscatter-region detector bins, defined here as detector centre >= 90 deg,
    #   are only allowed to count rays that have undergone at least one real
    #   scattering/reflection interaction in the kernel.
    #
    # This prevents non-real backscatter caused by finite beam waist/divergence, boundary
    # intersection, or detector acceptance overlap, while preserving circumference-detector
    # geometry.
    in_detector_semicircle = (
        valid_exit_mask &
        (exit_pos_angles >= 0) &
        (exit_pos_angles <= 180)
    )
    pos_angles_semicircle = exit_pos_angles[in_detector_semicircle]
    interacted_semicircle = (scatter_count[in_detector_semicircle] > 0)

    if pos_angles_semicircle.size == 0:
        counts = np.zeros_like(detector_angles, dtype=int)
    else:
        diffs = np.abs(pos_angles_semicircle[:, None] - detector_angles[None, :])
        hits_mask = diffs <= detector_accept

        backscatter_detector_bin = detector_angles >= 90
        ballistic_ray = ~interacted_semicircle

        # Suppress only ballistic contributions to backscatter detector bins.
        hits_mask[ballistic_ray[:, None] & backscatter_detector_bin[None, :]] = False

        counts = hits_mask.sum(axis=0).astype(int)

    detector_hit_counts[int(wl*1e9)] = counts

# ---------------- Final outputs: detector CSV and plots ----------------
df_det = pd.DataFrame({"Detector_deg": detector_angles})
for wl_nm, counts in detector_hit_counts.items():
    df_det[f"H_{wl_nm}nm"] = counts
det_csv_path = os.path.join(OUTDIR, "detector_hits.csv")
df_det.to_csv(det_csv_path, index=False)
print(f"✅ Saved {det_csv_path}")

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

# Bulk angular scattering plots
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


# ---------------- Histogram parameters ----------------
n_bins = 1000  # number of bins
hist_color = "skyblue"
hist_edge = "black"
hist_alpha = 0.7

# Compute histogram using only valid exit path lengths.
# Absorbed rays are stored as NaN in CLARITAS_42b/42c and must not be
# included in path-length diagnostics.
exit_rpl_valid = exit_rpl[np.isfinite(exit_rpl)]

if len(exit_rpl_valid) > 0:
    hist_counts, bin_edges = np.histogram(exit_rpl_valid, bins=n_bins)
else:
    hist_counts = np.zeros(n_bins, dtype=np.int64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

# Replace zeros with 1 for log-scale plotting
hist_counts_safe = np.where(hist_counts == 0, 1, hist_counts)

# Compute bin centers
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# ---------------- Plot histogram (log scale) ----------------
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

# ---------------- Save histogram data to CSV ----------------
df_hist = pd.DataFrame({
    "bin_center_m": bin_centers,
    "counts": hist_counts,          # raw counts
    "counts_for_log_plot": hist_counts_safe  # counts used for plotting
})
hist_csv_path = "./exit_rpl_histogram.csv"
df_hist.to_csv(hist_csv_path, index=False)
print(f"✅ Saved histogram data to {hist_csv_path}")

# ----------------- Backscatter Fraction ---------------------
# ================= DETECTOR-BASED BACKSCATTER (INTEGRATED) =================

angles = detector_angles  # already defined
counts = detector_hit_counts[622]  # or loop over wavelengths if needed

# Convert to numpy arrays
angles = np.array(angles)
counts = np.array(counts)

# Compute angular bin width (assumes uniform spacing)
dtheta = np.deg2rad(angles[1] - angles[0])  # radians

# Total signal (0-180°)
total = np.sum(counts * dtheta)

# Backscatter region (90-180°)
mask_back = angles >= 90
backscatter = np.sum(counts[mask_back] * dtheta)

backscatter_fraction = backscatter / total if total > 0 else 0.0

print("\n=========== DETECTOR-INTEGRATED BACKSCATTER ===========")
print(f"Backscatter fraction (integrated): {backscatter_fraction:.6f}")
print("======================================================")

# ================= OPTICAL OBJECT DIAGNOSTICS =================

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

# ================= PSD COMPARISON OUTPUTS =================

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
    # One-dimensional legacy-style floc-bin summary.  Do not put the full
    # source-band x floc-bin matrices directly into this DataFrame; pandas
    # requires each column here to be one-dimensional.  The full matrices are
    # saved separately below.
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

    # Full 2-D kernel diagnostics in long form: one row per source-band/floc-bin
    # cell.  This preserves the new many-to-many pooling information without
    # breaking the legacy one-dimensional summary output.
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

# Use particle_is_floc as the authoritative floc mask.
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
# ================= FLOC MODEL DIAGNOSTIC OUTPUTS =================

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

# ================= REFRACTIVE INDEX DIAGNOSTIC OUTPUTS =================

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

# ================= FLOC BIN CONCENTRATION DIAGNOSTICS =================

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

# ================= FLOC EVENT-WEIGHT BY BAND DIAGNOSTICS =================

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
