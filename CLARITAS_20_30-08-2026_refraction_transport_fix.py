#!/usr/bin/env python3
# CLARITAS_20_30-08-2026_refraction_transport_fix.py
#
# Refraction-only particle-direction fork of CLARITAS_18_17-06-2026.
#
# PURPOSE
# -------
# This version is intentionally a controlled comparison against the original
# CLARITAS transport model:
#
#   PRESERVED
#   - primary sediment PSDs and PSD weighting modes
#   - mass-concentration -> physical particle number density conversion
#   - Mie Qsca cross-sections ONLY for the original ray-particle interaction rate
#   - mu_s and n_i*sigma_i particle-event weighting
#   - source beam, circular sample geometry, CUDA stepping/chunking, heatmaps,
#     HDF5 ray outputs, detector binning and downstream plots
#
#   REMOVED
#   - all flocculation / pooled-floc code and diagnostics
#   - porous/effective floc refractive-index mixing
#   - Mie S1/S2 angular scattering profiles and cached angular CDFs
#   - particle roughness angular jitter
#   - empirical primary/floc reflection branches
#
#   REPLACED
#   - every particle interaction changes direction by geometrical refraction through
#     a homogeneous spherical particle using Snell's law at entry and exit.
#
# IMPORTANT PHYSICS NOTE
# ----------------------
# Mie theory remains ONLY in efficiencies_mx() so that this experimental fork keeps
# the original interaction probabilities/cross-sections. It does not determine the
# interaction direction. The angular physics is pure geometric refraction.
#
# The sphere interaction samples a point uniformly over the projected particle disk:
#     rho = b/R = sqrt(U),  U~Uniform(0,1)
# so the incidence angle is i = asin(rho). Snell's law gives
#     n_medium sin(i) = n_particle sin(r)
# and the direct transmitted ray has total deflection
#     delta = 2 * (i - r).
# A random scattering-plane azimuth is then projected into the existing 2-D transport
# plane. No Fresnel reflection, absorption, diffraction or interference is included.

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
from tqdm import tqdm

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

mass_concentration_g_per_L = 0.01
# Particle interactions are sampled as a continuous exponential free path from mu_s.
# There is deliberately no fixed per-step interaction probability in this version.


# ============================ PRIMARY PARTICLE MODEL ONLY ============================
# There is deliberately no floc state in this fork. The selected source PSD is used
# directly by transport.

#particle_diameter_m = null_diameter
#particle_weights = null_weights
#particle_density_kg_per_m3 = 1.0

#particle_diameter_m = loess_diameter
#particle_weights = loess_weights
#particle_density_kg_per_m3 = 2600.0  # loess density

particle_diameter_m = kaolin_diameter
particle_weights = kaolin_weights
particle_density_kg_per_m3 = 2600.0  # kaolin density

particle_diameter_m = np.asarray(particle_diameter_m, dtype=np.float64)
particle_weights = np.asarray(particle_weights, dtype=np.float64)
particle_weights /= np.sum(particle_weights)
particle_radius_m = particle_diameter_m / 2.0
particle_density_by_bin_kg_per_m3 = np.full_like(
    particle_diameter_m,
    particle_density_kg_per_m3,
    dtype=np.float64
)

# Keep aliases used by some of the legacy diagnostics/output conventions.
primary_particle_diameter_m = particle_diameter_m.copy()
primary_particle_weights = particle_weights.copy()
primary_particle_density_by_bin_kg_per_m3 = particle_density_by_bin_kg_per_m3.copy()

# PSD weight interpretation. This is unchanged from the supplied version.
PSD_WEIGHT_MODE = "mass_fraction"  # options: "mass_fraction", "number_fraction"
#PSD_WEIGHT_MODE = "number_fraction"

# Optical indices for the homogeneous particle and suspending medium.
n_particle = 1.59
n_medium = 1.33

# Direction model. The current implementation is a 3-D spherical encounter projected
# into the existing 2-D CLARITAS transport plane.
REFRACTION_DIRECTION_MODEL = "homogeneous_sphere_snell_3d_projected"
REFRACTION_PROFILE_SAMPLES = 500_000
REFRACTION_PROFILE_SEED = 24681357

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
#RAY_OFFSET = 0.05  # Ray initial y-offset (m) TARDIIS
RAY_OFFSET = 0.005  # Ray initial y-offset (m)
#STEP_SIZE = 1.0e-6  # integration step size (m)
#STEP_SIZE = 1.0e-7  # integration step size (m)
#VISUAL_SCALE = 100.0 TARDIIS
#VIS_SIZE = 2048      # Heatmap resolution TARDIIS
VISUAL_SCALE = 1.0
VIS_SIZE = 4096      # Heatmap resolution
N_RAYS = 1_000_00  # number of rays to simulate
MAX_ITERATIONS = int(100e6)  # maximum number of steps per ray to avoid infinite loops

#### Kernel parameters for a cuvette ####
#R_REAL = 0.005    # Sample radius (m)
#RAY_OFFSET = 0.03  # Ray initial y-offset (m)
#STEP_SIZE = 1.0e-6  # integration step size (m)
#VISUAL_SCALE = 100.0
#VIS_SIZE = 1024      # Heatmap resolution
#N_RAYS = 1_000_000  # number of rays to simulate
#MAX_ITERATIONS = int(1e12)  # maximum number of steps per ray to avoid infinite loops

###### OUTPUT DIRECTORY #######
OUTDIR = "."
os.makedirs(OUTDIR, exist_ok=True)

# ============================ PHYSICAL PSD / INTERACTION-RATE SETUP ============================
# Unit conversion: 1 g/L == 1 kg/m^3.
mass_concentration_kg_per_m3 = mass_concentration_g_per_L

particle_volumes = (4.0 / 3.0) * np.pi * particle_radius_m**3
particle_masses = particle_volumes * particle_density_by_bin_kg_per_m3

if PSD_WEIGHT_MODE == "mass_fraction":
    particle_number_density_by_bin = (
        mass_concentration_kg_per_m3 *
        particle_weights /
        particle_masses
    )
elif PSD_WEIGHT_MODE == "number_fraction":
    number_weights = particle_weights / np.sum(particle_weights)
    average_particle_mass_from_number_distribution = np.sum(
        number_weights * particle_masses
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
average_particle_separation_m = (
    n_particles_per_m3 ** (-1.0 / 3.0)
    if n_particles_per_m3 > 0.0 else np.inf
)
average_particle_mass = (
    mass_concentration_kg_per_m3 / n_particles_per_m3
    if n_particles_per_m3 > 0 else 0.0
)
pm = particle_masses
particle_mass = average_particle_mass

# ============================ PRESERVED INTERACTION PROBABILITIES ============================
# IMPORTANT: Mie theory is retained ONLY here, to preserve the supplied model's
# interaction cross-sections and particle-event probabilities. It is not used to
# choose a new ray direction anywhere in this file.
scatter_probability_wavelength = wavelengths[0]
relative_refractive_index_by_bin = np.full_like(
    particle_diameter_m,
    n_particle / n_medium,
    dtype=np.float64
)
particle_refractive_index_by_bin = np.full_like(
    particle_diameter_m,
    n_particle,
    dtype=np.float64
)

sigma_s = []
for r, m_rel in zip(particle_radius_m, relative_refractive_index_by_bin):
    x_mie = 2.0 * np.pi * n_medium * r / scatter_probability_wavelength
    qext, qsca, qback, g = miepython.efficiencies_mx(m_rel, x_mie)
    sigma_s.append(qsca * np.pi * r**2)

sigma_s = np.asarray(sigma_s, dtype=np.float64)
mu_s_by_bin = particle_number_density_by_bin * sigma_s
mu_s = np.sum(mu_s_by_bin)

if mu_s < 0.0:
    raise ValueError("mu_s must not be negative")

# Continuous-flight transport:
#   P(no interaction over path length L) = exp(-mu_s * L)
# and the distance to the next interaction is sampled exactly as
#   s = -ln(U) / mu_s.
#
# This gives the required transparent limit: as concentration -> 0,
# mu_s -> 0, mean free path -> infinity, and the interaction probability
# across the finite sample tends to zero.
mean_free_path_m = (1.0 / mu_s) if mu_s > 0.0 else np.inf
diameter_optical_depth = mu_s * (2.0 * R_REAL)
diameter_interaction_probability = -np.expm1(-diameter_optical_depth)

particle_event_weights = np.zeros_like(mu_s_by_bin, dtype=np.float64)
if mu_s > 0.0:
    particle_event_weights = mu_s_by_bin / mu_s

particle_event_cdf = np.cumsum(particle_event_weights)
if particle_event_cdf.size > 0 and particle_event_cdf[-1] > 0:
    particle_event_cdf /= particle_event_cdf[-1]

print("\n=========== CLARITAS REFRACTION-ONLY FORK ===========")
print(f"Direction model: {REFRACTION_DIRECTION_MODEL}")
print("Mie angular scattering: DISABLED")
print("Mie Qsca interaction-rate calculation: PRESERVED")
print("Flocculation: REMOVED")
print("Empirical particle reflection: REMOVED")
print("Roughness angular jitter: REMOVED")
print(f"PSD_WEIGHT_MODE: {PSD_WEIGHT_MODE}")
print(f"particle diameter range: {np.min(particle_diameter_m)*1e6:.3f} - {np.max(particle_diameter_m)*1e6:.3f} um")
print(f"particle density: {particle_density_kg_per_m3:.3e} kg/m^3")
print(f"particle refractive index: {n_particle:.5f}")
print(f"medium refractive index: {n_medium:.5f}")
print(f"n_particles_per_m3: {n_particles_per_m3:.3e}")
print(f"average_particle_mass: {average_particle_mass:.3e} kg")
print(f"average_particle_separation_m: {average_particle_separation_m:.3e}")
print(f"mu_s: {mu_s:.3e} 1/m")
print(f"mean_free_path_m: {mean_free_path_m:.3e}")
print(f"diameter_optical_depth_mu_s_2R: {diameter_optical_depth:.3e}")
print(f"single_diameter_path_interaction_probability: {diameter_interaction_probability:.6e}")
print(f"particle_event_weights_sum: {np.sum(particle_event_weights):.6f}")
if mu_s > 0.0:
    print(f"dominant_event_diameter_um: {particle_diameter_m[np.argmax(particle_event_weights)]*1e6:.3f}")
else:
    print("dominant_event_diameter_um: n/a (mu_s = 0)")
print("Interaction transport: continuous exponential free-path sampling")
print("=====================================================\n")

# ============================ REFRACTION ANGULAR DIAGNOSTICS ============================
def sphere_refraction_deflection_rad(rho, n_outside, n_inside):
    """Direct transmitted-ray deflection for a homogeneous sphere.

    rho is normalized impact parameter b/R in [0,1]. For n_inside > n_outside,
    every geometrical incident ray has a transmitted direct path. The returned
    deflection is the magnitude delta = 2(i-r).
    """
    rho = np.clip(np.asarray(rho, dtype=np.float64), 0.0, 1.0)
    incidence = np.arcsin(rho)
    sin_refracted = (n_outside / n_inside) * rho
    valid = np.abs(sin_refracted) <= 1.0
    refracted = np.full_like(incidence, np.nan)
    refracted[valid] = np.arcsin(sin_refracted[valid])
    delta = np.full_like(incidence, np.nan)
    delta[valid] = 2.0 * (incidence[valid] - refracted[valid])
    return delta

# Curve showing how one sphere redirects a ray as a function of impact parameter.
rho_curve = np.linspace(0.0, 0.999999, 5000)
delta_curve_rad = sphere_refraction_deflection_rad(rho_curve, n_medium, n_particle)
delta_curve_deg = np.rad2deg(delta_curve_rad)

pd.DataFrame({
    "normalized_impact_parameter_b_over_R": rho_curve,
    "deflection_deg": delta_curve_deg
}).to_csv(os.path.join(OUTDIR, "refraction_deflection_vs_impact_parameter.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.plot(rho_curve, delta_curve_deg)
plt.xlabel("Normalized impact parameter b/R")
plt.ylabel("Net refractive deflection (deg)")
plt.title("Homogeneous Sphere: Snell-Law Deflection vs Impact Parameter")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "refraction_deflection_vs_impact_parameter.png"), dpi=200)
plt.close()

# Generate a deterministic reference distribution matching the CUDA direction model:
# point uniform over projected disk (rho=sqrt(U)), random scattering-plane azimuth,
# then project the 3-D deflection into the existing 2-D transport plane.
rng_refraction = np.random.default_rng(REFRACTION_PROFILE_SEED)
rho_profile = np.sqrt(rng_refraction.random(REFRACTION_PROFILE_SAMPLES))
delta_profile = sphere_refraction_deflection_rad(rho_profile, n_medium, n_particle)
phi_profile = rng_refraction.random(REFRACTION_PROFILE_SAMPLES) * 2.0 * np.pi
projected_turn = np.arctan2(
    np.sin(delta_profile) * np.cos(phi_profile),
    np.cos(delta_profile)
)
projected_turn_abs_deg = np.abs(np.rad2deg(projected_turn))

refraction_profile_density, refraction_profile_edges = np.histogram(
    projected_turn_abs_deg,
    bins=theta_deg,
    density=True
)
refraction_profile_centers = 0.5 * (
    refraction_profile_edges[:-1] + refraction_profile_edges[1:]
)

pd.DataFrame({
    "Angle_deg": refraction_profile_centers,
    "Probability_density": refraction_profile_density
}).to_csv(os.path.join(OUTDIR, "single_event_refraction_profile.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.plot(refraction_profile_centers, refraction_profile_density)
plt.xlabel("Absolute projected deflection angle (deg)")
plt.ylabel("Probability density")
plt.title("Single-Interaction Angular Profile: Spherical Refraction Only")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "single_event_refraction_profile.png"), dpi=200)
plt.close()

print("✅ Saved refraction_deflection_vs_impact_parameter.csv/png")
print("✅ Saved single_event_refraction_profile.csv/png")

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


# ================= CUDA KERNEL: REFRACTION-ONLY PARTICLE DIRECTION =================
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

__device__ float sphere_refraction_projected_turn(
    unsigned int* state,
    const float n_medium,
    const float n_particle)
{
    // Sample uniformly over the projected disk of a 3-D sphere.
    // Area CDF is P(<rho)=rho^2, therefore rho=sqrt(U).
    float rho = sqrtf(fminf(fmaxf(rnd_uniform(state), 0.0f), 0.99999994f));

    // Incidence at the spherical surface: sin(i)=b/R=rho.
    float incidence = asinf(rho);

    // Snell: n_medium sin(i) = n_particle sin(r).
    float sin_refracted = (n_medium / n_particle) * rho;

    // For the current CLARITAS indices (n_particle > n_medium) this is always valid.
    // If a future configuration creates no transmitted solution, leave the ray
    // direction unchanged rather than introducing a reflection mechanism into this
    // deliberately refraction-only fork.
    if (fabsf(sin_refracted) >= 1.0f) {
        return 0.0f;
    }

    float refracted = asinf(sin_refracted);

    // Direct transmitted ray through a sphere, two Snell refractions.
    float delta = 2.0f * (incidence - refracted);

    // Random orientation of the ray/sphere scattering plane about the incident ray.
    float phi = rnd_uniform(state) * 2.0f * 3.1415927f;

    // Project the 3-D outgoing direction into the existing CLARITAS 2-D plane.
    return atan2f(
        sinf(delta) * cosf(phi),
        cosf(delta)
    );
}

__global__ void trace_kernel(
    const float MAX_ITERATIONS,
    const float MU_S,
    const float N_MEDIUM,
    const float R_REAL,
    const float R_OFF,
    const int VIS_SIZE,
    const float VISUAL_SCALE,
    const double* angles_init,
    const int N_rays,
    const double* particle_cdf_table,
    const double* particle_refractive_index_table,
    const int n_particles,
    float* heatmap_flat,
    float* exit_dir_out,
    float* exit_x_out,
    float* exit_y_out,
    float* ray_path_length_out,
    int* interaction_count_out,
    unsigned int seed0,
    unsigned int seed1)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N_rays) return;

    unsigned int state = seed0 + (unsigned int)tid * 74729u + 13u;
    unsigned int stateREFR = seed1 + (unsigned int)tid * 104729u + 29u;

    float beam_sigma = 0.00001f;

    float u1 = fmaxf(rnd_uniform(&state), 1.0e-12f);
    float u2 = rnd_uniform(&state);
    float gaussian =
        sqrtf(-2.0f * logf(u1)) *
        cosf(2.0f * 3.1415927f * u2);

    float x0 = beam_sigma * gaussian;
    float y0 = -(R_REAL + R_OFF);

    double angle_d = angles_init[tid];
    float angle_init = (float)angle_d;

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

    const int max_steps = MAX_ITERATIONS;
    int step_count = 0;
    int absorbed = 0;
    float rpl = 0.0f;
    int interaction_count = 0;

    while (x * x + y * y <= R_REAL * R_REAL + 1.0e-8f) {

        // Distance from the current interior point to the forward intersection
        // with the circular sample boundary.
        float rv = x * vx + y * vy;
        float rr_minus_R2 = x * x + y * y - R_REAL * R_REAL;
        float boundary_disc = rv * rv - rr_minus_R2;

        if (boundary_disc < 0.0f) {
            absorbed = 1;
            break;
        }

        float distance_to_boundary = -rv + sqrtf(fmaxf(boundary_disc, 0.0f));

        // Numerical guard for a point already at the outward boundary.
        if (distance_to_boundary <= 1.0e-9f) {
            break;
        }

        // Exact Poisson-flight sampling from the physical interaction
        // coefficient.  For MU_S == 0 the free path is infinite.
        float free_path = 3.402823466e+38F;
        if (MU_S > 0.0f) {
            float u_path = fmaxf(rnd_uniform(&state), 1.0e-12f);
            free_path = -logf(u_path) / MU_S;
        }

        bool interaction_before_boundary = free_path < distance_to_boundary;
        float travel_dist = interaction_before_boundary
            ? free_path
            : distance_to_boundary;

        // Sample the travelled ray segment into the heatmap.  This is only a
        // visualisation sampling interval; it has no role in the interaction
        // probability or ray physics.
        const float HEATMAP_SAMPLE_SPACING = 1.0e-6f;
        int heatmap_steps = (int)ceilf(travel_dist / HEATMAP_SAMPLE_SPACING);
        if (heatmap_steps < 1) {
            heatmap_steps = 1;
        }

        float x_start = x;
        float y_start = y;

        for (int hs = 0; hs < heatmap_steps; hs++) {
            float frac = ((float)hs + 1.0f) / (float)heatmap_steps;
            float xs = x_start + vx * travel_dist * frac;
            float ys = y_start + vy * travel_dist * frac;

            if (xs * xs + ys * ys > R_REAL * R_REAL + 1.0e-8f) {
                break;
            }

            int ix = (int)(((xs + R_REAL) / (2.0f * R_REAL)) * (float)VIS_SIZE);
            if (ix < 0) ix = 0;
            if (ix > VIS_SIZE - 1) ix = VIS_SIZE - 1;

            int iy = VIS_SIZE - 1 - (int)(((ys + R_REAL) / (2.0f * R_REAL)) * (float)VIS_SIZE);
            if (iy < 0) iy = 0;
            if (iy > VIS_SIZE - 1) iy = VIS_SIZE - 1;

            int pix_idx = iy * VIS_SIZE + ix;
            atomicAdd(&heatmap_flat[pix_idx], VISUAL_SCALE);
        }

        x = x_start + vx * travel_dist;
        y = y_start + vy * travel_dist;
        rpl += travel_dist;
        step_count++;

        if (!interaction_before_boundary) {
            // Ballistic/refracted segment has reached the sample boundary.
            break;
        }

        interaction_count++;

        // Preserve the original particle-event selection probabilities
        // conditional on an interaction having occurred.
        float u_particle = rnd_uniform(&state);
        int pidx = 0;

        for (int j = 0; j < n_particles - 1; ++j) {
            if (u_particle <= (float)particle_cdf_table[j]) {
                pidx = j;
                break;
            }
            if (j == n_particles - 2) {
                pidx = n_particles - 1;
            }
        }

        float n_particle_this = (float)particle_refractive_index_table[pidx];
        float theta_projected = sphere_refraction_projected_turn(
            &stateREFR,
            N_MEDIUM,
            n_particle_this
        );

        float ray_angle = atan2f(vy, vx);
        float new_angle = ray_angle + theta_projected;

        vx = cosf(new_angle);
        vy = sinf(new_angle);

        if (step_count >= max_steps) {
            absorbed = 1;
            break;
        }
    }

    if (absorbed == 0) {
        exit_x_out[tid] = x;
        exit_y_out[tid] = y;
        exit_dir_out[tid] = atan2f(vy, vx);
        ray_path_length_out[tid] = rpl;
        interaction_count_out[tid] = interaction_count;
    }
}
}
"""

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
def trace_rays_gpu(
    angles_init_np,
    mu_s,
    n_medium,
    R_REAL,
    RAY_OFFSET,
    VIS_SIZE,
    VISUAL_SCALE,
    particle_cdf_table_np,
    particle_refractive_index_table_np,
    hdf5_file="ray_exits.h5",
    safety_fraction=0.01,
    min_chunk=100_000,
    max_chunk=1_000_000
):
    """GPU ray tracing with spherical-refraction direction changes.

    Per-ray outputs are streamed to HDF5. Particle interaction distances are
    sampled directly from the exponential free-path distribution defined by mu_s.
    The legacy dataset name ``scatter_count`` is retained for downstream
    compatibility, but it now means particle interaction count; every such
    interaction uses the refraction model above.
    """
    N = angles_init_np.shape[0]

    particle_cdf_dev = cp.asarray(particle_cdf_table_np, dtype=cp.float64)
    particle_refractive_index_dev = cp.asarray(
        particle_refractive_index_table_np,
        dtype=cp.float64
    )
    heatmap_dev = cp.zeros((VIS_SIZE * VIS_SIZE,), dtype=cp.float32)
    threads_per_block = 256

    free_bytes, total_bytes = get_gpu_free_bytes()
    if free_bytes is None:
        estimated_chunk = 2_000_000
    else:
        est = estimate_chunk_size_bytes(free_bytes, safety_fraction=safety_fraction)
        estimated_chunk = int(max(min_chunk, min(est, max_chunk)))

    with h5py.File(hdf5_file, "w") as f:
        dset_exit_x = f.create_dataset("exit_x", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_exit_y = f.create_dataset("exit_y", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_exit_dir = f.create_dataset("exit_dir", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_rpl = f.create_dataset("exit_rpl", shape=(N,), dtype='f4', chunks=(estimated_chunk,))
        dset_scatter_count = f.create_dataset("scatter_count", shape=(N,), dtype='i4', chunks=(estimated_chunk,))

        for start in tqdm(
            range(0, N, estimated_chunk),
            total=(N + estimated_chunk - 1) // estimated_chunk,
            desc="Tracing rays"
        ):
            end = min(N, start + estimated_chunk)
            sz = end - start

            angles_chunk = cp.asarray(angles_init_np[start:end], dtype=cp.float64)
            exit_dir_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
            exit_x_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
            exit_y_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
            rpl_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
            scatter_count_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)

            blocks = (sz + threads_per_block - 1) // threads_per_block
            seed0 = np.uint32(np.random.randint(1, 2**31 - 1))
            seed1 = np.uint32(np.random.randint(1, 2**31 - 1))

            launched = False
            attempt_chunk = estimated_chunk
            while not launched:
                try:
                    trace_kernel(
                        (blocks,),
                        (threads_per_block,),
                        (
                            np.float32(MAX_ITERATIONS),
                            np.float32(mu_s),
                            np.float32(n_medium),
                            np.float32(R_REAL),
                            np.float32(RAY_OFFSET),
                            np.int32(VIS_SIZE),
                            np.float32(VISUAL_SCALE),
                            angles_chunk,
                            np.int32(sz),
                            particle_cdf_dev,
                            particle_refractive_index_dev,
                            np.int32(len(particle_cdf_table_np)),
                            heatmap_dev,
                            exit_dir_chunk_dev,
                            exit_x_chunk_dev,
                            exit_y_chunk_dev,
                            rpl_chunk_dev,
                            scatter_count_chunk_dev,
                            seed0,
                            seed1
                        )
                    )
                    cp.cuda.Stream.null.synchronize()
                    launched = True
                except cp.cuda.memory.OutOfMemoryError:
                    attempt_chunk = max(min_chunk, attempt_chunk // 2)
                    if attempt_chunk < 2:
                        raise
                    end = min(N, start + attempt_chunk)
                    sz = end - start
                    cp._default_memory_pool.free_all_blocks()
                    angles_chunk = cp.asarray(angles_init_np[start:end], dtype=cp.float64)
                    exit_dir_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
                    exit_x_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
                    exit_y_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
                    rpl_chunk_dev = cp.full((sz,), cp.nan, dtype=cp.float32)
                    scatter_count_chunk_dev = cp.full((sz,), -1, dtype=cp.int32)
                    blocks = (sz + threads_per_block - 1) // threads_per_block
                    estimated_chunk = attempt_chunk

            dset_exit_dir[start:end] = cp.asnumpy(exit_dir_chunk_dev)
            dset_exit_x[start:end] = cp.asnumpy(exit_x_chunk_dev)
            dset_exit_y[start:end] = cp.asnumpy(exit_y_chunk_dev)
            dset_rpl[start:end] = cp.asnumpy(rpl_chunk_dev)
            dset_scatter_count[start:end] = cp.asnumpy(scatter_count_chunk_dev)

            del (
                angles_chunk,
                exit_dir_chunk_dev,
                exit_x_chunk_dev,
                exit_y_chunk_dev,
                rpl_chunk_dev,
                scatter_count_chunk_dev
            )
            cp._default_memory_pool.free_all_blocks()

    heatmap = cp.asnumpy(heatmap_dev).reshape((VIS_SIZE, VIS_SIZE))
    return heatmap

# ================= SIMULATION (main loop) =================
bulk_profiles = []

detector_centers_rad = np.deg2rad(detector_angles)
detector_accept = detector_acceptance_deg
detector_hit_counts = {}

def sample_laser_angles(N, half_angle_deg=2.0):
    angles = np.random.uniform(
        -np.deg2rad(half_angle_deg),
        np.deg2rad(half_angle_deg),
        N
    )
    return angles.astype(np.float64)

for wl_idx, wl in enumerate(wavelengths):
    print(f"--- Wavelength {int(wl*1e9)} nm ---")
    angles_init = sample_beta_angles(N_RAYS, alpha1, alpha2)
    #angles_init = sample_laser_angles(N_RAYS, half_angle_deg=2.0)

    particle_cdf_table_np = np.asarray(particle_event_cdf, dtype=np.float64)
    particle_refractive_index_table_np = np.asarray(
        particle_refractive_index_by_bin,
        dtype=np.float64
    )

    hdf5_file = os.path.join(OUTDIR, f"ray_exits_{int(wl*1e9)}nm.h5")

    t0 = time.time()
    heatmap = trace_rays_gpu(
        angles_init,
        np.float32(mu_s),
        n_medium,
        R_REAL,
        RAY_OFFSET,
        VIS_SIZE,
        VISUAL_SCALE,
        particle_cdf_table_np,
        particle_refractive_index_table_np,
        hdf5_file=hdf5_file
    )
    t1 = time.time()
    print(f"[INFO] trace_rays_gpu completed in {t1-t0:.2f} s")

    with h5py.File(hdf5_file, "r") as f:
        exit_x = f["exit_x"][:]
        exit_y = f["exit_y"][:]
        exit_dirs = f["exit_dir"][:]
        exit_rpl = f["exit_rpl"][:]
        scatter_count = f["scatter_count"][:]

    valid_exit = (
        np.isfinite(exit_x) &
        np.isfinite(exit_y) &
        np.isfinite(exit_dirs) &
        (scatter_count >= 0)
    )
    valid_counts = scatter_count[valid_exit]
    if valid_counts.size:
        print(f"Valid exiting rays: {valid_counts.size} / {N_RAYS}")
        print(f"Mean interaction count: {np.mean(valid_counts):.3e}")
        print(f"Max interaction count: {np.max(valid_counts)}")
        print(f"Ballistic fraction: {np.mean(valid_counts == 0):.6f}")
    else:
        print("Valid exiting rays: 0")

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

    # Bulk exit-direction histogram: ignore rays that never entered/exited validly.
    exit_dirs_deg = np.rad2deg(exit_dirs[valid_exit])
    if exit_dirs_deg.size:
        hist_bulk, _ = np.histogram(exit_dirs_deg, bins=theta_deg, density=True)
    else:
        hist_bulk = np.zeros(len(theta_deg) - 1, dtype=np.float64)
    bulk_profiles.append(hist_bulk)

    # Detector angle from exit position, folded about the beam axis.
    # atan2(x,y) gives 0 deg at the forward/top detector and +/- angles on the
    # two mirror halves.  Folding maps both physical halves onto the same
    # 0..180 degree detector/scattering-angle scale.
    exit_pos_signed_deg = np.rad2deg(np.arctan2(exit_x, exit_y))
    exit_pos_angles = np.abs(
        (exit_pos_signed_deg + 180.0) % 360.0 - 180.0
    )

    # Save exit positions CSV
    df_exits = pd.DataFrame({
        "exit_x_m": exit_x,
        "exit_y_m": exit_y,
        "exit_rpl_m": exit_rpl,
        "scatter_count": scatter_count,
        "valid_exit": valid_exit,
        "is_ballistic": valid_exit & (scatter_count == 0),
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
    #   refractive particle interaction in the kernel.
    #
    # This prevents non-real backscatter caused by finite beam waist/divergence, boundary
    # intersection, or detector acceptance overlap, while preserving circumference-detector
    # geometry.
    detector_valid = valid_exit & np.isfinite(exit_pos_angles)
    pos_angles_valid = exit_pos_angles[detector_valid]
    interacted_valid = (scatter_count[detector_valid] > 0)

    if pos_angles_valid.size == 0:
        counts = np.zeros_like(detector_angles, dtype=int)
    else:
        diffs = np.abs(pos_angles_valid[:, None] - detector_angles[None, :])
        hits_mask = diffs <= detector_accept

        backscatter_detector_bin = detector_angles >= 90
        ballistic_ray = ~interacted_valid

        # Preserve the previous instrument rule: a ray cannot be called
        # backscatter unless it actually interacted with a particle.
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

# Bulk exit-direction plots
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
plt.title("Bulk Exit-Direction Profiles — Refraction Only")
plt.legend(title="Wavelength")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "bulk_angular_scattering_profiles.png"), dpi=200)
plt.close()
print("✅ Simulation complete. Refraction-only heatmaps + bulk exit directions + detector hits + exit point plots saved.")

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

# Compute histogram
hist_counts, bin_edges = np.histogram(exit_rpl, bins=n_bins)

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

# Total signal (0–180°)
total = np.sum(counts * dtheta)

# Backscatter region (90–180°)
mask_back = angles >= 90
backscatter = np.sum(counts[mask_back] * dtheta)

backscatter_fraction = backscatter / total if total > 0 else 0.0

print("\n=========== DETECTOR-INTEGRATED BACKSCATTER ===========")
print(f"Backscatter fraction (integrated): {backscatter_fraction:.6f}")
print("======================================================")


# ================= PRIMARY PSD / INTERACTION DIAGNOSTICS =================
particle_diag_csv = os.path.join(OUTDIR, "primary_particle_interaction_diagnostics.csv")

df_particle_diag = pd.DataFrame({
    "diameter_um": particle_diameter_m * 1e6,
    "weight_fraction": particle_weights,
    "density_kg_per_m3": particle_density_by_bin_kg_per_m3,
    "refractive_index": particle_refractive_index_by_bin,
    "number_density_per_m3": particle_number_density_by_bin,
    "mie_qsca_cross_section_m2_rate_only": sigma_s,
    "mu_s_by_bin_per_m": mu_s_by_bin,
    "particle_event_weight": particle_event_weights
})
df_particle_diag.to_csv(particle_diag_csv, index=False)
print(f"✅ Saved {particle_diag_csv}")

plt.figure(figsize=(9, 5))
plt.bar(
    particle_diameter_m * 1e6,
    particle_weights,
    width=(particle_diameter_m * 1e6) * 0.08,
    alpha=0.8
)
plt.xscale("log")
plt.xlabel("Primary particle diameter (um)")
plt.ylabel("PSD weight fraction")
plt.title("Primary PSD Used Directly — No Floc Transformation")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "primary_psd_no_flocs.png"), dpi=200)
plt.close()
print("✅ Saved primary_psd_no_flocs.png")

# Summary useful when comparing this fork against the Mie-direction version.
max_single_event_deflection_deg = np.nanmax(delta_curve_deg)
print("\n=========== REFRACTION-ONLY PHYSICS SUMMARY ===========")
print(f"Maximum direct single-sphere deflection: {max_single_event_deflection_deg:.3f} deg")
print("Single-event direction depends on n_medium, n_particle and normalized impact parameter b/R.")
print("For a homogeneous single-index sphere it does NOT depend directly on particle diameter.")
print("Particle diameter still affects interaction frequency and event-bin selection through the preserved Qsca cross-sections.")
print("Any >90 deg detector response must therefore arise from multiple refractive interactions / transport geometry, not a single direct transmitted sphere event.")
print("Transparent-limit check: as mass_concentration_g_per_L -> 0, mu_s -> 0 and particle interaction probability across the sample -> 0.")
print("=======================================================")
