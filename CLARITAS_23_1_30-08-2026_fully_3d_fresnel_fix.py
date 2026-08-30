#!/usr/bin/env python3
# CLARITAS_23_1_30-08-2026_fully_3d_fresnel_fix.py
#
# Fully 3-D geometric-optics particle transport fork of CLARITAS with Fresnel physics.
#
# PHYSICS MODEL
# -------------
# Particle encounters remain purely geometric: there are no Mie cross sections,
# Mie phase functions, flocs, empirical reflection terms or fitted scatter boosts.
# Once a ray geometrically intersects a homogeneous spherical particle, both
# refraction and reflection are handled by full 3-D Fresnel/Snell interface physics.
#
#   PARTICLE ENCOUNTER MODEL
#   - projected sphere encounter cross-section:
#         sigma_geom_i = pi * r_i^2
#   - total encounter coefficient:
#         mu_geom = sum_i(n_i * sigma_geom_i)
#   - free path:
#         s = -ln(U) / mu_geom
#   - particle-bin probability:
#         P(i | encounter) = n_i * sigma_geom_i / mu_geom
#
#   FULL 3-D SPHERE / FRESNEL MODEL
#   - impact point sampled uniformly over projected sphere disk:
#         rho = b/R = sqrt(U)
#   - the actual 3-D entry surface normal and sphere centre are constructed
#   - unpolarised Fresnel reflectance:
#         R = (Rs + Rp)/2
#   - entry interface:
#         medium -> particle; probabilistic reflection or Snell transmission
#   - transmitted rays propagate along the true internal 3-D chord
#   - exit interface:
#         particle -> medium; Fresnel reflection or Snell transmission
#   - internal Fresnel reflections are followed geometrically until the ray exits
#     (subject to a generous safety bounce limit)
#   - all six ray-state components x,y,z,vx,vy,vz are retained throughout
#
# Non-absorbing media are assumed, so for a non-TIR interface T = 1-R. Fresnel
# polarisation is presently treated as unpolarised light. Absorption, surface
# roughness and diffraction/interference remain deliberately absent.
#
# IMPORTANT CONSISTENCY NOTE
# --------------------------
# Geometric encounter area, impact-parameter sampling, actual particle radius,
# Snell transmission and Fresnel reflection now belong to one self-consistent
# geometric-optics sphere model. The sample-scale transport is fully 3-D.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy.ma as ma
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

mass_concentration_g_per_L = 0.5
# Particle interactions are sampled as a continuous exponential free path from mu_geom.
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

# Direction/interface model. V23 retains the complete 3-D ray state through
# entry reflection, transmission, internal propagation and any internal reflections.
OPTICAL_DIRECTION_MODEL = "homogeneous_sphere_snell_fresnel_fully_3d"
FRESNEL_POLARIZATION = "unpolarized"
MAX_INTERNAL_FRESNEL_BOUNCES = 64
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
# Nominal illumination wavelength(s), retained for output labels and future
# wavelength-dependent refractive-index dispersion. With constant n_particle and
# n_medium, Fresnel/Snell coefficients are wavelength-independent in this release.
wavelengths = [622e-9]  # in meters


#### Kernel parameters for TARDIIS####
#R_REAL = 0.049    # Sample radius (m) TARDIIS
R_REAL = 0.049    # 3-D spherical sample radius (m)
SAMPLE_GEOMETRY = "sphere"
#RAY_OFFSET = 0.05  # Ray initial y-offset (m) TARDIIS
RAY_OFFSET = 0.005  # Ray initial y-offset (m)
#STEP_SIZE = 1.0e-6  # integration step size (m)
#STEP_SIZE = 1.0e-7  # integration step size (m)
#VISUAL_SCALE = 100.0 TARDIIS
#VIS_SIZE = 2048      # Heatmap resolution TARDIIS
VISUAL_SCALE = 1.0
VIS_SIZE = 4096      # Heatmap resolution
N_RAYS = 1_000_000  # number of rays to simulate
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

# ============================ PURE GEOMETRIC PARTICLE ENCOUNTERS ============================
# A ray encounters a spherical particle whenever its straight trajectory crosses
# the particle's projected disk. Therefore the physical encounter cross-section is
# simply the projected geometric area pi*r^2.
#
# For a dilute statistically homogeneous suspension, independent encounters form a
# Poisson process along ray path length with coefficient
#     mu_geom = sum_i(n_i * pi*r_i^2)        [1/m]
# and exact free-path sampling
#     s = -ln(U)/mu_geom.
#
# Conditional on an encounter, the selected size bin is weighted by its contribution
# n_i*pi*r_i^2 to the total encounter coefficient. No wavelength-dependent efficiency
# factor or wave-scattering correction is applied.
particle_refractive_index_by_bin = np.full_like(
    particle_diameter_m,
    n_particle,
    dtype=np.float64
)

geometric_cross_section_m2 = np.pi * particle_radius_m**2
mu_geom_by_bin = particle_number_density_by_bin * geometric_cross_section_m2
mu_geom = np.sum(mu_geom_by_bin)

if mu_geom < 0.0:
    raise ValueError("mu_geom must not be negative")

# Continuous-flight transport:
#   P(no particle encounter over path L) = exp(-mu_geom*L)
#   s = -ln(U)/mu_geom
# so concentration -> 0 gives mu_geom -> 0, infinite mean free path and the
# transparent limit automatically.
mean_free_path_m = (1.0 / mu_geom) if mu_geom > 0.0 else np.inf
diameter_optical_depth = mu_geom * (2.0 * R_REAL)
diameter_interaction_probability = -np.expm1(-diameter_optical_depth)

particle_event_weights = np.zeros_like(mu_geom_by_bin, dtype=np.float64)
if mu_geom > 0.0:
    particle_event_weights = mu_geom_by_bin / mu_geom

particle_event_cdf = np.cumsum(particle_event_weights)
if particle_event_cdf.size > 0 and particle_event_cdf[-1] > 0:
    particle_event_cdf /= particle_event_cdf[-1]

print("\n=========== CLARITAS FULLY 3-D GEOMETRIC + FRESNEL ===========")
print(f"Direction model: {OPTICAL_DIRECTION_MODEL}")
print("Particle encounter cross-section: pi*r^2")
print("Particle direction physics: full 3-D Snell + unpolarized Fresnel interfaces")
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
print(f"mu_geom: {mu_geom:.3e} 1/m")
print(f"mean_free_path_m: {mean_free_path_m:.3e}")
print(f"diameter_optical_depth_mu_geom_2R: {diameter_optical_depth:.3e}")
print(f"single_diameter_path_interaction_probability: {diameter_interaction_probability:.6e}")
print(f"particle_event_weights_sum: {np.sum(particle_event_weights):.6f}")
if mu_geom > 0.0:
    print(f"dominant_event_diameter_um: {particle_diameter_m[np.argmax(particle_event_weights)]*1e6:.3f}")
else:
    print("dominant_event_diameter_um: n/a (mu_geom = 0)")
print("Interaction transport: continuous exponential geometric free-path sampling")
print("===========================================================\n")


# ============================ FRESNEL / REFRACTION DIAGNOSTICS ============================
def sphere_refraction_deflection_rad(rho, n_outside, n_inside):
    """Direct transmitted-ray deflection for the zero-internal-reflection branch."""
    rho = np.clip(np.asarray(rho, dtype=np.float64), 0.0, 1.0)
    incidence = np.arcsin(rho)
    sin_refracted = (n_outside / n_inside) * rho
    valid = np.abs(sin_refracted) <= 1.0
    refracted = np.full_like(incidence, np.nan)
    refracted[valid] = np.arcsin(sin_refracted[valid])
    delta = np.full_like(incidence, np.nan)
    delta[valid] = 2.0 * (incidence[valid] - refracted[valid])
    return delta


def fresnel_unpolarized_R_from_cos(cos_i, n1, n2):
    """Unpolarised power reflectance for a lossless dielectric interface.

    Returns R and a TIR mask. Inputs may be scalars or numpy arrays. cos_i is
    the positive cosine of the incidence angle in medium n1.
    """
    cos_i = np.clip(np.asarray(cos_i, dtype=np.float64), 0.0, 1.0)
    sin2_i = np.maximum(0.0, 1.0 - cos_i*cos_i)
    eta = n1 / n2
    sin2_t = eta*eta * sin2_i
    tir = sin2_t >= 1.0

    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))
    rs_den = n1*cos_i + n2*cos_t
    rp_den = n1*cos_t + n2*cos_i

    rs = np.ones_like(cos_i)
    rp = np.ones_like(cos_i)
    good_s = (~tir) & (np.abs(rs_den) > 1.0e-15)
    good_p = (~tir) & (np.abs(rp_den) > 1.0e-15)

    rs[good_s] = ((n1*cos_i[good_s] - n2*cos_t[good_s]) / rs_den[good_s])**2
    rp[good_p] = ((n1*cos_t[good_p] - n2*cos_i[good_p]) / rp_den[good_p])**2

    R = np.where(tir, 1.0, 0.5*(rs + rp))
    return np.clip(R, 0.0, 1.0), tir


rho_curve = np.linspace(0.0, 0.999999, 5000)
incidence_curve = np.arcsin(rho_curve)
cos_inc_curve = np.cos(incidence_curve)
delta_curve_rad = sphere_refraction_deflection_rad(rho_curve, n_medium, n_particle)
delta_curve_deg = np.rad2deg(delta_curve_rad)

R_entry_curve, entry_tir_curve = fresnel_unpolarized_R_from_cos(
    cos_inc_curve, n_medium, n_particle
)

# For the direct transmitted sphere branch, the first exit incidence angle equals
# the internal refracted angle.
sin_r_curve = (n_medium / n_particle) * rho_curve
valid_r_curve = np.abs(sin_r_curve) <= 1.0
r_curve = np.full_like(rho_curve, np.nan)
r_curve[valid_r_curve] = np.arcsin(sin_r_curve[valid_r_curve])
R_exit_curve = np.full_like(rho_curve, np.nan)
if np.any(valid_r_curve):
    R_exit_curve[valid_r_curve], _ = fresnel_unpolarized_R_from_cos(
        np.cos(r_curve[valid_r_curve]), n_particle, n_medium
    )

pd.DataFrame({
    "normalized_impact_parameter_b_over_R": rho_curve,
    "incidence_deg": np.rad2deg(incidence_curve),
    "direct_transmitted_deflection_deg": delta_curve_deg,
    "entry_fresnel_R_unpolarized": R_entry_curve,
    "first_exit_fresnel_R_unpolarized": R_exit_curve,
    "entry_TIR": entry_tir_curve.astype(int)
}).to_csv(os.path.join(OUTDIR, "sphere_fresnel_vs_impact_parameter.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.plot(rho_curve, R_entry_curve, label="entry: medium → particle")
plt.plot(rho_curve, R_exit_curve, label="first exit: particle → medium")
plt.xlabel("Normalized impact parameter b/R")
plt.ylabel("Unpolarized Fresnel reflectance R")
plt.title("Homogeneous Sphere Fresnel Reflectance vs Impact Parameter")
plt.ylim(0.0, 1.02)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "sphere_fresnel_vs_impact_parameter.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(rho_curve, delta_curve_deg)
plt.xlabel("Normalized impact parameter b/R")
plt.ylabel("Direct transmitted deflection (deg)")
plt.title("Homogeneous Sphere: Direct Snell Deflection vs Impact Parameter")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "refraction_deflection_vs_impact_parameter.png"), dpi=200)
plt.close()

# Direct-transmission-only angular diagnostic retained for comparison with V22.
rng_refraction = np.random.default_rng(REFRACTION_PROFILE_SEED)
rho_profile = np.sqrt(rng_refraction.random(REFRACTION_PROFILE_SAMPLES))
delta_profile = sphere_refraction_deflection_rad(rho_profile, n_medium, n_particle)
delta_profile_deg = np.rad2deg(delta_profile)

refraction_profile_density, refraction_profile_edges = np.histogram(
    delta_profile_deg,
    bins=theta_deg,
    density=True
)
refraction_profile_centers = 0.5 * (
    refraction_profile_edges[:-1] + refraction_profile_edges[1:]
)

pd.DataFrame({
    "Angle_deg": refraction_profile_centers,
    "Probability_density_direct_transmission_only": refraction_profile_density
}).to_csv(os.path.join(OUTDIR, "single_event_direct_transmission_profile_3d.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.plot(refraction_profile_centers, refraction_profile_density)
plt.xlabel("True 3-D deflection angle (deg)")
plt.ylabel("Probability density")
plt.title("Direct-Transmission Branch: 3-D Spherical Refraction")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "single_event_direct_transmission_profile_3d.png"), dpi=200)
plt.close()

# Area-weighted mean entry reflectance: rho has PDF 2*rho for uniform disk hits.
rho_int = np.linspace(0.0, 0.999999999, 200_000)
R_int, _ = fresnel_unpolarized_R_from_cos(
    np.sqrt(np.maximum(0.0, 1.0-rho_int*rho_int)),
    n_medium,
    n_particle
)
mean_entry_fresnel_R = float(np.trapezoid(R_int * 2.0*rho_int, rho_int))
single_event_direct_g = float(np.nanmean(np.cos(delta_profile)))


def simulate_single_sphere_fresnel_profile(N, n_outside, n_inside, max_internal_bounces, seed):
    """Vectorised 2-D meridional calculation of true 3-D deflection magnitudes.

    A homogeneous sphere interaction is axisymmetric about the incident ray, so the
    magnitude of the final 3-D deflection can be calculated in one meridional plane.
    Random azimuth changes only the orientation of that plane, not the deflection.
    """
    rng = np.random.default_rng(seed)
    rho = np.sqrt(rng.random(N))
    ci = np.sqrt(np.maximum(0.0, 1.0-rho*rho))

    # Incoming direction is +y. Entry outward normal is (rho, -ci).
    ix = np.zeros(N, dtype=np.float64)
    iy = np.ones(N, dtype=np.float64)
    nx = rho.copy()
    ny = -ci.copy()

    R_entry, tir_entry = fresnel_unpolarized_R_from_cos(ci, n_outside, n_inside)
    entry_reflected = tir_entry | (rng.random(N) < R_entry)
    internal_reflection_counts = np.zeros(N, dtype=np.int32)

    ox = np.full(N, np.nan, dtype=np.float64)
    oy = np.full(N, np.nan, dtype=np.float64)

    # Entry-reflected rays.
    d = ix*nx + iy*ny
    rx = ix - 2.0*d*nx
    ry = iy - 2.0*d*ny
    ox[entry_reflected] = rx[entry_reflected]
    oy[entry_reflected] = ry[entry_reflected]

    active = ~entry_reflected
    if np.any(active):
        # Sphere radius is 1. Current entry point P=(0,0), centre C=-N.
        cx = -nx
        cy = -ny

        cos_i = ci
        eta = n_outside / n_inside
        k = 1.0 - eta*eta*(1.0-cos_i*cos_i)
        cos_t = np.sqrt(np.maximum(0.0, k))
        # Vector Snell: T = eta*I + (eta*cos_i-cos_t)*N.
        vx = eta*ix + (eta*cos_i-cos_t)*nx
        vy = eta*iy + (eta*cos_i-cos_t)*ny
        vn = np.sqrt(vx*vx + vy*vy)
        vx /= vn
        vy /= vn

        px = np.zeros(N, dtype=np.float64)
        py = np.zeros(N, dtype=np.float64)

        for bounce in range(max_internal_bounces + 1):
            ids = np.flatnonzero(active)
            if ids.size == 0:
                break

            rpx = px[ids] - cx[ids]
            rpy = py[ids] - cy[ids]
            chord = -2.0*(rpx*vx[ids] + rpy*vy[ids])
            px[ids] += chord*vx[ids]
            py[ids] += chord*vy[ids]

            nox = px[ids] - cx[ids]
            noy = py[ids] - cy[ids]
            nn = np.sqrt(nox*nox + noy*noy)
            nox /= nn
            noy /= nn

            cos_inside = np.clip(vx[ids]*nox + vy[ids]*noy, 0.0, 1.0)
            R_exit, tir_exit = fresnel_unpolarized_R_from_cos(
                cos_inside, n_inside, n_outside
            )
            do_reflect = tir_exit | (rng.random(ids.size) < R_exit)
            do_transmit = ~do_reflect

            if np.any(do_transmit):
                tid = ids[do_transmit]
                noxt = nox[do_transmit]
                noyt = noy[do_transmit]
                cit = cos_inside[do_transmit]
                eta2 = n_inside / n_outside
                k2 = 1.0 - eta2*eta2*(1.0-cit*cit)
                ct2 = np.sqrt(np.maximum(0.0, k2))
                # Normal against incident direction is -Nout.
                tx = eta2*vx[tid] + (eta2*cit-ct2)*(-noxt)
                ty = eta2*vy[tid] + (eta2*cit-ct2)*(-noyt)
                tn = np.sqrt(tx*tx + ty*ty)
                tx /= tn
                ty /= tn
                ox[tid] = tx
                oy[tid] = ty
                active[tid] = False

            if np.any(do_reflect):
                rid = ids[do_reflect]
                noxr = nox[do_reflect]
                noyr = noy[do_reflect]
                dd = vx[rid]*noxr + vy[rid]*noyr
                vx[rid] = vx[rid] - 2.0*dd*noxr
                vy[rid] = vy[rid] - 2.0*dd*noyr
                internal_reflection_counts[rid] += 1

                if bounce == max_internal_bounces:
                    active[rid] = False

    good = np.isfinite(ox) & np.isfinite(oy)
    final_angle = np.full(N, np.nan, dtype=np.float64)
    final_angle[good] = np.arccos(np.clip(oy[good], -1.0, 1.0))

    return final_angle, entry_reflected, internal_reflection_counts


fresnel_profile_delta, fresnel_profile_entry_reflected, fresnel_profile_internal_counts = (
    simulate_single_sphere_fresnel_profile(
        REFRACTION_PROFILE_SAMPLES,
        n_medium,
        n_particle,
        MAX_INTERNAL_FRESNEL_BOUNCES,
        REFRACTION_PROFILE_SEED + 1
    )
)
fresnel_profile_deg = np.rad2deg(fresnel_profile_delta)
fresnel_good = np.isfinite(fresnel_profile_deg)
fresnel_density, fresnel_edges = np.histogram(
    fresnel_profile_deg[fresnel_good],
    bins=theta_deg,
    density=True
)
fresnel_centers = 0.5*(fresnel_edges[:-1] + fresnel_edges[1:])

pd.DataFrame({
    "Angle_deg": fresnel_centers,
    "Probability_density_all_Fresnel_outcomes": fresnel_density
}).to_csv(os.path.join(OUTDIR, "single_event_fresnel_profile_3d.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.plot(fresnel_centers, fresnel_density)
plt.xlabel("True 3-D deflection angle (deg)")
plt.ylabel("Probability density")
plt.title("Single Geometric Sphere Encounter: Full Fresnel + Snell")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "single_event_fresnel_profile_3d.png"), dpi=200)
plt.close()

single_event_fresnel_g = float(np.nanmean(np.cos(fresnel_profile_delta)))
single_event_entry_reflection_fraction = float(np.mean(fresnel_profile_entry_reflected))
single_event_internal_reflection_fraction = float(
    np.mean(fresnel_profile_internal_counts > 0)
)
single_event_backscatter_fraction = float(
    np.mean(fresnel_profile_deg[fresnel_good] >= 90.0)
)

print("✅ Saved sphere_fresnel_vs_impact_parameter.csv/png")
print("✅ Saved refraction_deflection_vs_impact_parameter.png")
print("✅ Saved single_event_direct_transmission_profile_3d.csv/png")
print("✅ Saved single_event_fresnel_profile_3d.csv/png")
print(f"Area-weighted mean entry Fresnel R: {mean_entry_fresnel_R:.6f}")
print(f"Direct-transmission branch g=<cos(delta)>: {single_event_direct_g:.6f}")
print(f"Full single-encounter Fresnel g: {single_event_fresnel_g:.6f}")
print(f"Single-encounter entry-reflection fraction: {single_event_entry_reflection_fraction:.6f}")
print(f"Single-encounter >=1 internal-reflection fraction: {single_event_internal_reflection_fraction:.6f}")
print(f"Single-encounter backscatter fraction (>=90 deg): {single_event_backscatter_fraction:.6f}")

# ================= 3-D SOURCE SAMPLING (host-side) =================
def sample_beta_directions_3d(N, a1, a2):
    """Axisymmetric 3-D extension of the legacy CLARITAS beta beam divergence.

    The legacy signed in-plane angle had |angle| = Beta(a1,a2)*pi/2. V23 keeps
    exactly that polar-angle magnitude distribution and samples a uniform azimuth,
    making the source rotationally symmetric about the +y beam axis.
    """
    polar = np.random.beta(a1, a2, N) * (np.pi / 2.0)
    azimuth = np.random.uniform(0.0, 2.0 * np.pi, N)
    return polar.astype(np.float64), azimuth.astype(np.float64)


def sample_laser_directions_3d(N, half_angle_deg=2.0):
    # Uniform solid angle within a cone: cos(theta) is uniform over the cone.
    theta_max = np.deg2rad(half_angle_deg)
    u = np.random.random(N)
    cos_theta = 1.0 - u * (1.0 - np.cos(theta_max))
    polar = np.arccos(cos_theta)
    azimuth = np.random.uniform(0.0, 2.0 * np.pi, N)
    return polar.astype(np.float64), azimuth.astype(np.float64)


# ================= CUDA KERNEL: FULLY 3-D GEOMETRIC + FRESNEL =================
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

__device__ void normalize3(float* x, float* y, float* z) {
    float n2 = (*x)*(*x) + (*y)*(*y) + (*z)*(*z);
    if (n2 <= 0.0f) {
        *x = 0.0f; *y = 1.0f; *z = 0.0f;
        return;
    }
    float invn = rsqrtf(n2);
    *x *= invn; *y *= invn; *z *= invn;
}

__device__ float dot3(
    const float ax, const float ay, const float az,
    const float bx, const float by, const float bz)
{
    return ax*bx + ay*by + az*bz;
}

__device__ void reflect3(
    const float ix, const float iy, const float iz,
    const float nx, const float ny, const float nz,
    float* ox, float* oy, float* oz)
{
    // n is the geometric surface normal. Reflection is independent of which
    // medium side the normal points toward.
    float d = dot3(ix, iy, iz, nx, ny, nz);
    *ox = ix - 2.0f*d*nx;
    *oy = iy - 2.0f*d*ny;
    *oz = iz - 2.0f*d*nz;
    normalize3(ox, oy, oz);
}

__device__ int refract3(
    const float ix, const float iy, const float iz,
    const float nx_against, const float ny_against, const float nz_against,
    const float n1, const float n2,
    float* ox, float* oy, float* oz)
{
    // normal_against points into incident medium, so dot(I,N) <= 0.
    float cos_i = -dot3(ix, iy, iz, nx_against, ny_against, nz_against);
    cos_i = fminf(fmaxf(cos_i, 0.0f), 1.0f);

    float eta = n1 / n2;
    float k = 1.0f - eta*eta*(1.0f - cos_i*cos_i);
    if (k <= 0.0f) return 0;  // total internal reflection / no transmitted ray

    float cos_t = sqrtf(k);
    *ox = eta*ix + (eta*cos_i - cos_t)*nx_against;
    *oy = eta*iy + (eta*cos_i - cos_t)*ny_against;
    *oz = eta*iz + (eta*cos_i - cos_t)*nz_against;
    normalize3(ox, oy, oz);
    return 1;
}

__device__ float fresnel_unpolarized_R(
    float cos_i,
    const float n1,
    const float n2,
    int* tir)
{
    cos_i = fminf(fmaxf(cos_i, 0.0f), 1.0f);
    float sin2_i = fmaxf(0.0f, 1.0f - cos_i*cos_i);
    float eta = n1 / n2;
    float sin2_t = eta*eta*sin2_i;

    if (sin2_t >= 1.0f) {
        *tir = 1;
        return 1.0f;
    }

    *tir = 0;
    float cos_t = sqrtf(fmaxf(0.0f, 1.0f - sin2_t));

    float rs_den = n1*cos_i + n2*cos_t;
    float rp_den = n1*cos_t + n2*cos_i;

    float Rs = 1.0f;
    float Rp = 1.0f;
    if (fabsf(rs_den) > 1.0e-12f) {
        float rs = (n1*cos_i - n2*cos_t) / rs_den;
        Rs = rs*rs;
    }
    if (fabsf(rp_den) > 1.0e-12f) {
        float rp = (n1*cos_t - n2*cos_i) / rp_den;
        Rp = rp*rp;
    }

    return fminf(fmaxf(0.5f*(Rs + Rp), 0.0f), 1.0f);
}

__device__ void perpendicular_basis(
    const float vx, const float vy, const float vz,
    float* e1x, float* e1y, float* e1z,
    float* e2x, float* e2y, float* e2z)
{
    float rx, ry, rz;
    if (fabsf(vy) < 0.9f) {
        rx = 0.0f; ry = 1.0f; rz = 0.0f;
    } else {
        rx = 1.0f; ry = 0.0f; rz = 0.0f;
    }

    *e1x = ry*vz - rz*vy;
    *e1y = rz*vx - rx*vz;
    *e1z = rx*vy - ry*vx;
    normalize3(e1x, e1y, e1z);

    *e2x = vy*(*e1z) - vz*(*e1y);
    *e2y = vz*(*e1x) - vx*(*e1z);
    *e2z = vx*(*e1y) - vy*(*e1x);
    normalize3(e2x, e2y, e2z);
}

__device__ int sphere_fresnel_interaction_3d(
    unsigned int* state,
    const float n_medium,
    const float n_particle,
    const float radius,
    const int max_internal_bounces,
    float* x,
    float* y,
    float* z,
    float* vx,
    float* vy,
    float* vz,
    float* internal_path_added,
    int* fresnel_reflection_count,
    int* entry_reflection_count,
    int* internal_reflection_count)
{
    if (radius <= 0.0f) return 1;

    // Uniform interception point over the projected disk of a 3-D sphere.
    float rho = sqrtf(fminf(fmaxf(rnd_uniform(state), 0.0f), 0.99999994f));
    float cos_i = sqrtf(fmaxf(0.0f, 1.0f - rho*rho));
    float phi = rnd_uniform(state) * 2.0f * 3.1415927f;

    // q is the transverse impact direction around the incident ray.
    float e1x, e1y, e1z, e2x, e2y, e2z;
    perpendicular_basis(*vx, *vy, *vz, &e1x, &e1y, &e1z, &e2x, &e2y, &e2z);
    float qx = cosf(phi)*e1x + sinf(phi)*e2x;
    float qy = cosf(phi)*e1y + sinf(phi)*e2y;
    float qz = cosf(phi)*e1z + sinf(phi)*e2z;

    // Outward normal at entry. It points into the incident medium and therefore
    // satisfies dot(v, n_entry) = -cos(i).
    float nex = -cos_i*(*vx) + rho*qx;
    float ney = -cos_i*(*vy) + rho*qy;
    float nez = -cos_i*(*vz) + rho*qz;
    normalize3(&nex, &ney, &nez);

    // Place the actual particle centre so the current transport point is the
    // sampled entry point on a sphere of the selected bin radius.
    float cx = *x - radius*nex;
    float cy = *y - radius*ney;
    float cz = *z - radius*nez;

    int tir_entry = 0;
    float R_entry = fresnel_unpolarized_R(cos_i, n_medium, n_particle, &tir_entry);

    if (tir_entry || rnd_uniform(state) < R_entry) {
        float rvx, rvy, rvz;
        reflect3(*vx, *vy, *vz, nex, ney, nez, &rvx, &rvy, &rvz);
        *vx = rvx; *vy = rvy; *vz = rvz;
        (*fresnel_reflection_count)++;
        (*entry_reflection_count)++;
        return 1;
    }

    // Entry transmission by vector Snell law.
    float ivx, ivy, ivz;
    if (!refract3(
        *vx, *vy, *vz,
        nex, ney, nez,
        n_medium, n_particle,
        &ivx, &ivy, &ivz))
    {
        // Numerically this should coincide with TIR and hence entry reflection.
        float rvx, rvy, rvz;
        reflect3(*vx, *vy, *vz, nex, ney, nez, &rvx, &rvy, &rvz);
        *vx = rvx; *vy = rvy; *vz = rvz;
        (*fresnel_reflection_count)++;
        (*entry_reflection_count)++;
        return 1;
    }

    *vx = ivx; *vy = ivy; *vz = ivz;

    // Traverse the real internal chord. After each internal reflection, the same
    // geometric sphere is followed to the next surface intersection.
    for (int bounce = 0; bounce <= max_internal_bounces; ++bounce) {
        float rx = *x - cx;
        float ry = *y - cy;
        float rz = *z - cz;
        float rdv = dot3(rx, ry, rz, *vx, *vy, *vz);
        float chord = -2.0f * rdv;

        if (chord <= 1.0e-12f) return 0;

        *x += chord*(*vx);
        *y += chord*(*vy);
        *z += chord*(*vz);
        *internal_path_added += chord;

        // Geometric outward normal at this particle surface point.
        float nox = (*x - cx) / radius;
        float noy = (*y - cy) / radius;
        float noz = (*z - cz) / radius;
        normalize3(&nox, &noy, &noz);

        // Incident ray is inside the particle. For the generic refraction function,
        // the normal must point against the incident ray, i.e. inward = -n_out.
        float cos_inside = dot3(*vx, *vy, *vz, nox, noy, noz);
        cos_inside = fminf(fmaxf(cos_inside, 0.0f), 1.0f);

        int tir_exit = 0;
        float R_exit = fresnel_unpolarized_R(
            cos_inside, n_particle, n_medium, &tir_exit
        );

        if (!tir_exit && rnd_uniform(state) >= R_exit) {
            float ovx, ovy, ovz;
            if (refract3(
                *vx, *vy, *vz,
                -nox, -noy, -noz,
                n_particle, n_medium,
                &ovx, &ovy, &ovz))
            {
                *vx = ovx; *vy = ovy; *vz = ovz;
                return 1;
            }
            // Fall through to reflection if numerical refraction fails.
        }

        // Internal Fresnel reflection.
        float rvx, rvy, rvz;
        reflect3(*vx, *vy, *vz, nox, noy, noz, &rvx, &rvy, &rvz);
        *vx = rvx; *vy = rvy; *vz = rvz;
        (*fresnel_reflection_count)++;
        (*internal_reflection_count)++;

        if (bounce == max_internal_bounces) return 0;
    }

    return 0;
}

__global__ void trace_kernel(
    const float MAX_ITERATIONS,
    const float MU_GEOM,
    const float N_MEDIUM,
    const int MAX_INTERNAL_BOUNCES,
    const float R_REAL,
    const float R_OFF,
    const int VIS_SIZE,
    const float VISUAL_SCALE,
    const double* polar_init,
    const double* azimuth_init,
    const int N_rays,
    const double* particle_cdf_table,
    const double* particle_refractive_index_table,
    const double* particle_radius_table,
    const int n_particles,
    float* heatmap_xy_flat,
    float* heatmap_zy_flat,
    float* exit_x_out,
    float* exit_y_out,
    float* exit_z_out,
    float* exit_vx_out,
    float* exit_vy_out,
    float* exit_vz_out,
    float* ray_path_length_out,
    int* interaction_count_out,
    int* fresnel_reflection_count_out,
    int* entry_reflection_count_out,
    int* internal_reflection_count_out,
    unsigned int seed0,
    unsigned int seed1)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N_rays) return;

    unsigned int state = seed0 + (unsigned int)tid * 74729u + 13u;
    unsigned int stateOPT = seed1 + (unsigned int)tid * 104729u + 29u;

    // Circular Gaussian beam waist in source plane (x-z).
    float beam_sigma = 0.00001f;
    float u1 = fmaxf(rnd_uniform(&state), 1.0e-12f);
    float u2 = rnd_uniform(&state);
    float mag = sqrtf(-2.0f * logf(u1));
    float x0 = beam_sigma * mag * cosf(2.0f * 3.1415927f * u2);
    float z0 = beam_sigma * mag * sinf(2.0f * 3.1415927f * u2);
    float y0 = -(R_REAL + R_OFF);

    float polar = (float)polar_init[tid];
    float azimuth = (float)azimuth_init[tid];
    float spolar = sinf(polar);
    float vx = spolar * cosf(azimuth);
    float vy = cosf(polar);
    float vz = spolar * sinf(azimuth);
    normalize3(&vx, &vy, &vz);

    if (vy <= 0.0f) return;

    // Ray-sphere sample entry intersection.
    float b = x0*vx + y0*vy + z0*vz;
    float c = x0*x0 + y0*y0 + z0*z0 - R_REAL*R_REAL;
    float disc = b*b - c;
    if (disc <= 0.0f) return;

    float t = -b - sqrtf(disc);
    if (t < 0.0f) return;

    float x = x0 + t*vx;
    float y = y0 + t*vy;
    float z = z0 + t*vz;

    const int max_steps = (int)MAX_ITERATIONS;
    int step_count = 0;
    int failed = 0;
    float rpl = 0.0f;
    int interaction_count = 0;
    int fresnel_reflection_count = 0;
    int entry_reflection_count = 0;
    int internal_reflection_count = 0;

    while (x*x + y*y + z*z <= R_REAL*R_REAL + 1.0e-8f) {
        float rv = x*vx + y*vy + z*vz;
        float rr_minus_R2 = x*x + y*y + z*z - R_REAL*R_REAL;
        float boundary_disc = rv*rv - rr_minus_R2;
        if (boundary_disc < 0.0f) {
            failed = 1;
            break;
        }

        float distance_to_boundary = -rv + sqrtf(fmaxf(boundary_disc, 0.0f));
        if (distance_to_boundary <= 1.0e-9f) break;

        float free_path = 3.402823466e+38F;
        if (MU_GEOM > 0.0f) {
            float u_path = fmaxf(rnd_uniform(&state), 1.0e-12f);
            free_path = -logf(u_path) / MU_GEOM;
        }

        bool interaction_before_boundary = free_path < distance_to_boundary;
        float travel_dist = interaction_before_boundary ? free_path : distance_to_boundary;

        // Sample external-medium path into two orthogonal diagnostic projections.
        const float HEATMAP_SAMPLE_SPACING = 1.0e-6f;
        int heatmap_steps = (int)ceilf(travel_dist / HEATMAP_SAMPLE_SPACING);
        if (heatmap_steps < 1) heatmap_steps = 1;

        float x_start = x;
        float y_start = y;
        float z_start = z;

        for (int hs = 0; hs < heatmap_steps; ++hs) {
            float frac = ((float)hs + 1.0f) / (float)heatmap_steps;
            float xs = x_start + vx*travel_dist*frac;
            float ys = y_start + vy*travel_dist*frac;
            float zs = z_start + vz*travel_dist*frac;

            if (xs*xs + ys*ys + zs*zs > R_REAL*R_REAL + 1.0e-8f) break;

            int ix = (int)(((xs + R_REAL)/(2.0f*R_REAL))*(float)VIS_SIZE);
            int iz = (int)(((zs + R_REAL)/(2.0f*R_REAL))*(float)VIS_SIZE);
            int iy = VIS_SIZE - 1 - (int)(((ys + R_REAL)/(2.0f*R_REAL))*(float)VIS_SIZE);
            if (ix < 0) ix = 0; if (ix > VIS_SIZE-1) ix = VIS_SIZE-1;
            if (iz < 0) iz = 0; if (iz > VIS_SIZE-1) iz = VIS_SIZE-1;
            if (iy < 0) iy = 0; if (iy > VIS_SIZE-1) iy = VIS_SIZE-1;

            atomicAdd(&heatmap_xy_flat[iy*VIS_SIZE + ix], VISUAL_SCALE);
            atomicAdd(&heatmap_zy_flat[iy*VIS_SIZE + iz], VISUAL_SCALE);
        }

        x = x_start + vx*travel_dist;
        y = y_start + vy*travel_dist;
        z = z_start + vz*travel_dist;
        rpl += travel_dist;
        step_count++;

        if (!interaction_before_boundary) break;

        interaction_count++;

        float u_particle = rnd_uniform(&state);
        int pidx = n_particles - 1;
        for (int j = 0; j < n_particles - 1; ++j) {
            if (u_particle <= (float)particle_cdf_table[j]) {
                pidx = j;
                break;
            }
        }

        float n_particle_this = (float)particle_refractive_index_table[pidx];
        float particle_radius_this = (float)particle_radius_table[pidx];
        float internal_added = 0.0f;

        int ok = sphere_fresnel_interaction_3d(
            &stateOPT,
            N_MEDIUM,
            n_particle_this,
            particle_radius_this,
            MAX_INTERNAL_BOUNCES,
            &x, &y, &z,
            &vx, &vy, &vz,
            &internal_added,
            &fresnel_reflection_count,
            &entry_reflection_count,
            &internal_reflection_count
        );
        rpl += internal_added;

        if (!ok) {
            failed = 1;
            break;
        }

        if (step_count >= max_steps) {
            failed = 1;
            break;
        }
    }

    if (!failed) {
        exit_x_out[tid] = x;
        exit_y_out[tid] = y;
        exit_z_out[tid] = z;
        exit_vx_out[tid] = vx;
        exit_vy_out[tid] = vy;
        exit_vz_out[tid] = vz;
        ray_path_length_out[tid] = rpl;
        interaction_count_out[tid] = interaction_count;
        fresnel_reflection_count_out[tid] = fresnel_reflection_count;
        entry_reflection_count_out[tid] = entry_reflection_count;
        internal_reflection_count_out[tid] = internal_reflection_count;
    }
}
}
"""

module = cp.RawModule(code=cuda_src, options=('-std=c++11',))
trace_kernel = module.get_function('trace_kernel')

# ================= Adaptive chunk size helper =================
def get_gpu_free_bytes():
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        return int(free), int(total)
    except Exception:
        try:
            dev = cp.cuda.Device()
            mem = dev.mem_info
            if isinstance(mem, tuple) and len(mem) == 2:
                return int(mem[0]), int(mem[1])
        except Exception:
            pass
    return None, None


def estimate_chunk_size_bytes(free_bytes, safety_fraction=0.2, overhead_bytes=384*1024*1024):
    # V23 stores six float exit components, path length and four integer event counters.
    per_ray_bytes = 128
    usable = int(free_bytes * safety_fraction) - overhead_bytes
    if usable <= 0:
        return 0
    return max(1, usable // per_ray_bytes)


# ================= GPU wrapper function (adaptive chunking) =================
def trace_rays_gpu(
    polar_init_np,
    azimuth_init_np,
    mu_geom,
    n_medium,
    R_REAL,
    RAY_OFFSET,
    VIS_SIZE,
    VISUAL_SCALE,
    particle_cdf_table_np,
    particle_refractive_index_table_np,
    particle_radius_table_np,
    hdf5_file="ray_exits.h5",
    safety_fraction=0.01,
    min_chunk=100_000,
    max_chunk=1_000_000
):
    """Fully 3-D GPU ray tracing with geometric sphere encounters plus Snell/Fresnel interfaces."""
    N = polar_init_np.shape[0]
    if azimuth_init_np.shape[0] != N:
        raise ValueError("polar_init_np and azimuth_init_np must have equal length")

    particle_cdf_dev = cp.asarray(particle_cdf_table_np, dtype=cp.float64)
    particle_refractive_index_dev = cp.asarray(particle_refractive_index_table_np, dtype=cp.float64)
    particle_radius_dev = cp.asarray(particle_radius_table_np, dtype=cp.float64)
    heatmap_xy_dev = cp.zeros((VIS_SIZE * VIS_SIZE,), dtype=cp.float32)
    heatmap_zy_dev = cp.zeros((VIS_SIZE * VIS_SIZE,), dtype=cp.float32)
    threads_per_block = 256

    free_bytes, total_bytes = get_gpu_free_bytes()
    if free_bytes is None:
        chunk_size = min(max_chunk, max(min_chunk, 250_000))
    else:
        est = estimate_chunk_size_bytes(free_bytes, safety_fraction=safety_fraction)
        chunk_size = int(max(min_chunk, min(est, max_chunk)))

    # Fixed-size HDF5 chunks must not exceed dataset length.
    h5_chunk = max(1, min(chunk_size, N))

    with h5py.File(hdf5_file, "w") as f:
        dsets = {
            "exit_x": f.create_dataset("exit_x", shape=(N,), dtype='f4', chunks=(h5_chunk,)),
            "exit_y": f.create_dataset("exit_y", shape=(N,), dtype='f4', chunks=(h5_chunk,)),
            "exit_z": f.create_dataset("exit_z", shape=(N,), dtype='f4', chunks=(h5_chunk,)),
            "exit_vx": f.create_dataset("exit_vx", shape=(N,), dtype='f4', chunks=(h5_chunk,)),
            "exit_vy": f.create_dataset("exit_vy", shape=(N,), dtype='f4', chunks=(h5_chunk,)),
            "exit_vz": f.create_dataset("exit_vz", shape=(N,), dtype='f4', chunks=(h5_chunk,)),
            "exit_rpl": f.create_dataset("exit_rpl", shape=(N,), dtype='f4', chunks=(h5_chunk,)),
            "interaction_count": f.create_dataset("interaction_count", shape=(N,), dtype='i4', chunks=(h5_chunk,)),
            "fresnel_reflection_count": f.create_dataset("fresnel_reflection_count", shape=(N,), dtype='i4', chunks=(h5_chunk,)),
            "entry_reflection_count": f.create_dataset("entry_reflection_count", shape=(N,), dtype='i4', chunks=(h5_chunk,)),
            "internal_reflection_count": f.create_dataset("internal_reflection_count", shape=(N,), dtype='i4', chunks=(h5_chunk,)),
        }
        f.attrs["transport_dimensions"] = 3
        f.attrs["sample_geometry"] = SAMPLE_GEOMETRY
        f.attrs["sample_radius_m"] = R_REAL
        f.attrs["mu_geom_per_m"] = mu_geom
        f.attrs["optical_direction_model"] = OPTICAL_DIRECTION_MODEL
        f.attrs["fresnel_polarization"] = FRESNEL_POLARIZATION
        f.attrs["max_internal_fresnel_bounces"] = MAX_INTERNAL_FRESNEL_BOUNCES

        start = 0
        pbar = tqdm(total=N, desc="Tracing rays")
        while start < N:
            this_chunk = min(chunk_size, N - start)

            while True:
                end = start + this_chunk
                try:
                    polar_chunk = cp.asarray(polar_init_np[start:end], dtype=cp.float64)
                    azimuth_chunk = cp.asarray(azimuth_init_np[start:end], dtype=cp.float64)
                    out_f = [cp.full((this_chunk,), cp.nan, dtype=cp.float32) for _ in range(7)]
                    interaction_dev = cp.full((this_chunk,), -1, dtype=cp.int32)
                    fresnel_reflection_dev = cp.full((this_chunk,), -1, dtype=cp.int32)
                    entry_reflection_dev = cp.full((this_chunk,), -1, dtype=cp.int32)
                    internal_reflection_dev = cp.full((this_chunk,), -1, dtype=cp.int32)

                    blocks = (this_chunk + threads_per_block - 1) // threads_per_block
                    seed0 = np.uint32(np.random.randint(1, 2**31 - 1))
                    seed1 = np.uint32(np.random.randint(1, 2**31 - 1))

                    trace_kernel(
                        (blocks,),
                        (threads_per_block,),
                        (
                            np.float32(MAX_ITERATIONS),
                            np.float32(mu_geom),
                            np.float32(n_medium),
                            np.int32(MAX_INTERNAL_FRESNEL_BOUNCES),
                            np.float32(R_REAL),
                            np.float32(RAY_OFFSET),
                            np.int32(VIS_SIZE),
                            np.float32(VISUAL_SCALE),
                            polar_chunk,
                            azimuth_chunk,
                            np.int32(this_chunk),
                            particle_cdf_dev,
                            particle_refractive_index_dev,
                            particle_radius_dev,
                            np.int32(len(particle_cdf_table_np)),
                            heatmap_xy_dev,
                            heatmap_zy_dev,
                            out_f[0], out_f[1], out_f[2],
                            out_f[3], out_f[4], out_f[5],
                            out_f[6],
                            interaction_dev,
                            fresnel_reflection_dev,
                            entry_reflection_dev,
                            internal_reflection_dev,
                            seed0,
                            seed1
                        )
                    )
                    cp.cuda.Stream.null.synchronize()
                    break
                except cp.cuda.memory.OutOfMemoryError:
                    cp._default_memory_pool.free_all_blocks()
                    if this_chunk <= 1:
                        raise
                    this_chunk = max(1, this_chunk // 2)
                    chunk_size = min(chunk_size, this_chunk)
                    print(f"[WARN] GPU OOM; reducing chunk size to {this_chunk:,}")

            names = ["exit_x", "exit_y", "exit_z", "exit_vx", "exit_vy", "exit_vz", "exit_rpl"]
            for name, arr in zip(names, out_f):
                dsets[name][start:end] = cp.asnumpy(arr)
            dsets["interaction_count"][start:end] = cp.asnumpy(interaction_dev)
            dsets["fresnel_reflection_count"][start:end] = cp.asnumpy(fresnel_reflection_dev)
            dsets["entry_reflection_count"][start:end] = cp.asnumpy(entry_reflection_dev)
            dsets["internal_reflection_count"][start:end] = cp.asnumpy(internal_reflection_dev)

            del polar_chunk, azimuth_chunk, out_f
            del interaction_dev, fresnel_reflection_dev, entry_reflection_dev, internal_reflection_dev
            cp._default_memory_pool.free_all_blocks()
            pbar.update(this_chunk)
            start = end
        pbar.close()

    heatmap_xy = cp.asnumpy(heatmap_xy_dev).reshape((VIS_SIZE, VIS_SIZE))
    heatmap_zy = cp.asnumpy(heatmap_zy_dev).reshape((VIS_SIZE, VIS_SIZE))
    return heatmap_xy, heatmap_zy


# ================= PLOTTING HELPERS =================
def save_projected_heatmap(heatmap, horizontal_label, title, path):
    masked = ma.masked_where(heatmap == 0, heatmap)
    if masked.count() > 0:
        vmin = masked.min()
        vmax = masked.max()
    else:
        vmin, vmax = 1e-12, 1.0

    plt.figure(figsize=(6, 6))
    colors = [(0, 0, 0), (1, 1, 1)]
    cmap = LinearSegmentedColormap.from_list("black_white", colors, N=256)
    cmap.set_bad(color='black')
    extent_mm = [-R_REAL*1000, R_REAL*1000, -R_REAL*1000, R_REAL*1000]
    plt.imshow(
        masked,
        cmap=cmap,
        aspect="equal",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=extent_mm,
        interpolation='gaussian'
    )
    circle = plt.Circle((0, 0), R_REAL*1000, color='red', linewidth=1.5, fill=False, linestyle='--')
    plt.gca().add_patch(circle)
    plt.xlabel(f"{horizontal_label} (mm)")
    plt.ylabel("y (mm)")
    plt.title(title)
    start = -R_REAL*1000
    end = R_REAL*1000
    step = (end-start)/5.0
    plt.xticks(np.arange(start, end+step, step))
    plt.yticks(np.arange(start, end+step, step))
    plt.colorbar(label="Projected path samples")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ================= SIMULATION (main loop) =================
bulk_profiles = []
detector_hit_counts = {}
last_valid_exit_rpl = np.array([], dtype=np.float64)
last_interaction_count = np.array([], dtype=np.int32)
last_valid_exit = np.array([], dtype=bool)

# Detector centres are 3-D points on the spherical sample boundary. To preserve
# the legacy folded left/right detector convention, each 0..170 degree detector
# represents the union of two symmetric spherical caps in the x-y plane.
detector_theta_rad = np.deg2rad(detector_angles)
detector_sin = np.sin(detector_theta_rad)
detector_cos = np.cos(detector_theta_rad)
detector_cos_accept = np.cos(np.deg2rad(detector_acceptance_deg))

for wl_idx, wl in enumerate(wavelengths):
    print(f"--- Wavelength {int(wl*1e9)} nm ---")
    polar_init, azimuth_init = sample_beta_directions_3d(N_RAYS, alpha1, alpha2)
    #polar_init, azimuth_init = sample_laser_directions_3d(N_RAYS, half_angle_deg=2.0)

    particle_cdf_table_np = np.asarray(particle_event_cdf, dtype=np.float64)
    particle_refractive_index_table_np = np.asarray(particle_refractive_index_by_bin, dtype=np.float64)
    particle_radius_table_np = np.asarray(particle_radius_m, dtype=np.float64)
    hdf5_file = os.path.join(OUTDIR, f"ray_exits_{int(wl*1e9)}nm.h5")

    t0 = time.time()
    heatmap_xy, heatmap_zy = trace_rays_gpu(
        polar_init,
        azimuth_init,
        np.float32(mu_geom),
        n_medium,
        R_REAL,
        RAY_OFFSET,
        VIS_SIZE,
        VISUAL_SCALE,
        particle_cdf_table_np,
        particle_refractive_index_table_np,
        particle_radius_table_np,
        hdf5_file=hdf5_file
    )
    t1 = time.time()
    print(f"[INFO] trace_rays_gpu completed in {t1-t0:.2f} s")

    with h5py.File(hdf5_file, "r") as f:
        exit_x = f["exit_x"][:]
        exit_y = f["exit_y"][:]
        exit_z = f["exit_z"][:]
        exit_vx = f["exit_vx"][:]
        exit_vy = f["exit_vy"][:]
        exit_vz = f["exit_vz"][:]
        exit_rpl = f["exit_rpl"][:]
        interaction_count = f["interaction_count"][:]
        fresnel_reflection_count = f["fresnel_reflection_count"][:]
        entry_reflection_count = f["entry_reflection_count"][:]
        internal_reflection_count = f["internal_reflection_count"][:]

    valid_exit = (
        np.isfinite(exit_x) & np.isfinite(exit_y) & np.isfinite(exit_z) &
        np.isfinite(exit_vx) & np.isfinite(exit_vy) & np.isfinite(exit_vz) &
        np.isfinite(exit_rpl) & (interaction_count >= 0)
    )
    valid_counts = interaction_count[valid_exit]
    if valid_counts.size:
        print(f"Valid exiting rays: {valid_counts.size} / {N_RAYS}")
        print(f"Mean interaction count: {np.mean(valid_counts):.3e}")
        print(f"Median interaction count: {np.median(valid_counts):.3f}")
        print(f"Max interaction count: {np.max(valid_counts)}")
        print(f"Ballistic fraction: {np.mean(valid_counts == 0):.6f}")
        valid_fresnel = fresnel_reflection_count[valid_exit]
        valid_entry_ref = entry_reflection_count[valid_exit]
        valid_internal_ref = internal_reflection_count[valid_exit]
        print(f"Mean Fresnel surface reflections/ray: {np.mean(valid_fresnel):.6f}")
        print(f"Rays with >=1 entry Fresnel reflection: {np.mean(valid_entry_ref > 0):.6f}")
        print(f"Rays with >=1 internal Fresnel reflection: {np.mean(valid_internal_ref > 0):.6f}")
    else:
        print("Valid exiting rays: 0")

    last_valid_exit_rpl = exit_rpl[valid_exit]
    last_interaction_count = interaction_count
    last_valid_exit = valid_exit

    title = f"{mass_concentration_g_per_L} g/L, {int(wl*1e9)} nm — 3-D transport projection"
    xy_path = os.path.join(OUTDIR, f"conc_{mass_concentration_g_per_L}_{int(wl*1e9)}nm_xy.png")
    zy_path = os.path.join(OUTDIR, f"conc_{mass_concentration_g_per_L}_{int(wl*1e9)}nm_zy.png")
    save_projected_heatmap(heatmap_xy, "x", title + " (x-y)", xy_path)
    save_projected_heatmap(heatmap_zy, "z", title + " (z-y)", zy_path)
    # Legacy filename points to x-y projection for easier comparison with earlier releases.
    legacy_heatmap_path = os.path.join(OUTDIR, f"conc_{mass_concentration_g_per_L}_{int(wl*1e9)}nm.png")
    save_projected_heatmap(heatmap_xy, "x", title + " (x-y)", legacy_heatmap_path)
    print(f"✅ Saved {xy_path}")
    print(f"✅ Saved {zy_path}")

    # True 3-D exit direction relative to the +y beam axis.
    vnorm = np.sqrt(exit_vx**2 + exit_vy**2 + exit_vz**2)
    cos_exit_dir = np.full_like(exit_vy, np.nan, dtype=np.float64)
    good_v = valid_exit & (vnorm > 0)
    cos_exit_dir[good_v] = np.clip(exit_vy[good_v] / vnorm[good_v], -1.0, 1.0)
    exit_dir_polar_deg = np.full_like(exit_y, np.nan, dtype=np.float64)
    exit_dir_polar_deg[good_v] = np.rad2deg(np.arccos(cos_exit_dir[good_v]))
    if np.any(good_v):
        hist_bulk, _ = np.histogram(exit_dir_polar_deg[good_v], bins=theta_deg, density=True)
    else:
        hist_bulk = np.zeros(len(theta_deg)-1, dtype=np.float64)
    bulk_profiles.append(hist_bulk)

    # True 3-D exit position on spherical sample boundary.
    r_exit = np.sqrt(exit_x**2 + exit_y**2 + exit_z**2)
    ux = np.full_like(exit_x, np.nan, dtype=np.float64)
    uy = np.full_like(exit_y, np.nan, dtype=np.float64)
    uz = np.full_like(exit_z, np.nan, dtype=np.float64)
    good_p = valid_exit & (r_exit > 0)
    ux[good_p] = exit_x[good_p] / r_exit[good_p]
    uy[good_p] = exit_y[good_p] / r_exit[good_p]
    uz[good_p] = exit_z[good_p] / r_exit[good_p]

    exit_pos_polar_deg = np.full_like(exit_y, np.nan, dtype=np.float64)
    exit_pos_azimuth_deg = np.full_like(exit_y, np.nan, dtype=np.float64)
    exit_pos_polar_deg[good_p] = np.rad2deg(np.arccos(np.clip(uy[good_p], -1.0, 1.0)))
    exit_pos_azimuth_deg[good_p] = (np.rad2deg(np.arctan2(uz[good_p], ux[good_p])) + 360.0) % 360.0

    # Folded physical detector pair: cap centres (+/-sin(theta), cos(theta), 0).
    if np.any(good_p):
        uxx = ux[good_p][:, None]
        uyy = uy[good_p][:, None]
        dots_plus = uxx*detector_sin[None, :] + uyy*detector_cos[None, :]
        dots_minus = -uxx*detector_sin[None, :] + uyy*detector_cos[None, :]
        hits_mask = (dots_plus >= detector_cos_accept) | (dots_minus >= detector_cos_accept)
        counts = hits_mask.sum(axis=0).astype(int)
    else:
        counts = np.zeros_like(detector_angles, dtype=int)
    detector_hit_counts[int(wl*1e9)] = counts

    df_exits = pd.DataFrame({
        "exit_x_m": exit_x,
        "exit_y_m": exit_y,
        "exit_z_m": exit_z,
        "exit_vx": exit_vx,
        "exit_vy": exit_vy,
        "exit_vz": exit_vz,
        "exit_rpl_m": exit_rpl,
        "interaction_count": interaction_count,
        "fresnel_reflection_count": fresnel_reflection_count,
        "entry_reflection_count": entry_reflection_count,
        "internal_reflection_count": internal_reflection_count,
        "transmitted_particle_encounter_count": np.where(
            interaction_count >= 0,
            interaction_count - entry_reflection_count,
            -1
        ),
        "valid_exit": valid_exit,
        "is_ballistic": valid_exit & (interaction_count == 0),
        "exit_position_polar_deg_from_forward": exit_pos_polar_deg,
        "exit_position_azimuth_deg": exit_pos_azimuth_deg,
        "exit_direction_polar_deg_from_forward": exit_dir_polar_deg,
    })
    exits_csv_path = os.path.join(OUTDIR, f"exit_points_3d_{int(wl*1e9)}nm.csv")
    df_exits.to_csv(exits_csv_path, index=False)
    print(f"✅ Saved {exits_csv_path}")

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
plt.xlabel("Detector polar angle from +y beam axis (deg)")
plt.ylabel("Hit count")
plt.title("3-D Detector Hit Counts — Folded Symmetric Spherical Caps")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "detector_hits.png"), dpi=200)
plt.close()
print("✅ Saved detector_hits.png")

plt.figure(figsize=(6,6))
ax = plt.subplot(111, projection='polar')
for wl_nm, counts in detector_hit_counts.items():
    ax.plot(np.deg2rad(detector_angles), counts, '-o', label=f"{wl_nm} nm")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetalim(0, np.pi)
ax.set_title("3-D detector hits (polar-angle representation)")
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "detector_hits_polar.png"), dpi=200)
plt.close()
print("✅ Saved detector_hits_polar.png")

# Bulk true 3-D exit-direction plots.
df_bulk = pd.DataFrame({"Angle_deg": theta_deg[:-1]})
for wl_idx, wl in enumerate(wavelengths):
    df_bulk[f"I_bulk_{int(wl*1e9)}nm"] = bulk_profiles[wl_idx]
bulk_csv_path = os.path.join(OUTDIR, "bulk_angular_exit_profiles_3d.csv")
df_bulk.to_csv(bulk_csv_path, index=False)

plt.figure(figsize=(8,5))
for wl_idx, wl in enumerate(wavelengths):
    plt.plot(theta_deg[:-1], bulk_profiles[wl_idx], label=f"{int(wl*1e9)} nm")
plt.xlabel("True 3-D exit-direction polar angle from +y (deg)")
plt.ylabel("Probability density")
plt.title("Bulk 3-D Exit-Direction Profiles — Geometric Refraction + Fresnel")
plt.legend(title="Wavelength")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "bulk_angular_exit_profiles_3d.png"), dpi=200)
plt.close()
print("✅ Simulation complete. Fully 3-D geometric/Fresnel transport + projected heatmaps + 3-D detector outputs saved.")

# Wavelength-dependent detector response.
plt.figure(figsize=(9,5))
for wl_nm, counts in detector_hit_counts.items():
    counts_norm = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    plt.plot(detector_angles, counts_norm, '-', label=f"{wl_nm} nm")
plt.xlabel("Detector polar angle (deg)")
plt.ylabel("Normalized detector response")
plt.title("3-D Detector Response")
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 181, 10))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "detector_response_vs_wavelength.png"), dpi=200)
plt.close()

# ---------------- Exit path-length histogram ----------------
n_bins = 1000
if last_valid_exit_rpl.size:
    hist_counts, bin_edges = np.histogram(last_valid_exit_rpl, bins=n_bins)
    hist_counts_safe = np.where(hist_counts == 0, 1, hist_counts)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(8,5))
    plt.bar(bin_centers, hist_counts_safe, width=(bin_edges[1]-bin_edges[0]))
    plt.yscale("log")
    plt.xlabel("Exit Ray Path Length (m)")
    plt.ylabel("Counts (log scale)")
    plt.title("3-D Exit Ray Path Lengths")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    hist_png_path = os.path.join(OUTDIR, "exit_rpl_histogram_log.png")
    plt.savefig(hist_png_path, dpi=200)
    plt.close()

    pd.DataFrame({
        "bin_center_m": bin_centers,
        "counts": hist_counts,
        "counts_for_log_plot": hist_counts_safe
    }).to_csv(os.path.join(OUTDIR, "exit_rpl_histogram.csv"), index=False)
    print("✅ Saved exit path-length histogram CSV/PNG")

# ----------------- Detector-based backscatter fraction ---------------------
if 622 in detector_hit_counts:
    angles = np.asarray(detector_angles)
    counts = np.asarray(detector_hit_counts[622])
    dtheta = np.deg2rad(angles[1] - angles[0])
    total = np.sum(counts * dtheta)
    backscatter = np.sum(counts[angles >= 90] * dtheta)
    backscatter_fraction = backscatter / total if total > 0 else 0.0
    print("\n=========== DETECTOR-INTEGRATED BACKSCATTER ===========")
    print(f"Backscatter fraction (integrated): {backscatter_fraction:.6f}")
    print("======================================================")

# ================= PRIMARY PSD / INTERACTION DIAGNOSTICS =================
particle_diag_csv = os.path.join(OUTDIR, "primary_particle_interaction_diagnostics.csv")
pd.DataFrame({
    "diameter_um": particle_diameter_m * 1e6,
    "weight_fraction": particle_weights,
    "density_kg_per_m3": particle_density_by_bin_kg_per_m3,
    "refractive_index": particle_refractive_index_by_bin,
    "number_density_per_m3": particle_number_density_by_bin,
    "geometric_cross_section_m2": geometric_cross_section_m2,
    "mu_geom_by_bin_per_m": mu_geom_by_bin,
    "particle_event_weight": particle_event_weights
}).to_csv(particle_diag_csv, index=False)
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

max_single_event_deflection_deg = np.nanmax(delta_curve_deg)
print("\n=========== V23 FULLY 3-D FRESNEL PHYSICS SUMMARY ===========")
print(f"Transport dimensions: 3")
print(f"Sample geometry: {SAMPLE_GEOMETRY}, radius={R_REAL:.6f} m")
print("Ray state: x,y,z and vx,vy,vz retained through all particle-interface events")
print("Source: circular Gaussian x-z waist + axisymmetric 3-D beta angular divergence")
print("Particle encounters: sigma_geom=pi*r^2, exponential free path from mu_geom")
print("Particle interfaces: unpolarized Fresnel Rs/Rp + vector Snell transmission")
print("Particle geometry: actual radius used for 3-D internal chord propagation")
print(f"Internal Fresnel reflections followed up to {MAX_INTERNAL_FRESNEL_BOUNCES} bounces")
print("Detector geometry: symmetric 3-D spherical-cap pairs at legacy polar angles")
print("Heatmaps: x-y and z-y projections; particle internal chords affect physics/path length but are not separately rasterised")
print(f"Maximum direct zero-internal-reflection transmission deflection: {max_single_event_deflection_deg:.3f} deg")
print(f"Direct-transmission branch anisotropy g: {single_event_direct_g:.6f}")
print(f"Area-weighted mean entry Fresnel R: {mean_entry_fresnel_R:.6f}")
print(f"Full single-encounter Fresnel anisotropy g: {single_event_fresnel_g:.6f}")
print(f"Full single-encounter backscatter fraction: {single_event_backscatter_fraction:.6f}")
print("Single-particle >90 deg response is now possible through Fresnel surface/internal reflection.")
print("==============================================================")
