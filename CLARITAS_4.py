#!/usr/bin/env python3
# adaptive_chunked_tracer.py
# Full simulation script with adaptive GPU chunk_size probing via CuPy.
# Physics, inputs, outputs, kernel logic, and public API preserved exactly.

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

### Reflection supresses wavelenth sensitivity ###
#REFLECT_PROB = 0.23 # probability of reflection at particle - TARDIIS & CLARITAS tuning
REFLECT_PROB = 0.0
# REFLECT_PROB = 0.1202 * np.log(mass_concentration_g_per_L) + 0.0833  # probability of reflection at particle - loess tuning
RELFECT_SIZE_THRESHOLD = 0.0e-6  # minimum diameter (in CDF) for reflection to occur

#reflection_path_length = 0.11
#scatter_prob_per_step = 4.0e-04


#reflection_path_length = 0.0
#scatter_prob_per_step = 0.0
#particle_diameter_m = null_diameter
#particle_weights = null_weights
#particle_density_kg_per_m3 = 1.0  # loess density

particle_diameter_m = loess_diameter
particle_weights = loess_weights
particle_density_kg_per_m3 = 2600.0  # loess density

#particle_diameter_m = kaolin_diameter
#particle_weights = kaolin_weights
#particle_density_kg_per_m3 = 2600.0  # kaolin density

particle_weights = np.float64(particle_weights)
particle_weights /= np.sum(particle_weights)  # normalize
particle_radius_m = particle_diameter_m / 2
cdf = np.cumsum(particle_weights)
cdf /= np.sum(cdf)

n_particle = 1.59  # refractive index of particle
n_medium = 1.33
detector_angles = np.arange(0, 180, 10)   # centres in degrees 0 to 170 - matches TARDIIS & CLARITAS outputs
#detector_angles = np.arange(0, 190, 10)   # centres in degrees 0 to 180
detector_acceptance_deg = 6.5  # degrees
#detector_acceptance_deg = 10.0  # degrees
#theta_deg = np.arange(0.0, 191.0, 0.001)
theta_deg = np.arange(0.0, 181.0, 0.001)


#### LED Beam parameters ####
alpha1 = 1.0
alpha2 = 300.0

#wavelengths = [200e-9, 622e-9, 850e-9]  # in meters
#wavelengths = [200e-9, 622e-9, 950e-9]  # in meters
wavelengths = [622e-9]  # in meters
SCATTER_2D_BOOST = 3.5

#### Kernel parameters for TARDIIS####
#R_REAL = 0.049    # Sample radius (m) TARDIIS
R_REAL = 0.049    # Sample radius (m)
#RAY_OFFSET = 0.05  # Ray initial y-offset (m) TARDIIS
RAY_OFFSET = 0.005  # Ray initial y-offset (m)
#STEP_SIZE = 1.0e-6  # integration step size (m)
STEP_SIZE = 1.0e-7  # integration step size (m)
#VISUAL_SCALE = 100.0 TARDIIS
#VIS_SIZE = 2048      # Heatmap resolution TARDIIS
VISUAL_SCALE = 1.0
VIS_SIZE = 2048      # Heatmap resolution
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

pm = (4/3) * np.pi * particle_radius_m**3 * particle_density_kg_per_m3
particle_mass = np.sum(pm * particle_weights)
n_particles_per_m3 = mass_concentration_g_per_L / particle_mass
dist_csa = np.sum(particle_radius_m**2 * particle_weights) 
samp_csa = dist_csa * np.pi * R_REAL**2 * 0.001 * n_particles_per_m3 

#samp_csa = 1.5e-3

#reflection_path_length = 0.0178 * np.log(samp_csa) + 0.2459
#scatter_prob_per_step = 0.0888 * samp_csa ** 0.7084
#scatter_prob_per_step *= 1
#scatter_prob_per_step = 5.0e-6

######### PSD/concentration-based scattering probability ###############################
# Calculates scatter_prob_per_step from the particle size distribution, mass
# concentration, particle density, wavelength, and refractive indices.
# The model uses:
#   mu_s = n_particles_per_m3 * <Q_sca * pi * r^2>
#   average_particle_separation_m = n_particles_per_m3^(-1/3)
#   effective_depth_m = SCATTER_2D_DEPTH_FACTOR * average_particle_separation_m
#   scatter_prob_per_step = 1 - exp(-mu_s * effective_depth_m)
# where effective_depth_m provides the 3D-to-2D interaction-depth correction.

mass_concentration_kg_per_m3 = mass_concentration_g_per_L  # 1 g/L = 1 kg/m^3

particle_volumes = (4.0 / 3.0) * np.pi * particle_radius_m**3
particle_masses = particle_volumes * particle_density_kg_per_m3
average_particle_mass = np.sum(particle_masses * particle_weights)

n_particles_per_m3 = mass_concentration_kg_per_m3 / average_particle_mass
average_particle_separation_m = n_particles_per_m3 ** (-1.0 / 3.0)

# Use the first configured wavelength for the single scalar kernel scatter probability.
# If multiple wavelengths are used, this scalar is shared by the current kernel call,
# matching the existing code structure.
scatter_probability_wavelength = wavelengths[0]
relative_refractive_index = n_particle / n_medium

sigma_s = []
for r in particle_radius_m:
    try:
        qext, qsca, qback, g = miepython.efficiencies(
            relative_refractive_index,
            2.0 * r,
            scatter_probability_wavelength,
            n_medium
        )
    except TypeError:
        # Fallback for older miepython versions that do not accept n_env.
        x_mie = 2.0 * np.pi * n_medium * r / scatter_probability_wavelength
        qext, qsca, qback, g = miepython.efficiencies_mx(
            relative_refractive_index,
            x_mie
        )

    sigma_s.append(qsca * np.pi * r**2)

sigma_s = np.array(sigma_s, dtype=np.float64)
sigma_s_avg = np.sum(sigma_s * particle_weights)
mu_s = n_particles_per_m3 * sigma_s_avg

# 3D -> 2D empirical correction.
# Keep STEP_SIZE in the exponent; tune SCATTER_2D_BOOST if needed.

scatter_prob_per_step = 1.0 - np.exp(-mu_s * STEP_SIZE * SCATTER_2D_BOOST)
scatter_prob_linear = mu_s * STEP_SIZE * SCATTER_2D_BOOST
#######################################################################################
reflection_path_length = 0.0

######### PHCIP Hermite cubic spline function #############################################################
def get_spline_at_y(x, y, xq):
    N = len(x) - 1
    h = np.diff(x)
    delta = np.diff(y) / h

    m = np.zeros(len(x))

    # Interior slopes
    for i in range(1, N):
        if delta[i-1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            m[i] = (3*(h[i-1]+h[i])) / (
                (2*h[i]+h[i-1])/delta[i-1] +
                (h[i]+2*h[i-1])/delta[i]
            )

    # Endpoints
    m[0] = 0.0
    m[N] = delta[-1]

    # Cubic coefficients
    a = (m[:-1] + m[1:] - 2*delta) / h**2
    b = (3*delta - 2*m[:-1] - m[1:]) / h
    c = m[:-1]
    d = y[:-1]

    #def pchip_eval(xq):
    i = np.searchsorted(x, xq) - 1
    i = np.clip(i, 0, N-1)
    dx = xq - x[i]
    return ((a[i]*dx + b[i])*dx + c[i])*dx + d[i]
######################################################################################

######### spline parameters for x=scatter_prob_per_step, y=samp_csa from TARDIIS calibration data #####################
x = np.array([0.0, 3.98e-05, 1.59e-04, 3.19e-04,
              3.97e-04, 1.59e-03, 3.17e-03])
y = np.array([0.0, 7.25e-06, 2.5e-05, 4.0e-05,
              4.50e-05, 1.2e-04, 1.5e-04])
### Calculate scatter probability per step ###
#scatter_prob_per_step = get_spline_at_y(x,y,samp_csa)
###########################################################################################################

######### spline parameters for x=reflection_path_length, y=samp_csa from TARDIIS calibration data #####################
x = np.array([0.0, 3.98e-05, 1.59e-04, 3.19e-04,
              3.97e-04, 1.59e-03, 3.17e-03])
y = np.array([0.0, 0.08, 0.15, 0.17,
              0.18, 0.23, 0.25])
### Calculate scatter probability per step ###
#reflection_path_length = get_spline_at_y(x,y,samp_csa)
###########################################################################################################

#reflection_path_length = 0.0
#scatter_prob_per_step = 5.0e-3

##############################################

print(f"n_particles_per_m3: {n_particles_per_m3:.3e}")
print(f"particle_mass: {particle_mass:.3e}")
print(f"dist_csa: {dist_csa:.3e}")
print(f"samp_csa: {samp_csa:.3e}")
print(f"average_particle_mass: {average_particle_mass:.3e}")
print(f"average_particle_separation_m: {average_particle_separation_m:.3e}")
print(f"sigma_s_avg: {sigma_s_avg:.3e}")
print(f"mu_s: {mu_s:.3e}")
print(f"scatter_prob_linear: {scatter_prob_linear:.3e}")
print(f"Scattering probability per physical step: {scatter_prob_per_step:.3e}")
print(f"Reflection path length (m): {reflection_path_length:.3e}")

def closest_index(arr, value):
    i = np.searchsorted(arr, value)
    if i == 0:
        return 0
    if i == len(arr):
        return len(arr) - 1
    left = i - 1
    right = i
    return left if abs(arr[left] - value) <= abs(arr[right] - value) else right

# ================= MIE PROFILES (host side - per-particle angular CDFs, cached) =================
theta_rad = np.deg2rad(theta_deg)

# The per-particle Mie angular CDF tables are expensive to build and do not depend
# on concentration. They depend only on PSD, wavelength list, refractive indices,
# and angular resolution, so cache them and re-use them on later runs.
mie_cache_key = (
    str(wavelengths) +
    str(n_particle) +
    str(n_medium) +
    str(float(theta_deg[1] - theta_deg[0])) +
    str(particle_diameter_m.tolist()) +
    str(particle_weights.tolist())
)

mie_cache_hash = hashlib.md5(mie_cache_key.encode()).hexdigest()
mie_cache_file = os.path.join(OUTDIR, f"mie_cache_{mie_cache_hash}.npz")

if os.path.exists(mie_cache_file):
    print(f"Loading cached Mie tables: {mie_cache_file}")

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
    print("Building Mie tables...")

    all_profiles = []
    cdf_profiles = []

    for wl in wavelengths:
        # Build one angular scattering CDF per particle-size bin.
        # The CUDA kernel samples a particle size per scattering event using the PSD,
        # then samples the scattering angle from that particle's own Mie CDF.
        m = n_particle / n_medium
        mu = np.cos(theta_rad)

        psd_weighted_profile = np.zeros_like(theta_rad, dtype=np.float64)
        per_particle_cdfs = []

        for radius, weight in zip(particle_radius_m, particle_weights):
            x = 2*np.pi*n_medium * radius / wl

            S1, S2 = miepython.S1_S2(m, x, mu)
            I = np.sqrt(S1**2) + np.sqrt(S2**2)
            I = np.real(I).astype(np.float64)

            # Preserve the plotted/exported profile as the PSD-weighted mean profile.
            psd_weighted_profile += weight * I

            angle_cdf = np.cumsum(I).astype(np.float64)
            if angle_cdf[-1] > 0:
                angle_cdf /= angle_cdf[-1]
            per_particle_cdfs.append(angle_cdf)

        all_profiles.append(psd_weighted_profile.astype(np.float64))
        cdf_profiles.append(np.asarray(per_particle_cdfs, dtype=np.float64))

    np.savez_compressed(
        mie_cache_file,
        all_profiles=np.asarray(all_profiles, dtype=np.float64),
        cdf_profiles=np.asarray(cdf_profiles, dtype=np.float64)
    )

    print(f"Saved Mie cache: {mie_cache_file}")

# Export angular scattering as before
df_angles = pd.DataFrame({"Angle_deg": theta_deg})
for wl_idx, wl in enumerate(wavelengths):
    df_angles[f"I_{int(wl*1e9)}nm"] = all_profiles[wl_idx]
csv_angles_path = os.path.join(OUTDIR, "angular_scattering_profiles.csv")
df_angles.to_csv(csv_angles_path, index=False)
print(f"✅ Saved {csv_angles_path}")

plt.figure(figsize=(8,5))
for wl_idx, wl in enumerate(wavelengths):
    plt.plot(theta_deg, all_profiles[wl_idx], label=f"{int(wl*1e9)} nm")
plt.xlabel("Scattering angle (deg)")
plt.ylabel("Normalized intensity (a.u.)")
plt.title("Angular scattering profiles (Mie) — I vs θ")
plt.legend(title="Wavelength")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "angular_scattering_profiles.png"), dpi=200)
plt.close()
print(f"✅ Saved angular_scattering_profiles.png")

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
cuda_src = r'''
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

__global__ void trace_kernel(
    const float MAX_ITERATIONS,
    const float RELFECT_SIZE_THRESHOLD,
    const float REFLECTION_PATH_LENGTH,
    const float REFLECT_PROB,
    const float STEP_SIZE,
    const float scatter_prob,
    const float R_REAL,
    const float R_OFF,
    const int VIS_SIZE,
    const float VISUAL_SCALE,
    const double* angles_init,
    const int N_rays,
    const double* particle_cdf_table,
    const double* particle_diameter_table,
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
    unsigned int seed0,
    unsigned int seed1,
    unsigned int seed2)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N_rays) return;

    unsigned int state = seed0 + (unsigned int)tid * 74729u + 13u;
    unsigned int stateREFL = seed1 + (unsigned int)tid * 74729u + 13u;
    unsigned int stateJITTER = seed2 + (unsigned int)tid * 74729u + 13u;

    // ----------------------------
    // Gaussian beam waist
    // ----------------------------
    float beam_sigma = 0.00001f; // 0.01 mm beam radius sigma

    float u1 = fmaxf(rnd_uniform(&state), 1.0e-12f);
    float u2 = rnd_uniform(&state);

    float gaussian =
        sqrtf(-2.0f * logf(u1)) *
        cosf(2.0f * 3.1415927f * u2);

    float x0 = beam_sigma * gaussian;
    float y0 = -(R_REAL + R_OFF);

    // ----------------------------
    // Ray angle and direction
    // ----------------------------
    double angle_d = angles_init[tid];
    float angle_init = (float)angle_d;

    float vx = sinf(angle_init);
    float vy = cosf(angle_init);

    if (vy < 0.0f) {
        vx = -vx;
        vy = -vy;
    }

    // ----------------------------
    // Intersect ray with circular sample
    // ----------------------------
    float b = x0 * vx + y0 * vy;
    float c = x0 * x0 + y0 * y0 - R_REAL * R_REAL;
    float disc = b * b - c;

    if (disc <= 0.0f) return;

    float t = -b - sqrtf(disc);

    if (t < 0.0f) return;

    float x = x0 + t * vx;
    float y = y0 + t * vy;

    // ----------------------------
    // Propagate inside sample
    // ----------------------------
    const int max_steps = MAX_ITERATIONS;
    int step_count = 0;
    int absorbed = 0;
    float rpl = 0.0f;
    int scatter_count = 0;

    while (x * x + y * y <= R_REAL * R_REAL) {

        int ix = (int)(((x + R_REAL) / (2.0f * R_REAL)) * (float)VIS_SIZE);
        if (ix < 0) ix = 0;
        if (ix > VIS_SIZE - 1) ix = VIS_SIZE - 1;

        int iy = VIS_SIZE - 1 - (int)(((y + R_REAL) / (2.0f * R_REAL)) * (float)VIS_SIZE);
        if (iy < 0) iy = 0;
        if (iy > VIS_SIZE - 1) iy = VIS_SIZE - 1;

        int pix_idx = iy * VIS_SIZE + ix;
        atomicAdd(&heatmap_flat[pix_idx], VISUAL_SCALE);

        // ----------------------------
        // Scattering
        // ----------------------------
        if (rnd_uniform(&state) < scatter_prob) {

            scatter_count++;

            // Sample particle size from the PSD for this scattering event.
            float u_particle = rnd_uniform(&state);
            int pidx = 0;

            for (int j = 0; j < n_particles - 1; ++j) {
                if (u_particle <= (float)particle_cdf_table[j]) {
                    pidx = j;
                    break;
                }
                if (j == n_particles - 2) pidx = n_particles - 1;
            }

            // Sample scattering angle from this particle size's own Mie angular CDF.
            float u_angle = rnd_uniform(&state);
            int idx = 0;
            int angle_offset = pidx * n_theta;

            for (int j = 0; j < n_theta - 1; ++j) {
                if (u_angle <= (float)angle_cdf_table[angle_offset + j]) {
                    idx = j;
                    break;
                }
                if (j == n_theta - 2) idx = n_theta - 1;
            }

            float theta_3d = (float)theta_table[idx];
            float sign2 = (rnd_uniform(&stateJITTER) < 0.5f) ? -1.0f : 1.0f;

            // Project a 3D Mie polar deflection into the 2D simulation plane.
            // theta_3d is the polar deflection relative to the incoming ray.
            // phi is the random azimuth around the incoming ray.
            float phi =
                rnd_uniform(&stateJITTER) *
                2.0f * 3.1415927f;

            float theta_projected =
                atan2f(
                    sinf(theta_3d) * cosf(phi),
                    cosf(theta_3d)
                );

            float ray_angle = atan2f(vy, vx);

            float new_angle =
                ray_angle + theta_projected;

            rpl = STEP_SIZE * step_count;

//            if ((rnd_uniform(&stateREFL) < REFLECT_PROB &&
//                cdf_table[idx] >= RELFECT_SIZE_THRESHOLD) ||
//                 rpl <= REFLECTION_PATH_LENGTH) {
//                new_angle += sign2 * 3.1415927f;
//            }

                bool reflection_path_enabled = REFLECTION_PATH_LENGTH > 0.0f;

                if ((rnd_uniform(&stateREFL) < REFLECT_PROB &&
                    particle_diameter_table[pidx] >= RELFECT_SIZE_THRESHOLD) ||
                    (reflection_path_enabled && rpl < REFLECTION_PATH_LENGTH)) {
                    new_angle += sign2 * 3.1415927f;
                }

            vx = cosf(new_angle);
            vy = sinf(new_angle);
        }

        x += vx * STEP_SIZE;
        y += vy * STEP_SIZE;

        step_count++;

        if (step_count >= max_steps) {
            absorbed = 1;
            break;
        }
    }

    // ----------------------------
    // Save exit data
    // ----------------------------
    if (absorbed == 0) {
        exit_x_out[tid] = x;
        exit_y_out[tid] = y;
        exit_dir_out[tid] = atan2f(vy, vx);
        ray_path_length_out[tid] = rpl;
        scatter_count_out[tid] = scatter_count;
    }
}
}
'''

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
def trace_rays_gpu(angles_init_np, STEP_SIZE, scatter_prob, reflection_path_length, R_REAL, RAY_OFFSET, VIS_SIZE,
                   VISUAL_SCALE, particle_cdf_table_np, particle_diameter_table_np, angle_cdf_table_np, theta_table_np,
                   hdf5_file="ray_exits.h5",
                   safety_fraction=0.01, min_chunk=100_000, max_chunk=1_000_000):
    """
    GPU ray tracing with adaptive chunking, streaming per-ray exit data to HDF5.
    Returns only heatmap; exit data is streamed to HDF5.
    """
    N = angles_init_np.shape[0]

    particle_cdf_dev = cp.asarray(particle_cdf_table_np, dtype=cp.float64)
    particle_diameter_dev = cp.asarray(particle_diameter_table_np, dtype=cp.float64)
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

        # Wrap per-chunk loop with tqdm
        for start in tqdm(range(0, N, estimated_chunk), total=(N + estimated_chunk - 1)//estimated_chunk, desc="Tracing rays"):
            end = min(N, start + estimated_chunk)
            sz = end - start

            angles_chunk = cp.asarray(angles_init_np[start:end], dtype=cp.float64)
            exit_dir_chunk_dev = cp.zeros((sz,), dtype=cp.float32)
            exit_x_chunk_dev = cp.zeros((sz,), dtype=cp.float32)
            exit_y_chunk_dev = cp.zeros((sz,), dtype=cp.float32)
            rpl_chunk_dev = cp.zeros((sz,), dtype=cp.float32)
            scatter_count_chunk_dev = cp.zeros((sz,), dtype=cp.int32)

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
                                 (np.float32(MAX_ITERATIONS),
                                  np.float32(RELFECT_SIZE_THRESHOLD),
                                  np.float32(reflection_path_length),
                                  np.float32(REFLECT_PROB),
                                  np.float32(STEP_SIZE),
                                  np.float32(scatter_prob),
                                  np.float32(R_REAL),
                                  np.float32(RAY_OFFSET),
                                  np.int32(VIS_SIZE),
                                  np.float32(VISUAL_SCALE),
                                  angles_chunk,
                                  np.int32(sz),
                                  particle_cdf_dev,
                                  particle_diameter_dev,
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
                                  seed0,
                                  seed1,
                                  seed2
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
                    exit_dir_chunk_dev = cp.zeros((sz,), dtype=cp.float32)
                    exit_x_chunk_dev = cp.zeros((sz,), dtype=cp.float32)
                    exit_y_chunk_dev = cp.zeros((sz,), dtype=cp.float32)
                    rpl_chunk_dev = cp.zeros((sz,), dtype=cp.float32)
                    scatter_count_chunk_dev = cp.zeros((sz,), dtype=cp.int32)
                    blocks = (sz + threads_per_block - 1) // threads_per_block
                    estimated_chunk = attempt_chunk

            # Write chunk to HDF5
            dset_exit_dir[start:end] = cp.asnumpy(exit_dir_chunk_dev)
            dset_exit_x[start:end] = cp.asnumpy(exit_x_chunk_dev)
            dset_exit_y[start:end] = cp.asnumpy(exit_y_chunk_dev)
            dset_rpl[start:end] = cp.asnumpy(rpl_chunk_dev)
            dset_scatter_count[start:end] = cp.asnumpy(scatter_count_chunk_dev)

            # Free GPU memory
            del angles_chunk, exit_dir_chunk_dev, exit_x_chunk_dev, exit_y_chunk_dev, rpl_chunk_dev, scatter_count_chunk_dev
            cp._default_memory_pool.free_all_blocks()

    heatmap = cp.asnumpy(heatmap_dev).reshape((VIS_SIZE, VIS_SIZE))
    return heatmap

# ================= SIMULATION (main loop) =================
from tqdm import tqdm
import h5py
import hashlib

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
    #angles_init = sample_beta_angles(N_RAYS, alpha1, alpha2)  # host numpy float64
    angles_init = sample_laser_angles(N_RAYS, half_angle_deg=2.0)
    particle_cdf_table_np = np.array(cdf, dtype=np.float64)
    particle_diameter_table_np = np.array(particle_diameter_m, dtype=np.float64)
    angle_cdf_table_np = np.array(cdf_profiles[wl_idx], dtype=np.float64)
    theta_table_np = np.array(theta_rad, dtype=np.float64)

    hdf5_file = os.path.join(OUTDIR, f"ray_exits_{int(wl*1e9)}nm.h5")

    t0 = time.time()
    # Call GPU tracer (adaptive chunking, HDF5 streaming)
    heatmap = trace_rays_gpu(angles_init, np.float32(STEP_SIZE),
                             scatter_prob_per_step,
                             reflection_path_length,
                             R_REAL, RAY_OFFSET, VIS_SIZE, VISUAL_SCALE,
                             particle_cdf_table_np, particle_diameter_table_np, angle_cdf_table_np, theta_table_np,
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

    print(f"Mean scatter count: {np.mean(scatter_count):.3e}")
    print(f"Max scatter count: {np.max(scatter_count)}")

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

    # Bulk angular scattering histogram
    exit_dirs_deg = np.rad2deg(exit_dirs)
    hist_bulk, _ = np.histogram(exit_dirs_deg, bins=theta_deg, density=True)
    bulk_profiles.append(hist_bulk)

    # Compute exit position angles
    exit_pos_angles = (np.rad2deg(np.arctan2(exit_x, exit_y)) + 360) % 360

    # Save exit positions CSV
    df_exits = pd.DataFrame({
        "exit_x_m": exit_x,
        "exit_y_m": exit_y,
        "exit_rpl_m": exit_rpl,
        "scatter_count": scatter_count,
        "is_ballistic": scatter_count == 0,
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
    in_detector_semicircle = (exit_pos_angles >= 0) & (exit_pos_angles <= 180)
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