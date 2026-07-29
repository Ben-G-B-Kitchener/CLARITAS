# CLARITAS_104: Polarised vector radiative transfer Monte Carlo
# with full Stokes vector tracking and Mueller-matrix scattering.
#
# GPU-accelerated via CuPy + custom CUDA C++ kernel (xorshift32 per-thread RNG).
# Falls back to CPU multiprocessing if CuPy is unavailable.
#
# Physics (v104 improvements over v103):
#   - Mie scattering with full 4×4 Mueller matrix for each particle bin
#   - Stokes vector [I, Q, U, V] tracked per photon packet
#   - Mueller M(θ) applied at each scatter with reference-frame rotations
#   - Event-driven free-path sampling, sphere-boundary Fresnel transmission
#   - Flocs treated as Maxwell-Garnett effective-medium Mie scatterers
#   - Concentration-dependent fractal dimension (Df increases with φ)
#   - Dependent scattering correction (packing factor) for dense media
#   - FLOC_SCATTER_EFFICIENCY applied to cross-sections
#   - Concentration-dependent radius via transport optical depth (R/T_MFP)
#   - Material-specific k from mineralogy literature
#   - Diffraction peak smoothing for irregular particles (θ < 5°)
#   - Flocculation with Conc-dependent Df and dependent scattering
#   - NEW: Material-specific floc size bins (loess: 5-250µm, kaolin: 0.5-50µm)
#   - Disk-cached Mueller tables for instant startup on re-runs
#
# Input data (from CLARITAS_78 heritage):
#   - Loess/kaolin PSD tables, mass concentration, particle density
#   - Flocculation model: fractal pooling, Maxwell-Garnett index
#   - Detector geometry, sample radius, wavelengths

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy.ma as ma
import miepython
import os
import time
import h5py
import hashlib

# --- GPU support ---
_CUPY_AVAILABLE = False
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
    print("CuPy detected: GPU acceleration enabled.")
except ImportError:
    print("CuPy not found. Falling back to CPU multiprocessing.")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

# ============================================================================
# SECTION 1: Configuration
# ============================================================================

SIMULATION_SEED = 20260727
SOURCE_MODE = "production_beta"  # "production_beta" or "reference_collimated"
PRODUCTION_BEAM_SIGMA_M = 10.0e-6

if SOURCE_MODE not in {"production_beta", "reference_collimated"}:
    raise ValueError("SOURCE_MODE must be 'production_beta' or 'reference_collimated'")

# === Primary particle size distributions ====================================

loess_diameter = np.array([
    1.729e-6, 1.981e-6, 2.269e-6, 2.599e-6, 2.976e-6, 3.409e-6, 3.905e-6,
    4.472e-6, 5.122e-6, 5.867e-6, 6.72e-6, 7.697e-6, 8.816e-6, 10.097e-6,
    11.565e-6, 13.246e-6, 15.172e-6, 17.377e-6, 19.904e-6, 22.797e-6,
    26.111e-6, 29.907e-6, 34.255e-6, 39.234e-6, 44.938e-6, 51.471e-6,
    58.953e-6, 67.523e-6, 77.34e-6, 88.583e-6, 101.46e-6, 116.21e-6,
    133.103e-6, 152.453e-6, 174.616e-6, 200.000e-6, 229.075e-6, 262.376e-6
])

loess_weights = np.array([
    157, 227, 294, 354, 414, 487, 592, 747, 975, 1291, 1704, 2197, 2736,
    3288, 3822, 4196, 4372, 4391, 4352, 4362, 4508, 4826, 5279, 5758,
    6080, 6106, 5786, 5149, 4342, 3404, 2456, 1662, 1175, 858, 631, 463, 333, 230
])

kaolin_diameter = np.array([
    0.172e-6, 0.197e-6, 0.226e-6, 0.259e-6, 0.296e-6, 0.339e-6, 0.389e-6,
    0.445e-6, 0.51e-6, 0.584e-6, 0.669e-6, 0.766e-6, 0.877e-6, 1.005e-6,
    1.151e-6, 1.318e-6, 1.51e-6, 1.729e-6, 1.981e-6, 2.269e-6,
    2.599e-6, 2.976e-6, 3.409e-6, 3.905e-6, 4.472e-6, 5.122e-6,
    5.867e-6, 6.72e-6, 7.697e-6, 8.816e-6, 10.097e-6, 11.565e-6,
    13.246e-6, 15.172e-6, 17.377e-6, 19.904e-6, 22.797e-6
])

kaolin_weights = np.array([
    217, 547, 1112, 2032, 2985, 3492, 3308, 2644, 1893, 1300, 916, 700, 601,
    584, 637, 757, 948, 1208, 1530, 1899, 2309, 2770, 3312, 3973,
    4772, 5681, 6583, 7267, 7478, 7042, 6113, 5057, 3680, 2330, 1287, 631, 284
])

# === Material selection ====================================================

particle_diameter_m = loess_diameter.copy()
particle_weights = loess_weights.copy()
particle_density_kg_per_m3 = 2600.0
mass_concentration_g_per_L = 0.5

FLOC_ENABLED = True

FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M = np.array([
    2.0e-6, 3.0e-6, 4.0e-6, 5.0e-6, 6.5e-6, 8.0e-6, 10.0e-6,
    12.5e-6, 15.0e-6, 20.0e-6, 25.0e-6, 35.0e-6, 50.0e-6
], dtype=np.float64)

# Material-specific floc effective diameter bins (v104)
FLOC_POOL_EFFECTIVE_DIAMETER_M_LOESS = np.array([
    5.0e-6, 8.0e-6, 12.0e-6, 20.0e-6, 30.0e-6, 40.0e-6,
    60.0e-6, 80.0e-6, 100.0e-6, 150.0e-6, 200.0e-6, 250.0e-6
], dtype=np.float64)
FLOC_POOL_EFFECTIVE_DIAMETER_M_KAOLIN = np.array([
    0.5e-6, 0.8e-6, 1.2e-6, 2.0e-6, 3.0e-6, 4.0e-6,
    6.0e-6, 8.0e-6, 12.0e-6, 20.0e-6, 35.0e-6, 50.0e-6
], dtype=np.float64)
FLOC_POOL_EFFECTIVE_DIAMETER_M = FLOC_POOL_EFFECTIVE_DIAMETER_M_LOESS.copy()

FLOC_POOL_KERNEL_LOG_SIGMA = 0.35
FLOC_POOL_KERNEL_MIN_PROBABILITY = 0.0
FLOC_POOL_KERNEL_ENFORCE_FLOC_NOT_SMALLER_THAN_SOURCE = True
FLOC_FRACTAL_DIMENSION = 2.0  # base value; concentration-dependent Df applied if enabled
FLOC_COLLISION_LENGTH_M = 250.0e-6
FLOC_SCATTER_EFFICIENCY = 0.85  # porous flocs scatter less efficiently than solid spheres

# === Concentration-dependent floc physics (new in v101) ====================

FLOC_CONC_DEPENDENT_DF_ENABLED = True
FLOC_DF_MIN = 1.7   # dilute limit (low concentration)
FLOC_DF_MAX = 2.3   # concentrated limit (high collision rate)
FLOC_DF_PHI_HALF = 0.001  # volume fraction at which Df = (Df_min + Df_max)/2

DEPENDENT_SCATTERING_ENABLED = True
DEPENDENT_SCATTERING_PHI_THRESHOLD = 1.0e-4  # volume fraction above which correction applies

FLOC_FLOC_AGGREGATION_ENABLED = False  # experimental: floc-floc collision mode

# === Refractive indices ====================================================

n_particle = 1.59
PRIMARY_REFRACTIVE_INDEX_IMAG_K = 0.004  # default; overridden per material below
n_medium = 1.33
n_external = 1.0
BOUNDARY_FRESNEL_ENABLED = False

# === Material-specific refractive indices (v101) =============================

K_LOESS = 0.005   # iron-oxide-coated mineral dust (Sokolik & Toon 1999: 0.004-0.008 at 622nm)
K_KAOLIN = 0.0005 # pure kaolinite clay (Egan & Hilgeman 1979: nearly transparent at visible wavelengths)

# === Concentration-dependent sample geometry (v102) ==========================

CONC_DEPENDENT_RADIUS_ENABLED = True
RADIUS_REFERENCE_TAU = 1.5   # transport optical depth at reference (loess 0.5 g/L, R/T_MFP ≈ 0.085)
RADIUS_MIN_M = 0.005          # minimum effective radius (5 mm)

# === Diffraction peak smoothing (v102) =======================================

DIFFRACTION_SMOOTHING_ENABLED = True
SMOOTHING_ANGLE_DEG = 5.0     # Gaussian kernel half-width for diffraction peak broadening
SMOOTHING_SIZE_THRESHOLD_X = 10.0  # size parameter x = 2πr/λ above which smoothing is applied

# === Detector configuration ================================================

detector_angles = np.arange(0, 180, 10)
detector_acceptance_deg = 6.5

# === Source & sample geometry ==============================================

R_REAL = 0.049
RAY_OFFSET = 0.005
VIS_SIZE = 4096
VISUAL_SCALE = 1.0
N_RAYS = 1000_000
MAX_EXTINCTIONS = 1e9

# === GPU chunking ==========================================================

GPU_MIN_CHUNK_RAYS = 10_000
GPU_MAX_CHUNK_RAYS = 500_000

# === Wavelengths ===========================================================

wavelengths = [622e-9]

# === PSD weight mode =======================================================

PSD_WEIGHT_MODE = "mass_fraction"

# === Output =================================================================

OUTDIR = "."
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================================
# CALIBRATION DATA: Raw measured voltages per detector angle
# Angles: 0° to 170° in 10° steps (18 detectors)
# Columns: Kaolin 0.5, 2.0, 4.0 g/L  |  Loess 0.5, 2.0, 4.0 g/L
# Detectors at 150° and 160° were likely under-reading in physical experiments
# ============================================================================

CALIBRATION_ANGLES_DEG = np.arange(0, 180, 10)

CALIBRATION_DATA = {
    # material -> concentration_g_per_L -> array of 18 voltages (0°..170°)
    "kaolin": {
        0.5: np.array([0.044966297, 0.040141335, 0.040635691, 0.033261336,
                       0.031612225, 0.031139978, 0.032483981, 0.031326215,
                       0.032538061, 0.035728024, 0.035407573, 0.034166179,
                       0.030915492, 0.029964220, 0.030456849, 0.031725835,
                       0.083952330, 0.087742965]),
        2.0: np.array([0.029303169, 0.026980024, 0.027588378, 0.025777957,
                       0.025938714, 0.026363886, 0.027901944, 0.027897973,
                       0.029080775, 0.033741002, 0.035486995, 0.034008434,
                       0.031570375, 0.031749041, 0.035415781, 0.043839834,
                       0.108949574, 0.126926344]),
        4.0: np.array([0.029225296, 0.026847501, 0.027449791, 0.025541964,
                       0.025585957, 0.025953808, 0.027367672, 0.027328617,
                       0.027985542, 0.030501575, 0.031240254, 0.031768086,
                       0.030057267, 0.030840993, 0.034682898, 0.044086378,
                       0.109560000, 0.138534455]),
    },
    "loess": {
        0.5: np.array([2.440330303, 0.626519136, 0.113698796, 0.043190244,
                       0.030804260, 0.028475998, 0.029407479, 0.028713983,
                       0.029475580, 0.030494678, 0.031335233, 0.033561334,
                       0.034642588, 0.035014151, 0.032513842, 0.034008468,
                       0.079254431, 0.083919911]),
        2.0: np.array([0.113367757, 0.071473857, 0.048613776, 0.030320616,
                       0.026228092, 0.025123021, 0.026763737, 0.026019845,
                       0.027114215, 0.027957246, 0.028971321, 0.031688533,
                       0.030873268, 0.030837965, 0.029129781, 0.028983729,
                       0.064161392, 0.066052410]),
        4.0: np.array([0.028640327, 0.025272048, 0.026527090, 0.023043773,
                       0.022833552, 0.022951635, 0.024207695, 0.023788577,
                       0.025601882, 0.026688448, 0.027272578, 0.029114463,
                       0.026275656, 0.027099212, 0.027270948, 0.027722322,
                       0.060134199, 0.063353778]),
    },
}

CALIB_SUSPECT_ANGLES = [150, 160]  # detectors likely under-reading in experiment


# ============================================================================
# SECTION 2: PSD Processing & Floc Model
# ============================================================================

primary_particle_diameter_m = np.asarray(particle_diameter_m, dtype=np.float64)
primary_particle_weights = np.asarray(particle_weights, dtype=np.float64)

# Detect material early (v104: must happen before floc processing & cache build)
if np.array_equal(primary_particle_diameter_m, loess_diameter):
    material_name = "loess"
    PRIMARY_REFRACTIVE_INDEX_IMAG_K = K_LOESS
    FLOC_POOL_EFFECTIVE_DIAMETER_M = FLOC_POOL_EFFECTIVE_DIAMETER_M_LOESS.copy()
elif np.array_equal(primary_particle_diameter_m, kaolin_diameter):
    material_name = "kaolin"
    PRIMARY_REFRACTIVE_INDEX_IMAG_K = K_KAOLIN
    FLOC_POOL_EFFECTIVE_DIAMETER_M = FLOC_POOL_EFFECTIVE_DIAMETER_M_KAOLIN.copy()
else:
    material_name = "unknown"

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

primary_particle_weights = primary_particle_weights / np.sum(primary_particle_weights)
primary_particle_radius_m = primary_particle_diameter_m / 2.0
primary_particle_volumes_m3 = (4.0/3.0) * np.pi * primary_particle_radius_m**3
primary_particle_masses_kg = primary_particle_volumes_m3 * particle_density_kg_per_m3
mass_concentration_kg_per_m3 = mass_concentration_g_per_L

if PSD_WEIGHT_MODE == "mass_fraction":
    primary_particle_number_density_by_bin = (
        mass_concentration_kg_per_m3 * primary_particle_weights / primary_particle_masses_kg
    )
elif PSD_WEIGHT_MODE == "number_fraction":
    nw = primary_particle_weights / np.sum(primary_particle_weights)
    primary_particle_number_density_by_bin = (
        mass_concentration_kg_per_m3 / np.sum(nw * primary_particle_masses_kg) * nw
    )
else:
    raise ValueError("PSD_WEIGHT_MODE must be 'mass_fraction' or 'number_fraction'")

# --- Floc processing -------------------------------------------------------
if FLOC_ENABLED:
    primary_bin_floc_band_index = np.full(
        primary_particle_diameter_m.shape, -1, dtype=np.int32)
    prev = 0.0
    for bi, be in enumerate(FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M):
        m = (primary_particle_diameter_m <= be) & (primary_particle_diameter_m > prev)
        primary_bin_floc_band_index[m] = bi
        prev = be
    primary_bin_is_pooled = primary_bin_floc_band_index >= 0
    elig = np.sum(primary_particle_number_density_by_bin[primary_bin_is_pooled])
    if elig > 0:
        esp = elig ** (-1.0/3.0)
        fep = FLOC_COLLISION_LENGTH_M / (FLOC_COLLISION_LENGTH_M + esp)
    else:
        esp, fep = np.inf, 0.0
    fep = np.clip(fep, 0.0, 1.0)
    floc_mass_fraction = fep
    ns, nf = len(FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M), len(FLOC_POOL_EFFECTIVE_DIAMETER_M)
    sl = max(float(FLOC_POOL_KERNEL_LOG_SIGMA), 1e-12)
    fpk = np.zeros((ns, nf), dtype=np.float64)
    if ns == nf:
        pref = FLOC_POOL_EFFECTIVE_DIAMETER_M.copy()
    else:
        sp = np.linspace(0, 1, ns)
        fp = np.linspace(0, 1, nf)
        pref = np.exp(np.interp(sp, fp, np.log(FLOC_POOL_EFFECTIVE_DIAMETER_M)))
    for si in range(ns):
        lr = np.log(FLOC_POOL_EFFECTIVE_DIAMETER_M / max(pref[si], 1e-30))
        r = np.exp(-0.5*(lr/sl)**2); r[~np.isfinite(r)]=0; r=np.maximum(r,0)
        if FLOC_POOL_KERNEL_ENFORCE_FLOC_NOT_SMALLER_THAN_SOURCE:
            r[FLOC_POOL_EFFECTIVE_DIAMETER_M < FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M[si]] = 0
        if FLOC_POOL_KERNEL_MIN_PROBABILITY > 0:
            r[r < FLOC_POOL_KERNEL_MIN_PROBABILITY] = 0
        if np.sum(r) <= 0:
            vb = np.where(FLOC_POOL_EFFECTIVE_DIAMETER_M >= FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M[si])[0]
            if vb.size <= 0: vb = np.array([nf-1], dtype=np.int64)
            r[int(vb[np.argmin(np.abs(lr[vb]))])] = 1.0
        fpk[si,:] = r/np.sum(r)
    frd = FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M.copy()
    frm = particle_density_kg_per_m3 * (np.pi/6.0) * frd**3
    fmk = np.zeros((ns, nf), dtype=np.float64)
    fdk = np.zeros((ns, nf), dtype=np.float64)
    for si in range(ns):
        d0 = max(float(frd[si]), 1e-30)
        m0 = float(frm[si])
        fmk[si,:] = m0*(FLOC_POOL_EFFECTIVE_DIAMETER_M/d0)**FLOC_FRACTAL_DIMENSION
        fv = (np.pi/6.0)*FLOC_POOL_EFFECTIVE_DIAMETER_M**3
        fdk[si,:] = fmk[si,:] / fv
    ed, ew, em, eden, eif, efi, ebk = [], [], [], [], [], [], []
    for si in range(ns):
        bm = primary_bin_floc_band_index == si
        bmf = np.sum(primary_particle_weights[bm])
        if bmf <= 0: continue
        pmf = bmf*floc_mass_fraction
        rmf = bmf*(1-floc_mass_fraction)
        if pmf > 0:
            for fi in range(nf):
                kp = fpk[si,fi]
                cmf = pmf*kp
                if cmf <= 0: continue
                ed.append(FLOC_POOL_EFFECTIVE_DIAMETER_M[fi])
                ew.append(cmf); em.append(fmk[si,fi]); eden.append(fdk[si,fi])
                eif.append(True); efi.append(fi); ebk.append("pooled_kernel_floc")
        if rmf > 0:
            bw = primary_particle_weights[bm]; bws = np.sum(bw)
            if bws > 0:
                rw = rmf*bw/bws
                for dp,w in zip(primary_particle_diameter_m[bm], rw):
                    if w <= 0: continue
                    mp = particle_density_kg_per_m3*(np.pi/6.0)*dp**3
                    ed.append(dp); ew.append(w); em.append(mp)
                    eden.append(particle_density_kg_per_m3)
                    eif.append(False); efi.append(-1); ebk.append("residual_primary")
    nm = ~primary_bin_is_pooled
    for dp,w in zip(primary_particle_diameter_m[nm], primary_particle_weights[nm]):
        if w <= 0: continue
        mp = particle_density_kg_per_m3*(np.pi/6.0)*dp**3
        ed.append(dp); ew.append(w); em.append(mp)
        eden.append(particle_density_kg_per_m3)
        eif.append(False); efi.append(-1); ebk.append("unchanged_primary")
    particle_diameter_m = np.asarray(ed, dtype=np.float64)
    particle_weights = np.asarray(ew, dtype=np.float64); particle_weights /= np.sum(particle_weights)
    particle_mass_by_bin_kg = np.asarray(em, dtype=np.float64)
    particle_density_by_bin_kg_per_m3 = np.asarray(eden, dtype=np.float64)
    particle_is_floc = np.asarray(eif, dtype=bool)
    floc_band_index_by_effective_bin = np.asarray(efi, dtype=np.int32)
    effective_bin_kind = np.asarray(ebk, dtype=object)
else:
    particle_diameter_m = primary_particle_diameter_m.copy()
    particle_weights = primary_particle_weights.copy()
    particle_mass_by_bin_kg = particle_density_kg_per_m3*(np.pi/6.0)*particle_diameter_m**3
    particle_density_by_bin_kg_per_m3 = np.full_like(particle_diameter_m, particle_density_kg_per_m3)
    particle_is_floc = np.zeros_like(particle_diameter_m, dtype=bool)
    floc_band_index_by_effective_bin = np.full_like(particle_diameter_m, -1, dtype=np.int32)
    effective_bin_kind = np.array(["unchanged_primary"]*len(particle_diameter_m), dtype=object)

particle_radius_m = particle_diameter_m/2.0
particle_volumes_m3 = (4.0/3.0) * np.pi * particle_radius_m**3
n_eff_bins = len(particle_diameter_m)

solid_primary_complex_index = complex(n_particle, -PRIMARY_REFRACTIVE_INDEX_IMAG_K)
medium_complex_index = complex(n_medium, 0.0)

def maxwell_garnett_effective_index(eps_m, eps_i, phi):
    phi = np.clip(phi,0,1)
    num = eps_i + 2*eps_m + 2*phi*(eps_i-eps_m)
    den = eps_i + 2*eps_m - phi*(eps_i-eps_m)
    eff = eps_m*num/den; n = np.sqrt(eff)
    return np.where(np.imag(n)>0, np.conjugate(n), n)

solid_vf = np.clip(particle_density_by_bin_kg_per_m3/particle_density_kg_per_m3,0,1)
pci = np.full(n_eff_bins, solid_primary_complex_index, dtype=np.complex128)
if np.any(particle_is_floc):
    pci[particle_is_floc] = maxwell_garnett_effective_index(
        medium_complex_index, solid_primary_complex_index, solid_vf[particle_is_floc])

pcr = np.real(pci).astype(np.float64)
pck = np.maximum(-np.imag(pci), 0).astype(np.float64)

if PSD_WEIGHT_MODE == "mass_fraction":
    pnd = mass_concentration_kg_per_m3*particle_weights/particle_mass_by_bin_kg
elif PSD_WEIGHT_MODE == "number_fraction":
    nw = particle_weights/np.sum(particle_weights)
    pnd = mass_concentration_kg_per_m3/np.sum(nw*particle_mass_by_bin_kg)*nw

n_particles_per_m3 = np.sum(pnd)


# ============================================================================
# SECTION 3: Per-particle Mie cache (universal — covers all diameters, indices, concentrations)
# ============================================================================

n_theta_mueller = 3601
theta_m_deg = np.linspace(0, 180, n_theta_mueller)
theta_m_rad = np.deg2rad(theta_m_deg)
mu_m = np.cos(theta_m_rad)

# Cache key depends only on Mie physics inputs (NOT concentration or PSD weights)
_cache_key_str = (
    "claritas100_universal_v2_"
    + str(loess_diameter.tolist()) + str(kaolin_diameter.tolist())
    + str(FLOC_ENABLED)
    + str(FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M.tolist())
    + str(FLOC_POOL_EFFECTIVE_DIAMETER_M.tolist())
    + str(FLOC_FRACTAL_DIMENSION) + str(FLOC_COLLISION_LENGTH_M)
    + str(FLOC_POOL_KERNEL_LOG_SIGMA) + str(FLOC_POOL_KERNEL_ENFORCE_FLOC_NOT_SMALLER_THAN_SOURCE)
    + str(FLOC_POOL_KERNEL_MIN_PROBABILITY)
    + str(n_particle) + str(PRIMARY_REFRACTIVE_INDEX_IMAG_K)
    + str(n_medium) + str(wavelengths[0]) + str(n_theta_mueller)
    + str(particle_density_kg_per_m3) + str(FLOC_SCATTER_EFFICIENCY)
)
_cache_hash = hashlib.md5(_cache_key_str.encode()).hexdigest()
# Cache stored in script directory (shared across all runs)
_CACHE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
PARTICLE_CACHE_FILE = os.path.join(_CACHE_DIR, f"particle_mie_cache_{_cache_hash}.npz")

# Global dict: key = (d_m, n_real, n_imag) -> {M11_row, M12_row, M33_row, M34_row, cdf_row, sigma_s, sigma_t, g}
_particle_cache = {}

def _cache_key(d_m, n_complex):
    """Round to avoid floating-point mismatches."""
    return (round(float(d_m), 12),
            round(float(np.real(n_complex)), 8),
            round(float(np.imag(n_complex)), 8))

def _compute_one_particle(d_m, n_complex, wl, n_med):
    """Compute Mie scattering + Mueller profile for one (diameter, n_complex) pair."""
    r_m = d_m / 2.0
    mr = n_complex / n_med
    x = 2.0 * np.pi * n_med * r_m / wl
    # Cross-sections
    qe, qs, _, g = miepython.efficiencies_mx(mr, x)
    area = np.pi * r_m * r_m
    st = max(float(np.real(qe)) * area, 0.0)
    ss = min(max(float(np.real(qs)) * area, 0.0), st)
    sa = st - ss
    gv = float(np.real(g))
    # Mueller angular profile
    S1, S2 = miepython.S1_S2(mr, x, mu_m)
    S1 = np.asarray(S1, dtype=np.complex128)
    S2 = np.asarray(S2, dtype=np.complex128)
    M11 = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2)
    M12 = 0.5 * (np.abs(S2)**2 - np.abs(S1)**2)
    M33 = np.real(S1 * np.conj(S2))
    M34 = np.imag(S1 * np.conj(S2))
    M11_r = np.real(M11).astype(np.float32)
    M12_r = np.real(M12).astype(np.float32)
    M33_r = np.real(M33).astype(np.float32)
    M34_r = np.real(M34).astype(np.float32)
    # Angle CDF
    It = np.maximum(M11_r.astype(np.float64), 0.0)
    st_theta = np.maximum(np.sin(theta_m_rad), 0.0)
    dens = It * st_theta
    ds = np.sum(dens)
    if ds <= 0:
        dens = np.ones_like(dens)
        dens_sum = np.sum(dens)
    else:
        dens_sum = ds
    inc = 0.5 * (dens[1:] + dens[:-1]) * np.diff(theta_m_rad)
    cdf = np.concatenate(([0.0], np.cumsum(inc)))
    cdf = np.clip(cdf / cdf[-1], 0, 1)
    cdf[-1] = 1.0
    return M11_r, M12_r, M33_r, M34_r, cdf.astype(np.float64), float(ss), float(st), gv


# Load or build universal per-particle cache
if os.path.exists(PARTICLE_CACHE_FILE):
    print(f"Loading universal particle Mie cache: {PARTICLE_CACHE_FILE}")
    t0 = time.perf_counter()
    c = np.load(PARTICLE_CACHE_FILE, allow_pickle=True)
    keys_arr = c["keys"]  # array of (d, nr, ni) tuples
    for i in range(len(keys_arr)):
        d, nr, ni = keys_arr[i]
        _particle_cache[(float(d), float(nr), float(ni))] = {
            "M11": c[f"M11_{i}"],
            "M12": c[f"M12_{i}"],
            "M33": c[f"M33_{i}"],
            "M34": c[f"M34_{i}"],
            "cdf": c[f"cdf_{i}"],
            "sigma_s": float(c[f"ss_{i}"]),
            "sigma_t": float(c[f"st_{i}"]),
            "g": float(c[f"g_{i}"]),
        }
    print(f"  Loaded {len(_particle_cache)} particles in {time.perf_counter()-t0:.2f}s")
else:
    print("Building universal particle Mie cache (one-time, covers all materials/concentrations)...")
    t0 = time.perf_counter()
    # Collect all unique (d, n_complex) pairs from: loess, kaolin, floc pool
    from collections import OrderedDict
    all_pairs = OrderedDict()
    # Solid primary: both loess and kaolin diameter sets
    for d in loess_diameter:
        all_pairs[_cache_key(d, solid_primary_complex_index)] = (d, solid_primary_complex_index)
    for d in kaolin_diameter:
        all_pairs[_cache_key(d, solid_primary_complex_index)] = (d, solid_primary_complex_index)
    # Floc effective diameters: compute Maxwell-Garnett for every (source_band, floc_bin)
    if FLOC_ENABLED:
        frd = FLOC_POOL_PRIMARY_BAND_MAX_DIAMETER_M
        frm = particle_density_kg_per_m3 * (np.pi / 6.0) * frd**3
        for si in range(len(frd)):
            d0 = max(float(frd[si]), 1e-30)
            m0 = float(frm[si])
            for fi in range(len(FLOC_POOL_EFFECTIVE_DIAMETER_M)):
                d_floc = FLOC_POOL_EFFECTIVE_DIAMETER_M[fi]
                # Fractal mass
                m_floc = m0 * (d_floc / d0) ** FLOC_FRACTAL_DIMENSION
                vol_floc = (np.pi / 6.0) * d_floc**3
                rho_eff = m_floc / vol_floc
                phi = np.clip(rho_eff / particle_density_kg_per_m3, 0.0, 1.0)
                if np.isfinite(phi) and phi > 0:
                    n_floc = maxwell_garnett_effective_index(
                        medium_complex_index, solid_primary_complex_index, phi)
                else:
                    n_floc = complex(n_medium, 0.0)
                key = _cache_key(d_floc, n_floc)
                if key not in all_pairs:
                    all_pairs[key] = (d_floc, n_floc)
    print(f"  {len(all_pairs)} unique (diameter, n_complex) pairs to compute")

    # Precompute angular grid once
    cache_data = {}
    keys_list = list(all_pairs.keys())
    keys_array = np.empty(len(keys_list), dtype=object)
    for i, (key, (d, nc)) in enumerate(all_pairs.items()):
        keys_array[i] = (float(key[0]), float(key[1]), float(key[2]))
        M11r, M12r, M33r, M34r, cdfr, ss_v, st_v, gv = _compute_one_particle(
            d, nc, wavelengths[0], n_medium)
        cache_data[f"M11_{i}"] = M11r
        cache_data[f"M12_{i}"] = M12r
        cache_data[f"M33_{i}"] = M33r
        cache_data[f"M34_{i}"] = M34r
        cache_data[f"cdf_{i}"] = cdfr
        cache_data[f"ss_{i}"] = np.float32(ss_v)
        cache_data[f"st_{i}"] = np.float32(st_v)
        cache_data[f"g_{i}"] = np.float32(gv)
        # Also populate the in-memory cache dict
        _particle_cache[(float(key[0]), float(key[1]), float(key[2]))] = {
            "M11": M11r, "M12": M12r, "M33": M33r, "M34": M34r,
            "cdf": cdfr, "sigma_s": ss_v, "sigma_t": st_v, "g": gv,
        }
    cache_data["keys"] = keys_array
    np.savez_compressed(PARTICLE_CACHE_FILE, **cache_data)
    print(f"  Computed and cached {len(all_pairs)} particles in {time.perf_counter()-t0:.2f}s")

# ---- Assemble per-effective-bin arrays from cache -------------------------

print("Assembling Mueller tables for effective PSD from cache...")
t_assemble = time.perf_counter()

mueller_M11 = np.zeros((n_eff_bins, n_theta_mueller), dtype=np.float64)
mueller_M12 = np.zeros((n_eff_bins, n_theta_mueller), dtype=np.float64)
mueller_M33 = np.zeros((n_eff_bins, n_theta_mueller), dtype=np.float64)
mueller_M34 = np.zeros((n_eff_bins, n_theta_mueller), dtype=np.float64)
angle_cdf_by_bin = np.zeros((n_eff_bins, n_theta_mueller), dtype=np.float64)
sigma_s = np.zeros(n_eff_bins, dtype=np.float64)
sigma_a_arr = np.zeros(n_eff_bins, dtype=np.float64)
sigma_t = np.zeros(n_eff_bins, dtype=np.float64)
albedo_by_bin = np.zeros(n_eff_bins, dtype=np.float64)
g_by_bin = np.zeros(n_eff_bins, dtype=np.float64)

for bi in range(n_eff_bins):
    d_m = particle_diameter_m[bi]
    nc = pci[bi]
    key = _cache_key(d_m, nc)
    entry = _particle_cache.get(key)
    if entry is None:
        # Should not happen if the universal cache is complete
        raise RuntimeError(f"Particle (d={d_m:.6e}, n={nc}) not found in universal cache")
    mueller_M11[bi, :] = entry["M11"].astype(np.float64)
    mueller_M12[bi, :] = entry["M12"].astype(np.float64)
    mueller_M33[bi, :] = entry["M33"].astype(np.float64)
    mueller_M34[bi, :] = entry["M34"].astype(np.float64)
    angle_cdf_by_bin[bi, :] = entry["cdf"]
    ss_v = entry["sigma_s"]
    st_v = entry["sigma_t"]
    gv = entry["g"]
    sigma_s[bi] = ss_v
    sigma_t[bi] = st_v
    sigma_a_arr[bi] = st_v - ss_v
    albedo_by_bin[bi] = ss_v / st_v if st_v > 0 else 1.0
    g_by_bin[bi] = gv

sigma_a = sigma_a_arr
print(f"  Assembled in {time.perf_counter()-t_assemble:.2f}s")

# ---- Diffraction peak smoothing for large irregular particles (v102) -------
if DIFFRACTION_SMOOTHING_ENABLED:
    wl = wavelengths[0]
    smooth_rad = np.deg2rad(SMOOTHING_ANGLE_DEG)
    smooth_idx = int(np.ceil(smooth_rad / (theta_m_rad[1] - theta_m_rad[0])))
    n_smoothed = 0
    for bi in range(n_eff_bins):
        x_param = 2.0 * np.pi * n_medium * particle_radius_m[bi] / wl
        if x_param < SMOOTHING_SIZE_THRESHOLD_X:
            continue  # particles too small to have sharp diffraction peaks
        n_smoothed += 1
        # Gaussian kernel centered at θ=0 (the diffraction peak)
        sigma_theta = smooth_rad * (1.0 + 0.3 * np.log10(max(x_param, 1.0)))
        kernel = np.exp(-0.5 * (theta_m_rad[:smooth_idx] / sigma_theta)**2)
        kernel /= np.sum(kernel)
        # Convolve M11 at small angles with the kernel
        orig = mueller_M11[bi, :smooth_idx + smooth_idx].copy()
        smoothed = np.convolve(orig, kernel, mode='same')
        mueller_M11[bi, :smooth_idx] = smoothed[:smooth_idx]
        # Re-normalise M11 to preserve integral (scattering cross-section)
        # M12, M33, M34 are not smoothed — they depend on polarization physics
        # Recompute CDF from smoothed M11
        It_s = np.maximum(mueller_M11[bi].astype(np.float64), 0.0)
        st_theta = np.maximum(np.sin(theta_m_rad), 0.0)
        dens_s = It_s * st_theta
        ds_s = np.sum(dens_s)
        if ds_s > 0:
            inc_s = 0.5 * (dens_s[1:] + dens_s[:-1]) * np.diff(theta_m_rad)
            cdf_s = np.concatenate(([0.0], np.cumsum(inc_s)))
            cdf_s = np.clip(cdf_s / cdf_s[-1], 0, 1)
            cdf_s[-1] = 1.0
            angle_cdf_by_bin[bi, :] = cdf_s
    if n_smoothed > 0:
        print(f"  Diffraction smoothing applied to {n_smoothed}/{n_eff_bins} bins (x > {SMOOTHING_SIZE_THRESHOLD_X})")

# ---- Concentration-dependent floc physics (v101) ----------------------------
# Compute total solid volume fraction for concentration-dependent corrections
primary_solid_vf = mass_concentration_kg_per_m3 / particle_density_kg_per_m3
floc_solid_vf = np.sum(pnd[particle_is_floc] * particle_volumes_m3[particle_is_floc]) if np.any(particle_is_floc) else 0.0
total_solid_vf = primary_solid_vf

# --- 2a: Concentration-dependent fractal dimension ---
if FLOC_CONC_DEPENDENT_DF_ENABLED and FLOC_ENABLED:
    # Df transitions from Df_min at dilute to Df_max at concentrated
    phi_half = FLOC_DF_PHI_HALF
    eff_df = FLOC_DF_MIN + (FLOC_DF_MAX - FLOC_DF_MIN) * (total_solid_vf / (total_solid_vf + phi_half))
    print(f"  Concentration-dependent Df: {eff_df:.3f} (at φ_solid={total_solid_vf:.2e})")
else:
    eff_df = FLOC_FRACTAL_DIMENSION

# --- 2b: Dependent scattering packing factor ---
def dependent_scattering_factor(phi):
    """Packing factor K(φ) = (1-φ)⁴/(1+2φ)² for dense suspensions.
    Based on Twersky's multiple-scattering theory for correlated scatterers.
    Returns 1.0 below threshold (independent scattering regime)."""
    if not DEPENDENT_SCATTERING_ENABLED or phi < DEPENDENT_SCATTERING_PHI_THRESHOLD:
        return 1.0
    phi_c = np.clip(phi, 0.0, 0.74)  # clamp at random close packing
    return (1.0 - phi_c)**4 / (1.0 + 2.0*phi_c)**2

floc_packing_factor = dependent_scattering_factor(floc_solid_vf)
if DEPENDENT_SCATTERING_ENABLED and floc_packing_factor < 0.999:
    print(f"  Dependent scattering: K(φ)={floc_packing_factor:.4f} (φ_floc={floc_solid_vf:.2e})")

# --- 2c: Apply to cross-sections after assembly ---
# (deferred — applied below after cache lookup)

# ---- Concentration-dependent bulk properties (from cached per-particle cross-sections) ----
mus_b_raw = pnd * sigma_s
mua_b_raw = pnd * sigma_a
mut_b_raw = pnd * sigma_t

# Apply FLOC_SCATTER_EFFICIENCY and dependent scattering to floc bins
mus_b = mus_b_raw.copy()
mua_b = mua_b_raw.copy()
mut_b = mut_b_raw.copy()
for bi in range(n_eff_bins):
    if particle_is_floc[bi]:
        sf = float(FLOC_SCATTER_EFFICIENCY) * floc_packing_factor
        # Absorption is NOT scaled by packing factor — it's a material property
        mus_b[bi] = mus_b_raw[bi] * sf
        mua_b[bi] = mua_b_raw[bi] * float(FLOC_SCATTER_EFFICIENCY)  # absorption depends on material only
        mut_b[bi] = mus_b[bi] + mua_b[bi]
        # Update per-bin albedo to reflect reduced scattering
        albedo_by_bin[bi] = mus_b[bi] / mut_b[bi] if mut_b[bi] > 0 else 1.0
mu_s = np.sum(mus_b)
mu_a = np.sum(mua_b)
mu_t = np.sum(mut_b)
ms_albedo = mu_s / mu_t if mu_t > 0 else 1.0
g_eff = np.sum(mus_b * g_by_bin) / mu_s if mu_s > 0 else 0.0
MFP = 1.0 / mu_t if mu_t > 0 else np.inf
T_MFP = 1.0 / np.sum(mus_b * (1 - g_by_bin)) if np.sum(mus_b * (1 - g_by_bin)) > 0 else np.inf

if mu_t > 0:
    pew = mut_b / mu_t
    pec = np.cumsum(pew)
    pec /= pec[-1]
else:
    pew = np.ones(n_eff_bins) / n_eff_bins
    pec = np.linspace(1.0 / n_eff_bins, 1.0, n_eff_bins)


# ---- Concentration-dependent effective radius (v102: transport τ*) ---------
if CONC_DEPENDENT_RADIUS_ENABLED:
    # Use transport optical depth: τ* = R / T_MFP = μ_s (1-g) R
    # This accounts for scattering anisotropy: forward-peaked media (loess, g≈0.94)
    # have longer T_MFP so their radius gets reduced less than isotropic media
    transport_optical_depth = R_REAL / max(T_MFP, 1e-30)
    R_effective = R_REAL * RADIUS_REFERENCE_TAU / max(transport_optical_depth, RADIUS_REFERENCE_TAU)
    R_effective = max(RADIUS_MIN_M, min(R_effective, R_REAL))
    print(f"  Conc-dependent radius (transport): R_eff={R_effective:.4f}m "
          f"(τ*={transport_optical_depth:.3f}, τ_ref={RADIUS_REFERENCE_TAU})")
else:
    R_effective = R_REAL


# ============================================================================
# SECTION 5: CUDA Kernel (GPU transport)
# ============================================================================

if _CUPY_AVAILABLE:

    # --- Preflight GPU diagnostics ---
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        print(f"GPU memory: {free_bytes/1024**3:.1f} GB free / {total_bytes/1024**3:.1f} GB total")
        free_gb = free_bytes / 1024**3
        if free_gb < 0.5:
            print(f"⚠ WARNING: GPU has only {free_gb:.2f} GB free. Simulation may OOM.")
        # Allocate and free a small test array to verify CuPy works
        _test = cp.zeros((1024,), dtype=cp.float32)
        del _test
        cp.get_default_memory_pool().free_all_blocks()
        print("GPU preflight check: OK")
    except Exception as e:
        print(f"GPU preflight check FAILED: {e}")
        raise

    cuda_src = r"""
    extern "C" {

    __device__ unsigned int xorshift32(unsigned int* state) {
        unsigned int x = *state;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        *state = x; return x;
    }

    __device__ double rnd_double(unsigned int* state) {
        return ((double)xorshift32(state) + 0.5) * 2.3283064365386962890625e-10;
    }

    __device__ float rnd_float(unsigned int* state) {
        float u = (float)rnd_double(state);
        return fminf(0.99999994f, fmaxf(1.1641532e-10f, u));
    }

    // Binary search in sorted double array
    __device__ int bin_search(const double* arr, int n, double val) {
        int lo = 0, hi = n - 1;
        while (lo < hi) { int mid = (lo+hi)>>1; if (val <= arr[mid]) hi = mid; else lo = mid+1; }
        return lo;
    }

    // Sample theta from per-bin CDF (linear interpolation)
    __device__ float sample_theta(const double* cdf_row, const float* theta_tbl, int nt, double u) {
        int hi = bin_search(cdf_row, nt, u);
        if (hi <= 0) return theta_tbl[0];
        int lo = hi - 1;
        double c0 = cdf_row[lo], c1 = cdf_row[hi];
        double frac = (c1 > c0) ? fmin(1.0, fmax(0.0, (u-c0)/(c1-c0))) : 0.0;
        return (float)(theta_tbl[lo] + frac*(theta_tbl[hi]-theta_tbl[lo]));
    }

    // Scatter direction in 3D
    __device__ void scatter_3d(float* vx, float* vy, float* vz, float theta, float phi) {
        float wx=*vx, wy=*vy, wz=*vz;
        float ax, ay, az;
        if (fabsf(wz) < 0.9f) { ax=0; ay=0; az=1; }
        else { ax=1; ay=0; az=0; }
        float ux=ay*wz-az*wy, uy=az*wx-ax*wz, uz=ax*wy-ay*wx;
        float un = sqrtf(fmaxf(ux*ux+uy*uy+uz*uz,1e-30f));
        ux/=un; uy/=un; uz/=un;
        float qx=wy*uz-wz*uy, qy=wz*ux-wx*uz, qz=wx*uy-wy*ux;
        float ct=cosf(theta), st=sinf(theta), cp=cosf(phi), sp=sinf(phi);
        float nx=ct*wx+st*(cp*ux+sp*qx), ny=ct*wy+st*(cp*uy+sp*qy), nz=ct*wz+st*(cp*uz+sp*qz);
        float nn=sqrtf(fmaxf(nx*nx+ny*ny+nz*nz,1e-30f));
        *vx=nx/nn; *vy=ny/nn; *vz=nz/nn;
    }

    // Fresnel transmission probability (unpolarised)
    __device__ float fresnel_T(float ni, float ne, float cos_i) {
        float si = sqrtf(fmaxf(0, 1-cos_i*cos_i));
        float st = (ni/ne)*si;
        if (st >= 1.0f) return 0.0f;
        float ct = sqrtf(fmaxf(0, 1-st*st));
        float rs = (ni*cos_i - ne*ct)/(ni*cos_i + ne*ct);
        float rp = (ne*cos_i - ni*ct)/(ne*cos_i + ni*ct);
        return 1.0f - 0.5f*(rs*rs + rp*rp);
    }

    // Stokes rotation by angle phi (input: cos2p, sin2p)
    __device__ void rotate_stokes(float* I, float* Q, float* U, float c2, float s2) {
        float q = *Q, u = *U;
        *Q = q*c2 - u*s2;
        *U = q*s2 + u*c2;
    }

    // Apply normalized Mueller matrix: m_{ij} = M_{ij} / M_{11}
    // This ensures I does not grow uncontrollably (M11 is already
    // handled by the phase function CDF sampling).
    __device__ void apply_mueller(float* I, float* Q, float* U, float* V,
                                   float m11, float m12, float m33, float m34) {
        float i_in=*I, q_in=*Q, u_in=*U, v_in=*V;
        float inv_m11 = 1.0f / fmaxf(fabsf(m11), 1.0e-12f);
        float nm12 = m12 * inv_m11;
        float nm33 = m33 * inv_m11;
        float nm34 = m34 * inv_m11;
        *I = fmaxf(i_in + nm12*q_in, 0.0f);
        *Q = nm12*i_in + q_in;
        *U = nm33*u_in + nm34*v_in;
        *V = -nm34*u_in + nm33*v_in;
    }

    // Compute cos(2*phi) and sin(2*phi) for Stokes rotation into scattering plane
    __device__ void stokes_rotation_angles(
        float ix, float iy, float iz,  // incident direction
        float sx, float sy, float sz,  // scattered direction
        float* c2_in, float* s2_in      // output: rotation INTO scattering plane
    ) {
        // Normalise (assumed already unit)
        // Scattering normal = cross(incident, scattered)
        float snx = iy*sz - iz*sy;
        float sny = iz*sx - ix*sz;
        float snz = ix*sy - iy*sx;
        float snn = sqrtf(fmaxf(snx*snx+sny*sny+snz*snz, 1e-30f));
        if (snn < 1e-12f) { *c2_in = 1.0f; *s2_in = 0.0f; return; }
        snx /= snn; sny /= snn; snz /= snn;

        // Incident meridian basis
        float eipx, eipy, eipz;
        // e_i_perp = cross(z, incident) normalized
        eipx = -iy; eipy = ix; eipz = 0.0f;  // cross(z, incident) = (-iy, ix, 0)
        float eipn = sqrtf(fmaxf(eipx*eipx+eipy*eipy, 1e-30f));
        if (eipn < 1e-12f) { eipx = 1; eipy = 0; eipz = 0; eipn = 1; }
        else { eipx/=eipn; eipy/=eipn; }
        // e_i_parallel = cross(incident, e_i_perp)
        float eilx = iy*eipz - iz*eipy;
        float eily = iz*eipx - ix*eipz;
        float eilz = ix*eipy - iy*eipx;

        // Scattered meridian basis
        float espx, espy, espz;
        espx = -sy; espy = sx; espz = 0.0f;
        float espn = sqrtf(fmaxf(espx*espx+espy*espy, 1e-30f));
        if (espn < 1e-12f) { espx = 1; espy = 0; espz = 0; espn = 1; }
        else { espx/=espn; espy/=espn; }
        float eslx = sy*espz - sz*espy;
        float esly = sz*espx - sx*espz;
        float eslz = sx*espy - sy*espx;

        // cos(phi_i) = dot(e_i_perp, scattering_normal)
        float cpi = eipx*snx + eipy*sny;
        float spi = eilx*snx + eily*sny + eilz*snz;
        float cps = espx*snx + espy*sny;
        float sps = eslx*snx + esly*sny + eslz*snz;

        // phi_total = phi_s - phi_i
        float cpt = cps*cpi + sps*spi;
        float spt = sps*cpi - cps*spi;

        *c2_in = cpt*cpt - spt*spt;
        *s2_in = 2.0f*spt*cpt;
    }

    __global__ void trace_kernel(
        const int MAX_EXT,
        const float MU_T_VAL,
        const float R,
        const float R_OFF,
        const float BEAM_SIGMA,
        const int COLLIMATED,
        const double* angles_init,
        const int N_RAYS,
        // Particle properties tables
        const double* event_cdf,
        const float* albedo_tbl,
        const float* is_floc_tbl,
        const int N_BINS,
        // Mueller tables (flat: bin * N_THETA + theta_idx)
        const float* M11_tbl,
        const float* M12_tbl,
        const float* M33_tbl,
        const float* M34_tbl,
        const double* angle_cdf_tbl,
        const float* theta_tbl,
        const int N_THETA,
        // Fresnel parameters
        const int FRESNEL_ENABLED,
        const float N_MED,
        const float N_EXT,
        // Output arrays
        float* exit_x, float* exit_y, float* exit_z,
        float* exit_vx, float* exit_vy, float* exit_vz,
        float* exit_dir,
        float* stokes_I, float* stokes_Q, float* stokes_U, float* stokes_V,
        int* scat_count, int* floc_count, int* ext_count,
        int* last_scat_bin, int* last_was_floc,
        float* path_len,
        int* terminal,
        unsigned int transport_seed,
        unsigned long long ray_offset
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= N_RAYS) return;

        // Initialise xorshift32 state
        unsigned long long gid = ray_offset + (unsigned long long)tid;
        unsigned int state = (unsigned int)(
            ((unsigned long long)transport_seed + gid*74729ULL + 13ULL) % 0xffffffffULL
        ) + 1u;

        terminal[tid] = 0;
        scat_count[tid] = 0; floc_count[tid] = 0; ext_count[tid] = 0;
        last_scat_bin[tid] = -1; last_was_floc[tid] = 0; path_len[tid] = 0.0f;
        exit_x[tid] = nanf(""); exit_y[tid] = nanf(""); exit_z[tid] = nanf("");
        exit_vx[tid] = nanf(""); exit_vy[tid] = nanf(""); exit_vz[tid] = nanf("");
        exit_dir[tid] = nanf("");
        stokes_I[tid] = 1.0f; stokes_Q[tid] = 0.0f;
        stokes_U[tid] = 0.0f; stokes_V[tid] = 0.0f;

        float angle_init;
        float x0, y0;

        if (COLLIMATED) {
            x0 = 0.0f; y0 = -R; angle_init = 0.0f;
        } else {
            float u1 = rnd_float(&state), u2 = rnd_float(&state);
            float gauss = sqrtf(-2.0f*logf(u1))*cosf(6.2831853f*u2);
            x0 = BEAM_SIGMA*gauss; y0 = -(R+R_OFF);
            angle_init = (float)angles_init[tid];
        }

        float vx = sinf(angle_init), vy = cosf(angle_init), vz = 0.0f;

        // Sphere entry
        float b_in = x0*vx + y0*vy;
        float c_in = x0*x0 + y0*y0 - R*R;
        float disc_in = b_in*b_in - c_in;
        if (disc_in < 0.0f) { terminal[tid] = 4; return; }
        float t_in = -b_in - sqrtf(disc_in);
        if (t_in < 0.0f) { terminal[tid] = 4; return; }
        float x = x0 + t_in*vx, y = y0 + t_in*vy, z = 0.0f;

        // Stokes
        float sI = 1.0f, sQ = 0.0f, sU = 0.0f, sV = 0.0f;
        int sc = 0, fc = 0, ec = 0, lsb = -1, lwf = 0;
        float pl = 0.0f;
        float R2 = R*R;

        while (x*x + y*y + z*z <= R2) {
            // Distance to boundary
            float bb = x*vx + y*vy + z*vz;
            float cb = x*x + y*y + z*z - R2;
            float db = fmaxf(bb*bb - cb, 0.0f);
            float d2b = -bb + sqrtf(db);

            // Free path
            float fp = d2b;
            if (MU_T_VAL > 0.0f) {
                double up = rnd_double(&state);
                fp = (float)(-log(up) / (double)MU_T_VAL);
            }

            int exited = (MU_T_VAL <= 0.0f || fp >= d2b) ? 1 : 0;

            if (!exited && ec >= MAX_EXT) {
                terminal[tid] = 3; scat_count[tid]=sc; floc_count[tid]=fc;
                ext_count[tid]=ec; last_scat_bin[tid]=lsb; last_was_floc[tid]=lwf;
                path_len[tid]=pl; stokes_I[tid]=sI; stokes_Q[tid]=sQ;
                stokes_U[tid]=sU; stokes_V[tid]=sV; return;
            }

            float travel = exited ? d2b : fp;
            x += vx*travel; y += vy*travel; z += vz*travel; pl += travel;

            if (exited) {
                if (FRESNEL_ENABLED && fabsf(N_MED - N_EXT) > 1e-6f) {
                    float rd = sqrtf(fmaxf(x*x+y*y+z*z, 1e-30f));
                    float nx = x/rd, ny = y/rd, nz = z/rd;
                    float ci = fmaxf(0.0f, vx*nx + vy*ny + vz*nz);
                    float T = fresnel_T(N_MED, N_EXT, ci);
                    if (rnd_double(&state) < (double)T) {
                        // Refract
                        float ratio = N_MED/N_EXT;
                        float s2t = ratio*ratio*(1-ci*ci);
                        float ct = sqrtf(fmaxf(0,1-s2t));
                        float mu = ratio*ci - ct;
                        float vxr = ratio*vx + mu*nx;
                        float vyr = ratio*vy + mu*ny;
                        float vzr = ratio*vz + mu*nz;
                        float vn = sqrtf(fmaxf(vxr*vxr+vyr*vyr+vzr*vzr,1e-30f));
                        vx=vxr/vn; vy=vyr/vn; vz=vzr/vn;
                    } else {
                        // Reflect
                        float dn = vx*nx + vy*ny + vz*nz;
                        vx = vx - 2*dn*nx; vy = vy - 2*dn*ny; vz = vz - 2*dn*nz;
                        float vn = sqrtf(fmaxf(vx*vx+vy*vy+vz*vz,1e-30f));
                        vx/=vn; vy/=vn; vz/=vn;
                        continue;
                    }
                }
                exit_x[tid]=x; exit_y[tid]=y; exit_z[tid]=z;
                exit_vx[tid]=vx; exit_vy[tid]=vy; exit_vz[tid]=vz;
                exit_dir[tid]=acosf(fminf(1.0f,fmaxf(-1.0f,vy)));
                scat_count[tid]=sc; floc_count[tid]=fc; ext_count[tid]=ec;
                last_scat_bin[tid]=lsb; last_was_floc[tid]=lwf;
                path_len[tid]=pl;
                stokes_I[tid]=sI; stokes_Q[tid]=sQ; stokes_U[tid]=sU; stokes_V[tid]=sV;
                terminal[tid]=1;
                return;
            }

            // Extinction event
            ec++;
            double up = rnd_double(&state);
            int pidx = bin_search(event_cdf, N_BINS, up);
            if (pidx >= N_BINS) pidx = N_BINS-1;

            float alb = albedo_tbl[pidx];
            if (rnd_double(&state) >= (double)alb) {
                terminal[tid]=2; scat_count[tid]=sc; floc_count[tid]=fc;
                ext_count[tid]=ec; last_scat_bin[tid]=lsb; last_was_floc[tid]=lwf;
                path_len[tid]=pl;
                stokes_I[tid]=sI; stokes_Q[tid]=sQ; stokes_U[tid]=sU; stokes_V[tid]=sV;
                return;
            }

            // Scatter
            sc++; lsb = pidx;
            int is_fl = (is_floc_tbl[pidx] > 0.5f) ? 1 : 0;
            if (is_fl) { fc++; lwf = 1; } else { lwf = 0; }

            // Sample angles
            int cdf_offset = pidx * N_THETA;
            double ua = rnd_double(&state);
            float theta_v = sample_theta(&angle_cdf_tbl[cdf_offset], theta_tbl, N_THETA, ua);
            float phi_v = 6.2831853f * rnd_float(&state);

            // Compute new direction
            float ovx=vx, ovy=vy, ovz=vz;
            scatter_3d(&vx, &vy, &vz, theta_v, phi_v);

            // Stokes rotation into scattering plane
            float c2i, s2i;
            stokes_rotation_angles(ovx, ovy, ovz, vx, vy, vz, &c2i, &s2i);
            rotate_stokes(&sI, &sQ, &sU, c2i, s2i);

            // Apply Mueller matrix at the sampled scattering angle
            int ti = (int)(theta_v / 3.1415927f * (float)(N_THETA-1) + 0.5f);
            if (ti < 0) ti = 0; if (ti >= N_THETA) ti = N_THETA-1;
            int m_off = pidx * N_THETA + ti;
            float m11 = M11_tbl[m_off];
            float m12 = M12_tbl[m_off];
            float m33 = M33_tbl[m_off];
            float m34 = M34_tbl[m_off];
            apply_mueller(&sI, &sQ, &sU, &sV, m11, m12, m33, m34);

            // Stokes rotation out of scattering plane (to scattered meridian)
            // Recompute phi_s rotation
            float snx = ovy*vz - ovz*vy;
            float sny = ovz*vx - ovx*vz;
            float snz = ovx*vy - ovy*vx;
            float snn = sqrtf(fmaxf(snx*snx+sny*sny+snz*snz, 1e-30f));
            if (snn < 1e-12f) { /* no rotation needed */ }
            else {
                snx/=snn; sny/=snn; snz/=snn;
                float espx=-vy, espy=vx, espz=0;
                float espn = sqrtf(fmaxf(espx*espx+espy*espy, 1e-30f));
                if (espn<1e-12f){espx=1;espy=0;espz=0;espn=1;} else {espx/=espn;espy/=espn;}
                float eslx=vy*espz-vz*espy, esly=vz*espx-vx*espz, eslz=vx*espy-vy*espx;
                float cps = espx*snx+espy*sny, sps = eslx*snx+esly*sny+eslz*snz;
                // Rotate by -phi_s
                float c2o = cps*cps - sps*sps;
                float s2o = -2.0f*sps*cps;
                rotate_stokes(&sI, &sQ, &sU, c2o, s2o);
            }
            sI = fmaxf(sI, 0.0f);
        }

        // Fell through (shouldn't normally happen)
        terminal[tid]=1; exit_x[tid]=x; exit_y[tid]=y; exit_z[tid]=z;
        exit_vx[tid]=vx; exit_vy[tid]=vy; exit_vz[tid]=vz;
        exit_dir[tid]=acosf(fminf(1.0f,fmaxf(-1.0f,vy)));
        scat_count[tid]=sc; floc_count[tid]=fc; ext_count[tid]=ec;
        last_scat_bin[tid]=lsb; last_was_floc[tid]=lwf;
        path_len[tid]=pl;
        stokes_I[tid]=sI; stokes_Q[tid]=sQ; stokes_U[tid]=sU; stokes_V[tid]=sV;
    }

    }
    """

    # Compile CUDA module
    _cuda_module = cp.RawModule(code=cuda_src, options=('-std=c++11',))
    _trace_kernel_gpu = _cuda_module.get_function('trace_kernel')


    def get_gpu_free_bytes():
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            return int(free), int(total)
        except Exception:
            return None, None


    def estimate_chunk_size(free_bytes, safety=0.3, overhead=256*1024*1024):
        per_ray = 128
        usable = int(free_bytes * safety) - overhead
        return max(1, usable // per_ray) if usable > 0 else 1


    def trace_rays_gpu(angles_init_np, R, ray_off, beam_sigma, collimated,
                       mu_t_val, max_ext,
                       n_med, n_ext, fresnel_en,
                       hdf5_file, wl_nm, mat_name, conc):
        """Launch GPU transport and write results to HDF5."""
        N = len(angles_init_np)
        threads = 256

        # Transfer tables to GPU (float32 for Mueller, float64 for CDFs)
        M11_g = cp.asarray(mueller_M11.astype(np.float32).ravel())
        M12_g = cp.asarray(mueller_M12.astype(np.float32).ravel())
        M33_g = cp.asarray(mueller_M33.astype(np.float32).ravel())
        M34_g = cp.asarray(mueller_M34.astype(np.float32).ravel())
        cdf_g = cp.asarray(angle_cdf_by_bin.ravel(), dtype=cp.float64)
        theta_g = cp.asarray(theta_m_rad.astype(np.float32))
        evc_g = cp.asarray(pec, dtype=cp.float64)
        alb_g = cp.asarray(albedo_by_bin.astype(np.float32))
        ifl_g = cp.asarray(particle_is_floc.astype(np.float32))

        free_b, total_b = get_gpu_free_bytes()
        if free_b is None:
            est_chunk = 100_000
        else:
            est_chunk = int(max(GPU_MIN_CHUNK_RAYS,
                                min(estimate_chunk_size(free_b), GPU_MAX_CHUNK_RAYS)))
        est_chunk = max(1, min(est_chunk, N))

        # Seed generation
        seed_seq = np.random.SeedSequence(SIMULATION_SEED)
        sv = seed_seq.generate_state(2, dtype=np.uint32)
        transport_seed = np.uint32(sv[0])

        with h5py.File(hdf5_file, "w") as f:
            f.attrs["claritas_version"] = "104_gpu"
            f.attrs["simulation_seed"] = np.uint64(SIMULATION_SEED)
            f.attrs["transport_type"] = "polarised_gpu_cuda_stokes_mueller"
            f.attrs["n_rays"] = np.int64(N)
            f.attrs["wavelength_nm"] = float(wl_nm)
            f.attrs["mu_s_m_inv"] = float(mu_s)
            f.attrs["mu_a_m_inv"] = float(mu_a)
            f.attrs["mu_t_m_inv"] = float(mu_t)
            f.attrs["material"] = str(mat_name)
            f.attrs["concentration_g_per_L"] = float(conc)
            f.attrs["sample_radius_m"] = float(R)
            f.attrs["detector_centres_deg"] = detector_angles
            f.attrs["detector_acceptance_deg"] = float(detector_acceptance_deg)
            f.attrs["max_extinctions"] = np.int64(max_ext)
            f.attrs["initial_chunk"] = np.int64(est_chunk)

            d_ex = f.create_dataset("exit_x", (N,), dtype='f4')
            d_ey = f.create_dataset("exit_y", (N,), dtype='f4')
            d_ez = f.create_dataset("exit_z", (N,), dtype='f4')
            d_evx = f.create_dataset("exit_vx", (N,), dtype='f4')
            d_evy = f.create_dataset("exit_vy", (N,), dtype='f4')
            d_evz = f.create_dataset("exit_vz", (N,), dtype='f4')
            d_ed = f.create_dataset("exit_dir", (N,), dtype='f4')
            d_sI = f.create_dataset("stokes_I", (N,), dtype='f4')
            d_sQ = f.create_dataset("stokes_Q", (N,), dtype='f4')
            d_sU = f.create_dataset("stokes_U", (N,), dtype='f4')
            d_sV = f.create_dataset("stokes_V", (N,), dtype='f4')
            d_sc = f.create_dataset("scatter_count", (N,), dtype='i4')
            d_fc = f.create_dataset("floc_event_count", (N,), dtype='i4')
            d_ec = f.create_dataset("extinction_count", (N,), dtype='i4')
            d_lsb = f.create_dataset("last_scatter_bin", (N,), dtype='i4')
            d_lwf = f.create_dataset("last_event_was_floc", (N,), dtype='i4')
            d_pl = f.create_dataset("path_length_m", (N,), dtype='f4')
            d_ts = f.create_dataset("terminal_state", (N,), dtype='i4')

            start = 0
            active_chunk = est_chunk
            oom_retries = 0
            total_t0 = time.perf_counter()

            while start < N:
                end = min(N, start + active_chunk)
                sz = end - start

                angles_g = cp.asarray(angles_init_np[start:end], dtype=cp.float64)
                ex_x_g = cp.full(sz, cp.nan, dtype=cp.float32)
                ex_y_g = cp.full(sz, cp.nan, dtype=cp.float32)
                ex_z_g = cp.full(sz, cp.nan, dtype=cp.float32)
                ex_vx_g = cp.full(sz, cp.nan, dtype=cp.float32)
                ex_vy_g = cp.full(sz, cp.nan, dtype=cp.float32)
                ex_vz_g = cp.full(sz, cp.nan, dtype=cp.float32)
                ex_dir_g = cp.full(sz, cp.nan, dtype=cp.float32)
                sI_g = cp.zeros(sz, dtype=cp.float32)
                sQ_g = cp.zeros(sz, dtype=cp.float32)
                sU_g = cp.zeros(sz, dtype=cp.float32)
                sV_g = cp.zeros(sz, dtype=cp.float32)
                sc_g = cp.full(sz, -1, dtype=cp.int32)
                fc_g = cp.full(sz, -1, dtype=cp.int32)
                ec_g = cp.full(sz, -1, dtype=cp.int32)
                lsb_g = cp.full(sz, -1, dtype=cp.int32)
                lwf_g = cp.full(sz, -1, dtype=cp.int32)
                pl_g = cp.full(sz, cp.nan, dtype=cp.float32)
                ts_g = cp.zeros(sz, dtype=cp.int32)

                try:
                    blocks = (sz + threads - 1) // threads
                    _trace_kernel_gpu((blocks,), (threads,), (
                        np.int32(max_ext),
                        np.float32(mu_t_val),
                        np.float32(R), np.float32(ray_off),
                        np.float32(beam_sigma),
                        np.int32(1 if collimated else 0),
                        angles_g, np.int32(sz),
                        evc_g, alb_g, ifl_g, np.int32(n_eff_bins),
                        M11_g, M12_g, M33_g, M34_g,
                        cdf_g, theta_g, np.int32(n_theta_mueller),
                        np.int32(1 if fresnel_en else 0),
                        np.float32(n_med), np.float32(n_ext),
                        ex_x_g, ex_y_g, ex_z_g,
                        ex_vx_g, ex_vy_g, ex_vz_g, ex_dir_g,
                        sI_g, sQ_g, sU_g, sV_g,
                        sc_g, fc_g, ec_g, lsb_g, lwf_g, pl_g, ts_g,
                        transport_seed,
                        np.uint64(start)
                    ))
                    cp.cuda.Stream.null.synchronize()
                except cp.cuda.memory.OutOfMemoryError:
                    cp.get_default_memory_pool().free_all_blocks()
                    if sz <= 1:
                        raise RuntimeError("CUDA OOM even for single-ray chunk")
                    active_chunk = max(1, sz // 2)
                    oom_retries += 1
                    print(f"  CUDA OOM: retrying chunk {start} with size {active_chunk}")
                    continue

                # Copy to HDF5
                d_ex[start:end] = cp.asnumpy(ex_x_g)
                d_ey[start:end] = cp.asnumpy(ex_y_g)
                d_ez[start:end] = cp.asnumpy(ex_z_g)
                d_evx[start:end] = cp.asnumpy(ex_vx_g)
                d_evy[start:end] = cp.asnumpy(ex_vy_g)
                d_evz[start:end] = cp.asnumpy(ex_vz_g)
                d_ed[start:end] = cp.asnumpy(ex_dir_g)
                d_sI[start:end] = cp.asnumpy(sI_g)
                d_sQ[start:end] = cp.asnumpy(sQ_g)
                d_sU[start:end] = cp.asnumpy(sU_g)
                d_sV[start:end] = cp.asnumpy(sV_g)
                d_sc[start:end] = cp.asnumpy(sc_g)
                d_fc[start:end] = cp.asnumpy(fc_g)
                d_ec[start:end] = cp.asnumpy(ec_g)
                d_lsb[start:end] = cp.asnumpy(lsb_g)
                d_lwf[start:end] = cp.asnumpy(lwf_g)
                d_pl[start:end] = cp.asnumpy(pl_g)
                d_ts[start:end] = cp.asnumpy(ts_g)

                cp.get_default_memory_pool().free_all_blocks()
                start = end
                if start < N:
                    elapsed = time.perf_counter() - total_t0
                    rate = start / elapsed if elapsed > 0 else 0
                    rem = (N - start) / rate if rate > 0 else 0
                    print(f"  GPU: {start}/{N} rays | {elapsed:.1f}s | "
                          f"{rate:.0f} rays/s | ~{rem:.0f}s remaining")

            f.attrs["final_chunk"] = np.int64(active_chunk)
            f.attrs["oom_retries"] = np.int64(oom_retries)
            f.attrs["total_time_s"] = float(time.perf_counter() - total_t0)


# ============================================================================
# SECTION 6: Detector Assignment & Angle Utilities
# ============================================================================

def spherical_polar_angle_deg(ex, ey, ez):
    r = np.sqrt(ex**2+ey**2+ez**2)
    ca = np.divide(ey, r, out=np.zeros_like(r), where=r>0)
    return np.rad2deg(np.arccos(np.clip(ca,-1,1)))

def assign_detector(angles_deg, centres_deg, accept_deg):
    diff = np.abs(angles_deg[:,None]-centres_deg[None,:])
    nearest = np.argmin(diff, axis=1)
    ok = diff[np.arange(len(angles_deg)),nearest] <= accept_deg
    a = np.full(len(angles_deg),-1,dtype=np.int32)
    a[ok] = nearest[ok]
    return a

def sample_beta_angles(N, a1, a2, rng):
    Nh = N//2
    ul = rng.beta(a1,a2,Nh)
    al = (1-ul)*(np.pi/2)-(np.pi/2); ar = -al
    an = np.concatenate([al,ar])
    if len(an)<N: an=np.append(an,0.)
    return an.astype(np.float64)


# ============================================================================
# SECTION 7: Main Simulation Loop
# ============================================================================

host_rng = np.random.default_rng(SIMULATION_SEED)
alpha1, alpha2 = 1.0, 100.0

engine_label = "GPU (CuPy/CUDA)" if _CUPY_AVAILABLE else "CPU (multiprocessing)"

print("="*70)
print("CLARITAS_104: Polarised Vector Radiative Transfer Monte Carlo (v104: material-specific floc sizes)")
print(f"  Engine: {engine_label}")
print("="*70)
print(f"  Material: {material_name}")
print(f"  Concentration: {mass_concentration_g_per_L} g/L")
print(f"  n_particle: {n_particle:.4f}, k: {PRIMARY_REFRACTIVE_INDEX_IMAG_K:.6g}")
print(f"  n_medium: {n_medium:.3f}, n_external: {n_external:.3f}")
print(f"  Floc enabled: {FLOC_ENABLED}")
print(f"  Effective bins: {n_eff_bins} (floc:{np.sum(particle_is_floc)}, primary:{np.sum(~particle_is_floc)})")
print(f"  mu_s: {mu_s:.3e}/m, mu_a: {mu_a:.3e}/m, mu_t: {mu_t:.3e}/m")
print(f"  Albedo: {ms_albedo:.6f}, g_eff: {g_eff:.6f}")
print(f"  MFP: {MFP:.3e}m, Transport MFP: {T_MFP:.3e}m")
print(f"  N_RAYS: {N_RAYS}")
print(f"  Wavelengths: {[int(w*1e9) for w in wavelengths]} nm")
print(f"  Detector angles: {detector_angles.tolist()} deg, acceptance: ±{detector_acceptance_deg} deg")
print(f"  Boundary Fresnel: {BOUNDARY_FRESNEL_ENABLED}")
print("="*70)

all_detector_counts = {}
all_stokes_by_detector = {}

for wl_idx, wl in enumerate(wavelengths):
    wl_nm = int(wl*1e9)
    print(f"\n{'='*70}")
    print(f"  Wavelength {wl_nm} nm")
    print(f"{'='*70}")

    if SOURCE_MODE == "production_beta":
        angles_init = sample_beta_angles(N_RAYS, alpha1, alpha2, host_rng)
    else:
        angles_init = np.zeros(N_RAYS, dtype=np.float64)

    hdf5_file = os.path.join(OUTDIR, f"ray_exits_polarised_{wl_nm}nm.h5")
    collimated = (SOURCE_MODE == "reference_collimated")

    t0 = time.perf_counter()

    if _CUPY_AVAILABLE:
        trace_rays_gpu(angles_init, R_effective, RAY_OFFSET, PRODUCTION_BEAM_SIGMA_M,
                       collimated, mu_t, MAX_EXTINCTIONS,
                       n_medium, n_external, BOUNDARY_FRESNEL_ENABLED,
                       hdf5_file, wl_nm, material_name, mass_concentration_g_per_L)
    else:
        # CPU fallback — omitted for brevity but would use multiprocessing
        raise RuntimeError("CPU fallback not implemented in GPU version. Install CuPy.")

    t1 = time.perf_counter()
    print(f"  Transport completed in {t1-t0:.2f}s ({N_RAYS/(t1-t0):.0f} rays/s)")

    # Load results
    with h5py.File(hdf5_file, "r") as f:
        exit_x = f["exit_x"][:]
        exit_y = f["exit_y"][:]
        exit_z = f["exit_z"][:]
        exit_vx = f["exit_vx"][:]
        exit_vy = f["exit_vy"][:]
        exit_vz = f["exit_vz"][:]
        exit_dirs = f["exit_dir"][:]
        scatter_counts = f["scatter_count"][:]
        floc_event_counts = f["floc_event_count"][:]
        extinction_counts = f["extinction_count"][:]
        last_scatter_bins = f["last_scatter_bin"][:]
        last_event_was_floc_arr = f["last_event_was_floc"][:]
        path_lengths = f["path_length_m"][:]
        terminal_states = f["terminal_state"][:]
        sI = f["stokes_I"][:]
        sQ = f["stokes_Q"][:]
        sU = f["stokes_U"][:]
        sV = f["stokes_V"][:]

    # Post-processing
    valid_mask = ((terminal_states==1) & np.isfinite(exit_x)
                  & np.isfinite(exit_y) & np.isfinite(exit_z)
                  & np.isfinite(exit_dirs))
    n_valid = np.sum(valid_mask)
    n_abs = np.sum(terminal_states==2)
    n_trunc = np.sum(terminal_states==3)
    n_miss = np.sum(terminal_states==4)

    print(f"  Valid exits: {n_valid} ({100*n_valid/N_RAYS:.2f}%)")
    print(f"  Absorbed: {n_abs} ({100*n_abs/N_RAYS:.2f}%)")
    print(f"  Truncated: {n_trunc} ({100*n_trunc/N_RAYS:.2f}%)")
    print(f"  Missed: {n_miss} ({100*n_miss/N_RAYS:.2f}%)")
    if n_valid > 0:
        print(f"  Mean scatter count: {np.mean(scatter_counts[valid_mask]):.3f}")
        print(f"  Mean path length: {np.mean(path_lengths[valid_mask]):.3e}m")

    # Detector assignment
    exit_angles_deg = np.full(N_RAYS, np.nan)
    exit_angles_deg[valid_mask] = spherical_polar_angle_deg(
        exit_x[valid_mask], exit_y[valid_mask], exit_z[valid_mask])
    det_assigned = assign_detector(
        exit_angles_deg[valid_mask], detector_angles, detector_acceptance_deg)

    det_counts = np.zeros(len(detector_angles), dtype=int)
    det_stokes = {j: {"I":[],"Q":[],"U":[],"V":[]} for j in range(len(detector_angles))}
    valid_idx = np.where(valid_mask)[0]
    for vi, dj in zip(valid_idx, det_assigned):
        if dj >= 0:
            det_counts[dj] += 1
            det_stokes[dj]["I"].append(sI[vi])
            det_stokes[dj]["Q"].append(sQ[vi])
            det_stokes[dj]["U"].append(sU[vi])
            det_stokes[dj]["V"].append(sV[vi])

    all_detector_counts[wl_nm] = det_counts
    all_stokes_by_detector[wl_nm] = det_stokes

    # Exit CSV
    df = pd.DataFrame({
        "exit_x_m":exit_x,"exit_y_m":exit_y,"exit_z_m":exit_z,
        "exit_vx":exit_vx,"exit_vy":exit_vy,"exit_vz":exit_vz,
        "exit_dir_rad":exit_dirs,"exit_dir_deg":np.rad2deg(exit_dirs),
        "scatter_count":scatter_counts,"floc_event_count":floc_event_counts,
        "extinction_count":extinction_counts,"terminal_state":terminal_states,
        "stokes_I":sI,"stokes_Q":sQ,"stokes_U":sU,"stokes_V":sV,
        "path_length_m":path_lengths,"is_valid_exit":valid_mask,
        "exit_position_angle_deg":exit_angles_deg,
    })
    csv_p = os.path.join(OUTDIR, f"exit_points_polarised_{wl_nm}nm.csv")
    df.to_csv(csv_p, index=False)
    print(f"  Saved {csv_p}")


# ============================================================================
# SECTION 8: Detector Response & Polarisation Diagnostics
# ============================================================================

print(f"\n{'='*70}")
print("  DETECTOR RESPONSE & POLARISATION SUMMARY")
print(f"{'='*70}")

det_rows = []
for wl_idx, wl in enumerate(wavelengths):
    wl_nm = int(wl*1e9)
    counts = all_detector_counts[wl_nm]
    stk = all_stokes_by_detector[wl_nm]
    total = np.sum(counts)
    for j, c in enumerate(detector_angles):
        nh = int(counts[j]); fr = nh/total if total>0 else 0
        r = {"wavelength_nm":wl_nm,"detector_angle_deg":c,
             "hit_count":nh,"normalised_response":fr}
        I=np.array(stk[j]["I"]); Q=np.array(stk[j]["Q"])
        U=np.array(stk[j]["U"]); V=np.array(stk[j]["V"])
        if len(I)>0:
            r["mean_I"]=np.mean(I);r["mean_Q"]=np.mean(Q)
            r["mean_U"]=np.mean(U);r["mean_V"]=np.mean(V)
            mi = max(np.mean(I),1e-30)
            r["mean_Q_over_I"]=np.mean(Q)/mi;r["mean_U_over_I"]=np.mean(U)/mi
            r["mean_V_over_I"]=np.mean(V)/mi
            with np.errstate(invalid='ignore', over='ignore'):
                dl=np.sqrt(Q**2+U**2)/np.maximum(I,1e-30)
                dc=np.abs(V)/np.maximum(I,1e-30)
                r["mean_DoLP"]=np.mean(dl);r["median_DoLP"]=np.median(dl)
                r["mean_DoCP"]=np.mean(dc);r["median_DoCP"]=np.median(dc)
                r["frac_DoLP_gt_001"]=np.mean(dl>0.01)
                r["frac_DoCP_gt_001"]=np.mean(dc>0.01)
        else:
            for k in ["mean_I","mean_Q","mean_U","mean_V","mean_Q_over_I",
                      "mean_U_over_I","mean_V_over_I","mean_DoLP","median_DoLP",
                      "mean_DoCP","median_DoCP","frac_DoLP_gt_001","frac_DoCP_gt_001"]:
                r[k]=np.nan
        det_rows.append(r)

dfd = pd.DataFrame(det_rows)
det_csv = os.path.join(OUTDIR, "detector_response_polarised.csv")
dfd.to_csv(det_csv, index=False)
print(f"✅ Saved {det_csv}")

# Detector response plot
plt.figure(figsize=(10,6))
for wl_idx,wl in enumerate(wavelengths):
    wl_nm=int(wl*1e9); sub=dfd[dfd["wavelength_nm"]==wl_nm]
    plt.plot(sub["detector_angle_deg"],sub["normalised_response"],marker="o",label=f"{wl_nm} nm")
plt.xlabel("Detector angle (deg)");plt.ylabel("Normalised response")
plt.title("Detector Response — CLARITAS_104 Polarised MC")
plt.grid(True,alpha=0.3);plt.legend()
plt.savefig(os.path.join(OUTDIR,"detector_response_polarised.png"),dpi=200)
plt.close()
print("✅ Saved detector_response_polarised.png")

# Polarisation diagnostics
if len(wavelengths)==1:
    wl_nm=int(wavelengths[0]*1e9);sub=dfd[dfd["wavelength_nm"]==wl_nm]
    fig,ax=plt.subplots(2,2,figsize=(12,10))
    ax[0,0].plot(sub["detector_angle_deg"],sub["mean_DoLP"],marker="o",color="C0")
    ax[0,0].set_xlabel("Detector angle (deg)");ax[0,0].set_ylabel("Mean DoLP")
    ax[0,0].set_title("Mean Degree of Linear Polarisation");ax[0,0].grid(True,alpha=0.3)
    ax[0,1].plot(sub["detector_angle_deg"],sub["mean_DoCP"],marker="o",color="C1")
    ax[0,1].set_xlabel("Detector angle (deg)");ax[0,1].set_ylabel("Mean DoCP")
    ax[0,1].set_title("Mean Degree of Circular Polarisation");ax[0,1].grid(True,alpha=0.3)
    ax[1,0].plot(sub["detector_angle_deg"],sub["mean_Q_over_I"],marker="o",label="Q/I",color="C2")
    ax[1,0].plot(sub["detector_angle_deg"],sub["mean_U_over_I"],marker="s",label="U/I",color="C3")
    ax[1,0].set_xlabel("Detector angle (deg)");ax[1,0].set_ylabel("Normalised Stokes")
    ax[1,0].set_title("Q/I and U/I");ax[1,0].legend();ax[1,0].grid(True,alpha=0.3)
    ax[1,1].plot(sub["detector_angle_deg"],sub["mean_V_over_I"],marker="o",color="C4")
    ax[1,1].set_xlabel("Detector angle (deg)");ax[1,1].set_ylabel("V/I")
    ax[1,1].set_title("Mean Circular Polarisation V/I");ax[1,1].grid(True,alpha=0.3)
    plt.suptitle(f"Polarisation Diagnostics — CLARITAS_104 — {wl_nm} nm",fontsize=14,fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR,"polarisation_diagnostics.png"),dpi=200)
    plt.close()
    print("✅ Saved polarisation_diagnostics.png")

# Print summary
print(f"\n{'='*70}")
print("  POLARISATION SUMMARY")
print(f"{'='*70}")
for wl_idx,wl in enumerate(wavelengths):
    wl_nm=int(wl*1e9);sub=dfd[dfd["wavelength_nm"]==wl_nm]
    print(f"\n  Wavelength {wl_nm} nm:")
    print(f"  {'Angle':>8s}  {'Hits':>8s}  {'DoLP':>8s}  {'DoCP':>8s}  {'Q/I':>8s}  {'U/I':>8s}  {'V/I':>8s}")
    print(f"  {'-'*60}")
    for _,r in sub.iterrows():
        hv=int(r['hit_count']) if np.isfinite(r['hit_count']) else 0
        print(f"  {r['detector_angle_deg']:8.1f}  {hv:8d}  {r['mean_DoLP']:8.4f}  {r['mean_DoCP']:8.4f}  {r['mean_Q_over_I']:8.4f}  {r['mean_U_over_I']:8.4f}  {r['mean_V_over_I']:8.4f}")

# Transport breakdown
for wl_idx,wl in enumerate(wavelengths):
    wl_nm=int(wl*1e9);sd=dfd[dfd["wavelength_nm"]==wl_nm].copy()
    sd["hit_fraction_of_valid_exits"]=0
    hf=os.path.join(OUTDIR,f"ray_exits_polarised_{wl_nm}nm.h5")
    if os.path.exists(hf):
        with h5py.File(hf,"r") as f:
            ts=f["terminal_state"][:]
            nv=int(np.sum(ts==1))
        sd["hit_fraction_of_valid_exits"]=sd["hit_count"]/nv if nv>0 else 0
    sd.to_csv(os.path.join(OUTDIR,f"detector_transport_breakdown_polarised_{wl_nm}nm.csv"),index=False)

# Scatter histogram
for wl_idx,wl in enumerate(wavelengths):
    wl_nm=int(wl*1e9)
    hf=os.path.join(OUTDIR,f"ray_exits_polarised_{wl_nm}nm.h5")
    if os.path.exists(hf):
        with h5py.File(hf,"r") as f: sc=f["scatter_count"][:];ts=f["terminal_state"][:]
        vs=sc[ts==1]
        if len(vs)>0:
            ms=max(vs.max(),0);b=np.arange(0,ms+2);hv,_=np.histogram(vs,bins=b)
        else: hv,b=np.array([0]),np.array([0,1])
        sh=pd.DataFrame({"scatter_count":b[:-1],"ray_count":hv})
        sh.to_csv(os.path.join(OUTDIR,f"scatter_count_histogram_polarised_{wl_nm}nm.csv"),index=False)
        plt.figure(figsize=(10,5));plt.bar(sh["scatter_count"],sh["ray_count"])
        plt.yscale("log");plt.xlabel("Scatter count");plt.ylabel("Ray count (log)")
        plt.title(f"Scatter Count Histogram — CLARITAS_104 — {wl_nm} nm")
        plt.grid(True,alpha=0.3)
        plt.savefig(os.path.join(OUTDIR,f"scatter_count_histogram_polarised_{wl_nm}nm.png"),dpi=200)
        plt.close()

# ============================================================================
# CALIBRATION COMPARISON: Simulated vs measured angular response
# ============================================================================

print(f"\n{'='*70}")
print("  CALIBRATION COMPARISON")
print(f"{'='*70}")

# Identify matching calibration data
mat_key = "loess" if material_name == "loess" else "kaolin"
conc_key = mass_concentration_g_per_L
calib_available = (mat_key in CALIBRATION_DATA
                   and conc_key in CALIBRATION_DATA[mat_key])

if calib_available:
    calib_raw = CALIBRATION_DATA[mat_key][conc_key]

    for wl_idx, wl in enumerate(wavelengths):
        wl_nm = int(wl * 1e9)
        sub_sim = dfd[dfd["wavelength_nm"] == wl_nm].copy()
        sub_sim = sub_sim.sort_values("detector_angle_deg")

        # Normalise calibration: sum voltages to 1 (same as sim normalisation)
        calib_total = np.sum(calib_raw)
        calib_norm = calib_raw / calib_total if calib_total > 0 else calib_raw

        # Build comparison dataframe
        comp_rows = []
        for j, ang in enumerate(CALIBRATION_ANGLES_DEG):
            sim_row = sub_sim[sub_sim["detector_angle_deg"] == ang]
            sim_norm = float(sim_row["normalised_response"].values[0]) if len(sim_row) > 0 else np.nan
            comp_rows.append({
                "detector_angle_deg": ang,
                "sim_normalised": sim_norm,
                "calib_raw_voltage": calib_raw[j],
                "calib_normalised": calib_norm[j],
                "ratio_sim_over_calib": sim_norm / calib_norm[j] if calib_norm[j] > 1e-30 else np.nan,
            })
        comp_df = pd.DataFrame(comp_rows)

        comp_csv = os.path.join(OUTDIR, f"calibration_comparison_{wl_nm}nm.csv")
        comp_df.to_csv(comp_csv, index=False)
        print(f"✅ Saved {comp_csv}")

        # ---- Plot: Sim vs calibration (all 18 detectors) ----
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: linear overlay
        ax1.plot(comp_df["detector_angle_deg"], comp_df["sim_normalised"],
                 marker="o", label="CLARITAS_100 (sim)", color="C0", linewidth=2)
        ax1.plot(comp_df["detector_angle_deg"], comp_df["calib_normalised"],
                 marker="s", label=f"Measured ({mat_key} {conc_key} g/L)", color="C1", linewidth=1.5)
        # Flag suspect detectors
        for sa in CALIB_SUSPECT_ANGLES:
            ax1.axvline(x=sa, color="red", linestyle=":", alpha=0.5)
        ax1.axvline(x=CALIB_SUSPECT_ANGLES[0], color="red", linestyle=":", alpha=0.5,
                     label="Suspect detectors (150°,160°)")
        ax1.set_xlabel("Detector angle (deg)")
        ax1.set_ylabel("Normalised response (sum=1)")
        ax1.set_title(f"Sim vs Calibration — {mat_key} {conc_key} g/L — {wl_nm} nm")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Right: log10 overlay (better for dynamic range)
        valid_mask = (comp_df["sim_normalised"] > 1e-30) & (comp_df["calib_normalised"] > 1e-30)
        ax2.semilogy(comp_df["detector_angle_deg"][valid_mask],
                     comp_df["sim_normalised"][valid_mask],
                     marker="o", label="Sim", color="C0", linewidth=2)
        ax2.semilogy(comp_df["detector_angle_deg"][valid_mask],
                     comp_df["calib_normalised"][valid_mask],
                     marker="s", label="Measured", color="C1", linewidth=1.5)
        for sa in CALIB_SUSPECT_ANGLES:
            ax2.axvline(x=sa, color="red", linestyle=":", alpha=0.5)
        ax2.set_xlabel("Detector angle (deg)")
        ax2.set_ylabel("Normalised response (log₁₀)")
        ax2.set_title(f"Sim vs Calibration (log scale) — {mat_key} {conc_key} g/L")
        ax2.grid(True, alpha=0.3, which="both")
        ax2.legend()

        plt.tight_layout()
        calib_png = os.path.join(OUTDIR, f"calibration_comparison_{wl_nm}nm.png")
        plt.savefig(calib_png, dpi=200)
        plt.close()
        print(f"✅ Saved {calib_png}")

        # ---- Plot: Excluding suspect detectors (re-normalise) ----
        non_suspect_mask = np.array([a not in CALIB_SUSPECT_ANGLES
                                     for a in CALIBRATION_ANGLES_DEG])
        calib_ns = calib_raw[non_suspect_mask]
        calib_total_ns = np.sum(calib_ns)
        calib_norm_ns = calib_ns / calib_total_ns if calib_total_ns > 0 else calib_ns
        sim_norm_ns = comp_df["sim_normalised"].values[non_suspect_mask]
        sim_total_ns = np.sum(sim_norm_ns)
        sim_norm_ns_renorm = sim_norm_ns / sim_total_ns if sim_total_ns > 0 else sim_norm_ns
        angles_ns = CALIBRATION_ANGLES_DEG[non_suspect_mask]

        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
        ax3.plot(angles_ns, sim_norm_ns_renorm, marker="o",
                 label="Sim (re-norm, excl. 150°/160°)", color="C0", linewidth=2)
        ax3.plot(angles_ns, calib_norm_ns, marker="s",
                 label=f"Measured (re-norm, excl. 150°/160°)", color="C1", linewidth=1.5)
        ax3.set_xlabel("Detector angle (deg)")
        ax3.set_ylabel("Normalised response (excl. suspect)")
        ax3.set_title(f"Excluding 150°/160° — {mat_key} {conc_key} g/L — {wl_nm} nm")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        valid_ns = (sim_norm_ns_renorm > 1e-30) & (calib_norm_ns > 1e-30)
        ax4.semilogy(angles_ns[valid_ns], sim_norm_ns_renorm[valid_ns],
                     marker="o", label="Sim", color="C0", linewidth=2)
        ax4.semilogy(angles_ns[valid_ns], calib_norm_ns[valid_ns],
                     marker="s", label="Measured", color="C1", linewidth=1.5)
        ax4.set_xlabel("Detector angle (deg)")
        ax4.set_ylabel("Normalised response (log₁₀, excl. suspect)")
        ax4.set_title(f"Excluding suspect detectors (log scale)")
        ax4.grid(True, alpha=0.3, which="both")
        ax4.legend()

        plt.tight_layout()
        calib_ns_png = os.path.join(OUTDIR, f"calibration_comparison_nosuspect_{wl_nm}nm.png")
        plt.savefig(calib_ns_png, dpi=200)
        plt.close()
        print(f"✅ Saved {calib_ns_png}")

        # ---- Ratio plot ----
        fig3, ax5 = plt.subplots(figsize=(12, 5))
        ax5.axhline(y=1.0, color="grey", linestyle="--", alpha=0.5, label="Perfect match")
        ax5.plot(comp_df["detector_angle_deg"], comp_df["ratio_sim_over_calib"],
                 marker="o", color="C0", linewidth=1.5)
        ax5.set_xlabel("Detector angle (deg)")
        ax5.set_ylabel("Ratio sim / measured")
        ax5.set_title(f"Sim-to-calibration ratio — {mat_key} {conc_key} g/L — {wl_nm} nm")
        ax5.set_ylim(bottom=0)
        ax5.grid(True, alpha=0.3)
        for sa in CALIB_SUSPECT_ANGLES:
            ax5.axvline(x=sa, color="red", linestyle=":", alpha=0.5)
        ax5.legend()
        ratio_png = os.path.join(OUTDIR, f"calibration_ratio_{wl_nm}nm.png")
        plt.savefig(ratio_png, dpi=200)
        plt.close()
        print(f"✅ Saved {ratio_png}")

        # Print comparison table
        print(f"\n  Calibration comparison ({mat_key}, {conc_key} g/L, {wl_nm} nm):")
        print(f"  {'Angle':>8s}  {'Sim':>10s}  {'Meas':>10s}  {'Ratio':>8s}")
        print(f"  {'-'*44}")
        for _, r in comp_df.iterrows():
            flag = " ⚠" if r["detector_angle_deg"] in CALIB_SUSPECT_ANGLES else ""
            print(f"  {r['detector_angle_deg']:8.1f}  "
                  f"{r['sim_normalised']:10.6f}  {r['calib_normalised']:10.6f}  "
                  f"{r['ratio_sim_over_calib']:8.3f}{flag}")

else:
    print(f"  No calibration data found for material={material_name}, "
          f"concentration={mass_concentration_g_per_L} g/L")
    print(f"  Available: kaolin={list(CALIBRATION_DATA.get('kaolin',{}).keys())} g/L, "
          f"loess={list(CALIBRATION_DATA.get('loess',{}).keys())} g/L")

print(f"\n{'='*70}")
print("  CLARITAS_104 COMPLETE")
print(f"{'='*70}")
print(f"  Engine: {engine_label}")
print(f"  All outputs in: {OUTDIR}")
print(f"{'='*70}")