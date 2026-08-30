#!/usr/bin/env python3
"""Inference-oriented CLARITAS V23.2 forward model.

This module preserves the V23.2 particle physics but removes plotting, HDF5 and
heatmap accumulation so that many deterministic forward evaluations can be made
inside a PSD optimiser.

Physics retained from V23.2:
  * 3-D spherical sample and 3-D ray state
  * geometric particle encounter cross section pi*r^2
  * exponential free paths from mu_geom
  * particle selection from n_i*pi*r_i^2
  * full 3-D Snell refraction
  * unpolarised Fresnel reflection at entry and exit interfaces
  * geometric internal Fresnel reflections inside the same sphere
  * the same 3-D source divergence and detector-cap geometry

No Mie, floc, empirical reflection, roughness or fitted optical multipliers are
introduced here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

LOESS_DIAMETER_M = np.array([1.729e-6, 1.981e-6, 2.269e-6, 2.599e-6, 2.976e-6, 3.409e-6, 3.905e-6,
    4.472e-6, 5.122e-6, 5.867e-6, 6.72e-6, 7.697e-6, 8.816e-6, 10.097e-6,
    11.565e-6, 13.246e-6, 15.172e-6, 17.377e-6, 19.904e-6, 22.797e-6,
    26.111e-6, 29.907e-6, 34.255e-6, 39.234e-6, 44.938e-6, 51.471e-6,
    58.953e-6, 67.523e-6, 77.34e-6, 88.583e-6, 101.46e-6, 116.21e-6,
    133.103e-6, 152.453e-6, 174.616e-6, 200.000e-6, 229.075e-6, 262.376e-6], dtype=np.float64)
LOESS_WEIGHTS = np.array([157, 227, 294, 354, 414, 487, 592, 747, 975, 1291, 1704, 2197, 2736,
    3288, 3822, 4196, 4372, 4391, 4352, 4362, 4508, 4826, 5279, 5758,
    6080, 6106, 5786, 5149, 4342, 3404, 2456, 1662, 1175, 858, 631, 463, 333, 230], dtype=np.float64)
KAOLIN_DIAMETER_M = np.array([0.172e-6, 0.197e-6, 0.226e-6, 0.259e-6, 0.296e-6, 0.339e-6, 0.389e-6,
    0.445e-6, 0.51e-6, 0.584e-6, 0.669e-6, 0.766e-6, 0.877e-6, 1.005e-6,
    1.151e-6, 1.318e-6, 1.51e-6, 1.729e-6, 1.981e-6, 2.269e-6,
    2.599e-6, 2.976e-6, 3.409e-6, 3.905e-6, 4.472e-6, 5.122e-6,
    5.867e-6, 6.72e-6, 7.697e-6, 8.816e-6, 10.097e-6, 11.565e-6,
    13.246e-6, 15.172e-6, 17.377e-6, 19.904e-6, 22.797e-6], dtype=np.float64)
KAOLIN_WEIGHTS = np.array([217, 547, 1112, 2032, 2985, 3492, 3308, 2644, 1893, 1300, 916, 700, 601,
    584, 637, 757, 948, 1208, 1530, 1899, 2309, 2770, 3312, 3973,
    4772, 5681, 6583, 7267, 7478, 7042, 6113, 5057, 3680, 2330, 1287, 631, 284], dtype=np.float64)

DETECTOR_ANGLES_DEG = np.arange(0, 180, 10, dtype=np.float64)

CUDA_SRC = 'extern "C" {\n\n__device__ unsigned int xorshift32_state(unsigned int* state) {\n    unsigned int x = *state;\n    x ^= x << 13;\n    x ^= x >> 17;\n    x ^= x << 5;\n    *state = x;\n    return x;\n}\n\n__device__ float rnd_uniform(unsigned int* state) {\n    unsigned int r = xorshift32_state(state);\n    return (float)(r * 2.3283064e-10f);\n}\n\n__device__ void normalize3(float* x, float* y, float* z) {\n    float n2 = (*x)*(*x) + (*y)*(*y) + (*z)*(*z);\n    if (n2 <= 0.0f) {\n        *x = 0.0f; *y = 1.0f; *z = 0.0f;\n        return;\n    }\n    float invn = rsqrtf(n2);\n    *x *= invn; *y *= invn; *z *= invn;\n}\n\n__device__ float dot3(\n    const float ax, const float ay, const float az,\n    const float bx, const float by, const float bz)\n{\n    return ax*bx + ay*by + az*bz;\n}\n\n__device__ void reflect3(\n    const float ix, const float iy, const float iz,\n    const float nx, const float ny, const float nz,\n    float* ox, float* oy, float* oz)\n{\n    // n is the geometric surface normal. Reflection is independent of which\n    // medium side the normal points toward.\n    float d = dot3(ix, iy, iz, nx, ny, nz);\n    *ox = ix - 2.0f*d*nx;\n    *oy = iy - 2.0f*d*ny;\n    *oz = iz - 2.0f*d*nz;\n    normalize3(ox, oy, oz);\n}\n\n__device__ int refract3(\n    const float ix, const float iy, const float iz,\n    const float nx_against, const float ny_against, const float nz_against,\n    const float n1, const float n2,\n    float* ox, float* oy, float* oz)\n{\n    // normal_against points into incident medium, so dot(I,N) <= 0.\n    float cos_i = -dot3(ix, iy, iz, nx_against, ny_against, nz_against);\n    cos_i = fminf(fmaxf(cos_i, 0.0f), 1.0f);\n\n    float eta = n1 / n2;\n    float k = 1.0f - eta*eta*(1.0f - cos_i*cos_i);\n    if (k <= 0.0f) return 0;  // total internal reflection / no transmitted ray\n\n    float cos_t = sqrtf(k);\n    *ox = eta*ix + (eta*cos_i - cos_t)*nx_against;\n    *oy = eta*iy + (eta*cos_i - cos_t)*ny_against;\n    *oz = eta*iz + (eta*cos_i - cos_t)*nz_against;\n    normalize3(ox, oy, oz);\n    return 1;\n}\n\n__device__ float fresnel_unpolarized_R(\n    float cos_i,\n    const float n1,\n    const float n2,\n    int* tir)\n{\n    cos_i = fminf(fmaxf(cos_i, 0.0f), 1.0f);\n    float sin2_i = fmaxf(0.0f, 1.0f - cos_i*cos_i);\n    float eta = n1 / n2;\n    float sin2_t = eta*eta*sin2_i;\n\n    if (sin2_t >= 1.0f) {\n        *tir = 1;\n        return 1.0f;\n    }\n\n    *tir = 0;\n    float cos_t = sqrtf(fmaxf(0.0f, 1.0f - sin2_t));\n\n    float rs_den = n1*cos_i + n2*cos_t;\n    float rp_den = n1*cos_t + n2*cos_i;\n\n    float Rs = 1.0f;\n    float Rp = 1.0f;\n    if (fabsf(rs_den) > 1.0e-12f) {\n        float rs = (n1*cos_i - n2*cos_t) / rs_den;\n        Rs = rs*rs;\n    }\n    if (fabsf(rp_den) > 1.0e-12f) {\n        float rp = (n1*cos_t - n2*cos_i) / rp_den;\n        Rp = rp*rp;\n    }\n\n    return fminf(fmaxf(0.5f*(Rs + Rp), 0.0f), 1.0f);\n}\n\n__device__ void perpendicular_basis(\n    const float vx, const float vy, const float vz,\n    float* e1x, float* e1y, float* e1z,\n    float* e2x, float* e2y, float* e2z)\n{\n    float rx, ry, rz;\n    if (fabsf(vy) < 0.9f) {\n        rx = 0.0f; ry = 1.0f; rz = 0.0f;\n    } else {\n        rx = 1.0f; ry = 0.0f; rz = 0.0f;\n    }\n\n    *e1x = ry*vz - rz*vy;\n    *e1y = rz*vx - rx*vz;\n    *e1z = rx*vy - ry*vx;\n    normalize3(e1x, e1y, e1z);\n\n    *e2x = vy*(*e1z) - vz*(*e1y);\n    *e2y = vz*(*e1x) - vx*(*e1z);\n    *e2z = vx*(*e1y) - vy*(*e1x);\n    normalize3(e2x, e2y, e2z);\n}\n\n__device__ int sphere_fresnel_interaction_3d(\n    unsigned int* state,\n    const float n_medium,\n    const float n_particle,\n    const float radius,\n    const int max_internal_bounces,\n    float* x,\n    float* y,\n    float* z,\n    float* vx,\n    float* vy,\n    float* vz,\n    float* internal_path_added,\n    int* fresnel_reflection_count,\n    int* entry_reflection_count,\n    int* internal_reflection_count)\n{\n    if (radius <= 0.0f) return 1;\n\n    // Uniform interception point over the projected disk of a 3-D sphere.\n    float rho = sqrtf(fminf(fmaxf(rnd_uniform(state), 0.0f), 0.99999994f));\n    float cos_i = sqrtf(fmaxf(0.0f, 1.0f - rho*rho));\n    float phi = rnd_uniform(state) * 2.0f * 3.1415927f;\n\n    // q is the transverse impact direction around the incident ray.\n    float e1x, e1y, e1z, e2x, e2y, e2z;\n    perpendicular_basis(*vx, *vy, *vz, &e1x, &e1y, &e1z, &e2x, &e2y, &e2z);\n    float qx = cosf(phi)*e1x + sinf(phi)*e2x;\n    float qy = cosf(phi)*e1y + sinf(phi)*e2y;\n    float qz = cosf(phi)*e1z + sinf(phi)*e2z;\n\n    // Outward normal at entry. It points into the incident medium and therefore\n    // satisfies dot(v, n_entry) = -cos(i).\n    float nex = -cos_i*(*vx) + rho*qx;\n    float ney = -cos_i*(*vy) + rho*qy;\n    float nez = -cos_i*(*vz) + rho*qz;\n    normalize3(&nex, &ney, &nez);\n\n    // Place the actual particle centre so the current transport point is the\n    // sampled entry point on a sphere of the selected bin radius.\n    float cx = *x - radius*nex;\n    float cy = *y - radius*ney;\n    float cz = *z - radius*nez;\n\n    int tir_entry = 0;\n    float R_entry = fresnel_unpolarized_R(cos_i, n_medium, n_particle, &tir_entry);\n\n    if (tir_entry || rnd_uniform(state) < R_entry) {\n        float rvx, rvy, rvz;\n        reflect3(*vx, *vy, *vz, nex, ney, nez, &rvx, &rvy, &rvz);\n        *vx = rvx; *vy = rvy; *vz = rvz;\n        (*fresnel_reflection_count)++;\n        (*entry_reflection_count)++;\n        return 1;\n    }\n\n    // Entry transmission by vector Snell law.\n    float ivx, ivy, ivz;\n    if (!refract3(\n        *vx, *vy, *vz,\n        nex, ney, nez,\n        n_medium, n_particle,\n        &ivx, &ivy, &ivz))\n    {\n        // Numerically this should coincide with TIR and hence entry reflection.\n        float rvx, rvy, rvz;\n        reflect3(*vx, *vy, *vz, nex, ney, nez, &rvx, &rvy, &rvz);\n        *vx = rvx; *vy = rvy; *vz = rvz;\n        (*fresnel_reflection_count)++;\n        (*entry_reflection_count)++;\n        return 1;\n    }\n\n    *vx = ivx; *vy = ivy; *vz = ivz;\n\n    // Traverse the real internal chord. After each internal reflection, the same\n    // geometric sphere is followed to the next surface intersection.\n    for (int bounce = 0; bounce <= max_internal_bounces; ++bounce) {\n        float rx = *x - cx;\n        float ry = *y - cy;\n        float rz = *z - cz;\n        float rdv = dot3(rx, ry, rz, *vx, *vy, *vz);\n        float chord = -2.0f * rdv;\n\n        if (chord <= 1.0e-12f) return 0;\n\n        *x += chord*(*vx);\n        *y += chord*(*vy);\n        *z += chord*(*vz);\n        *internal_path_added += chord;\n\n        // Geometric outward normal at this particle surface point.\n        float nox = (*x - cx) / radius;\n        float noy = (*y - cy) / radius;\n        float noz = (*z - cz) / radius;\n        normalize3(&nox, &noy, &noz);\n\n        // Incident ray is inside the particle. For the generic refraction function,\n        // the normal must point against the incident ray, i.e. inward = -n_out.\n        float cos_inside = dot3(*vx, *vy, *vz, nox, noy, noz);\n        cos_inside = fminf(fmaxf(cos_inside, 0.0f), 1.0f);\n\n        int tir_exit = 0;\n        float R_exit = fresnel_unpolarized_R(\n            cos_inside, n_particle, n_medium, &tir_exit\n        );\n\n        if (!tir_exit && rnd_uniform(state) >= R_exit) {\n            float ovx, ovy, ovz;\n            if (refract3(\n                *vx, *vy, *vz,\n                -nox, -noy, -noz,\n                n_particle, n_medium,\n                &ovx, &ovy, &ovz))\n            {\n                *vx = ovx; *vy = ovy; *vz = ovz;\n                return 1;\n            }\n            // Fall through to reflection if numerical refraction fails.\n        }\n\n        // Internal Fresnel reflection.\n        float rvx, rvy, rvz;\n        reflect3(*vx, *vy, *vz, nox, noy, noz, &rvx, &rvy, &rvz);\n        *vx = rvx; *vy = rvy; *vz = rvz;\n        (*fresnel_reflection_count)++;\n        (*internal_reflection_count)++;\n\n        if (bounce == max_internal_bounces) return 0;\n    }\n\n    return 0;\n}\n\n__global__ void trace_kernel(\n    const float MAX_ITERATIONS,\n    const float MU_GEOM,\n    const float N_MEDIUM,\n    const int MAX_INTERNAL_BOUNCES,\n    const float R_REAL,\n    const float R_OFF,\n    const double* polar_init,\n    const double* azimuth_init,\n    const int N_rays,\n    const double* particle_cdf_table,\n    const double* particle_refractive_index_table,\n    const double* particle_radius_table,\n    const int n_particles,\n    float* exit_x_out,\n    float* exit_y_out,\n    float* exit_z_out,\n    int* interaction_count_out,\n    int* fresnel_reflection_count_out,\n    int* entry_reflection_count_out,\n    int* internal_reflection_count_out,\n    unsigned int seed0,\n    unsigned int seed1,\n    const unsigned int ray_offset)\n{\n    int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid >= N_rays) return;\n\n    unsigned int gid = ray_offset + (unsigned int)tid;\n    unsigned int state = seed0 + gid * 74729u + 13u;\n    unsigned int stateOPT = seed1 + gid * 104729u + 29u;\n\n    // Circular Gaussian beam waist in source plane (x-z), identical to V23.2.\n    float beam_sigma = 0.00001f;\n    float u1 = fmaxf(rnd_uniform(&state), 1.0e-12f);\n    float u2 = rnd_uniform(&state);\n    float mag = sqrtf(-2.0f * logf(u1));\n    float x0 = beam_sigma * mag * cosf(2.0f * 3.1415927f * u2);\n    float z0 = beam_sigma * mag * sinf(2.0f * 3.1415927f * u2);\n    float y0 = -(R_REAL + R_OFF);\n\n    float polar = (float)polar_init[tid];\n    float azimuth = (float)azimuth_init[tid];\n    float spolar = sinf(polar);\n    float vx = spolar * cosf(azimuth);\n    float vy = cosf(polar);\n    float vz = spolar * sinf(azimuth);\n    normalize3(&vx, &vy, &vz);\n\n    if (vy <= 0.0f) return;\n\n    // Ray-sphere sample entry intersection.\n    float b = x0*vx + y0*vy + z0*vz;\n    float c = x0*x0 + y0*y0 + z0*z0 - R_REAL*R_REAL;\n    float disc = b*b - c;\n    if (disc <= 0.0f) return;\n\n    float t = -b - sqrtf(disc);\n    if (t < 0.0f) return;\n\n    float x = x0 + t*vx;\n    float y = y0 + t*vy;\n    float z = z0 + t*vz;\n\n    const int max_steps = (int)MAX_ITERATIONS;\n    int step_count = 0;\n    int failed = 0;\n    int interaction_count = 0;\n    int fresnel_reflection_count = 0;\n    int entry_reflection_count = 0;\n    int internal_reflection_count = 0;\n\n    while (x*x + y*y + z*z <= R_REAL*R_REAL + 1.0e-8f) {\n        float rv = x*vx + y*vy + z*vz;\n        float rr_minus_R2 = x*x + y*y + z*z - R_REAL*R_REAL;\n        float boundary_disc = rv*rv - rr_minus_R2;\n        if (boundary_disc < 0.0f) {\n            failed = 1;\n            break;\n        }\n\n        float distance_to_boundary = -rv + sqrtf(fmaxf(boundary_disc, 0.0f));\n        if (distance_to_boundary <= 1.0e-9f) break;\n\n        float free_path = 3.402823466e+38F;\n        if (MU_GEOM > 0.0f) {\n            float u_path = fmaxf(rnd_uniform(&state), 1.0e-12f);\n            free_path = -logf(u_path) / MU_GEOM;\n        }\n\n        bool interaction_before_boundary = free_path < distance_to_boundary;\n        float travel_dist = interaction_before_boundary ? free_path : distance_to_boundary;\n\n        x += vx * travel_dist;\n        y += vy * travel_dist;\n        z += vz * travel_dist;\n        step_count++;\n\n        if (!interaction_before_boundary) break;\n\n        interaction_count++;\n\n        float u_particle = rnd_uniform(&state);\n        int pidx = n_particles - 1;\n        for (int j = 0; j < n_particles - 1; ++j) {\n            if (u_particle <= (float)particle_cdf_table[j]) {\n                pidx = j;\n                break;\n            }\n        }\n\n        float n_particle_this = (float)particle_refractive_index_table[pidx];\n        float particle_radius_this = (float)particle_radius_table[pidx];\n        float internal_added = 0.0f;\n\n        int ok = sphere_fresnel_interaction_3d(\n            &stateOPT,\n            N_MEDIUM,\n            n_particle_this,\n            particle_radius_this,\n            MAX_INTERNAL_BOUNCES,\n            &x, &y, &z,\n            &vx, &vy, &vz,\n            &internal_added,\n            &fresnel_reflection_count,\n            &entry_reflection_count,\n            &internal_reflection_count\n        );\n\n        if (!ok) {\n            failed = 1;\n            break;\n        }\n\n        if (step_count >= max_steps) {\n            failed = 1;\n            break;\n        }\n    }\n\n    if (!failed) {\n        exit_x_out[tid] = x;\n        exit_y_out[tid] = y;\n        exit_z_out[tid] = z;\n        interaction_count_out[tid] = interaction_count;\n        fresnel_reflection_count_out[tid] = fresnel_reflection_count;\n        entry_reflection_count_out[tid] = entry_reflection_count;\n        internal_reflection_count_out[tid] = internal_reflection_count;\n    }\n}\n}\n'

@dataclass
class ForwardResult:
    detector_angles_deg: np.ndarray
    raw_hits: np.ndarray
    normalized_response: np.ndarray
    valid_exit_count: int
    n_rays: int
    mu_geom_per_m: float
    optical_depth_diameter: float
    mean_interactions: float
    median_interactions: float
    ballistic_fraction: float
    mean_fresnel_reflections: float
    entry_reflection_fraction: float
    internal_reflection_fraction: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "detector_angles_deg": self.detector_angles_deg.tolist(),
            "raw_hits": self.raw_hits.tolist(),
            "normalized_response": self.normalized_response.tolist(),
            "valid_exit_count": self.valid_exit_count,
            "n_rays": self.n_rays,
            "mu_geom_per_m": self.mu_geom_per_m,
            "optical_depth_diameter": self.optical_depth_diameter,
            "mean_interactions": self.mean_interactions,
            "median_interactions": self.median_interactions,
            "ballistic_fraction": self.ballistic_fraction,
            "mean_fresnel_reflections": self.mean_fresnel_reflections,
            "entry_reflection_fraction": self.entry_reflection_fraction,
            "internal_reflection_fraction": self.internal_reflection_fraction,
        }


def get_material_psd(material: str):
    m = material.strip().lower()
    if m == "loess":
        d, w = LOESS_DIAMETER_M.copy(), LOESS_WEIGHTS.copy()
    elif m == "kaolin":
        d, w = KAOLIN_DIAMETER_M.copy(), KAOLIN_WEIGHTS.copy()
    else:
        raise ValueError("material must be 'loess' or 'kaolin'")
    w = w / w.sum()
    return d, w


def build_geometric_transport(diameters_m, weights, concentration_g_per_L,
                              density_kg_per_m3=2600.0, weight_mode="mass_fraction"):
    d = np.asarray(diameters_m, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if d.ndim != 1 or w.ndim != 1 or d.size != w.size:
        raise ValueError("diameters and weights must be equal-length 1-D arrays")
    if np.any(d <= 0) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("diameters must be >0 and weights non-negative with non-zero total")
    w = w / w.sum()
    r = d / 2.0
    particle_mass = (4.0/3.0) * np.pi * r**3 * float(density_kg_per_m3)
    c = float(concentration_g_per_L)  # 1 g/L == 1 kg/m^3
    if weight_mode == "mass_fraction":
        number_density = c * w / particle_mass
    elif weight_mode == "number_fraction":
        mean_mass = np.sum(w * particle_mass)
        total_n = c / mean_mass if mean_mass > 0 else 0.0
        number_density = total_n * w
    else:
        raise ValueError("weight_mode must be mass_fraction or number_fraction")
    sigma = np.pi * r**2
    mu_by_bin = number_density * sigma
    mu = float(mu_by_bin.sum())
    event_weights = mu_by_bin / mu if mu > 0 else np.zeros_like(mu_by_bin)
    cdf = np.cumsum(event_weights)
    if cdf.size and cdf[-1] > 0:
        cdf /= cdf[-1]
    return {
        "weights": w, "radii_m": r, "number_density_by_bin": number_density,
        "geometric_cross_section_m2": sigma, "mu_geom_by_bin": mu_by_bin,
        "mu_geom": mu, "particle_event_weights": event_weights,
        "particle_event_cdf": cdf,
    }


class ClaritasForwardModel:
    def __init__(self, n_medium=1.33, n_particle=1.59, sample_radius_m=0.049,
                 ray_offset_m=0.005, detector_acceptance_deg=6.5,
                 alpha1=1.0, alpha2=100.0, density_kg_per_m3=2600.0,
                 max_iterations=int(100e6), max_internal_bounces=64,
                 chunk_size=250_000):
        try:
            import cupy as cp
        except Exception as exc:
            raise RuntimeError(
                "CuPy is required for the inference forward model. Use the same CUDA-enabled "
                "CLARITAS environment as V23.2."
            ) from exc
        self.cp = cp
        self.n_medium = float(n_medium)
        self.n_particle = float(n_particle)
        self.sample_radius_m = float(sample_radius_m)
        self.ray_offset_m = float(ray_offset_m)
        self.detector_acceptance_deg = float(detector_acceptance_deg)
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.density_kg_per_m3 = float(density_kg_per_m3)
        self.max_iterations = int(max_iterations)
        self.max_internal_bounces = int(max_internal_bounces)
        self.chunk_size = int(chunk_size)
        self.module = cp.RawModule(code=CUDA_SRC, options=('-std=c++11',))
        self.kernel = self.module.get_function('trace_kernel')
        theta = np.deg2rad(DETECTOR_ANGLES_DEG)
        self.detector_sin = np.sin(theta)
        self.detector_cos = np.cos(theta)
        self.detector_cos_accept = math.cos(math.radians(self.detector_acceptance_deg))

    def simulate(self, material: str, concentration_g_per_L: float,
                 weights: Optional[np.ndarray] = None, n_rays: int = 100_000,
                 seed: int = 12345, weight_mode: str = "mass_fraction") -> ForwardResult:
        cp = self.cp
        diameters_m, base_weights = get_material_psd(material)
        if weights is None:
            weights = base_weights
        weights = np.asarray(weights, dtype=np.float64)
        if weights.size != diameters_m.size:
            raise ValueError(f"{material} requires {diameters_m.size} PSD weights")
        transport = build_geometric_transport(
            diameters_m, weights, concentration_g_per_L,
            density_kg_per_m3=self.density_kg_per_m3, weight_mode=weight_mode
        )
        mu_geom = transport["mu_geom"]
        cdf = transport["particle_event_cdf"]
        radii = transport["radii_m"]
        n_index = np.full_like(radii, self.n_particle, dtype=np.float64)

        # Common-random-number source directions: identical for every candidate at a given seed.
        rng = np.random.default_rng(int(seed))
        polar_all = rng.beta(self.alpha1, self.alpha2, int(n_rays)) * (np.pi / 2.0)
        azimuth_all = rng.uniform(0.0, 2.0*np.pi, int(n_rays))

        cdf_dev = cp.asarray(cdf, dtype=cp.float64)
        n_dev = cp.asarray(n_index, dtype=cp.float64)
        r_dev = cp.asarray(radii, dtype=cp.float64)
        threads = 256
        raw_hits = np.zeros(DETECTOR_ANGLES_DEG.size, dtype=np.int64)
        valid_total = 0
        sum_interactions = 0.0
        interaction_values = []
        ballistic_total = 0
        sum_fresnel = 0.0
        entry_reflected_rays = 0
        internal_reflected_rays = 0

        # Fixed seeds plus global ray offset make the kernel random stream deterministic and
        # independent of the chosen chunk size.
        seed0 = np.uint32((int(seed) * 1664525 + 1013904223) & 0x7fffffff or 1)
        seed1 = np.uint32((int(seed) * 22695477 + 1) & 0x7fffffff or 7)

        for start in range(0, int(n_rays), self.chunk_size):
            end = min(int(n_rays), start + self.chunk_size)
            n = end - start
            polar_dev = cp.asarray(polar_all[start:end], dtype=cp.float64)
            azimuth_dev = cp.asarray(azimuth_all[start:end], dtype=cp.float64)
            exit_x = cp.full(n, cp.nan, dtype=cp.float32)
            exit_y = cp.full(n, cp.nan, dtype=cp.float32)
            exit_z = cp.full(n, cp.nan, dtype=cp.float32)
            interaction = cp.full(n, -1, dtype=cp.int32)
            fresnel = cp.full(n, -1, dtype=cp.int32)
            entry_ref = cp.full(n, -1, dtype=cp.int32)
            internal_ref = cp.full(n, -1, dtype=cp.int32)
            blocks = (n + threads - 1)//threads
            self.kernel(
                (blocks,), (threads,),
                (
                    np.float32(self.max_iterations), np.float32(mu_geom),
                    np.float32(self.n_medium), np.int32(self.max_internal_bounces),
                    np.float32(self.sample_radius_m), np.float32(self.ray_offset_m),
                    polar_dev, azimuth_dev, np.int32(n), cdf_dev, n_dev, r_dev,
                    np.int32(cdf.size), exit_x, exit_y, exit_z, interaction,
                    fresnel, entry_ref, internal_ref, seed0, seed1, np.uint32(start)
                )
            )
            cp.cuda.Stream.null.synchronize()
            x = cp.asnumpy(exit_x); y = cp.asnumpy(exit_y); z = cp.asnumpy(exit_z)
            ic = cp.asnumpy(interaction); fr = cp.asnumpy(fresnel)
            er = cp.asnumpy(entry_ref); ir = cp.asnumpy(internal_ref)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (ic >= 0)
            if np.any(valid):
                rv = np.sqrt(x[valid]**2 + y[valid]**2 + z[valid]**2)
                good = rv > 0
                xv = x[valid][good] / rv[good]
                yv = y[valid][good] / rv[good]
                dots_plus = xv[:,None]*self.detector_sin[None,:] + yv[:,None]*self.detector_cos[None,:]
                dots_minus = -xv[:,None]*self.detector_sin[None,:] + yv[:,None]*self.detector_cos[None,:]
                raw_hits += ((dots_plus >= self.detector_cos_accept) | (dots_minus >= self.detector_cos_accept)).sum(axis=0)
                icv = ic[valid]
                valid_total += icv.size
                sum_interactions += float(icv.sum())
                interaction_values.append(icv.astype(np.int32, copy=False))
                ballistic_total += int(np.count_nonzero(icv == 0))
                sum_fresnel += float(fr[valid].sum())
                entry_reflected_rays += int(np.count_nonzero(er[valid] > 0))
                internal_reflected_rays += int(np.count_nonzero(ir[valid] > 0))
            del polar_dev, azimuth_dev, exit_x, exit_y, exit_z, interaction, fresnel, entry_ref, internal_ref

        total_hits = int(raw_hits.sum())
        normalized = raw_hits.astype(np.float64) / total_hits if total_hits > 0 else np.zeros_like(raw_hits, dtype=np.float64)
        if interaction_values:
            all_ic = np.concatenate(interaction_values)
            median_interactions = float(np.median(all_ic))
        else:
            median_interactions = float('nan')
        denom = max(valid_total, 1)
        return ForwardResult(
            detector_angles_deg=DETECTOR_ANGLES_DEG.copy(), raw_hits=raw_hits,
            normalized_response=normalized, valid_exit_count=valid_total, n_rays=int(n_rays),
            mu_geom_per_m=mu_geom, optical_depth_diameter=mu_geom*(2.0*self.sample_radius_m),
            mean_interactions=sum_interactions/denom if valid_total else float('nan'),
            median_interactions=median_interactions,
            ballistic_fraction=ballistic_total/denom if valid_total else float('nan'),
            mean_fresnel_reflections=sum_fresnel/denom if valid_total else float('nan'),
            entry_reflection_fraction=entry_reflected_rays/denom if valid_total else float('nan'),
            internal_reflection_fraction=internal_reflected_rays/denom if valid_total else float('nan'))


def load_weights_csv(path: str, expected_diameters_m: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path)
    if "weight" not in df.columns:
        raise ValueError("weights CSV must contain a 'weight' column")
    if "diameter_m" in df.columns:
        got = df["diameter_m"].to_numpy(float)
        if got.size != expected_diameters_m.size or not np.allclose(got, expected_diameters_m, rtol=1e-8, atol=0):
            raise ValueError("diameter_m column does not match the selected built-in PSD bins")
    w = df["weight"].to_numpy(float)
    if w.size != expected_diameters_m.size:
        raise ValueError("weights CSV has the wrong number of rows")
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser(description="Fast inference-oriented V23.2 forward evaluation")
    ap.add_argument("--material", choices=["kaolin","loess"], required=True)
    ap.add_argument("--concentration", type=float, required=True, help="g/L")
    ap.add_argument("--n-rays", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--weights-csv")
    ap.add_argument("--output", default="forward_response.csv")
    ap.add_argument("--diagnostics-json", default="forward_diagnostics.json")
    args = ap.parse_args()
    d, w = get_material_psd(args.material)
    if args.weights_csv:
        w = load_weights_csv(args.weights_csv, d)
    model = ClaritasForwardModel()
    res = model.simulate(args.material, args.concentration, weights=w, n_rays=args.n_rays, seed=args.seed)
    pd.DataFrame({
        "Detector_deg": res.detector_angles_deg.astype(int),
        "raw_hits": res.raw_hits,
        "normalized_response": res.normalized_response,
    }).to_csv(args.output, index=False)
    Path(args.diagnostics_json).write_text(json.dumps(res.to_dict(), indent=2))
    print(f"Saved {args.output}")
    print(f"mu_geom={res.mu_geom_per_m:.6g}  optical_depth={res.optical_depth_diameter:.6g}  valid={res.valid_exit_count}/{res.n_rays}")

if __name__ == "__main__":
    main()
