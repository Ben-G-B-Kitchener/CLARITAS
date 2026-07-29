from dataclasses import dataclass
from typing import Optional

import miepython
import numpy as np

from .config import MaterialConfig


@dataclass(frozen=True)
class OpticalMedium:
    diameters_m: np.ndarray
    number_density_m3: np.ndarray
    sigma_s_m2: np.ndarray
    sigma_a_m2: np.ndarray
    sigma_t_m2: np.ndarray
    event_cdf: np.ndarray
    albedo: np.ndarray
    theta_rad: np.ndarray
    phase_cdf_by_bin: np.ndarray
    mu_s_m_inv: float
    mu_a_m_inv: float
    mu_t_m_inv: float

    def __post_init__(self) -> None:
        n = len(self.diameters_m)
        arrays = (
            self.number_density_m3, self.sigma_s_m2, self.sigma_a_m2,
            self.sigma_t_m2, self.event_cdf, self.albedo,
        )
        if n == 0 or any(np.asarray(a).shape != (n,) for a in arrays):
            raise ValueError("All per-bin optical arrays must have the same non-zero length")
        if np.asarray(self.phase_cdf_by_bin).shape != (n, len(self.theta_rad)):
            raise ValueError("phase_cdf_by_bin shape must be (n_bins, n_theta)")
        if self.mu_t_m_inv < 0.0 or self.mu_s_m_inv < 0.0 or self.mu_a_m_inv < 0.0:
            raise ValueError("Bulk optical coefficients cannot be negative")


def _cdf_from_density(theta_rad: np.ndarray, density: np.ndarray) -> np.ndarray:
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(theta_rad)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    if cdf[-1] <= 0.0:
        raise ValueError("Angular probability density has zero integral")
    cdf /= cdf[-1]
    cdf[-1] = 1.0
    return cdf


def isotropic_phase_cdf(theta_rad: np.ndarray) -> np.ndarray:
    """Polar CDF for a uniform distribution over the unit sphere."""
    return 0.5 * (1.0 - np.cos(theta_rad))


def build_primary_medium(
    material: MaterialConfig,
    concentration_kg_m3: float,
    wavelength_m: float,
    phase_grid_size: int = 20_001,
) -> OpticalMedium:
    if concentration_kg_m3 < 0.0:
        raise ValueError("concentration_kg_m3 cannot be negative")
    diameters_m = np.asarray(material.diameters_m, dtype=np.float64)
    weights = np.asarray(material.mass_fraction_weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    radii_m = 0.5 * diameters_m
    particle_mass_kg = material.density_kg_m3 * (np.pi / 6.0) * diameters_m**3
    number_density_m3 = concentration_kg_m3 * weights / particle_mass_kg

    relative_index = complex(
        material.refractive_index_real,
        -material.refractive_index_imag_k,
    ) / 1.33
    theta_rad = np.linspace(0.0, np.pi, phase_grid_size, dtype=np.float64)
    mu_cos = np.cos(theta_rad)

    sigma_s_m2 = np.empty_like(radii_m)
    sigma_a_m2 = np.empty_like(radii_m)
    sigma_t_m2 = np.empty_like(radii_m)
    phase_cdfs = np.empty((len(radii_m), phase_grid_size), dtype=np.float64)

    for i, radius_m in enumerate(radii_m):
        size_parameter = 2.0 * np.pi * 1.33 * radius_m / wavelength_m
        qext, qsca, _, _ = miepython.efficiencies_mx(relative_index, size_parameter)
        area_m2 = np.pi * radius_m**2
        sigma_t_m2[i] = max(float(np.real(qext)) * area_m2, 0.0)
        sigma_s_m2[i] = min(max(float(np.real(qsca)) * area_m2, 0.0), sigma_t_m2[i])
        sigma_a_m2[i] = max(sigma_t_m2[i] - sigma_s_m2[i], 0.0)

        s1, s2 = miepython.S1_S2(relative_index, size_parameter, mu_cos)
        intensity = 0.5 * (np.abs(s1)**2 + np.abs(s2)**2)
        polar_density = np.real(intensity).astype(np.float64) * np.sin(theta_rad)
        phase_cdfs[i] = _cdf_from_density(theta_rad, polar_density)

    mu_s_by_bin = number_density_m3 * sigma_s_m2
    mu_a_by_bin = number_density_m3 * sigma_a_m2
    mu_t_by_bin = number_density_m3 * sigma_t_m2
    mu_s = float(np.sum(mu_s_by_bin))
    mu_a = float(np.sum(mu_a_by_bin))
    mu_t = float(np.sum(mu_t_by_bin))

    if mu_t > 0.0:
        event_cdf = np.cumsum(mu_t_by_bin) / mu_t
        event_cdf[-1] = 1.0
    else:
        event_cdf = np.linspace(1.0 / len(radii_m), 1.0, len(radii_m))
    albedo = np.divide(
        sigma_s_m2,
        sigma_t_m2,
        out=np.ones_like(sigma_s_m2),
        where=sigma_t_m2 > 0.0,
    )

    return OpticalMedium(
        diameters_m=diameters_m,
        number_density_m3=number_density_m3,
        sigma_s_m2=sigma_s_m2,
        sigma_a_m2=sigma_a_m2,
        sigma_t_m2=sigma_t_m2,
        event_cdf=event_cdf,
        albedo=albedo,
        theta_rad=theta_rad,
        phase_cdf_by_bin=phase_cdfs,
        mu_s_m_inv=mu_s,
        mu_a_m_inv=mu_a,
        mu_t_m_inv=mu_t,
    )


def make_test_medium(
    mu_s_m_inv: float,
    mu_a_m_inv: float,
    theta_grid_size: int = 4001,
    phase_cdf: Optional[np.ndarray] = None,
) -> OpticalMedium:
    if mu_s_m_inv < 0.0 or mu_a_m_inv < 0.0:
        raise ValueError("Test optical coefficients cannot be negative")
    mu_t = mu_s_m_inv + mu_a_m_inv
    theta = np.linspace(0.0, np.pi, theta_grid_size)
    cdf = isotropic_phase_cdf(theta) if phase_cdf is None else np.asarray(phase_cdf)
    sigma_t = np.array([mu_t])
    sigma_s = np.array([mu_s_m_inv])
    sigma_a = np.array([mu_a_m_inv])
    return OpticalMedium(
        diameters_m=np.array([1e-6]),
        number_density_m3=np.array([1.0]),
        sigma_s_m2=sigma_s,
        sigma_a_m2=sigma_a,
        sigma_t_m2=sigma_t,
        event_cdf=np.array([1.0]),
        albedo=np.array([mu_s_m_inv / mu_t if mu_t > 0.0 else 1.0]),
        theta_rad=theta,
        phase_cdf_by_bin=cdf.reshape(1, -1),
        mu_s_m_inv=mu_s_m_inv,
        mu_a_m_inv=mu_a_m_inv,
        mu_t_m_inv=mu_t,
    )

