from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .geometry import assign_detector_bins, distance_to_sphere_boundary, rotate_directions
from .physics import OpticalMedium


@dataclass(frozen=True)
class SimulationResult:
    detector_counts: np.ndarray
    detector_index: np.ndarray
    exit_positions_m: np.ndarray
    exit_directions: np.ndarray
    path_length_m: np.ndarray
    scatter_count: np.ndarray
    extinction_count: np.ndarray
    absorbed: np.ndarray
    truncated: np.ndarray

    @property
    def exited(self) -> np.ndarray:
        return ~(self.absorbed | self.truncated)

    def summary(self) -> dict:
        n = len(self.absorbed)
        exited = self.exited
        detected = self.detector_index >= 0
        paths = self.path_length_m[exited]
        return {
            "n_rays": n,
            "ballistic_exit_fraction": float(np.mean(exited & (self.scatter_count == 0))),
            "scattered_exit_fraction": float(np.mean(exited & (self.scatter_count > 0))),
            "total_exit_fraction": float(np.mean(exited)),
            "absorbed_fraction": float(np.mean(self.absorbed)),
            "truncated_fraction": float(np.mean(self.truncated)),
            "total_detected_fraction": float(np.mean(detected)),
            "mean_scatter_count": float(np.mean(self.scatter_count)),
            "mean_scatter_count_exited": float(np.mean(self.scatter_count[exited])) if np.any(exited) else 0.0,
            "mean_path_length_m_exited": float(np.mean(paths)) if paths.size else 0.0,
            "median_path_length_m_exited": float(np.median(paths)) if paths.size else 0.0,
            "p95_path_length_m_exited": float(np.percentile(paths, 95.0)) if paths.size else 0.0,
            "accounting_sum": float(np.mean(exited) + np.mean(self.absorbed) + np.mean(self.truncated)),
        }


def _sample_phase_angles(
    rng: np.random.Generator,
    medium: OpticalMedium,
    bin_indices: np.ndarray,
) -> np.ndarray:
    values = np.empty(len(bin_indices), dtype=np.float64)
    uniforms = rng.random(len(bin_indices))
    for bin_index in np.unique(bin_indices):
        mask = bin_indices == bin_index
        values[mask] = np.interp(
            uniforms[mask],
            medium.phase_cdf_by_bin[bin_index],
            medium.theta_rad,
        )
    return values


def sample_event_bins(
    rng: np.random.Generator,
    event_cdf: np.ndarray,
    count: int,
) -> np.ndarray:
    """Sample extinction-object indices from a validated cumulative probability."""
    cdf = np.asarray(event_cdf, dtype=np.float64)
    if cdf.ndim != 1 or cdf.size == 0 or count < 0:
        raise ValueError("event_cdf must be non-empty and count non-negative")
    if np.any(np.diff(cdf) < 0.0) or not np.isclose(cdf[-1], 1.0):
        raise ValueError("event_cdf must be monotonic and end at one")
    return np.searchsorted(cdf, rng.random(count), side="left")


def simulate(config: SimulationConfig, medium: OpticalMedium) -> SimulationResult:
    rng = np.random.default_rng(config.seed)
    n_rays = config.n_rays
    radius_m = config.sample_radius_m

    positions = np.zeros((n_rays, 3), dtype=np.float64)
    positions[:, 1] = -radius_m
    directions = np.zeros((n_rays, 3), dtype=np.float64)
    directions[:, 1] = 1.0

    path_length = np.zeros(n_rays, dtype=np.float64)
    scatter_count = np.zeros(n_rays, dtype=np.int32)
    extinction_count = np.zeros(n_rays, dtype=np.int32)
    absorbed = np.zeros(n_rays, dtype=bool)
    truncated = np.zeros(n_rays, dtype=bool)
    active = np.ones(n_rays, dtype=bool)

    for _ in range(config.max_events + 1):
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break
        if medium.mu_t_m_inv <= 0.0:
            boundary_distance = distance_to_sphere_boundary(
                positions[active_indices], directions[active_indices], radius_m
            )
            positions[active_indices] += directions[active_indices] * boundary_distance[:, None]
            path_length[active_indices] += boundary_distance
            active[active_indices] = False
            break

        boundary_distance = distance_to_sphere_boundary(
            positions[active_indices], directions[active_indices], radius_m
        )
        free_path = -np.log(np.maximum(rng.random(active_indices.size), 1e-15)) / medium.mu_t_m_inv
        exits_now = free_path >= boundary_distance

        exiting_indices = active_indices[exits_now]
        if exiting_indices.size:
            distance = boundary_distance[exits_now]
            positions[exiting_indices] += directions[exiting_indices] * distance[:, None]
            path_length[exiting_indices] += distance
            active[exiting_indices] = False

        event_indices = active_indices[~exits_now]
        if not event_indices.size:
            continue
        distance = free_path[~exits_now]
        positions[event_indices] += directions[event_indices] * distance[:, None]
        path_length[event_indices] += distance
        extinction_count[event_indices] += 1

        selected_bins = sample_event_bins(rng, medium.event_cdf, event_indices.size)
        scatter_event = rng.random(event_indices.size) < medium.albedo[selected_bins]
        absorbed_indices = event_indices[~scatter_event]
        absorbed[absorbed_indices] = True
        active[absorbed_indices] = False

        scattering_indices = event_indices[scatter_event]
        if scattering_indices.size:
            scattering_bins = selected_bins[scatter_event]
            theta = _sample_phase_angles(rng, medium, scattering_bins)
            phi = rng.uniform(0.0, 2.0 * np.pi, scattering_indices.size)
            directions[scattering_indices] = rotate_directions(
                directions[scattering_indices], theta, phi
            )
            scatter_count[scattering_indices] += 1

    if np.any(active):
        truncated[active] = True
        active[:] = False

    exited = ~(absorbed | truncated)
    detector_index = np.full(n_rays, -1, dtype=np.int32)
    detector_index[exited] = assign_detector_bins(positions[exited], config.detector)
    counts = np.bincount(
        detector_index[detector_index >= 0],
        minlength=len(config.detector.centres_deg),
    ).astype(np.int64)

    return SimulationResult(
        detector_counts=counts,
        detector_index=detector_index,
        exit_positions_m=positions,
        exit_directions=directions,
        path_length_m=path_length,
        scatter_count=scatter_count,
        extinction_count=extinction_count,
        absorbed=absorbed,
        truncated=truncated,
    )
