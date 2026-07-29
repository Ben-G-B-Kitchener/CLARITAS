import numpy as np

from .config import DetectorConfig


def distance_to_sphere_boundary(
    positions_m: np.ndarray,
    directions: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    """Positive distance from interior points to a sphere along each direction."""
    b = np.einsum("ij,ij->i", positions_m, directions)
    c = np.einsum("ij,ij->i", positions_m, positions_m) - radius_m**2
    discriminant = np.maximum(b * b - c, 0.0)
    return -b + np.sqrt(discriminant)


def rotate_directions(
    directions: np.ndarray,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
) -> np.ndarray:
    """Rotate unit directions by polar theta and local azimuth phi."""
    directions = np.asarray(directions, dtype=np.float64)
    reference = np.zeros_like(directions)
    use_z = np.abs(directions[:, 2]) < 0.9
    reference[use_z, 2] = 1.0
    reference[~use_z, 0] = 1.0

    e1 = np.cross(reference, directions)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(directions, e1)

    cos_theta = np.cos(theta_rad)[:, None]
    sin_theta = np.sin(theta_rad)[:, None]
    transverse = (
        np.cos(phi_rad)[:, None] * e1 +
        np.sin(phi_rad)[:, None] * e2
    )
    rotated = cos_theta * directions + sin_theta * transverse
    rotated /= np.linalg.norm(rotated, axis=1)[:, None]
    return rotated


def detector_polar_angle_deg(exit_positions_m: np.ndarray) -> np.ndarray:
    """Polar position angle: 0° at forward +y and 180° at incident -y."""
    radius = np.linalg.norm(exit_positions_m, axis=1)
    cos_angle = np.divide(
        exit_positions_m[:, 1],
        radius,
        out=np.zeros_like(radius),
        where=radius > 0.0,
    )
    return np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def assign_detector_bins(
    exit_positions_m: np.ndarray,
    detector: DetectorConfig,
) -> np.ndarray:
    """Return detector index per exit, or -1 outside every detector acceptance."""
    polar_deg = detector_polar_angle_deg(exit_positions_m)
    centres = np.asarray(detector.centres_deg, dtype=np.float64)
    differences = np.abs(polar_deg[:, None] - centres[None, :])
    nearest = np.argmin(differences, axis=1)
    accepted = differences[np.arange(len(polar_deg)), nearest] <= detector.acceptance_half_angle_deg
    return np.where(accepted, nearest, -1).astype(np.int32)

