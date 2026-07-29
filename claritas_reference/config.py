from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


LOESS_DIAMETERS_M = np.array([
    1.729e-6, 1.981e-6, 2.269e-6, 2.599e-6, 2.976e-6, 3.409e-6,
    3.905e-6, 4.472e-6, 5.122e-6, 5.867e-6, 6.720e-6, 7.697e-6,
    8.816e-6, 10.097e-6, 11.565e-6, 13.246e-6, 15.172e-6,
    17.377e-6, 19.904e-6, 22.797e-6, 26.111e-6, 29.907e-6,
    34.255e-6, 39.234e-6, 44.938e-6, 51.471e-6, 58.953e-6,
    67.523e-6, 77.340e-6, 88.583e-6, 101.460e-6, 116.210e-6,
    133.103e-6, 152.453e-6, 174.616e-6, 200.000e-6, 229.075e-6,
    262.376e-6,
], dtype=np.float64)

LOESS_WEIGHTS = np.array([
    157, 227, 294, 354, 414, 487, 592, 747, 975, 1291, 1704, 2197,
    2736, 3288, 3822, 4196, 4372, 4391, 4352, 4362, 4508, 4826,
    5279, 5758, 6080, 6106, 5786, 5149, 4342, 3404, 2456, 1662,
    1175, 858, 631, 463, 333, 230,
], dtype=np.float64)

KAOLIN_DIAMETERS_M = np.array([
    0.172e-6, 0.197e-6, 0.226e-6, 0.259e-6, 0.296e-6, 0.339e-6,
    0.389e-6, 0.445e-6, 0.510e-6, 0.584e-6, 0.669e-6, 0.766e-6,
    0.877e-6, 1.005e-6, 1.151e-6, 1.318e-6, 1.510e-6, 1.729e-6,
    1.981e-6, 2.269e-6, 2.599e-6, 2.976e-6, 3.409e-6, 3.905e-6,
    4.472e-6, 5.122e-6, 5.867e-6, 6.720e-6, 7.697e-6, 8.816e-6,
    10.097e-6, 11.565e-6, 13.246e-6, 15.172e-6, 17.377e-6,
    19.904e-6, 22.797e-6,
], dtype=np.float64)

KAOLIN_WEIGHTS = np.array([
    217, 547, 1112, 2032, 2985, 3492, 3308, 2644, 1893, 1300, 916,
    700, 601, 584, 637, 757, 948, 1208, 1530, 1899, 2309, 2770,
    3312, 3973, 4772, 5681, 6583, 7267, 7478, 7042, 6113, 5057,
    3680, 2330, 1287, 631, 284,
], dtype=np.float64)


@dataclass(frozen=True)
class MaterialConfig:
    name: str
    diameters_m: np.ndarray
    mass_fraction_weights: np.ndarray
    density_kg_m3: float = 2600.0
    refractive_index_real: float = 1.59
    refractive_index_imag_k: float = 0.001

    def __post_init__(self) -> None:
        diameters = np.asarray(self.diameters_m, dtype=np.float64)
        weights = np.asarray(self.mass_fraction_weights, dtype=np.float64)
        if diameters.ndim != 1 or weights.shape != diameters.shape:
            raise ValueError("diameters_m and mass_fraction_weights must be equal 1-D arrays")
        if np.any(diameters <= 0.0) or np.any(weights < 0.0):
            raise ValueError("Particle diameters must be positive and weights non-negative")
        if not np.sum(weights) > 0.0 or self.density_kg_m3 <= 0.0:
            raise ValueError("PSD total weight and material density must be positive")


@dataclass(frozen=True)
class DetectorConfig:
    centres_deg: np.ndarray = field(
        default_factory=lambda: np.arange(0.0, 180.0, 10.0, dtype=np.float64)
    )
    acceptance_half_angle_deg: float = 6.5

    def __post_init__(self) -> None:
        centres = np.asarray(self.centres_deg, dtype=np.float64)
        if centres.ndim != 1 or np.any((centres < 0.0) | (centres > 180.0)):
            raise ValueError("Detector centres must be a 1-D array in [0, 180] degrees")
        if self.acceptance_half_angle_deg <= 0.0:
            raise ValueError("Detector acceptance must be positive")


@dataclass(frozen=True)
class SimulationConfig:
    n_rays: int = 100_000
    seed: int = 20260727
    wavelength_m: float = 622e-9
    concentration_kg_m3: float = 0.5
    sample_radius_m: float = 0.049
    max_events: int = 10_000
    phase_grid_size: int = 20_001
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def __post_init__(self) -> None:
        if self.n_rays <= 0 or self.wavelength_m <= 0.0 or self.sample_radius_m <= 0.0:
            raise ValueError("Ray count, wavelength, and sample radius must be positive")
        if self.concentration_kg_m3 < 0.0 or self.max_events <= 0:
            raise ValueError("Concentration must be non-negative and max_events positive")
        if self.phase_grid_size < 1001:
            raise ValueError("phase_grid_size must be at least 1001")


MATERIALS = {
    "loess": MaterialConfig("loess", LOESS_DIAMETERS_M, LOESS_WEIGHTS),
    "kaolin": MaterialConfig("kaolin", KAOLIN_DIAMETERS_M, KAOLIN_WEIGHTS),
}

DATASETS: Tuple[Tuple[str, float], ...] = (
    ("loess", 0.5),
    ("loess", 4.0),
    ("kaolin", 0.5),
    ("kaolin", 4.0),
)

