"""Validated CPU reference transport for primary particles."""

from .config import DetectorConfig, MaterialConfig, SimulationConfig
from .physics import OpticalMedium, build_primary_medium
from .transport import SimulationResult, simulate

__all__ = [
    "DetectorConfig",
    "MaterialConfig",
    "OpticalMedium",
    "SimulationConfig",
    "SimulationResult",
    "build_primary_medium",
    "simulate",
]

