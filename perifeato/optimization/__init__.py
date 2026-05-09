"""Optimization algorithms and orchestration loops."""

from .shell_thickness.dual_physics_loop import run_dual_physics_optimization
from .shell_thickness.thickness_loop import run_thickness_optimization
from .shell_thickness.theta_band_params import ThetaBandParams, create_uniform_bands

__all__ = [
    "run_dual_physics_optimization",
    "run_thickness_optimization",
    "ThetaBandParams",
    "create_uniform_bands",
]
