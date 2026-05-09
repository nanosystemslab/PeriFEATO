"""Shell thickness optimization loops and update rules."""

from .theta_band_params import ThetaBandParams, create_uniform_bands
from .thickness_updater import (
    update_thickness,
    update_fracture_factor,
    check_convergence,
)
from .thickness_loop import run_thickness_optimization
from .dual_physics_loop import run_dual_physics_optimization

__all__ = [
    "ThetaBandParams",
    "create_uniform_bands",
    "update_thickness",
    "update_fracture_factor",
    "check_convergence",
    "run_thickness_optimization",
    "run_dual_physics_optimization",
]
