"""
Adapters for integrating production modules into MVP loop.
==========================================================

This package provides unified interfaces for:
- FEA (simple or truth_contact, local or HPC)
- Peridigm (mock or real, local or HPC)
- Mesh generation (fallback, pregenerated, theta_mesh)
"""

from .fea_adapter import run_fea, validate_fea_config
from .mesh_adapter import (
    load_mesh_for_mvp,
    load_mesh_for_mvp_legacy,
    get_region_masks,
    validate_mesh_config,
    MeshData,
)
from .peridigm_adapter import run_peridigm, validate_peridigm_config, parse_peridigm_results

__all__ = [
    "run_fea",
    "validate_fea_config",
    "run_peridigm",
    "validate_peridigm_config",
    "parse_peridigm_results",
    "load_mesh_for_mvp",
    "load_mesh_for_mvp_legacy",
    "get_region_masks",
    "validate_mesh_config",
    "MeshData",
]
