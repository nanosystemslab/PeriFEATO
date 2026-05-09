"""
Mesh Adapter - Unified interface for mesh generation/loading
==============================================================

Supports:
1. Fallback mesh (simple box for testing)
2. Pregenerated mesh (load existing XDMF)
3. Theta mesh (generate using modules/theta_mesh/)

Configuration-driven backend selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import yaml


@dataclass
class MeshData:
    """Lightweight mesh container used across the MVP loop."""

    mesh: Any
    cell_regions: np.ndarray
    cell_centers: np.ndarray
    height: float
    xdmf_path: Optional[str] = None

    @property
    def n_cells(self) -> int:
        return int(self.cell_regions.size)


def load_mesh_for_mvp(config: Dict[str, Any]) -> MeshData:
    """
    Generate or load a mesh based on configuration.

    Config structure:
        mesh:
            backend: "fallback" | "pregenerated" | "theta_mesh"
            ... backend-specific parameters

    Returns:
        MeshData object with mesh and metadata
    """
    mesh_cfg = config.get("mesh", {}) if isinstance(config, dict) else {}

    # Handle legacy config format
    if "backend" not in mesh_cfg:
        if mesh_cfg.get("use_fallback", False):
            backend = "fallback"
        elif mesh_cfg.get("use_pregenerated", False):
            backend = "pregenerated"
        elif "theta_config_path" in mesh_cfg or "theta_overrides" in mesh_cfg:
            backend = "theta_mesh"
        else:
            backend = "fallback"  # Default for very old configs
    else:
        backend = mesh_cfg.get("backend", "fallback")

    if backend == "fallback":
        return _create_fallback_mesh(mesh_cfg)
    elif backend == "pregenerated":
        return _load_pregenerated_mesh(mesh_cfg)
    elif backend == "theta_mesh":
        return _generate_theta_mesh(mesh_cfg, config)
    else:
        raise ValueError(
            f"Invalid mesh backend: {backend}. Must be 'fallback', 'pregenerated', or 'theta_mesh'."
        )


def get_region_masks(mesh: MeshData, fracture_zone_ids, shell_body_ids):
    """Return boolean masks for fracture and shell regions."""
    fracture_mask = np.isin(mesh.cell_regions, fracture_zone_ids)
    shell_mask = np.isin(mesh.cell_regions, shell_body_ids)
    return fracture_mask, shell_mask


# ============================================================================
# Backend Implementations
# ============================================================================


def _create_fallback_mesh(mesh_cfg: Dict) -> MeshData:
    """
    Create simple box mesh for testing.

    Fast, no dependencies, good for MVP testing.
    """
    from dolfinx import mesh as dmesh
    from mpi4py import MPI

    # Box dimensions (meters)
    dims = mesh_cfg.get("dimensions", [0.014, 0.014, 0.028])
    n_elements = mesh_cfg.get("n_elements", [8, 8, 16])

    box = dmesh.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], dims],
        n_elements,
        cell_type=dmesh.CellType.tetrahedron,
    )

    centers = _compute_cell_centers(box)

    # Simple region assignment: lower half = protected (3), upper half = fracture (2)
    mid_z = dims[2] / 2.0
    region_ids = np.where(centers[:, 2] < mid_z, 3, 2).astype(np.int32)

    print(f"[Mesh Adapter] Created fallback mesh: {box.topology.index_map(3).size_local} cells")

    return MeshData(
        mesh=box,
        cell_regions=region_ids,
        cell_centers=centers,
        height=dims[2],
        xdmf_path=None,
    )


def _load_pregenerated_mesh(mesh_cfg: Dict) -> MeshData:
    """
    Load existing XDMF mesh from disk.

    Expects mesh with region_id cell data.
    """
    xdmf_path = mesh_cfg.get("xdmf_path")
    if not xdmf_path:
        raise ValueError("mesh.xdmf_path required for 'pregenerated' backend")

    xdmf_path = Path(xdmf_path)
    if not xdmf_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {xdmf_path}")

    print(f"[Mesh Adapter] Loading pregenerated mesh: {xdmf_path}")

    return _read_mesh_with_tags(xdmf_path)


def _generate_theta_mesh(mesh_cfg: Dict, full_config: Dict) -> MeshData:
    """
    Generate mesh using modules/theta_mesh/ generator.

    Uses ShellGeometryGeneratorRefined with configuration.
    """
    import os

    # Use HPC path (from environment variable or hardcoded)
    # NO FALLBACK - fail loudly if path is wrong
    of_root = Path(os.environ.get("OF_ROOT", "/home/mtdsn/Optimization_Framework"))
    theta_mesh_src = of_root / "modules" / "theta_mesh" / "src"

    if not theta_mesh_src.exists():
        raise RuntimeError(
            f"theta_mesh module not found at {theta_mesh_src}. "
            f"Ensure OF_ROOT is set correctly or sync code to HPC."
        )

    # Determine project root
    project_root = of_root

    # Add theta_mesh to path BEFORE import
    _ensure_theta_mesh_on_path(theta_mesh_src)

    # Import theta_mesh generator (NO FALLBACK)
    try:
        from shell_geometry_generator_refined import ShellGeometryGeneratorRefined
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import shell_geometry_generator_refined from {theta_mesh_src}. "
            f"Error: {exc}. Ensure gmsh and all dependencies are available in container."
        ) from exc

    # Prepare configuration
    base_config_path = mesh_cfg.get(
        "theta_config_path",
        project_root / "modules" / "theta_mesh" / "config" / "base_config.yaml",
    )
    base_config_path = Path(base_config_path)

    if not base_config_path.exists():
        raise FileNotFoundError(f"Theta mesh config not found: {base_config_path}")

    # Merge base config with overrides
    theta_overrides = mesh_cfg.get("theta_overrides", {})
    output_dir = Path(mesh_cfg.get("output_dir", project_root / "to_do" / "local_mvp" / "meshes"))

    config_path = _write_theta_config(base_config_path, theta_overrides, output_dir)

    # Generate mesh (NO FALLBACK - fail loudly if this doesn't work)
    print(f"[Mesh Adapter] Generating theta mesh with config: {config_path}")

    generator = ShellGeometryGeneratorRefined(str(config_path))
    generator.generate_geometry()
    generator.create_volumetric_mesh(include_solid_core=True, output_dir=str(output_dir))
    files = generator.export_mesh(output_dir=str(output_dir))

    xdmf_path = Path(files["xdmf"])
    print(f"[Mesh Adapter] Generated theta mesh: {xdmf_path}")

    return _read_mesh_with_tags(xdmf_path)


# ============================================================================
# Helper Functions
# ============================================================================


def _compute_cell_centers(mesh) -> np.ndarray:
    """Compute cell centroids."""
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    connectivity = mesh.topology.connectivity(tdim, 0)
    points = mesh.geometry.x

    num_cells = mesh.topology.index_map(tdim).size_local
    centers = np.zeros((num_cells, mesh.geometry.dim), dtype=float)
    for cell in range(num_cells):
        vertex_ids = connectivity.links(cell)
        centers[cell] = points[vertex_ids].mean(axis=0)
    return centers


def _read_mesh_with_tags(xdmf_path: Path) -> MeshData:
    """
    Read XDMF mesh with region_id tags.

    Tries multiple methods to extract region tags.
    """
    from dolfinx.io import XDMFFile
    from mpi4py import MPI
    import meshio

    with XDMFFile(MPI.COMM_WORLD, str(xdmf_path), "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cell_regions = None

        # Try to read region_id meshtags
        try:
            region_tags = xdmf.read_meshtags(mesh, name="region_id")
            num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
            cell_regions = np.zeros(num_cells, dtype=np.int32)
            cell_regions[region_tags.indices] = region_tags.values
        except RuntimeError:
            pass

    # If meshtags failed, try meshio
    if cell_regions is None:
        try:
            meshio_mesh = meshio.read(str(xdmf_path))
            region_data = meshio_mesh.cell_data_dict.get("region_id", {})
            if "tetra" in region_data:
                cell_regions = np.asarray(region_data["tetra"], dtype=np.int32)
            elif "hexahedron" in region_data:
                cell_regions = np.asarray(region_data["hexahedron"], dtype=np.int32)
        except Exception as exc:
            print(f"[Mesh Adapter] Failed to read region_id with meshio: {exc}")

    # If still no regions, assign default
    if cell_regions is None:
        print("[Mesh Adapter] Warning: No region_id found, using default regions")
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        cell_regions = np.full(num_cells, 2, dtype=np.int32)  # All protected

    centers = _compute_cell_centers(mesh)
    bbox_min = mesh.geometry.x.min(axis=0)
    bbox_max = mesh.geometry.x.max(axis=0)
    height = float(bbox_max[2] - bbox_min[2]) if mesh.geometry.dim >= 3 else 1.0

    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    print(f"[Mesh Adapter] Loaded mesh: {num_cells} cells, height={height*1000:.1f} mm")

    return MeshData(
        mesh=mesh,
        cell_regions=cell_regions,
        cell_centers=centers,
        height=height,
        xdmf_path=str(xdmf_path),
    )


def _ensure_theta_mesh_on_path(theta_path: Path) -> None:
    """Add theta_mesh to Python path if not already there."""
    theta_str = str(theta_path)
    if theta_str not in sys.path:
        sys.path.insert(0, theta_str)


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge of dictionaries."""
    updated = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(updated.get(key), dict):
            updated[key] = _deep_update(updated[key], value)
        else:
            updated[key] = value
    return updated


def _write_theta_config(
    base_config_path: Path,
    overrides: Optional[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """
    Create theta_mesh config by merging base + overrides.

    Returns path to temporary config file.
    """
    with base_config_path.open("r") as f:
        base_cfg = yaml.safe_load(f)

    merged = _deep_update(base_cfg, overrides or {})

    output_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="theta_mesh_", suffix=".yaml", dir=output_dir)
    os.close(fd)
    tmp_path = Path(tmp_path)

    with tmp_path.open("w") as f:
        yaml.safe_dump(merged, f, default_flow_style=False, sort_keys=False)

    print(f"[Mesh Adapter] Created theta config: {tmp_path}")

    return tmp_path


# ============================================================================
# Configuration Validation
# ============================================================================


def validate_mesh_config(config: Dict) -> list[str]:
    """
    Validate mesh configuration.

    Returns list of error messages (empty if valid).
    """
    errors = []
    mesh_cfg = config.get("mesh", {})

    # Check backend (default to "fallback" for backward compatibility)
    # Also check for legacy "use_fallback" field
    if "backend" not in mesh_cfg:
        if mesh_cfg.get("use_fallback", False):
            backend = "fallback"
        elif mesh_cfg.get("use_pregenerated", False):
            backend = "pregenerated"
        else:
            backend = "fallback"  # Default
    else:
        backend = mesh_cfg.get("backend", "fallback")

    if backend not in ["fallback", "pregenerated", "theta_mesh"]:
        errors.append(
            f"Invalid mesh.backend: {backend}. Must be 'fallback', 'pregenerated', or 'theta_mesh'."
        )

    # Check backend-specific requirements
    if backend == "pregenerated":
        if "xdmf_path" not in mesh_cfg:
            errors.append("mesh.xdmf_path required for 'pregenerated' backend")
        elif not Path(mesh_cfg["xdmf_path"]).exists():
            errors.append(f"Mesh file not found: {mesh_cfg['xdmf_path']}")

    if backend == "theta_mesh":
        theta_config = mesh_cfg.get("theta_config_path")
        if theta_config and not Path(theta_config).exists():
            errors.append(f"Theta config not found: {theta_config}")

    return errors


# ============================================================================
# Backward Compatibility
# ============================================================================


def load_mesh_for_mvp_legacy(config: Dict[str, Any]) -> MeshData:
    """
    Legacy interface for backward compatibility.

    Supports old config format with:
    - mesh.use_fallback: bool
    - mesh.use_pregenerated: bool
    - mesh.pregenerated_xdmf: str
    """
    mesh_cfg = config.get("mesh", {})

    # Map old config to new backend format
    if mesh_cfg.get("use_fallback", False):
        mesh_cfg["backend"] = "fallback"
    elif mesh_cfg.get("use_pregenerated", False):
        mesh_cfg["backend"] = "pregenerated"
        if "pregenerated_xdmf" in mesh_cfg:
            mesh_cfg["xdmf_path"] = mesh_cfg["pregenerated_xdmf"]
    elif "theta_config_path" in mesh_cfg or "theta_overrides" in mesh_cfg:
        mesh_cfg["backend"] = "theta_mesh"
    else:
        mesh_cfg["backend"] = "fallback"

    # Use new interface
    return load_mesh_for_mvp(config)
