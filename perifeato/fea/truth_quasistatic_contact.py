#!/usr/bin/env python3
"""
Transient dynamics FEA for 3D core-shell mesh impact simulation.

NOTE: Quasi-static mode is deprecated. See old_dev/truth_quasistatic_contact_backup.py
      for the original code. Only transient dynamics mode is supported.

Loads mesh with cell data (material_id, region_id, thickness_mm, theta_band, hemisphere),
rotates to test drop orientations, then runs Newmark-beta transient dynamics with
unilateral (penalty) contact against a rigid ground plane (z=0).

Outputs per-orientation results with:
- displacement (CG1 vector) - time-resolved VTK
- von_mises (DG0 scalar) - time-resolved VTK
- max_stress (DG0 scalar) - peak stress per cell over simulation
- theta_band, hemisphere (DG0 int) - for stress binning by band (in XDMF)
- cell tags: material_id (1=steel, 2=PLA), region_id (2=PLA protected, 3=PLA fracture)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import os
import time
from typing import Iterable

import numpy as np
import yaml
import meshio
import re

# Work around a common macOS MPICH/OFI shutdown error (`OFI poll failed ... utun*`)
# by preventing mpi4py from calling MPI_Finalize at interpreter exit.
os.environ.setdefault("MPI4PY_RC_FINALIZE", "0")

# FFCx/DOLFINx JIT can be slow or can get stuck if a previous compile left stale cache artifacts.
# Increase the default timeout a bit; HPC job scripts should also set XDG_CACHE_HOME to a job-local dir.
os.environ.setdefault("DOLFINX_JIT_TIMEOUT", "300")


def _cleanup_ffcx_cache_from_timeout_message(message: str) -> list[str]:
    """
    Best-effort cleanup for common FFCx JIT timeout cases where a stale `.c` file exists in the cache.
    Returns a list of removed paths (as strings).
    """
    removed: list[str] = []
    m = re.search(r"remove\\s+([^\\s\\)]+\\.c)", message)
    if not m:
        return removed
    from pathlib import Path as _Path

    c_path = _Path(m.group(1))
    try:
        parent = c_path.parent
        stem = c_path.stem  # libffcx_forms_<hash>
        if parent.is_dir():
            for p in parent.glob(stem + ".*"):
                try:
                    p.unlink()
                    removed.append(str(p))
                except FileNotFoundError:
                    pass
    except Exception:
        return removed
    return removed


@dataclass(frozen=True)
class Orientation:
    name: str
    mode: str
    rotation_deg_xyz: tuple[float, float, float] | None = None
    theta_deg: float | None = None
    phi_deg: float | None = None
    reference_axis: str = "x"


DEFAULT_ORIENTATIONS: list[Orientation] = [
    # Requested set (7 orientations for comprehensive coverage):
    # 1) Vertical (θ=180°) pole/hole impact - hole rim contacts ground, load at base
    #    This is consistent with other orientations where the specified point hits ground
    Orientation(name="vertical_theta180_pole", mode="impact", theta_deg=180.0, phi_deg=0.0, reference_axis="x"),
    # 2) Tipped 45° on protected zone (θ=45°, φ=90°)
    Orientation(name="tipped_theta45_phi90_protected", mode="impact", theta_deg=45.0, phi_deg=90.0, reference_axis="x"),
    # 3) Tipped 45° on fracture zone (θ=45°, φ=0°)
    Orientation(name="tipped_theta45_phi0_fracture", mode="impact", theta_deg=45.0, phi_deg=0.0, reference_axis="x"),
    # 4) Tipped 45° on edge (θ=45°, φ=10°)
    Orientation(name="tipped_theta45_phi10_edge", mode="impact", theta_deg=45.0, phi_deg=10.0, reference_axis="x"),
    # 5) Horizontal (θ=90°, φ=90°) protected side
    Orientation(name="horizontal_theta90_phi90_protected", mode="impact", theta_deg=90.0, phi_deg=90.0, reference_axis="z"),
    # 6) Horizontal (θ=90°, φ=0°) fracture side
    Orientation(name="horizontal_theta90_phi0_fracture", mode="impact", theta_deg=90.0, phi_deg=0.0, reference_axis="z"),
    # 7) Horizontal (θ=90°, φ=10°) near fracture edge
    Orientation(name="horizontal_theta90_phi10_edge", mode="impact", theta_deg=90.0, phi_deg=10.0, reference_axis="z"),
]


ORIENTATION_ALIASES: dict[str, str] = {
    # Backward-compatible names from earlier iterations of this script:
    "long_axis_horizontal_x": "horizontal_theta90_phi0_fracture",
    # Common user-friendly aliases:
    "vertical_theta0_base": "vertical_theta180_pole",  # Old name → new pole-first impact
    "vertical_theta0_fracture": "vertical_theta180_pole",
    "vertical_theta0_protected": "vertical_theta180_pole",
    "vertical": "vertical_theta180_pole",  # Short alias
    "horizontal_theta90_phi0_protected": "horizontal_theta90_phi90_protected",
    "horizontal_theta90_phi90_fracture": "horizontal_theta90_phi0_fracture",
}


def _rotation_matrix_xyz_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_m @ ry_m @ rx_m


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0:
        raise ValueError("Zero-length vector")
    return v / n


def _rotation_from_primary_secondary(
    *,
    primary_from: np.ndarray,
    secondary_from: np.ndarray,
    primary_to: np.ndarray,
    secondary_to: np.ndarray,
) -> np.ndarray:
    """
    Construct a rotation R such that:
      R * primary_from   = primary_to
      R * secondary_from ≈ secondary_to (after removing any component parallel to primary)
    """
    u1 = _unit(primary_from)
    s_from = secondary_from - np.dot(secondary_from, u1) * u1
    if np.linalg.norm(s_from) < 1e-12:
        raise ValueError("secondary_from is parallel to primary_from")
    u2 = _unit(s_from)
    u3 = np.cross(u1, u2)

    v1 = _unit(primary_to)
    s_to = secondary_to - np.dot(secondary_to, v1) * v1
    if np.linalg.norm(s_to) < 1e-12:
        raise ValueError("secondary_to is parallel to primary_to")
    v2 = _unit(s_to)
    v3 = np.cross(v1, v2)

    U = np.column_stack([u1, u2, u3])
    V = np.column_stack([v1, v2, v3])
    return V @ U.T


def _direction_from_theta_phi_deg(theta_deg: float, phi_deg: float) -> np.ndarray:
    """
    Convert (theta, phi) to a unit direction in the body frame.

    Conventions:
    - theta = 0 at +z pole, theta = 90 at equator
    - phi   = 0 along +x, phi = 90 along +y
    We use the LOWER hemisphere (impact side): z component is negative.
    """
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = -np.cos(theta)
    return _unit(np.array([x, y, z], dtype=float))


def _load_xdmf_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    msh = meshio.read(str(mesh_path))
    if "tetra" not in msh.cells_dict:
        raise RuntimeError(f"No tetra cells in {mesh_path}")
    if "material_id" not in msh.cell_data_dict or "tetra" not in msh.cell_data_dict["material_id"]:
        raise RuntimeError(f"Missing cell_data 'material_id' for tetra in {mesh_path}")
    if "region_id" not in msh.cell_data_dict or "tetra" not in msh.cell_data_dict["region_id"]:
        raise RuntimeError(f"Missing cell_data 'region_id' for tetra in {mesh_path}")
    if "thickness_mm" not in msh.cell_data_dict or "tetra" not in msh.cell_data_dict["thickness_mm"]:
        raise RuntimeError(f"Missing cell_data 'thickness_mm' for tetra in {mesh_path}")

    points = msh.points.astype(np.float64)
    tets = msh.cells_dict["tetra"].astype(np.int64)
    material_id = msh.cell_data_dict["material_id"]["tetra"].astype(np.int32)
    region_id = msh.cell_data_dict["region_id"]["tetra"].astype(np.int32)
    thickness_mm = msh.cell_data_dict["thickness_mm"]["tetra"].astype(np.float64)

    # Load theta_band if present (assigned during mesh generation, survives rotation)
    if "theta_band" in msh.cell_data_dict and "tetra" in msh.cell_data_dict["theta_band"]:
        theta_band = msh.cell_data_dict["theta_band"]["tetra"].astype(np.int32)
    else:
        # Fallback: compute from original z-coordinates (assumes mesh not yet rotated)
        print("  Warning: theta_band not in mesh, computing from z-coordinates")
        z_centroid = points[tets].mean(axis=1)[:, 2]
        c_core = 0.034  # Default 34mm
        theta = np.arccos(np.clip(np.abs(z_centroid) / c_core, 0.0, 1.0))
        theta_edges = np.deg2rad(np.linspace(0, 90, 11))
        theta_band = np.searchsorted(theta_edges, theta, side="right") - 1
        theta_band = np.clip(theta_band, 0, 9).astype(np.int32)

    # Load hemisphere if present (1=upper/+z, 0=lower/-z, assigned before rotation)
    if "hemisphere" in msh.cell_data_dict and "tetra" in msh.cell_data_dict["hemisphere"]:
        hemisphere = msh.cell_data_dict["hemisphere"]["tetra"].astype(np.int32)
    else:
        # Fallback: compute from original z-coordinates (assumes mesh not yet rotated)
        print("  Warning: hemisphere not in mesh, computing from z-coordinates")
        z_centroid = points[tets].mean(axis=1)[:, 2]
        hemisphere = (z_centroid >= 0).astype(np.int32)

    return points, tets, material_id, region_id, thickness_mm, theta_band, hemisphere


def _make_oriented_mesh(
    points: np.ndarray,
    tets: np.ndarray,
    material_id: np.ndarray,
    region_id: np.ndarray,
    thickness_mm: np.ndarray,
    theta_band: np.ndarray,
    hemisphere: np.ndarray,
    orientation: Orientation,
    clearance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if orientation.mode == "euler":
        if orientation.rotation_deg_xyz is None:
            raise ValueError(f"{orientation.name}: missing rotation_deg_xyz")
        r = _rotation_matrix_xyz_deg(*orientation.rotation_deg_xyz)
    elif orientation.mode == "impact":
        if orientation.theta_deg is None or orientation.phi_deg is None:
            raise ValueError(f"{orientation.name}: missing theta_deg/phi_deg")
        d_body = _direction_from_theta_phi_deg(orientation.theta_deg, orientation.phi_deg)

        if orientation.reference_axis == "x":
            ref_body = np.array([1.0, 0.0, 0.0])
        elif orientation.reference_axis == "z":
            ref_body = np.array([0.0, 0.0, 1.0])
        else:
            raise ValueError(f"{orientation.name}: reference_axis must be 'x' or 'z'")

        # Map desired impact direction to world down (-z).
        # Use world +x as the secondary reference to fix twist.
        r = _rotation_from_primary_secondary(
            primary_from=d_body,
            secondary_from=ref_body,
            primary_to=np.array([0.0, 0.0, -1.0]),
            secondary_to=np.array([1.0, 0.0, 0.0]),
        )
    else:
        raise ValueError(f"{orientation.name}: unknown mode {orientation.mode!r}")
    rotated = points @ r.T

    # Shift so the lowest point is just above the ground plane
    z_min = float(rotated[:, 2].min())
    rotated[:, 2] += (clearance_m - z_min)

    # theta_band and hemisphere are unchanged by rotation - assigned based on original mesh coordinates
    return rotated, tets, material_id, region_id, thickness_mm, theta_band, hemisphere


def _summarize_von_mises_by_region(von_mises: np.ndarray, material_id: np.ndarray, region_id: np.ndarray) -> str:
    def stat(mask: np.ndarray) -> tuple[float, float, float, float]:
        if not np.any(mask):
            return float("nan"), float("nan"), float("nan"), float("nan")
        vals = von_mises[mask]
        return (
            float(np.max(vals)),
            float(np.percentile(vals, 99.0)),
            float(np.percentile(vals, 95.0)),
            float(np.mean(vals)),
        )

    steel_max, steel_p99, steel_p95, steel_mean = stat(material_id == 1)
    pla_max, pla_p99, pla_p95, pla_mean = stat(material_id == 2)
    prot_max, prot_p99, prot_p95, prot_mean = stat((material_id == 2) & (region_id == 2))
    frac_max, frac_p99, frac_p95, frac_mean = stat((material_id == 2) & (region_id == 3))

    def fmt(mx: float, p99: float, p95: float, mean: float) -> str:
        return f"max={mx/1e6:.2f} MPa, p99={p99/1e6:.2f} MPa, p95={p95/1e6:.2f} MPa, mean={mean/1e6:.2f} MPa"

    return (
        f"Steel: {fmt(steel_max, steel_p99, steel_p95, steel_mean)}\n"
        f"PLA (all): {fmt(pla_max, pla_p99, pla_p95, pla_mean)}\n"
        f"PLA protected (region_id=2): {fmt(prot_max, prot_p99, prot_p95, prot_mean)}\n"
        f"PLA fracture (region_id=3): {fmt(frac_max, frac_p99, frac_p95, frac_mean)}"
    )


def _summarize_von_mises_by_region_parallel(von_mises: np.ndarray, material_id: np.ndarray, region_id: np.ndarray, comm) -> str:
    """
    MPI-safe summary without global percentiles (avoids gathering massive arrays).
    Reports: max and mean per region/material.
    """

    def stat(mask: np.ndarray) -> tuple[float, float]:
        if not np.any(mask):
            local_max = -np.inf
            local_sum = 0.0
            local_n = 0
        else:
            vals = von_mises[mask]
            local_max = float(np.max(vals))
            local_sum = float(np.sum(vals))
            local_n = int(vals.size)

        from mpi4py import MPI

        gmax = float(comm.allreduce(local_max, op=MPI.MAX))
        gsum = float(comm.allreduce(local_sum, op=MPI.SUM))
        gn = int(comm.allreduce(local_n, op=MPI.SUM))
        gmean = float(gsum / gn) if gn > 0 else float("nan")
        if not np.isfinite(gmax):
            gmax = float("nan")
        return float(gmax), float(gmean)

    steel_max, steel_mean = stat(material_id == 1)
    pla_max, pla_mean = stat(material_id == 2)
    prot_max, prot_mean = stat((material_id == 2) & (region_id == 2))
    frac_max, frac_mean = stat((material_id == 2) & (region_id == 3))

    def fmt(mx: float, mean: float) -> str:
        return f"max={mx/1e6:.2f} MPa, mean={mean/1e6:.2f} MPa"

    return (
        f"Steel: {fmt(steel_max, steel_mean)}\n"
        f"PLA (all): {fmt(pla_max, pla_mean)}\n"
        f"PLA protected (region_id=2): {fmt(prot_max, prot_mean)}\n"
        f"PLA fracture (region_id=3): {fmt(frac_max, frac_mean)}"
    )


def _hist_percentiles_parallel(values: np.ndarray, comm, percentiles: tuple[float, ...] = (95.0, 99.0), nbins: int = 256):
    """
    Approximate percentiles on MPI without gathering the full array, using a global histogram on [0, max(values)].
    Returns a dict percentile->value (float). Values are assumed non-negative.
    """
    from mpi4py import MPI

    local_max = float(np.max(values)) if values.size else -float("inf")
    gmax = float(comm.allreduce(local_max, op=MPI.MAX))
    if not np.isfinite(gmax) or gmax <= 0.0:
        return {p: 0.0 if gmax == 0.0 else float("nan") for p in percentiles}

    nbins_eff = int(max(32, nbins))
    edges = np.linspace(0.0, gmax, nbins_eff + 1, dtype=np.float64)
    local_hist, _ = np.histogram(values, bins=edges)
    global_hist = np.zeros_like(local_hist, dtype=np.int64)
    comm.Allreduce(local_hist.astype(np.int64), global_hist, op=MPI.SUM)

    total = int(np.sum(global_hist))
    if total <= 0:
        return {p: float("nan") for p in percentiles}
    cdf = np.cumsum(global_hist)

    out: dict[float, float] = {}
    for p in percentiles:
        target = int(np.ceil(float(p) / 100.0 * total))
        idx = int(np.searchsorted(cdf, target, side="left"))
        idx = max(0, min(idx, len(edges) - 2))
        out[p] = float(edges[idx + 1])
    return out


def _von_mises_metrics_parallel(von_mises: np.ndarray, material_id: np.ndarray, region_id: np.ndarray, comm) -> dict[str, dict[str, float]]:
    """
    Return numeric MPI-safe von Mises metrics (Pa) that are easy to parse.
    """
    from mpi4py import MPI

    def stat(mask: np.ndarray) -> dict[str, float]:
        if not np.any(mask):
            local_max = -np.inf
            local_sum = 0.0
            local_n = 0
            local_vals = np.zeros((0,), dtype=np.float64)
        else:
            local_vals = von_mises[mask].astype(np.float64, copy=False)
            local_max = float(np.max(local_vals))
            local_sum = float(np.sum(local_vals))
            local_n = int(local_vals.size)

        gmax = float(comm.allreduce(local_max, op=MPI.MAX))
        gsum = float(comm.allreduce(local_sum, op=MPI.SUM))
        gn = int(comm.allreduce(local_n, op=MPI.SUM))
        gmean = float(gsum / gn) if gn > 0 else float("nan")
        if not np.isfinite(gmax):
            gmax = float("nan")

        pct = _hist_percentiles_parallel(local_vals, comm, percentiles=(95.0, 99.0), nbins=256)
        return {
            "max_pa": float(gmax),
            "p99_pa": float(pct[99.0]) if 99.0 in pct else float("nan"),
            "p95_pa": float(pct[95.0]) if 95.0 in pct else float("nan"),
            "mean_pa": float(gmean),
            "n_cells": float(gn),
        }

    overall = stat(np.ones_like(von_mises, dtype=bool))
    steel = stat(material_id == 1)
    pla_all = stat(material_id == 2)
    pla_protected = stat((material_id == 2) & (region_id == 2))
    pla_fracture = stat((material_id == 2) & (region_id == 3))

    return {
        "overall": overall,
        "steel": steel,
        "pla_all": pla_all,
        "pla_protected": pla_protected,
        "pla_fracture": pla_fracture,
    }


def _smooth_pos(s, eps: float):
    """Smooth approximation of max(s, 0) for Newton stability."""
    import ufl

    return 0.5 * (s + ufl.sqrt(s * s + eps * eps))


def _first_dof_array(dofs) -> np.ndarray:
    """
    DOLFINx may return dofs as:
    - a 1D ndarray of int32
    - a 2D ndarray (e.g., Nx2)
    - a list/tuple of arrays when locating dofs for (subspace, collapsed_space)
    This helper normalizes to a 1D int32 array for use in `dirichletbc`.
    """
    if isinstance(dofs, (list, tuple)):
        if len(dofs) == 0:
            return np.zeros((0,), dtype=np.int32)
        dofs0 = dofs[0]
        return np.asarray(dofs0, dtype=np.int32).ravel()
    arr = np.asarray(dofs, dtype=np.int32)
    if arr.ndim == 2 and arr.shape[1] >= 1:
        return np.asarray(arr[:, 0], dtype=np.int32).ravel()
    return arr.ravel()


def _write_bc_markers_vtu(
    path: Path,
    *,
    coords: np.ndarray,
    contact_vertices: np.ndarray,
    top_vertices: np.ndarray,
    pinned_a: np.ndarray,
    pinned_b: np.ndarray,
    contact_centroid: np.ndarray,
    top_centroid: np.ndarray,
) -> None:
    """
    Write a small point-cloud VTU to make BC/contact selection obvious in ParaView.

    `marker_id` meanings:
      1 = contact vertices
      2 = top/load vertices
      3 = pinned point A
      4 = pinned point B
      5 = contact centroid
      6 = top centroid
    """

    points_list: list[np.ndarray] = []
    marker_id: list[int] = []

    if contact_vertices.size:
        points_list.append(coords[contact_vertices])
        marker_id.extend([1] * int(contact_vertices.size))
    if top_vertices.size:
        points_list.append(coords[top_vertices])
        marker_id.extend([2] * int(top_vertices.size))

    points_list.append(np.asarray(pinned_a, dtype=float).reshape(1, 3))
    marker_id.append(3)
    points_list.append(np.asarray(pinned_b, dtype=float).reshape(1, 3))
    marker_id.append(4)
    points_list.append(np.asarray(contact_centroid, dtype=float).reshape(1, 3))
    marker_id.append(5)
    points_list.append(np.asarray(top_centroid, dtype=float).reshape(1, 3))
    marker_id.append(6)

    points = np.vstack(points_list) if points_list else np.zeros((0, 3), dtype=float)
    marker_id_arr = np.asarray(marker_id, dtype=np.int32)

    cells = [("vertex", np.arange(len(points), dtype=np.int32).reshape(-1, 1))]
    markers = meshio.Mesh(points=points, cells=cells, point_data={"marker_id": marker_id_arr})
    meshio.write(str(path), markers)


# ============================================================================
# Transient Dynamics Helper Functions
# ============================================================================


def _assemble_mass_matrix(domain, V, rho: float):
    """
    Assemble consistent mass matrix M for transient dynamics.

    Args:
        domain: DOLFINx mesh domain
        V: Function space (vector CG1)
        rho: Material density in kg/m³

    Returns:
        M: PETSc matrix (assembled mass matrix)
    """
    from dolfinx import fem
    from petsc4py import PETSc
    import ufl

    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)

    # Consistent mass matrix: M_ij = ∫ ρ N_i · N_j dV
    rho_const = fem.Constant(domain, PETSc.ScalarType(rho))
    m_form = rho_const * ufl.dot(u_trial, v_test) * ufl.dx

    M = fem.petsc.assemble_matrix(fem.form(m_form))
    M.assemble()

    return M


def _compute_rayleigh_damping(damping_ratio: float, omega_1: float, omega_2: float):
    """
    Compute Rayleigh damping coefficients alpha and beta.

    C = alpha * M + beta * K

    For a given damping ratio ξ at two frequencies ω_1 and ω_2:
    ξ = (alpha/(2*ω) + beta*ω/2)

    Solving for alpha and beta:
    alpha = 2 * ξ * ω_1 * ω_2 / (ω_1 + ω_2)
    beta = 2 * ξ / (ω_1 + ω_2)

    Args:
        damping_ratio: Critical damping ratio (e.g., 0.03 for 3%)
        omega_1: First angular frequency (rad/s)
        omega_2: Second angular frequency (rad/s)

    Returns:
        (alpha, beta): Rayleigh damping coefficients
    """
    alpha = 2.0 * damping_ratio * omega_1 * omega_2 / (omega_1 + omega_2)
    beta = 2.0 * damping_ratio / (omega_1 + omega_2)

    return alpha, beta


def _initialize_transient_state(V, top_dofs_z, drop_height_m: float):
    """
    Initialize displacement, velocity, and acceleration for transient dynamics.

    For a drop test, the ENTIRE shell is falling at impact velocity.
    This applies initial velocity to all z-component DOFs, not just top surface.

    Args:
        V: Function space (vector CG1)
        top_dofs_z: DOF indices for z-component of top surface (kept for compatibility)
        drop_height_m: Drop height in meters (used to compute impact velocity)

    Returns:
        (u_n, v_n, a_n): Displacement, velocity, acceleration Functions
    """
    from dolfinx import fem
    import numpy as np

    u_n = fem.Function(V)
    v_n = fem.Function(V)
    a_n = fem.Function(V)

    # Zero initial displacement and acceleration
    u_n.x.array[:] = 0.0
    a_n.x.array[:] = 0.0

    # Compute impact velocity from drop height
    # v_0 = sqrt(2 * g * h), negative for downward motion
    v_0 = -np.sqrt(2.0 * 9.81 * drop_height_m)

    # Apply initial velocity to ENTIRE SHELL (all z-component DOFs)
    # The shell is falling as a rigid body at moment of impact
    # V is a vector function space with 3 components (x, y, z)
    # We need to set all z-component DOFs (every 3rd DOF starting at index 2)
    n_dofs = len(v_n.x.array)
    z_dofs = np.arange(2, n_dofs, 3)  # z-component is index 2 in (x,y,z)
    v_n.x.array[z_dofs] = v_0

    print(f"[Transient Init] Applied v_0={v_0:.2f} m/s to {len(z_dofs)} z-DOFs (entire shell)")

    return u_n, v_n, a_n


def _run_transient_dynamics(
    *,
    domain,
    V,
    u,  # Will be used as solution container
    v,  # UFL test function
    sigma,  # Stress function
    eps,  # Strain function
    contact_term,  # UFL contact force term
    bcs: list,  # Boundary conditions (pin constraints)
    top_dofs_z,  # Top surface DOFs (z-component)
    drop_mass_kg: float | None,
    drop_height_m: float | None,
    time_end_s: float,
    timestep_s: float,
    newmark_beta: float,
    newmark_gamma: float,
    damping_ratio: float,
    rayleigh_alpha: float,
    rayleigh_beta: float,
    contact_penalty: float,
    newton_max_it: int,
    newton_rtol: float,
    newton_atol: float,
    newton_relax: float,
    ksp_type: str,
    pc_type: str,
    ksp_rtol: float,
    ksp_atol: float,
    ksp_max_it: int,
    ksp_monitor: bool,
    ksp_monitor_true: bool,
    ksp_norm_type: str,
    pc_factor_solver: str | None,
    ksp_error_if_not_converged: bool,
    orientation,
    comm,
    out_dir: Path,
    output_interval: int,
    rho: float,
    theta_band_f,  # DG0 function with theta_band data for each cell
    hemisphere_f,  # DG0 function with hemisphere data (1=upper/+z, 0=lower/-z)
):
    """
    Run transient dynamics simulation using Newmark-beta time integration.

    Returns:
        (compression_used, energy_work_j, energy_steps_used, energy_stop_reason, total_newton_iters, iters_info)
    """
    from dolfinx import fem
    from dolfinx.nls.petsc import NewtonSolver
    from petsc4py import PETSc
    import ufl
    import numpy as np

    print(f"[{orientation.name}] Initializing transient dynamics...", flush=True)

    # Validate inputs
    if drop_mass_kg is None or drop_height_m is None:
        raise ValueError("Transient dynamics requires --drop-mass-kg and --drop-height-m")

    if drop_height_m <= 0:
        raise ValueError("--drop-height-m must be positive")

    # 1. Assemble mass matrix
    print(f"[{orientation.name}] Assembling mass matrix (rho={rho:.1f} kg/m³)...", flush=True)
    M = _assemble_mass_matrix(domain, V, rho)

    # 2. Compute Rayleigh damping coefficients (if not specified)
    if rayleigh_alpha == 0.0 and rayleigh_beta == 0.0 and damping_ratio > 0.0:
        # Estimate fundamental frequency from geometry and material
        # For a cylindrical shell: f_1 ≈ 100-500 Hz (rough estimate)
        omega_1 = 2.0 * np.pi * 100.0  # 100 Hz (fundamental mode)
        omega_2 = 2.0 * np.pi * 1000.0  # 1 kHz (higher mode to damp)
        rayleigh_alpha, rayleigh_beta = _compute_rayleigh_damping(damping_ratio, omega_1, omega_2)
        print(
            f"[{orientation.name}] Rayleigh damping: α={rayleigh_alpha:.3e}, β={rayleigh_beta:.3e} (ξ={damping_ratio:.3f})",
            flush=True,
        )

    # 3. Initialize state vectors
    print(f"[{orientation.name}] Initializing state (v_0={-np.sqrt(2*9.81*drop_height_m):.3f} m/s)...", flush=True)
    u_n, v_n, a_n = _initialize_transient_state(V, top_dofs_z, drop_height_m)

    # 4. Time integration setup
    n_steps = int(time_end_s / timestep_s)
    dt = timestep_s
    beta = newmark_beta
    gamma = newmark_gamma

    print(
        f"[{orientation.name}] Time integration: {n_steps} steps, dt={dt*1e6:.2f} μs, β={beta:.3f}, γ={gamma:.3f}",
        flush=True,
    )

    # Newmark effective coefficients
    c_M = 1.0 / (beta * dt**2)  # Mass coefficient
    c_C = gamma / (beta * dt)    # Damping coefficient (if used)

    # 5. Define Newmark residual form
    # At each timestep, we solve for u_{n+1} such that:
    # M * a_{n+1} + C * v_{n+1} + f_int(u_{n+1}) + f_contact(u_{n+1}) = 0
    #
    # Where:
    # a_{n+1} = (u_{n+1} - u_pred) / (beta * dt^2)
    # v_{n+1} = v_pred + gamma * dt * a_{n+1}
    #
    # Substituting:
    # M * (u_{n+1} - u_pred) / (beta*dt^2) + C * (v_{n+1} - v_pred) / (gamma*dt)
    #   + f_int(u_{n+1}) + f_contact(u_{n+1}) = 0
    #
    # Rearranging:
    # f_int(u_{n+1}) + f_contact(u_{n+1}) + M*c_M*u_{n+1} + C*c_C*u_{n+1}
    #   = M*c_M*u_pred + C*c_C*v_pred

    # Create functions for predictor states
    u_pred = fem.Function(V)
    v_pred = fem.Function(V)

    # Trial and test functions for residual
    u_trial = ufl.TrialFunction(V)

    # Internal forces (strain energy)
    f_internal = ufl.inner(sigma(u), eps(v)) * ufl.dx

    # Mass term contribution to residual
    # We'll add this dynamically in the time loop

    # 6. Time integration loop
    peak_stress = 0.0
    max_displacement = 0.0
    time = 0.0
    step = 0
    total_newton_iters = 0

    # Energy tracking
    initial_KE = 0.0

    print(f"[{orientation.name}] Starting time integration...", flush=True)

    # Define Newmark residual form
    # We need to solve: f_int(u) + f_contact(u) + c_M*M*u = c_M*M*u_pred (+ damping terms)
    #
    # Since M is a matrix, we can't directly include it in UFL.
    # Instead, we'll use a custom RHS approach with PETSc

    # Define mass contribution as UFL form (for action)
    u_trial_mass = ufl.TrialFunction(V)
    mass_form = rho * ufl.dot(u_trial_mass, v) * ufl.dx

    # Track peak stress and displacement
    peak_stress_pa = 0.0
    max_displacement_m = 0.0

    # Energy tracking
    Mv = M.createVecRight()
    M.mult(v_n.x.petsc_vec, Mv)
    initial_ke = 0.5 * Mv.dot(v_n.x.petsc_vec)

    print(f"[{orientation.name}] Initial KE = {initial_ke:.6f} J", flush=True)

    # Create VTK writer for time-resolved output
    from dolfinx import io
    vtk_cell_path = out_dir / f"{orientation.name}_cell.pvd"
    vtk_writer = io.VTKFile(comm, str(vtk_cell_path), "w")
    vtk_writer.write_mesh(domain)

    # Create DG0 function space for cell data (von Mises stress)
    from dolfinx.fem import functionspace
    import basix
    DG0 = functionspace(domain, ("DG", 0))
    vm_output = fem.Function(DG0, name="von_mises")

    # Per-cell max stress tracking across all timesteps
    # This captures peak stress that may occur mid-simulation during wave propagation
    max_stress_per_cell = fem.Function(DG0, name="max_stress")
    max_stress_per_cell.x.array[:] = 0.0

    # Track max stress more frequently than VTK output (every 5 steps = 0.1ms)
    # This balances accuracy with computational cost
    stress_track_interval = max(1, output_interval // 50) if output_interval > 50 else 1

    print(f"[{orientation.name}] VTK output: {vtk_cell_path}", flush=True)
    print(f"[{orientation.name}] Max stress tracking every {stress_track_interval} steps", flush=True)

    for step in range(1, n_steps + 1):
        time = step * dt

        # Newmark predictor
        u_pred.x.array[:] = u_n.x.array + dt * v_n.x.array + (dt**2 / 2.0) * (1.0 - 2.0*beta) * a_n.x.array
        v_pred.x.array[:] = v_n.x.array + dt * (1.0 - gamma) * a_n.x.array

        # Set initial guess for Newton (use predictor)
        u.x.array[:] = u_pred.x.array[:]

        # ====================================================================
        # Solve Newmark nonlinear system:
        # f_internal(u) + f_contact(u) + c_M * (M*u - M*u_pred) = 0
        # ====================================================================

        # Define residual with internal forces, contact, and mass term
        # Internal forces: ∫ σ(u) : ε(v) dx
        # Contact forces: already in contact_term
        # Mass term: c_M * ∫ ρ (u - u_pred) · v dx

        rho_const = fem.Constant(domain, PETSc.ScalarType(rho))
        c_M_const = fem.Constant(domain, PETSc.ScalarType(c_M))

        F_residual = ufl.inner(sigma(u), eps(v)) * ufl.dx + contact_term
        F_residual += c_M_const * rho_const * ufl.dot(u - u_pred, v) * ufl.dx

        # Solve nonlinear problem
        problem = fem.petsc.NonlinearProblem(F_residual, u, bcs=bcs)
        solver = NewtonSolver(comm, problem)

        # Configure Newton solver
        solver.convergence_criterion = "incremental"
        solver.rtol = newton_rtol
        solver.atol = newton_atol
        solver.max_it = newton_max_it
        if hasattr(solver, "relaxation_parameter"):
            solver.relaxation_parameter = newton_relax

        # Configure KSP
        ksp = solver.krylov_solver
        opts = PETSc.Options()
        prefix = ksp.getOptionsPrefix()
        opts[f"{prefix}ksp_type"] = ksp_type
        opts[f"{prefix}ksp_rtol"] = ksp_rtol
        opts[f"{prefix}ksp_atol"] = ksp_atol
        opts[f"{prefix}ksp_max_it"] = ksp_max_it
        opts[f"{prefix}pc_type"] = pc_type

        if pc_factor_solver and pc_type in {"lu", "cholesky"}:
            opts[f"{prefix}pc_factor_mat_solver_type"] = pc_factor_solver
            if pc_factor_solver.lower() == "mumps":
                opts[f"{prefix}mat_mumps_icntl_14"] = 80
                opts[f"{prefix}mat_mumps_icntl_23"] = 8000

        if ksp_norm_type != "default":
            opts[f"{prefix}ksp_norm_type"] = ksp_norm_type

        if ksp_monitor:
            if ksp_monitor_true:
                opts[f"{prefix}ksp_monitor_true_residual"] = None
            else:
                opts[f"{prefix}ksp_monitor"] = None
            opts[f"{prefix}ksp_converged_reason"] = None

        if ksp_error_if_not_converged:
            opts[f"{prefix}ksp_error_if_not_converged"] = 1

        ksp.setFromOptions()

        # Solve
        n_iters, converged = solver.solve(u)
        total_newton_iters += n_iters

        if not converged:
            print(
                f"[{orientation.name}] WARNING: Newton did not converge at t={time*1000:.3f} ms (step {step}, iters={n_iters})",
                flush=True,
            )
            # Continue anyway for now (could make this fatal)

        # Newmark corrector: compute acceleration and velocity
        a_n.x.array[:] = (u.x.array - u_pred.x.array) / (beta * dt**2)
        v_n.x.array[:] = v_pred.x.array + gamma * dt * a_n.x.array

        # Track displacement
        current_disp = abs(u.x.array[top_dofs_z].min()) if top_dofs_z.size > 0 else 0.0
        max_displacement_m = max(max_displacement_m, current_disp)

        # Track per-cell max stress more frequently than VTK output
        # This captures peak stress during transient wave propagation
        if step % stress_track_interval == 0 or step == 1:
            # Compute von Mises stress for max tracking
            sig = sigma(u)
            s_dev = sig - (1.0 / 3.0) * ufl.tr(sig) * ufl.Identity(3)
            von_mises_expr = ufl.sqrt(3.0 / 2.0 * ufl.inner(s_dev, s_dev))
            vm_output.interpolate(fem.Expression(von_mises_expr, DG0.element.interpolation_points()))

            # Update per-cell maximum (element-wise max)
            max_stress_per_cell.x.array[:] = np.maximum(
                max_stress_per_cell.x.array,
                vm_output.x.array
            )

            # Track global peak
            current_max_stress = float(vm_output.x.array.max()) if vm_output.x.array.size > 0 else 0.0
            peak_stress_pa = max(peak_stress_pa, current_max_stress)

        # Write VTK output (less frequently to save disk space)
        if step % output_interval == 0 or step == 1:
            # Ensure stress is computed if not already done in tracking block
            if step % stress_track_interval != 0 and step != 1:
                sig = sigma(u)
                s_dev = sig - (1.0 / 3.0) * ufl.tr(sig) * ufl.Identity(3)
                von_mises_expr = ufl.sqrt(3.0 / 2.0 * ufl.inner(s_dev, s_dev))
                vm_output.interpolate(fem.Expression(von_mises_expr, DG0.element.interpolation_points()))
                current_max_stress = float(vm_output.x.array.max()) if vm_output.x.array.size > 0 else 0.0
                peak_stress_pa = max(peak_stress_pa, current_max_stress)
                # Also update max tracking
                max_stress_per_cell.x.array[:] = np.maximum(
                    max_stress_per_cell.x.array,
                    vm_output.x.array
                )

            # Write current stress to VTK
            vtk_writer.write_function(vm_output, time)

            # Compute kinetic energy
            v_vec_temp = M.createVecRight()
            M.mult(v_n.x.petsc_vec, v_vec_temp)
            ke = 0.5 * v_vec_temp.dot(v_n.x.petsc_vec)

            # Compute P95 stress for more meaningful convergence metric
            current_p95_stress = float(np.percentile(vm_output.x.array, 95)) if vm_output.x.array.size > 0 else 0.0

            print(
                f"[{orientation.name}] t={time*1000:.3f} ms: disp={current_disp*1000:.3f} mm, "
                f"max={current_max_stress/1e6:.1f} P95={current_p95_stress/1e6:.1f} MPa, KE={ke:.6f} J",
                flush=True,
            )

        # Update for next step
        u_n.x.array[:] = u.x.array[:]

        # Early stopping if simulation has settled
        if time > 0.005:  # After 5 ms
            v_max = abs(v_n.x.array).max()
            if v_max < 1e-3:  # Negligible velocity (1 mm/s)
                print(f"[{orientation.name}] Settled at t={time*1000:.1f} ms (v_max={v_max:.3e} m/s)", flush=True)
                break

    # Close time-resolved VTK writer
    vtk_writer.close()
    print(f"[{orientation.name}] VTK output complete: {vtk_cell_path}", flush=True)

    # Write max stress per cell to separate VTK file (for optimization)
    # This contains the peak stress each cell experienced during the entire simulation
    # NOTE: DOLFINx VTKFile.write_function with a list doesn't reliably write all functions.
    # See: https://fenicsproject.discourse.group/t/how-to-write-multiple-functions-into-one-file-with-dolfinx-io-vtkfile/14980
    # Fix: Write theta_band and hemisphere to separate files, or use XDMF.
    max_stress_vtk_path = out_dir / f"{orientation.name}_max_stress.pvd"
    max_stress_writer = io.VTKFile(comm, str(max_stress_vtk_path), "w")
    max_stress_writer.write_mesh(domain)
    max_stress_writer.write_function([max_stress_per_cell], 0.0)
    max_stress_writer.close()

    # Write theta_band and hemisphere to XDMF (more reliable for multiple fields)
    # This allows proper stress binning by theta_band after mesh rotation
    cell_data_xdmf_path = out_dir / f"{orientation.name}_cell_data.xdmf"
    with io.XDMFFile(comm, str(cell_data_xdmf_path), "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(theta_band_f, 0.0)
        xdmf.write_function(hemisphere_f, 0.0)

    global_max_stress = float(max_stress_per_cell.x.array.max()) if max_stress_per_cell.x.array.size > 0 else 0.0
    print(f"[{orientation.name}] Max stress per cell written: {max_stress_vtk_path}", flush=True)
    print(f"[{orientation.name}] Peak stress during simulation: {global_max_stress/1e6:.2f} MPa", flush=True)

    # Compute final metrics
    compression_used = max_displacement_m
    energy_work_j = initial_ke  # Approximate (could compute strain energy too)
    energy_steps_used = step
    energy_stop_reason = "time_end" if step >= n_steps else "settled"
    iters_info = f"transient_newmark: {step} steps, {total_newton_iters} Newton iters"

    print(
        f"[{orientation.name}] Transient simulation complete: {step} steps, compression={compression_used*1000:.3f} mm",
        flush=True,
    )

    return compression_used, energy_work_j, energy_steps_used, energy_stop_reason, total_newton_iters, iters_info


def run_orientation(
    *,
    mesh_path: Path,
    config_path: Path,
    orientation: Orientation,
    out_dir: Path,
    loadcase: str = "ground",
    simulation_mode: str = "transient",  # Only transient mode is supported (quasi-static deprecated)
    max_displacement_mm: float = 20.0,
    displacement_step_mm: float = 0.2,
    cone_half_angle_deg: float = 55.0,
    cone_penetration_mm: float = 3.0,
    cone_axial_band_mm: float = 3.0,
    cone_radial_band_mm: float = 2.0,
    contact_penalty: float,
    contact_model: str,
    clearance_mm: float,
    top_tol_mm: float,
    contact_tol_mm: float,
    top_material: str = "pla",
    neumann_pressure_mpa: float | None = None,
    neumann_max_displacement_mm: float = 50.0,
    # Transient dynamics parameters (NEW)
    time_end_s: float = 0.030,
    timestep_s: float = 1.0e-6,
    time_integration: str = "newmark",
    newmark_beta: float = 0.25,
    newmark_gamma: float = 0.5,
    damping_ratio: float = 0.03,
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 0.0,
    output_interval: int = 100,
    # Solver parameters
    ksp_monitor: bool = False,
    ksp_type: str | None = None,
    pc_type: str | None = None,
    ksp_rtol: float = 1e-8,
    ksp_atol: float = 0.0,
    ksp_max_it: int = 2000,
    ksp_norm_type: str = "default",
    ksp_monitor_true: bool = False,
    pc_factor_solver: str | None = None,
    ksp_error_if_not_converged: bool = True,
    newton_max_it: int = 40,
    newton_rtol: float = 1e-8,
    newton_atol: float = 1e-10,
    newton_relax: float = 1.0,
    max_load_steps: int = 100,
    contact_smooth_eps: float = 1e-9,
    penalty_initial_guess: str = "dirichlet",
    pin_mode: str = "contact",
    pin_z: bool = True,
    write_bc_markers: bool = True,
    output_format: str = "auto",
    jit_only: bool = False,
    target_work_j: float | None = None,
    drop_mass_kg: float | None = None,
    drop_height_m: float | None = None,
    stop_on_energy: bool = False,
) -> Path:
    # Lazy import of DOLFINx stack (so `--help` works even without it)
    from mpi4py import MPI
    from dolfinx import fem, mesh as dmesh, io
    from dolfinx.fem.petsc import LinearProblem
    from dolfinx.nls.petsc import NewtonSolver
    import ufl
    from petsc4py import PETSc
    import basix
    import traceback

    comm = MPI.COMM_WORLD

    config = yaml.safe_load(config_path.read_text())
    t0 = time.perf_counter()
    loadcase_eff = (loadcase or "ground").strip().lower()
    if loadcase_eff not in {"ground", "cone_rim"}:
        raise ValueError("--loadcase must be one of: ground, cone_rim")

    # Validate simulation mode
    simulation_mode_eff = (simulation_mode or "transient").strip().lower()
    if simulation_mode_eff != "transient":
        raise ValueError("--simulation-mode must be 'transient' (quasi-static is deprecated, see old_dev/)")

    energy_target_j = float(target_work_j) if target_work_j is not None else None
    if energy_target_j is not None and energy_target_j <= 0.0:
        raise ValueError("--target-work-j must be positive")

    mat: np.ndarray
    reg: np.ndarray
    thickness_mm: np.ndarray

    # Prefer `.msh` for MPI runs (distributed mesh) to avoid OOM on large problems.
    if mesh_path.suffix.lower() == ".msh":
        from dolfinx.io import gmshio

        domain, cell_tags, _ = gmshio.read_from_msh(str(mesh_path), comm, rank=0, gdim=3)
        domain.name = f"truth_{orientation.name}"

        # Apply orientation rotation + clearance shift in-place on distributed coordinates.
        coords = domain.geometry.x
        if orientation.mode == "euler":
            if orientation.rotation_deg_xyz is None:
                raise ValueError(f"{orientation.name}: missing rotation_deg_xyz")
            r = _rotation_matrix_xyz_deg(*orientation.rotation_deg_xyz)
        elif orientation.mode == "impact":
            if orientation.theta_deg is None or orientation.phi_deg is None:
                raise ValueError(f"{orientation.name}: missing theta_deg/phi_deg")
            d_body = _direction_from_theta_phi_deg(orientation.theta_deg, orientation.phi_deg)
            if orientation.reference_axis == "x":
                ref_body = np.array([1.0, 0.0, 0.0])
            elif orientation.reference_axis == "z":
                ref_body = np.array([0.0, 0.0, 1.0])
            else:
                raise ValueError(f"{orientation.name}: reference_axis must be 'x' or 'z'")
            r = _rotation_from_primary_secondary(
                primary_from=d_body,
                secondary_from=ref_body,
                primary_to=np.array([0.0, 0.0, -1.0]),
                secondary_to=np.array([1.0, 0.0, 0.0]),
            )
        else:
            raise ValueError(f"{orientation.name}: unknown mode {orientation.mode!r}")

        coords[:] = coords @ r.T
        local_min_z = float(coords[:, 2].min()) if coords.size else float("inf")
        global_min_z = float(comm.allreduce(local_min_z, op=MPI.MIN))
        coords[:, 2] += (clearance_mm / 1000.0 - global_min_z)

        tdim = domain.topology.dim
        n_cells = domain.topology.index_map(tdim).size_local

        # Map gmsh physical tags -> region/material/thickness
        if cell_tags is None or cell_tags.dim != tdim:
            raise RuntimeError(
                "Missing cell physical tags in .msh mesh. Ensure the gmsh physical groups "
                "are exported for all volume regions (steel/protected/fracture)."
            )
        physical = np.zeros(n_cells, dtype=np.int32)
        physical[cell_tags.indices] = cell_tags.values.astype(np.int32)
        if np.any(physical == 0):
            raise RuntimeError(
                "Some cells are missing physical tags (tag=0). This will corrupt material/region assignment; "
                "regenerate the mesh with full physical groups."
            )
        unknown_tags = sorted({int(v) for v in np.unique(physical) if v not in (1, 2, 3)})
        if unknown_tags:
            raise RuntimeError(f"Unexpected physical tags in mesh: {unknown_tags}. Expected only 1, 2, or 3.")

        # Convention for refined mesh generator:
        # physical 1=steel, 2=pla_protected, 3=pla_fracture
        mat = np.full(n_cells, 2, dtype=np.int32)
        mat[physical == 1] = 1
        reg = physical.copy()
        reg[reg == 0] = mat[reg == 0]

        t_prot = float(config["material"]["min_thickness_mm"])
        t_frac = float(config["material"]["min_thickness_mm"]) * float(config["fracture_design"].get("initial_thickness_factor", 0.6))
        thickness_mm = np.zeros(n_cells, dtype=np.float64)
        thickness_mm[physical == 2] = t_prot
        thickness_mm[physical == 3] = t_frac

        # Compute theta_band from ORIGINAL (pre-rotation) mesh coordinates
        # For .msh path, we rotated coords in-place, so use the rotation inverse to get original z
        r_inv = r.T  # Rotation matrix inverse = transpose for orthogonal matrices
        # Get cell centers from domain using topology
        domain.topology.create_connectivity(tdim, 0)
        c_to_v = domain.topology.connectivity(tdim, 0)
        cell_centers_rotated = np.zeros((n_cells, 3), dtype=np.float64)
        for cell_idx in range(n_cells):
            vert_indices = c_to_v.links(cell_idx)
            cell_centers_rotated[cell_idx] = coords[vert_indices].mean(axis=0)
        # Undo rotation to get original coordinates (before shift and rotation)
        cell_centers_original = (cell_centers_rotated - np.array([0, 0, clearance_mm / 1000.0 - global_min_z])) @ r_inv.T
        z_centroid_original = cell_centers_original[:, 2]
        c_core = float(config.get("geometry", {}).get("inner_radius_z_mm", 34.0)) / 1000.0
        theta = np.arccos(np.clip(np.abs(z_centroid_original) / c_core, 0.0, 1.0))
        theta_edges = np.deg2rad(np.linspace(0, 90, 11))
        theta_band = np.searchsorted(theta_edges, theta, side="right") - 1
        theta_band = np.clip(theta_band, 0, 9).astype(np.int32)

        # Compute hemisphere from original z-coordinates (1=upper/+z, 0=lower/-z)
        hemisphere = (z_centroid_original >= 0).astype(np.int32)

        if comm.rank == 0:
            if not np.any(physical == 1):
                print(f"[{orientation.name}] warning: mesh has no steel cells (physical tag 1).", flush=True)
            if not np.any((physical == 2) | (physical == 3)):
                print(f"[{orientation.name}] warning: mesh has no PLA cells (physical tags 2/3).", flush=True)

        if comm.rank == 0:
            print(
                f"[{orientation.name}] mesh loaded+rotated in {time.perf_counter()-t0:.2f}s (mpi ranks={comm.size})",
                flush=True,
            )
    else:
        if comm.size != 1:
            raise RuntimeError(
                "For MPI runs, pass a `.msh` mesh so DOLFINx can distribute it "
                "(e.g. results_refined/core_shell_refined.msh)."
            )
        pts, tets, mat, reg, thickness_mm, theta_band, hemisphere = _load_xdmf_mesh(mesh_path)
        pts, tets, mat, reg, thickness_mm, theta_band, hemisphere = _make_oriented_mesh(
            pts,
            tets,
            mat,
            reg,
            thickness_mm,
            theta_band,
            hemisphere,
            orientation=orientation,
            clearance_m=clearance_mm / 1000.0,
        )

        coord_element = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
        domain = dmesh.create_mesh(comm, tets, pts, coord_element)
        domain.name = f"truth_{orientation.name}"
        if comm.rank == 0:
            print(f"[{orientation.name}] mesh loaded+rotated in {time.perf_counter()-t0:.2f}s", flush=True)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(fdim, 0)
    domain.topology.create_connectivity(0, tdim)

    n_cells = domain.topology.index_map(tdim).size_local
    if n_cells != len(mat) or n_cells != len(reg) or n_cells != len(thickness_mm) or n_cells != len(theta_band) or n_cells != len(hemisphere):
        raise RuntimeError(
            f"Cell tag length mismatch: mesh has {n_cells} cells but "
            f"material_id={len(mat)} region_id={len(reg)} thickness_mm={len(thickness_mm)} theta_band={len(theta_band)} hemisphere={len(hemisphere)}"
        )

    # DOLFINx may reorder cells internally for efficiency. Use original_cell_index
    # to map our cell data arrays to the correct DOLFINx cell ordering.
    # original_cell_index[i] = original input cell index for DOLFINx cell i
    orig_idx = domain.topology.original_cell_index
    if orig_idx is not None and len(orig_idx) == n_cells:
        mat = mat[orig_idx]
        reg = reg[orig_idx]
        thickness_mm = thickness_mm[orig_idx]
        theta_band = theta_band[orig_idx]
        hemisphere = hemisphere[orig_idx]
        if comm.rank == 0:
            print(f"[{orientation.name}] Cell data reordered using original_cell_index", flush=True)

    cell_entities = np.arange(n_cells, dtype=np.int32)

    material_tags = dmesh.meshtags(domain, tdim, cell_entities, mat.astype(np.int32))
    material_tags.name = f"{domain.name}_material_id"
    region_tags = dmesh.meshtags(domain, tdim, cell_entities, reg.astype(np.int32))
    region_tags.name = f"{domain.name}_region_id"
    theta_band_tags = dmesh.meshtags(domain, tdim, cell_entities, theta_band.astype(np.int32))
    theta_band_tags.name = f"{domain.name}_theta_band"

    # Function spaces
    V = fem.functionspace(domain, ("CG", 1, (3,)))
    DG0 = fem.functionspace(domain, ("DG", 0))
    ndofs_local = int(V.dofmap.index_map.size_local * V.dofmap.index_map_bs)
    ndofs_global = int(comm.allreduce(ndofs_local, op=MPI.SUM))
    if comm.rank == 0:
        print(f"[{orientation.name}] dofs={ndofs_global}", flush=True)
        if comm.size == 1 and ndofs_global >= 1_000_000:
            print(
                f"[{orientation.name}] warning: very large problem on 1 rank; expect OOM/slow solves. "
                f"Prefer MPI + `.msh` input (e.g. `srun -n 8 ... --mesh results_refined/core_shell_refined.msh`).",
                flush=True,
            )

    # Also export as DG0 Functions so ParaView always shows them as cell arrays.
    # (dolfinx MeshTags sometimes don't appear as "Cell Data" arrays in ParaView.)
    mat_f = fem.Function(DG0, name="material_id")
    reg_f = fem.Function(DG0, name="region_id")
    thick_f = fem.Function(DG0, name="thickness_mm")
    theta_band_f = fem.Function(DG0, name="theta_band")
    hemisphere_f = fem.Function(DG0, name="hemisphere")
    mat_f.x.array[:] = mat.astype(np.float64)
    reg_f.x.array[:] = reg.astype(np.float64)
    thick_f.x.array[:] = thickness_mm.astype(np.float64)
    theta_band_f.x.array[:] = theta_band.astype(np.float64)
    hemisphere_f.x.array[:] = hemisphere.astype(np.float64)

    # Collapse component subspaces for robust DOF location on subspaces (DOLFINx 0.9)
    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    Vz, _ = V.sub(2).collapse()

    # Material fields (DG0, per cell)
    E_steel = 200e9
    nu_steel = 0.30
    E_pla = float(config["material"]["youngs_modulus_gpa"]) * 1e9
    nu_pla = float(config["material"]["poissons_ratio"])
    rho = float(config["material"].get("density_kg_m3", 1250.0))  # PLA density, default 1250 kg/m³

    E = fem.Function(DG0)
    nu = fem.Function(DG0)
    E_vals = E.x.array
    nu_vals = nu.x.array
    E_vals[:] = E_pla
    nu_vals[:] = nu_pla
    steel_mask = mat == 1
    E_vals[steel_mask] = E_steel
    nu_vals[steel_mask] = nu_steel
    E.x.array[:] = E_vals
    nu.x.array[:] = nu_vals

    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Displacement (unknown)
    u = fem.Function(V, name="displacement")
    v = ufl.TestFunction(V)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lmbda * ufl.tr(eps(w)) * ufl.Identity(3)

    # Boundary detection
    x = ufl.SpatialCoordinate(domain)
    coords = domain.geometry.x
    local_z_max = float(coords[:, 2].max()) if coords.size else -np.inf
    z_max = float(comm.allreduce(local_z_max, op=MPI.MAX))
    local_z_min = float(coords[:, 2].min()) if coords.size else np.inf
    z_min = float(comm.allreduce(local_z_min, op=MPI.MIN))
    top_tol = top_tol_mm / 1000.0
    contact_tol = contact_tol_mm / 1000.0

    boundary_facets = np.asarray(dmesh.exterior_facet_indices(domain.topology), dtype=np.int32)
    f_to_v = domain.topology.connectivity(fdim, 0)

    facet_z_max = np.array([coords[f_to_v.links(int(f)), 2].max() for f in boundary_facets], dtype=float)
    facet_z_min = np.array([coords[f_to_v.links(int(f)), 2].min() for f in boundary_facets], dtype=float)

    top_facets = np.unique(boundary_facets[facet_z_max > (z_max - top_tol)]).astype(np.int32)
    top_material_eff = (top_material or "pla").strip().lower()
    if top_material_eff not in {"all", "pla", "steel"}:
        raise ValueError("--top-material must be one of: all, pla, steel")
    if top_material_eff != "all":
        target = 2 if top_material_eff == "pla" else 1
        f_to_c = domain.topology.connectivity(fdim, tdim)
        filtered = []
        for f in top_facets:
            cells = f_to_c.links(int(f))
            # Use majority logic: facet belongs to target material if >50% of adjacent cells match.
            # This prevents selecting facets that only touch a single cell of the target material
            # while most adjacent cells are a different material.
            if cells.size and np.sum(mat[cells] == target) > len(cells) / 2:
                filtered.append(int(f))
        top_facets = np.asarray(filtered, dtype=np.int32)
    top_facets_global = int(comm.allreduce(int(top_facets.size), op=MPI.SUM))
    if top_facets_global == 0:
        # Fallback 1: retry without material filter.
        if top_material_eff != "all":
            top_material_eff = "all"
            top_facets = np.unique(boundary_facets[facet_z_max > (z_max - top_tol)]).astype(np.int32)
            top_facets_global = int(comm.allreduce(int(top_facets.size), op=MPI.SUM))
        # Fallback 2: widen tolerance based on span.
        if top_facets_global == 0:
            z_span = float(z_max - z_min)
            fallback_tol = max(top_tol, 0.05 * z_span)
            top_facets = np.unique(boundary_facets[facet_z_max > (z_max - fallback_tol)]).astype(np.int32)
            top_facets_global = int(comm.allreduce(int(top_facets.size), op=MPI.SUM))
        # Fallback 3: grab facets touching the absolute max z.
        if top_facets_global == 0:
            eps = max(1e-12, 1e-6 * max(1.0, abs(z_max)))
            top_facets = np.unique(boundary_facets[facet_z_max >= (z_max - eps)]).astype(np.int32)
            top_facets_global = int(comm.allreduce(int(top_facets.size), op=MPI.SUM))
        # Fallback 4: percentile of boundary facets by max z (always non-empty).
        if top_facets_global == 0 and boundary_facets.size:
            keep = max(1, int(np.ceil(0.02 * len(boundary_facets))))
            idx = np.argsort(facet_z_max)[-keep:]
            top_facets = np.unique(boundary_facets[idx]).astype(np.int32)
            top_facets_global = int(comm.allreduce(int(top_facets.size), op=MPI.SUM))
        if top_facets_global == 0:
            raise RuntimeError(
                "Could not locate any top facets; increase `--top-tol-mm` or set `--top-material all`."
            )

    contact_facets = np.unique(boundary_facets[facet_z_min < (z_min + contact_tol)]).astype(np.int32)
    contact_facets_global = int(comm.allreduce(int(contact_facets.size), op=MPI.SUM))
    if contact_facets_global == 0:
        # Fallback 1: widen tolerance based on span (mirrors top facet fallback 2).
        z_span = float(z_max - z_min)
        fallback_tol = max(contact_tol, 0.05 * z_span)
        contact_facets = np.unique(boundary_facets[facet_z_min < (z_min + fallback_tol)]).astype(np.int32)
        contact_facets_global = int(comm.allreduce(int(contact_facets.size), op=MPI.SUM))
    if contact_facets_global == 0:
        # Fallback 2: grab facets touching the absolute min z (mirrors top facet fallback 3).
        eps = max(1e-12, 1e-6 * max(1.0, abs(z_min)))
        contact_facets = np.unique(boundary_facets[facet_z_min <= (z_min + eps)]).astype(np.int32)
        contact_facets_global = int(comm.allreduce(int(contact_facets.size), op=MPI.SUM))
    if contact_facets_global == 0 and boundary_facets.size:
        # Fallback 3: percentile of boundary facets by min z (mirrors top facet fallback 4).
        keep = max(1, int(np.ceil(0.02 * len(boundary_facets))))
        idx = np.argsort(facet_z_min)[:keep]  # lowest z facets
        contact_facets = np.unique(boundary_facets[idx]).astype(np.int32)
        contact_facets_global = int(comm.allreduce(int(contact_facets.size), op=MPI.SUM))
    if contact_facets_global == 0:
        raise RuntimeError("Could not locate any contact facets; increase `--contact-tol-mm` or decrease clearance.")
    boundary_facets_global = int(comm.allreduce(int(len(boundary_facets)), op=MPI.SUM))
    if comm.rank == 0:
        print(
            f"[{orientation.name}] boundary facets: total={boundary_facets_global} "
            f"top={top_facets_global} (material={top_material_eff}) contact={contact_facets_global}",
            flush=True,
        )

    # Cone loadcase uses a different "contact" selection: facets near the open-end hole rim.
    # Keep the meshtag name `contact_mt` so output files remain consistent.
    contact_facets_eff = contact_facets
    if loadcase_eff == "cone_rim":
        hole_r_m = float(config.get("geometry", {}).get("hole_radius_mm", 0.0)) / 1000.0
        if hole_r_m <= 0:
            raise RuntimeError("cone_rim loadcase requires geometry.hole_radius_mm > 0 in config.yaml")
        axial_band = float(cone_axial_band_mm) / 1000.0
        radial_band = float(cone_radial_band_mm) / 1000.0
        facet_r_mean = np.array(
            [
                float(np.mean(np.sqrt(np.sum(coords[f_to_v.links(int(f)), :2] ** 2, axis=1))))
                for f in boundary_facets
            ],
            dtype=float,
        )
        cone_mask = (facet_z_min < (z_min + axial_band)) & (np.abs(facet_r_mean - hole_r_m) < radial_band)
        contact_facets_eff = np.unique(boundary_facets[cone_mask]).astype(np.int32)
        cone_global = int(comm.allreduce(int(contact_facets_eff.size), op=MPI.SUM))
        if cone_global == 0:
            raise RuntimeError(
                "cone_rim: could not locate hole-rim facets; try increasing --cone-axial-band-mm or --cone-radial-band-mm"
            )
        if comm.rank == 0:
            print(
                f"[{orientation.name}] cone_rim facets: {cone_global} (hole_r={hole_r_m*1000:.2f}mm, axial_band={cone_axial_band_mm}mm, radial_band={cone_radial_band_mm}mm)",
                flush=True,
            )

    # Mark contact (or cone) facets for ds
    contact_values = np.full(len(contact_facets_eff), 1, dtype=np.int32)
    contact_mt = dmesh.meshtags(domain, fdim, contact_facets_eff, contact_values)
    contact_mt.name = f"{domain.name}_contact_facets"
    ds = ufl.Measure("ds", domain=domain, subdomain_data=contact_mt)

    # Mark top facets too (diagnostics + visualization)
    top_values = np.full(len(top_facets), 1, dtype=np.int32)
    top_mt = dmesh.meshtags(domain, fdim, top_facets, top_values)
    top_mt.name = f"{domain.name}_top_facets"

    ds_top = ufl.Measure("ds", domain=domain, subdomain_data=top_mt)
    traction = ufl.dot(sigma(u), ufl.FacetNormal(domain))
    reaction_forms = [fem.form(traction[i] * ds_top(1)) for i in range(3)]

    def _assemble_reaction_force() -> list[float]:
        reaction = []
        for form in reaction_forms:
            local = fem.assemble_scalar(form)
            total = comm.allreduce(local, op=MPI.SUM)
            reaction.append(-float(total))
        return reaction

    # Diagnostic masks (vertex-based) to visualize which regions were selected for contact/loading
    Q = fem.functionspace(domain, ("CG", 1))
    contact_mask = fem.Function(Q, name="contact_mask")
    top_mask = fem.Function(Q, name="top_mask")
    bc_id = fem.Function(Q, name="bc_id")  # 0=none, 1=contact, 2=top, 3=both
    contact_mask.x.array[:] = 0.0
    top_mask.x.array[:] = 0.0
    bc_id.x.array[:] = 0.0

    # In MPI, a given rank may own zero of the selected facets even if the global set is non-empty.
    # Guard against concatenating an empty list.
    if contact_facets_eff.size:
        contact_vertices = np.unique(np.concatenate([f_to_v.links(int(f)) for f in contact_facets_eff])).astype(np.int32)
    else:
        contact_vertices = np.zeros((0,), dtype=np.int32)
    if top_facets.size:
        top_vertices = np.unique(np.concatenate([f_to_v.links(int(f)) for f in top_facets])).astype(np.int32)
    else:
        top_vertices = np.zeros((0,), dtype=np.int32)

    contact_dofs = fem.locate_dofs_topological(Q, 0, contact_vertices)
    top_dofs = fem.locate_dofs_topological(Q, 0, top_vertices)
    contact_mask.x.array[contact_dofs] = 1.0
    top_mask.x.array[top_dofs] = 1.0
    bc_id.x.array[contact_dofs] = 1.0
    bc_id.x.array[top_dofs] += 2.0

    # Dirichlet: prescribed displacement on top, z-component only
    # - ground: adaptive displacement up to max_displacement_mm (stops early when energy target reached)
    # - cone_rim: hold the far end fixed in z (max_displacement=0) while the cone obstacle penetrates from below
    max_displacement = float(max_displacement_mm) / 1000.0  # Convert mm to m
    displacement_step = float(displacement_step_mm) / 1000.0  # Convert mm to m
    if loadcase_eff == "cone_rim":
        max_displacement = 0.0
    top_dofs_z = _first_dof_array(fem.locate_dofs_topological((V.sub(2), Vz), fdim, top_facets))
    bc_top = fem.dirichletbc(PETSc.ScalarType(0.0), top_dofs_z, V.sub(2))  # Initial BC (will be updated in loop)

    # Rigid-body stabilization:
    # - Contact (u_z=0) does not constrain x/y translations nor rotation about z.
    # - Using a full 3D pin at one point can create large local stress artifacts.
    # Here we add minimal constraints on CONTACT vertices only:
    #   * Fix u_x=u_y=0 at one contact vertex (removes x/y translation)
    #   * Fix one tangential component at a second contact vertex (removes rotation about z)
    #
    # This keeps the solution well-posed while reducing spurious stress singularities.
    pin_mode_eff = (pin_mode or "top").strip().lower()
    if pin_mode_eff not in {"top", "contact"}:
        raise ValueError("--pin-mode must be one of: top, contact")

    # Also track the global lowest point for debug output (not used for BCs).
    local_min_idx = int(np.argmin(coords[:, 2])) if coords.size else -1
    local_min_z = float(coords[local_min_idx, 2]) if local_min_idx >= 0 else float("inf")
    local_min_pt = coords[local_min_idx].copy() if local_min_idx >= 0 else np.array([np.nan, np.nan, np.nan], dtype=float)
    gathered_min = comm.gather((local_min_z, local_min_pt.tolist()), root=0)
    if comm.rank == 0:
        z_val, pt_val = min(gathered_min, key=lambda t: float(t[0]))
        min_pt = np.array(pt_val, dtype=float)
    else:
        min_pt = None
    min_pt = np.array(comm.bcast(min_pt.tolist() if comm.rank == 0 else None, root=0), dtype=float)

    pin_vertices = top_vertices if pin_mode_eff == "top" else contact_vertices

    def _bcast_best_point_min_key(local_key: float, local_point: np.ndarray) -> np.ndarray:
        gathered = comm.gather((float(local_key), local_point.tolist()), root=0)
        if comm.rank == 0:
            key, pt = min(gathered, key=lambda t: float(t[0]))
            best = np.array(pt, dtype=float) if np.isfinite(key) else np.array([np.nan, np.nan, np.nan], dtype=float)
        else:
            best = None
        return np.array(comm.bcast(best.tolist() if comm.rank == 0 else None, root=0), dtype=float)

    def _bcast_best_point_max_key(local_key: float, local_point: np.ndarray) -> np.ndarray:
        gathered = comm.gather((float(local_key), local_point.tolist()), root=0)
        if comm.rank == 0:
            key, pt = max(gathered, key=lambda t: float(t[0]))
            best = np.array(pt, dtype=float) if np.isfinite(key) else np.array([np.nan, np.nan, np.nan], dtype=float)
        else:
            best = None
        return np.array(comm.bcast(best.tolist() if comm.rank == 0 else None, root=0), dtype=float)

    if comm.size > 1:
        if pin_vertices.size:
            pin_pts = coords[pin_vertices]
            r2 = np.sum(pin_pts[:, :2] ** 2, axis=1)
            i_a = int(np.argmin(r2))
            local_key_a = float(r2[i_a])
            local_a = pin_pts[i_a].copy()
        else:
            local_key_a = float("inf")
            local_a = np.array([np.nan, np.nan, np.nan], dtype=float)

        pinned_pt = _bcast_best_point_min_key(local_key_a, local_a)
        if not np.all(np.isfinite(pinned_pt)):
            raise RuntimeError("Failed to select a global pinned point A (no pin vertices found).")

        if pin_vertices.size:
            pin_pts = coords[pin_vertices]
            d2 = np.sum((pin_pts[:, :2] - pinned_pt[:2]) ** 2, axis=1)
            i_b = int(np.argmax(d2))
            local_key_b = float(d2[i_b])
            b_pt = pin_pts[i_b].copy()
        else:
            local_key_b = float("-inf")
            b_pt = np.array([np.nan, np.nan, np.nan], dtype=float)

        b_pt = _bcast_best_point_max_key(local_key_b, b_pt)
        if not np.all(np.isfinite(b_pt)):
            raise RuntimeError("Failed to select a global pinned point B (no pin vertices found).")

        d = b_pt[:2] - pinned_pt[:2]
        locate_tol = 1e-10

        def near_point(pt: np.ndarray):
            def marker(x):
                return (
                    np.isclose(x[0], pt[0], atol=locate_tol)
                    & np.isclose(x[1], pt[1], atol=locate_tol)
                    & np.isclose(x[2], pt[2], atol=locate_tol)
                )

            return marker

        a_dofs_x = _first_dof_array(fem.locate_dofs_geometrical((V.sub(0), Vx), near_point(pinned_pt)))
        a_dofs_y = _first_dof_array(fem.locate_dofs_geometrical((V.sub(1), Vy), near_point(pinned_pt)))
    else:
        pin_xy = coords[pin_vertices][:, :2]
        # Choose a reference vertex near the symmetry axis if possible (small r^2), to reduce local artifacts.
        r2 = np.sum(pin_xy**2, axis=1)
        a_vertex = int(pin_vertices[int(np.argmin(r2))])
        a_xy = coords[a_vertex, :2]
        d2 = np.sum((pin_xy - a_xy) ** 2, axis=1)
        b_vertex = int(pin_vertices[int(np.argmax(d2))])

        a_dofs_x = _first_dof_array(fem.locate_dofs_topological((V.sub(0), Vx), 0, np.array([a_vertex], dtype=np.int32)))
        a_dofs_y = _first_dof_array(fem.locate_dofs_topological((V.sub(1), Vy), 0, np.array([a_vertex], dtype=np.int32)))

        pinned_pt = coords[a_vertex].copy()
        b_pt = coords[b_vertex].copy()
        d = b_pt[:2] - pinned_pt[:2]

    # If b is mostly along +x/-x from a, rotation about z produces mostly u_y at b (and vice versa).
    b_component = 1 if abs(float(d[0])) >= abs(float(d[1])) else 0
    b_space = Vx if b_component == 0 else Vy
    if comm.size > 1:
        b_dofs = _first_dof_array(fem.locate_dofs_geometrical((V.sub(b_component), b_space), near_point(b_pt)))
    else:
        b_dofs = _first_dof_array(
            fem.locate_dofs_topological((V.sub(b_component), b_space), 0, np.array([b_vertex], dtype=np.int32))
        )

    bc_pin_ax = fem.dirichletbc(PETSc.ScalarType(0.0), a_dofs_x, V.sub(0))
    bc_pin_ay = fem.dirichletbc(PETSc.ScalarType(0.0), a_dofs_y, V.sub(1))
    bc_pin_b = fem.dirichletbc(PETSc.ScalarType(0.0), b_dofs, V.sub(b_component))

    # Optional: also pin one z DOF to remove the rigid-body z-translation mode.
    # This is especially important for penalty contact where contact may be inactive/weak early in Newton.
    bcs = [bc_top, bc_pin_ax, bc_pin_ay, bc_pin_b]
    bc_pin_az = None
    if bool(pin_z):
        if comm.size > 1:
            a_dofs_z = _first_dof_array(fem.locate_dofs_geometrical((V.sub(2), Vz), near_point(pinned_pt)))
        else:
            a_dofs_z = _first_dof_array(
                fem.locate_dofs_topological((V.sub(2), Vz), 0, np.array([a_vertex], dtype=np.int32))
            )
        bc_pin_az = fem.dirichletbc(PETSc.ScalarType(0.0), a_dofs_z, V.sub(2))
        bcs.append(bc_pin_az)

    # Contact model
    contact_model = contact_model.lower().strip()
    if contact_model not in {"penalty", "dirichlet", "neumann"}:
        raise ValueError("--contact-model must be one of: penalty, dirichlet, neumann")
    if loadcase_eff == "cone_rim" and contact_model != "penalty":
        raise ValueError("cone_rim requires --contact-model penalty (Dirichlet/Neumann are plane-only).")
    if contact_model == "neumann" and neumann_pressure_mpa is None and energy_target_j is None:
        raise ValueError("--contact-model neumann requires --neumann-pressure-mpa or --target-work-j/--drop-*.")
    if energy_target_j is not None and contact_model == "dirichlet" and comm.rank == 0:
        print(
            f"[{orientation.name}] warning: target work specified with dirichlet contact; "
            f"consider --contact-model penalty for a more physical drop surrogate.",
            flush=True,
        )

    neumann_pressure_mpa_used: float | None = None

    # Option A: fast approximation (frictionless hard contact on initially-near-ground facets)
    # Enforce u_z = 0 on contact facets.
    if contact_model == "dirichlet":
        contact_dofs_z = _first_dof_array(fem.locate_dofs_topological((V.sub(2), Vz), fdim, contact_facets))
        bc_contact = fem.dirichletbc(PETSc.ScalarType(0.0), contact_dofs_z, V.sub(2))
        bcs = [bc_top, bc_contact, bc_pin_ax, bc_pin_ay, bc_pin_b]
        if bc_pin_az is not None:
            bcs.append(bc_pin_az)
    elif contact_model == "neumann":
        contact_dofs_z = _first_dof_array(fem.locate_dofs_topological((V.sub(2), Vz), fdim, contact_facets))
        bc_contact = fem.dirichletbc(PETSc.ScalarType(0.0), contact_dofs_z, V.sub(2))
        bcs = [bc_contact, bc_pin_ax, bc_pin_ay, bc_pin_b]
        if bc_pin_az is not None:
            bcs.append(bc_pin_az)

    def ppos(s):
        return ufl.conditional(ufl.gt(s, 0.0), s, 0.0)

    if contact_model == "penalty":
        # Option B: unilateral penalty contact on candidate facets against z=0 plane (nonlinear).
        k = fem.Constant(domain, PETSc.ScalarType(contact_penalty))
        # Smooth the positive-part to help Newton convergence near activation.
        #
        # ground: obstacle is plane z=0 (body must satisfy z+u_z >= 0)
        # cone_rim: obstacle is an analytic rigid cone surface. For robustness (and to keep Newton stable),
        # we penalize ONLY the z-penetration to the cone surface (proxy for an impactor), i.e. traction is vertical.
        tip_z: fem.Constant | None = None
        tip_z0: float | None = None
        cone_pen: float | None = None
        tan_alpha: float | None = None
        if loadcase_eff == "ground":
            penetration = _smooth_pos(-(x[2] + u[2]), float(contact_smooth_eps))  # >0 only if below ground
            contact_term = -k * penetration * v[2] * ds(1)
        else:
            hole_r_m = float(config.get("geometry", {}).get("hole_radius_mm", 0.0)) / 1000.0
            alpha = float(cone_half_angle_deg) * np.pi / 180.0
            if not (0.0 < alpha < 0.5 * np.pi):
                raise ValueError("--cone-half-angle-deg must be between 0 and 90")
            tan_alpha = float(np.tan(alpha))
            if not np.isfinite(tan_alpha) or tan_alpha <= 0:
                raise ValueError("--cone-half-angle-deg produced an invalid tan()")
            cone_pen = float(cone_penetration_mm) / 1000.0
            tip_z0 = -hole_r_m / tan_alpha  # so that z_surface(hole_r)=0 at cone_pen=0
            tip_z = fem.Constant(domain, PETSc.ScalarType(tip_z0))

            # Use deformed (current) configuration for gap calculation to maintain Lagrangian consistency.
            # The cone surface z_surface(r) is evaluated at the current radial position (x+u), and
            # the current z-coordinate (x[2]+u[2]) is compared to it. This is physically correct for
            # a rigid obstacle that the deformed body must not penetrate.
            # Note: This adds nonlinearity in u[0], u[1], but improves accuracy under radial bulge.
            r_current = ufl.sqrt((x[0] + u[0]) ** 2 + (x[1] + u[1]) ** 2 + PETSc.ScalarType(1e-18))
            z_surface = tip_z + r_current / PETSc.ScalarType(tan_alpha)
            gap = (x[2] + u[2]) - z_surface
            penetration = _smooth_pos(-gap, float(contact_smooth_eps))
            contact_term = -k * penetration * v[2] * ds(1)

        # ========================================================================
        # BRANCH: Transient dynamics or Quasi-static
        # ========================================================================

        if simulation_mode_eff == "transient":
            # ====================================================================
            # TRANSIENT DYNAMICS PATH
            # ====================================================================
            print(f"[{orientation.name}] Running transient dynamics (Newmark-beta)", flush=True)

            # Call transient dynamics solver
            compression_used, energy_work_j, energy_steps_used, energy_stop_reason, total_newton_iters, iters_info = _run_transient_dynamics(
                domain=domain,
                V=V,
                u=u,
                v=v,
                sigma=sigma,
                eps=eps,
                contact_term=contact_term,
                bcs=[bc_pin_ax, bc_pin_ay, bc_pin_b] + ([bc_pin_az] if bc_pin_az is not None else []),
                top_dofs_z=top_dofs_z,
                drop_mass_kg=drop_mass_kg,
                drop_height_m=drop_height_m,
                time_end_s=time_end_s,
                timestep_s=timestep_s,
                newmark_beta=newmark_beta,
                newmark_gamma=newmark_gamma,
                damping_ratio=damping_ratio,
                rayleigh_alpha=rayleigh_alpha,
                rayleigh_beta=rayleigh_beta,
                contact_penalty=contact_penalty,
                newton_max_it=newton_max_it,
                newton_rtol=newton_rtol,
                newton_atol=newton_atol,
                newton_relax=newton_relax,
                ksp_type=(ksp_type or "cg").strip(),
                pc_type=(pc_type or "gamg").strip(),
                ksp_rtol=ksp_rtol,
                ksp_atol=ksp_atol,
                ksp_max_it=ksp_max_it,
                ksp_monitor=ksp_monitor,
                ksp_monitor_true=ksp_monitor_true,
                ksp_norm_type=ksp_norm_type,
                pc_factor_solver=pc_factor_solver,
                ksp_error_if_not_converged=ksp_error_if_not_converged,
                orientation=orientation,
                comm=comm,
                out_dir=out_dir,
                output_interval=output_interval,
                rho=rho,
                theta_band_f=theta_band_f,
                hemisphere_f=hemisphere_f,
            )

            # Compute reaction force after transient simulation
            reaction_force = _assemble_reaction_force()

        elif simulation_mode_eff == "quasi-static" and contact_model == "penalty":
            # ====================================================================
            # QUASI-STATIC EQUILIBRIUM PATH (existing code)
            # ====================================================================

            # Quasi-static equilibrium: internal + contact = 0 (no inertia; no gravity)
            F = ufl.inner(sigma(u), eps(v)) * ufl.dx + contact_term

            max_steps = max(1, int(max_load_steps))
            ksp_type_eff = (ksp_type or "cg").strip()
            pc_type_eff = (pc_type or "gamg").strip()

            total_newton_iters = 0
            energy_work_j = 0.0
            energy_steps_used = 0
            energy_stop_reason = "max_steps"
            prev_top = 0.0
            prev_fz = 0.0
            current_displacement = 0.0  # Adaptive displacement (starts at 0)
            step = 0

            while current_displacement < max_displacement and step < max_steps:
                step += 1
                current_displacement = min(current_displacement + displacement_step, max_displacement)
                target_top = PETSc.ScalarType(-current_displacement)
                bc_top_step = fem.dirichletbc(target_top, top_dofs_z, V.sub(2))
                # IMPORTANT: For nonlinear solves, DOLFINx expects the current iterate `u`
                # to already satisfy Dirichlet BC values. When we change the prescribed
                # displacement between continuation steps, update the constrained dofs in `u`.
                try:
                    u.sub(2).x.array[top_dofs_z] = target_top
                    u.x.scatter_forward()
                except Exception:
                    pass
                print(
                    f"[{orientation.name}] step {step}/{max_steps}: top_disp={current_displacement*1000:.3f} mm, k={float(contact_penalty):.3e}, smooth_eps={contact_smooth_eps:g}",
                    flush=True,
                )

                # For cone loadcase, ramp the cone penetration via the constant tip_z.
                if loadcase_eff == "cone_rim":
                    if tip_z is None or tip_z0 is None or cone_pen is None or tan_alpha is None:
                        raise RuntimeError("cone_rim internal error: missing cone parameters")
                    # Move the cone up by cone_pen*frac, so the rim penetration increases gradually.
                    tip_z.value = PETSc.ScalarType(tip_z0 + cone_pen * frac)

                # Optional robust initial guess: solve the linearized problem with hard contact (u_z=0)
                # on the candidate contact facets, then start Newton from that state.
                guess_mode = (penalty_initial_guess or "dirichlet").strip().lower()
                if step == 1 and guess_mode != "zero":
                    try:
                        contact_dofs_z = _first_dof_array(
                            fem.locate_dofs_topological((V.sub(2), Vz), fdim, contact_facets_eff)
                        )
                        bc_contact_guess = fem.dirichletbc(PETSc.ScalarType(0.0), contact_dofs_z, V.sub(2))
                        bcs_guess = [bc_top_step, bc_contact_guess, bc_pin_ax, bc_pin_ay, bc_pin_b]

                        u_trial = ufl.TrialFunction(V)
                        a = ufl.inner(sigma(u_trial), eps(v)) * ufl.dx
                        L = ufl.dot(fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0))), v) * ufl.dx

                        t_guess = time.perf_counter()
                        petsc_opts_guess: dict[str, str] = {
                            "ksp_type": str(ksp_type_eff),
                            "pc_type": str(pc_type_eff),
                            "ksp_rtol": str(float(ksp_rtol)),
                            "ksp_atol": str(float(ksp_atol)),
                            "ksp_max_it": str(int(ksp_max_it)),
                        }
                        if pc_factor_solver and pc_type_eff in {"lu", "cholesky"}:
                            petsc_opts_guess["pc_factor_mat_solver_type"] = str(pc_factor_solver)
                            if str(pc_factor_solver).lower() == "mumps":
                                petsc_opts_guess["mat_mumps_icntl_14"] = "80"
                                petsc_opts_guess["mat_mumps_icntl_23"] = "8000"
                        problem_guess = LinearProblem(
                            a,
                            L,
                            bcs=bcs_guess,
                            petsc_options=petsc_opts_guess,
                        )
                        u_guess = problem_guess.solve()
                        u.x.array[:] = u_guess.x.array[:]
                        print(f"[{orientation.name}] initial guess (dirichlet-contact) in {time.perf_counter()-t_guess:.2f}s", flush=True)
                    except Exception as e:
                        if comm.rank == 0:
                            print(f"[{orientation.name}] initial guess skipped (failed): {e}", flush=True)
                            print(traceback.format_exc(), flush=True)

                bcs_step = [bc_top_step, bc_pin_ax, bc_pin_ay, bc_pin_b]
                if bc_pin_az is not None:
                    bcs_step.append(bc_pin_az)
                problem = fem.petsc.NonlinearProblem(F, u, bcs=bcs_step)
                solver = NewtonSolver(comm, problem)
                if hasattr(solver, "report"):
                    solver.report = True
                solver.convergence_criterion = "incremental"
                solver.rtol = float(newton_rtol)
                solver.atol = float(newton_atol)
                solver.max_it = int(newton_max_it)
                if hasattr(solver, "relaxation_parameter"):
                    solver.relaxation_parameter = float(newton_relax)

                ksp = solver.krylov_solver
                opts = PETSc.Options()
                prefix = ksp.getOptionsPrefix()
                opts[f"{prefix}ksp_type"] = ksp_type_eff
                opts[f"{prefix}ksp_rtol"] = float(ksp_rtol)
                opts[f"{prefix}ksp_atol"] = float(ksp_atol)
                opts[f"{prefix}ksp_max_it"] = int(ksp_max_it)
                opts[f"{prefix}pc_type"] = pc_type_eff
                if pc_factor_solver and pc_type_eff in {"lu", "cholesky"}:
                    opts[f"{prefix}pc_factor_mat_solver_type"] = str(pc_factor_solver)
                    if str(pc_factor_solver).lower() == "mumps":
                        opts[f"{prefix}mat_mumps_icntl_14"] = 80
                        opts[f"{prefix}mat_mumps_icntl_23"] = 8000
                norm_type_eff = (ksp_norm_type or "default").strip().lower()
                if norm_type_eff != "default":
                    opts[f"{prefix}ksp_norm_type"] = norm_type_eff
                if ksp_monitor:
                    if ksp_monitor_true:
                        opts[f"{prefix}ksp_monitor_true_residual"] = None
                    else:
                        opts[f"{prefix}ksp_monitor"] = None
                    opts[f"{prefix}ksp_converged_reason"] = None
                if ksp_error_if_not_converged:
                    # Avoid silently "converging" Newton with a failed linear solve (can produce all-zero stress outputs).
                    opts[f"{prefix}ksp_error_if_not_converged"] = 1
                ksp.setFromOptions()

                print(
                    f"[{orientation.name}] Newton: max_it={newton_max_it} rtol={newton_rtol:g} atol={newton_atol:g} relax={newton_relax:g} | "
                    f"KSP: type={ksp_type_eff} pc={pc_type_eff} rtol={ksp_rtol:g} atol={ksp_atol:g} max_it={ksp_max_it} "
                    f"norm={norm_type_eff} monitor={ksp_monitor}{' (true)' if (ksp_monitor and ksp_monitor_true) else ''}"
                    f"{f' factor={pc_factor_solver}' if pc_factor_solver else ''}",
                    flush=True,
                )

                n, converged = solver.solve(u)
                total_newton_iters += int(n)
                if not converged:
                    raise RuntimeError(
                        f"Newton solver did not converge at load step {step}/{max_steps} (iterations={n}). "
                        f"Try: more --max-load-steps, smaller --displacement-step-mm, smaller --contact-penalty, larger --contact-tol-mm, or --pc-type ilu."
                    )

                reaction_force = _assemble_reaction_force()
                fz = float(reaction_force[2])
                delta_u = abs(float(target_top) - prev_top)
                energy_work_j += 0.5 * (abs(prev_fz) + abs(fz)) * delta_u
                prev_top = float(target_top)
                prev_fz = fz
                energy_steps_used = step
                if energy_target_j is not None and comm.rank == 0:
                    print(
                        f"[{orientation.name}] step {step}/{max_steps} energy: "
                        f"work={energy_work_j:.4g} J target={energy_target_j:.4g} J",
                        flush=True,
                    )

                # Sanity check: ensure the imposed top displacement is actually present.
                try:
                    # top_dofs_z are dofs for V.sub(2); check against the subfunction view.
                    uz = u.sub(2)
                    u_top = uz.x.array[top_dofs_z]
                    target = float(target_top)
                    err = float(np.max(np.abs(u_top - target))) if u_top.size else float("nan")
                    umax = float(np.max(np.abs(u.x.array)))
                    print(
                        f"[{orientation.name}] step {step}/{max_steps} checks: max|u|={umax:.3e} m, max|u_top-target|={err:.3e} m",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[{orientation.name}] step {step}/{max_steps} checks skipped: {e}", flush=True)

                if energy_target_j is not None and stop_on_energy and energy_work_j >= energy_target_j:
                    energy_stop_reason = "target_reached"
                    if comm.rank == 0:
                        print(
                            f"[{orientation.name}] energy target reached: work={energy_work_j:.4g} J "
                            f"(target={energy_target_j:.4g} J) at step {step}/{max_steps}",
                            flush=True,
                        )
                    break
            if energy_target_j is None:
                energy_stop_reason = "no_target"
            elif not stop_on_energy:
                energy_stop_reason = "disabled"
            compression_used = abs(prev_top)
            iters_info = f"newton_total={total_newton_iters} (steps={energy_steps_used}/{max_steps})"
    else:
        # Linear elasticity with inhomogeneous Dirichlet BCs (compression + contact).
        u_trial = ufl.TrialFunction(V)
        a = ufl.inner(sigma(u_trial), eps(v)) * ufl.dx
        neumann_pressure_pa = None
        traction_vec = None
        if contact_model == "neumann":
            neumann_pressure_pa = float(neumann_pressure_mpa) * 1e6 if neumann_pressure_mpa is not None else 1.0
            traction_vec = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, -neumann_pressure_pa)))
            L = ufl.dot(traction_vec, v) * ds_top(1)
        else:
            L = ufl.dot(fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0))), v) * ufl.dx

        ksp_type_eff = (ksp_type or "cg").strip()
        pc_type_eff = (pc_type or "gamg").strip()
        petsc_opts: dict[str, str] = {
            "ksp_type": str(ksp_type_eff),
            "pc_type": str(pc_type_eff),
            "ksp_rtol": str(float(ksp_rtol)),
            "ksp_atol": str(float(ksp_atol)),
            "ksp_max_it": str(int(ksp_max_it)),
        }
        norm_type_eff = (ksp_norm_type or "default").strip().lower()
        if norm_type_eff != "default":
            petsc_opts["ksp_norm_type"] = norm_type_eff
        if ksp_monitor:
            petsc_opts["ksp_monitor_true_residual" if ksp_monitor_true else "ksp_monitor"] = "1"
            petsc_opts["ksp_converged_reason"] = "1"
        if ksp_error_if_not_converged:
            petsc_opts["ksp_error_if_not_converged"] = "1"
        if pc_factor_solver and pc_type_eff in {"lu", "cholesky"}:
            petsc_opts["pc_factor_mat_solver_type"] = str(pc_factor_solver)
            # MUMPS memory tuning: increase working space to avoid INFOG(1)=-9 errors.
            if str(pc_factor_solver).lower() == "mumps":
                petsc_opts["mat_mumps_icntl_14"] = "80"  # % increase in estimated workspace (default=20)
                petsc_opts["mat_mumps_icntl_23"] = "8000"  # max memory per MPI rank in MB
        try:
            try:
                if comm.rank == 0:
                    print(
                        f"[{orientation.name}] LinearProblem: building (JIT/assembly may take a while) | "
                        f"KSP: type={ksp_type_eff} pc={pc_type_eff} rtol={ksp_rtol:g} atol={ksp_atol:g} max_it={ksp_max_it} "
                        f"norm={norm_type_eff} monitor={ksp_monitor}{' (true)' if (ksp_monitor and ksp_monitor_true) else ''}"
                        f"{f' factor={pc_factor_solver}' if pc_factor_solver else ''}",
                        flush=True,
                    )
                comm.barrier()
                t_lp = time.perf_counter()
                problem = LinearProblem(a, L, bcs=bcs, petsc_options=petsc_opts)
                if comm.rank == 0:
                    print(f"[{orientation.name}] LinearProblem: built in {time.perf_counter()-t_lp:.2f}s", flush=True)
                comm.barrier()
                if jit_only:
                    if comm.rank == 0:
                        print(f"[{orientation.name}] --jit-only: stopping after LinearProblem construction.", flush=True)
                    return out_dir
                u_sol = problem.solve()
            except (TimeoutError, RuntimeError) as e:
                # Common on shared filesystems: stale libffcx cache file from a failed compile.
                # Try cleaning the offending cache artifacts once, then retry.
                msg = str(e)
                if isinstance(e, RuntimeError) and "jit compilation timed out" not in msg.lower():
                    raise
                removed = []
                if comm.rank == 0:
                    removed = _cleanup_ffcx_cache_from_timeout_message(msg)
                    if removed:
                        print(f"[{orientation.name}] JIT timeout; removed stale cache: {removed[:5]}", flush=True)
                        if len(removed) > 5:
                            print(f"[{orientation.name}] ... and {len(removed)-5} more", flush=True)
                    else:
                        print(f"[{orientation.name}] JIT timeout; no cache file parsed from message.", flush=True)
                comm.barrier()
                t_lp2 = time.perf_counter()
                problem = LinearProblem(a, L, bcs=bcs, petsc_options=petsc_opts)
                if comm.rank == 0:
                    print(f"[{orientation.name}] LinearProblem: rebuilt in {time.perf_counter()-t_lp2:.2f}s", flush=True)
                comm.barrier()
                if jit_only:
                    if comm.rank == 0:
                        print(f"[{orientation.name}] --jit-only: stopping after LinearProblem construction (retry path).", flush=True)
                    return out_dir
                u_sol = problem.solve()
        except Exception as e:
            if comm.rank == 0:
                print(f"[{orientation.name}] LinearProblem failed: {e}", flush=True)
                print(f"[{orientation.name}] petsc_options={petsc_opts}", flush=True)
                print(traceback.format_exc(), flush=True)
            raise
        u.x.array[:] = u_sol.x.array[:]
        if contact_model == "neumann":
            if traction_vec is None or neumann_pressure_pa is None:
                raise RuntimeError("neumann internal error: missing traction definition")
            # Compute elastic strain energy: U = (1/2) × ∫ traction · u dS.
            # The factor of 0.5 is correct for linear elasticity: the strain energy stored
            # equals half the external work done by a suddenly applied constant traction.
            # This is the appropriate quantity to compare against drop impact energy (KE = 0.5 m v²).
            work_form = fem.form(ufl.dot(traction_vec, u) * ds_top(1))
            work_local = fem.assemble_scalar(work_form)
            work_total = comm.allreduce(work_local, op=MPI.SUM)
            energy_work_j = 0.5 * abs(float(work_total))

            if energy_target_j is not None and neumann_pressure_mpa is None:
                if energy_work_j <= 0.0:
                    raise RuntimeError("neumann target work: unit pressure produced zero/negative work")
                scale = float(np.sqrt(energy_target_j / energy_work_j))
                u.x.array[:] *= scale
                neumann_pressure_pa *= scale
                traction_vec.value = PETSc.ScalarType((0.0, 0.0, -neumann_pressure_pa))
                energy_work_j *= scale * scale

            # Mean top displacement (for reporting only).
            uz = u.sub(2)
            u_top_local = uz.x.array[top_dofs_z] if top_dofs_z.size else np.zeros((0,), dtype=float)
            top_sum = float(np.sum(u_top_local)) if u_top_local.size else 0.0
            top_n = int(u_top_local.size)
            top_sum = comm.allreduce(top_sum, op=MPI.SUM)
            top_n = int(comm.allreduce(top_n, op=MPI.SUM))
            top_mean = top_sum / top_n if top_n > 0 else float("nan")
            compression_used = abs(float(top_mean))
            energy_steps_used = 1
            if energy_target_j is None:
                energy_stop_reason = "no_target"
            elif neumann_pressure_mpa is None:
                energy_stop_reason = "target_reached"
            elif stop_on_energy:
                energy_stop_reason = "max_steps"
            else:
                energy_stop_reason = "disabled"
            neumann_pressure_mpa_used = neumann_pressure_pa / 1e6
        else:
            reaction_force = _assemble_reaction_force()
            fz = float(reaction_force[2])
            compression_used = abs(compression)
            energy_work_j = abs(fz) * compression_used
            energy_steps_used = 1
            if energy_target_j is None:
                energy_stop_reason = "no_target"
            elif stop_on_energy and energy_work_j >= energy_target_j:
                energy_stop_reason = "target_reached"
            elif stop_on_energy:
                energy_stop_reason = "max_steps"
            else:
                energy_stop_reason = "disabled"
        if contact_model == "neumann":
            reaction_force = _assemble_reaction_force()
            fz = float(reaction_force[2])
        iters_info = "linear"

    # von Mises (DG0)
    sig = sigma(u)
    s_dev = sig - (1.0 / 3.0) * ufl.tr(sig) * ufl.Identity(3)
    von_mises_expr = ufl.sqrt(3.0 / 2.0 * ufl.inner(s_dev, s_dev))

    vm = fem.Function(DG0, name="von_mises")
    vm.interpolate(fem.Expression(von_mises_expr, DG0.element.interpolation_points()))

    # Reaction force on the top boundary (useful for load/energy calibration).
    reaction_force_mag = float(np.linalg.norm(reaction_force))
    reaction_force_z = float(reaction_force[2])
    if contact_model == "neumann":
        reaction_work_est = float(energy_work_j)
    else:
        reaction_work_est = float(abs(reaction_force_z) * abs(compression_used))

    # Output
    if comm.rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()
    output_format_eff = (output_format or "auto").strip().lower()
    if output_format_eff not in {"auto", "xdmf", "vtx", "vtk"}:
        raise ValueError("--output-format must be one of: auto, xdmf, vtx, vtk")
    if output_format_eff == "auto":
        output_format_eff = "xdmf" if comm.size == 1 else "vtx"

    out_path = out_dir / f"{orientation.name}.xdmf"
    vtx_path = out_dir / f"{orientation.name}.bp"
    vtx_cell_path = out_dir / f"{orientation.name}_cell.bp"
    vtx_masks_path = out_dir / f"{orientation.name}_masks.bp"
    vtk_path = out_dir / f"{orientation.name}.pvd"
    vtk_cell_path = out_dir / f"{orientation.name}_cell.pvd"
    vtk_masks_path = out_dir / f"{orientation.name}_masks.pvd"
    summary_path = out_dir / f"{orientation.name}_summary.yaml"
    metrics_path = out_dir / f"{orientation.name}_metrics.yaml"

    wrote_path: Path
    extra_outputs: dict[str, str] = {}
    if output_format_eff == "xdmf":
        try:
            geometry_xpath = f"/Xdmf/Domain/Grid[@Name='{domain.name}']/Geometry"
            with io.XDMFFile(comm, str(out_path), "w") as xdmf:
                xdmf.write_mesh(domain)
                xdmf.write_meshtags(material_tags, domain.geometry, geometry_xpath=geometry_xpath)
                xdmf.write_meshtags(region_tags, domain.geometry, geometry_xpath=geometry_xpath)
                xdmf.write_meshtags(contact_mt, domain.geometry, geometry_xpath=geometry_xpath)
                xdmf.write_meshtags(top_mt, domain.geometry, geometry_xpath=geometry_xpath)
                xdmf.write_function(u, 0.0)
                xdmf.write_function(vm, 0.0)
                xdmf.write_function(mat_f, 0.0)
                xdmf.write_function(reg_f, 0.0)
                xdmf.write_function(thick_f, 0.0)
                xdmf.write_function(contact_mask, 0.0)
                xdmf.write_function(top_mask, 0.0)
                xdmf.write_function(bc_id, 0.0)
            wrote_path = out_path
        except Exception as e:
            # Common on HPC when HDF5 is built without MPI-IO and multiple ranks try to open the same file.
            if comm.rank == 0:
                print(
                    f"[{orientation.name}] warning: XDMF/HDF5 write failed ({e}). Falling back to VTX/ADIOS2 (.bp).",
                    flush=True,
                )
            output_format_eff = "vtx"
            comm.barrier()
    if output_format_eff == "vtx":
        # Parallel-friendly output (no HDF5 locking). ParaView can open the `.bp` dataset.
        #
        # Note: VTXWriter requires all Functions in a list to share the same element type
        # (family/degree/value shape). Keep separate writers for vector CG1, scalar CG1,
        # and scalar DG0 fields.
        with io.VTXWriter(comm, str(vtx_path), u, engine="BP4") as vtx:
            vtx.write(0.0)
        wrote_path = vtx_path
        try:
            with io.VTXWriter(comm, str(vtx_cell_path), [vm, mat_f, reg_f, thick_f], engine="BP4") as vtx:
                vtx.write(0.0)
            extra_outputs["cell_fields"] = str(vtx_cell_path)
        except Exception as e:
            if comm.rank == 0:
                print(f"[{orientation.name}] warning: failed to write VTX cell fields: {e}", flush=True)
        try:
            with io.VTXWriter(comm, str(vtx_masks_path), [contact_mask, top_mask, bc_id], engine="BP4") as vtx:
                vtx.write(0.0)
            extra_outputs["point_masks"] = str(vtx_masks_path)
        except Exception as e:
            if comm.rank == 0:
                print(f"[{orientation.name}] warning: failed to write VTX point masks: {e}", flush=True)
    if output_format_eff == "vtk":
        # Widely compatible ParaView output (XML VTK). Writes .pvd plus per-rank .vtu pieces.
        # Transient mode: VTK files are already written during time loop
        if comm.rank == 0:
            print(f"[{orientation.name}] VTK output already written during transient simulation", flush=True)
        wrote_path = vtk_cell_path  # Point to time-resolved data
        extra_outputs["cell_fields"] = str(vtk_cell_path)

    if comm.size > 1:
        summary = _summarize_von_mises_by_region_parallel(vm.x.array, mat, reg, comm)
    else:
        summary = _summarize_von_mises_by_region(vm.x.array, mat, reg)

    # Scalar metrics (easy to parse without ParaView)
    # Displacement: use owned dofs only (avoid double-counting ghosts).
    owned_u = u.x.array[: int(V.dofmap.index_map.size_local * V.dofmap.index_map_bs)]
    u3 = owned_u.reshape((-1, 3)) if owned_u.size else np.zeros((0, 3), dtype=float)
    local_max_u_norm = float(np.linalg.norm(u3, axis=1).max()) if u3.size else 0.0
    local_max_u_abs = float(np.abs(owned_u).max()) if owned_u.size else 0.0
    local_min_uz = float(u3[:, 2].min()) if u3.size else float("inf")
    local_max_uz = float(u3[:, 2].max()) if u3.size else -float("inf")
    max_u_norm = float(comm.allreduce(local_max_u_norm, op=MPI.MAX))
    max_u_abs = float(comm.allreduce(local_max_u_abs, op=MPI.MAX))
    min_uz = float(comm.allreduce(local_min_uz, op=MPI.MIN))
    max_uz = float(comm.allreduce(local_max_uz, op=MPI.MAX))
    if not np.isfinite(min_uz):
        min_uz = float("nan")
    if not np.isfinite(max_uz):
        max_uz = float("nan")

    invalid_reason = None
    invalid_reasons = []

    # Displacement cap check for ALL contact modes (not just Neumann).
    # A physically unrealistic deformation indicates solver failure or bad BCs.
    disp_cap_m = float(neumann_max_displacement_mm) / 1000.0
    if max_u_norm > disp_cap_m:
        invalid_reasons.append(f"max_displacement_exceeds_cap_{disp_cap_m:.4f}m")

    # BC enforcement check on top surface (z-component Dirichlet)
    uz = u.sub(2)
    u_top_local = uz.x.array[top_dofs_z] if top_dofs_z.size else np.zeros((0,), dtype=float)
    if contact_model == "neumann":
        top_bc_err_max = float("nan")
    else:
        local_top_err = float(np.max(np.abs(u_top_local - (-compression_used)))) if u_top_local.size else 0.0
        top_bc_err_max = float(comm.allreduce(local_top_err, op=MPI.MAX))

    # von Mises numeric metrics (Pa)
    vm_metrics: dict[str, dict[str, float]] = {}
    if comm.size > 1:
        vm_metrics = _von_mises_metrics_parallel(vm.x.array, mat, reg, comm)
    else:
        overall_max = float(np.max(vm.x.array)) if vm.x.array.size else float("nan")
        overall_mean = float(np.mean(vm.x.array)) if vm.x.array.size else float("nan")
        vm_metrics = {"overall": {"max_pa": overall_max, "mean_pa": overall_mean, "n_cells": float(vm.x.array.size)}}

    # PLA yield stress check: flag if von Mises stress exceeds PLA yield (~50-60 MPa).
    # Linear elastic model is invalid beyond yield; results should be treated with caution.
    # NOTE: This check must come AFTER vm_metrics is computed above.
    PLA_YIELD_STRESS_PA = 55e6  # ~55 MPa typical PLA yield stress
    pla_max_vm = vm_metrics.get("pla_protected", vm_metrics.get("overall", {})).get("max_pa", 0.0)
    if pla_max_vm is None:
        pla_max_vm = 0.0
    # Also check fracture zone if present
    pla_frac_vm = vm_metrics.get("pla_fracture", {}).get("max_pa", 0.0)
    if pla_frac_vm is None:
        pla_frac_vm = 0.0
    # Check for NaN stress values, which indicate an issue, before comparing for yield.
    has_nan_stress = False
    max_stress = -1.0
    if pla_max_vm is not None and np.isfinite(pla_max_vm):
        max_stress = max(max_stress, pla_max_vm)
    # Only flag NaN as an error if the corresponding region has cells.
    elif vm_metrics.get("pla_protected", {}).get("n_cells", 0) > 0:
        has_nan_stress = True

    if pla_frac_vm is not None and np.isfinite(pla_frac_vm):
        max_stress = max(max_stress, pla_frac_vm)
    elif vm_metrics.get("pla_fracture", {}).get("n_cells", 0) > 0:
        has_nan_stress = True

    if has_nan_stress:
        invalid_reasons.append("pla_stress_is_nan_or_inf")
    elif max_stress > PLA_YIELD_STRESS_PA:
        invalid_reasons.append(f"pla_stress_exceeds_yield_{max_stress/1e6:.1f}MPa")

    if invalid_reasons:
        invalid_reason = ";".join(invalid_reasons)

    contact_sum_local = coords[contact_vertices].sum(axis=0) if contact_vertices.size else np.zeros((3,), dtype=float)
    top_sum_local = coords[top_vertices].sum(axis=0) if top_vertices.size else np.zeros((3,), dtype=float)
    contact_n_local = int(contact_vertices.size)
    top_n_local = int(top_vertices.size)
    contact_sum = comm.allreduce(contact_sum_local, op=MPI.SUM)
    top_sum = comm.allreduce(top_sum_local, op=MPI.SUM)
    contact_n = int(comm.allreduce(contact_n_local, op=MPI.SUM))
    top_n = int(comm.allreduce(top_n_local, op=MPI.SUM))
    contact_centroid = contact_sum / contact_n if contact_n > 0 else np.array([np.nan, np.nan, np.nan], dtype=float)
    top_centroid = top_sum / top_n if top_n > 0 else np.array([np.nan, np.nan, np.nan], dtype=float)

    markers_path = out_dir / f"{orientation.name}_bc_markers.vtu"
    if write_bc_markers and comm.rank == 0:
        try:
            # In MPI, avoid writing huge point clouds; write just pins + centroids.
            _write_bc_markers_vtu(
                markers_path,
                coords=coords,
                contact_vertices=contact_vertices if comm.size == 1 else np.zeros((0,), dtype=np.int32),
                top_vertices=top_vertices if comm.size == 1 else np.zeros((0,), dtype=np.int32),
                pinned_a=pinned_pt,
                pinned_b=b_pt,
                contact_centroid=contact_centroid,
                top_centroid=top_centroid,
            )
        except Exception as e:
            print(f"[{orientation.name}] warning: failed to write BC markers VTU: {e}", flush=True)

    if comm.rank == 0:
        print(f"\n[{orientation.name}] solved ({iters_info}, contact={contact_model})")
        print(f"Loadcase: {loadcase_eff}", flush=True)
        if loadcase_eff == "cone_rim":
            print(
                f"Cone: half_angle={float(cone_half_angle_deg):.1f}deg  penetration={float(cone_penetration_mm):.3f}mm  axial_band={float(cone_axial_band_mm):.1f}mm  radial_band={float(cone_radial_band_mm):.1f}mm",
                flush=True,
            )
        print(
            "Selection diagnostics:\n"
            f"- min_z point: [{min_pt[0]:.6g}, {min_pt[1]:.6g}, {min_pt[2]:.6g}] m\n"
            f"- contact vertices: {contact_n} (centroid [{contact_centroid[0]:.6g}, {contact_centroid[1]:.6g}, {contact_centroid[2]:.6g}] m)\n"
            f"- top vertices: {top_n} (centroid [{top_centroid[0]:.6g}, {top_centroid[1]:.6g}, {top_centroid[2]:.6g}] m)\n"
            f"- pinned point A: [{pinned_pt[0]:.6g}, {pinned_pt[1]:.6g}, {pinned_pt[2]:.6g}] m\n"
            f"- pinned point B: [{b_pt[0]:.6g}, {b_pt[1]:.6g}, {b_pt[2]:.6g}] m (fixed component={'y' if b_component==1 else 'x'})"
        )
        print(summary)
        print(f"Wrote: {wrote_path}")
        if extra_outputs:
            print(f"Also wrote: {extra_outputs}", flush=True)

    if energy_target_j is not None and energy_target_j > 0:
        energy_work_ratio = float(energy_work_j / energy_target_j)
    else:
        energy_work_ratio = None

    summary_data = {
        "orientation": {
            "name": orientation.name,
            "mode": orientation.mode,
            "theta_deg": orientation.theta_deg,
            "phi_deg": orientation.phi_deg,
            "rotation_deg_xyz": orientation.rotation_deg_xyz,
            "reference_axis": orientation.reference_axis,
        },
        "bc": {
            "loadcase": loadcase_eff,
            "compression_mm": float(compression_used * 1000.0),
            "contact_model": contact_model,
            "neumann_pressure_mpa": neumann_pressure_mpa_used,
            "clearance_mm": float(clearance_mm),
            "top_tol_mm": float(top_tol_mm),
            "contact_tol_mm": float(contact_tol_mm),
            "top_material": top_material_eff,
            "cone": {
                "half_angle_deg": float(cone_half_angle_deg),
                "penetration_mm": float(cone_penetration_mm),
                "axial_band_mm": float(cone_axial_band_mm),
                "radial_band_mm": float(cone_radial_band_mm),
            }
            if loadcase_eff == "cone_rim"
            else None,
        },
        "energy": {
            "target_j": energy_target_j,
            "work_j": float(energy_work_j),
            "work_ratio": energy_work_ratio,
            "stop_reason": energy_stop_reason,
            "steps_used": int(energy_steps_used),
            "drop_mass_kg": float(drop_mass_kg) if drop_mass_kg is not None else None,
            "drop_height_m": float(drop_height_m) if drop_height_m is not None else None,
        },
        "selection": {
            "min_z_point_m": [float(x) for x in min_pt],
            "pinned_point_a_m": [float(x) for x in pinned_pt],
            "pinned_point_b_m": [float(x) for x in b_pt],
            "pinned_b_component": "y" if b_component == 1 else "x",
            "pin_mode": pin_mode_eff,
            "contact_vertices": int(len(contact_vertices)),
            "contact_centroid_m": [float(x) for x in contact_centroid],
            "top_vertices": int(len(top_vertices)),
            "top_centroid_m": [float(x) for x in top_centroid],
        },
        "von_mises_summary": summary,
        "metrics": {
            "mpi_ranks": int(comm.size),
            "dofs_global": float(ndofs_global),
            "invalid_reason": invalid_reason,
            "displacement": {
                "max_norm_m": max_u_norm,
                "max_abs_component_m": max_u_abs,
                "min_uz_m": min_uz,
                "max_uz_m": max_uz,
                "top_bc_err_max_m": top_bc_err_max,
            },
            "reaction": {
                "force_N": [float(v) for v in reaction_force],
                "force_mag_N": reaction_force_mag,
                "force_z_N": reaction_force_z,
                "work_est_J": reaction_work_est,
            },
            "von_mises": vm_metrics,
        },
        "outputs": {
            "result": str(wrote_path),
            "bc_markers_vtu": str(markers_path) if write_bc_markers else None,
            "extras": extra_outputs,
        },
    }
    if comm.rank == 0:
        summary_path.write_text(yaml.safe_dump(summary_data, sort_keys=False))
        metrics_path.write_text(
            yaml.safe_dump(
                {
                    "orientation": orientation.name,
                    "loadcase": loadcase_eff,
                    "mpi_ranks": int(comm.size),
                    "dofs_global": float(ndofs_global),
                    "neumann_pressure_mpa": neumann_pressure_mpa_used,
                    "max_displacement_norm_m": max_u_norm,
                    "max_displacement_abs_component_m": max_u_abs,
                    "min_uz_m": min_uz,
                    "max_uz_m": max_uz,
                    "top_bc_err_max_m": top_bc_err_max,
                    "energy_target_j": energy_target_j,
                    "energy_work_j": float(energy_work_j),
                    "energy_work_ratio": energy_work_ratio,
                    "energy_stop_reason": energy_stop_reason,
                    "energy_steps_used": int(energy_steps_used),
                    "drop_mass_kg": float(drop_mass_kg) if drop_mass_kg is not None else None,
                    "drop_height_m": float(drop_height_m) if drop_height_m is not None else None,
                    "invalid_reason": invalid_reason,
                    "reaction_force_N": [float(v) for v in reaction_force],
                    "reaction_force_mag_N": reaction_force_mag,
                    "reaction_force_z_N": reaction_force_z,
                    "reaction_work_est_J": reaction_work_est,
                    "von_mises": vm_metrics,
                    "cone": {
                        "half_angle_deg": float(cone_half_angle_deg),
                        "penetration_mm": float(cone_penetration_mm),
                        "axial_band_mm": float(cone_axial_band_mm),
                        "radial_band_mm": float(cone_radial_band_mm),
                    }
                    if loadcase_eff == "cone_rim"
                    else None,
                },
                sort_keys=False,
            )
        )

    return out_path, invalid_reason


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=Path, default=Path("results_fixed/core_shell_with_hole.xdmf"))
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--out", type=Path, default=Path("results_truth_contact"))
    p.add_argument(
        "--loadcase",
        type=str,
        default="ground",
        help="Loadcase: ground (existing compression+ground contact) or cone_rim (rigid cone at hole rim, penalty contact).",
    )
    p.add_argument(
        "--output-format",
        type=str,
        default="auto",
        help="Output format: auto (xdmf on 1 rank, vtx on MPI), xdmf, vtx (.bp), or vtk (.pvd).",
    )
    p.add_argument("--max-displacement-mm", type=float, default=20.0, help="Maximum top displacement (mm) - safety limit")
    p.add_argument("--displacement-step-mm", type=float, default=0.2, help="Displacement increment per load step (mm)")
    p.add_argument(
        "--neumann-pressure-mpa",
        type=float,
        default=None,
        help="Neumann traction magnitude on top surface (MPa); only used when --contact-model neumann.",
    )
    p.add_argument("--target-work-j", type=float, default=None, help="Target work (J) to stop early when reached.")
    p.add_argument("--drop-mass-kg", type=float, default=None, help="Drop mass (kg) for target work m*g*h.")
    p.add_argument("--drop-height-m", type=float, default=None, help="Drop height (m) for target work m*g*h.")
    p.add_argument(
        "--stop-on-energy",
        action="store_true",
        help="Stop load stepping when accumulated work exceeds target (default if target is set).",
    )
    p.add_argument(
        "--no-stop-on-energy",
        action="store_true",
        help="Disable early stop even if a target work is provided.",
    )
    p.add_argument("--cone-half-angle-deg", type=float, default=55.0, help="cone_rim: cone half-angle in degrees.")
    p.add_argument(
        "--cone-penetration-mm",
        type=float,
        default=3.0,
        help="cone_rim: vertical rim penetration (at r=hole_radius) in mm; 0 means just touching at the rim plane.",
    )
    p.add_argument(
        "--cone-axial-band-mm",
        type=float,
        default=3.0,
        help="cone_rim: select hole-rim facets within this axial distance from the impact end.",
    )
    p.add_argument(
        "--cone-radial-band-mm",
        type=float,
        default=2.0,
        help="cone_rim: select hole-rim facets within +/- this radial distance of hole_radius.",
    )
    p.add_argument("--contact-penalty", type=float, default=5e11, help="Penalty stiffness (N/m^3-ish in weak form).")
    p.add_argument(
        "--contact-model",
        type=str,
        default="dirichlet",
        help="Contact/BC model: dirichlet (fast), penalty (nonlinear), or neumann (traction on top).",
    )
    p.add_argument(
        "--neumann-max-displacement-mm",
        type=float,
        default=50.0,
        help="Mark Neumann runs invalid if max displacement exceeds this value (mm).",
    )
    p.add_argument("--clearance-mm", type=float, default=0.05)
    p.add_argument("--top-tol-mm", type=float, default=0.2)
    p.add_argument(
        "--top-material",
        type=str,
        default="pla",
        choices=["all", "pla", "steel"],
        help="Restrict top displacement BC to a material (default: pla).",
    )
    p.add_argument("--contact-tol-mm", type=float, default=0.2)
    p.add_argument(
        "--ksp-monitor",
        action="store_true",
        help="Enable PETSc KSP monitor output (requires iterative KSP/PC).",
    )
    p.add_argument(
        "--ksp-monitor-true",
        action="store_true",
        help="Use `ksp_monitor_true_residual` instead of the default monitor.",
    )
    p.add_argument("--ksp-type", type=str, default=None, help="Override PETSc KSP type (penalty mode).")
    p.add_argument("--pc-type", type=str, default=None, help="Override PETSc PC type (penalty mode).")
    p.add_argument("--ksp-rtol", type=float, default=1e-8, help="PETSc KSP relative tolerance (penalty mode).")
    p.add_argument("--ksp-atol", type=float, default=0.0, help="PETSc KSP absolute tolerance (penalty mode).")
    p.add_argument("--ksp-max-it", type=int, default=2000, help="PETSc KSP max iterations (penalty mode).")
    p.add_argument(
        "--ksp-norm-type",
        type=str,
        default="default",
        help="PETSc KSP norm type: default | unpreconditioned | preconditioned | none.",
    )
    p.add_argument(
        "--pc-factor-solver",
        type=str,
        default=None,
        help="If using `--pc-type lu|cholesky`, set `pc_factor_mat_solver_type` (e.g., mumps, superlu_dist).",
    )
    p.add_argument(
        "--ksp-error-if-not-converged",
        action="store_true",
        help="Raise immediately if the inner linear solve fails (recommended).",
    )
    p.add_argument(
        "--no-ksp-error-if-not-converged",
        action="store_true",
        help="Allow Newton to continue even if a linear solve fails (may produce invalid/zero results).",
    )
    p.add_argument("--newton-max-it", type=int, default=40, help="Newton max iterations (penalty mode).")
    p.add_argument("--newton-rtol", type=float, default=1e-8, help="Newton relative tolerance (penalty mode).")
    p.add_argument("--newton-atol", type=float, default=1e-10, help="Newton absolute tolerance (penalty mode).")
    p.add_argument("--newton-relax", type=float, default=1.0, help="Newton relaxation parameter if supported.")
    p.add_argument("--max-load-steps", type=int, default=100, help="Maximum load steps - will stop early when energy target reached")
    p.add_argument(
        "--contact-smooth-eps",
        type=float,
        default=1e-9,
        help="Smoothing epsilon for contact activation (meters). Increase if Newton struggles.",
    )
    p.add_argument(
        "--penalty-initial-guess",
        type=str,
        default="dirichlet",
        help="penalty mode: initial guess strategy: dirichlet (default) or zero",
    )
    p.add_argument(
        "--pin-mode",
        type=str,
        default="contact",
        help="Rigid-body stabilization reference set: contact (default, most stable) or top (use to avoid polluting contact stresses).",
    )
    p.add_argument(
        "--pin-z",
        action="store_true",
        help="Also pin one z DOF (recommended for penalty contact stability).",
    )
    p.add_argument(
        "--no-pin-z",
        action="store_true",
        help="Disable the extra z pin DOF (not recommended for penalty contact).",
    )
    p.add_argument(
        "--orientations",
        type=str,
        default="default",
        help="default | all | list | comma-separated names",
    )
    p.add_argument(
        "--write-bc-markers",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Write `*_bc_markers.vtu` point cloud for ParaView (shows contact/top selections and pinned points).",
    )
    p.add_argument(
        "--jit-only",
        action="store_true",
        help="Stop after building the linear system (forces UFL/FFCx JIT); useful for warming caches/debugging long startups.",
    )
    return p.parse_args()


def _select_orientations(spec: str) -> Iterable[Orientation]:
    if spec in ("default", "all"):
        return DEFAULT_ORIENTATIONS
    wanted_raw = [s.strip() for s in spec.split(",") if s.strip()]
    wanted = {ORIENTATION_ALIASES.get(s, s) for s in wanted_raw}
    by_name = {o.name: o for o in DEFAULT_ORIENTATIONS}
    missing = sorted(wanted - set(by_name))
    if missing:
        raise SystemExit(f"Unknown orientation(s): {missing}. Available: {sorted(by_name)}")
    return [by_name[n] for n in wanted]


def list_orientations() -> None:
    print("Available orientations:")
    for o in DEFAULT_ORIENTATIONS:
        if o.mode == "impact":
            print(f"- {o.name} (impact theta={o.theta_deg}°, phi={o.phi_deg}°, ref={o.reference_axis})")
        else:
            print(f"- {o.name} (euler {o.rotation_deg_xyz})")


def main() -> None:
    args = _parse_args()
    if args.orientations == "list":
        list_orientations()
        return
    if (args.drop_mass_kg is None) ^ (args.drop_height_m is None):
        raise SystemExit("--drop-mass-kg and --drop-height-m must be set together.")
    target_work_j = args.target_work_j
    if target_work_j is None and args.drop_mass_kg is not None and args.drop_height_m is not None:
        if args.drop_mass_kg <= 0 or args.drop_height_m <= 0:
            raise SystemExit("--drop-mass-kg and --drop-height-m must be positive.")
        target_work_j = float(args.drop_mass_kg) * 9.81 * float(args.drop_height_m)
    stop_on_energy = bool(args.stop_on_energy) or (target_work_j is not None and not bool(args.no_stop_on_energy))
    orientations = list(_select_orientations(args.orientations))

    # Track invalid runs to exit with non-zero code if any fail validation checks.
    invalid_runs: list[tuple[str, str]] = []

    for o in orientations:
        _out_path, invalid_reason = run_orientation(
            mesh_path=args.mesh,
            config_path=args.config,
            orientation=o,
            out_dir=args.out,
            output_format=str(args.output_format),
            loadcase=str(args.loadcase),
            compression_mm=args.compression_mm,
            cone_half_angle_deg=float(args.cone_half_angle_deg),
            cone_penetration_mm=float(args.cone_penetration_mm),
            cone_axial_band_mm=float(args.cone_axial_band_mm),
            cone_radial_band_mm=float(args.cone_radial_band_mm),
            contact_penalty=args.contact_penalty,
            contact_model=args.contact_model,
            clearance_mm=args.clearance_mm,
            top_tol_mm=args.top_tol_mm,
            contact_tol_mm=args.contact_tol_mm,
            top_material=args.top_material,
            neumann_pressure_mpa=args.neumann_pressure_mpa,
            neumann_max_displacement_mm=float(args.neumann_max_displacement_mm),
            ksp_monitor=bool(args.ksp_monitor),
            ksp_type=args.ksp_type,
            pc_type=args.pc_type,
            ksp_rtol=float(args.ksp_rtol),
            ksp_atol=float(args.ksp_atol),
            ksp_max_it=int(args.ksp_max_it),
            ksp_norm_type=str(args.ksp_norm_type),
            ksp_monitor_true=bool(args.ksp_monitor_true),
            pc_factor_solver=args.pc_factor_solver,
            ksp_error_if_not_converged=bool(args.ksp_error_if_not_converged)
            or not bool(args.no_ksp_error_if_not_converged),
            newton_max_it=int(args.newton_max_it),
            newton_rtol=float(args.newton_rtol),
            newton_atol=float(args.newton_atol),
            newton_relax=float(args.newton_relax),
            load_steps=int(args.load_steps),
            contact_smooth_eps=float(args.contact_smooth_eps),
            penalty_initial_guess=str(args.penalty_initial_guess),
            pin_mode=str(args.pin_mode),
            pin_z=bool(args.pin_z) or not bool(args.no_pin_z),
            write_bc_markers=bool(args.write_bc_markers),
            jit_only=bool(args.jit_only),
            target_work_j=target_work_j,
            drop_mass_kg=args.drop_mass_kg,
            drop_height_m=args.drop_height_m,
            stop_on_energy=stop_on_energy,
        )
        if invalid_reason:
            invalid_runs.append((o.name, invalid_reason))
            print(f"[WARNING] {o.name}: invalid_reason={invalid_reason}", flush=True)

    # Report summary and exit with non-zero code if any runs were invalid.
    if invalid_runs:
        print(f"\n[SUMMARY] {len(invalid_runs)}/{len(orientations)} orientation(s) flagged as invalid:", flush=True)
        for name, reason in invalid_runs:
            print(f"  - {name}: {reason}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
