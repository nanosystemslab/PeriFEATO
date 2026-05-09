"""
Simple FEA Module for Local MVP
================================

Minimal linear elasticity solve with SIMP stiffness scaling.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _get_mesh(mesh_input: Any):
    return mesh_input.mesh if hasattr(mesh_input, "mesh") else mesh_input


def _normalize_density(density: np.ndarray, target_size: int) -> np.ndarray:
    density = np.asarray(density, dtype=float)
    if density.size == 1:
        return np.full(target_size, float(density), dtype=float)
    if density.size != target_size:
        raise ValueError(f"Density size mismatch: got {density.size}, expected {target_size}")
    return density


def run_simple_fea(mesh, density, config: Dict) -> Dict:
    """
    Simplified FEA stress check using FEniCSx.

    Returns:
        dict with 'max_stress', 'stress_field', 'passed'
    """
    from dolfinx import fem, mesh as dmesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc
    import ufl
    import basix.ufl

    mesh = _get_mesh(mesh)
    fea_cfg = config.get("fea", {}) if isinstance(config, dict) else {}
    mat_cfg = config.get("material", {}) if isinstance(config, dict) else {}

    E_max = float(fea_cfg.get("E_max", mat_cfg.get("E", 3.5e9)))
    nu = float(fea_cfg.get("nu", mat_cfg.get("nu", 0.36)))
    penalty = float(fea_cfg.get("simp_penalty", 3.0))
    E_min = float(fea_cfg.get("E_min", 1e-3 * E_max))
    load_magnitude = float(fea_cfg.get("load_magnitude_N", 1000.0))
    stress_limit = float(fea_cfg.get("stress_limit_Pa", 44e6))

    tdim = mesh.topology.dim

    # DOLFINx 0.9.0 API: Use basix.ufl elements and fem.functionspace()
    element_v = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
    V = fem.functionspace(mesh, element_v)
    element_v0 = basix.ufl.element("DG", mesh.topology.cell_name(), 0)
    V0 = fem.functionspace(mesh, element_v0)

    num_cells = V0.dofmap.index_map.size_local
    density_local = _normalize_density(density, num_cells)

    rho = fem.Function(V0)
    rho.x.array[:] = density_local

    E = E_min + (E_max - E_min) * rho**penalty
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(len(u))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(sigma(u), eps(v)) * ufl.dx

    coords = mesh.geometry.x
    z_min = coords[:, 2].min() if mesh.geometry.dim >= 3 else coords[:, 1].min()
    z_max = coords[:, 2].max() if mesh.geometry.dim >= 3 else coords[:, 1].max()

    fdim = tdim - 1

    def fixed_boundary(x):
        axis = 2 if mesh.geometry.dim >= 3 else 1
        return np.isclose(x[axis], z_max)

    fixed_facets = dmesh.locate_entities_boundary(mesh, fdim, fixed_boundary)
    fixed_dofs = fem.locate_dofs_topological(V, fdim, fixed_facets)
    u0 = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
    bc = fem.dirichletbc(u0, fixed_dofs, V)

    def load_boundary(x):
        axis = 2 if mesh.geometry.dim >= 3 else 1
        return np.isclose(x[axis], z_min)

    load_facets = dmesh.locate_entities_boundary(mesh, fdim, load_boundary)
    if len(load_facets) > 0:
        tag = np.full(len(load_facets), 1, dtype=np.int32)
        mt = dmesh.meshtags(mesh, fdim, load_facets, tag)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)
        area = fem.assemble_scalar(fem.form(1.0 * ds(1)))
        area = area if area > 0 else 1.0

        if mesh.geometry.dim >= 3:
            traction_vec = np.array((0.0, 0.0, -load_magnitude / area), dtype=PETSc.ScalarType)
        else:
            traction_vec = np.array((0.0, -load_magnitude / area), dtype=PETSc.ScalarType)
        traction = fem.Constant(mesh, traction_vec)
        L = ufl.dot(traction, v) * ds(1)
    else:
        body_force = (0.0, 0.0, -load_magnitude) if mesh.geometry.dim >= 3 else (0.0, -load_magnitude)
        f = fem.Constant(mesh, np.array(body_force, dtype=PETSc.ScalarType))
        L = ufl.dot(f, v) * ufl.dx

    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    sigma_u = sigma(uh)
    deviator = sigma_u - ufl.tr(sigma_u) * ufl.Identity(len(uh)) / 3.0
    von_mises = ufl.sqrt(1.5 * ufl.inner(deviator, deviator))

    stress = fem.Function(V0)
    stress_expr = fem.Expression(von_mises, V0.element.interpolation_points())
    stress.interpolate(stress_expr)

    stress_field = stress.x.array.copy()
    max_stress = float(np.max(stress_field)) if stress_field.size else 0.0

    return {
        "max_stress": max_stress,
        "stress_field": stress_field,
        "passed": max_stress <= stress_limit,
    }


if __name__ == "__main__":
    print("Testing simple_fea_local module...")
    print("Run via main_loop.py for a full integration test.")
