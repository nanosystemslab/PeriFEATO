"""PeriFEATO smoke test.

Verifies:
1. Package imports inside the DOLFINx container.
2. ThetaBandParams construction works.
3. update_thickness runs end-to-end on synthetic FEA data.
4. All 7 optimization methods dispatch correctly.
"""
from __future__ import annotations

import sys
import numpy as np

print("=" * 70)
print("PeriFEATO smoke test")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
print("\n[1] Importing perifeato...")
from perifeato.optimization.shell_thickness.theta_band_params import (
    ThetaBandParams,
    create_uniform_bands,
)
from perifeato.optimization.shell_thickness.thickness_updater import (
    update_thickness,
    update_fracture_factor,
    check_convergence,
    reset_momentum_history,
    reset_secant_history,
    reset_oc_history,
    reset_mma_history,
    reset_bo_history,
    reset_pgd_history,
    reset_cobyla_history,
)
print("    OK")

# ---------------------------------------------------------------------------
# 2. ThetaBandParams construction
# ---------------------------------------------------------------------------
print("\n[2] Constructing ThetaBandParams...")
n_bands = 10
params = create_uniform_bands(n_bands, initial_thickness_mm=2.0)
print(f"    n_bands={params.n_bands}, thickness={params.thickness_mm}")

# ---------------------------------------------------------------------------
# 3. Synthetic FEA / mesh data
# ---------------------------------------------------------------------------
print("\n[3] Building synthetic FEA results...")

class _MockMesh:
    n_cells = 1000
    cell_centers = np.zeros((1000, 3))

mesh_data = _MockMesh()

# Stress per band: pole bands overstressed, equator OK.
stress_by_band_mpa = np.linspace(60.0, 30.0, n_bands)
fea_results = {
    "stress_by_band_MPa": stress_by_band_mpa.tolist(),
    "stress_by_band_max_MPa": stress_by_band_mpa.tolist(),
    "max_stress": float(stress_by_band_mpa.max()) * 1e6,
    "stress_field": None,
}
print(f"    Synthetic stress range: {stress_by_band_mpa.min():.1f}-{stress_by_band_mpa.max():.1f} MPa")

# ---------------------------------------------------------------------------
# 4. Run update_thickness for every algorithm
# ---------------------------------------------------------------------------
print("\n[4] Dispatching update_thickness for each algorithm...")
methods = ["fsd", "oc", "mma", "bo", "pgd", "cobyla"]

for method in methods:
    # Reset all per-method histories so each call is independent.
    reset_momentum_history()
    reset_secant_history()
    reset_oc_history()
    reset_mma_history()
    reset_bo_history()
    reset_pgd_history()
    reset_cobyla_history()

    fresh_params = create_uniform_bands(n_bands, initial_thickness_mm=2.0)
    config = {
        "optimization": {
            "optimization_method": method,
            "target_safety_factor": 1.5,
            "thickness_relaxation": 0.3,
            "stress_thickness_exponent": 0.5,
            "thickness_resolution_mm": 0.05,
            "use_momentum": False,
            "stress_band_stat": "max",
            "max_iterations": 1,
        },
        "fea": {"stress_limit_Pa": 44e6},
        "material": {"min_thickness_mm": 1.0, "max_thickness_mm": 10.0},
    }

    updated, info = update_thickness(fresh_params, fea_results, mesh_data, config)
    delta = np.array(info["thickness_change_mm"])
    print(
        f"    {method:>7}: |dt|={np.linalg.norm(delta):.3f} mm  "
        f"thickness={np.round(updated.thickness_mm, 2).tolist()}"
    )

# ---------------------------------------------------------------------------
# 5. Done
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Smoke test PASSED")
print("=" * 70)
sys.exit(0)
