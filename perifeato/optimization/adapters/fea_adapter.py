"""
FEA Adapter - Unified interface for FEA backends
=================================================

Routes FEA calls to:
1. Simple FEA (fast, local MVP testing)
2. Truth Contact FEA (production quality, local or HPC)

Configuration-driven backend selection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict

import numpy as np


def compute_stress_by_theta_band(
    stress_field: np.ndarray,
    cell_centers: np.ndarray,
    theta_edges_deg: np.ndarray,
    c_core: float = 0.034,  # Default 34mm vertical core radius
    theta_band_data: np.ndarray = None,  # Pre-computed theta band from mesh (survives rotation)
    hemisphere_data: np.ndarray = None,  # Pre-computed hemisphere from mesh (1=upper/+z, 0=lower/-z)
) -> Dict[str, np.ndarray]:
    """
    Bin stress field by theta bands, optionally separating by hemisphere.

    If theta_band_data is provided (from mesh cell data), use it directly.
    This is the CORRECT approach because theta_band was assigned during mesh
    generation before any rotation, so it correctly identifies which cells
    belong to which theta band regardless of mesh orientation.

    If hemisphere_data is provided, also compute stress separately for each
    hemisphere (upper=+z, lower=-z in original mesh coordinates).

    Args:
        stress_field: Per-cell stress values (Pa)
        cell_centers: Cell centroid coordinates (n_cells, 3) with z in meters
        theta_edges_deg: Theta band edges in degrees [0, 30, 60, 90]
        c_core: Vertical radius of steel core (meters)
        theta_band_data: Pre-computed theta band indices from mesh cell data
        hemisphere_data: Pre-computed hemisphere (1=upper/+z, 0=lower/-z)

    Returns:
        {
            # Combined (backwards compatible)
            'stress_by_band_MPa': [max_stress_band0, max_stress_band1, ...],
            'stress_p99_by_band_MPa': [...],
            'stress_p95_by_band_MPa': [...],
            'stress_p80_by_band_MPa': [...],
            'stress_mean_by_band_MPa': [...],
            # Hemisphere-specific (if hemisphere_data provided)
            'stress_upper_by_band_MPa': [...],  # Upper hemisphere (+z)
            'stress_lower_by_band_MPa': [...],  # Lower hemisphere (-z)
            'stress_p95_upper_by_band_MPa': [...],
            'stress_p95_lower_by_band_MPa': [...],
        }
    """
    n_bands = len(theta_edges_deg) - 1

    # Initialize output arrays
    max_stress_by_band = np.zeros(n_bands)
    p99_stress_by_band = np.zeros(n_bands)
    p95_stress_by_band = np.zeros(n_bands)
    p80_stress_by_band = np.zeros(n_bands)
    mean_stress_by_band = np.zeros(n_bands)

    # Hemisphere-specific arrays (only populated if hemisphere_data provided)
    max_stress_upper_by_band = np.zeros(n_bands)
    max_stress_lower_by_band = np.zeros(n_bands)
    p95_stress_upper_by_band = np.zeros(n_bands)
    p95_stress_lower_by_band = np.zeros(n_bands)

    if theta_band_data is not None:
        # USE PRE-COMPUTED THETA BAND (correct for rotated meshes)
        band_idx = theta_band_data.astype(int)
        for i in range(n_bands):
            mask = band_idx == i
            if mask.any():
                band_stress = stress_field[mask]
                max_stress_by_band[i] = np.max(band_stress)
                p99_stress_by_band[i] = np.percentile(band_stress, 99)
                p95_stress_by_band[i] = np.percentile(band_stress, 95)
                p80_stress_by_band[i] = np.percentile(band_stress, 80)
                mean_stress_by_band[i] = np.mean(band_stress)

                # Hemisphere-specific stress (if available)
                if hemisphere_data is not None:
                    upper_mask = mask & (hemisphere_data == 1)
                    lower_mask = mask & (hemisphere_data == 0)

                    if upper_mask.any():
                        upper_stress = stress_field[upper_mask]
                        max_stress_upper_by_band[i] = np.max(upper_stress)
                        p95_stress_upper_by_band[i] = np.percentile(upper_stress, 95)

                    if lower_mask.any():
                        lower_stress = stress_field[lower_mask]
                        max_stress_lower_by_band[i] = np.max(lower_stress)
                        p95_stress_lower_by_band[i] = np.percentile(lower_stress, 95)
    else:
        # LEGACY: compute from z-coordinates (only correct for unrotated meshes)
        z_coords = cell_centers[:, 2]

        # Convert theta edges to z-coordinates on ellipsoid
        theta_edges_rad = np.deg2rad(theta_edges_deg)
        z_edges = c_core * np.cos(theta_edges_rad)
        z_edges = np.sort(z_edges)[::-1]

        for i in range(n_bands):
            z_top = z_edges[i]
            z_bottom = z_edges[i + 1]
            mask = ((np.abs(z_coords) <= z_top) & (np.abs(z_coords) >= z_bottom))

            if mask.any():
                band_stress = stress_field[mask]
                max_stress_by_band[i] = np.max(band_stress)
                p99_stress_by_band[i] = np.percentile(band_stress, 99)
                p95_stress_by_band[i] = np.percentile(band_stress, 95)
                p80_stress_by_band[i] = np.percentile(band_stress, 80)
                mean_stress_by_band[i] = np.mean(band_stress)

                # Hemisphere-specific (if available)
                if hemisphere_data is not None:
                    upper_mask = mask & (hemisphere_data == 1)
                    lower_mask = mask & (hemisphere_data == 0)

                    if upper_mask.any():
                        upper_stress = stress_field[upper_mask]
                        max_stress_upper_by_band[i] = np.max(upper_stress)
                        p95_stress_upper_by_band[i] = np.percentile(upper_stress, 95)

                    if lower_mask.any():
                        lower_stress = stress_field[lower_mask]
                        max_stress_lower_by_band[i] = np.max(lower_stress)
                        p95_stress_lower_by_band[i] = np.percentile(lower_stress, 95)

    result = {
        "stress_by_band_MPa": max_stress_by_band / 1e6,
        "stress_p99_by_band_MPa": p99_stress_by_band / 1e6,
        "stress_p95_by_band_MPa": p95_stress_by_band / 1e6,
        "stress_p80_by_band_MPa": p80_stress_by_band / 1e6,
        "stress_mean_by_band_MPa": mean_stress_by_band / 1e6,
    }

    # Add hemisphere-specific results if hemisphere_data was provided
    if hemisphere_data is not None:
        result["stress_upper_by_band_MPa"] = max_stress_upper_by_band / 1e6
        result["stress_lower_by_band_MPa"] = max_stress_lower_by_band / 1e6
        result["stress_p95_upper_by_band_MPa"] = p95_stress_upper_by_band / 1e6
        result["stress_p95_lower_by_band_MPa"] = p95_stress_lower_by_band / 1e6

    return result


def run_fea(mesh_data, density: np.ndarray, config: Dict) -> Dict:
    """
    Unified FEA interface with backend routing.

    Config structure:
        fea:
            backend: "simple" | "truth_contact"
            mode: "local" | "hpc"
            ... backend-specific parameters

    Args:
        mesh_data: MeshData object with mesh and region info
        density: Per-element density field
        config: Full configuration dict

    Returns:
        {
            'max_stress': float (Pa),
            'stress_field': np.ndarray,
            'passed': bool,
            'backend': str,
            'mode': str,
            'orientation': str (if truth_contact),
            'results_path': str (if truth_contact),
        }
    """
    fea_cfg = config.get("fea", {})
    backend = fea_cfg.get("backend", "simple")
    mode = fea_cfg.get("mode", "local")

    if backend == "simple":
        return _run_simple_fea(mesh_data, density, config)
    elif backend == "truth_contact":
        if mode == "local":
            return _run_truth_contact_local(mesh_data, density, config)
        elif mode == "hpc":
            return _run_truth_contact_hpc(mesh_data, density, config)
        else:
            raise ValueError(f"Invalid FEA mode: {mode}. Must be 'local' or 'hpc'.")
    else:
        raise ValueError(f"Invalid FEA backend: {backend}. Must be 'simple' or 'truth_contact'.")


def _run_simple_fea(mesh_data, density: np.ndarray, config: Dict) -> Dict:
    """
    Run simple FEA backend (existing MVP implementation).

    Fast, local-only, good for testing loop logic.
    """
    # Import the simple FEA module from the same directory
    sys.path.insert(0, str(Path(__file__).parent))
    from simple_fea_local import run_simple_fea

    results = run_simple_fea(mesh_data, density, config)

    # Add adapter metadata
    results["backend"] = "simple"
    results["mode"] = "local"

    return results


def _run_truth_contact_local(mesh_data, density: np.ndarray, config: Dict) -> Dict:
    """
    Run truth contact FEA locally (direct Python call).

    Uses modules/fea_truth_contact/src/truth_quasistatic_contact.py
    More accurate than simple FEA but slower.

    Supports both single-orientation and multi-orientation modes:
    - Single: Tests one orientation (config.fea.orientation)
    - Multi: Tests all orientations and returns worst-case (config.fea.use_worst_case_stress)
    """
    fea_cfg = config.get("fea", {})

    # Check for multi-orientation mode
    use_worst_case = fea_cfg.get("use_worst_case_stress", False)
    if use_worst_case:
        return _run_multi_orientation_fea(mesh_data, density, config)

    # Single-orientation mode (existing code)
    # Path to truth FEA module - use HPC path (NO FALLBACK)
    import os
    import time
    t_prep_start = time.time()

    of_root = Path(os.environ.get("OF_ROOT", "/home/mtdsn/Optimization_Framework"))
    truth_fea_src = of_root / "modules" / "fea_truth_contact" / "src"

    if not truth_fea_src.exists():
        raise RuntimeError(
            f"Truth FEA module not found at {truth_fea_src}. "
            f"Ensure OF_ROOT is set correctly or sync code to HPC."
        )

    # Find config relative to src directory
    truth_fea_config = truth_fea_src.parent / "config" / "base_config.yaml"
    if not truth_fea_config.exists():
        raise RuntimeError(f"Truth FEA config not found at {truth_fea_config}")

    # Prepare mesh for truth FEA
    # Truth FEA expects XDMF mesh with material_id and region_id tags
    mesh_path, temp_dir = _prepare_mesh_for_truth_fea(mesh_data, density, config)
    t_prep_end = time.time()
    print(f"[FEA Adapter] ⏱ Mesh preparation took {t_prep_end - t_prep_start:.1f}s")

    try:
        t_simulation_start = time.time()
        # Import truth FEA module
        sys.path.insert(0, str(truth_fea_src))
        from truth_quasistatic_contact import run_orientation, Orientation, DEFAULT_ORIENTATIONS

        # Get orientation
        orientation_name = fea_cfg.get("orientation", "vertical_theta0_base")
        orientation_obj = _find_orientation(orientation_name, DEFAULT_ORIENTATIONS)

        # Output directory
        output_dir = temp_dir / "fea_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run FEA
        print(f"[FEA Adapter] Running truth_contact FEA (orientation: {orientation_name})...")

        # Build parameters for run_orientation
        # Using FIXED defaults from convergence debugging (compression >= 0.1)
        params = {
            "mesh_path": mesh_path,
            "config_path": truth_fea_config,
            "orientation": orientation_obj,
            "out_dir": output_dir,
            "simulation_mode": fea_cfg.get("simulation_mode", "quasi-static"),  # NEW: transient or quasi-static
            "max_displacement_mm": fea_cfg.get("max_displacement_mm", 20.0),  # Safety limit
            "displacement_step_mm": fea_cfg.get("displacement_step_mm", 0.2),  # Adaptive stepping
            "contact_penalty": fea_cfg.get("contact_penalty", 1.0e9),
            "contact_model": fea_cfg.get("contact_model", "penalty"),
            "clearance_mm": fea_cfg.get("clearance_mm", 0.0),
            "top_tol_mm": fea_cfg.get("top_tol_mm", 0.2),
            "contact_tol_mm": fea_cfg.get("contact_tol_mm", 0.2),
            "max_load_steps": fea_cfg.get("max_load_steps", 100),     # Max steps (stops early on energy)
            "newton_max_it": fea_cfg.get("newton_max_it", 40),
            "newton_relax": fea_cfg.get("newton_relax", 1.0),          # ✅ FIXED: was 0.2 (too aggressive)
            # Transient dynamics parameters (NEW)
            "time_end_s": fea_cfg.get("time_end_s", 0.030),
            "timestep_s": fea_cfg.get("timestep_s", 1.0e-6),
            "time_integration": fea_cfg.get("time_integration", "newmark"),
            "newmark_beta": fea_cfg.get("newmark_beta", 0.25),
            "newmark_gamma": fea_cfg.get("newmark_gamma", 0.5),
            "damping_ratio": fea_cfg.get("damping_ratio", 0.03),
            "rayleigh_alpha": fea_cfg.get("rayleigh_alpha", 0.0),
            "rayleigh_beta": fea_cfg.get("rayleigh_beta", 0.0),
            "output_interval": fea_cfg.get("output_interval", 100),
            # Output control
            "output_format": "vtk",  # Simpler format for local use
            "write_bc_markers": False,  # Skip extra output for speed
        }
        params.update(_resolve_energy_params(fea_cfg))

        # Run the orientation
        # Note: run_orientation returns (out_path, invalid_reason) tuple
        out_path, invalid_reason = run_orientation(**params)
        t_simulation_end = time.time()
        print(f"[FEA Adapter] ⏱ Simulation took {t_simulation_end - t_simulation_start:.1f}s")

        # Check for critical invalid results
        # Note: Yield stress exceedance is expected during optimization (we're trying to reduce it!)
        # Only fail on critical issues like displacement cap violations
        if invalid_reason and "displacement_exceeds_cap" in invalid_reason:
            raise RuntimeError(f"Truth FEA returned invalid result: {invalid_reason}")
        elif invalid_reason:
            # Non-critical warning (e.g., yield stress exceedance)
            print(f"[FEA Adapter] Warning: {invalid_reason} (proceeding with results)")

        # Metrics are written to output_dir/{orientation_name}_metrics.yaml
        metrics_path = output_dir / f"{orientation_name}_metrics.yaml"

        # Parse results (gets max_stress but placeholder stress_field)
        t_parse_start = time.time()
        results = _parse_truth_fea_results(metrics_path, fea_cfg)
        t_parse_end = time.time()
        print(f"[FEA Adapter] ⏱ Metrics parsing took {t_parse_end - t_parse_start:.1f}s")

        # Read stress field from VTK output (same as multi-orientation path)
        t_vtk_start = time.time()
        vtk_cell_path = output_dir / f"{orientation_name}_cell.pvd"
        from .fea_vtk_reader import read_stress_field_from_vtk
        stress_field = read_stress_field_from_vtk(vtk_cell_path, mesh_data.n_cells)
        t_vtk_end = time.time()
        print(f"[FEA Adapter] ⏱ VTK reading took {t_vtk_end - t_vtk_start:.1f}s")

        # Fallback if VTK reading failed
        if stress_field is None:
            print(f"[FEA Adapter] Warning: VTK reading failed, using uniform stress fallback")
            stress_field = np.full(mesh_data.n_cells, results["max_stress"])

        # Update results with actual stress field
        results["stress_field"] = stress_field

        # Generate PNG screenshots for visualization (optional, doesn't affect optimization)
        try:
            from .fea_vtk_screenshot import generate_stress_screenshots
            screenshot_dir = output_dir / "screenshots"
            generate_stress_screenshots(vtk_cell_path, screenshot_dir, orientation_name)
        except Exception as e:
            print(f"[FEA Adapter] Screenshot generation failed (non-critical): {e}")

        # Add adapter metadata
        results["backend"] = "truth_contact"
        results["mode"] = "local"
        results["orientation"] = orientation_name
        results["results_path"] = str(output_dir)

        return results

    except Exception as e:
        # NO FALLBACK - fail loudly so errors are visible
        raise RuntimeError(f"Truth FEA failed: {e}") from e


def _run_single_orientation_worker(args):
    """
    Worker function for parallel orientation execution.

    Args:
        args: tuple of (i, orientation, common_params, output_dir, n_cells)

    Returns:
        dict with orientation results or None if failed
    """
    i, orientation, common_params, output_dir, n_cells = args

    try:
        # Import here to avoid issues with multiprocessing
        import yaml
        from .fea_vtk_reader import read_stress_field_from_vtk

        # Get required imports from common_params (passed as dict)
        truth_fea_src = common_params.pop('_truth_fea_src')
        sys.path.insert(0, str(truth_fea_src))
        from truth_quasistatic_contact import run_orientation

        print(f"[Worker {i}] Running {orientation.name}...", flush=True)

        # Run FEA
        out_path, invalid_reason = run_orientation(
            orientation=orientation,
            **common_params
        )

        # Handle errors
        if invalid_reason and "displacement_exceeds_cap" in invalid_reason:
            print(f"[Worker {i}] FAILED: {invalid_reason}")
            return None
        elif invalid_reason:
            print(f"[Worker {i}] Warning: {invalid_reason}")

        # Parse results
        metrics_path = output_dir / f"{orientation.name}_metrics.yaml"
        if not metrics_path.exists():
            print(f"[Worker {i}] FAILED: Metrics not found")
            return None

        with open(metrics_path, 'r') as f:
            metrics = yaml.safe_load(f)

        # Extract stress
        von_mises = metrics.get("von_mises", {})
        overall = von_mises.get("overall", {})
        max_stress_pa = float(overall.get("max_pa", 0.0)) if overall.get("max_pa") is not None else 0.0

        # Read stress field
        vtk_cell_path = output_dir / f"{orientation.name}_cell.pvd"
        stress_field = read_stress_field_from_vtk(vtk_cell_path, n_cells)

        if stress_field is None:
            import numpy as np
            print(f"[Worker {i}] Warning: Using uniform stress fallback")
            stress_field = np.full(n_cells, max_stress_pa)

        print(f"[Worker {i}] Completed {orientation.name}: {max_stress_pa/1e6:.2f} MPa", flush=True)

        return {
            "orientation": orientation.name,
            "max_stress": max_stress_pa,
            "stress_field": stress_field,
            "metrics": metrics,
        }

    except Exception as e:
        print(f"[Worker {i}] FAILED: {e}")
        return None


def _run_multi_orientation_fea(mesh_data, density: np.ndarray, config: Dict) -> Dict:
    """
    Run truth FEA for all orientations and return worst-case results.

    Loops through all 5 default orientations, runs FEA for each,
    and returns the maximum stress across all orientations.

    This ensures the optimization finds a design that survives all drop scenarios.

    Returns:
        Same dict interface as single-orientation FEA, but with additional fields:
        - max_stress: worst-case stress across all orientations
        - worst_case_orientation: name of orientation with max stress
        - all_orientations: list of {orientation, max_stress, stress_field} for each
        - stress_field: per-element stress from worst-case orientation
        - mode: "local_multi" to indicate multi-orientation mode
    """
    fea_cfg = config.get("fea", {})

    # Find truth FEA module - use HPC path (NO FALLBACK)
    import os
    of_root = Path(os.environ.get("OF_ROOT", "/home/mtdsn/Optimization_Framework"))
    truth_fea_src = of_root / "modules" / "fea_truth_contact" / "src"

    if not truth_fea_src.exists():
        raise RuntimeError(
            f"Truth FEA module not found at {truth_fea_src}. "
            f"Ensure OF_ROOT is set correctly or sync code to HPC."
        )

    # Find config
    truth_fea_config = truth_fea_src.parent / "config" / "base_config.yaml"
    if not truth_fea_config.exists():
        raise RuntimeError(f"Truth FEA config not found at {truth_fea_config}")

    # Prepare mesh
    mesh_path, temp_dir = _prepare_mesh_for_truth_fea(mesh_data, density, config)

    try:
        # Import truth FEA module
        sys.path.insert(0, str(truth_fea_src))
        from truth_quasistatic_contact import run_orientation, DEFAULT_ORIENTATIONS
        import yaml

        # Output directory for multi-orientation results
        output_dir = temp_dir / "fea_results_multi"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build common parameters for all orientations
        common_params = {
            "mesh_path": mesh_path,
            "config_path": truth_fea_config,
            "out_dir": output_dir,
            "simulation_mode": fea_cfg.get("simulation_mode", "quasi-static"),  # NEW: transient or quasi-static
            "max_displacement_mm": fea_cfg.get("max_displacement_mm", 20.0),
            "displacement_step_mm": fea_cfg.get("displacement_step_mm", 0.2),
            "contact_penalty": fea_cfg.get("contact_penalty", 1.0e9),
            "contact_model": fea_cfg.get("contact_model", "penalty"),
            "clearance_mm": fea_cfg.get("clearance_mm", 0.0),
            "top_tol_mm": fea_cfg.get("top_tol_mm", 0.2),
            "contact_tol_mm": fea_cfg.get("contact_tol_mm", 0.2),
            "max_load_steps": fea_cfg.get("max_load_steps", 100),
            "newton_max_it": fea_cfg.get("newton_max_it", 40),
            "newton_relax": fea_cfg.get("newton_relax", 1.0),
            # Transient dynamics parameters (NEW)
            "time_end_s": fea_cfg.get("time_end_s", 0.030),
            "timestep_s": fea_cfg.get("timestep_s", 1.0e-6),
            "time_integration": fea_cfg.get("time_integration", "newmark"),
            "newmark_beta": fea_cfg.get("newmark_beta", 0.25),
            "newmark_gamma": fea_cfg.get("newmark_gamma", 0.5),
            "damping_ratio": fea_cfg.get("damping_ratio", 0.03),
            "rayleigh_alpha": fea_cfg.get("rayleigh_alpha", 0.0),
            "rayleigh_beta": fea_cfg.get("rayleigh_beta", 0.0),
            "output_interval": fea_cfg.get("output_interval", 100),
            # Output control
            "output_format": "vtk",
            "write_bc_markers": False,
        }
        common_params.update(_resolve_energy_params(fea_cfg))

        # Check if parallel execution is enabled
        use_parallel = fea_cfg.get("parallel_orientations", False)  # Default to SEQUENTIAL for safety
        n_orientations = len(DEFAULT_ORIENTATIONS)

        # Determine number of workers for parallel execution
        n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4))
        # FEA is single-threaded, so 1 core per orientation
        n_parallel_workers = fea_cfg.get("n_parallel_workers", n_orientations)
        n_workers = min(n_orientations, n_cores, n_parallel_workers)

        print(f"[FEA Adapter] Running multi-orientation FEA ({n_orientations} orientations)...")
        if use_parallel and n_workers > 1:
            print(f"[FEA Adapter] Parallel execution: {n_workers} workers on {n_cores} cores")
        else:
            print(f"[FEA Adapter] Sequential execution")
        print()

        if use_parallel and n_workers > 1:  # Parallel execution enabled
            # PARALLEL EXECUTION
            # Store truth_fea_src in common_params for worker function
            common_params['_truth_fea_src'] = truth_fea_src

            # Prepare arguments for workers
            worker_args = [
                (i, orientation, common_params.copy(), output_dir, mesh_data.n_cells)
                for i, orientation in enumerate(DEFAULT_ORIENTATIONS, 1)
            ]

            # Run in parallel
            with Pool(processes=n_workers) as pool:
                all_results_raw = pool.map(_run_single_orientation_worker, worker_args)

            # Filter out failed orientations (None results)
            all_results = [r for r in all_results_raw if r is not None]

        else:
            # SEQUENTIAL EXECUTION (original code)
            # Don't add _truth_fea_src - not needed for sequential
            all_results = []
            for i, orientation in enumerate(DEFAULT_ORIENTATIONS, 1):
                print(f"[FEA Adapter]   ({i}/{n_orientations}) {orientation.name}...", end=" ", flush=True)

                try:
                    # Run this orientation
                    out_path, invalid_reason = run_orientation(
                        orientation=orientation,
                        **common_params
                    )

                    # Handle invalid_reason
                    if invalid_reason and "displacement_exceeds_cap" in invalid_reason:
                        print(f"FAILED: {invalid_reason}")
                        raise RuntimeError(f"Orientation {orientation.name} returned invalid result: {invalid_reason}")
                    elif invalid_reason:
                        print(f"Warning: {invalid_reason}", end=" ")

                    # Parse metrics
                    metrics_path = output_dir / f"{orientation.name}_metrics.yaml"
                    if not metrics_path.exists():
                        print(f"FAILED: Metrics file not found")
                        raise RuntimeError(f"Metrics file not found for {orientation.name}")

                    with open(metrics_path, 'r') as f:
                        metrics = yaml.safe_load(f)

                    # Extract stress
                    von_mises = metrics.get("von_mises", {})
                    overall = von_mises.get("overall", {})
                    max_stress_pa = float(overall.get("max_pa", 0.0)) if overall.get("max_pa") is not None else 0.0

                    # Read stress field, cell centers, and theta_band from VTK output
                    # Cell centers from VTK are critical - they match the stress field ordering!
                    # theta_band is assigned during mesh generation and survives rotation!
                    # Prefer max_stress file (peak stress across all timesteps) over final-frame stress
                    from .fea_vtk_reader import read_stress_field_from_vtk

                    vtk_max_stress_path = output_dir / f"{orientation.name}_max_stress.pvd"
                    vtk_cell_path = output_dir / f"{orientation.name}_cell.pvd"

                    stress_field = None
                    vtk_cell_centers = None
                    vtk_theta_band = None
                    vtk_hemisphere = None

                    # Try max_stress file first (contains peak stress during transient + theta_band + hemisphere)
                    if vtk_max_stress_path.exists():
                        vtk_result = read_stress_field_from_vtk(
                            vtk_max_stress_path, mesh_data.n_cells,
                            return_cell_centers=True, return_theta_band=True, return_hemisphere=True
                        )
                        stress_field, vtk_cell_centers, vtk_theta_band, vtk_hemisphere = vtk_result
                        if stress_field is not None:
                            has_theta = "with theta_band" if vtk_theta_band is not None else "WITHOUT theta_band"
                            has_hemi = "hemisphere" if vtk_hemisphere is not None else "no hemisphere"
                            print(f"   (using max_stress file {has_theta}, {has_hemi})")

                    # Fall back to regular cell.pvd (final frame only)
                    if stress_field is None and vtk_cell_path.exists():
                        vtk_result = read_stress_field_from_vtk(
                            vtk_cell_path, mesh_data.n_cells,
                            return_cell_centers=True, return_theta_band=True, return_hemisphere=True
                        )
                        stress_field, vtk_cell_centers, vtk_theta_band, vtk_hemisphere = vtk_result

                    # Final fallback if VTK reading failed
                    if stress_field is None:
                        print(f"   (using fallback: uniform stress)")
                        stress_field = np.full(mesh_data.n_cells, max_stress_pa)
                        vtk_cell_centers = None  # Can't use mesh_data.cell_centers - ordering mismatch
                        vtk_theta_band = None
                        vtk_hemisphere = None

                    # Generate PNG screenshots (optional, non-critical)
                    try:
                        from .fea_vtk_screenshot import generate_stress_screenshots
                        screenshot_dir = output_dir / "screenshots"
                        generate_stress_screenshots(vtk_cell_path, screenshot_dir, orientation.name)
                    except Exception:
                        pass  # Silent failure, screenshots are optional

                    # Store result
                    all_results.append({
                        "orientation": orientation.name,
                        "max_stress": max_stress_pa,
                        "stress_field": stress_field,
                        "vtk_cell_centers": vtk_cell_centers,  # For z-binning (matches stress_field ordering)
                        "vtk_theta_band": vtk_theta_band,  # For correct theta-band binning (survives rotation!)
                        "vtk_hemisphere": vtk_hemisphere,  # For hemisphere-specific stress (survives rotation!)
                        "metrics": metrics,
                        "vtk_path": vtk_cell_path,  # For comparison screenshot
                    })

                    print(f"{max_stress_pa/1e6:.2f} MPa")

                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

        if not all_results:
            # Return penalty stress instead of crashing — lets optimizer learn to avoid this region
            print("[FEA Adapter] WARNING: All orientations failed. Returning penalty stress (1 GPa).")
            print("[FEA Adapter] This geometry/thickness may trigger a DOLFINx/PETSc bug.")
            penalty_stress_pa = 1.0e9  # 1 GPa — well above any real stress
            n_bands_fallback = len(config.get("mesh", {}).get("theta_band_edges_deg", [0, 30, 60, 90])) - 1
            return {
                "max_stress": penalty_stress_pa,
                "stress_passed": False,
                "stress_limit_pa": float(fea_cfg.get("stress_limit_Pa", 44.0e6)),
                "worst_case_orientation": "ALL_FAILED",
                "all_orientations": [],
                "stress_field": np.full(mesh_data.n_cells, penalty_stress_pa),
                "stress_by_band_MPa": np.full(n_bands_fallback, penalty_stress_pa / 1e6),
                "stress_by_band_max_MPa": np.full(n_bands_fallback, penalty_stress_pa / 1e6),
                "stress_by_band_p95_MPa": np.full(n_bands_fallback, penalty_stress_pa / 1e6),
                "mode": "local_multi",
                "fea_failed_penalty": True,  # Flag so optimizer knows this was a failure
            }

        # Find worst-case (maximum stress)
        worst = max(all_results, key=lambda r: r["max_stress"])

        print()
        print(f"[FEA Adapter] Worst-case: {worst['orientation']} at {worst['max_stress']/1e6:.2f} MPa")

        # Generate comparison screenshot of all orientations (optional, non-critical)
        try:
            from .fea_vtk_screenshot import generate_multi_orientation_comparison
            vtk_paths = {r["orientation"]: r.get("vtk_path") for r in all_results if "vtk_path" in r}
            screenshot_dir = output_dir / "screenshots"
            generate_multi_orientation_comparison(vtk_paths, screenshot_dir)
        except Exception:
            pass  # Silent failure, screenshots are optional

        # Check stress constraint
        stress_limit_raw = fea_cfg.get("stress_limit_Pa", 44.0e6)
        stress_limit = float(stress_limit_raw) if stress_limit_raw is not None else 44.0e6
        passed = worst["max_stress"] <= stress_limit

        # Compute stress statistics by theta band
        mesh_cfg = config.get("mesh", {}) or {}
        theta_edges_deg = mesh_cfg.get("theta_band_edges_deg")
        if theta_edges_deg is None:
            # Check nested location from theta_overrides (used by theta_mesh module)
            theta_overrides = mesh_cfg.get("theta_overrides", {})
            theta_mesh_cfg = theta_overrides.get("mesh", {})
            theta_edges_deg = theta_mesh_cfg.get("theta_band_edges_deg", [0, 30, 60, 90])
        geom_cfg = config.get("geometry", {}) or {}
        # Also check nested location for geometry config
        if not geom_cfg:
            theta_overrides = mesh_cfg.get("theta_overrides", {})
            geom_cfg = theta_overrides.get("geometry", {})
        c_core_mm = float(geom_cfg.get("inner_radius_z_mm", 34.0))

        # ====================================================================
        # MULTI-ORIENTATION STRESS AGGREGATION
        # ====================================================================
        # Compute stress_by_band for ALL orientations, then aggregate.
        # This prevents oscillation caused by worst-case switching between orientations.
        #
        # Returns two sets of stress values:
        # 1. stress_by_band_MPa: Weighted average for FSD thickness updates
        # 2. stress_by_band_max_MPa: Max across orientations for yield checking
        # ====================================================================

        n_bands = len(theta_edges_deg) - 1
        all_stress_by_band = []  # List of (n_bands,) arrays, one per orientation
        all_stress_p95_by_band = []  # List of P95 stress per band per orientation
        all_stress_p80_by_band = []  # List of P80 stress per band per orientation
        all_max_stress = []  # Max stress per orientation (for weighting)

        # Hemisphere-specific stress arrays
        all_stress_upper_by_band = []  # Upper hemisphere (+z) stress per band per orientation
        all_stress_lower_by_band = []  # Lower hemisphere (-z) stress per band per orientation
        all_stress_p95_upper_by_band = []
        all_stress_p95_lower_by_band = []

        # Energy absorption data
        all_work_j = []  # Work done (energy absorbed) per orientation
        all_energy_absorption_ratio = []  # Ratio of absorbed to input energy per orientation

        print(f"\n[FEA Adapter] Computing stress by band for all {len(all_results)} orientations...")

        for result in all_results:
            orient_name = result["orientation"]
            orient_stress_field = result["stress_field"]
            orient_cell_centers = result.get("vtk_cell_centers")
            orient_theta_band = result.get("vtk_theta_band")  # Pre-computed theta band (survives rotation!)
            orient_hemisphere = result.get("vtk_hemisphere")  # Pre-computed hemisphere (survives rotation!)

            # Determine cell centers and units
            if orient_cell_centers is not None:
                max_z = np.max(np.abs(orient_cell_centers[:, 2]))
                if max_z > 1.0:
                    c_core = c_core_mm  # mm
                else:
                    c_core = c_core_mm / 1000.0  # m
                cell_centers = orient_cell_centers
            else:
                # Fallback (may have ordering issues)
                max_z = np.max(np.abs(mesh_data.cell_centers[:, 2]))
                if max_z > 1.0:
                    c_core = c_core_mm
                else:
                    c_core = c_core_mm / 1000.0
                cell_centers = mesh_data.cell_centers

            # Compute stress by band for this orientation
            # Use theta_band_data and hemisphere_data if available (correct for rotated meshes!)
            orient_stress_by_band = compute_stress_by_theta_band(
                stress_field=orient_stress_field,
                cell_centers=cell_centers,
                theta_edges_deg=np.array(theta_edges_deg),
                c_core=c_core,
                theta_band_data=orient_theta_band,  # Use pre-computed theta band!
                hemisphere_data=orient_hemisphere,  # Use pre-computed hemisphere!
            )

            all_stress_by_band.append(orient_stress_by_band["stress_by_band_MPa"])
            all_max_stress.append(result["max_stress"] / 1e6)  # MPa

            # Also collect P95 and P80 stress (computed in same call)
            all_stress_p95_by_band.append(orient_stress_by_band["stress_p95_by_band_MPa"])
            all_stress_p80_by_band.append(orient_stress_by_band["stress_p80_by_band_MPa"])

            # Collect hemisphere-specific stress if available
            if "stress_upper_by_band_MPa" in orient_stress_by_band:
                all_stress_upper_by_band.append(orient_stress_by_band["stress_upper_by_band_MPa"])
                all_stress_lower_by_band.append(orient_stress_by_band["stress_lower_by_band_MPa"])
                all_stress_p95_upper_by_band.append(orient_stress_by_band["stress_p95_upper_by_band_MPa"])
                all_stress_p95_lower_by_band.append(orient_stress_by_band["stress_p95_lower_by_band_MPa"])
            else:
                # No hemisphere data - use combined as fallback
                all_stress_upper_by_band.append(orient_stress_by_band["stress_by_band_MPa"])
                all_stress_lower_by_band.append(orient_stress_by_band["stress_by_band_MPa"])
                all_stress_p95_upper_by_band.append(orient_stress_by_band["stress_p95_by_band_MPa"])
                all_stress_p95_lower_by_band.append(orient_stress_by_band["stress_p95_by_band_MPa"])

            print(f"  {orient_name}: max={result['max_stress']/1e6:.1f} MPa, "
                  f"bands={[f'{s:.0f}' for s in orient_stress_by_band['stress_by_band_MPa']]}")

            # Collect energy absorption data
            work_j = result.get("work_j")
            energy_ratio = result.get("energy_absorption_ratio")
            if work_j is not None:
                all_work_j.append(work_j)
            if energy_ratio is not None:
                all_energy_absorption_ratio.append(energy_ratio)

        all_stress_p95_by_band = np.array(all_stress_p95_by_band)
        all_stress_p80_by_band = np.array(all_stress_p80_by_band)

        # Convert to arrays for aggregation
        all_stress_by_band = np.array(all_stress_by_band)  # (n_orientations, n_bands)
        all_max_stress = np.array(all_max_stress)  # (n_orientations,)

        # Convert hemisphere-specific arrays
        all_stress_upper_by_band = np.array(all_stress_upper_by_band)  # (n_orientations, n_bands)
        all_stress_lower_by_band = np.array(all_stress_lower_by_band)
        all_stress_p95_upper_by_band = np.array(all_stress_p95_upper_by_band)
        all_stress_p95_lower_by_band = np.array(all_stress_p95_lower_by_band)

        # ====================================================================
        # AGGREGATION METHOD: Stress-weighted average
        # ====================================================================
        # Weight each orientation by its max stress (higher stress = higher weight)
        # This ensures high-stress orientations contribute more to the FSD update,
        # but doesn't let a single worst-case dominate and cause oscillation.
        #
        # Formula: stress_avg[band] = sum(w_i * stress_i[band]) / sum(w_i)
        #          where w_i = max_stress_i^2 (quadratic weighting)
        # ====================================================================

        # Quadratic weights (emphasize high-stress orientations)
        weights = all_max_stress ** 2
        weights = weights / np.sum(weights)  # Normalize

        # Weighted average stress per band
        stress_by_band_weighted = np.zeros(n_bands)
        for band_idx in range(n_bands):
            stress_by_band_weighted[band_idx] = np.sum(weights * all_stress_by_band[:, band_idx])

        # Also compute max across orientations per band (for yield checking)
        stress_by_band_max = np.max(all_stress_by_band, axis=0)

        # Compute P95 aggregations (weighted and max)
        stress_p95_by_band_weighted = np.zeros(n_bands)
        for band_idx in range(n_bands):
            stress_p95_by_band_weighted[band_idx] = np.sum(weights * all_stress_p95_by_band[:, band_idx])
        stress_p95_by_band_max = np.max(all_stress_p95_by_band, axis=0)

        # Compute P80 aggregations (weighted and max) - more aggressive filtering for pole bands
        stress_p80_by_band_weighted = np.zeros(n_bands)
        for band_idx in range(n_bands):
            stress_p80_by_band_weighted[band_idx] = np.sum(weights * all_stress_p80_by_band[:, band_idx])
        stress_p80_by_band_max = np.max(all_stress_p80_by_band, axis=0)

        # ====================================================================
        # HEMISPHERE-SPECIFIC STRESS AGGREGATION
        # ====================================================================
        # Compute max stress per hemisphere per band across all orientations
        # This lets the optimization loop decide how to aggregate (e.g., take max)
        stress_upper_by_band_max = np.max(all_stress_upper_by_band, axis=0)
        stress_lower_by_band_max = np.max(all_stress_lower_by_band, axis=0)
        stress_p95_upper_by_band_max = np.max(all_stress_p95_upper_by_band, axis=0)
        stress_p95_lower_by_band_max = np.max(all_stress_p95_lower_by_band, axis=0)

        print(f"\n[FEA Adapter] Stress aggregation across {len(all_results)} orientations:")
        print(f"  Weights: {[f'{w:.2f}' for w in weights]}")
        print(f"  Weighted avg by band (MPa): {[f'{s:.1f}' for s in stress_by_band_weighted]}")
        print(f"  Max by band (MPa): {[f'{s:.1f}' for s in stress_by_band_max]}")
        print(f"  P95 max (MPa): {[f'{s:.1f}' for s in stress_p95_by_band_max]}")
        print(f"  P80 max (MPa): {[f'{s:.1f}' for s in stress_p80_by_band_max]}")
        print(f"  Upper hemisphere max (MPa): {[f'{s:.1f}' for s in stress_upper_by_band_max]}")
        print(f"  Lower hemisphere max (MPa): {[f'{s:.1f}' for s in stress_lower_by_band_max]}")
        print(f"  Worst-case only (MPa): {[f'{s:.1f}' for s in all_stress_by_band[np.argmax(all_max_stress)]]}")

        # Also compute p99 and mean from worst-case for backwards compatibility
        worst_idx = np.argmax(all_max_stress)
        vtk_cell_centers = worst.get("vtk_cell_centers")
        worst_theta_band = worst.get("vtk_theta_band")  # Pre-computed theta band
        worst_hemisphere = worst.get("vtk_hemisphere")  # Pre-computed hemisphere
        if vtk_cell_centers is not None:
            max_z = np.max(np.abs(vtk_cell_centers[:, 2]))
            if max_z > 1.0:
                c_core = c_core_mm
            else:
                c_core = c_core_mm / 1000.0
            cell_centers_for_binning = vtk_cell_centers
        else:
            max_z = np.max(np.abs(mesh_data.cell_centers[:, 2]))
            if max_z > 1.0:
                c_core = c_core_mm
            else:
                c_core = c_core_mm / 1000.0
            cell_centers_for_binning = mesh_data.cell_centers

        stress_by_band_worst = compute_stress_by_theta_band(
            stress_field=worst["stress_field"],
            cell_centers=cell_centers_for_binning,
            theta_edges_deg=np.array(theta_edges_deg),
            c_core=c_core,
            theta_band_data=worst_theta_band,  # Use pre-computed theta band!
            hemisphere_data=worst_hemisphere,  # Use pre-computed hemisphere!
        )

        # Return same interface as single-orientation, now with stress_field
        return {
            "max_stress": worst["max_stress"],
            "stress_field": worst["stress_field"],  # Now populated from worst-case orientation!
            "passed": passed,
            "backend": "truth_contact",
            "mode": "local_multi",
            "worst_case_orientation": worst["orientation"],
            "all_orientations": all_results,
            "results_path": str(output_dir),
            # ================================================================
            # MULTI-ORIENTATION STRESS AGGREGATION
            # ================================================================
            # stress_by_band_MPa: WEIGHTED AVERAGE across all orientations
            #   → Use for FSD thickness updates (prevents oscillation)
            # stress_by_band_max_MPa: MAX across all orientations per band
            #   → Use for yield constraint checking (conservative)
            # stress_by_band_worst_MPa: From worst-case orientation only
            #   → For backwards compatibility / debugging
            # ================================================================
            "stress_by_band_MPa": stress_by_band_weighted.tolist(),  # Weighted avg for FSD
            "stress_by_band_max_MPa": stress_by_band_max.tolist(),   # Max for yield check
            "stress_by_band_worst_MPa": all_stress_by_band[worst_idx].tolist(),  # Worst-case only
            "stress_p99_by_band_MPa": stress_by_band_worst["stress_p99_by_band_MPa"],
            "stress_p95_by_band_MPa": stress_p95_by_band_weighted.tolist(),  # P95 weighted avg
            "stress_p95_by_band_max_MPa": stress_p95_by_band_max.tolist(),   # P95 max across orientations
            "stress_p80_by_band_MPa": stress_p80_by_band_weighted.tolist(),  # P80 weighted avg
            "stress_p80_by_band_max_MPa": stress_p80_by_band_max.tolist(),   # P80 max across orientations
            "stress_mean_by_band_MPa": stress_by_band_worst["stress_mean_by_band_MPa"],
            # ================================================================
            # HEMISPHERE-SPECIFIC STRESS (NEW)
            # ================================================================
            # Max stress per hemisphere per band across all orientations
            # Lets optimization loop decide how to aggregate
            "stress_upper_by_band_MPa": stress_upper_by_band_max.tolist(),  # Upper hemisphere (+z)
            "stress_lower_by_band_MPa": stress_lower_by_band_max.tolist(),  # Lower hemisphere (-z)
            "stress_p95_upper_by_band_MPa": stress_p95_upper_by_band_max.tolist(),
            "stress_p95_lower_by_band_MPa": stress_p95_lower_by_band_max.tolist(),
            "stress_upper_by_band_max_MPa": stress_upper_by_band_max.tolist(),  # Alias for consistency
            "stress_lower_by_band_max_MPa": stress_lower_by_band_max.tolist(),  # Alias for consistency
            # Store aggregation metadata for debugging
            "stress_aggregation": {
                "method": "quadratic_weighted_average",
                "n_orientations": len(all_results),
                "weights": weights.tolist(),
                "all_max_stress_MPa": all_max_stress.tolist(),
            },
            # ================================================================
            # ENERGY ABSORPTION DATA
            # ================================================================
            # Aggregated energy absorption metrics across all orientations
            # Higher absorption ratio = better protection (more energy dissipated)
            "work_j": np.mean(all_work_j) if all_work_j else None,  # Average work done
            "work_j_min": np.min(all_work_j) if all_work_j else None,  # Min across orientations
            "work_j_max": np.max(all_work_j) if all_work_j else None,  # Max across orientations
            "energy_absorption_ratio": np.mean(all_energy_absorption_ratio) if all_energy_absorption_ratio else None,
            "energy_absorption_ratio_min": np.min(all_energy_absorption_ratio) if all_energy_absorption_ratio else None,
            "input_kinetic_energy_j": all_results[0].get("input_kinetic_energy_j") if all_results else None,
        }

    except Exception as e:
        # NO FALLBACK - fail loudly so errors are visible
        raise RuntimeError(f"Multi-orientation FEA failed: {e}") from e


def _run_truth_contact_hpc(mesh_data, density: np.ndarray, config: Dict) -> Dict:
    """
    Run truth contact FEA on HPC via SLURM submission.

    Submits job, waits for completion, parses results.
    """
    fea_cfg = config.get("fea", {})
    hpc_cfg = fea_cfg.get("hpc", {})

    # Prepare mesh and job files
    mesh_path, temp_dir = _prepare_mesh_for_truth_fea(mesh_data, density, config)

    # Submit SLURM job
    job_id = _submit_truth_fea_job(mesh_path, temp_dir, fea_cfg, hpc_cfg)

    print(f"[FEA Adapter] Submitted truth FEA job: {job_id}")

    # Wait for job completion
    timeout = hpc_cfg.get("timeout", 3600)  # 1 hour default
    wait_for_job(job_id, timeout=timeout)

    # Parse results
    results_dir = temp_dir / "fea_results"
    metrics_path = results_dir / f"{fea_cfg.get('orientation', 'vertical_theta0_base')}_metrics.yaml"

    results = _parse_truth_fea_results(metrics_path, fea_cfg)

    # Add adapter metadata
    results["backend"] = "truth_contact"
    results["mode"] = "hpc"
    results["job_id"] = job_id
    results["orientation"] = fea_cfg.get("orientation", "vertical_theta0_base")
    results["results_path"] = str(results_dir)

    return results


# ============================================================================
# Helper Functions
# ============================================================================


def _prepare_mesh_for_truth_fea(mesh_data, density: np.ndarray, config: Dict) -> tuple[Path, Path]:
    """
    Prepare mesh in format expected by truth FEA.

    Truth FEA expects XDMF with:
    - material_id (1=steel, 2=PLA)
    - region_id (2=protected, 3=fracture)
    - thickness_mm (optional, for visualization)

    For now, we'll use the existing mesh if it's already in XDMF format.
    TODO: Map density field to material properties if needed.
    """
    from pathlib import Path
    import tempfile
    import os
    from datetime import datetime

    # Get output directory from config (matches job output structure)
    output_cfg = config.get("output", {})
    base_output_dir = output_cfg.get("dir", None)

    # Get iteration from config (set by run_dual_physics_fea_only.py)
    iteration = config.get("_current_iteration", None)

    if base_output_dir is not None:
        # Use job-specific output directory with iteration subfolder
        # Structure: {job_output_dir}/fea/iter_{N}/
        fea_base = Path(base_output_dir) / "fea"
        if iteration is not None:
            temp_dir = fea_base / f"iter_{iteration}"
        else:
            # Fallback: use timestamp if no iteration provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = fea_base / f"run_{timestamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[FEA Adapter] Using job output directory: {temp_dir}")
    elif os.environ.get("HOME", "/tmp") != "/tmp":
        # Legacy fallback: use scratch/fea_results (for backwards compatibility)
        scratch_base = os.path.join(os.environ["HOME"], "scratch", "fea_results")
        os.makedirs(scratch_base, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        temp_dir = Path(scratch_base) / f"fea_{job_id}_{timestamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[FEA Adapter] WARNING: Using legacy scratch directory: {temp_dir}")
        print(f"[FEA Adapter] Set config.output.dir to use job-specific output")
    else:
        # Fallback to temp directory for local development
        temp_dir = Path(tempfile.mkdtemp(prefix="truth_fea_"))

    # Check if mesh_data already has XDMF path
    if hasattr(mesh_data, "xdmf_path") and mesh_data.xdmf_path is not None:
        mesh_path = Path(mesh_data.xdmf_path)
        if mesh_path.exists():
            print(f"[FEA Adapter] Using existing mesh: {mesh_path}")
            return mesh_path, temp_dir

    # Otherwise, write mesh to temp directory
    # This requires the mesh to have proper tags
    mesh_path = temp_dir / "mesh_for_fea.xdmf"
    _write_mesh_with_tags(mesh_data, density, mesh_path, config)

    return mesh_path, temp_dir


def _write_mesh_with_tags(mesh_data, density: np.ndarray, output_path: Path, config: Dict) -> None:
    """
    Write mesh with material_id, region_id, and thickness_mm tags.

    For MVP: Use existing region tags from mesh_data.
    TODO: Map density → material properties more sophisticatedly.
    """
    from dolfinx import fem
    from dolfinx.io import XDMFFile
    import basix.ufl

    mesh = mesh_data.mesh
    # DOLFINx 0.9.0 API: Use basix.ufl.element() and fem.functionspace()
    element_v0 = basix.ufl.element("DG", mesh.topology.cell_name(), 0)
    V0 = fem.functionspace(mesh, element_v0)

    # Write mesh only (no cell functions)
    # Truth FEA's meshio-based reader can only handle single-grid XDMF files
    # The truth FEA module will use default material/region IDs from config
    with XDMFFile(mesh.comm, str(output_path), "w") as xdmf:
        xdmf.write_mesh(mesh)

    print(f"[FEA Adapter] Wrote mesh to {output_path} (without tags - truth FEA will use defaults)")


def _find_orientation(name: str, orientations: list) -> Any:
    """Find orientation object by name."""
    for orient in orientations:
        if orient.name == name:
            return orient
    raise ValueError(
        f"Orientation '{name}' not found. Available: {[o.name for o in orientations]}"
    )


def _parse_truth_fea_results(metrics_path: Path, fea_cfg: Dict) -> Dict:
    """
    Parse truth FEA results from metrics YAML file.

    Returns dict compatible with MVP loop interface.
    """
    import yaml

    if not metrics_path.exists():
        raise RuntimeError(f"Truth FEA metrics file not found: {metrics_path}")

    with open(metrics_path, "r") as f:
        metrics = yaml.safe_load(f)

    # Extract stress information
    # Truth FEA outputs overall von Mises metrics
    von_mises = metrics.get("von_mises", {})
    overall = von_mises.get("overall", {})
    max_stress_pa = float(overall.get("max_pa", 0.0)) if overall.get("max_pa") is not None else 0.0

    # Check against stress limit (ensure it's a float)
    stress_limit_raw = fea_cfg.get("stress_limit_Pa", 44.0e6)
    stress_limit = float(stress_limit_raw) if stress_limit_raw is not None else 44.0e6
    passed = max_stress_pa <= stress_limit

    # For stress_field, we don't have per-element values from metrics file
    # Would need to read from VTK/XDMF output if needed
    # For now, return empty array (MVP loop doesn't use it currently)
    stress_field = np.array([max_stress_pa])  # Placeholder

    # Parse energy absorption from summary.yaml (sibling to metrics.yaml)
    # Energy absorption ratio = work_j / input_kinetic_energy
    work_j = None
    energy_absorption_ratio = None
    input_kinetic_energy_j = None

    summary_path = metrics_path.parent / metrics_path.name.replace("_metrics.yaml", "_summary.yaml")
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                summary = yaml.safe_load(f)
            energy_data = summary.get("energy", {})
            work_j = energy_data.get("work_j")
            drop_mass_kg = energy_data.get("drop_mass_kg")
            drop_height_m = energy_data.get("drop_height_m")

            # Compute input kinetic energy: KE = m * g * h
            if drop_mass_kg is not None and drop_height_m is not None:
                input_kinetic_energy_j = float(drop_mass_kg) * 9.81 * float(drop_height_m)

            # Compute energy absorption ratio
            if work_j is not None and input_kinetic_energy_j is not None and input_kinetic_energy_j > 0:
                energy_absorption_ratio = float(work_j) / input_kinetic_energy_j
        except Exception as e:
            print(f"[FEA Adapter] Warning: Failed to parse energy data from {summary_path}: {e}")

    return {
        "max_stress": max_stress_pa,
        "stress_field": stress_field,
        "passed": passed,
        "metrics": metrics,  # Include full metrics for debugging
        # Energy absorption data
        "work_j": work_j,
        "input_kinetic_energy_j": input_kinetic_energy_j,
        "energy_absorption_ratio": energy_absorption_ratio,
    }


def _resolve_energy_params(fea_cfg: Dict) -> Dict:
    """Resolve energy-based loading parameters from config."""
    target_work_j = fea_cfg.get("target_work_j")
    drop_mass_kg = fea_cfg.get("drop_mass_kg")
    drop_height_m = fea_cfg.get("drop_height_m")

    if (drop_mass_kg is None) ^ (drop_height_m is None):
        raise ValueError("drop_mass_kg and drop_height_m must be set together.")

    if target_work_j is None and drop_mass_kg is not None and drop_height_m is not None:
        if drop_mass_kg <= 0 or drop_height_m <= 0:
            raise ValueError("drop_mass_kg and drop_height_m must be positive.")
        target_work_j = float(drop_mass_kg) * 9.81 * float(drop_height_m)

    stop_on_energy = fea_cfg.get("stop_on_energy")
    if stop_on_energy is None:
        stop_on_energy = target_work_j is not None

    return {
        "target_work_j": target_work_j,
        "drop_mass_kg": drop_mass_kg,
        "drop_height_m": drop_height_m,
        "stop_on_energy": stop_on_energy,
    }


def _submit_truth_fea_job(
    mesh_path: Path, temp_dir: Path, fea_cfg: Dict, hpc_cfg: Dict
) -> str:
    """
    Submit truth FEA job to SLURM on HPC.

    Returns: job_id (str)
    """
    project_root = Path(__file__).resolve().parents[4]
    submit_script = project_root / "modules" / "fea_truth_contact" / "hpc" / "submit_truth_jobs.sh"

    if not submit_script.exists():
        raise RuntimeError(f"Submit script not found: {submit_script}")

    # Build sbatch command
    orientation = fea_cfg.get("orientation", "vertical_theta0_base")
    partition = hpc_cfg.get("partition", "shared")
    ranks = hpc_cfg.get("ranks", 8)

    cmd = [
        "bash",
        str(submit_script),
        orientation,
        "--partition", partition,
        "--mesh", str(mesh_path),
        "--ranks", str(ranks),
        "--output-format", "vtk",
    ]

    # Add optional parameters
    if "compression_mm" in fea_cfg:
        cmd.extend(["--compression-mm", str(fea_cfg["compression_mm"])])
    if "contact_penalty" in fea_cfg:
        cmd.extend(["--contact-penalty", str(fea_cfg["contact_penalty"])])
    if "target_work_j" in fea_cfg and fea_cfg["target_work_j"] is not None:
        cmd.extend(["--target-work-j", str(fea_cfg["target_work_j"])])
    if "drop_mass_kg" in fea_cfg and fea_cfg["drop_mass_kg"] is not None:
        cmd.extend(["--drop-mass-kg", str(fea_cfg["drop_mass_kg"])])
    if "drop_height_m" in fea_cfg and fea_cfg["drop_height_m"] is not None:
        cmd.extend(["--drop-height-m", str(fea_cfg["drop_height_m"])])
    if "stop_on_energy" in fea_cfg:
        if fea_cfg["stop_on_energy"]:
            cmd.append("--stop-on-energy")
        else:
            cmd.append("--no-stop-on-energy")

    # Submit job
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Parse job ID from output
    # Typical sbatch output: "Submitted batch job 12345"
    import re
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Failed to parse job ID from: {result.stdout}")

    job_id = match.group(1)
    return job_id


def wait_for_job(job_id: str, timeout: int = 3600, poll_interval: int = 10) -> None:
    """
    Wait for SLURM job to complete.

    Args:
        job_id: SLURM job ID
        timeout: Maximum wait time (seconds)
        poll_interval: Check interval (seconds)

    Raises:
        TimeoutError: If job doesn't complete within timeout
        RuntimeError: If job fails
    """
    start_time = time.time()

    while True:
        # Check job status with squeue
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0 or not result.stdout.strip():
            # Job no longer in queue - check if it completed successfully
            # Use sacct to get job state
            sacct_result = subprocess.run(
                ["sacct", "-j", job_id, "-n", "-o", "State"],
                capture_output=True,
                text=True,
            )

            if "COMPLETED" in sacct_result.stdout:
                print(f"[FEA Adapter] Job {job_id} completed successfully")
                return
            elif "FAILED" in sacct_result.stdout or "CANCELLED" in sacct_result.stdout:
                raise RuntimeError(f"Job {job_id} failed. Check logs.")
            else:
                # Job finished but status unclear
                print(f"[FEA Adapter] Job {job_id} finished (status unclear)")
                return

        # Job still running
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

        print(f"[FEA Adapter] Job {job_id} still running ({elapsed:.0f}s elapsed)...")
        time.sleep(poll_interval)


# ============================================================================
# Configuration Validation
# ============================================================================


def validate_fea_config(config: Dict) -> list[str]:
    """
    Validate FEA configuration.

    Returns list of error messages (empty if valid).
    """
    errors = []
    fea_cfg = config.get("fea", {})

    # Check backend (default to "simple" for backward compatibility)
    backend = fea_cfg.get("backend", "simple")
    if backend not in ["simple", "truth_contact"]:
        errors.append(f"Invalid fea.backend: {backend}. Must be 'simple' or 'truth_contact'.")

    # Check mode
    mode = fea_cfg.get("mode", "local")
    if mode not in ["local", "hpc"]:
        errors.append(f"Invalid fea.mode: {mode}. Must be 'local' or 'hpc'.")

    # Check backend-specific requirements
    if backend == "truth_contact":
        # Orientation required (unless using multi-orientation mode)
        use_worst_case = fea_cfg.get("use_worst_case_stress", False)
        if not use_worst_case and "orientation" not in fea_cfg:
            errors.append("fea.orientation required for truth_contact backend (or set use_worst_case_stress: true)")

        # HPC mode requires HPC config
        if mode == "hpc" and "hpc" not in fea_cfg:
            errors.append("fea.hpc configuration required for HPC mode")

    return errors
