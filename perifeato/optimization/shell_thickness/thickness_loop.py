#!/usr/bin/env python3
"""
Shell Thickness Optimization Loop
==================================

Multi-orientation shell thickness optimization for drop-impact applications.

This loop:
1. Initializes shell thickness parameterization (theta bands)
2. For each iteration:
   - Generates mesh with current thickness parameters
   - Runs multi-orientation FEA (all 5 drop orientations)
   - Updates thickness based on worst-case stress per theta band
   - Checks convergence
3. Saves optimized thickness and convergence history

Key differences from SIMP density optimization:
- Works with thickness parameters (3-5 values) instead of per-element density
- Regenerates mesh geometry each iteration (thickness → geometry)
- Uses theta-band stress distribution for updates
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

# Import adapters from parent src directory
from ..adapters import run_fea, validate_fea_config
from ..adapters.mesh_adapter import load_mesh_for_mvp

# Import thickness optimization components
from .theta_band_params import ThetaBandParams, create_uniform_bands
from .thickness_updater import update_thickness, check_convergence


def run_thickness_optimization(config_file: str) -> Tuple[ThetaBandParams, List[Dict]]:
    """
    Run shell thickness optimization loop.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Tuple of:
        - Optimized thickness parameters
        - Convergence history (list of iteration info dicts)
    """
    print("=" * 70)
    print("SHELL THICKNESS OPTIMIZATION")
    print("Multi-Orientation Drop Impact")
    print("=" * 70)

    # Load configuration
    config = _load_config(config_file)

    # Print configuration summary
    print(f"\nConfiguration: {config.get('version', 'unknown')}")
    print(f"FEA backend: {config.get('fea', {}).get('backend', 'simple')}")
    print(f"Mesh backend: {config.get('mesh', {}).get('backend', 'fallback')}")

    # Validate FEA configuration
    print("\nValidating configuration...")
    fea_errors = validate_fea_config(config)

    if fea_errors:
        print("\n⚠ FEA configuration errors:")
        for err in fea_errors:
            print(f"  - {err}")
        raise ValueError("Invalid FEA configuration. Fix errors above and retry.")

    print("✓ Configuration valid\n")

    # Setup output directory
    output_dir = _resolve_output_dir(config)

    # Initialize thickness parameters
    thickness_params = _initialize_thickness_params(config)
    print(f"Initial thickness: {thickness_params}\n")

    # Optimization settings
    opt_cfg = config.get("optimization", {})
    max_iterations = int(opt_cfg.get("max_iterations", 20))

    history = []

    for iteration in range(max_iterations):
        print(f"\n{'=' * 70}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'=' * 70}")
        print(f"Thickness: {thickness_params}")

        # ====================================================================
        # STEP 1: Generate mesh with current thickness
        # ====================================================================
        print("\n[1/3] Generating mesh...")
        import time
        t_mesh_start = time.time()

        # Update config with current thickness parameters
        config_iter = _update_config_with_thickness(config, thickness_params)

        # Generate mesh (regenerated each iteration!)
        mesh_data = load_mesh_for_mvp(config_iter)
        t_mesh_end = time.time()
        print(f"      Generated mesh: {mesh_data.n_cells} cells")
        print(f"      ⏱ Mesh generation took {t_mesh_end - t_mesh_start:.1f}s")

        # ====================================================================
        # STEP 2: Run multi-orientation FEA
        # ====================================================================
        print("\n[2/3] Running multi-orientation FEA...")
        t_fea_start = time.time()

        # Density is uniform (shell has constant material properties)
        # Thickness variation is encoded in the mesh geometry
        density = np.ones(mesh_data.n_cells, dtype=float)

        fea_results = run_fea(mesh_data, density, config_iter)
        t_fea_end = time.time()

        # Print FEA summary
        max_stress_mpa = fea_results['max_stress'] / 1e6
        worst_orientation = fea_results.get('worst_case_orientation', 'N/A')
        print(f"      Worst-case: {worst_orientation} at {max_stress_mpa:.2f} MPa")
        print(f"      ⏱ FEA took {t_fea_end - t_fea_start:.1f}s")

        # Print per-orientation results if available
        if 'all_orientations' in fea_results:
            print(f"      All orientations:")
            for orient_result in fea_results['all_orientations']:
                orient_name = orient_result['orientation']
                orient_stress_mpa = orient_result['max_stress'] / 1e6
                print(f"        - {orient_name}: {orient_stress_mpa:.2f} MPa")

        # ====================================================================
        # STEP 3: Update thickness based on stress
        # ====================================================================
        print("\n[3/3] Updating thickness...")
        t_update_start = time.time()

        thickness_params, update_info = update_thickness(
            thickness_params,
            fea_results,
            mesh_data,
            config_iter,
        )
        t_update_end = time.time()

        print(f"      Thickness change: {update_info['thickness_change_norm']:.3f} mm")
        print(f"      New thickness: {thickness_params}")
        print(f"      ⏱ Thickness update took {t_update_end - t_update_start:.1f}s")

        # Compute and log volume
        mean_thickness_current = float(np.mean(thickness_params.thickness_mm))
        volume_rel = mean_thickness_current / float(opt_cfg.get("initial_thickness_mm", 1.0))
        print(f"      Mean thickness: {mean_thickness_current:.3f} mm (volume={volume_rel:.2f}x initial)")

        # ====================================================================
        # Check convergence
        # ====================================================================
        converged, reasons = check_convergence(
            thickness_params,
            fea_results,
            update_info,
            history,
            config_iter,
        )

        # Compute relative volume (proportional to mean thickness)
        mean_thickness_mm = float(np.mean(thickness_params.thickness_mm))
        volume_relative = mean_thickness_mm / float(opt_cfg.get("initial_thickness_mm", 1.0))

        # Build history entry
        history_entry = {
            "iteration": iteration + 1,
            "thickness_mm": thickness_params.thickness_mm.tolist(),
            "mean_thickness_mm": mean_thickness_mm,
            "volume_relative": volume_relative,  # Relative to initial uniform thickness
            "max_stress_MPa": max_stress_mpa,
            "worst_case_orientation": worst_orientation,
            "stress_by_band_MPa": update_info.get("stress_by_band_MPa", []),
            "thickness_change_norm": update_info["thickness_change_norm"],
            "converged": converged,
            "convergence_reasons": reasons,
        }

        # Add per-orientation stress if available
        if 'all_orientations' in fea_results:
            history_entry["all_orientations"] = [
                {
                    "orientation": r["orientation"],
                    "max_stress_MPa": r["max_stress"] / 1e6,
                }
                for r in fea_results['all_orientations']
            ]

        history.append(history_entry)

        # Save intermediate results
        _save_results(output_dir, thickness_params, history, iteration)

        if converged:
            print(f"\n✓ Converged after {iteration + 1} iterations!")
            print(f"  Reasons: {', '.join(reasons)}")
            break

    # Save final results
    _save_results(output_dir, thickness_params, history, iteration, final=True)

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Final thickness: {thickness_params}")
    print(f"Final max stress: {history[-1]['max_stress_MPa']:.2f} MPa")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 70}\n")

    return thickness_params, history


def _expand_env_vars(value):
    """Recursively expand ${VAR} and ~ in string leaves of a config tree."""
    import os
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def _load_config(config_file: str) -> Dict:
    """Load configuration from YAML file.

    Expands ``${VAR}`` and ``~`` references in any string value so configs
    can use ``${HOME}/...`` paths portably across users.
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return _expand_env_vars(config)


def _resolve_output_dir(config: Dict) -> Path:
    """Resolve output directory from config."""
    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("dir", "results_thickness_optimization"))

    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _initialize_thickness_params(config: Dict) -> ThetaBandParams:
    """
    Initialize thickness parameterization from config.

    Supports two modes:
    1. Explicit theta bands (if provided in config)
    2. Uniform bands (default: 3 bands, 1.0 mm initial thickness)

    Also initializes fracture factors per band.
    """
    opt_cfg = config.get("optimization", {})
    mat_cfg = config.get("material", {})

    # Check if explicit theta bands provided
    theta_edges = opt_cfg.get("theta_band_edges_deg", None)
    initial_thickness = opt_cfg.get("initial_thickness_mm", None)

    # Thickness bounds
    min_thickness = float(mat_cfg.get("min_thickness_mm", 0.5))
    max_thickness = float(mat_cfg.get("max_thickness_mm", 10.0))

    # Fracture factor parameters
    # Priority: mesh config per-band > optimization scalar > default 0.6
    mesh_cfg = config.get("mesh", {})
    theta_overrides = mesh_cfg.get("theta_overrides", {})
    fracture_design = theta_overrides.get("fracture_design", {})
    fracture_factor_theta = fracture_design.get("fracture_factor_theta", None)

    initial_fracture_factor = opt_cfg.get("initial_fracture_factor", 0.6)
    min_fracture_factor = float(opt_cfg.get("min_fracture_factor", 0.1))
    max_fracture_factor = float(opt_cfg.get("max_fracture_factor", 0.9))

    # B-spline thickness interpolation (smooth outer surface)
    use_spline = bool(opt_cfg.get("use_spline_thickness", False))

    if theta_edges is not None and initial_thickness is not None:
        # Explicit parameterization
        n_bands = len(theta_edges) - 1

        if isinstance(initial_thickness, (int, float)):
            # Uniform initial thickness
            thickness_mm = [float(initial_thickness)] * n_bands
        else:
            # Per-band initial thickness
            thickness_mm = [float(t) for t in initial_thickness]

        # Handle fracture factor: prefer per-band from mesh config
        if fracture_factor_theta is not None and len(fracture_factor_theta) == n_bands:
            fracture_factor = [float(f) for f in fracture_factor_theta]
        elif isinstance(initial_fracture_factor, (int, float)):
            fracture_factor = [float(initial_fracture_factor)] * n_bands
        else:
            fracture_factor = [float(f) for f in initial_fracture_factor]

        return ThetaBandParams(
            theta_edges_deg=theta_edges,
            thickness_mm=thickness_mm,
            min_thickness_mm=min_thickness,
            max_thickness_mm=max_thickness,
            fracture_factor=fracture_factor,
            min_fracture_factor=min_fracture_factor,
            max_fracture_factor=max_fracture_factor,
            use_spline=use_spline,
        )
    else:
        # Default: uniform bands
        n_bands = int(opt_cfg.get("n_theta_bands", 3))
        initial_thickness_val = float(opt_cfg.get("initial_thickness_mm", 1.0))
        initial_fracture_val = float(initial_fracture_factor) if isinstance(initial_fracture_factor, (int, float)) else 0.6

        return create_uniform_bands(
            n_bands=n_bands,
            initial_thickness_mm=initial_thickness_val,
            min_thickness_mm=min_thickness,
            max_thickness_mm=max_thickness,
            initial_fracture_factor=initial_fracture_val,
            min_fracture_factor=min_fracture_factor,
            max_fracture_factor=max_fracture_factor,
            use_spline=use_spline,
        )


def _update_config_with_thickness(config: Dict, thickness_params: ThetaBandParams) -> Dict:
    """
    Update configuration with current thickness parameters.

    CRITICAL: Updates material.protected_thickness_theta_mm which is what
    the theta_mesh generator (ShellGeometryGeneratorRefined) actually reads.

    The theta_mesh generator reads:
    - material.protected_thickness_theta_mm for shell thickness per theta band
    - fracture_design.fracture_factor_theta for fracture zone factor per band
    - mesh.theta_band_edges_deg for theta band boundaries

    These must be at the TOP LEVEL of the config, NOT nested inside theta_overrides.
    """
    import copy
    config_iter = copy.deepcopy(config)

    # Get theta_overrides from thickness params (returns top-level structure)
    overrides = thickness_params.get_theta_overrides()

    # =========================================================================
    # Merge overrides into TOP-LEVEL config keys (not inside theta_overrides!)
    # The theta_mesh generator reads these at the root level.
    # =========================================================================

    # Update material section (for protected_thickness_theta_mm)
    material_cfg = config_iter.setdefault("material", {})
    if "material" in overrides:
        material_cfg.update(overrides["material"])

    # Update fracture_design section (for fracture_factor_theta)
    fracture_cfg = config_iter.setdefault("fracture_design", {})
    if "fracture_design" in overrides:
        fracture_cfg.update(overrides["fracture_design"])

    # Update mesh section (for theta_band_edges_deg)
    mesh_cfg = config_iter.setdefault("mesh", {})
    if "mesh" in overrides:
        mesh_cfg.update(overrides["mesh"])

    # =========================================================================
    # CRITICAL: Also put material/fracture_design inside mesh.theta_overrides
    # because mesh_adapter._generate_theta_mesh() ONLY passes theta_overrides
    # to _write_theta_config(), ignoring top-level config sections.
    # =========================================================================
    theta_overrides = mesh_cfg.setdefault("theta_overrides", {})

    # Put material in theta_overrides (this is what theta_mesh actually reads!)
    if "material" in overrides:
        theta_overrides.setdefault("material", {}).update(overrides["material"])

    # Put fracture_design in theta_overrides
    if "fracture_design" in overrides:
        theta_overrides.setdefault("fracture_design", {}).update(overrides["fracture_design"])

    # Also keep thickness_design for backwards compatibility
    theta_overrides.setdefault("thickness_design", {}).update({
        "theta_edges_deg": thickness_params.theta_edges_deg.tolist(),
        "thickness_mm": thickness_params.thickness_mm.tolist(),
    })

    # =========================================================================
    # HOLE FILLET: Compute fillet size from factor × t0 (pole thickness)
    # This reduces stress concentration at the hole rim.
    # BUT: fillet must not exceed minimum shell thickness (geometry constraint)
    # =========================================================================
    geom_overrides = theta_overrides.setdefault("geometry", {})
    hole_fillet_factor = geom_overrides.get("hole_fillet_factor", 0.0)
    if hole_fillet_factor > 0:
        t0_mm = float(thickness_params.thickness_mm[0])  # Pole thickness (band 0)
        t_min_mm = float(np.min(thickness_params.thickness_mm))  # Minimum shell thickness
        fillet_size_mm = hole_fillet_factor * t0_mm
        # Limit fillet to minimum shell thickness to avoid geometry intersection
        if fillet_size_mm > t_min_mm:
            print(f"[Config] Hole fillet: {fillet_size_mm:.2f}mm exceeds min shell {t_min_mm:.2f}mm, clamping")
            fillet_size_mm = t_min_mm * 0.95  # 95% of min thickness for safety margin
        geom_overrides["hole_fillet_torus_mm"] = fillet_size_mm
        print(f"[Config] Hole fillet: {hole_fillet_factor:.0%} × {t0_mm:.2f}mm = {fillet_size_mm:.2f}mm (min_t={t_min_mm:.2f}mm)")

    return config_iter


def _save_results(
    output_dir: Path,
    thickness_params: ThetaBandParams,
    history: List[Dict],
    iteration: int,
    final: bool = False,
):
    """Save optimization results."""
    # Save thickness parameters
    thickness_file = output_dir / ("thickness_final.json" if final else f"thickness_iter_{iteration}.json")
    with open(thickness_file, "w") as f:
        json.dump(thickness_params.to_dict(), f, indent=2)

    # Save history
    history_file = output_dir / "history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shell Thickness Optimization")
    parser.add_argument("--config", required=True, help="Configuration YAML file")
    args = parser.parse_args()

    run_thickness_optimization(args.config)
