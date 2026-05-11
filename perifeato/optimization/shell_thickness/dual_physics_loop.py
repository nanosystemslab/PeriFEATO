#!/usr/bin/env python3
"""
Dual-Physics Shell Thickness Optimization Loop
==============================================

Multi-objective shell thickness optimization combining:
1. FEA (stress-based): Multi-orientation drop impact testing
2. Peridigm (fracture-based): High-velocity cone impact fracture testing

This loop:
1. Initializes shell thickness parameterization (theta bands)
2. For each iteration:
   - Generates mesh with current thickness parameters
   - PARALLEL EXECUTION:
     * Runs multi-orientation FEA (all 5 drop orientations)
     * Runs Peridigm fracture simulation (vertical orientation, high velocity)
   - Combines results in weighted fitness function
   - Updates thickness based on combined objectives
   - Checks convergence
3. Saves optimized thickness and convergence history

Key differences from FEA-only optimization:
- Runs Peridigm in addition to FEA
- Weighted fitness function combines stress and damage metrics
- Optimization balances structural integrity AND fracture performance
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

# Import adapters from parent src directory
from ..adapters import run_fea, run_peridigm, validate_fea_config, validate_peridigm_config
from ..adapters.mesh_adapter import load_mesh_for_mvp
from ..adapters.peridigm_adapter import (
    compute_path_based_breach,
    read_peridigm_discretization,
    read_exodus_damage_with_coords,
)

# Import thickness optimization components
from .theta_band_params import ThetaBandParams, create_uniform_bands
from .thickness_updater import (
    update_thickness, check_convergence,
    reset_momentum_history, reset_fracture_bisection_history,
    get_fracture_bisection_history, set_fracture_bisection_history,
    reset_secant_history, get_secant_history, set_secant_history
)


def run_dual_physics_optimization(config_file: str) -> Tuple[ThetaBandParams, List[Dict]]:
    """
    Run dual-physics shell thickness optimization loop.

    Combines FEA (stress-based) and Peridigm (fracture-based) objectives.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Tuple of:
        - Optimized thickness parameters
        - Convergence history (list of iteration info dicts)
    """
    print("=" * 70)
    print("DUAL-PHYSICS SHELL THICKNESS OPTIMIZATION")
    print("Multi-Orientation FEA + Peridigm Fracture Simulation")
    print("=" * 70)

    # Load configuration
    config = _load_config(config_file)

    # Print configuration summary
    print(f"\nConfiguration: {config.get('version', 'unknown')}")
    print(f"FEA backend: {config.get('fea', {}).get('backend', 'simple')}")
    print(f"Peridigm backend: {config.get('peridigm', {}).get('backend', 'mock')}")
    print(f"Mesh backend: {config.get('mesh', {}).get('backend', 'fallback')}")

    # Validate FEA configuration
    print("\nValidating FEA configuration...")
    fea_errors = validate_fea_config(config)
    if fea_errors:
        print("\n⚠ FEA configuration errors:")
        for err in fea_errors:
            print(f"  - {err}")
        raise ValueError("Invalid FEA configuration. Fix errors above and retry.")
    print("✓ FEA configuration valid")

    # Validate Peridigm configuration
    print("\nValidating Peridigm configuration...")
    peridigm_errors = validate_peridigm_config(config)
    if peridigm_errors:
        print("\n⚠ Peridigm configuration errors:")
        for err in peridigm_errors:
            print(f"  - {err}")
        raise ValueError("Invalid Peridigm configuration. Fix errors above and retry.")
    print("✓ Peridigm configuration valid\n")

    # Setup output directory
    output_dir = _resolve_output_dir(config)

    # Initialize thickness parameters
    thickness_params = _initialize_thickness_params(config)
    print(f"Initial thickness: {thickness_params}\n")

    # Reset history for new campaign
    reset_momentum_history()
    reset_fracture_bisection_history()
    reset_secant_history()

    # Optimization settings
    opt_cfg = config.get("optimization", {})
    max_iterations = int(opt_cfg.get("max_iterations", 20))

    # Weighted fitness function weights
    weights = opt_cfg.get("dual_physics_weights", {})
    w_stress = weights.get("stress", 1.0)
    w_fracture_miss = weights.get("fracture_miss", 1.0)
    w_shell_damage = weights.get("shell_damage", 2.0)
    w_energy = weights.get("energy_absorption", 0.0)  # Default 0 for backwards compatibility

    print(f"Fitness weights: stress={w_stress:.1f}, fracture_miss={w_fracture_miss:.1f}, "
          f"shell_damage={w_shell_damage:.1f}, energy_absorption={w_energy:.1f}\n")

    history = []

    for iteration in range(max_iterations):
        print(f"\n{'=' * 70}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'=' * 70}")
        print(f"Thickness: {thickness_params}")

        # ====================================================================
        # STEP 1: Generate mesh with current thickness
        # ====================================================================
        print("\n[1/4] Generating mesh...")
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
        print("\n[2/4] Running multi-orientation FEA...")
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

        # Save FEA outputs to persistent storage (if enabled)
        fea_cfg = config_iter.get("fea", {})
        if fea_cfg.get("save_outputs", True):
            _save_fea_outputs(fea_results, output_dir, iteration + 1)

        # ====================================================================
        # STEP 3: Run Peridigm fracture simulation
        # ====================================================================
        print("\n[3/4] Running Peridigm fracture simulation...")
        t_peridigm_start = time.time()

        # Get region IDs from config
        regions_cfg = config.get("regions", {})
        fracture_zone_ids = regions_cfg.get("fracture_zone_ids", [1, 2])
        shell_body_ids = regions_cfg.get("shell_body_ids", [3, 4, 5])

        # Get impact velocity from config (high velocity for fracture testing)
        peridigm_cfg = config.get("peridigm", {})
        # Support both "impact_velocity" and legacy "impact_velocity_m_s" keys
        impact_velocity = peridigm_cfg.get("impact_velocity", peridigm_cfg.get("impact_velocity_m_s", 250.0))
        print(f"      Using Peridigm impact velocity: {impact_velocity} m/s")

        peridigm_results = run_peridigm(
            mesh_data,
            density,
            fracture_zone_ids,
            shell_body_ids,
            impact_velocity,
            config_iter,
        )
        t_peridigm_end = time.time()

        # Print Peridigm summary
        D_fracture = peridigm_results['D_fracture_zone']
        D_shell = peridigm_results['D_shell_body']
        backend = peridigm_results.get('backend', 'unknown')
        print(f"      Backend: {backend}")
        print(f"      Fracture zone damage: {D_fracture:.3f}")
        print(f"      Shell body damage: {D_shell:.3f}")
        print(f"      ⏱ Peridigm took {t_peridigm_end - t_peridigm_start:.1f}s")

        # ====================================================================
        # PATH-BASED BREACH DETECTION
        # ====================================================================
        # Compute breach based on connected path through broken bonds
        # This replaces the simple D_fracture > 0.5 heuristic
        breach_info = _compute_path_breach(output_dir, iteration + 1, config_iter, fracture_zone_ids)
        if breach_info is not None:
            peridigm_results['path_breach'] = breach_info
            print(f"      Path breach: {'YES' if breach_info['breach'] else 'NO'}")
            print(f"      Crack depth: {breach_info['crack_depth']:.1%}")
            if not breach_info['breach'] and breach_info['blocked_at_band'] is not None:
                print(f"      Blocked at band: {breach_info['blocked_at_band']}")

        # ====================================================================
        # STEP 4: Update thickness based on dual-physics objectives
        # ====================================================================
        print("\n[4/4] Updating thickness (dual-physics)...")
        t_update_start = time.time()

        # Compute weighted fitness function
        fitness_metrics = _compute_dual_physics_fitness(
            fea_results,
            peridigm_results,
            config_iter,
            w_stress,
            w_fracture_miss,
            w_shell_damage,
            w_energy,
        )

        print(f"      Fitness metrics:")
        print(f"        Stress violation: {fitness_metrics['stress_violation']:.3f}")
        print(f"        Fracture miss: {fitness_metrics['fracture_miss']:.3f}")
        print(f"        Shell damage: {fitness_metrics['shell_damage_excess']:.3f}")
        print(f"        Energy absorption: {fitness_metrics['energy_absorption_deficit']:.3f} "
              f"(ratio={fitness_metrics.get('energy_absorption_ratio', 0):.2f})")
        print(f"        Total fitness: {fitness_metrics['total_fitness']:.3f}")

        # Update thickness (using combined objectives)
        # FEA stress drives direction, Peridigm fracture constrains increases
        thickness_params, update_info = update_thickness(
            thickness_params,
            fea_results,
            mesh_data,
            config_iter,
            peridigm_results=peridigm_results,  # For fracture-aware cap
        )

        # Apply Peridigm-based adjustments (experimental)
        # IMPORTANT: Pass original `config` (not config_iter) so fracture_factor_theta
        # changes persist across iterations
        # NEW: Also pass FEA results to balance fracture strength vs breakability
        thickness_params = _apply_peridigm_adjustments(
            thickness_params,
            peridigm_results,
            fitness_metrics,
            config,  # Original config - changes persist!
            fea_results=fea_results,  # For FEA stress in fracture zone
        )

        # Apply hole fillet adjustment based on pole stress
        # Fillet helps reduce stress concentration at hole rim
        hole_fillet_info = _update_hole_fillet(
            fea_results,
            thickness_params,
            config,  # Original config - changes persist!
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

        # Add dual-physics convergence criteria
        if fitness_metrics['total_fitness'] < opt_cfg.get("fitness_tolerance", 0.1):
            converged = True
            reasons.append("dual_physics_fitness_converged")

        # Compute relative volume (proportional to mean thickness)
        mean_thickness_mm = float(np.mean(thickness_params.thickness_mm))
        volume_relative = mean_thickness_mm / float(opt_cfg.get("initial_thickness_mm", 1.0))

        # Build history entry
        history_entry = {
            "iteration": iteration + 1,
            "thickness_mm": thickness_params.thickness_mm.tolist(),
            "mean_thickness_mm": mean_thickness_mm,
            "volume_relative": volume_relative,
            "max_stress_MPa": max_stress_mpa,
            "worst_case_orientation": worst_orientation,
            "stress_by_band_MPa": update_info.get("stress_by_band_MPa", []),
            "stress_by_band_max_MPa": fea_results.get("stress_by_band_max_MPa", []),
            "stress_p95_by_band_max_MPa": fea_results.get("stress_p95_by_band_max_MPa", []),
            "stress_p80_by_band_max_MPa": fea_results.get("stress_p80_by_band_max_MPa", []),
            "stress_percentile_used": opt_cfg.get("stress_band_stat", "max"),
            "thickness_change_norm": update_info["thickness_change_norm"],
            # Peridigm metrics
            "D_fracture_zone": D_fracture,
            "D_shell_body": D_shell,
            "peridigm_backend": backend,
            # Path-based breach detection (new)
            "fracture_breach": breach_info['breach'] if breach_info else (D_fracture > 0.5),
            "crack_depth": breach_info['crack_depth'] if breach_info else D_fracture,
            "blocked_at_band": breach_info.get('blocked_at_band') if breach_info else None,
            # Dual-physics fitness
            "fitness_stress_violation": fitness_metrics['stress_violation'],
            "fitness_fracture_miss": fitness_metrics['fracture_miss'],
            "fitness_shell_damage": fitness_metrics['shell_damage_excess'],
            "fitness_energy_absorption": fitness_metrics['energy_absorption_deficit'],
            "fitness_total": fitness_metrics['total_fitness'],
            # Energy absorption metrics
            "energy_absorption_ratio": fitness_metrics.get('energy_absorption_ratio', 0.0),
            "work_j": fea_results.get('work_j'),
            "input_kinetic_energy_j": fea_results.get('input_kinetic_energy_j'),
            # Convergence
            "converged": converged,
            "convergence_reasons": reasons,
            # Hole fillet (stress concentration reduction)
            "hole_fillet_factor": hole_fillet_info.get("hole_fillet_factor", 0.0),
            "hole_fillet_mm": hole_fillet_info.get("hole_fillet_mm", 0.0),
            # Fracture factor (for completeness)
            "fracture_factor": thickness_params.fracture_factor.tolist(),
            "fracture_thickness_mm": (thickness_params.thickness_mm * thickness_params.fracture_factor).tolist(),
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
        _save_iteration_results(
            iteration + 1,
            thickness_params,
            history_entry,
            output_dir,
        )

        # Print convergence status
        if converged:
            print(f"\n✓ Converged after {iteration + 1} iterations!")
            print(f"  Reasons: {', '.join(reasons)}")
            break

    # Save final results
    _save_final_results(thickness_params, history, output_dir)

    print("\n" + "=" * 70)
    print("DUAL-PHYSICS OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Final thickness: {thickness_params}")
    print(f"Iterations: {len(history)}")
    print(f"Final stress: {history[-1]['max_stress_MPa']:.2f} MPa")
    print(f"Final fracture damage: {history[-1]['D_fracture_zone']:.3f}")
    print(f"Final shell damage: {history[-1]['D_shell_body']:.3f}")
    print(f"Final energy absorption: {history[-1].get('energy_absorption_ratio', 0):.2f}")
    print(f"Final fitness: {history[-1]['fitness_total']:.3f}")
    print(f"Results saved to: {output_dir}")

    return thickness_params, history


def _compute_dual_physics_fitness(
    fea_results: Dict,
    peridigm_results: Dict,
    config: Dict,
    w_stress: float,
    w_fracture_miss: float,
    w_shell_damage: float,
    w_energy: float = 0.0,
) -> Dict:
    """
    Compute weighted fitness function from FEA and Peridigm results.

    Objectives:
    1. Minimize stress violation (FEA): max_stress <= stress_limit
    2. Maximize fracture completeness (Peridigm): fracture zone must break
    3. Maximize shell integrity (Peridigm): shell body must stay intact
    4. Maximize energy absorption (FEA): higher absorption = better protection

    Uses new completeness-based metrics:
    - Fracture completeness: % of fracture zone with damage >= 90%
    - Fracture breach: Binary - did it break through anywhere?
    - Shell integrity: % of shell body with damage < 10%
    - Energy absorption ratio: work_done / input_kinetic_energy

    Args:
        fea_results: FEA simulation results
        peridigm_results: Peridigm simulation results
        config: Configuration dict
        w_stress: Weight for stress violation term
        w_fracture_miss: Weight for fracture miss term
        w_shell_damage: Weight for shell damage term
        w_energy: Weight for energy absorption term (default 0 for backwards compatibility)

    Returns:
        {
            'stress_violation': float,
            'fracture_miss': float,
            'shell_damage_excess': float,
            'no_breach_penalty': float,
            'energy_absorption_deficit': float,
            'total_fitness': float,
        }
    """
    fea_cfg = config.get("fea", {})
    peridigm_cfg = config.get("peridigm", {})

    # Stress violation (minimize)
    stress_limit_pa = fea_cfg.get("stress_limit_Pa", 44.0e6)
    max_stress = fea_results['max_stress']
    stress_violation = max(0.0, max_stress - stress_limit_pa) / stress_limit_pa

    # Get new fracture metrics
    fracture_metrics = peridigm_results.get("fracture_metrics", {})
    shell_metrics = peridigm_results.get("shell_metrics", {})

    # Fracture completeness miss (minimize) - want high completeness
    target_completeness = peridigm_cfg.get("fracture_completeness_target", 0.5)
    fracture_completeness = fracture_metrics.get("completeness", 0.0)
    fracture_breach = fracture_metrics.get("breach", False)

    # Fallback to old metrics if new ones not available
    if not fracture_metrics:
        D_fracture = peridigm_results.get('D_fracture_zone', 0.0)
        fracture_completeness = D_fracture
        fracture_breach = D_fracture > 0.5

    fracture_miss = max(0.0, target_completeness - fracture_completeness)

    # NO BREACH PENALTY - strong penalty if fracture zone didn't break at all
    no_breach_penalty = 0.0 if fracture_breach else 1.0  # Binary: 0 if breach, 1 if no breach

    # Shell integrity deficit (minimize) - want high integrity
    min_integrity = peridigm_cfg.get("min_shell_integrity", 0.8)
    shell_integrity = shell_metrics.get("integrity", 1.0)

    # Fallback to old metrics
    if not shell_metrics:
        D_shell = peridigm_results.get('D_shell_body', 0.0)
        shell_integrity = 1.0 - D_shell

    shell_damage_excess = max(0.0, min_integrity - shell_integrity)

    # Also store legacy metrics for backwards compatibility
    D_fracture = peridigm_results.get('D_fracture_zone', fracture_completeness)
    D_shell = peridigm_results.get('D_shell_body', 1.0 - shell_integrity)

    # Energy absorption (maximize - so we minimize the deficit)
    # Higher absorption ratio = better (more energy dissipated by deformation)
    opt_cfg = config.get("optimization", {})
    target_energy_ratio = opt_cfg.get("target_energy_absorption_ratio", 0.7)  # Target 70% absorption
    energy_absorption_ratio = fea_results.get("energy_absorption_ratio", 0.0) or 0.0

    # Deficit: how far below target (0 if at or above target)
    energy_absorption_deficit = max(0.0, target_energy_ratio - energy_absorption_ratio)

    # Weighted total fitness (minimize)
    # Add strong weight for no_breach_penalty
    w_no_breach = 2.0  # Strong penalty for no breach
    total_fitness = (
        w_stress * stress_violation +
        w_fracture_miss * fracture_miss +
        w_shell_damage * shell_damage_excess +
        w_no_breach * no_breach_penalty +
        w_energy * energy_absorption_deficit
    )

    return {
        'stress_violation': stress_violation,
        'fracture_miss': fracture_miss,
        'shell_damage_excess': shell_damage_excess,
        'no_breach_penalty': no_breach_penalty,
        'energy_absorption_deficit': energy_absorption_deficit,
        'energy_absorption_ratio': energy_absorption_ratio,
        'target_energy_ratio': target_energy_ratio,
        'fracture_completeness': fracture_completeness,
        'fracture_breach': fracture_breach,
        'shell_integrity': shell_integrity,
        'D_fracture_zone': D_fracture,
        'D_shell_body': D_shell,
        'total_fitness': total_fitness,
    }


def _apply_peridigm_adjustments(
    thickness_params: ThetaBandParams,
    peridigm_results: Dict,
    fitness_metrics: Dict,
    config: Dict,
    fea_results: Dict = None,
) -> ThetaBandParams:
    """
    Apply dual-physics adjustments to fracture_factor.

    Balances two competing objectives:
    1. FEA (drop impact): Wants THICKER fracture zone to survive normal drops
    2. Peridigm (flyback): Wants THINNER fracture zone to break during flyback

    OPTIMAL = As thick as possible while still breaking during flyback

    Logic (priority order):
    1. If NO fracture breach: DECREASE factor (must break!) - HIGHEST PRIORITY
    2. If breach=True AND FEA stress high: INCREASE factor (need strength for drops)
    3. If breach=True AND FEA stress OK: maintain current factor

    Args:
        thickness_params: Current thickness parameters (NOT modified by Peridigm)
        peridigm_results: Peridigm simulation results
        fitness_metrics: Computed fitness metrics
        config: Configuration dict (fracture_factor may be modified)
        fea_results: FEA results for stress-based adjustment (optional)

    Returns:
        Unchanged thickness parameters (Peridigm only adjusts fracture_factor)
    """
    # Get current fracture_factor_theta from config
    mesh_cfg = config.get("mesh", {})
    theta_overrides = mesh_cfg.get("theta_overrides", {})
    fracture_design = theta_overrides.get("fracture_design", {})
    fracture_factor = fracture_design.get("fracture_factor_theta", None)

    if fracture_factor is None:
        print("[Peridigm Adjust] No fracture_factor_theta in config, skipping adjustment")
        return thickness_params

    fracture_factor = np.array(fracture_factor, dtype=float)
    n_bands = len(fracture_factor)

    # Get peridigm config for thresholds
    peridigm_cfg = config.get("peridigm", {})
    target_completeness = peridigm_cfg.get("fracture_completeness_target", 0.5)  # 50% fully broken
    min_shell_integrity = peridigm_cfg.get("min_shell_integrity", 0.8)  # 80% intact

    # ==========================================================================
    # PATH-BASED BREACH DETECTION (preferred)
    # ==========================================================================
    # Use path-based breach if available - this is the NEW approach that tracks
    # connected broken bonds from top to bottom
    path_breach = peridigm_results.get('path_breach', None)

    if path_breach is not None:
        fracture_breach = path_breach['breach']
        blocked_at_band = path_breach.get('blocked_at_band', None)
        crack_depth = path_breach.get('crack_depth', 0.0)
        print(f"[Peridigm Adjust] Using PATH-BASED breach detection")
        print(f"[Peridigm Adjust] Breach: {'YES' if fracture_breach else 'NO'}")
        print(f"[Peridigm Adjust] Crack depth: {crack_depth:.1%}")
        if blocked_at_band is not None:
            print(f"[Peridigm Adjust] Blocked at band: {blocked_at_band}")
    else:
        # Fallback to legacy metrics
        fracture_metrics = peridigm_results.get("fracture_metrics", {})
        D_fracture = peridigm_results.get('D_fracture_zone', 0.0)

        if fracture_metrics:
            fracture_breach = fracture_metrics.get("breach", False)
            if not fracture_breach and D_fracture > 0.5:
                fracture_breach = True
        else:
            fracture_breach = D_fracture > 0.5

        blocked_at_band = None  # Unknown with legacy detection
        crack_depth = D_fracture  # Use mean damage as proxy

        print(f"[Peridigm Adjust] Using LEGACY breach detection (D_fracture={D_fracture:.3f})")
        print(f"[Peridigm Adjust] Breach: {'YES' if fracture_breach else 'NO'}")

    # Adjustment parameters
    min_fracture_factor = 0.15  # Don't go below 15% of protected thickness
    max_fracture_factor = 1.0   # Don't exceed protected thickness

    adjusted = False
    original_factor = fracture_factor.copy()

    # ==========================================================================
    # Get FEA stress info for fracture zone (if available)
    # ==========================================================================
    fea_cfg = config.get("fea", {})
    stress_limit_pa = float(fea_cfg.get("stress_limit_Pa", 44.0e6))
    opt_cfg = config.get("optimization", {})
    target_sf = float(opt_cfg.get("target_safety_factor", 1.0))
    target_stress_pa = stress_limit_pa / target_sf
    target_stress_mpa = target_stress_pa / 1e6

    # Check if FEA stress in fracture zone bands is high
    fea_stress_high = False
    fracture_zone_stress_mpa = 0.0
    if fea_results is not None:
        # Fracture zones are typically at bands near equator and pole edges
        # For now, use max stress across all bands as proxy
        stress_by_band = fea_results.get("stress_p95_by_band_max_MPa")
        if stress_by_band is None:
            stress_by_band = fea_results.get("stress_by_band_max_MPa", [])
        if stress_by_band:
            fracture_zone_stress_mpa = max(stress_by_band)
            # Consider stress "high" if > 80% of target
            fea_stress_high = fracture_zone_stress_mpa > (target_stress_mpa * 0.8)
            print(f"[Peridigm Adjust] FEA stress in fracture zone: {fracture_zone_stress_mpa:.1f} MPa "
                  f"(target: {target_stress_mpa:.1f} MPa, high={fea_stress_high})")

    # ==========================================================================
    # CRITICAL: If fracture zone didn't breach, thin the BLOCKING BAND
    # This is HIGHEST PRIORITY - fracture MUST break during flyback
    # ==========================================================================
    if not fracture_breach:
        strong_reduction = 0.15  # 15% reduction for blocking band

        if blocked_at_band is not None and 0 <= blocked_at_band < n_bands:
            # TARGET THE SPECIFIC BLOCKING BAND
            # Only reduce the band that's preventing crack propagation
            old_val = fracture_factor[blocked_at_band]
            fracture_factor[blocked_at_band] -= strong_reduction

            # Also reduce adjacent bands slightly (crack needs to propagate through)
            adjacent_reduction = strong_reduction * 0.5
            if blocked_at_band > 0:
                fracture_factor[blocked_at_band - 1] -= adjacent_reduction
            if blocked_at_band < n_bands - 1:
                fracture_factor[blocked_at_band + 1] -= adjacent_reduction

            fracture_factor = np.clip(fracture_factor, min_fracture_factor, max_fracture_factor)
            print(f"[Peridigm Adjust] NO BREACH - Crack blocked at band {blocked_at_band}")
            print(f"                  Thinning band {blocked_at_band}: {old_val:.3f} -> {fracture_factor[blocked_at_band]:.3f}")
            print(f"                  (Adjacent bands also reduced by {adjacent_reduction:.3f})")
        else:
            # Fallback: uniform reduction if we don't know which band blocked
            fracture_factor = fracture_factor - strong_reduction
            fracture_factor = np.clip(fracture_factor, min_fracture_factor, max_fracture_factor)
            print(f"[Peridigm Adjust] NO BREACH - Applying uniform reduction: -{strong_reduction}")
            print(f"                  (Blocking band unknown - reducing all bands)")

        adjusted = True

    # ==========================================================================
    # If breach happened AND FEA stress is high: INCREASE fracture_factor
    # We can afford more thickness since it's still breaking
    # ==========================================================================
    elif fracture_breach and fea_stress_high:
        # Breach works AND stress is high → increase factor for better drop survival
        stress_ratio = fracture_zone_stress_mpa / target_stress_mpa
        increase = 0.05 * min(stress_ratio - 0.8, 0.5)  # Cap increase
        fracture_factor = fracture_factor + increase
        fracture_factor = np.clip(fracture_factor, min_fracture_factor, max_fracture_factor)
        print(f"[Peridigm Adjust] BREACH OK + FEA stress high ({fracture_zone_stress_mpa:.1f} MPa)")
        print(f"                  Increasing fracture_factor by {increase:.3f} for drop survival")
        adjusted = True

    # ==========================================================================
    # If breach happened but crack depth is low, reduce bands above blocked point
    # ==========================================================================
    elif crack_depth < 0.9:  # Crack didn't get close to bottom
        # Crack propagated but stopped early - thin bands below current crack tip
        moderate_reduction = 0.05 * (1.0 - crack_depth)
        fracture_factor = fracture_factor - moderate_reduction
        fracture_factor = np.clip(fracture_factor, min_fracture_factor, max_fracture_factor)
        print(f"[Peridigm Adjust] Low crack depth ({crack_depth:.1%})")
        print(f"                  Reducing fracture_factor by {moderate_reduction:.3f}")
        adjusted = True

    # ==========================================================================
    # SHELL DAMAGE HANDLING - DISABLED
    # ==========================================================================
    # Shell damage from Peridigm is IGNORED. The flyback event (200 m/s) is
    # catastrophic and destroys everything - that's expected behavior.
    # Shell structural integrity is handled by the FEA stress constraints.
    # Peridigm ONLY validates that the fracture zone breaks properly.

    # ==========================================================================
    # CRITICAL: Enforce minimum meshable fracture zone thickness
    # ==========================================================================
    # The geometry generator needs minimum thickness for fracture zones to mesh
    opt_cfg = config.get("optimization", {})
    min_meshable_mm = float(opt_cfg.get("min_fracture_thickness_mm", 0.4))
    fracture_zone_mm = thickness_params.thickness_mm * fracture_factor
    too_thin_mask = fracture_zone_mm < min_meshable_mm

    if np.any(too_thin_mask):
        # Calculate minimum fracture_factor needed for each band
        min_factor_needed = min_meshable_mm / thickness_params.thickness_mm
        # Only adjust bands that are too thin
        fracture_factor = np.where(
            too_thin_mask,
            np.maximum(fracture_factor, min_factor_needed),
            fracture_factor
        )
        fracture_factor = np.clip(fracture_factor, min_fracture_factor, max_fracture_factor)

        # Calculate final fracture zone thicknesses
        final_fracture_mm = thickness_params.thickness_mm * fracture_factor

        print(f"[Peridigm Adjust] Enforcing min meshable thickness ({min_meshable_mm}mm):")
        for i in range(len(fracture_factor)):
            if too_thin_mask[i]:
                print(f"                  Band {i}: protected={thickness_params.thickness_mm[i]:.2f}mm, "
                      f"factor={fracture_factor[i]:.3f} -> fracture={final_fracture_mm[i]:.2f}mm")
        adjusted = True

    if adjusted:
        # Update config with new fracture_factor
        fracture_design["fracture_factor_theta"] = fracture_factor.tolist()
        # CRITICAL: Also update thickness_params so it persists to next iteration
        # (otherwise _update_config_with_thickness overwrites with old value)
        thickness_params.fracture_factor = fracture_factor.copy()
        print(f"[Peridigm Adjust] Updated fracture_factor_theta:")
        print(f"                  Before: {original_factor.tolist()}")
        print(f"                  After:  {fracture_factor.tolist()}")
    else:
        print(f"[Peridigm Adjust] Fracture zone performing well - no adjustment needed")

    return thickness_params


def _update_hole_fillet(
    fea_results: Dict,
    thickness_params: ThetaBandParams,
    config: Dict,
) -> Dict:
    """
    Update hole fillet factor based on pole (band 0) stress.

    The fillet reduces stress concentration at the hole rim. If pole stress
    is too high despite thickness increases, the fillet should grow.

    Strategy:
    - If band 0 stress > target: increase fillet factor
    - If band 0 stress < target: optionally decrease fillet (save material)
    - Fillet is bounded by [min_fillet_factor, max_fillet_factor]

    Args:
        fea_results: FEA results with stress by band
        thickness_params: Current thickness parameters
        config: Configuration dict (hole_fillet_factor may be modified)

    Returns:
        Dict with hole_fillet_factor and hole_fillet_mm
    """
    # Get optimization config
    opt_cfg = config.get("optimization", {})
    fea_cfg = config.get("fea", {})

    # Target stress (Pa)
    stress_limit_pa = float(fea_cfg.get("stress_limit_Pa", 44.0e6))
    target_sf = float(opt_cfg.get("target_safety_factor", 1.0))
    target_stress_pa = stress_limit_pa / target_sf

    # Fillet optimization settings
    fillet_cfg = opt_cfg.get("hole_fillet", {})
    optimize_fillet = fillet_cfg.get("optimize", True)
    min_fillet_factor = fillet_cfg.get("min_factor", 0.0)
    max_fillet_factor = fillet_cfg.get("max_factor", 1.0)
    fillet_relaxation = fillet_cfg.get("relaxation", 0.3)  # How fast to adjust

    # Get current fillet factor from config
    mesh_cfg = config.setdefault("mesh", {})
    theta_overrides = mesh_cfg.setdefault("theta_overrides", {})
    geom_overrides = theta_overrides.setdefault("geometry", {})

    current_fillet_factor = geom_overrides.get("hole_fillet_factor", 0.0)

    # Get pole (band 0) stress - use P95 if available, else max
    stress_by_band = fea_results.get("stress_p95_by_band_max_MPa")
    if stress_by_band is None:
        stress_by_band = fea_results.get("stress_by_band_max_MPa", [0])

    if len(stress_by_band) == 0:
        print("[Fillet] No stress data available, keeping current fillet")
        t0_mm = float(thickness_params.thickness_mm[0])
        return {
            "hole_fillet_factor": current_fillet_factor,
            "hole_fillet_mm": current_fillet_factor * t0_mm,
        }

    pole_stress_mpa = stress_by_band[0]
    pole_stress_pa = pole_stress_mpa * 1e6
    target_stress_mpa = target_stress_pa / 1e6

    # Current pole thickness and minimum shell thickness
    t0_mm = float(thickness_params.thickness_mm[0])
    t_min_mm = float(np.min(thickness_params.thickness_mm))

    # Compute actual fillet size (limited by pole band thickness)
    def compute_fillet_mm(factor):
        raw_fillet = factor * t0_mm
        # Limit to 95% of pole thickness to avoid geometry intersection
        # (fillet is at the pole hole, so it's bounded by pole thickness, not min global thickness)
        return min(raw_fillet, t0_mm * 0.95)

    if not optimize_fillet:
        print(f"[Fillet] Optimization disabled, keeping factor={current_fillet_factor:.2f}")
        return {
            "hole_fillet_factor": current_fillet_factor,
            "hole_fillet_mm": compute_fillet_mm(current_fillet_factor),
        }

    # Compute stress ratio for band 0
    stress_ratio = pole_stress_pa / target_stress_pa if target_stress_pa > 0 else 1.0

    print(f"[Fillet] Band 0 stress: {pole_stress_mpa:.1f} MPa (target: {target_stress_mpa:.1f} MPa, ratio: {stress_ratio:.2f})")
    print(f"[Fillet] Current fillet factor: {current_fillet_factor:.2f} ({current_fillet_factor * t0_mm:.2f} mm)")

    new_fillet_factor = current_fillet_factor

    if stress_ratio > 1.2:
        # Stress significantly above target - increase fillet
        # Increase proportional to how much over target
        increase = fillet_relaxation * (stress_ratio - 1.0) * 0.5
        new_fillet_factor = current_fillet_factor + increase
        print(f"[Fillet] Stress {stress_ratio:.1f}× target -> increasing fillet by {increase:.3f}")

    elif stress_ratio > 1.05:
        # Stress slightly above target - small increase
        increase = fillet_relaxation * 0.1
        new_fillet_factor = current_fillet_factor + increase
        print(f"[Fillet] Stress slightly high -> small increase by {increase:.3f}")

    elif stress_ratio < 0.8 and current_fillet_factor > min_fillet_factor:
        # Stress well below target - could reduce fillet (optional)
        # Be conservative - only reduce slowly
        decrease = fillet_relaxation * 0.05
        new_fillet_factor = current_fillet_factor - decrease
        print(f"[Fillet] Stress low -> could reduce by {decrease:.3f}")

    else:
        print(f"[Fillet] Stress acceptable - no change needed")

    # Clamp to bounds
    new_fillet_factor = np.clip(new_fillet_factor, min_fillet_factor, max_fillet_factor)

    # Update config if changed
    if abs(new_fillet_factor - current_fillet_factor) > 0.001:
        geom_overrides["hole_fillet_factor"] = float(new_fillet_factor)
        print(f"[Fillet] Updated: {current_fillet_factor:.3f} -> {new_fillet_factor:.3f}")
        actual_fillet = compute_fillet_mm(new_fillet_factor)
        print(f"[Fillet] New fillet size: {actual_fillet:.2f} mm (factor={new_fillet_factor:.0%}, t0={t0_mm:.2f}mm, t_min={t_min_mm:.2f}mm)")

    return {
        "hole_fillet_factor": float(new_fillet_factor),
        "hole_fillet_mm": compute_fillet_mm(new_fillet_factor),
    }


def _compute_path_breach(
    output_dir: Path,
    iteration: int,
    config: Dict,
    fracture_zone_ids: List[int],
) -> Dict:
    """
    Compute path-based breach detection from Peridigm results.

    Reads the discretization and Exodus damage files, then determines if there's
    a connected path of broken bonds from top to bottom of the fracture zone.

    Args:
        output_dir: Output directory containing peridigm/ subdirectory
        iteration: Current iteration number
        config: Configuration dict
        fracture_zone_ids: Block IDs for fracture zones

    Returns:
        Breach info dict or None if files not found
    """
    # Find Peridigm output directory - try multiple naming conventions
    peridigm_dir = output_dir / "peridigm" / f"iter_{iteration:03d}"  # iter_001
    if not peridigm_dir.exists():
        peridigm_dir = output_dir / "peridigm" / f"iter_{iteration}"  # iter_1
    if not peridigm_dir.exists():
        peridigm_dir = output_dir / "peridigm"  # fallback to root

    if not peridigm_dir.exists():
        print(f"[Path Breach] Peridigm directory not found: {peridigm_dir}")
        return None

    # Find discretization file
    disc_files = list(peridigm_dir.glob("*_peridigm.txt")) + list(peridigm_dir.glob("split_core_peridigm.txt"))
    if not disc_files:
        # Try parent peridigm directory
        parent_peridigm = output_dir / "peridigm"
        disc_files = list(parent_peridigm.glob("*_peridigm.txt"))

    if not disc_files:
        print(f"[Path Breach] No discretization file found in {peridigm_dir}")
        return None

    disc_file = disc_files[0]
    print(f"[Path Breach] Using discretization: {disc_file.name}")

    # Read discretization
    try:
        points, block_ids, _ = read_peridigm_discretization(disc_file)
    except Exception as e:
        print(f"[Path Breach] Error reading discretization: {e}")
        return None

    # Read Exodus damage
    damage_points, damage = read_exodus_damage_with_coords(peridigm_dir)

    if damage is None:
        print(f"[Path Breach] Could not read Exodus damage files")
        return None

    # If points don't match, try to align by using discretization file block IDs
    # and damage field in order
    if len(damage) != len(points):
        print(f"[Path Breach] Point count mismatch: disc={len(points)}, exodus={len(damage)}")
        # Use the exodus points and damage directly if available
        if damage_points is not None and len(damage_points) == len(damage):
            points = damage_points
            # Estimate block IDs from position (fracture zone is typically at phi=0,180)
            # For now, assume all points are potentially fracture zone
            block_ids = np.full(len(points), fracture_zone_ids[0] if fracture_zone_ids else 3)
            print(f"[Path Breach] Using Exodus coordinates with estimated block IDs")
        else:
            print(f"[Path Breach] Cannot align points and damage")
            return None

    # Get theta band edges from config
    mesh_cfg = config.get("mesh", {})
    theta_overrides = mesh_cfg.get("theta_overrides", {})
    mesh_params = theta_overrides.get("mesh", {})
    theta_band_edges = mesh_params.get("theta_band_edges_deg", [0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90])

    # Get horizon from config
    peridigm_cfg = config.get("peridigm", {})
    horizon = peridigm_cfg.get("horizon_mm", 9.0) / 1000.0  # Convert to meters

    # Compute path-based breach
    breach_info = compute_path_based_breach(
        points=points,
        damage=damage,
        block_ids=block_ids,
        fracture_zone_ids=fracture_zone_ids,
        theta_band_edges_deg=theta_band_edges,
        horizon=horizon,
        damage_threshold=0.9,
    )

    return breach_info


# ============================================================================
# Configuration and I/O Helpers
# ============================================================================

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
    """Load YAML configuration file.

    Expands ``${VAR}`` and ``~`` references in any string value so configs
    can use ``${HOME}/...`` paths portably across users.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return _expand_env_vars(config)


def _resolve_output_dir(config: Dict) -> Path:
    """Resolve output directory, creating if needed."""
    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("dir", "results_dual_physics"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _initialize_thickness_params(config: Dict) -> ThetaBandParams:
    """Initialize thickness parameters from config."""
    opt_cfg = config.get("optimization", {})
    n_bands = int(opt_cfg.get("n_theta_bands", 3))
    initial_thickness = float(opt_cfg.get("initial_thickness_mm", 1.0))

    return create_uniform_bands(n_bands, initial_thickness)


def _update_config_with_thickness(config: Dict, thickness_params: ThetaBandParams) -> Dict:
    """
    Update configuration with current thickness parameters.

    CRITICAL: Updates material.protected_thickness_theta_mm which is what
    the theta_mesh generator (ShellGeometryGeneratorRefined) actually reads.
    Also updates mesh.theta_overrides.thickness_design for backwards compat.

    Also computes hole_fillet_torus_mm from hole_fillet_factor × t0.
    """
    import copy
    config_iter = copy.deepcopy(config)

    # =========================================================================
    # PRIMARY: Update material.protected_thickness_theta_mm
    # This is what the theta_mesh generator (shell_geometry_generator_refined.py)
    # actually reads for protected shell thickness per theta band.
    # =========================================================================
    material_cfg = config_iter.setdefault("material", {})
    material_cfg["protected_thickness_theta_mm"] = thickness_params.thickness_mm.tolist()

    # =========================================================================
    # CRITICAL: Put material inside mesh.theta_overrides
    # The mesh_adapter._generate_theta_mesh() ONLY passes theta_overrides
    # to _write_theta_config(), ignoring top-level config sections.
    # So we MUST put material here for the theta_mesh generator to see it.
    # =========================================================================
    mesh_cfg = config_iter.setdefault("mesh", {})
    theta_overrides = mesh_cfg.setdefault("theta_overrides", {})

    # This is the KEY fix - put material in theta_overrides!
    theta_overrides.setdefault("material", {})["protected_thickness_theta_mm"] = thickness_params.thickness_mm.tolist()

    # Also update fracture_design in theta_overrides
    theta_overrides.setdefault("fracture_design", {})["fracture_factor_theta"] = thickness_params.fracture_factor.tolist()

    # Keep thickness_design for backwards compatibility
    thickness_design = theta_overrides.setdefault("thickness_design", {})
    thickness_design["theta_edges_deg"] = thickness_params.theta_edges_deg.tolist()
    thickness_design["thickness_mm"] = thickness_params.thickness_mm.tolist()

    # =========================================================================
    # HOLE FILLET: Compute from factor × t0 (pole thickness)
    # Reduces stress concentration at hole rim during vertical pole impact.
    # fillet_size = hole_fillet_factor × thickness_mm[0]
    # BUT: fillet must not exceed minimum shell thickness (geometry constraint)
    # =========================================================================
    geom_overrides = theta_overrides.setdefault("geometry", {})
    hole_fillet_factor = geom_overrides.get("hole_fillet_factor", 0.0)
    if hole_fillet_factor > 0:
        t0_mm = float(thickness_params.thickness_mm[0])  # Pole thickness
        t_min_mm = float(np.min(thickness_params.thickness_mm))  # Minimum shell thickness
        fillet_size_mm = hole_fillet_factor * t0_mm
        # Limit fillet to minimum shell thickness to avoid geometry intersection
        if fillet_size_mm > t_min_mm:
            print(f"[Config] Hole fillet: {fillet_size_mm:.2f}mm exceeds min shell {t_min_mm:.2f}mm, clamping")
            fillet_size_mm = t_min_mm * 0.95  # 95% of min thickness for safety margin
        geom_overrides["hole_fillet_torus_mm"] = fillet_size_mm
        print(f"[Config] Hole fillet: {hole_fillet_factor:.0%} × {t0_mm:.2f}mm = {fillet_size_mm:.2f}mm (min_t={t_min_mm:.2f}mm)")

    return config_iter


def _save_fea_outputs(fea_results: Dict, output_dir: Path, iteration: int):
    """
    Copy FEA output files to persistent storage.

    Copies VTK/PVD files from temporary FEA directory to output_dir/fea/iter_N/

    Args:
        fea_results: FEA results dict containing 'results_path'
        output_dir: Persistent output directory
        iteration: Current iteration number
    """
    import shutil

    fea_temp_path = fea_results.get("results_path")
    if not fea_temp_path:
        print("[FEA Output] No results_path in FEA results, skipping save")
        return

    fea_temp_dir = Path(fea_temp_path)
    print(f"[FEA Output] Source directory: {fea_temp_dir}")

    if not fea_temp_dir.exists():
        print(f"[FEA Output] FEA temp directory not found: {fea_temp_dir}")
        return

    # List what's in the source directory
    all_files = list(fea_temp_dir.glob("*"))
    print(f"[FEA Output] Found {len(all_files)} items in source directory")
    for f in all_files[:10]:  # Show first 10
        print(f"[FEA Output]   - {f.name}")
    if len(all_files) > 10:
        print(f"[FEA Output]   ... and {len(all_files) - 10} more")

    # Create persistent FEA output directory
    fea_output_dir = output_dir / "fea" / f"iter_{iteration:03d}"
    fea_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[FEA Output] Destination directory: {fea_output_dir}")

    # Copy VTK/PVD files and metrics - ONLY copy worst-case orientation to save space
    worst_orientation = fea_results.get("worst_case_orientation", "")
    patterns_to_copy = ["*.pvd", "*.vtu", "*.pvtu", "*_metrics.yaml"]
    files_copied = 0

    for pattern in patterns_to_copy:
        for src_file in fea_temp_dir.glob(pattern):
            # Only copy files for worst-case orientation (or all metrics)
            if pattern == "*_metrics.yaml" or worst_orientation in src_file.name:
                dst_file = fea_output_dir / src_file.name
                try:
                    shutil.copy2(src_file, dst_file)
                    files_copied += 1
                    print(f"[FEA Output]   Copied: {src_file.name}")
                except Exception as e:
                    print(f"[FEA Output] Warning: Could not copy {src_file.name}: {e}")

    # Write a summary file with worst-case info
    summary_file = fea_output_dir / "worst_case_summary.txt"
    try:
        with open(summary_file, 'w') as f:
            f.write(f"Iteration: {iteration}\n")
            f.write(f"Worst-case orientation: {worst_orientation}\n")
            f.write(f"Max stress: {fea_results.get('max_stress', 0) / 1e6:.2f} MPa\n")
            f.write(f"Files copied: {files_copied}\n")
        files_copied += 1
    except Exception as e:
        print(f"[FEA Output] Warning: Could not write summary: {e}")

    print(f"[FEA Output] Saved {files_copied} files to {fea_output_dir}")


def _save_iteration_results(
    iteration: int,
    thickness_params: ThetaBandParams,
    history_entry: Dict,
    output_dir: Path,
):
    """Save iteration results to JSON files."""
    # Save thickness parameters
    thickness_file = output_dir / f"thickness_iter{iteration:03d}.json"
    with open(thickness_file, 'w') as f:
        json.dump({
            "iteration": iteration,
            "theta_edges_deg": thickness_params.theta_edges_deg.tolist(),
            "thickness_mm": thickness_params.thickness_mm.tolist(),
        }, f, indent=2)

    # Append to history
    history_file = output_dir / "history.json"
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []

    history.append(history_entry)

    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def _save_final_results(
    thickness_params: ThetaBandParams,
    history: List[Dict],
    output_dir: Path,
):
    """Save final optimization results."""
    # Save final thickness
    final_thickness_file = output_dir / "thickness_final.json"
    with open(final_thickness_file, 'w') as f:
        json.dump({
            "theta_edges_deg": thickness_params.theta_edges_deg.tolist(),
            "thickness_mm": thickness_params.thickness_mm.tolist(),
        }, f, indent=2)

    # Save complete history
    history_file = output_dir / "history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nFinal results saved:")
    print(f"  - {final_thickness_file}")
    print(f"  - {history_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dual-physics shell thickness optimization"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()

    run_dual_physics_optimization(args.config)
