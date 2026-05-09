#!/usr/bin/env python3
"""
Run FEA for dual-physics optimization (FEA only, mesh already generated).

This script runs FEA using the mesh generated in the previous phase.
It runs in parallel with Peridigm for maximum efficiency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description="Run FEA for dual-physics optimization")
    parser.add_argument("--config", required=True, help="Configuration YAML file")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    # Add modules to path
    perifeato_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(perifeato_root))

    from perifeato.optimization.shell_thickness.thickness_loop import (
        _load_config,
        _update_config_with_thickness,
        _save_results,
    )
    from perifeato.optimization.adapters import run_fea
    from perifeato.optimization.adapters.mesh_adapter import _read_mesh_with_tags
    from perifeato.optimization.shell_thickness.thickness_updater import (
        update_thickness,
        update_fracture_factor,
        check_convergence,
        get_momentum_history,
        set_momentum_history,
        get_secant_history,
        set_secant_history,
        get_oc_history,
        set_oc_history,
        get_mma_history,
        set_mma_history,
        get_bo_history,
        set_bo_history,
        get_pgd_history,
        set_pgd_history,
        get_cobyla_history,
        set_cobyla_history,
    )
    from perifeato.optimization.shell_thickness.theta_band_params import ThetaBandParams
    from perifeato.optimization.shell_thickness.dual_physics_loop import _update_hole_fillet

    # Load config
    config = _load_config(args.config)
    output_dir = Path(args.output_dir)
    iteration = args.iteration
    state_file = output_dir / "optimization_state.json"

    print(f"\n{'=' * 70}")
    print(f"FEA PHASE - Iteration {iteration}")
    print(f"{'=' * 70}")

    # Load state (must exist from mesh phase)
    if not state_file.exists():
        raise RuntimeError(f"State file not found: {state_file}. Run mesh phase first.")

    with open(state_file, "r") as f:
        state = json.load(f)

    if not state.get("mesh_generated", False):
        raise RuntimeError("Mesh not generated. Run mesh phase first.")

    thickness_params = ThetaBandParams.from_dict(state["thickness_params"])
    history = state.get("history", [])

    # Restore momentum history (critical for stabilization across container runs)
    if "momentum_history" in state:
        set_momentum_history(state["momentum_history"])

    # Restore secant history (critical for secant method to accumulate data points)
    if "secant_history" in state:
        set_secant_history(state["secant_history"])
        print(f"Restored secant history: {len(state['secant_history'].get('history', []))} data points")

    # Restore OC/MMA history (critical for move limits and asymptotes across iterations)
    if "oc_history" in state:
        set_oc_history(state["oc_history"])
    if "mma_history" in state:
        set_mma_history(state["mma_history"])
    if "bo_history" in state:
        set_bo_history(state["bo_history"])
    if "pgd_history" in state:
        set_pgd_history(state["pgd_history"])
    if "cobyla_history" in state:
        set_cobyla_history(state["cobyla_history"])

    # Restore hole fillet factor from previous iteration
    # (critical: config is re-read from yaml each iteration, losing in-memory updates)
    if "hole_fillet_factor" in state:
        restored_ff = state["hole_fillet_factor"]
        config.setdefault("mesh", {}).setdefault("theta_overrides", {}).setdefault(
            "geometry", {})["hole_fillet_factor"] = restored_ff
        print(f"Restored hole fillet factor: {restored_ff:.3f}")

    print(f"Current thickness: {thickness_params.thickness_mm}")
    print(f"Current fracture factor: {thickness_params.fracture_factor}")

    # Update config with current thickness
    config_iter = _update_config_with_thickness(config, thickness_params)

    # Pass iteration number to FEA adapter for organized output
    config_iter["_current_iteration"] = iteration

    # Load existing mesh (already generated in mesh phase)
    print("\nLoading existing mesh...")
    mesh_output_dir = Path(state.get("mesh_output_dir", output_dir))
    mesh_xdmf = mesh_output_dir / "core_shell_refined.xdmf"

    if not mesh_xdmf.exists():
        raise RuntimeError(f"Mesh file not found: {mesh_xdmf}. Run mesh phase first.")

    print(f"  Loading: {mesh_xdmf}")
    mesh_data = _read_mesh_with_tags(mesh_xdmf)
    print(f"  Mesh: {mesh_data.n_cells} cells")

    # Run FEA
    print("\nRunning multi-orientation FEA...")
    density = np.ones(mesh_data.n_cells, dtype=float)
    fea_results = run_fea(mesh_data, density, config_iter)

    max_stress_mpa = fea_results["max_stress"] / 1e6
    worst_orientation = fea_results.get("worst_case_orientation", "N/A")
    print(f"  Worst-case: {worst_orientation} at {max_stress_mpa:.2f} MPa")

    # Print per-orientation results
    if "all_orientations" in fea_results:
        print("  All orientations:")
        for orient_result in fea_results["all_orientations"]:
            orient_name = orient_result["orientation"]
            orient_stress_mpa = orient_result["max_stress"] / 1e6
            print(f"    - {orient_name}: {orient_stress_mpa:.2f} MPa")

    # Print multi-orientation stress aggregation info (NEW - prevents oscillation)
    if "stress_aggregation" in fea_results:
        print("\n  Multi-orientation stress aggregation:")
        agg_info = fea_results["stress_aggregation"]
        print(f"    Method: {agg_info.get('method', 'unknown')}")
        print(f"    Orientations: {agg_info.get('n_orientations', '?')}")

        stress_weighted = fea_results.get("stress_by_band_MPa", [])
        stress_max = fea_results.get("stress_by_band_max_MPa", [])
        stress_worst = fea_results.get("stress_by_band_worst_MPa", [])

        if stress_weighted and stress_max:
            print(f"    Weighted avg (for FSD): {[f'{s:.0f}' for s in stress_weighted]}")
            print(f"    Max (for yield check): {[f'{s:.0f}' for s in stress_max]}")
            if stress_worst:
                print(f"    Worst-case only:       {[f'{s:.0f}' for s in stress_worst]}")

    # Update thickness
    # Pass previous Peridigm results so fracture-aware cap can block thickness increases
    # when fracture zone isn't breaching (prevents death spiral of ever-thicker shells)
    prev_peridigm = state.get("peridigm_results", None)
    opt_method = config.get("optimization", {}).get("optimization_method", "fsd")
    print(f"\nUpdating thickness ({opt_method.upper()})...")
    thickness_params, update_info = update_thickness(
        thickness_params, fea_results, mesh_data, config_iter,
        peridigm_results=prev_peridigm,
    )
    print(f"  Thickness change: {update_info['thickness_change_norm']:.4f} mm")
    print(f"  New thickness: {thickness_params.thickness_mm}")

    # Update hole fillet based on pole stress
    print("\nUpdating hole fillet...")
    hole_fillet_info = _update_hole_fillet(
        fea_results, thickness_params, config_iter
    )
    print(f"  Fillet factor: {hole_fillet_info.get('hole_fillet_factor', 0):.3f}")
    print(f"  Fillet size: {hole_fillet_info.get('hole_fillet_mm', 0):.2f} mm")

    # ====================================================================
    # INTEGRATED FRACTURE FACTOR UPDATE (FEA + Peridigm)
    # ====================================================================
    # Peridigm drives fracture_factor:
    #   - Breach → increase (thicken to resist crack)
    #   - Blocked at band → decrease that band (thin to allow crack)
    # FEA provides a FLOOR:
    #   - If stress in a band is near yield, don't thin further
    #   - fracture_factor must keep fracture zone < shell (max 0.99)
    # ====================================================================
    fracture_update_info = {}
    opt_cfg = config.get("optimization", {})
    mat_cfg = config.get("material", {})
    yield_stress_mpa = mat_cfg.get("yield_stress_Pa", 44e6) / 1e6
    min_fracture_factor = float(opt_cfg.get("min_fracture_factor", 0.2))
    max_fracture_factor = float(opt_cfg.get("max_fracture_factor", 0.99))
    use_crack_line = opt_cfg.get("use_crack_line_metrics", True)

    if opt_cfg.get("optimize_fracture_factor", True) and len(history) > 0:
        prev_entry = history[-1]
        if "D_fracture_zone" in prev_entry:
            print("\nUpdating fracture factors (absolute thickness mode)...")

            n_bands = len(thickness_params.fracture_factor)
            fracture_factor_before = thickness_params.fracture_factor.copy()
            protected_thickness = thickness_params.thickness_mm.copy()

            # --- ABSOLUTE FRACTURE THICKNESS ---
            # Work in absolute mm, not ratios. This prevents fracture thickness
            # from drifting when FEA changes protected thickness.
            #
            # 1. Recover target fracture thickness from previous iteration
            # 2. Apply Peridigm damage-driven adjustment in mm
            # 3. Recompute factor = target_frac_thick / current_protected_thick
            #
            # On first fracture update, initialize from current factor × thickness.
            # Look for target in top-level first, then inside fracture_update
            _prev_target = prev_entry.get("target_fracture_thickness_mm", None)
            if _prev_target is None:
                fu = prev_entry.get("fracture_update", {})
                _prev_target = fu.get("target_fracture_thickness_mm", None)
            if _prev_target is None:
                # First time: initialize from current factor × protected thickness
                _prev_target = (fracture_factor_before * protected_thickness).tolist()
            prev_frac_thick = np.array(_prev_target)
            frac_thick = prev_frac_thick.copy()

            print(f"  Previous fracture thickness (mm): [{','.join(f'{t:.2f}' for t in prev_frac_thick)}]")

            # --- Peridigm-driven fracture factor: MAXIMIZE FF while ensuring breach ---
            # Goal: fracture zone as thick as possible (FF close to 1.0) while still
            # fracturing. Uses bisection: breach → try thicker; no breach → thin.
            fracture_breach = prev_entry.get("fracture_breach", None)
            blocked_at_band = prev_entry.get("blocked_at_band", None)
            crack_depth = prev_entry.get("crack_depth", None)

            min_frac_thick = float(opt_cfg.get("min_fracture_thickness_mm", 0.4))
            max_frac_thick = float(opt_cfg.get("max_fracture_thickness_mm", 0.0))
            breach_continuity_threshold = float(opt_cfg.get("fracture_continuity_threshold", 0.8))

            # === Bisection state: per-band upper/lower FF bounds ===
            # ff_lower = highest FF known to achieve breach
            # ff_upper = lowest FF known to fail breach (starts at max_fracture_factor)
            prev_fu = prev_entry.get("fracture_update", {})
            ff_lower = np.array(prev_fu.get("ff_bisection_lower", [min_fracture_factor] * n_bands))
            ff_upper = np.array(prev_fu.get("ff_bisection_upper", [max_fracture_factor] * n_bands))

            crack_line = prev_entry.get("crack_line_metrics", None)
            per_band_damage = prev_entry.get("per_band_damage", None)

            # Get volumetric protected zone damage (diagnostic)
            prot_vol_damage = None
            if per_band_damage is not None:
                prot_vol_damage = per_band_damage.get("prot_damaged_fraction", None)

            if crack_line is not None and use_crack_line:
                frac_continuity = crack_line["fracture_zone"]["per_band_continuity"]
                n_particles = crack_line["fracture_zone"]["per_band_n_particles"]
                prot_continuity = crack_line["protected_zone"]["per_band_continuity"]

                print(f"  [FF Bisection] Maximize FF while ensuring breach (threshold={breach_continuity_threshold})")
                for b in range(n_bands):
                    if b < len(n_particles) and n_particles[b] == 0:
                        # No fracture particles → can't evaluate, keep current
                        print(f"    Band {b}: SKIP (no fracture particles)")
                        continue

                    fc = frac_continuity[b] if b < len(frac_continuity) else 0.0
                    current_ff = fracture_factor_before[b]
                    pv = prot_vol_damage[b] if prot_vol_damage and b < len(prot_vol_damage) else 0.0

                    if fc >= breach_continuity_threshold:
                        # Check if protected zone is also destroyed — if so, this is
                        # indiscriminate shattering, not selective fracture.
                        # Don't let bisection increase FF when we can't distinguish zones.
                        prot_damage_limit = 0.5  # >50% protected zone destroyed = invalid breach
                        if pv > prot_damage_limit:
                            # Invalid breach: protected zone also shattered
                            # Hold FF steady — can't determine if fracture zone is too thick or too thin
                            new_ff = current_ff
                            print(f"    Band {b}: cont={fc:.0%} BREACH but prot_vol={pv:.1%} > {prot_damage_limit:.0%} → HOLD FF {current_ff:.3f} (indiscriminate)")
                        else:
                            # Valid breach: fracture zone broke, protected survived
                            ff_lower[b] = max(ff_lower[b], current_ff)
                            # Bisect upward: try thicker fracture zone
                            new_ff = (current_ff + ff_upper[b]) / 2.0
                            print(f"    Band {b}: cont={fc:.0%} BREACH → FF {current_ff:.3f} → {new_ff:.3f} (↑ thicken, prot_vol={pv:.1%})")
                    else:
                        # Breach failed → current FF is too high
                        ff_upper[b] = min(ff_upper[b], current_ff)
                        # Bisect downward: try thinner fracture zone
                        new_ff = (ff_lower[b] + current_ff) / 2.0
                        print(f"    Band {b}: cont={fc:.0%} NO BREACH → FF {current_ff:.3f} → {new_ff:.3f} (↓ thin, prot_vol={pv:.1%})")

                    # Set new fracture thickness from bisected FF
                    frac_thick[b] = new_ff * protected_thickness[b]

                # Log protected zone volumetric damage
                if prot_vol_damage:
                    for b in range(n_bands):
                        pv = prot_vol_damage[b] if b < len(prot_vol_damage) else 0.0
                        if pv > 0.05:
                            print(f"    Band {b}: WARN protected zone vol damage = {pv:.1%}")

            elif per_band_damage is not None:
                # === FALLBACK: volumetric damage fraction ===
                frac_vol = per_band_damage.get("frac_damaged_fraction", [0.0] * n_bands)
                frac_vol_threshold = float(opt_cfg.get("frac_vol_breach_threshold", 0.5))

                print(f"  [FF Bisection] Using volumetric damage (threshold={frac_vol_threshold})")
                for b in range(n_bands):
                    n_frac = per_band_damage.get("n_frac_particles", [])
                    if n_frac and b < len(n_frac) and n_frac[b] == 0:
                        print(f"    Band {b}: SKIP (no fracture particles)")
                        continue

                    fv = frac_vol[b] if b < len(frac_vol) else 0.0
                    current_ff = fracture_factor_before[b]

                    if fv >= frac_vol_threshold:
                        ff_lower[b] = max(ff_lower[b], current_ff)
                        new_ff = (current_ff + ff_upper[b]) / 2.0
                        print(f"    Band {b}: frac_vol={fv:.1%} BREACH → FF {current_ff:.3f} → {new_ff:.3f} (↑)")
                    else:
                        ff_upper[b] = min(ff_upper[b], current_ff)
                        new_ff = (ff_lower[b] + current_ff) / 2.0
                        print(f"    Band {b}: frac_vol={fv:.1%} NO BREACH → FF {current_ff:.3f} → {new_ff:.3f} (↓)")

                    frac_thick[b] = new_ff * protected_thickness[b]

            elif fracture_breach is True:
                # Global breach detected → try thickening everywhere
                for b in range(n_bands):
                    ff_lower[b] = max(ff_lower[b], fracture_factor_before[b])
                    new_ff = (fracture_factor_before[b] + ff_upper[b]) / 2.0
                    frac_thick[b] = new_ff * protected_thickness[b]
                print(f"  [FF Bisection] Global breach → thickening all bands")

            elif fracture_breach is False:
                # Global no-breach → thin everywhere
                for b in range(n_bands):
                    ff_upper[b] = min(ff_upper[b], fracture_factor_before[b])
                    new_ff = (ff_lower[b] + fracture_factor_before[b]) / 2.0
                    frac_thick[b] = new_ff * protected_thickness[b]
                print(f"  [FF Bisection] No breach → thinning all bands")

            else:
                print(f"  [FF Bisection] No Peridigm data yet — no adjustment")

            # Enforce min/max fracture thickness
            frac_thick = np.maximum(frac_thick, min_frac_thick)
            if max_frac_thick > 0:
                frac_thick = np.minimum(frac_thick, max_frac_thick)

            # --- Recompute fracture factor from absolute thickness ---
            fracture_factor = frac_thick / protected_thickness
            fracture_factor = np.clip(fracture_factor, min_fracture_factor, max_fracture_factor)

            # Re-derive actual fracture thickness after factor clamping
            frac_thick_actual = fracture_factor * protected_thickness

            thickness_params.fracture_factor = fracture_factor

            print(f"  Fracture thickness (mm): [{','.join(f'{t:.2f}' for t in frac_thick_actual)}]")

            fracture_update_info = {
                "fracture_factor_old": fracture_factor_before.tolist(),
                "fracture_factor_new": fracture_factor.tolist(),
                "fracture_factor_change": (fracture_factor - fracture_factor_before).tolist(),
                "target_fracture_thickness_mm": frac_thick_actual.tolist(),
                "ff_bisection_lower": ff_lower.tolist(),
                "ff_bisection_upper": ff_upper.tolist(),
                "path_breach": fracture_breach,
                "blocked_at_band": blocked_at_band,
                "crack_depth": crack_depth,
                "D_fracture_zone": prev_entry.get("D_fracture_zone", 0.0),
                "D_shell_body": prev_entry.get("D_shell_body", 0.0),
                "used_crack_line_metrics": crack_line is not None and use_crack_line,
            }

            print(f"  Fracture factors: {[f'{f:.2f}' for f in fracture_factor]}")
            change = fracture_factor - fracture_factor_before
            if np.any(change != 0):
                print(f"  Changes:          {[f'{c:+.2f}' for c in change]}")
        else:
            print("\n[Fracture] No previous Peridigm results - skipping fracture factor update")
    elif len(history) == 0:
        print("\n[Fracture] First iteration - skipping fracture factor update")

    # Check convergence
    converged, reasons = check_convergence(
        thickness_params, fea_results, update_info, history, config_iter,
        peridigm_results=prev_peridigm,
    )

    # Build history entry
    mean_thickness_mm = float(np.mean(thickness_params.thickness_mm))
    volume_relative = mean_thickness_mm / float(opt_cfg.get("initial_thickness_mm", 1.0))

    history_entry = {
        "iteration": iteration,
        "thickness_mm": thickness_params.thickness_mm.tolist(),
        "fracture_factor": thickness_params.fracture_factor.tolist(),
        "fracture_thickness_mm": thickness_params.fracture_thickness_mm.tolist(),
        "mean_thickness_mm": mean_thickness_mm,
        "volume_relative": volume_relative,
        "max_stress_MPa": max_stress_mpa,
        "worst_case_orientation": worst_orientation,
        # Multi-orientation stress aggregation (NEW)
        # stress_by_band_MPa: Weighted average across orientations (used for FSD)
        # stress_by_band_max_MPa: Max across orientations (used for yield check)
        "stress_by_band_MPa": update_info.get("stress_by_band_MPa", []),
        "stress_by_band_max_MPa": fea_results.get("stress_by_band_max_MPa", []),
        "stress_by_band_worst_MPa": fea_results.get("stress_by_band_worst_MPa", []),
        "stress_aggregation": fea_results.get("stress_aggregation", {}),
        "thickness_change_norm": update_info["thickness_change_norm"],
        "thickness_change_mm": update_info.get("thickness_change_mm", []),
        "converged": converged,
        "convergence_reasons": reasons,
        # Momentum info for debugging stabilization
        "momentum": update_info.get("momentum", {"enabled": False}),
        # Hole fillet optimization
        "hole_fillet_factor": hole_fillet_info.get("hole_fillet_factor", 0.0),
        "hole_fillet_mm": hole_fillet_info.get("hole_fillet_mm", 0.0),
    }

    # Add fracture update info if available
    if fracture_update_info:
        history_entry["fracture_update"] = fracture_update_info
        # Store target fracture thickness at top level for easy access next iteration
        if "target_fracture_thickness_mm" in fracture_update_info:
            history_entry["target_fracture_thickness_mm"] = fracture_update_info["target_fracture_thickness_mm"]

    if "all_orientations" in fea_results:
        history_entry["all_orientations"] = [
            {"orientation": r["orientation"], "max_stress_MPa": r["max_stress"] / 1e6}
            for r in fea_results["all_orientations"]
        ]

    history.append(history_entry)

    # Update state (include momentum, secant, OC, MMA history for continuation)
    state.update({
        "iteration": iteration,
        "thickness_params": thickness_params.to_dict(),
        "history": history,
        "momentum_history": get_momentum_history(),
        "secant_history": get_secant_history(),
        "oc_history": get_oc_history(),
        "mma_history": get_mma_history(),
        "bo_history": get_bo_history(),
        "pgd_history": get_pgd_history(),
        "cobyla_history": get_cobyla_history(),
        "hole_fillet_factor": hole_fillet_info.get("hole_fillet_factor", 0.0),
        "hole_fillet_mm": hole_fillet_info.get("hole_fillet_mm", 0.0),
        "fea_results": {
            "max_stress": float(fea_results["max_stress"]),
            "worst_case_orientation": worst_orientation,
        },
        "fea_completed": True,
        "converged": converged,
    })

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    # Save intermediate results
    _save_results(output_dir, thickness_params, history, iteration - 1)

    print(f"\nFEA Phase complete.")
    print(f"Converged: {converged}")
    if converged:
        print(f"Reasons: {', '.join(reasons)}")


if __name__ == "__main__":
    main()
