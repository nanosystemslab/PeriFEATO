"""
Thickness Update Logic
======================

Update shell thickness based on FEA stress distribution and convergence criteria.

Update strategy:
- Increase thickness where stress > stress_limit (critical regions)
- Decrease thickness where stress << stress_limit (over-designed regions)
- Use gradual updates with learning rate to ensure stability
- **Momentum-based stabilization**: Track thickness change history per band and
  dampen direction reversals to prevent oscillation

Momentum Algorithm:
- Track the exponential moving average (EMA) of thickness changes per band
- When the proposed change reverses direction from the EMA trend, apply damping
- This prevents "flip-flopping" where a band alternates between thickening/thinning
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from .theta_band_params import ThetaBandParams


# Module-level storage for momentum history (persists across calls)
_momentum_history: Dict[str, np.ndarray] = {}

# Module-level storage for fracture bisection history
_fracture_bisection_history: Dict[str, any] = {}

# Module-level storage for secant method history (thickness-stress pairs per band)
_secant_history: Dict[str, any] = {}

# Module-level storage for OC method history
_oc_history: Dict[str, any] = {}

# Module-level storage for MMA method history
_mma_history: Dict[str, any] = {}

# Module-level storage for Bayesian Optimization history
_bo_history: Dict[str, any] = {}

# Module-level storage for Projected Gradient Descent history
_pgd_history: Dict[str, any] = {}

# Module-level storage for COBYLA history
_cobyla_history: Dict[str, any] = {}


def update_thickness(
    thickness_params: ThetaBandParams,
    fea_results: Dict,
    mesh_data,
    config: Dict,
    peridigm_results: Dict = None,
) -> Tuple[ThetaBandParams, Dict]:
    """
    Update shell thickness based on FEA stress results.

    Args:
        thickness_params: Current thickness parameterization
        fea_results: FEA results with stress field and metadata
        mesh_data: Mesh data with cell centers and regions
        config: Optimization configuration

    Returns:
        Tuple of:
        - Updated thickness parameters
        - Update info dictionary (for logging/convergence)

    Update logic:
        1. Compute average stress per theta band
        2. For each band:
           - If stress > limit: increase thickness
           - If stress < limit * safety_margin: decrease thickness
        3. Apply learning rate and enforce bounds
    """
    opt_cfg = config.get("optimization", {})
    fea_cfg = config.get("fea", {})

    # Optimization parameters
    relaxation_base = float(opt_cfg.get("thickness_relaxation", 0.3))  # Base blend factor (0-1)
    stress_limit_pa = float(fea_cfg.get("stress_limit_Pa", 44e6))
    target_sf = float(opt_cfg.get("target_safety_factor", 2.0))
    thickness_resolution_mm = float(opt_cfg.get("thickness_resolution_mm", 0.05))  # Round to this
    stress_exponent = float(opt_cfg.get("stress_thickness_exponent", 0.5))  # 0.5 for bending, 1.0 for membrane

    # Momentum parameters (for stabilization)
    momentum_beta = float(opt_cfg.get("momentum_beta", 0.7))  # EMA smoothing (0=no memory, 1=full memory)
    direction_damping = float(opt_cfg.get("direction_damping", 0.5))  # Damping when reversing direction (0=none, 1=full block)
    use_momentum = bool(opt_cfg.get("use_momentum", True))  # Enable/disable momentum

    # === ACCELERATION FEATURES ===
    # Stress diffusion: smooth stress signal across neighbor bands before FSD update.
    # This makes FSD aware of inter-band coupling (stress redistribution from stiffness
    # mismatch). Without it, FSD greedily thickens one band, creating a stiffness cliff
    # that pushes stress to neighbors. With diffusion, neighbors "feel" the overstress
    # and thicken proportionally, producing a natural taper instead of a cliff.
    fsd_stress_diffusion = float(opt_cfg.get("fsd_stress_diffusion", 0.0))  # 0=off, 0.4=recommended
    # Update smoothing: Laplacian regularization of the FSD thickness change.
    # Prevents abrupt thickness jumps between adjacent bands.
    fsd_update_smoothing = float(opt_cfg.get("fsd_update_smoothing", 0.0))  # 0=off, 0.3=recommended

    # Secant method: Use 2 data points to extrapolate optimal thickness (Newton-like)
    use_secant_method = bool(opt_cfg.get("use_secant_method", False))

    # Adaptive relaxation: Aggressive when far from target, conservative when close
    use_adaptive_relaxation = bool(opt_cfg.get("use_adaptive_relaxation", False))
    adaptive_relaxation_thresholds = opt_cfg.get("adaptive_relaxation_thresholds", {
        "far": {"error": 0.5, "relaxation": 0.9},      # >50% error: 90% relaxation
        "medium": {"error": 0.2, "relaxation": 0.6},  # 20-50% error: 60% relaxation
        "close": {"error": 0.0, "relaxation": 0.3},   # <20% error: 30% relaxation
    })

    # Target stress = yield / SF
    target_stress_pa = stress_limit_pa / target_sf

    # Extract stress by theta band
    stress_by_band = _compute_stress_by_theta_band(
        fea_results,
        mesh_data,
        thickness_params,
        opt_cfg,
    )

    # Get material bounds
    material_cfg = config.get("material", {})
    min_thickness = float(material_cfg.get("min_thickness_mm", 1.0))
    max_thickness = float(material_cfg.get("max_thickness_mm", 10.0))

    # Compute thickness updates
    thickness_old = thickness_params.thickness_mm.copy()

    # === OPTIMIZATION METHOD DISPATCH ===
    optimization_method = str(opt_cfg.get("optimization_method", "fsd")).strip().lower()

    if optimization_method == "oc":
        # Optimality Criteria method (KKT-derived FSD with move limits)
        oc_move_limit = float(opt_cfg.get("oc_move_limit", 0.3))
        oc_move_decay = float(opt_cfg.get("oc_move_decay", 0.95))
        oc_move_floor = float(opt_cfg.get("oc_move_floor", 0.05))

        thickness_new_raw, method_info = _compute_oc_update(
            thickness_old=thickness_old,
            stress_by_band=stress_by_band,
            target_stress_pa=target_stress_pa,
            stress_exponent=stress_exponent,
            min_thickness=min_thickness,
            max_thickness=max_thickness,
            oc_move_limit=oc_move_limit,
            oc_move_decay=oc_move_decay,
            oc_move_floor=oc_move_floor,
        )
        momentum_info = {"enabled": False}
        print(f"[Thickness Update] Using OC method (move_limit={method_info['move_limit_used']:.3f})")

    elif optimization_method == "mma":
        # Method of Moving Asymptotes (Svanberg 1987)
        mma_asymptote_init = float(opt_cfg.get("mma_asymptote_init", 0.5))
        mma_oscillation_shrink = float(opt_cfg.get("mma_oscillation_shrink", 0.7))
        mma_monotone_expand = float(opt_cfg.get("mma_monotone_expand", 1.2))

        thickness_new_raw, method_info = _compute_mma_update(
            thickness_old=thickness_old,
            stress_by_band=stress_by_band,
            target_stress_pa=target_stress_pa,
            stress_exponent=stress_exponent,
            min_thickness=min_thickness,
            max_thickness=max_thickness,
            mma_asymptote_init=mma_asymptote_init,
            mma_oscillation_shrink=mma_oscillation_shrink,
            mma_monotone_expand=mma_monotone_expand,
        )
        momentum_info = {"enabled": False}
        print(f"[Thickness Update] Using MMA method (iteration={method_info['n_iterations']})")

    elif optimization_method == "bo":
        # Bayesian Optimization (GP surrogate + Expected Improvement)
        bo_length_scale = float(opt_cfg.get("bo_length_scale", 1.0))
        bo_noise = float(opt_cfg.get("bo_noise", 1e-4))
        bo_xi = float(opt_cfg.get("bo_exploration_xi", 0.01))
        bo_n_random = int(opt_cfg.get("bo_n_random_init", 3))

        thickness_new_raw, method_info = _compute_bo_update(
            thickness_old=thickness_old,
            stress_by_band=stress_by_band,
            target_stress_pa=target_stress_pa,
            stress_exponent=stress_exponent,
            min_thickness=min_thickness,
            max_thickness=max_thickness,
            length_scale=bo_length_scale,
            noise=bo_noise,
            xi=bo_xi,
            n_random_init=bo_n_random,
        )
        momentum_info = {"enabled": False}
        print(f"[Thickness Update] Using BO method (observations={method_info['n_observations']})")

    elif optimization_method == "pgd":
        # Projected Gradient Descent with backtracking line search
        pgd_initial_step = float(opt_cfg.get("pgd_initial_step", 1.0))
        pgd_backtrack = float(opt_cfg.get("pgd_backtrack_factor", 0.5))
        pgd_min_step = float(opt_cfg.get("pgd_min_step", 0.01))

        thickness_new_raw, method_info = _compute_pgd_update(
            thickness_old=thickness_old,
            stress_by_band=stress_by_band,
            target_stress_pa=target_stress_pa,
            stress_exponent=stress_exponent,
            min_thickness=min_thickness,
            max_thickness=max_thickness,
            initial_step=pgd_initial_step,
            backtrack_factor=pgd_backtrack,
            min_step=pgd_min_step,
        )
        momentum_info = {"enabled": False}
        print(f"[Thickness Update] Using PGD method (step={method_info['step_size_used']:.4f})")

    elif optimization_method == "cobyla":
        # COBYLA: derivative-free trust-region with linear models (Powell 1994)
        cobyla_rhobeg = float(opt_cfg.get("cobyla_rhobeg", 0.5))
        cobyla_rhoend = float(opt_cfg.get("cobyla_rhoend", 0.01))
        cobyla_maxiter = int(opt_cfg.get("cobyla_maxiter", 1000))

        thickness_new_raw, method_info = _compute_cobyla_update(
            thickness_old=thickness_old,
            stress_by_band=stress_by_band,
            target_stress_pa=target_stress_pa,
            stress_exponent=stress_exponent,
            min_thickness=min_thickness,
            max_thickness=max_thickness,
            rhobeg=cobyla_rhobeg,
            rhoend=cobyla_rhoend,
            maxiter=cobyla_maxiter,
        )
        momentum_info = {"enabled": False}
        print(f"[Thickness Update] Using COBYLA method (trust_radius={method_info['final_trust_radius']:.4f})")

    else:
        # === FSD METHOD (default) ===
        method_info = {"method": "fsd"}

        # === SECANT METHOD (Newton-like, uses 2 data points) ===
        secant_info = {"used": False}
        thickness_secant = None

        if use_secant_method:
            thickness_secant, secant_info = _apply_secant_method(
                thickness_old=thickness_old,
                stress_by_band=stress_by_band,
                target_stress_pa=target_stress_pa,
                stress_exponent=stress_exponent,
                min_thickness=min_thickness,
                max_thickness=max_thickness,
            )

        # === ADAPTIVE RELAXATION ===
        # Compute average stress error to determine relaxation
        if use_adaptive_relaxation:
            avg_stress_ratio = np.mean(stress_by_band) / target_stress_pa
            max_stress_ratio = np.max(stress_by_band) / target_stress_pa
            # Use the worse of avg or max to determine error level
            stress_error = max(abs(avg_stress_ratio - 1.0), abs(max_stress_ratio - 1.0))

            thresholds = adaptive_relaxation_thresholds
            if stress_error > thresholds["far"]["error"]:
                relaxation = thresholds["far"]["relaxation"]
                error_level = "far"
            elif stress_error > thresholds["medium"]["error"]:
                relaxation = thresholds["medium"]["relaxation"]
                error_level = "medium"
            else:
                relaxation = thresholds["close"]["relaxation"]
                error_level = "close"

            print(f"[Adaptive Relaxation] Error={stress_error:.2f} ({error_level}) → relaxation={relaxation:.2f}")
        else:
            relaxation = relaxation_base

        # === BAND VOLUME WEIGHTS ===
        # Bands near the pole cover tiny surface area; equator bands cover large area.
        # Volume-weighted FSD uses per-band relaxation that is inversely proportional
        # to the band's area fraction — small bands (pole) get higher relaxation
        # because thickening them is "cheap" in total volume.
        use_volume_weights = bool(opt_cfg.get("use_band_volume_weights", False))
        volume_weight_exponent = float(opt_cfg.get("volume_weight_exponent", 0.5))

        if use_volume_weights:
            area_fractions = thickness_params.compute_band_area_weights()
            mean_area = np.mean(area_fractions)
            # Per-band relaxation: boost for small bands, reduce for large bands
            # relaxation_i = base * (mean_area / area_i)^exponent, capped at 0.95
            relaxation_per_band = relaxation * (mean_area / np.maximum(area_fractions, 1e-10)) ** volume_weight_exponent
            relaxation_per_band = np.clip(relaxation_per_band, 0.1, 0.95)
            print(f"[Volume Weights] Area fractions: [{', '.join(f'{a:.3f}' for a in area_fractions)}]")
            print(f"[Volume Weights] Relaxation/band: [{', '.join(f'{r:.2f}' for r in relaxation_per_band)}]")
        else:
            relaxation_per_band = np.full(thickness_params.n_bands, relaxation)

        # Compute Fully Stressed Design (FSD) target with MATERIAL MINIMIZATION
        # Three cases:
        # 1. stress > target: OVERSTRESSED - thicken to reduce stress
        # 2. safe_threshold * target < stress <= target: OPTIMAL - leave alone
        # 3. stress <= safe_threshold * target: OVER-DESIGNED - thin to save material
        thickness_fsd = np.zeros(thickness_params.n_bands)

        # safe_stress_threshold controls material minimization behavior:
        # - 1.0 = never thin (only thicken overstressed, leave safe alone)
        # - 0.85 = thin bands with stress < 85% of target (material minimization)
        # - 0.0 = always apply FSD (aggressive thinning toward uniform stress)
        safe_threshold = opt_cfg.get("safe_stress_threshold", 1.0)

        # --- FSD-G: Stress diffusion (neighbor-aware stress signal) ---
        # Convolve stress with a 3-point kernel so each band "feels" neighbor stress.
        # This prevents the cliff pattern where FSD slams one band to max while
        # neighbors stay at min, because neighbors now see elevated stress too.
        stress_for_fsd = stress_by_band.copy()
        if fsd_stress_diffusion > 0 and thickness_params.n_bands >= 3:
            w_self = 1.0 - fsd_stress_diffusion
            w_adj = fsd_stress_diffusion / 2.0
            diffused = np.zeros_like(stress_by_band)
            for i in range(thickness_params.n_bands):
                left = stress_by_band[max(0, i - 1)]
                right = stress_by_band[min(thickness_params.n_bands - 1, i + 1)]
                diffused[i] = w_self * stress_by_band[i] + w_adj * (left + right)
            stress_for_fsd = diffused
            print(f"[FSD-G] Stress diffusion (w={fsd_stress_diffusion:.2f}): "
                  f"raw_max={max(stress_by_band)/1e6:.1f} MPa → diffused_max={max(diffused)/1e6:.1f} MPa")

        for band_idx in range(thickness_params.n_bands):
            stress_pa = stress_for_fsd[band_idx]
            t_current = thickness_old[band_idx]

            if stress_pa > 0:
                stress_ratio = stress_pa / target_stress_pa

                if stress_ratio > 1.0:
                    # OVERSTRESSED: need to thicken to reduce stress
                    # FSD: t_new = t_current * (σ_current / σ_target)^exponent
                    thickness_fsd[band_idx] = t_current * (stress_ratio ** stress_exponent)
                elif stress_ratio > safe_threshold:
                    # OPTIMAL: stress is close to target - leave thickness alone
                    # This prevents oscillation when near the goal
                    thickness_fsd[band_idx] = t_current
                else:
                    # OVER-DESIGNED: stress is well below target - thin to save material
                    # FSD thinning: same formula, but stress_ratio < 1 means thickness decreases
                    thickness_fsd[band_idx] = t_current * (stress_ratio ** stress_exponent)
            else:
                # No stress data for this band - keep current thickness
                thickness_fsd[band_idx] = t_current

        # Choose target thickness: secant prediction (if available) or FSD
        if thickness_secant is not None and secant_info.get("used", False):
            # Use secant method prediction (physics-based extrapolation from 2 data points)
            thickness_target = thickness_secant
            print(f"[Thickness Update] Using SECANT method prediction")
        else:
            # Use traditional FSD
            thickness_target = thickness_fsd
            print(f"[Thickness Update] Using FSD method")

        # Relaxation: blend current thickness with target (per-band when volume-weighted)
        thickness_new_raw = (1.0 - relaxation_per_band) * thickness_old + relaxation_per_band * thickness_target

        # --- FSD-G: Laplacian update smoothing ---
        # Smooth the thickness CHANGE (not thickness itself) across neighbors.
        # Prevents abrupt jumps: if band 0 wants +5mm but band 1 wants +0mm,
        # smoothing distributes some growth to band 1, producing a taper.
        if fsd_update_smoothing > 0 and thickness_params.n_bands >= 3:
            delta_raw = thickness_new_raw - thickness_old
            delta_smooth = delta_raw.copy()
            lam = fsd_update_smoothing
            for i in range(thickness_params.n_bands):
                left = delta_raw[max(0, i - 1)]
                right = delta_raw[min(thickness_params.n_bands - 1, i + 1)]
                delta_smooth[i] = (1.0 - lam) * delta_raw[i] + lam / 2.0 * (left + right)
            thickness_new_raw = thickness_old + delta_smooth
            print(f"[FSD-G] Update smoothing (λ={lam:.2f}): "
                  f"Δ_max raw={max(abs(delta_raw)):.2f}mm → smooth={max(abs(delta_smooth)):.2f}mm")

        # Stochastic perturbation to break limit cycles (fixed seed for reproducibility)
        # Iteration count derived from thickness_params which tracks history length
        perturbation_scale = float(opt_cfg.get("fsd_perturbation_scale", 0.0))
        if perturbation_scale > 0:
            perturbation_seed = int(opt_cfg.get("fsd_perturbation_seed", 42))
            perturbation_decay = float(opt_cfg.get("fsd_perturbation_decay", 0.98))
            iteration = int(config.get("_current_iteration", opt_cfg.get("_current_iteration", 0)))

            # Decaying amplitude: large early, small when close to solution
            amplitude = perturbation_scale * (perturbation_decay ** iteration)

            # Deterministic per-iteration jitter (seed + iteration = unique but reproducible)
            rng = np.random.RandomState(perturbation_seed + iteration)
            jitter = rng.uniform(-1.0, 1.0, size=len(thickness_new_raw))

            # Scale jitter by current thickness delta magnitude (proportional to update size)
            delta_magnitude = np.abs(thickness_new_raw - thickness_old)
            perturbation = amplitude * jitter * np.maximum(delta_magnitude, 0.05)

            thickness_new_raw = thickness_new_raw + perturbation
            print(f"[FSD Perturbation] iter={iteration}, amplitude={amplitude:.4f}, "
                  f"max_jitter={np.max(np.abs(perturbation)):.4f}mm")

        # Compute proposed delta before momentum
        delta_proposed = thickness_new_raw - thickness_old

        # Apply momentum-based stabilization
        if use_momentum:
            delta_stabilized, momentum_info = _apply_momentum_stabilization(
                delta_proposed=delta_proposed,
                n_bands=thickness_params.n_bands,
                momentum_beta=momentum_beta,
                direction_damping=direction_damping,
            )
            thickness_new_raw = thickness_old + delta_stabilized
        else:
            momentum_info = {"enabled": False}

        method_info["secant"] = secant_info if use_secant_method else {"used": False}
        method_info["adaptive_relaxation"] = use_adaptive_relaxation
        method_info["relaxation_used"] = relaxation

    # Round to manufacturing resolution (e.g., 0.05mm)
    thickness_new_rounded = np.round(thickness_new_raw / thickness_resolution_mm) * thickness_resolution_mm

    # Compute delta and apply
    delta_mm = thickness_new_rounded - thickness_old

    # Fixed bands: zero out delta for bands that shouldn't be optimized
    # (e.g., bands 0-1 near the pole hole where thickness doesn't contribute to mesh)
    fixed_band_indices = opt_cfg.get("fixed_band_indices", [])
    for band_idx in fixed_band_indices:
        if 0 <= band_idx < len(delta_mm):
            delta_mm[band_idx] = 0.0

    # FRACTURE-AWARE THICKNESS CAP
    # If fracture zone is not breaking (D_frac < target), don't allow thickness increases.
    # This prevents the algorithm from thickening the shell indefinitely when fracture fails.
    # Thicker protected shell = stiffer structure = harder for fracture zone to break.
    fracture_cap_enabled = bool(opt_cfg.get("fracture_aware_cap", True))
    if fracture_cap_enabled and peridigm_results is not None:
        # Prefer crack_line_metrics (all-timestep continuity) over per_band_damage
        crack_line = peridigm_results.get("crack_line_metrics")
        use_crack_line = bool(opt_cfg.get("use_crack_line_metrics", True))

        if crack_line is not None and use_crack_line:
            # Crack line cap: block increases on bands with low fracture score
            frac_scores = crack_line["fracture_zone"]["per_band_score"]
            n_part = crack_line["fracture_zone"]["per_band_n_particles"]
            score_threshold = float(opt_cfg.get("fracture_score_threshold", 0.5))
            n_capped = 0
            for i in range(len(delta_mm)):
                if i < len(n_part) and n_part[i] == 0:
                    continue  # No particles in band — skip
                if i < len(frac_scores) and frac_scores[i] < score_threshold and delta_mm[i] > 0:
                    delta_mm[i] = 0.0  # Block increase
                    n_capped += 1

            if n_capped > 0:
                print(f"[Fracture Cap] Crack line: blocked increases on {n_capped} bands with score < {score_threshold}")
                print(f"[Fracture Cap] Frac scores: {[f'{s:.3f}' for s in frac_scores]}")
        else:
            # Fallback: per_band_damage or global D_frac
            per_band_damage = peridigm_results.get("per_band_damage")
            target_D_frac = float(opt_cfg.get("target_D_fracture", 0.3))
            use_per_band = bool(opt_cfg.get("use_per_band_damage", False))

            if per_band_damage is not None and use_per_band:
                # Per-band cap: only block increases on bands where fracture isn't breaking
                D_frac_bands = np.array(per_band_damage["worst_frac_damage"])
                n_frac = per_band_damage.get("n_frac_particles", [])
                n_capped = 0
                for i in range(len(delta_mm)):
                    # Skip bands with no fracture particles (no data, not zero damage)
                    if n_frac and i < len(n_frac) and n_frac[i] == 0:
                        continue
                    if i < len(D_frac_bands) and D_frac_bands[i] < target_D_frac and delta_mm[i] > 0:
                        delta_mm[i] = 0.0  # Block increase on this band
                        n_capped += 1

                if n_capped > 0:
                    print(f"[Fracture Cap] Per-band: blocked increases on {n_capped} bands with D_frac < {target_D_frac}")
                    print(f"[Fracture Cap] D_frac per band: {[f'{d:.3f}' for d in D_frac_bands]}")
            else:
                # Global cap: use overall D_fracture_zone (fallback)
                D_frac = float(peridigm_results.get("D_fracture_zone", 1.0))

                if D_frac < target_D_frac:
                    n_capped = 0
                    for i in range(len(delta_mm)):
                        if delta_mm[i] > 0:
                            delta_mm[i] = 0.0  # Block increase
                            n_capped += 1

                    if n_capped > 0:
                        print(f"[Fracture Cap] D_frac={D_frac:.3f} < target={target_D_frac:.3f}")
                        print(f"[Fracture Cap] Blocked thickness increases on {n_capped} bands")

    # Apply update (with bounds enforcement in ThetaBandParams)
    thickness_params.update_thickness(delta_mm)

    # After changing protected thickness, enforce fracture zone thickness constraints
    # (fracture_factor × thickness >= min_fracture_thickness_mm)
    min_fracture_thickness_mm = float(opt_cfg.get("min_fracture_thickness_mm", 0.4))
    thickness_params.enforce_min_fracture_thickness(min_fracture_thickness_mm)

    # Cap fracture zone thickness to prevent unbreakable zones when shell thickens
    # (fracture_factor × thickness <= max_fracture_thickness_mm)
    max_fracture_thickness_mm = float(opt_cfg.get("max_fracture_thickness_mm", 0.0))
    if max_fracture_thickness_mm > 0:
        thickness_params.enforce_max_fracture_thickness(max_fracture_thickness_mm)

    # Enforce smoothness constraint to prevent extreme thickness variations between adjacent bands
    # This prevents "bulging" designs where one band is much thicker than its neighbors
    max_gradient_mm = float(opt_cfg.get("max_thickness_gradient_mm", 0.0))
    # lift_mode: If True, lift thin bands to accommodate thick bands (bidirectional)
    # This allows overstressed regions to get thicker without being clamped by gradient
    gradient_lift_mode = bool(opt_cfg.get("gradient_lift_mode", True))
    if max_gradient_mm > 0:
        thickness_params.enforce_smoothness(max_gradient_mm, lift_mode=gradient_lift_mode)

    # Enforce monotone-decreasing profile: pole (band 0) thickest, equator thinnest.
    # Prevents spline undershoot from non-monotone profiles (e.g. COBYLA [1.7, 10, 1.2×8]).
    enforce_monotone = opt_cfg.get("enforce_monotone_thickness",
                                   opt_cfg.get("use_spline_thickness", False))
    if enforce_monotone:
        t = thickness_params.thickness_mm
        for i in range(1, len(t)):
            if t[i] > t[i - 1]:
                t[i] = t[i - 1]
        thickness_params.thickness_mm = t
        print(f"[Monotone] Enforced: {np.array2string(t, precision=2)}")

    thickness_new = thickness_params.thickness_mm

    # Compute change metrics
    thickness_change = thickness_new - thickness_old
    thickness_change_norm = float(np.linalg.norm(thickness_change))
    max_stress_pa = float(np.max(stress_by_band))

    # Build update info
    update_info = {
        "thickness_change_norm": thickness_change_norm,
        "thickness_change_mm": thickness_change.tolist(),
        "thickness_old_mm": thickness_old.tolist(),
        "thickness_new_mm": thickness_new.tolist(),
        "stress_by_band_MPa": (stress_by_band / 1e6).tolist(),
        "max_stress_MPa": max_stress_pa / 1e6,
        "max_stress_band_idx": int(np.argmax(stress_by_band)),
        "stress_band_stat": opt_cfg.get("stress_band_stat", "max"),
        "stress_band_percentile": float(opt_cfg.get("stress_band_percentile", 99.0)),
        "optimization_method": optimization_method,
        "method_info": method_info,
        "momentum": momentum_info,
        # Acceleration features (FSD-specific, kept for backward compat)
        "acceleration": {
            "secant_method": method_info.get("secant", {"used": False}),
            "adaptive_relaxation": method_info.get("adaptive_relaxation", False),
            "relaxation_used": method_info.get("relaxation_used", 0.0),
        },
    }

    return thickness_params, update_info


def _apply_momentum_stabilization(
    delta_proposed: np.ndarray,
    n_bands: int,
    momentum_beta: float = 0.7,
    direction_damping: float = 0.5,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply momentum-based stabilization to thickness changes.

    Tracks the exponential moving average (EMA) of thickness changes per band.
    When the proposed change reverses direction from the trend, apply damping.

    Args:
        delta_proposed: Proposed thickness changes per band (mm)
        n_bands: Number of theta bands
        momentum_beta: EMA smoothing factor (0=no memory, 1=full memory)
        direction_damping: Damping factor when reversing (0=none, 1=full block)

    Returns:
        Tuple of (stabilized deltas, momentum info dict)
    """
    global _momentum_history

    # Initialize momentum EMA if not exists
    if "ema" not in _momentum_history or len(_momentum_history["ema"]) != n_bands:
        _momentum_history["ema"] = np.zeros(n_bands)
        _momentum_history["n_updates"] = 0

    ema = _momentum_history["ema"]
    n_updates = _momentum_history["n_updates"]

    # Compute stabilized deltas
    delta_stabilized = np.zeros(n_bands)
    direction_reversals = []
    damping_applied = []

    for i in range(n_bands):
        proposed = delta_proposed[i]
        trend = ema[i]

        # Check for direction reversal (signs differ and both non-zero)
        reversal = False
        damped = False

        if n_updates > 0 and abs(trend) > 1e-6 and abs(proposed) > 1e-6:
            if np.sign(proposed) != np.sign(trend):
                # Direction reversal detected - apply damping
                reversal = True
                damped = True
                delta_stabilized[i] = proposed * (1.0 - direction_damping)
            else:
                # Same direction - allow full change
                delta_stabilized[i] = proposed
        else:
            # First iteration or near-zero - allow full change
            delta_stabilized[i] = proposed

        direction_reversals.append(reversal)
        damping_applied.append(damped)

    # Update EMA with the stabilized deltas
    if n_updates == 0:
        # First update - initialize EMA to current deltas
        _momentum_history["ema"] = delta_stabilized.copy()
    else:
        # Exponential moving average update
        _momentum_history["ema"] = momentum_beta * ema + (1.0 - momentum_beta) * delta_stabilized

    _momentum_history["n_updates"] = n_updates + 1

    # Build info dict
    momentum_info = {
        "enabled": True,
        "momentum_beta": momentum_beta,
        "direction_damping": direction_damping,
        "n_updates": _momentum_history["n_updates"],
        "ema": _momentum_history["ema"].tolist(),
        "direction_reversals": direction_reversals,
        "n_reversals": sum(direction_reversals),
        "damping_applied": damping_applied,
        "delta_proposed": delta_proposed.tolist(),
        "delta_stabilized": delta_stabilized.tolist(),
    }

    print(f"[Momentum] Iteration {_momentum_history['n_updates']}: "
          f"{sum(direction_reversals)} direction reversals damped")

    return delta_stabilized, momentum_info


def reset_momentum_history():
    """Reset momentum history (call at start of new optimization campaign)."""
    global _momentum_history
    _momentum_history = {}
    print("[Momentum] History reset")


def reset_fracture_bisection_history():
    """Reset fracture bisection history (call at start of new optimization campaign)."""
    global _fracture_bisection_history
    _fracture_bisection_history = {}
    print("[Bisection] History reset")


def reset_secant_history():
    """Reset secant method history (call at start of new optimization campaign)."""
    global _secant_history
    _secant_history = {}
    print("[Secant] History reset")


def get_secant_history() -> Dict:
    """Get current secant history for persistence."""
    global _secant_history
    return _secant_history.copy()


def set_secant_history(state: Dict):
    """Restore secant history from persisted state."""
    global _secant_history
    _secant_history = state.copy() if state else {}
    if _secant_history:
        n_points = len(_secant_history.get("history", []))
        print(f"[Secant] Restored history: {n_points} data points")


def _apply_secant_method(
    thickness_old: np.ndarray,
    stress_by_band: np.ndarray,
    target_stress_pa: float,
    stress_exponent: float,
    min_thickness: float,
    max_thickness: float,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply secant method to predict optimal thickness using historical data.

    The secant method uses two (thickness, stress) data points to estimate
    the local stress-thickness relationship and extrapolate to the target.

    For each band:
        slope = (σ₁ - σ₀) / (t₁ - t₀)
        t_target = t₁ - (σ₁ - σ_target) / slope

    Args:
        thickness_old: Current thickness per band (mm)
        stress_by_band: Current stress per band (Pa)
        target_stress_pa: Target stress (Pa)
        stress_exponent: FSD exponent for fallback
        min_thickness: Minimum allowed thickness (mm)
        max_thickness: Maximum allowed thickness (mm)

    Returns:
        Tuple of (predicted thickness, secant info dict)
    """
    global _secant_history

    n_bands = len(thickness_old)

    # Initialize history if needed
    if "history" not in _secant_history:
        _secant_history["history"] = []

    # Record current data point
    current_point = {
        "thickness_mm": thickness_old.tolist(),
        "stress_Pa": stress_by_band.tolist(),
    }
    _secant_history["history"].append(current_point)

    n_points = len(_secant_history["history"])

    # Need at least 2 points for secant method
    if n_points < 2:
        print(f"[Secant] Only {n_points} data point(s), need 2 for secant method")
        return None, {"n_points": n_points, "used": False}

    # Get the two most recent points
    p0 = _secant_history["history"][-2]  # Previous
    p1 = _secant_history["history"][-1]  # Current

    t0 = np.array(p0["thickness_mm"])
    s0 = np.array(p0["stress_Pa"])
    t1 = np.array(p1["thickness_mm"])
    s1 = np.array(p1["stress_Pa"])

    # Compute secant prediction for each band
    thickness_predicted = np.zeros(n_bands)
    secant_used = np.zeros(n_bands, dtype=bool)

    for i in range(n_bands):
        dt = t1[i] - t0[i]
        ds = s1[i] - s0[i]

        # Check if we have meaningful variation
        if abs(dt) < 0.01 or abs(ds) < 1e3:  # Less than 0.01mm or 1kPa change
            # Not enough variation - use FSD fallback
            if s1[i] > 0:
                stress_ratio = s1[i] / target_stress_pa
                thickness_predicted[i] = t1[i] * (stress_ratio ** stress_exponent)
            else:
                thickness_predicted[i] = t1[i]
            secant_used[i] = False
        else:
            # Secant method: t_new = t1 - (s1 - s_target) / slope
            slope = ds / dt  # dσ/dt (stress change per thickness change)

            # Extrapolate to find thickness where stress = target
            t_predicted = t1[i] - (s1[i] - target_stress_pa) / slope

            # Sanity check: predicted thickness should be reasonable
            if t_predicted < min_thickness or t_predicted > max_thickness:
                # Out of bounds - use bounded FSD instead
                if s1[i] > 0:
                    stress_ratio = s1[i] / target_stress_pa
                    thickness_predicted[i] = t1[i] * (stress_ratio ** stress_exponent)
                else:
                    thickness_predicted[i] = t1[i]
                secant_used[i] = False
            else:
                thickness_predicted[i] = t_predicted
                secant_used[i] = True

    # Enforce bounds
    thickness_predicted = np.clip(thickness_predicted, min_thickness, max_thickness)

    secant_info = {
        "n_points": n_points,
        "used": True,
        "secant_used_per_band": secant_used.tolist(),
        "n_bands_secant": int(np.sum(secant_used)),
        "t0": t0.tolist(),
        "s0_MPa": (s0 / 1e6).tolist(),
        "t1": t1.tolist(),
        "s1_MPa": (s1 / 1e6).tolist(),
        "predicted": thickness_predicted.tolist(),
    }

    print(f"[Secant] Using {int(np.sum(secant_used))}/{n_bands} bands with secant extrapolation")

    return thickness_predicted, secant_info


# =============================================================================
# Optimality Criteria (OC) Method
# =============================================================================

def _compute_oc_update(
    thickness_old: np.ndarray,
    stress_by_band: np.ndarray,
    target_stress_pa: float,
    stress_exponent: float,
    min_thickness: float,
    max_thickness: float,
    oc_move_limit: float = 0.3,
    oc_move_decay: float = 0.95,
    oc_move_floor: float = 0.05,
) -> Tuple[np.ndarray, Dict]:
    """
    Optimality Criteria (OC) method for minimum-weight design under stress constraints.

    Derived from KKT conditions for the problem:
        Minimize: V = Σ t[i]  (volume proxy)
        Subject to: σ[i] ≤ σ_target for all i
                    t_min ≤ t[i] ≤ t_max

    The OC update rule uses the stress ratio as the optimality condition:
        B[i] = (σ[i] / σ_target)
        t_new[i] = t_old[i] * B[i]^η   where η = 1/(1+p), p = stress_exponent

    Move limits prevent oscillation and decay each iteration:
        |t_new - t_old| ≤ move * t_old

    Key difference from FSD: explicit move limits + iteration-dependent step control.

    Args:
        thickness_old: Current thickness per band (mm)
        stress_by_band: Current stress per band (Pa)
        target_stress_pa: Target stress (Pa)
        stress_exponent: Power-law exponent relating stress to thickness
        min_thickness: Minimum allowed thickness (mm)
        max_thickness: Maximum allowed thickness (mm)
        oc_move_limit: Move limit as fraction of current thickness
        oc_move_decay: Decay factor for move limit per iteration
        oc_move_floor: Minimum move limit (fraction)

    Returns:
        Tuple of (updated thickness, OC method info dict)
    """
    global _oc_history

    n_bands = len(thickness_old)

    # Initialize or update OC history
    if "n_iterations" not in _oc_history:
        _oc_history["n_iterations"] = 0
        _oc_history["move_limit"] = oc_move_limit

    # Decay move limit each iteration (floor at oc_move_floor)
    iteration = _oc_history["n_iterations"]
    if iteration > 0:
        _oc_history["move_limit"] = max(
            _oc_history["move_limit"] * oc_move_decay,
            oc_move_floor,
        )
    move = _oc_history["move_limit"]

    # OC update exponent: η = 1/(1+p)
    eta = 1.0 / (1.0 + stress_exponent)

    thickness_new = np.zeros(n_bands)
    for i in range(n_bands):
        if stress_by_band[i] > 0:
            # Optimality condition ratio
            B_i = stress_by_band[i] / target_stress_pa
            # OC update: t_new = t_old * B^η
            t_oc = thickness_old[i] * (B_i ** eta)
        else:
            t_oc = thickness_old[i]

        # Apply move limits (fraction of current thickness)
        move_abs = move * thickness_old[i]
        t_lower = max(thickness_old[i] - move_abs, min_thickness)
        t_upper = min(thickness_old[i] + move_abs, max_thickness)
        thickness_new[i] = max(t_lower, min(t_upper, t_oc))

    # Enforce global bounds
    thickness_new = np.clip(thickness_new, min_thickness, max_thickness)

    _oc_history["n_iterations"] = iteration + 1

    oc_info = {
        "method": "oc",
        "n_iterations": _oc_history["n_iterations"],
        "move_limit_used": move,
        "eta": eta,
        "stress_ratios": (stress_by_band / target_stress_pa).tolist(),
    }

    print(f"[OC] Iteration {_oc_history['n_iterations']}: move_limit={move:.3f}, η={eta:.3f}")

    return thickness_new, oc_info


def get_oc_history() -> Dict:
    """Get current OC method history for persistence."""
    global _oc_history
    return _oc_history.copy()


def set_oc_history(state: Dict):
    """Restore OC method history from persisted state."""
    global _oc_history
    _oc_history = state.copy() if state else {}
    if _oc_history:
        print(f"[OC] Restored history: {_oc_history.get('n_iterations', 0)} iterations, "
              f"move_limit={_oc_history.get('move_limit', 0):.3f}")


def reset_oc_history():
    """Reset OC method history."""
    global _oc_history
    _oc_history = {}
    print("[OC] History reset")


# =============================================================================
# Method of Moving Asymptotes (MMA) — Svanberg 1987
# =============================================================================

def _compute_mma_update(
    thickness_old: np.ndarray,
    stress_by_band: np.ndarray,
    target_stress_pa: float,
    stress_exponent: float,
    min_thickness: float,
    max_thickness: float,
    mma_asymptote_init: float = 0.5,
    mma_oscillation_shrink: float = 0.7,
    mma_monotone_expand: float = 1.2,
) -> Tuple[np.ndarray, Dict]:
    """
    Method of Moving Asymptotes (MMA) for structural optimization (Svanberg 1987).

    Uses convex separable approximations with moving asymptotes to solve:
        Minimize: V = Σ t[i]  (volume proxy)
        Subject to: σ[i] ≤ σ_target for all i
                    t_min ≤ t[i] ≤ t_max

    For each design variable t[i]:
    - Maintain lower asymptote L[i] and upper asymptote U[i]
    - Build convex approximation using reciprocal terms
    - Solve separable subproblem analytically

    Asymptote update rules:
    - Iterations 1-2: L[i] = t[i] - init*(t_max - t_min), U[i] = t[i] + init*(t_max - t_min)
    - Iteration 3+: Check oscillation
      - If (t[k] - t[k-1]) * (t[k-1] - t[k-2]) < 0: tighten by shrink factor
      - If same sign (monotone): loosen by expand factor

    Approximate gradient: ∂σ/∂t ≈ -exponent * σ / t  (power-law)

    Args:
        thickness_old: Current thickness per band (mm)
        stress_by_band: Current stress per band (Pa)
        target_stress_pa: Target stress (Pa)
        stress_exponent: Power-law exponent
        min_thickness: Minimum allowed thickness (mm)
        max_thickness: Maximum allowed thickness (mm)
        mma_asymptote_init: Initial asymptote distance (fraction of range)
        mma_oscillation_shrink: Tighten factor on oscillation detection
        mma_monotone_expand: Loosen factor on monotone progress

    Returns:
        Tuple of (updated thickness, MMA method info dict)
    """
    global _mma_history

    n_bands = len(thickness_old)
    t_range = max_thickness - min_thickness

    # Initialize MMA history
    if "n_iterations" not in _mma_history:
        _mma_history = {
            "n_iterations": 0,
            "thickness_history": [],
            "L": (thickness_old - mma_asymptote_init * t_range).tolist(),
            "U": (thickness_old + mma_asymptote_init * t_range).tolist(),
        }

    iteration = _mma_history["n_iterations"]

    # Record current thickness
    _mma_history["thickness_history"].append(thickness_old.tolist())
    # Keep only last 3 iterations for oscillation detection
    if len(_mma_history["thickness_history"]) > 3:
        _mma_history["thickness_history"] = _mma_history["thickness_history"][-3:]

    hist = _mma_history["thickness_history"]
    L = np.array(_mma_history["L"])
    U = np.array(_mma_history["U"])

    # === UPDATE ASYMPTOTES ===
    if iteration < 2:
        # First two iterations: use initial asymptote distance
        L = thickness_old - mma_asymptote_init * t_range
        U = thickness_old + mma_asymptote_init * t_range
    else:
        # Iteration 3+: detect oscillation vs monotone per variable
        t_k = thickness_old
        t_k1 = np.array(hist[-2])  # Previous
        t_k2 = np.array(hist[-3]) if len(hist) >= 3 else t_k1  # Two iterations ago

        for i in range(n_bands):
            osc = (t_k[i] - t_k1[i]) * (t_k1[i] - t_k2[i])

            if osc < 0:
                # Oscillation detected → tighten asymptotes
                L[i] = t_k[i] - mma_oscillation_shrink * (t_k1[i] - L[i])
                U[i] = t_k[i] + mma_oscillation_shrink * (U[i] - t_k1[i])
            else:
                # Monotone progress → loosen asymptotes
                L[i] = t_k[i] - mma_monotone_expand * (t_k1[i] - L[i])
                U[i] = t_k[i] + mma_monotone_expand * (U[i] - t_k1[i])

    # Enforce asymptote bounds (L must be below t_min, U above t_max isn't required
    # but they must maintain separation from current point)
    min_sep = 0.01 * t_range  # Minimum separation to avoid division by zero
    for i in range(n_bands):
        L[i] = min(L[i], thickness_old[i] - min_sep)
        U[i] = max(U[i], thickness_old[i] + min_sep)

    # === SOLVE MMA SUBPROBLEM ===
    # For each variable, the constraint approximation is:
    #   g_i(t) ≈ g_i(t_k) + p_i/(U_i - t) + q_i/(t - L_i)
    #
    # Approximate gradient of stress w.r.t. thickness (power-law):
    #   dσ/dt ≈ -exponent * σ / t  (thicker → lower stress)
    #
    # For stress constraint: σ(t) ≤ σ_target
    #   Constraint violation: g(t) = σ(t) - σ_target
    #   dg/dt = dσ/dt = -exponent * σ / t
    #
    # MMA coefficients:
    #   If dg/dt < 0: p = (U - t)^2 * |dg/dt|, q = 0   (increasing t helps)
    #   If dg/dt > 0: p = 0, q = (t - L)^2 * dg/dt      (decreasing t helps)
    #
    # Optimal t* for minimizing volume subject to linearized constraint:
    #   t* = (L*sqrt(p) + U*sqrt(q)) / (sqrt(p) + sqrt(q))

    thickness_new = np.zeros(n_bands)

    for i in range(n_bands):
        t_i = thickness_old[i]
        sigma_i = stress_by_band[i]

        if sigma_i <= 0:
            thickness_new[i] = t_i
            continue

        # Approximate constraint gradient: dg/dt = -exponent * σ / t
        dg_dt = -stress_exponent * sigma_i / t_i

        # MMA coefficients for asymptotes
        U_i = U[i]
        L_i = L[i]

        # --- Constraint term (stress feasibility: σ ≤ σ_target) ---
        # dg/dt < 0 means increasing t reduces stress → p_con pulls toward U
        if dg_dt < 0:
            p_con = (U_i - t_i) ** 2 * abs(dg_dt)
            q_con = 0.0
        else:
            p_con = 0.0
            q_con = (t_i - L_i) ** 2 * dg_dt

        # --- Volume objective term (minimize Σt[i]) ---
        # df/dt = 1 > 0 → increasing t increases volume → q_obj pulls toward L
        # Normalized by t_range so objective and constraint are comparable
        q_obj = (t_i - L_i) ** 2 * (1.0 / t_range)

        # Lagrange multiplier for constraint (scales stress vs volume tradeoff)
        # Capped at 5.0 to prevent volume term from being overwhelmed when
        # starting far from feasibility (e.g., stress_ratio=22 → lam=484 without cap)
        stress_ratio = sigma_i / target_stress_pa
        if stress_ratio > 1.0:
            # Overstressed: strong pull toward feasibility (capped)
            lam = min(stress_ratio ** 2, 5.0)
        elif stress_ratio < 0.8:
            # Over-designed: weak constraint, volume objective dominates
            lam = 0.2
        else:
            # Near-optimal: balanced
            lam = 1.0

        # Combined MMA subproblem: min f_approx + λ * g_approx
        # p_total pulls toward U (thicken for stress), q_total pulls toward L (thin for volume)
        p_total = lam * p_con
        q_total = q_obj + lam * q_con

        sqrt_p = np.sqrt(max(p_total, 0.0))
        sqrt_q = np.sqrt(max(q_total, 0.0))

        denom = sqrt_p + sqrt_q
        if denom > 1e-12:
            t_star = (U_i * sqrt_p + L_i * sqrt_q) / denom
        else:
            t_star = t_i

        # Bound within asymptotes (with margin)
        alpha_i = L_i + 0.1 * (t_i - L_i)
        beta_i = U_i - 0.1 * (U_i - t_i)
        t_star = max(alpha_i, min(beta_i, t_star))

        thickness_new[i] = t_star

    # Enforce global bounds
    thickness_new = np.clip(thickness_new, min_thickness, max_thickness)

    # Store updated asymptotes
    _mma_history["L"] = L.tolist()
    _mma_history["U"] = U.tolist()
    _mma_history["n_iterations"] = iteration + 1

    mma_info = {
        "method": "mma",
        "n_iterations": _mma_history["n_iterations"],
        "L": L.tolist(),
        "U": U.tolist(),
        "stress_ratios": (stress_by_band / target_stress_pa).tolist(),
    }

    n_osc = 0
    if iteration >= 2 and len(hist) >= 3:
        t_k = thickness_old
        t_k1 = np.array(hist[-2])
        t_k2 = np.array(hist[-3]) if len(hist) >= 3 else t_k1
        n_osc = int(np.sum((t_k - t_k1) * (t_k1 - t_k2) < 0))
    mma_info["n_oscillating_vars"] = n_osc

    print(f"[MMA] Iteration {_mma_history['n_iterations']}: {n_osc}/{n_bands} oscillating variables")

    return thickness_new, mma_info


def get_mma_history() -> Dict:
    """Get current MMA method history for persistence."""
    global _mma_history
    return _mma_history.copy()


def set_mma_history(state: Dict):
    """Restore MMA method history from persisted state."""
    global _mma_history
    _mma_history = state.copy() if state else {}
    if _mma_history:
        print(f"[MMA] Restored history: {_mma_history.get('n_iterations', 0)} iterations")


def reset_mma_history():
    """Reset MMA method history."""
    global _mma_history
    _mma_history = {}
    print("[MMA] History reset")


# =============================================================================
# Bayesian Optimization (GP + Expected Improvement)
# =============================================================================

def _gp_rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float, signal_var: float = 1.0) -> np.ndarray:
    """
    RBF (squared exponential) kernel matrix.

    K(x1, x2) = signal_var * exp(-0.5 * ||x1 - x2||^2 / length_scale^2)

    Args:
        X1: (n1, d) array
        X2: (n2, d) array
        length_scale: RBF length scale
        signal_var: Signal variance (amplitude^2)

    Returns:
        (n1, n2) kernel matrix
    """
    # Squared Euclidean distance matrix
    sq_dist = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2.0 * X1 @ X2.T
    sq_dist = np.maximum(sq_dist, 0.0)  # Numerical safety
    return signal_var * np.exp(-0.5 * sq_dist / (length_scale**2))


def _gp_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    length_scale: float,
    noise: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian Process posterior prediction.

    Args:
        X_train: (n, d) training inputs (normalized thickness vectors)
        y_train: (n,) training targets (objective values)
        X_test: (m, d) test inputs
        length_scale: RBF kernel length scale
        noise: Observation noise variance

    Returns:
        Tuple of (mean, variance) arrays, each shape (m,)
    """
    n = len(X_train)
    K = _gp_rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(n)
    K_s = _gp_rbf_kernel(X_train, X_test, length_scale)
    K_ss = _gp_rbf_kernel(X_test, X_test, length_scale)

    # Solve K^{-1} y and K^{-1} K_s via Cholesky
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        V = np.linalg.solve(L, K_s)
    except np.linalg.LinAlgError:
        # Fallback: add more jitter
        K += 1e-6 * np.eye(n)
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        V = np.linalg.solve(L, K_s)

    mu = K_s.T @ alpha
    var = np.diag(K_ss) - np.sum(V**2, axis=0)
    var = np.maximum(var, 1e-10)  # Numerical safety

    return mu, var


def _expected_improvement(mu: np.ndarray, var: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Expected Improvement acquisition function.

    EI(x) = (y_best - mu - xi) * Φ(Z) + σ * φ(Z)
    where Z = (y_best - mu - xi) / σ

    We minimize the objective, so improvement = y_best - mu.

    Args:
        mu: Predicted means
        var: Predicted variances
        y_best: Best (lowest) observed objective value
        xi: Exploration parameter (higher = more exploration)

    Returns:
        EI values (higher = more promising)
    """
    sigma = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = (y_best - mu - xi) / sigma
        # Standard normal CDF and PDF
        cdf_Z = 0.5 * (1.0 + _erf_approx(Z / np.sqrt(2.0)))
        pdf_Z = np.exp(-0.5 * Z**2) / np.sqrt(2.0 * np.pi)
        ei = (y_best - mu - xi) * cdf_Z + sigma * pdf_Z
        ei = np.where(sigma > 1e-10, ei, 0.0)
    return ei


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun approximation to erf (max error 1.5e-7)."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-x * x))


def _compute_bo_update(
    thickness_old: np.ndarray,
    stress_by_band: np.ndarray,
    target_stress_pa: float,
    stress_exponent: float,
    min_thickness: float,
    max_thickness: float,
    length_scale: float = 1.0,
    noise: float = 1e-4,
    xi: float = 0.01,
    n_random_init: int = 3,
) -> Tuple[np.ndarray, Dict]:
    """
    Bayesian Optimization for shell thickness using GP surrogate + Expected Improvement.

    Objective: minimize max stress violation + total volume
        f(t) = w_stress * max(0, σ_max/σ_target - 1)^2 + w_vol * mean(t)/t_max

    The GP builds a surrogate of f(t) from all observed (thickness, stress) pairs,
    then maximizes Expected Improvement to suggest the next thickness to try.

    For the first n_random_init iterations, uses Latin Hypercube-style perturbations
    to explore before the GP has enough data.

    Args:
        thickness_old: Current thickness per band (mm)
        stress_by_band: Current stress per band (Pa)
        target_stress_pa: Target stress (Pa)
        stress_exponent: Power-law exponent (not directly used by BO)
        min_thickness: Minimum allowed thickness (mm)
        max_thickness: Maximum allowed thickness (mm)
        length_scale: GP RBF kernel length scale (in normalized units)
        noise: GP observation noise variance
        xi: EI exploration parameter
        n_random_init: Number of initial random-ish iterations before GP kicks in

    Returns:
        Tuple of (suggested thickness, BO method info dict)
    """
    global _bo_history

    n_bands = len(thickness_old)
    t_range = max_thickness - min_thickness

    # Initialize BO history
    if "observations" not in _bo_history:
        _bo_history = {
            "observations": [],  # List of {"thickness": [...], "stress": [...], "objective": float}
            "n_iterations": 0,
        }

    # Compute objective for current observation
    # Objective: penalize stress violation + reward low volume
    stress_ratios = stress_by_band / target_stress_pa
    max_violation = max(0.0, np.max(stress_ratios) - 1.0)
    mean_stress_violation = float(np.mean(np.maximum(stress_ratios - 1.0, 0.0)))
    volume_fraction = float(np.mean(thickness_old)) / max_thickness

    # Combined objective (lower is better):
    # - Heavy penalty for stress constraint violation
    # - Light reward for material savings
    objective = 10.0 * max_violation**2 + 2.0 * mean_stress_violation + 0.5 * volume_fraction

    # Record observation
    _bo_history["observations"].append({
        "thickness": thickness_old.tolist(),
        "stress": stress_by_band.tolist(),
        "objective": objective,
        "max_stress_ratio": float(np.max(stress_ratios)),
        "volume_fraction": volume_fraction,
    })

    n_obs = len(_bo_history["observations"])
    _bo_history["n_iterations"] = n_obs

    print(f"[BO] Recorded observation {n_obs}: obj={objective:.4f}, "
          f"max_σ/σ_target={np.max(stress_ratios):.3f}, vol={volume_fraction:.3f}")

    # Phase 1: Not enough data for GP — use FSD-like perturbation with exploration
    if n_obs < n_random_init:
        # Use stress-guided perturbation (like FSD but with random exploration)
        thickness_new = np.zeros(n_bands)
        for i in range(n_bands):
            if stress_by_band[i] > 0:
                # FSD direction + small random perturbation for exploration
                ratio = stress_by_band[i] / target_stress_pa
                t_fsd = thickness_old[i] * (ratio ** (1.0 / (1.0 + stress_exponent)))
                # Add exploration noise (±10% of range, decreasing with observations)
                explore_scale = 0.1 * t_range * (1.0 - n_obs / n_random_init)
                # Deterministic "pseudo-random" based on band index and iteration
                # This ensures reproducibility without seeds
                phase = 2.0 * np.pi * (i * 0.618033988749895 + n_obs * 0.4142135623730951)
                perturbation = explore_scale * np.sin(phase)
                thickness_new[i] = t_fsd + perturbation
            else:
                thickness_new[i] = thickness_old[i]

        thickness_new = np.clip(thickness_new, min_thickness, max_thickness)

        bo_info = {
            "method": "bo",
            "phase": "exploration",
            "n_observations": n_obs,
            "objective": objective,
            "max_stress_ratio": float(np.max(stress_ratios)),
        }

        print(f"[BO] Phase: exploration ({n_obs}/{n_random_init} initial samples)")
        return thickness_new, bo_info

    # Phase 2: Enough data — use GP + Expected Improvement
    # Build training data (normalize to [0, 1])
    obs = _bo_history["observations"]
    X_train = np.array([o["thickness"] for o in obs])
    y_train = np.array([o["objective"] for o in obs])

    # Normalize inputs to [0, 1]
    X_norm = (X_train - min_thickness) / t_range

    # Normalize targets (zero mean, unit variance)
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) if np.std(y_train) > 1e-8 else 1.0
    y_norm = (y_train - y_mean) / y_std
    y_best_norm = np.min(y_norm)

    # Generate candidate points by perturbing current best + FSD suggestion
    best_idx = int(np.argmin(y_train))
    t_best = X_train[best_idx]

    # FSD suggestion from current stress
    t_fsd = np.zeros(n_bands)
    for i in range(n_bands):
        if stress_by_band[i] > 0:
            ratio = stress_by_band[i] / target_stress_pa
            t_fsd[i] = thickness_old[i] * (ratio ** (1.0 / (1.0 + stress_exponent)))
        else:
            t_fsd[i] = thickness_old[i]
    t_fsd = np.clip(t_fsd, min_thickness, max_thickness)

    # Generate candidates: grid of perturbations around best and FSD suggestion
    n_candidates = 200
    candidates = np.zeros((n_candidates, n_bands))

    for c in range(n_candidates):
        if c < n_candidates // 3:
            # Perturbations around best observed
            base = t_best.copy()
        elif c < 2 * n_candidates // 3:
            # Perturbations around FSD suggestion
            base = t_fsd.copy()
        else:
            # Perturbations around current
            base = thickness_old.copy()

        # Deterministic quasi-random perturbation (Halton-like using golden ratio)
        for i in range(n_bands):
            phase = 2.0 * np.pi * ((c * 0.618033988749895 + i * 0.4142135623730951) % 1.0)
            scale = 0.3 * t_range * (1.0 / (1.0 + 0.1 * n_obs))  # Shrink exploration over time
            candidates[c, i] = base[i] + scale * np.sin(phase)

    candidates = np.clip(candidates, min_thickness, max_thickness)
    X_cand_norm = (candidates - min_thickness) / t_range

    # GP prediction on candidates
    mu, var = _gp_predict(X_norm, y_norm, X_cand_norm, length_scale, noise)

    # Expected Improvement
    ei = _expected_improvement(mu, var, y_best_norm, xi)

    # Select candidate with highest EI
    best_cand_idx = int(np.argmax(ei))
    thickness_new = candidates[best_cand_idx]

    # Denormalize predicted objective for logging
    pred_obj = float(mu[best_cand_idx] * y_std + y_mean)

    bo_info = {
        "method": "bo",
        "phase": "gp_exploitation",
        "n_observations": n_obs,
        "objective": objective,
        "best_objective": float(np.min(y_train)),
        "best_observation_idx": best_idx,
        "predicted_objective": pred_obj,
        "max_ei": float(ei[best_cand_idx]),
        "max_stress_ratio": float(np.max(stress_ratios)),
    }

    print(f"[BO] Phase: GP+EI (best_obj={np.min(y_train):.4f}, pred={pred_obj:.4f}, EI={ei[best_cand_idx]:.4f})")

    return thickness_new, bo_info


def get_bo_history() -> Dict:
    """Get current BO history for persistence."""
    global _bo_history
    return _bo_history.copy()


def set_bo_history(state: Dict):
    """Restore BO history from persisted state."""
    global _bo_history
    _bo_history = state.copy() if state else {}
    if _bo_history:
        n = len(_bo_history.get("observations", []))
        print(f"[BO] Restored history: {n} observations")


def reset_bo_history():
    """Reset BO history."""
    global _bo_history
    _bo_history = {}
    print("[BO] History reset")


# =============================================================================
# Projected Gradient Descent (PGD)
# =============================================================================

def _compute_pgd_update(
    thickness_old: np.ndarray,
    stress_by_band: np.ndarray,
    target_stress_pa: float,
    stress_exponent: float,
    min_thickness: float,
    max_thickness: float,
    initial_step: float = 1.0,
    backtrack_factor: float = 0.5,
    min_step: float = 0.01,
) -> Tuple[np.ndarray, Dict]:
    """
    Projected Gradient Descent for minimum-weight design under stress constraints.

    Formulation (Lagrangian relaxation of constrained problem):
        L(t) = w_vol * Σ t[i] + w_stress * Σ max(0, σ[i]/σ_target - 1)^2

    Gradient (using power-law stress approximation σ ∝ t^{-p}):
        ∂L/∂t[i] = w_vol + w_stress * 2*(σ[i]/σ_target - 1) * (∂σ/∂t[i]) / σ_target
        where ∂σ/∂t[i] ≈ -p * σ[i] / t[i]

    Update: t_new = Project[t_old - α * ∇L]  onto [t_min, t_max]

    Step size α uses Barzilai-Borwein (BB) spectral method when history is
    available (iteration 2+), with a backtracking decay floor. BB gives
    superlinear convergence without a line search callback.

    Args:
        thickness_old: Current thickness per band (mm)
        stress_by_band: Current stress per band (Pa)
        target_stress_pa: Target stress (Pa)
        stress_exponent: Power-law exponent (p)
        min_thickness: Minimum allowed thickness (mm)
        max_thickness: Maximum allowed thickness (mm)
        initial_step: Initial step size (mm)
        backtrack_factor: Step size reduction per iteration when oscillating
        min_step: Floor for step size

    Returns:
        Tuple of (updated thickness, PGD method info dict)
    """
    global _pgd_history

    n_bands = len(thickness_old)

    # Initialize history
    if "n_iterations" not in _pgd_history:
        _pgd_history = {
            "n_iterations": 0,
            "step_size": initial_step,
            "prev_thickness": None,
            "prev_gradient": None,
            "prev_objective": None,
        }

    iteration = _pgd_history["n_iterations"]

    # === COMPUTE OBJECTIVE AND GRADIENT ===
    # Lagrangian: L = w_vol * mean(t) + w_stress * Σ max(0, σ_i/σ_target - 1)^2
    w_vol = 1.0
    w_stress = 10.0  # Heavy penalty for stress violation

    stress_ratios = stress_by_band / target_stress_pa
    violations = np.maximum(stress_ratios - 1.0, 0.0)

    objective = w_vol * np.mean(thickness_old) / max_thickness + w_stress * np.sum(violations**2)

    # Gradient of Lagrangian w.r.t. thickness
    # ∂L/∂t[i] = w_vol / (n * t_max)
    #           + w_stress * 2 * violation[i] * (∂σ/∂t[i]) / σ_target
    # where ∂σ/∂t[i] ≈ -p * σ[i] / t[i]  (power-law approximation)
    gradient = np.zeros(n_bands)
    for i in range(n_bands):
        # Volume gradient (always pushes toward thinner)
        grad_vol = w_vol / (n_bands * max_thickness)

        # Stress constraint gradient
        if stress_by_band[i] > 0 and thickness_old[i] > 0:
            dsigma_dt = -stress_exponent * stress_by_band[i] / thickness_old[i]
            grad_stress = w_stress * 2.0 * violations[i] * dsigma_dt / target_stress_pa
        else:
            grad_stress = 0.0

        gradient[i] = grad_vol + grad_stress

    # === BARZILAI-BORWEIN STEP SIZE (iteration 2+) ===
    step_size = _pgd_history["step_size"]

    if iteration > 0 and _pgd_history["prev_thickness"] is not None and _pgd_history["prev_gradient"] is not None:
        t_prev = np.array(_pgd_history["prev_thickness"])
        g_prev = np.array(_pgd_history["prev_gradient"])

        s = thickness_old - t_prev     # Δt
        y = gradient - g_prev           # Δg

        s_dot_y = np.dot(s, y)
        s_dot_s = np.dot(s, s)

        if abs(s_dot_y) > 1e-12 and s_dot_s > 1e-12:
            # BB Type 1: α = (s·s) / (s·y)
            bb_step = s_dot_s / abs(s_dot_y)
            # Clamp to reasonable range
            bb_step = max(min_step, min(bb_step, 5.0 * initial_step))
            step_size = bb_step
            print(f"[PGD] Barzilai-Borwein step: {bb_step:.4f}")
        else:
            # Not enough curvature info — decay previous step
            step_size = max(step_size * backtrack_factor, min_step)
            print(f"[PGD] Insufficient curvature, decayed step: {step_size:.4f}")

        # Check for oscillation (objective got worse) and backtrack
        if _pgd_history["prev_objective"] is not None:
            if objective > _pgd_history["prev_objective"] * 1.01:
                step_size = max(step_size * backtrack_factor, min_step)
                print(f"[PGD] Objective increased ({_pgd_history['prev_objective']:.4f} → {objective:.4f}), "
                      f"backtracking to step={step_size:.4f}")

    # === GRADIENT DESCENT STEP ===
    thickness_new = thickness_old - step_size * gradient

    # === PROJECT ONTO FEASIBLE SET [t_min, t_max] ===
    thickness_new = np.clip(thickness_new, min_thickness, max_thickness)

    # === UPDATE HISTORY ===
    _pgd_history["prev_thickness"] = thickness_old.tolist()
    _pgd_history["prev_gradient"] = gradient.tolist()
    _pgd_history["prev_objective"] = float(objective)
    _pgd_history["step_size"] = step_size
    _pgd_history["n_iterations"] = iteration + 1

    grad_norm = float(np.linalg.norm(gradient))
    pgd_info = {
        "method": "pgd",
        "n_iterations": _pgd_history["n_iterations"],
        "step_size_used": step_size,
        "objective": objective,
        "gradient_norm": grad_norm,
        "stress_ratios": stress_ratios.tolist(),
        "n_violated": int(np.sum(violations > 0)),
    }

    print(f"[PGD] Iteration {_pgd_history['n_iterations']}: obj={objective:.4f}, "
          f"|∇|={grad_norm:.4f}, step={step_size:.4f}, violated={int(np.sum(violations > 0))}/{n_bands}")

    return thickness_new, pgd_info


def get_pgd_history() -> Dict:
    """Get current PGD history for persistence."""
    global _pgd_history
    return _pgd_history.copy()


def set_pgd_history(state: Dict):
    """Restore PGD history from persisted state."""
    global _pgd_history
    _pgd_history = state.copy() if state else {}
    if _pgd_history:
        print(f"[PGD] Restored history: {_pgd_history.get('n_iterations', 0)} iterations, "
              f"step={_pgd_history.get('step_size', 0):.4f}")


def reset_pgd_history():
    """Reset PGD history."""
    global _pgd_history
    _pgd_history = {}
    print("[PGD] History reset")


# =============================================================================
# COBYLA (Constrained Optimization BY Linear Approximation)
# =============================================================================

def _compute_cobyla_update(
    thickness_old: np.ndarray,
    stress_by_band: np.ndarray,
    target_stress_pa: float,
    stress_exponent: float,
    min_thickness: float,
    max_thickness: float,
    rhobeg: float = 0.5,
    rhoend: float = 0.01,
    maxiter: int = 1000,
) -> Tuple[np.ndarray, Dict]:
    """
    COBYLA derivative-free optimizer for minimum-weight shell design.

    Uses scipy.optimize.minimize(method='COBYLA') to solve the constrained
    subproblem each iteration.  COBYLA builds local linear models of the
    objective and constraints within a trust region — no gradient information
    is used.

    Subproblem formulation (same physics model as PGD):
        minimize   Σ t[i]                          (total thickness)
        subject to σ_target − σ_model(t[i]) ≥ 0    for each band
                   t[i] − t_min ≥ 0                for each band
                   t_max − t[i] ≥ 0                for each band

    where σ_model(t[i]) = σ_current[i] * (t_current[i] / t[i])^p   (power-law)

    Unlike PGD which takes a single gradient step per iteration, COBYLA
    fully solves the subproblem each iteration — providing a reference for
    how much gradient information accelerates convergence.

    The trust-region initial radius (rhobeg) shrinks across outer iterations
    to keep COBYLA proposals within the power-law model's validity range.

    Args:
        thickness_old: Current thickness per band (mm)
        stress_by_band: Current stress per band (Pa)
        target_stress_pa: Target stress (Pa)
        stress_exponent: Power-law exponent (p)
        min_thickness: Minimum allowed thickness (mm)
        max_thickness: Maximum allowed thickness (mm)
        rhobeg: Initial trust-region radius (mm)
        rhoend: Final trust-region radius (convergence tolerance)
        maxiter: Maximum COBYLA inner iterations

    Returns:
        Tuple of (updated thickness, COBYLA method info dict)
    """
    from scipy.optimize import minimize as sp_minimize

    global _cobyla_history

    n_bands = len(thickness_old)

    # Initialize history
    if "n_iterations" not in _cobyla_history:
        _cobyla_history = {
            "n_iterations": 0,
            "prev_thickness": None,
            "prev_objective": None,
        }

    iteration = _cobyla_history["n_iterations"]

    # Scale trust-region radius based on how far we are from feasibility.
    # When stress ratios are large (far from target), use larger trust region
    # so COBYLA can make meaningful steps toward feasibility.
    max_stress_ratio = float(np.max(stress_by_band / target_stress_pa))
    if max_stress_ratio > 2.0:
        rho_scale = 2.0  # Far from feasible: double the trust region
    elif max_stress_ratio > 1.2:
        rho_scale = 1.0  # Near feasibility: standard trust region
    else:
        rho_scale = max(0.5, 1.0 / (1.0 + 0.1 * iteration))  # Feasible: shrink gently
    rhobeg_scaled = rhobeg * rho_scale

    # Current objective value (for logging)
    objective_current = float(np.sum(thickness_old))

    # --- Define COBYLA subproblem ---
    # Freeze current observation for the power-law model
    sigma_current = stress_by_band.copy()
    t_current = thickness_old.copy()

    def objective(t):
        return np.sum(t)

    # Stress constraints: (target - sigma_model) / target >= 0  ⟹  sigma <= target
    # CRITICAL: Normalize constraints to O(1) so COBYLA's linear models are well-conditioned.
    # Without normalization, constraints are O(1e7) Pa while objective is O(10) mm,
    # causing COBYLA to ignore constraints and drive to minimum thickness.
    constraints = []
    for i in range(n_bands):
        def stress_con(t, idx=i):
            if t[idx] <= 0:
                return -1e6  # infeasible
            sigma_model = sigma_current[idx] * (t_current[idx] / t[idx]) ** stress_exponent
            return (target_stress_pa - sigma_model) / target_stress_pa  # Normalized: 0 = at limit, +1 = zero stress
        constraints.append({"type": "ineq", "fun": stress_con})

    # Bound constraints (COBYLA uses inequality constraints for bounds)
    for i in range(n_bands):
        constraints.append({"type": "ineq", "fun": lambda t, idx=i: t[idx] - min_thickness})
        constraints.append({"type": "ineq", "fun": lambda t, idx=i: max_thickness - t[idx]})

    # Initial guess: current thickness
    t0 = thickness_old.copy()

    # Log pre-solve state for diagnostics
    stress_ratios_init = sigma_current / target_stress_pa
    print(f"[COBYLA] Pre-solve: t0={np.array2string(t0, precision=2)}, "
          f"σ/σt={np.array2string(stress_ratios_init, precision=2)}, "
          f"rho={rhobeg_scaled:.3f}")

    # Solve subproblem
    result = sp_minimize(
        objective,
        t0,
        method="COBYLA",
        constraints=constraints,
        options={
            "maxiter": maxiter,
            "rhobeg": rhobeg_scaled,
            "catol": 0.02,  # 2% constraint tolerance (normalized units)
        },
    )

    thickness_new = np.clip(result.x, min_thickness, max_thickness)

    # Compute stress ratios at proposed point (for logging)
    stress_ratios_new = np.array([
        sigma_current[i] * (t_current[i] / thickness_new[i]) ** stress_exponent
        for i in range(n_bands)
    ]) / target_stress_pa

    # Update history
    _cobyla_history["prev_thickness"] = thickness_old.tolist()
    _cobyla_history["prev_objective"] = objective_current
    _cobyla_history["n_iterations"] = iteration + 1

    objective_new = float(np.sum(thickness_new))

    cobyla_info = {
        "method": "cobyla",
        "n_iterations": _cobyla_history["n_iterations"],
        "scipy_success": result.success,
        "scipy_message": str(result.message),
        "scipy_nfev": result.nfev,
        "objective_before": objective_current,
        "objective_after": objective_new,
        "final_trust_radius": rhobeg_scaled,
        "max_stress_ratio": float(np.max(stress_ratios_new)),
        "n_violated": int(np.sum(stress_ratios_new > 1.0)),
    }

    print(f"[COBYLA] Iteration {_cobyla_history['n_iterations']}: "
          f"Σt={objective_current:.1f}→{objective_new:.1f}mm, "
          f"nfev={result.nfev}, rho={rhobeg_scaled:.3f}, "
          f"max_σ/σt={np.max(stress_ratios_new):.3f}, "
          f"success={result.success}, msg={result.message}")

    return thickness_new, cobyla_info


def get_cobyla_history() -> Dict:
    """Get current COBYLA history for persistence."""
    global _cobyla_history
    return _cobyla_history.copy()


def set_cobyla_history(state: Dict):
    """Restore COBYLA history from persisted state."""
    global _cobyla_history
    _cobyla_history = state.copy() if state else {}
    if _cobyla_history:
        print(f"[COBYLA] Restored history: {_cobyla_history.get('n_iterations', 0)} iterations")


def reset_cobyla_history():
    """Reset COBYLA history."""
    global _cobyla_history
    _cobyla_history = {}
    print("[COBYLA] History reset")


def get_fracture_bisection_history() -> Dict:
    """Get current fracture bisection history for persistence."""
    global _fracture_bisection_history
    return _fracture_bisection_history.copy()


def set_fracture_bisection_history(state: Dict):
    """Restore fracture bisection history from persisted state."""
    global _fracture_bisection_history
    _fracture_bisection_history = state.copy() if state else {}
    if _fracture_bisection_history:
        breach = len(_fracture_bisection_history.get("breach_factors", []))
        no_breach = len(_fracture_bisection_history.get("no_breach_factors", []))
        print(f"[Bisection] Restored history: {breach} breach, {no_breach} no-breach points")


def get_momentum_history() -> Dict:
    """
    Get current momentum history for persistence.

    Returns:
        Dictionary containing momentum state (EMA values, update count)
    """
    global _momentum_history
    if not _momentum_history:
        return {}
    return {
        "ema": _momentum_history.get("ema", []).tolist() if isinstance(_momentum_history.get("ema"), np.ndarray) else _momentum_history.get("ema", []),
        "n_updates": _momentum_history.get("n_updates", 0),
    }


def set_momentum_history(state: Dict):
    """
    Restore momentum history from persisted state.

    Use when continuing from a previous optimization run.

    Args:
        state: Dictionary containing momentum state (from get_momentum_history())
    """
    global _momentum_history
    if not state:
        return
    ema = state.get("ema", [])
    _momentum_history = {
        "ema": np.array(ema) if ema else np.array([]),
        "n_updates": state.get("n_updates", 0),
    }
    print(f"[Momentum] Restored history: {_momentum_history['n_updates']} updates")


def update_fracture_factor(
    thickness_params: ThetaBandParams,
    peridigm_results: Dict,
    config: Dict,
) -> Tuple[ThetaBandParams, Dict]:
    """
    Update fracture zone thickness factors based on Peridigm damage results.

    The goal is to tune fracture factors so that:
    - Fracture zone breaks reliably (high D_frac)
    - Shell body remains intact (low D_shell)

    Update logic:
    - If D_frac < target: decrease factor (thinner → breaks easier)
    - If D_frac >= target AND breach occurred: factor is good, maybe increase slightly
    - Apply relaxation and momentum for stability
    - BISECTION MODE: Once we have breach and no-breach, use binary search

    Args:
        thickness_params: Current thickness parameterization (modified in place)
        peridigm_results: Peridigm results with D_fracture_zone, D_shell_body, etc.
        config: Optimization configuration

    Returns:
        Tuple of (updated params, fracture update info dict)
    """
    global _fracture_bisection_history
    opt_cfg = config.get("optimization", {})

    # Fracture optimization parameters
    fracture_relaxation = float(opt_cfg.get("fracture_relaxation", 0.2))
    target_D_frac = float(opt_cfg.get("target_D_fracture", 0.3))  # Target mean damage in fracture zone
    max_D_shell = float(opt_cfg.get("max_D_shell", 0.1))  # Max acceptable shell body damage
    fracture_resolution = float(opt_cfg.get("fracture_resolution", 0.05))  # Round to this

    # === BISECTION MODE ===
    use_fracture_bisection = bool(opt_cfg.get("use_fracture_bisection", False))

    # Get current fracture factors
    factor_old = thickness_params.fracture_factor.copy()
    n_bands = thickness_params.n_bands
    avg_factor = float(np.mean(factor_old))

    # Get Peridigm damage metrics
    D_frac = float(peridigm_results.get("D_fracture_zone", 0.0))
    D_shell = float(peridigm_results.get("D_shell_body", 0.0))
    breach = peridigm_results.get("fracture_metrics", {}).get("breach", False)

    # === UPDATE BISECTION HISTORY ===
    if use_fracture_bisection:
        if "breach_factors" not in _fracture_bisection_history:
            _fracture_bisection_history = {
                "breach_factors": [],      # Factors that caused breach
                "no_breach_factors": [],   # Factors that didn't breach
            }

        if breach:
            _fracture_bisection_history["breach_factors"].append(avg_factor)
            print(f"[Bisection] Recorded BREACH at factor={avg_factor:.3f}")
        else:
            _fracture_bisection_history["no_breach_factors"].append(avg_factor)
            print(f"[Bisection] Recorded NO-BREACH at factor={avg_factor:.3f}")

    # Compute factor update
    factor_delta = np.zeros(n_bands)
    bisection_used = False

    # === TRY BISECTION FIRST ===
    if use_fracture_bisection:
        breach_factors = _fracture_bisection_history.get("breach_factors", [])
        no_breach_factors = _fracture_bisection_history.get("no_breach_factors", [])

        if len(breach_factors) > 0 and len(no_breach_factors) > 0:
            # We have both breach and no-breach data → use bisection
            max_breach = max(breach_factors)      # Highest factor that breached
            min_no_breach = min(no_breach_factors)  # Lowest factor that didn't breach

            if max_breach < min_no_breach:
                # Valid bisection range: breach happens below min_no_breach
                bisect_target = (max_breach + min_no_breach) / 2
                factor_delta[:] = bisect_target - avg_factor
                bisection_used = True
                print(f"[Bisection] Breach range: [{max_breach:.3f}, {min_no_breach:.3f}]")
                print(f"[Bisection] Target midpoint: {bisect_target:.3f} (delta={factor_delta[0]:.3f})")
            else:
                print(f"[Bisection] Invalid range (breach={max_breach:.3f} >= no_breach={min_no_breach:.3f}), using heuristic")

    # === FALLBACK TO HEURISTIC ===
    if not bisection_used:
        if D_frac < target_D_frac:
            # Fracture zone not breaking enough → decrease factor (make thinner)
            # Scale by how far we are from target
            gap = (target_D_frac - D_frac) / target_D_frac  # 0 to 1
            adjustment = -0.2 * gap  # Max -0.2 decrease (was -0.1, too small with relaxation)
            factor_delta[:] = adjustment
            print(f"[Fracture] D_frac={D_frac:.3f} < target={target_D_frac:.3f} → decreasing factors by {adjustment:.3f}")
        elif breach and D_shell <= max_D_shell:
            # Fracture working well and shell intact → can slightly increase factor for robustness
            adjustment = 0.02
            factor_delta[:] = adjustment
            print(f"[Fracture] Breach OK, D_shell={D_shell:.3f} acceptable → slight increase by {adjustment:.3f}")
        elif D_shell > max_D_shell:
            # Shell taking too much damage → might need thicker fracture zone or other adjustments
            # For now, don't change (this is more of a protected thickness issue)
            print(f"[Fracture] WARNING: D_shell={D_shell:.3f} > max={max_D_shell:.3f} (shell damage too high)")
        else:
            print(f"[Fracture] D_frac={D_frac:.3f}, D_shell={D_shell:.3f}, breach={breach} → no change")

    # Apply relaxation
    factor_delta_relaxed = fracture_relaxation * factor_delta

    # Fixed bands: zero out delta for bands near the pole hole
    fixed_band_indices = opt_cfg.get("fixed_band_indices", [])
    for band_idx in fixed_band_indices:
        if 0 <= band_idx < len(factor_delta_relaxed):
            factor_delta_relaxed[band_idx] = 0.0

    # Ensure minimum step size in mm (not factor units)
    # This guarantees each update makes a meaningful change to fracture thickness
    min_step_mm = float(opt_cfg.get("fracture_min_step_mm", 0.1))
    protected_thickness = thickness_params.thickness_mm

    for i in range(n_bands):
        # Convert factor delta to thickness delta
        thickness_delta = factor_delta_relaxed[i] * protected_thickness[i]

        # Enforce minimum step size (in the direction of the change)
        if abs(thickness_delta) > 0 and abs(thickness_delta) < min_step_mm:
            # Scale up to minimum step
            sign = np.sign(factor_delta_relaxed[i])
            min_factor_delta = min_step_mm / protected_thickness[i]
            factor_delta_relaxed[i] = sign * min_factor_delta
            print(f"[Fracture] Band {i}: enforced min step {min_step_mm}mm → factor delta {factor_delta_relaxed[i]:.3f}")

    # No rounding — min_step_mm already ensures changes are discretization-meaningful
    factor_new_raw = factor_old + factor_delta_relaxed
    factor_new_rounded = factor_new_raw

    # Enforce minimum fracture zone thickness (manufacturing constraint)
    min_fracture_thickness_mm = float(opt_cfg.get("min_fracture_thickness_mm", 0.4))
    for i in range(n_bands):
        # fracture_thickness = factor × protected_thickness >= min_fracture_thickness
        # factor >= min_fracture_thickness / protected_thickness
        min_factor_for_band = min_fracture_thickness_mm / protected_thickness[i]
        if factor_new_rounded[i] < min_factor_for_band:
            factor_new_rounded[i] = min_factor_for_band
            print(f"[Fracture] Band {i}: clamped factor to {min_factor_for_band:.3f} (min thickness constraint)")

    # Compute actual delta after rounding and clamping
    delta_final = factor_new_rounded - factor_old

    # Apply update (bounds enforced in ThetaBandParams)
    thickness_params.update_fracture_factor(delta_final)
    factor_new = thickness_params.fracture_factor

    # Build update info
    fracture_update_info = {
        "fracture_factor_old": factor_old.tolist(),
        "fracture_factor_new": factor_new.tolist(),
        "fracture_factor_change": (factor_new - factor_old).tolist(),
        "D_fracture_zone": D_frac,
        "D_shell_body": D_shell,
        "breach": breach,
        "target_D_fracture": target_D_frac,
        "max_D_shell": max_D_shell,
        "bisection_used": bisection_used,
        "bisection_history": _fracture_bisection_history.copy() if use_fracture_bisection else None,
    }

    return thickness_params, fracture_update_info


def _compute_stress_by_theta_band(
    fea_results: Dict,
    mesh_data,
    thickness_params: ThetaBandParams,
    opt_cfg: Dict,
) -> np.ndarray:
    """
    Compute average (or max) stress for each theta band.

    Uses FEA stress field and mesh cell centers to assign stresses to bands.
    Prefers pre-computed stress_by_band from FEA adapter if available.

    Args:
        fea_results: FEA results with 'stress_field' or 'stress_by_band_MPa'
        mesh_data: Mesh data with 'cell_centers' (xyz coordinates)
        thickness_params: Theta band parameterization
        opt_cfg: Optimization config with 'stress_band_stat'

    Returns:
        Array of stress values per band (Pa)
    """
    stat = str(opt_cfg.get("stress_band_stat", "max")).strip().lower()

    # Check if FEA adapter already computed stress by band (preferred - more accurate)
    if stat == "max" and "stress_by_band_MPa" in fea_results:
        stress_by_band_mpa = np.array(fea_results["stress_by_band_MPa"])
        print(f"[Thickness Updater] Using pre-computed stress by band from FEA adapter")
        return stress_by_band_mpa * 1e6  # Convert MPa to Pa
    elif stat == "adaptive":
        # ADAPTIVE MODE: Use P80 for pole bands (contact spikes), P95 for equator bands
        # This handles the issue where small bands near the pole have few cells
        # and P95 still catches contact stress spikes
        p95_bands = np.array(fea_results.get("stress_p95_by_band_MPa", []))
        p80_bands = np.array(fea_results.get("stress_p80_by_band_MPa", []))

        if len(p95_bands) == 0 or len(p80_bands) == 0:
            print(f"[Thickness Updater] WARNING: P95/P80 not available, falling back to max")
            stress_by_band_mpa = np.array(fea_results.get("stress_by_band_MPa", []))
            return stress_by_band_mpa * 1e6 if len(stress_by_band_mpa) > 0 else None

        # Get bands that should use P80 (pole bands with contact spikes)
        adaptive_bands = opt_cfg.get("adaptive_percentile_bands", [0, 1, 2, 3])

        # Build adaptive stress array: P80 for pole bands, P95 for equator bands
        stress_by_band_mpa = p95_bands.copy()  # Default to P95
        for band_idx in adaptive_bands:
            if band_idx < len(stress_by_band_mpa):
                stress_by_band_mpa[band_idx] = p80_bands[band_idx]

        print(f"[Thickness Updater] Using ADAPTIVE percentile: P80 for bands {adaptive_bands}, P95 for rest")
        print(f"  P80 (pole): {[f'{p80_bands[i]:.0f}' for i in adaptive_bands if i < len(p80_bands)]} MPa")
        print(f"  P95 (equator): {[f'{p95_bands[i]:.0f}' for i in range(len(p95_bands)) if i not in adaptive_bands]} MPa")
        return stress_by_band_mpa * 1e6  # Convert MPa to Pa
    elif stat == "p80" and "stress_p80_by_band_MPa" in fea_results:
        stress_by_band_mpa = np.array(fea_results["stress_p80_by_band_MPa"])
        print(f"[Thickness Updater] Using pre-computed P80 stress by band from FEA adapter")
        return stress_by_band_mpa * 1e6  # Convert MPa to Pa
    elif stat == "p95" and "stress_p95_by_band_MPa" in fea_results:
        stress_by_band_mpa = np.array(fea_results["stress_p95_by_band_MPa"])
        print(f"[Thickness Updater] Using pre-computed P95 stress by band from FEA adapter")
        return stress_by_band_mpa * 1e6  # Convert MPa to Pa
    elif stat in {"p99", "percentile"} and "stress_p99_by_band_MPa" in fea_results:
        stress_by_band_mpa = np.array(fea_results["stress_p99_by_band_MPa"])
        print(f"[Thickness Updater] Using pre-computed P99 stress by band from FEA adapter")
        return stress_by_band_mpa * 1e6  # Convert MPa to Pa

    # Fallback: compute from stress field (for backward compatibility or if preferred method not available)
    stress_field = fea_results.get("stress_field", None)
    percentile = float(opt_cfg.get("stress_band_percentile", 99.0))

    if stress_field is None:
        # Fallback: use worst-case stress for all bands (conservative)
        max_stress = fea_results.get("max_stress", 0.0)
        print(f"[Thickness Updater] WARNING: No stress field or pre-computed bands, using global max for all bands")
        return np.full(thickness_params.n_bands, max_stress)

    # Compute theta angle for each cell
    cell_centers = mesh_data.cell_centers
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    z = cell_centers[:, 2]

    # Theta = angle from z-axis (latitude)
    # theta = 0° at poles, 90° at equator
    r_xy = np.sqrt(x**2 + y**2)
    theta_rad = np.arctan2(r_xy, np.abs(z))
    theta_deg = np.rad2deg(theta_rad)

    # Assign cells to bands
    stress_by_band = np.zeros(thickness_params.n_bands)

    for band_idx in range(thickness_params.n_bands):
        theta_min = thickness_params.theta_edges_deg[band_idx]
        theta_max = thickness_params.theta_edges_deg[band_idx + 1]

        # Find cells in this band
        mask = (theta_deg >= theta_min) & (theta_deg < theta_max)

        if np.any(mask):
            band_vals = stress_field[mask]
            if stat == "max":
                stress_by_band[band_idx] = np.max(band_vals)
            elif stat in {"p95", "p99", "percentile"}:
                if stat in {"p95", "p99"}:
                    pct = 95.0 if stat == "p95" else 99.0
                else:
                    pct = percentile
                stress_by_band[band_idx] = np.percentile(band_vals, pct)
            else:
                # Fallback to max for unknown stat values
                stress_by_band[band_idx] = np.max(band_vals)
        else:
            # No cells in this band (shouldn't happen with proper mesh)
            stress_by_band[band_idx] = 0.0

    return stress_by_band


def check_convergence(
    thickness_params: ThetaBandParams,
    fea_results: Dict,
    update_info: Dict,
    history: list,
    config: Dict,
    peridigm_results: Dict = None,
) -> Tuple[bool, list]:
    """
    Check if optimization has converged.

    Convergence criteria:
    1. All theta bands have stress ≤ stress_limit (uses MAX across orientations)
    2. Thickness changes are small (stable)
    3. Fitness-based: total fitness < tolerance AND each component < its tolerance
    4. (Optional) Constrained convergence (stable at constraint limits)

    Args:
        thickness_params: Current thickness parameters
        fea_results: Latest FEA results
        update_info: Latest update info
        history: List of past update_info dicts
        config: Optimization configuration
        peridigm_results: Previous iteration Peridigm results (for fitness check)

    Returns:
        Tuple of (converged: bool, reasons: list of str)
    """
    opt_cfg = config.get("optimization", {})
    fea_cfg = config.get("fea", {})

    stress_limit_pa = float(fea_cfg.get("stress_limit_Pa", 44e6))
    thickness_tol_mm = float(opt_cfg.get("thickness_tolerance", 0.01))
    convergence_window = int(opt_cfg.get("convergence_window", 3))

    reasons = []

    # Check stress criterion
    # CONVERGENCE must use the MAX stress across orientations per band
    # (stress_by_band_max_MPa) — NOT the weighted average.  The weighted
    # average dramatically underestimates worst-case stress because it
    # down-weights the dominant orientation, leading to false convergence.
    stress_stat = str(opt_cfg.get("stress_band_stat", "max")).strip().lower()

    # Always prefer max-across-orientations for convergence checking
    if "stress_by_band_max_MPa" in fea_results:
        stress_for_convergence = np.array(fea_results["stress_by_band_max_MPa"]) * 1e6
        print(f"[Convergence] Using MAX stress across orientations per band (conservative)")
    elif stress_stat == "p95" and "stress_p95_by_band_max_MPa" in fea_results:
        stress_for_convergence = np.array(fea_results["stress_p95_by_band_max_MPa"]) * 1e6
        print(f"[Convergence] Using P95 max-across-orientations stress")
    elif "stress_by_band_MPa" in fea_results:
        # Fallback: weighted avg (log warning — this underestimates)
        stress_for_convergence = np.array(fea_results["stress_by_band_MPa"]) * 1e6
        print("[Convergence] WARNING: stress_by_band_max_MPa unavailable, "
              "falling back to weighted avg (may underestimate!)")
    else:
        # Last resort fallback from update_info
        stress_for_convergence = np.array(update_info.get("stress_by_band_MPa", [])) * 1e6
        print("[Convergence] WARNING: Using update_info stress (last resort fallback)")

    # Option to require stress convergence (default True for backwards compatibility)
    require_stress_convergence = opt_cfg.get("require_stress_convergence", True)

    if len(stress_for_convergence) > 0 and np.all(stress_for_convergence <= stress_limit_pa):
        reasons.append("stress")
        print(f"[Convergence] Stress criterion MET: all bands below {stress_limit_pa/1e6:.1f} MPa")
    elif len(stress_for_convergence) > 0:
        # Log which bands are over limit
        over_limit = stress_for_convergence > stress_limit_pa
        if np.any(over_limit):
            over_bands = np.where(over_limit)[0]
            over_values = stress_for_convergence[over_limit] / 1e6
            max_stress = np.max(stress_for_convergence) / 1e6
            print(f"[Convergence] Stress over limit in bands {over_bands.tolist()}: max={max_stress:.1f} MPa > {stress_limit_pa/1e6:.1f} MPa")

            # If stress convergence not required, still count as "stress" if improving
            if not require_stress_convergence:
                reasons.append("stress_waived")
                print(f"[Convergence] Stress convergence waived (require_stress_convergence=False)")

    # Check stability criterion (thickness changes small for N consecutive iterations)
    if len(history) >= convergence_window:
        recent_changes = [
            h["thickness_change_norm"]
            for h in history[-convergence_window:]
        ]
        if all(change <= thickness_tol_mm for change in recent_changes):
            reasons.append("stability")

    # Check for CONSTRAINED CONVERGENCE
    # If thickness is stable but stress goal isn't met, check if we're constraint-limited
    # This prevents infinite loops when optimizer wants to change but can't due to constraints
    has_stability = "stability" in reasons
    has_stress = "stress" in reasons or "stress_waived" in reasons

    if has_stability and not has_stress:
        # We're stable but not at stress target - check if constrained
        min_thickness_mm = float(config.get("material", {}).get("min_thickness_mm", 1.0))
        max_thickness_mm = float(config.get("material", {}).get("max_thickness_mm", 10.0))
        max_gradient_mm = float(opt_cfg.get("max_thickness_gradient_mm", 0.0))

        current_thickness = thickness_params.thickness_mm
        n_at_min = np.sum(np.abs(current_thickness - min_thickness_mm) < 0.01)
        n_at_max = np.sum(np.abs(current_thickness - max_thickness_mm) < 0.01)

        # Check if gradient constraint is limiting (adjacent bands differ by max_gradient)
        gradient_limited = False
        if max_gradient_mm > 0:
            for i in range(len(current_thickness) - 1):
                diff = abs(current_thickness[i] - current_thickness[i + 1])
                if abs(diff - max_gradient_mm) < 0.02:  # Within 0.02mm of gradient limit
                    gradient_limited = True
                    break

        # Constrained convergence: stable + at constraint limits + stress within 20% of target
        # Without the stress check, optimizer can declare convergence at 2x over target
        max_stress_ratio = float(np.max(stress_for_convergence)) / stress_limit_pa if len(stress_for_convergence) > 0 else 999.0
        stress_close_enough = max_stress_ratio <= 1.2  # Within 20% of target

        if (n_at_min > 0 or n_at_max > 0 or gradient_limited) and stress_close_enough:
            reasons.append("constrained_convergence")
            print(f"[Convergence] CONSTRAINED CONVERGENCE detected:")
            print(f"  - Bands at min ({min_thickness_mm}mm): {n_at_min}")
            print(f"  - Bands at max ({max_thickness_mm}mm): {n_at_max}")
            print(f"  - Gradient limited: {gradient_limited}")
            print(f"  - Max stress ratio: {max_stress_ratio:.2f} (within 20% threshold)")
            print(f"  - Accepting current design")
        elif (n_at_min > 0 or n_at_max > 0 or gradient_limited) and not stress_close_enough:
            print(f"[Convergence] Constrained but stress too high ({max_stress_ratio:.2f}x target) - NOT converged")

    # === FITNESS-BASED CONVERGENCE ===
    # Check if total fitness < tolerance AND each component < its own tolerance
    # Uses current FEA stress + previous iteration Peridigm results
    fitness_tol = opt_cfg.get("fitness_convergence", {})
    fitness_total_tol = float(fitness_tol.get("total_tolerance", 0.2))
    fitness_stress_tol = float(fitness_tol.get("stress_tolerance", 0.1))
    fitness_fracture_tol = float(fitness_tol.get("fracture_miss_tolerance", 0.1))
    fitness_breach_required = bool(fitness_tol.get("breach_required", True))
    fitness_shell_tol = float(fitness_tol.get("shell_damage_tolerance", 0.1))
    fitness_window = int(fitness_tol.get("window", 3))

    has_fitness = False
    if peridigm_results is not None and len(history) >= fitness_window:
        # Compute fitness from current FEA + previous Peridigm
        stress_key = "stress_by_band_max_MPa" if "stress_by_band_max_MPa" in fea_results else "stress_by_band_MPa"
        stress_bands_mpa = np.array(fea_results.get(stress_key, []))
        target_stress_mpa = stress_limit_pa / 1e6

        if len(stress_bands_mpa) > 0:
            # Stress violation
            max_stress_mpa = float(np.max(stress_bands_mpa))
            stress_violation = max(0.0, max_stress_mpa - target_stress_mpa) / target_stress_mpa

            # Fracture metrics from previous Peridigm
            frac_metrics = peridigm_results.get("fracture_metrics", {})
            completeness = frac_metrics.get("completeness", 0.0)
            breach = frac_metrics.get("breach", False)
            if not frac_metrics:
                D_frac = peridigm_results.get("D_fracture_zone", 0.0)
                completeness = D_frac
                breach = D_frac > 0.5

            target_completeness = float(opt_cfg.get("fracture_completeness_target",
                                   config.get("peridigm", {}).get("fracture_completeness_target", 0.5)))
            fracture_miss = max(0.0, target_completeness - completeness)

            # Breach penalty
            no_breach = 0.0 if breach else 1.0

            # Shell damage
            shell_metrics = peridigm_results.get("shell_metrics", {})
            integrity = shell_metrics.get("integrity", 1.0)
            min_integrity = float(config.get("peridigm", {}).get("min_shell_integrity", 0.8))
            shell_excess = max(0.0, min_integrity - integrity)

            # Weighted total (same weights as fitness function)
            total = (1.0 * stress_violation + 1.0 * fracture_miss +
                     2.0 * no_breach + 0.0 * shell_excess)

            # Check component tolerances
            stress_ok = stress_violation <= fitness_stress_tol
            fracture_ok = fracture_miss <= fitness_fracture_tol
            breach_ok = breach if fitness_breach_required else True
            shell_ok = shell_excess <= fitness_shell_tol
            total_ok = total <= fitness_total_tol

            print(f"[Fitness] total={total:.3f} (tol={fitness_total_tol}), "
                  f"stress={stress_violation:.3f} (tol={fitness_stress_tol}), "
                  f"frac_miss={fracture_miss:.3f} (tol={fitness_fracture_tol}), "
                  f"breach={'Y' if breach else 'N'}, shell={shell_excess:.3f}")

            if total_ok and stress_ok and fracture_ok and breach_ok and shell_ok:
                # Check if fitness has been good for the window
                # Look at recent history entries for sustained fitness
                recent_fitness_ok = True
                for h in history[-fitness_window:]:
                    h_frac = h.get("fracture_metrics", {})
                    h_breach = h_frac.get("breach", False)
                    h_compl = h_frac.get("completeness", 0.0)
                    if not h_frac:
                        h_d = h.get("D_fracture_zone", 0.0)
                        h_breach = h_d > 0.5
                        h_compl = h_d
                    h_fm = max(0.0, target_completeness - h_compl)

                    h_stress_key = "stress_p95_by_band_MPa" if "stress_p95_by_band_MPa" in h else "stress_by_band_MPa"
                    h_stress = h.get(h_stress_key, [0])
                    h_max_s = max(h_stress) if h_stress else 0
                    h_sv = max(0.0, h_max_s - target_stress_mpa) / target_stress_mpa
                    h_total = 1.0 * h_sv + 1.0 * h_fm + 2.0 * (0.0 if h_breach else 1.0)

                    if h_total > fitness_total_tol or h_sv > fitness_stress_tol or h_fm > fitness_fracture_tol:
                        recent_fitness_ok = False
                        break
                    if fitness_breach_required and not h_breach:
                        recent_fitness_ok = False
                        break

                if recent_fitness_ok:
                    has_fitness = True
                    reasons.append("fitness")
                    print(f"[Convergence] FITNESS CONVERGED: total={total:.3f} for {fitness_window} consecutive iterations")
            else:
                components_failing = []
                if not stress_ok: components_failing.append(f"stress({stress_violation:.3f}>{fitness_stress_tol})")
                if not fracture_ok: components_failing.append(f"frac({fracture_miss:.3f}>{fitness_fracture_tol})")
                if not breach_ok: components_failing.append("no_breach")
                if not total_ok: components_failing.append(f"total({total:.3f}>{fitness_total_tol})")
                print(f"[Fitness] Not converged: {', '.join(components_failing)}")

    # Converged if:
    # 1. Both stress + stability met (ideal convergence), OR
    # 2. Constrained convergence (stable at constraint limits), OR
    # 3. Fitness converged (total + all components within tolerance for N iterations)
    has_constrained = "constrained_convergence" in reasons
    converged = (has_stress and has_stability) or has_constrained or has_fitness

    return converged, reasons
