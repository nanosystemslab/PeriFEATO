#!/usr/bin/env python3
"""
Generate mesh for dual-physics optimization (mesh only, no FEA).

This script generates the mesh and Peridigm discretization so that
Peridigm can be submitted immediately while FEA runs in parallel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description="Generate mesh for dual-physics optimization")
    parser.add_argument("--config", required=True, help="Configuration YAML file")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    # Add modules to path
    perifeato_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(perifeato_root))

    from perifeato.optimization.shell_thickness.thickness_loop import (
        _load_config,
        _initialize_thickness_params,
        _update_config_with_thickness,
    )
    from perifeato.optimization.adapters.mesh_adapter import load_mesh_for_mvp
    from perifeato.optimization.shell_thickness.theta_band_params import ThetaBandParams
    from perifeato.optimization.shell_thickness.thickness_updater import (
        get_momentum_history,
        set_momentum_history,
        get_secant_history,
        set_secant_history,
    )

    # Load config
    config = _load_config(args.config)
    output_dir = Path(args.output_dir)
    iteration = args.iteration
    state_file = output_dir / "optimization_state.json"

    print(f"\n{'=' * 70}")
    print(f"MESH GENERATION - Iteration {iteration}")
    print(f"{'=' * 70}")

    # Load or initialize thickness params
    if state_file.exists() and iteration > 1:
        with open(state_file, "r") as f:
            state = json.load(f)
        thickness_params = ThetaBandParams.from_dict(state["thickness_params"])
        history = state.get("history", [])
        # Restore momentum history if continuing from previous run
        if "momentum_history" in state:
            set_momentum_history(state["momentum_history"])
        # Restore secant history if continuing from previous run
        if "secant_history" in state:
            set_secant_history(state["secant_history"])
        print(f"Loaded state from iteration {state['iteration']}")
    else:
        thickness_params = _initialize_thickness_params(config)
        history = []
        print("Initialized new optimization")

    print(f"Current thickness: {thickness_params.thickness_mm}")

    # Update config with current thickness
    config_iter = _update_config_with_thickness(config, thickness_params)

    # Generate mesh (this also exports Peridigm discretization if enabled)
    print("\nGenerating mesh...")
    mesh_data = load_mesh_for_mvp(config_iter)
    print(f"  Mesh cells: {mesh_data.n_cells}")
    print(f"  Mesh nodes: {mesh_data.n_nodes if hasattr(mesh_data, 'n_nodes') else 'N/A'}")

    # Get mesh output directory
    mesh_cfg = config_iter.get("mesh", {})
    mesh_output_dir = mesh_cfg.get("output_dir", str(output_dir / "meshes"))

    # Check if Peridigm discretization was generated
    peridigm_files = list(Path(mesh_output_dir).rglob("*_peridigm.txt"))
    if peridigm_files:
        print(f"  Peridigm discretization: {peridigm_files[0]}")
    else:
        print("  WARNING: No Peridigm discretization found")

    # Save state for next phases (include momentum and secant history for continuation)
    # Load existing state to preserve keys set by other phases (e.g., cone_half_angle_deg)
    existing_state = {}
    if state_file.exists():
        with open(state_file, "r") as f:
            existing_state = json.load(f)

    state = existing_state
    state.update({
        "iteration": iteration,
        "thickness_params": thickness_params.to_dict(),
        "history": history,
        "momentum_history": get_momentum_history(),
        "secant_history": get_secant_history(),
        "mesh_output_dir": mesh_output_dir,
        "mesh_n_cells": mesh_data.n_cells,
        "mesh_generated": True,
        "fea_completed": False,
        "peridigm_pending": True,
        "converged": False,
    })

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    print(f"\nMesh generation complete.")
    print(f"State saved to: {state_file}")
    print(f"Peridigm can now be submitted while FEA runs.")


if __name__ == "__main__":
    main()
