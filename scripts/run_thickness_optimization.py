#!/usr/bin/env python3
"""FEA-only shell thickness optimization driver for PeriFEATO.

Drives the optimization loop from the perifeato library. Adapters
(mesh, FEA) call into the underlying mesh/FEA tooling, which they
locate via the OF_ROOT env var (defaults to ~/Optimization_Framework
for now).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PeriFEATO shell thickness optimization (FEA-only)",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    # OF_ROOT points at the directory containing modules/{theta_mesh,fea_truth_contact}.
    # Falls back to ~/Optimization_Framework if not set.
    if "OF_ROOT" not in os.environ:
        default_of_root = Path.home() / "Optimization_Framework"
        os.environ["OF_ROOT"] = str(default_of_root)
    of_root = Path(os.environ["OF_ROOT"])

    perifeato_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(perifeato_root))

    from perifeato.optimization.shell_thickness.thickness_loop import (
        run_thickness_optimization,
    )

    print(f"PERIFEATO_ROOT: {perifeato_root}")
    print(f"OF_ROOT:        {of_root}")
    print(f"Config:         {args.config}")
    print()

    thickness_params, history = run_thickness_optimization(args.config)

    print()
    print("=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Iterations: {len(history)}")
    if history:
        last = history[-1]
        print(f"Final stress: {last.get('max_stress_MPa', 'N/A')} MPa")
        print(f"Converged:    {last.get('converged', False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
