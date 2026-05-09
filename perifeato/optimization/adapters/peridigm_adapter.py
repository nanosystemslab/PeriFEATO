"""
Peridigm Adapter - Unified interface for fracture simulation
=============================================================

Routes Peridigm calls to:
1. Mock Peridigm (fast, local testing, heuristic damage model)
2. Real Peridigm (HPC-only, physics-based fracture simulation)

Configuration-driven backend selection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def run_peridigm(
    mesh_data,
    density: np.ndarray,
    fracture_zone_ids: List[int],
    shell_body_ids: List[int],
    impact_velocity: float,
    config: Dict,
    prepare_only: bool = False,
) -> Dict:
    """
    Unified Peridigm interface with backend routing.

    Config structure:
        peridigm:
            backend: "mock" | "real"
            mode: "local" | "hpc"
            ... backend-specific parameters

    Args:
        mesh_data: MeshData object with mesh and region info
        density: Per-element density field
        fracture_zone_ids: Region IDs for fracture zones
        shell_body_ids: Region IDs for main shell body
        impact_velocity: Impact velocity (m/s)
        config: Full configuration dict
        prepare_only: If True, only prepare files without submitting (for container mode)

    Returns:
        {
            'damage_field': np.ndarray,      # Per-element damage [0, 1]
            'D_fracture_zone': float,        # Average damage in fracture zone
            'D_shell_body': float,           # Average damage in shell body
            'backend': str,                  # Which backend was used
            'mode': str,                     # local or hpc
            'job_id': str,                   # HPC job ID (if HPC mode)
            'exodus_file': str,              # Path to Exodus file (if real)
            'metadata_file': str,            # Path to metadata file (if prepare_only)
        }
    """
    peridigm_cfg = config.get("peridigm", {})
    backend = peridigm_cfg.get("backend", "mock")
    mode = peridigm_cfg.get("mode", "local")

    if backend == "mock":
        return _run_mock_peridigm(
            mesh_data, density, fracture_zone_ids, shell_body_ids, impact_velocity, config
        )
    elif backend == "real":
        if mode != "hpc":
            raise ValueError(
                "Real Peridigm backend requires mode='hpc'. Peridigm only runs on HPC."
            )
        return _run_real_peridigm_hpc(
            mesh_data, density, fracture_zone_ids, shell_body_ids, impact_velocity, config,
            prepare_only=prepare_only
        )
    else:
        raise ValueError(f"Invalid Peridigm backend: {backend}. Must be 'mock' or 'real'.")


def _run_mock_peridigm(
    mesh_data,
    density: np.ndarray,
    fracture_zone_ids: List[int],
    shell_body_ids: List[int],
    impact_velocity: float,
    config: Dict,
) -> Dict:
    """
    Run mock Peridigm backend (existing MVP implementation).

    Fast, local-only, heuristic damage model.
    Good for testing optimization loop logic.
    """
    # Import the mock Peridigm module from the same directory
    sys.path.insert(0, str(Path(__file__).parent))
    from peridigm_mock import run_peridigm_mock

    # Extract mock-specific config
    peridigm_cfg = config.get("peridigm", {})

    results = run_peridigm_mock(
        mesh_data,
        density,
        fracture_zone_ids,
        shell_body_ids,
        impact_velocity,
        peridigm_cfg,
    )

    # Add adapter metadata
    results["backend"] = "mock"
    results["mode"] = "local"
    results["job_id"] = None
    results["exodus_file"] = None

    return results


def _run_real_peridigm_hpc(
    mesh_data,
    density: np.ndarray,
    fracture_zone_ids: List[int],
    shell_body_ids: List[int],
    impact_velocity: float,
    config: Dict,
    prepare_only: bool = False,
) -> Dict:
    """
    Run real Peridigm on HPC (SLURM job submission).

    Workflow:
    1. Export mesh to Peridigm text format (use theta_mesh export)
    2. Map density → Peridigm material properties
    3. Generate Peridigm YAML input
    4. If prepare_only: Write metadata file and return
    5. Else: Submit SLURM job, wait, parse results

    Args:
        prepare_only: If True, only prepare files and write metadata (for container mode)

    Uses modules/peridigm_fracture/hpc/run_core_shell_from_theta_mesh.slurm
    """
    peridigm_cfg = config.get("peridigm", {})
    hpc_cfg = peridigm_cfg.get("hpc", {})

    # Get OF_ROOT
    of_root = Path(os.environ.get("OF_ROOT", "/home/mtdsn/Optimization_Framework"))
    peridigm_module = of_root / "modules" / "peridigm_fracture"

    if not peridigm_module.exists():
        raise RuntimeError(
            f"Peridigm module not found at {peridigm_module}. "
            f"Ensure OF_ROOT is set correctly."
        )

    # Prepare work directory
    output_dir = Path(config.get("output", {}).get("dir", "results_peridigm"))
    output_dir = output_dir / "peridigm"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Peridigm Adapter] Preparing Peridigm job submission...")

    # Step 1: Export mesh to Peridigm format
    # Assume theta_mesh has already generated the Peridigm discretization
    # Located in mesh output directory
    mesh_cfg = config.get("mesh", {})
    theta_output = Path(mesh_cfg.get("output_dir", "meshes"))

    # Look for Peridigm discretization file
    peridigm_txt = _find_peridigm_discretization(theta_output)

    if peridigm_txt is None:
        raise RuntimeError(
            f"Peridigm discretization not found in {theta_output}. "
            f"Ensure theta_mesh generates Peridigm export."
        )

    print(f"[Peridigm Adapter] Using Peridigm discretization: {peridigm_txt}")

    # Step 2: Map density → material properties
    # For now, use uniform material properties
    # Future: implement SIMP-like mapping for density
    material_params = _density_to_peridigm_materials(density, config)

    # Step 3: Generate Peridigm YAML input
    peridigm_yaml = _generate_peridigm_yaml(
        peridigm_txt, material_params, impact_velocity, config
    )

    yaml_path = output_dir / "peridigm_input.yaml"
    with open(yaml_path, "w") as f:
        f.write(peridigm_yaml)

    print(f"[Peridigm Adapter] Generated Peridigm input: {yaml_path}")

    # If prepare_only mode, write metadata and return
    if prepare_only:
        metadata_file = output_dir / "peridigm_submission_metadata.json"
        metadata = {
            "theta_run_dir": str(peridigm_txt.parent),
            "yaml_path": str(yaml_path),
            "output_dir": str(output_dir),
            "n_cells": len(density),
            "fracture_zone_ids": fracture_zone_ids,
            "shell_body_ids": shell_body_ids,
            "config": config,
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Peridigm Adapter] Wrote metadata to: {metadata_file}")
        print("[Peridigm Adapter] Prepare-only mode: files ready for submission")

        return {
            "damage_field": None,
            "D_fracture_zone": None,
            "D_shell_body": None,
            "backend": "real",
            "mode": "hpc_prepare",
            "job_id": None,
            "exodus_file": None,
            "metadata_file": str(metadata_file),
        }

    # Step 4: Submit SLURM job
    job_id = _submit_peridigm_slurm_job(
        peridigm_txt.parent, yaml_path, output_dir, config
    )

    print(f"[Peridigm Adapter] Submitted Peridigm job: {job_id}")

    # Step 5: Wait for job completion
    timeout = hpc_cfg.get("timeout", 7200)  # 2 hour default
    _wait_for_slurm_job(job_id, timeout)

    print(f"[Peridigm Adapter] Peridigm job {job_id} completed")

    # Step 6: Parse Exodus output
    exodus_file = output_dir / f"peridigm_{job_id}.exo"
    damage_field = _parse_exodus_damage(exodus_file, len(density))

    # Step 7: Compute damage metrics
    damage_metrics = _compute_damage_metrics(
        damage_field, mesh_data, fracture_zone_ids, shell_body_ids
    )

    # Return results
    return {
        "damage_field": damage_field,
        "D_fracture_zone": damage_metrics["D_fracture_zone"],
        "D_shell_body": damage_metrics["D_shell_body"],
        "backend": "real",
        "mode": "hpc",
        "job_id": job_id,
        "exodus_file": str(exodus_file),
    }


def _find_peridigm_discretization(theta_output: Path) -> Path:
    """
    Find Peridigm discretization file in theta_mesh output.

    Looks for:
    - peridigm/*_peridigm.txt
    - results_refined/peridigm/*_peridigm.txt
    """
    # Common patterns
    patterns = [
        theta_output / "peridigm" / "*_peridigm.txt",
        theta_output / "results_refined" / "peridigm" / "*_peridigm.txt",
    ]

    for pattern in patterns:
        matches = list(pattern.parent.glob(pattern.name))
        if matches:
            return matches[0]  # Return first match

    return None


def _density_to_peridigm_materials(density: np.ndarray, config: Dict) -> Dict:
    """
    Map density field to Peridigm material properties.

    For now, uses uniform material properties.
    Future: implement density-dependent critical stretch.

    Returns:
        {
            'pla_critical_stretch': float,
            'fracture_critical_stretch': float,
            'horizon_mm': float,
        }
    """
    peridigm_cfg = config.get("peridigm", {})

    return {
        "pla_critical_stretch": peridigm_cfg.get("pla_critical_stretch", 0.01),
        "fracture_critical_stretch": peridigm_cfg.get(
            "fracture_zone_critical_stretch", 0.005
        ),
        "horizon_mm": peridigm_cfg.get("horizon_mm", 2.0),
    }


def _generate_peridigm_yaml(
    peridigm_txt: Path,
    material_params: Dict,
    impact_velocity: float,
    config: Dict,
) -> str:
    """
    Generate Peridigm YAML input file.

    NOTE: This is the legacy inline generator. For optimization campaigns,
    use the flyback split core approach via run_flyback_split_core.slurm
    which calls split_core_generator.py with proper:
    - Self-contact (prevents steel folding)
    - Optimized contact parameters (Spring Constant 1e12)
    - Split steel core for realistic flyback physics
    - Optimized FINAL_TIME (100µs)

    This function is kept for backwards compatibility with simple tests.
    """
    peridigm_cfg = config.get("peridigm", {})

    yaml_content = f"""# Peridigm Input - Generated by optimization framework (legacy)
# For production use: run_flyback_split_core.slurm with split_core_generator.py
# Vertical cone impact at {impact_velocity} m/s

Discretization:
  Type: "Text File"
  Input Mesh File: "{peridigm_txt}"

Materials:
  PLA Shell:
    Material Model: "Elastic"
    Density: 1250.0
    Bulk Modulus: 3.0e9
    Horizon: {material_params['horizon_mm']}e-3
    Critical Stretch: {material_params['pla_critical_stretch']}

  Fracture Zone:
    Material Model: "Elastic"
    Density: 1250.0
    Bulk Modulus: 3.0e9
    Horizon: {material_params['horizon_mm']}e-3
    Critical Stretch: {material_params['fracture_critical_stretch']}

Blocks:
  PLA Shell Block:
    Block Names: "block_1"
    Material: "PLA Shell"
    Horizon: {material_params['horizon_mm']}e-3

  Fracture Zone Block:
    Block Names: "block_2"
    Material: "Fracture Zone"
    Horizon: {material_params['horizon_mm']}e-3

Boundary Conditions:
  Cone Impact:
    Type: "Initial Velocity"
    Node Set: "cone_nodes"
    Coordinate: "z"
    Value: "{-impact_velocity}"

Solver:
  Initial Time: 0.0
  Final Time: {peridigm_cfg.get('simulation_time_s', 0.001)}
  Verlet:
    Safety Factor: 0.9

Output:
  Output File Type: "ExodusII"
  Output Filename: "peridigm_output"
  Output Frequency: 10
  Output Variables:
    - "Displacement"
    - "Velocity"
    - "Damage"
    - "Number_Of_Neighbors"
"""

    return yaml_content


def _submit_peridigm_slurm_job(
    theta_run_dir: Path, yaml_path: Path, output_dir: Path, config: Dict
) -> str:
    """
    Submit Peridigm SLURM job using flyback split core simulation.

    Uses modules/peridigm_fracture/hpc/run_flyback_split_core.slurm which:
    - Splits steel core into two halves for realistic flyback physics
    - Includes self-contact to prevent steel folding
    - Uses optimized contact parameters (Spring Constant 1e12)
    - Runs for 100µs (captures impact + fracture, avoids bouncing artifacts)

    Note: This function needs to run outside a container to access sbatch.
    """
    import shutil

    peridigm_cfg = config.get("peridigm", {})
    hpc_cfg = peridigm_cfg.get("hpc", {})

    # Check if sbatch is available
    sbatch_path = shutil.which("sbatch")
    if sbatch_path is None:
        # We're likely inside a container - write submission script instead
        raise RuntimeError(
            "sbatch command not found. This function must be called outside the container.\n"
            "The Peridigm job submission requires access to SLURM commands which are not "
            "available inside containers.\n\n"
            "Solution: Run Peridigm submission test outside the container or restructure "
            "to have the SLURM wrapper script handle Peridigm job submission."
        )

    of_root = Path(os.environ.get("OF_ROOT", "/home/mtdsn/Optimization_Framework"))

    # Use flyback split core SLURM script for production
    slurm_script = (
        of_root / "modules" / "peridigm_fracture" / "hpc" / "run_flyback_split_core.slurm"
    )

    if not slurm_script.exists():
        raise RuntimeError(f"Peridigm SLURM script not found: {slurm_script}")

    # Find the Peridigm discretization file
    peridigm_disc = _find_peridigm_discretization(theta_run_dir)
    if peridigm_disc is None:
        raise RuntimeError(f"No Peridigm discretization found in {theta_run_dir}")

    # Environment variables for flyback split core SLURM script
    env = os.environ.copy()
    env["SOURCE_DISC"] = str(peridigm_disc)

    # Flyback parameters
    env["IMPACT_VZ"] = str(peridigm_cfg.get("impact_velocity", 250))
    env["FINAL_TIME"] = str(peridigm_cfg.get("simulation_time_s", 1.0e-4))  # 100µs default
    env["GAP"] = str(peridigm_cfg.get("steel_gap_m", 0.0005))  # 0.5mm tolerance gap
    env["SPLIT_AXIS"] = str(peridigm_cfg.get("split_axis", "x"))

    # Peridynamic material parameters
    env["PLA_HORIZON"] = str(peridigm_cfg.get("horizon_mm", 9.0) / 1000.0)  # Convert mm to m
    env["PLA_CRIT_STRETCH"] = str(peridigm_cfg.get("pla_critical_stretch", 0.011))
    env["FRACTURE_CRIT_STRETCH"] = str(
        peridigm_cfg.get("fracture_zone_critical_stretch", 0.007)
    )

    # SLURM options
    partition = hpc_cfg.get("partition", "shared")
    nodes = hpc_cfg.get("nodes", 1)
    ntasks = hpc_cfg.get("ntasks", 8)
    time_limit = hpc_cfg.get("time_limit", "01:00:00")  # 1 hour for 100µs sim

    # Build export string for SLURM - sbatch needs explicit --export to pass env vars
    export_vars = [
        f"SOURCE_DISC={env['SOURCE_DISC']}",
        f"IMPACT_VZ={env['IMPACT_VZ']}",
        f"FINAL_TIME={env['FINAL_TIME']}",
        f"GAP={env['GAP']}",
        f"SPLIT_AXIS={env['SPLIT_AXIS']}",
        f"PLA_HORIZON={env['PLA_HORIZON']}",
        f"PLA_CRIT_STRETCH={env['PLA_CRIT_STRETCH']}",
        f"FRACTURE_CRIT_STRETCH={env['FRACTURE_CRIT_STRETCH']}",
        f"OUTDIR={output_dir}",  # Tell flyback script where to put output
    ]
    export_string = ",".join(export_vars)

    # Submit job
    cmd = [
        sbatch_path,
        f"--partition={partition}",
        f"--nodes={nodes}",
        f"--ntasks={ntasks}",
        f"--time={time_limit}",
        f"--output={output_dir}/peridigm_%j.log",
        f"--error={output_dir}/peridigm_%j.err",
        f"--export=ALL,{export_string}",  # CRITICAL: explicitly export env vars
        "--parsable",  # Returns only job ID
        str(slurm_script),
    ]

    print(f"[Peridigm Adapter] Submitting flyback split core job:")
    print(f"  SOURCE_DISC: {env['SOURCE_DISC']}")
    print(f"  IMPACT_VZ: {env['IMPACT_VZ']} m/s")
    print(f"  FINAL_TIME: {env['FINAL_TIME']} s")
    print(f"  GAP: {env['GAP']} m")

    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, check=True
    )

    job_id = result.stdout.strip()
    return job_id


def _wait_for_slurm_job(job_id: str, timeout: int):
    """
    Wait for SLURM job to complete.

    Args:
        job_id: SLURM job ID
        timeout: Maximum wait time in seconds

    Raises:
        TimeoutError: If job doesn't complete within timeout
        RuntimeError: If job fails
    """
    start_time = time.time()
    check_interval = 30  # Check every 30 seconds

    print(f"[Peridigm Adapter] Waiting for job {job_id} (timeout: {timeout}s)...")

    while True:
        # Check job status
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0 or not result.stdout.strip():
            # Job no longer in queue - check if completed successfully
            result = subprocess.run(
                ["sacct", "-j", job_id, "-n", "-o", "State"],
                capture_output=True,
                text=True,
            )

            state = result.stdout.strip().split()[0] if result.stdout.strip() else "UNKNOWN"

            if state == "COMPLETED":
                print(f"[Peridigm Adapter] Job {job_id} completed successfully")
                return
            else:
                raise RuntimeError(
                    f"Peridigm job {job_id} failed with state: {state}"
                )

        # Job still running - check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            # Cancel job
            subprocess.run(["scancel", job_id])
            raise TimeoutError(
                f"Peridigm job {job_id} exceeded timeout of {timeout}s"
            )

        # Wait before next check
        time.sleep(check_interval)


def _parse_exodus_damage(exodus_file: Path, n_cells: int) -> np.ndarray:
    """
    Parse damage field from Exodus output.

    Uses PyVista or netCDF4 to read Exodus file.

    Args:
        exodus_file: Path to Exodus .exo file
        n_cells: Expected number of cells

    Returns:
        Damage field as np.ndarray of shape (n_cells,)
    """
    try:
        import pyvista as pv
    except ImportError:
        raise RuntimeError(
            "PyVista required to parse Exodus files. Install with: pip install pyvista"
        )

    if not exodus_file.exists():
        raise RuntimeError(f"Exodus file not found: {exodus_file}")

    # Read Exodus file
    mesh = pv.read(str(exodus_file))

    # Extract damage field (final timestep)
    if "Damage" not in mesh.array_names:
        raise RuntimeError(f"Damage field not found in {exodus_file}")

    damage = mesh["Damage"]

    # Ensure correct size
    if len(damage) != n_cells:
        print(
            f"[Peridigm Adapter] Warning: Exodus has {len(damage)} cells, "
            f"expected {n_cells}. Using what's available."
        )

    return np.array(damage)


def _compute_damage_metrics(
    damage_field: np.ndarray,
    mesh_data,
    fracture_zone_ids: List[int],
    shell_body_ids: List[int],
) -> Dict:
    """
    Compute damage metrics for fracture zone and shell body.

    Args:
        damage_field: Per-element damage [0, 1]
        mesh_data: MeshData with cell_regions
        fracture_zone_ids: Region IDs for fracture zones
        shell_body_ids: Region IDs for shell body

    Returns:
        {
            'D_fracture_zone': float,  # Mean damage in fracture zone
            'D_shell_body': float,     # Mean damage in shell body
        }
    """
    # Get region masks
    fracture_mask = np.isin(mesh_data.cell_regions, fracture_zone_ids)
    shell_mask = np.isin(mesh_data.cell_regions, shell_body_ids)

    # Compute mean damage
    D_fracture = np.mean(damage_field[fracture_mask]) if fracture_mask.any() else 0.0
    D_shell = np.mean(damage_field[shell_mask]) if shell_mask.any() else 0.0

    return {
        "D_fracture_zone": D_fracture,
        "D_shell_body": D_shell,
    }


def parse_peridigm_results(metadata_file: Path, mesh_data) -> Dict:
    """
    Parse Peridigm results after job completes.

    Used in Phase 3 after Peridigm job submission completes.

    Args:
        metadata_file: Path to metadata JSON file from prepare phase
        mesh_data: Original mesh data

    Returns:
        Results dict with damage metrics
    """
    if not metadata_file.exists():
        raise RuntimeError(f"Metadata file not found: {metadata_file}")

    # Load metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    output_dir = Path(metadata["output_dir"])
    n_cells = metadata["n_cells"]
    fracture_zone_ids = metadata["fracture_zone_ids"]
    shell_body_ids = metadata["shell_body_ids"]

    # Find Peridigm job ID and Exodus file
    # Look for peridigm_*.exo files in output_dir
    exodus_files = list(output_dir.glob("peridigm_*.exo"))

    if not exodus_files:
        raise RuntimeError(
            f"No Exodus output found in {output_dir}. "
            f"Peridigm job may have failed."
        )

    # Use the most recent Exodus file
    exodus_file = max(exodus_files, key=lambda p: p.stat().st_mtime)
    print(f"[Peridigm Adapter] Parsing Exodus file: {exodus_file}")

    # Parse damage field
    damage_field = _parse_exodus_damage(exodus_file, n_cells)

    # Compute damage metrics
    damage_metrics = _compute_damage_metrics(
        damage_field, mesh_data, fracture_zone_ids, shell_body_ids
    )

    # Extract job ID from filename (peridigm_<jobid>.exo)
    job_id = exodus_file.stem.replace("peridigm_", "")

    return {
        "damage_field": damage_field,
        "D_fracture_zone": damage_metrics["D_fracture_zone"],
        "D_shell_body": damage_metrics["D_shell_body"],
        "backend": "real",
        "mode": "hpc",
        "job_id": job_id,
        "exodus_file": str(exodus_file),
    }


def validate_peridigm_config(config: Dict) -> List[str]:
    """
    Validate Peridigm configuration.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    peridigm_cfg = config.get("peridigm", {})

    # Check backend
    backend = peridigm_cfg.get("backend")
    if backend not in {"mock", "real"}:
        errors.append(f"peridigm.backend must be 'mock' or 'real', got: {backend}")

    # Check mode
    mode = peridigm_cfg.get("mode")
    if mode not in {"local", "hpc"}:
        errors.append(f"peridigm.mode must be 'local' or 'hpc', got: {mode}")

    # Real backend requires HPC
    if backend == "real" and mode != "hpc":
        errors.append("peridigm.backend='real' requires mode='hpc'")

    # Check HPC config if using HPC
    if mode == "hpc":
        hpc_cfg = peridigm_cfg.get("hpc", {})
        required_hpc = ["partition", "ntasks", "time_limit"]
        for key in required_hpc:
            if key not in hpc_cfg:
                errors.append(f"peridigm.hpc.{key} is required for HPC mode")

    return errors


def compute_path_based_breach(
    points: np.ndarray,
    damage: np.ndarray,
    block_ids: np.ndarray,
    fracture_zone_ids: List[int],
    theta_band_edges_deg: List[float],
    horizon: float = 0.009,
    damage_threshold: float = 0.9,
) -> Dict:
    """
    Compute breach detection based on connected path through broken bonds.

    A breach occurs when there exists a continuous path of broken bonds
    from z_max to z_min within the fracture zone.

    Args:
        points: Nx3 array of point coordinates (x, y, z)
        damage: N array of damage values [0, 1]
        block_ids: N array of block IDs
        fracture_zone_ids: Block IDs that are fracture zones
        theta_band_edges_deg: Theta band edges in degrees [0, 9, 18, ..., 90]
        horizon: Neighbor search radius (m) - typically 3x mesh size
        damage_threshold: Damage level to consider a bond "broken"

    Returns:
        {
            'breach': bool,           # Did crack propagate from top to bottom?
            'crack_depth': float,     # How far crack got (0=top, 1=bottom)
            'blocked_at_band': int,   # Which theta band blocked (None if breach)
            'blocked_at_z': float,    # Z coordinate where crack stopped
            'path_length': int,       # Number of points in longest path
        }
    """
    from collections import deque
    from scipy.spatial import cKDTree

    # Filter to fracture zone points
    fracture_mask = np.isin(block_ids, fracture_zone_ids)
    fracture_indices = np.where(fracture_mask)[0]

    if len(fracture_indices) == 0:
        print("[Breach] No fracture zone points found")
        return {
            'breach': False,
            'crack_depth': 0.0,
            'blocked_at_band': None,
            'blocked_at_z': None,
            'path_length': 0,
        }

    # Get fracture zone point data
    frac_points = points[fracture_indices]
    frac_damage = damage[fracture_indices]
    frac_z = frac_points[:, 2]

    z_min, z_max = frac_z.min(), frac_z.max()
    z_range = z_max - z_min

    print(f"[Breach] Fracture zone: {len(fracture_indices)} points, z=[{z_min:.4f}, {z_max:.4f}]")

    # Find broken points (damage > threshold)
    broken_mask = frac_damage >= damage_threshold
    broken_indices = np.where(broken_mask)[0]  # Indices within fracture zone

    n_broken = len(broken_indices)
    pct_broken = 100.0 * n_broken / len(fracture_indices)
    print(f"[Breach] Broken points: {n_broken} ({pct_broken:.1f}% of fracture zone)")

    if n_broken == 0:
        print("[Breach] No broken points - no breach possible")
        return {
            'breach': False,
            'crack_depth': 0.0,
            'blocked_at_band': 0,  # Blocked at top
            'blocked_at_z': z_max,
            'path_length': 0,
        }

    # Get broken point coordinates
    broken_points = frac_points[broken_indices]
    broken_z = broken_points[:, 2]

    # Build KD-tree for neighbor search among broken points
    tree = cKDTree(broken_points)

    # Find starting points (broken points near z_max)
    z_top_threshold = z_max - 0.1 * z_range  # Top 10% of z range
    top_broken = np.where(broken_z >= z_top_threshold)[0]

    print(f"[Breach] Starting points near top: {len(top_broken)}")

    if len(top_broken) == 0:
        # No broken points at top - find the highest broken point
        highest_broken_idx = np.argmax(broken_z)
        blocked_z = broken_z[highest_broken_idx]
        crack_depth = (z_max - blocked_z) / z_range

        # Compute theta band at blocked location
        blocked_point = broken_points[highest_broken_idx]
        blocked_band = _compute_theta_band(blocked_point, theta_band_edges_deg)

        # Compute per-band breach scores (broken fraction, no connectivity)
        n_bands = len(theta_band_edges_deg) - 1
        edges_rad = np.deg2rad(theta_band_edges_deg)
        frac_r = np.sqrt(frac_points[:, 0]**2 + frac_points[:, 1]**2 + frac_points[:, 2]**2)
        frac_theta = np.where(frac_r > 1e-10,
                              np.arccos(np.clip(np.abs(frac_points[:, 2]) / frac_r, 0, 1)),
                              0.0)
        frac_bands = np.clip(np.searchsorted(edges_rad, frac_theta, side='right') - 1, 0, n_bands - 1)

        broken_r = np.sqrt(broken_points[:, 0]**2 + broken_points[:, 1]**2 + broken_points[:, 2]**2)
        broken_theta = np.where(broken_r > 1e-10,
                                np.arccos(np.clip(np.abs(broken_points[:, 2]) / broken_r, 0, 1)),
                                0.0)
        broken_bands = np.clip(np.searchsorted(edges_rad, broken_theta, side='right') - 1, 0, n_bands - 1)

        band_breach_scores = []
        print(f"[Breach] Per-band breach scores (no-path mode):")
        for b in range(n_bands):
            total_in_band = int(np.sum(frac_bands == b))
            broken_in_band = int(np.sum(broken_bands == b))
            if total_in_band == 0:
                band_breach_scores.append(0.0)
                continue
            score = broken_in_band / total_in_band
            band_breach_scores.append(float(score))
            status = "FULL_BREACH" if score >= 0.7 else "BREACH" if score >= 0.4 else "PARTIAL" if score > 0 else "INTACT"
            print(f"  Band {b}: {broken_in_band}/{total_in_band} broken → score={score:.2f} [{status}]")

        print(f"[Breach] No broken points at top - crack blocked at z={blocked_z:.4f} (band {blocked_band})")
        return {
            'breach': False,
            'crack_depth': crack_depth,
            'blocked_at_band': blocked_band,
            'blocked_at_z': float(blocked_z),
            'path_length': 0,
            'band_breach_scores': band_breach_scores,
        }

    # BFS from top broken points to find connected path to bottom
    visited = set()
    queue = deque()

    # Initialize with top broken points
    for idx in top_broken:
        queue.append(idx)
        visited.add(idx)

    # Track the lowest z reached
    lowest_z_reached = z_max
    lowest_z_point = None

    while queue:
        current_idx = queue.popleft()
        current_z = broken_z[current_idx]

        if current_z < lowest_z_reached:
            lowest_z_reached = current_z
            lowest_z_point = broken_points[current_idx]

        # Find neighbors within horizon
        neighbor_indices = tree.query_ball_point(broken_points[current_idx], horizon)

        for neighbor_idx in neighbor_indices:
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                queue.append(neighbor_idx)

    # Check if we reached the bottom
    z_bottom_threshold = z_min + 0.1 * z_range  # Bottom 10% of z range
    breach = bool(lowest_z_reached <= z_bottom_threshold)

    crack_depth = (z_max - lowest_z_reached) / z_range
    path_length = len(visited)

    # --- Per-band breach score ---
    # For each band, compute: score = (connected_cells / total_cells) * z_span
    # This tells the optimizer exactly which bands are breaching vs intact.
    n_bands = len(theta_band_edges_deg) - 1
    edges_rad = np.deg2rad(theta_band_edges_deg)

    # Assign bands to ALL fracture zone points using abs(z) / distance-to-origin
    frac_r = np.sqrt(frac_points[:, 0]**2 + frac_points[:, 1]**2 + frac_points[:, 2]**2)
    frac_theta = np.where(frac_r > 1e-10,
                          np.arccos(np.clip(np.abs(frac_points[:, 2]) / frac_r, 0, 1)),
                          0.0)
    frac_bands = np.clip(np.searchsorted(edges_rad, frac_theta, side='right') - 1, 0, n_bands - 1)

    # Assign bands to broken points
    broken_r = np.sqrt(broken_points[:, 0]**2 + broken_points[:, 1]**2 + broken_points[:, 2]**2)
    broken_theta = np.where(broken_r > 1e-10,
                            np.arccos(np.clip(np.abs(broken_points[:, 2]) / broken_r, 0, 1)),
                            0.0)
    broken_bands = np.clip(np.searchsorted(edges_rad, broken_theta, side='right') - 1, 0, n_bands - 1)

    visited_arr = np.array(sorted(visited))

    band_breach_scores = []
    print(f"[Breach] Per-band breach scores:")
    for b in range(n_bands):
        total_in_band = int(np.sum(frac_bands == b))
        if total_in_band == 0:
            band_breach_scores.append(0.0)
            continue

        # Connected broken cells in this band
        conn_in_band = [v for v in visited_arr if broken_bands[v] == b]
        n_conn = len(conn_in_band)

        if n_conn == 0:
            band_breach_scores.append(0.0)
            print(f"  Band {b}: 0/{total_in_band} connected → score=0.00 [INTACT]")
            continue

        # Z-span of connected path within this band
        band_abs_z = np.abs(frac_points[frac_bands == b, 2])
        bz_range = band_abs_z.max() - band_abs_z.min()

        conn_abs_z = np.abs(broken_points[np.array(conn_in_band), 2])
        if bz_range > 1e-6:
            z_span = (conn_abs_z.max() - conn_abs_z.min()) / bz_range
        else:
            z_span = 1.0 if n_conn > 0 else 0.0

        score = (n_conn / total_in_band) * z_span
        band_breach_scores.append(float(score))

        status = "FULL_BREACH" if score >= 0.7 else "BREACH" if score >= 0.4 else "PARTIAL" if score > 0 else "INTACT"
        print(f"  Band {b}: {n_conn}/{total_in_band} connected, z-span={z_span:.0%} → score={score:.2f} [{status}]")

    # Compute blocked_band from the lowest point reached by BFS
    if lowest_z_point is not None and not breach:
        blocked_band = _compute_theta_band(lowest_z_point, theta_band_edges_deg)
    else:
        blocked_band = None

    result = {
        'crack_depth': 1.0 if breach else crack_depth,
        'blocked_at_band': None if breach else (blocked_band if lowest_z_point is not None else 0),
        'blocked_at_z': None if breach else float(lowest_z_reached),
        'path_length': path_length,
        'breach': breach,
        'band_breach_scores': band_breach_scores,
    }

    if breach:
        print(f"[Breach] SUCCESS! Crack propagated from z={z_max:.4f} to z={lowest_z_reached:.4f}")
        print(f"[Breach] Path length: {path_length} connected broken points")
    else:
        print(f"[Breach] BLOCKED at z={lowest_z_reached:.4f} (depth={crack_depth:.1%})")
        print(f"[Breach] Blocking band: {result['blocked_at_band']}")
        print(f"[Breach] Path length: {path_length} connected broken points")

    return result


def _compute_theta_band(point: np.ndarray, theta_band_edges_deg: List[float]) -> int:
    """
    Compute which theta band a point belongs to.

    Theta is measured from the z-axis (pole), so:
    - theta=0 at pole (z_max)
    - theta=90 at equator (z=0)

    Args:
        point: (x, y, z) coordinates
        theta_band_edges_deg: Band edges like [0, 9, 18, ..., 90]

    Returns:
        Band index (0 = pole, n-1 = equator)
    """
    x, y, z = point
    r = np.sqrt(x**2 + y**2 + z**2)

    if r < 1e-10:
        return 0  # At origin, assign to pole

    # Theta from z-axis (use abs(z) since structure is symmetric)
    theta_rad = np.arccos(np.abs(z) / r)
    theta_deg = np.degrees(theta_rad)

    # Find which band
    edges = np.array(theta_band_edges_deg)
    band = np.searchsorted(edges[1:], theta_deg)  # Skip first edge (0)
    band = min(band, len(edges) - 2)  # Clamp to valid range

    return int(band)


def read_peridigm_discretization(disc_file: Path) -> tuple:
    """
    Read Peridigm discretization file.

    Args:
        disc_file: Path to *_peridigm.txt file

    Returns:
        (points, block_ids, volumes) - Nx3 array, N array, N array
    """
    points = []
    block_ids = []
    volumes = []

    with open(disc_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                block_id = int(parts[3])
                volume = float(parts[4])
                points.append([x, y, z])
                block_ids.append(block_id)
                volumes.append(volume)

    return np.array(points), np.array(block_ids), np.array(volumes)


def read_exodus_damage_with_coords(exodus_dir: Path) -> tuple:
    """
    Read damage field and coordinates from Exodus output files.

    Handles parallel Exodus files (*.e.N.P format).

    Args:
        exodus_dir: Directory containing Exodus files

    Returns:
        (points, damage) - Nx3 array, N array
        Returns (None, None) if files not found
    """
    try:
        import pyvista as pv
    except ImportError:
        print("[Exodus] PyVista not available")
        return None, None

    # Find Exodus files
    exodus_files = sorted(exodus_dir.glob("*.e.*"))
    if not exodus_files:
        exodus_files = sorted(exodus_dir.glob("*.exo"))

    if not exodus_files:
        print(f"[Exodus] No Exodus files found in {exodus_dir}")
        return None, None

    all_points = []
    all_damage = []

    # Group by base name (handle parallel files)
    base_files = {}
    for ef in exodus_files:
        # Extract base name (before .e.N.P)
        name = ef.name
        if '.e.' in name:
            base = name.split('.e.')[0]
        else:
            base = name.rsplit('.', 1)[0]

        if base not in base_files:
            base_files[base] = []
        base_files[base].append(ef)

    # Use the first base name found
    base_name = list(base_files.keys())[0]
    files_to_read = base_files[base_name]

    print(f"[Exodus] Reading {len(files_to_read)} files for '{base_name}'")

    for ef in files_to_read:
        try:
            reader = pv.ExodusIIReader(str(ef))
            n_times = reader.number_time_points
            if n_times > 0:
                reader.set_active_time_point(n_times - 1)  # Last timestep

            mesh = reader.read()
            if isinstance(mesh, pv.MultiBlock):
                combined = mesh.combine()
            else:
                combined = mesh

            if combined.n_points == 0:
                continue

            # Get coordinates
            pts = np.array(combined.points)

            # Get damage
            dmg = None
            for name in combined.array_names:
                if 'damage' in name.lower():
                    dmg = np.array(combined[name]).flatten()
                    break

            if dmg is None:
                print(f"[Exodus] No damage field in {ef.name}")
                continue

            all_points.append(pts)
            all_damage.append(dmg)

        except Exception as e:
            print(f"[Exodus] Error reading {ef.name}: {e}")

    if not all_points:
        return None, None

    points = np.vstack(all_points)
    damage = np.concatenate(all_damage)

    print(f"[Exodus] Loaded {len(points)} points with damage")

    return points, damage
