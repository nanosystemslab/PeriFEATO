#!/usr/bin/env python3
"""
Process Peridigm results for dual-physics optimization.

This script is called from the SLURM wrapper after Peridigm job completes.
It parses the Exodus output and updates the optimization state with damage metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def read_damage_from_exodus_vtk(file_path: Path, target_time_s: float | None = None) -> dict:
    """
    Read damage and GlobalElementId from a single Exodus file using VTK.

    Args:
        file_path: Path to Exodus file
        target_time_s: If provided, read the timestep closest to this time (seconds).
                       If None, reads the LAST timestep (default behavior).

    Returns:
        Tuple of (damage_by_block, coords_by_block, globalid_damage_pairs)
        - damage_by_block: {block_id: [damage_values]}
        - coords_by_block: {block_id: [[x,y,z], ...]}  (reference coords)
        - globalid_damage_pairs: [(global_element_id, damage, block_id), ...]
    """
    import vtk
    from collections import defaultdict

    damage_by_block = defaultdict(list)
    coords_by_block = defaultdict(list)
    globalid_damage_pairs = []

    reader = vtk.vtkExodusIIReader()
    reader.SetFileName(str(file_path))

    # Use original (undeformed) coordinates as fallback for theta band computation
    reader.SetApplyDisplacements(False)

    reader.UpdateInformation()

    # Enable the Damage element array (it's element data, not point data)
    for i in range(reader.GetNumberOfElementResultArrays()):
        name = reader.GetElementResultArrayName(i)
        if name.lower() == "damage":
            reader.SetElementResultArrayStatus(name, 1)

    # Enable ObjectId and GlobalElementId generation
    reader.GenerateObjectIdCellArrayOn()
    reader.GenerateGlobalElementIdArrayOn()

    # Get available timesteps via VTK pipeline info
    n_timesteps = reader.GetNumberOfTimeSteps()
    time_values = None
    if n_timesteps > 1:
        try:
            info = reader.GetExecutive().GetOutputInformation(0)
            time_key = vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()
            if info.Has(time_key):
                n_times = info.Length(time_key)
                time_values = [info.Get(time_key, i) for i in range(n_times)]
        except Exception:
            pass

    if target_time_s is not None and time_values and len(time_values) > 1:
        # Find closest timestep to target time
        target_ts = int(np.argmin(np.abs(np.array(time_values) - target_time_s)))
        reader.SetTimeStep(target_ts)
    elif target_time_s is not None and n_timesteps > 0:
        # Fallback: pick last timestep if can't read time values
        reader.SetTimeStep(n_timesteps - 1)
    elif n_timesteps > 1:
        # Default: read the LAST timestep (final state with damage)
        reader.SetTimeStep(n_timesteps - 1)
    elif n_timesteps == 1:
        reader.SetTimeStep(0)

    reader.Update()
    output = reader.GetOutput()

    def extract_recursive(data_object):
        if data_object is None:
            return
        if data_object.IsA('vtkMultiBlockDataSet'):
            for i in range(data_object.GetNumberOfBlocks()):
                extract_recursive(data_object.GetBlock(i))
        else:
            cell_data = data_object.GetCellData() if hasattr(data_object, 'GetCellData') else None
            if cell_data is None:
                return

            # Get Damage array
            damage_arr = None
            for name in ["Damage", "damage", "DAMAGE"]:
                damage_arr = cell_data.GetArray(name)
                if damage_arr:
                    break

            object_id_arr = cell_data.GetArray("ObjectId")
            global_id_arr = cell_data.GetArray("GlobalElementId")

            if damage_arr is None:
                return

            n_cells = damage_arr.GetNumberOfTuples()
            for k in range(n_cells):
                damage_val = damage_arr.GetValue(k)
                block_id = int(object_id_arr.GetValue(k)) if object_id_arr else 0
                global_id = int(global_id_arr.GetValue(k)) if global_id_arr else -1

                damage_by_block[block_id].append(damage_val)
                globalid_damage_pairs.append((global_id, damage_val, block_id))

                # Reference coords (fallback if no disc file)
                cell = data_object.GetCell(k)
                bounds = cell.GetBounds()
                cx = (bounds[0] + bounds[1]) / 2.0
                cy = (bounds[2] + bounds[3]) / 2.0
                cz = (bounds[4] + bounds[5]) / 2.0
                coords_by_block[block_id].append([cx, cy, cz])

    extract_recursive(output)
    return dict(damage_by_block), dict(coords_by_block), globalid_damage_pairs


def map_damage_to_discretization(
    globalid_damage_pairs: list,
    disc_file: Path,
) -> tuple:
    """
    Map damage from Exodus output back to original positions using GlobalElementId.

    Peridigm's GlobalElementId is 1-indexed and maps to the line number in the
    discretization file. This gives us ground-truth original coordinates and
    block IDs, independent of VTK's displacement handling.

    Args:
        globalid_damage_pairs: [(global_element_id, damage, exodus_block_id), ...]
        disc_file: Path to Peridigm discretization file (x y z block_id volume)

    Returns:
        Tuple of (damage_by_block, coords_by_block) using discretization file positions
    """
    from collections import defaultdict

    # Read discretization file: x y z block_id volume
    disc_data = np.loadtxt(str(disc_file))
    disc_coords = disc_data[:, :3]
    disc_block_ids = disc_data[:, 3].astype(int)
    n_particles = len(disc_data)

    print(f"[Disc Mapping] Discretization file: {n_particles} particles")
    print(f"[Disc Mapping] Block IDs in disc: {np.unique(disc_block_ids).tolist()}")

    # Build damage array indexed by disc file order
    damage_by_disc = np.full(n_particles, np.nan)
    n_mapped = 0
    n_skipped = 0  # Particles only in Exodus (e.g., cone block 99)

    for global_id, damage_val, _ in globalid_damage_pairs:
        if global_id < 1:
            continue
        disc_idx = global_id - 1  # 1-indexed → 0-indexed
        if 0 <= disc_idx < n_particles:
            damage_by_disc[disc_idx] = damage_val
            n_mapped += 1
        else:
            n_skipped += 1

    print(f"[Disc Mapping] Mapped: {n_mapped}, skipped: {n_skipped} (cone/extra particles)")

    # Group by block using disc file's block IDs (ground truth)
    damage_by_block = defaultdict(list)
    coords_by_block = defaultdict(list)

    for i in range(n_particles):
        if np.isnan(damage_by_disc[i]):
            continue
        bid = int(disc_block_ids[i])
        damage_by_block[bid].append(float(damage_by_disc[i]))
        coords_by_block[bid].append(disc_coords[i].tolist())

    return dict(damage_by_block), dict(coords_by_block)


def compute_eval_time(
    disc_file: Path,
    wave_speed_mps: float = 1420.0,
    reflection_number: float = 1,
) -> float:
    """
    Compute damage evaluation time from assembly geometry and wave speed.

    The stress wave travels from the cone-side pole (z_min) to the wall-side
    pole (z_max) in time t = L / v. The Nth reflection time = N * L / v.

    Args:
        disc_file: Path to Peridigm discretization file (x y z block_id volume)
        wave_speed_mps: Longitudinal wave speed in PLA (m/s)
        reflection_number: Which reflection to evaluate at (1 = first arrival,
            2 = first round trip, etc.)

    Returns:
        Evaluation time in seconds
    """
    disc_data = np.loadtxt(str(disc_file))
    z = disc_data[:, 2]  # z coordinates in meters
    L = z.max() - z.min()
    t_1st_arrival = L / wave_speed_mps
    eval_time = reflection_number * t_1st_arrival
    print(f"[Wave Timing] Assembly length: {L*1000:.1f}mm, "
          f"wave speed: {wave_speed_mps:.0f} m/s, "
          f"1st arrival: {t_1st_arrival*1e6:.1f}µs")
    print(f"[Wave Timing] Using reflection #{reflection_number:.1f}×: {eval_time*1e6:.1f}µs")
    return eval_time


def compute_crack_line_continuity(
    coords: np.ndarray,
    damage: np.ndarray,
    z_min_band: float,
    z_max_band: float,
    damage_threshold: float = 0.5,
    slice_mm: float = 1.0,
) -> tuple:
    """
    Measure whether fractured particles form a connected crack path from z_min to z_max.

    Uses nearest-neighbor BFS (KDTree) to find if damaged particles form a
    continuous chain from the impact end (z_min) to the far end (z_max).
    The neighbor radius is derived from slice_mm (treated as 1.5× particle
    spacing for backward-compatible config).

    Args:
        coords: Nx3 particle positions (meters)
        damage: N damage values
        z_min_band: Band z-boundary (bottom / cone side)
        z_max_band: Band z-boundary (top / wall side)
        damage_threshold: Damage level for "fractured" particle
        slice_mm: Neighbor connectivity radius in mm (default 1.0mm, ~2× 0.5mm spacing)

    Returns:
        Tuple of (continuity, n_total, n_damaged)
        - continuity: fraction of z-range reached by connected crack (0.0 to 1.0)
        - n_total: total particles in this region
        - n_damaged: number of damaged particles in connected component
    """
    from scipy.spatial import cKDTree
    from collections import deque

    if len(coords) == 0 or len(damage) == 0:
        return 0.0, 0, 0

    z = coords[:, 2]
    z_range = z_max_band - z_min_band
    if z_range < 1e-10:
        return 0.0, 0, 0

    n_total = len(coords)

    # Get damaged particles
    damaged_mask = damage >= damage_threshold
    n_damaged_total = int(np.sum(damaged_mask))
    if n_damaged_total == 0:
        return 0.0, n_total, 0

    damaged_coords = coords[damaged_mask]
    damaged_z = damaged_coords[:, 2]

    # Neighbor radius in meters (slice_mm repurposed as connectivity radius)
    neighbor_radius = slice_mm / 1000.0

    # Build KDTree of damaged particles
    tree = cKDTree(damaged_coords)

    # Seed: damaged particles near z_min (within 2× neighbor radius)
    seed_tolerance = neighbor_radius * 2
    seed_mask = damaged_z <= (z_min_band + seed_tolerance)
    if not np.any(seed_mask):
        # No damaged particles near z_min — crack hasn't reached this region
        return 0.0, n_total, 0

    # BFS from seed particles through connected damaged neighbors
    visited = set()
    queue = deque()
    for idx in np.where(seed_mask)[0]:
        visited.add(idx)
        queue.append(idx)

    max_z_reached = damaged_z[list(visited)].max()

    while queue:
        current = queue.popleft()
        for n in tree.query_ball_point(damaged_coords[current], neighbor_radius):
            if n not in visited:
                visited.add(n)
                queue.append(n)
                if damaged_z[n] > max_z_reached:
                    max_z_reached = damaged_z[n]

    # Check if we reached z_max
    reached_top = max_z_reached >= (z_max_band - seed_tolerance)
    continuity = 1.0 if reached_top else (max_z_reached - z_min_band) / z_range
    continuity = max(0.0, min(1.0, continuity))

    return continuity, n_total, len(visited)


def compute_full_zone_breach(
    coords: np.ndarray,
    damage: np.ndarray,
    damage_threshold: float = 0.5,
    connectivity_radius_m: float = 0.0015,
) -> dict:
    """
    Full-zone BFS breach check: determines if damaged fracture particles form
    a structurally connected path spanning from z_min to z_max.

    Unlike per-band crack line continuity (which measures optimization feedback),
    this answers the definitive question: "is the structure split?"

    Uses BFS through all connected damaged particles starting from those near z_min.
    If the connected component reaches z_max, the structure is breached.

    Args:
        coords: Nx3 fracture zone particle positions (meters)
        damage: N damage values for fracture zone particles
        damage_threshold: Damage level for "broken" particle (default 0.5)
        connectivity_radius_m: Max distance between connected particles (default 1.5mm)

    Returns:
        Dict with:
        - breach: bool (True if connected damaged path spans z_min to z_max)
        - z_span_mm: z-range spanned by largest connected component (mm)
        - coverage_pct: fraction of fracture zone that is damaged
        - n_damaged: number of damaged particles
        - n_total: total fracture zone particles
        - largest_cc_size: number of particles in largest connected component
        - n_components: number of disconnected damaged components
    """
    from scipy.spatial import cKDTree
    from collections import deque

    n_total = len(coords)
    if n_total == 0:
        return {
            "breach": False, "z_span_mm": 0.0, "coverage_pct": 0.0,
            "n_damaged": 0, "n_total": 0, "largest_cc_size": 0, "n_components": 0,
        }

    z = coords[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    z_range = z_max - z_min

    # Get damaged particles
    damaged_mask = damage >= damage_threshold
    n_damaged = int(np.sum(damaged_mask))
    coverage_pct = n_damaged / n_total if n_total > 0 else 0.0

    if n_damaged == 0:
        return {
            "breach": False, "z_span_mm": 0.0, "coverage_pct": 0.0,
            "n_damaged": 0, "n_total": n_total, "largest_cc_size": 0, "n_components": 0,
        }

    damaged_coords = coords[damaged_mask]
    damaged_z = damaged_coords[:, 2]

    # Build KDTree of damaged particles
    tree = cKDTree(damaged_coords)

    # Find all connected components via BFS
    visited = set()
    components = []

    for start_idx in range(len(damaged_coords)):
        if start_idx in visited:
            continue
        # BFS from this unvisited particle
        component = set()
        queue = deque([start_idx])
        visited.add(start_idx)
        component.add(start_idx)

        while queue:
            current = queue.popleft()
            for n in tree.query_ball_point(damaged_coords[current], connectivity_radius_m):
                if n not in visited:
                    visited.add(n)
                    component.add(n)
                    queue.append(n)

        components.append(component)

    n_components = len(components)

    # Find largest component
    largest_cc = max(components, key=len)
    largest_cc_size = len(largest_cc)

    # Check each component for z_min-to-z_max breach
    breach = False
    best_z_span = 0.0
    seed_tolerance = connectivity_radius_m * 2

    for comp in components:
        comp_z = damaged_z[list(comp)]
        comp_z_min = float(comp_z.min())
        comp_z_max = float(comp_z.max())
        comp_span = comp_z_max - comp_z_min

        if comp_span > best_z_span:
            best_z_span = comp_span

        # Breach if component reaches both ends
        reaches_bottom = comp_z_min <= (z_min + seed_tolerance)
        reaches_top = comp_z_max >= (z_max - seed_tolerance)
        if reaches_bottom and reaches_top:
            breach = True

    z_span_mm = best_z_span * 1000.0

    print(f"[Full Zone Breach] n_total={n_total}, n_damaged={n_damaged} ({coverage_pct:.1%})")
    print(f"[Full Zone Breach] Components: {n_components}, largest: {largest_cc_size}")
    print(f"[Full Zone Breach] Z-span: {z_span_mm:.1f}mm (total: {z_range*1000:.1f}mm)")
    print(f"[Full Zone Breach] Breach: {'YES' if breach else 'NO'}")

    return {
        "breach": breach,
        "z_span_mm": z_span_mm,
        "coverage_pct": coverage_pct,
        "n_damaged": n_damaged,
        "n_total": n_total,
        "largest_cc_size": largest_cc_size,
        "n_components": n_components,
    }


def evaluate_all_timesteps(
    exodus_file: Path | list[Path],
    disc_file: Path,
    theta_band_edges_deg: list,
    fracture_block: int = 3,
    protected_block: int = 2,
    damage_threshold: float = 0.5,
    slice_mm: float = 1.0,
) -> dict:
    """
    Evaluate crack line continuity across ALL timesteps in a Peridigm Exodus file.

    For each timestep, extracts damage via GlobalElementId mapping and computes
    crack line continuity for each theta band in both fracture and protected zones.

    Tracks when each band first achieves 100% continuity and computes time-weighted
    scores (faster fracture = higher score for fracture zone; no fracture = higher
    score for protected zone).

    Args:
        exodus_file: Path to Peridigm Exodus output (serial), or list of paths
                     for parallel partition files (e.g. [output.e.8.0, ..., output.e.8.7])
        disc_file: Path to Peridigm discretization file
        theta_band_edges_deg: Band edges e.g. [0, 9, 18, ..., 90]
        fracture_block: Block ID for fracture zone (default: 3)
        protected_block: Block ID for protected shell (default: 2)
        damage_threshold: Damage level for "fractured" particle
        slice_mm: Z-slice resolution for continuity measurement

    Returns:
        Dict with crack_line_metrics structure
    """
    import vtk

    theta_edges = np.array(theta_band_edges_deg, dtype=float)
    n_bands = len(theta_edges) - 1

    # Read discretization file for ground-truth positions
    disc_data = np.loadtxt(str(disc_file))
    disc_coords = disc_data[:, :3]  # x, y, z in meters
    disc_block_ids = disc_data[:, 3].astype(int)
    n_particles = len(disc_data)

    print(f"[Crack Line] Discretization file: {n_particles} particles")
    print(f"[Crack Line] Block IDs: {np.unique(disc_block_ids).tolist()}")

    # Precompute per-particle hemisphere and band assignments
    # (these don't change with timestep since we use reference coords)
    z = disc_coords[:, 2]
    z_min, z_max = z.min(), z.max()
    z_mid = (z_min + z_max) / 2.0
    half_len = (z_max - z_min) / 2.0

    if half_len < 1e-10:
        print("[Crack Line] WARNING: Zero assembly length")
        return _empty_crack_line_result(n_bands)

    is_lower = z < z_mid  # cone-side hemisphere

    # Theta from nearest pole (0° = pole, 90° = equator)
    dist_from_pole = np.where(is_lower, z_mid - z, z - z_mid) / half_len
    dist_from_pole = np.clip(dist_from_pole, -1, 1)
    theta_deg = np.degrees(np.arccos(dist_from_pole))
    band_idx = np.clip(np.digitize(theta_deg, theta_edges[1:]), 0, n_bands - 1)

    # Masks for fracture and protected zones
    is_frac = disc_block_ids == fracture_block
    is_prot = disc_block_ids == protected_block

    # Precompute z-boundaries per band per hemisphere (for crack line slicing)
    # For each band × hemisphere, find z extent of particles
    band_hemi_info = {}  # (band, hemi) -> {"mask": ..., "z_min": ..., "z_max": ...}
    for b in range(n_bands):
        for hemi, hemi_mask in [("lower", is_lower), ("upper", ~is_lower)]:
            for zone, zone_mask in [("frac", is_frac), ("prot", is_prot)]:
                mask = (band_idx == b) & hemi_mask & zone_mask
                if np.any(mask):
                    z_vals = z[mask]
                    band_hemi_info[(b, hemi, zone)] = {
                        "indices": np.where(mask)[0],
                        "z_min": float(z_vals.min()),
                        "z_max": float(z_vals.max()),
                        "n_particles": int(mask.sum()),
                    }

    # Setup VTK readers — one per partition file for parallel Exodus output
    # (vtkPExodusIIReader requires MPI init which may not be available)
    exodus_files = [exodus_file] if isinstance(exodus_file, Path) else list(exodus_file)
    readers = []
    for ef in exodus_files:
        r = vtk.vtkExodusIIReader()
        r.SetFileName(str(ef))
        r.SetApplyDisplacements(False)
        r.UpdateInformation()
        for i in range(r.GetNumberOfElementResultArrays()):
            name = r.GetElementResultArrayName(i)
            if name.lower() == "damage":
                r.SetElementResultArrayStatus(name, 1)
        r.GenerateObjectIdCellArrayOn()
        r.GenerateGlobalElementIdArrayOn()
        readers.append(r)

    print(f"[Crack Line] Opened {len(readers)} Exodus reader(s)")

    # Get time values from first reader
    reader0 = readers[0]
    n_timesteps = reader0.GetNumberOfTimeSteps()
    time_values = None
    try:
        info = reader0.GetExecutive().GetOutputInformation(0)
        time_key = vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()
        if info.Has(time_key):
            n_times = info.Length(time_key)
            time_values = [info.Get(time_key, i) for i in range(n_times)]
    except Exception:
        pass

    if time_values is None or len(time_values) == 0:
        time_values = list(range(n_timesteps))

    print(f"[Crack Line] {n_timesteps} timesteps, time range: "
          f"{time_values[0]*1e6:.1f}µs to {time_values[-1]*1e6:.1f}µs")

    sim_time_s = float(time_values[-1]) if time_values else 0.0
    sim_time_us = sim_time_s * 1e6

    # Initialize per-band tracking for max continuity and time-to-100%
    # We track worst hemisphere (max of lower/upper) per band
    frac_max_continuity = np.zeros(n_bands)
    frac_time_to_100_us = [None] * n_bands  # None = never reached 100%
    frac_n_particles = np.zeros(n_bands, dtype=int)
    frac_n_slices = np.zeros(n_bands, dtype=int)

    prot_max_continuity = np.zeros(n_bands)
    prot_time_to_100_us = [None] * n_bands
    prot_n_particles = np.zeros(n_bands, dtype=int)

    # Count particles per band (from precomputed info)
    for b in range(n_bands):
        for hemi in ["lower", "upper"]:
            key_f = (b, hemi, "frac")
            key_p = (b, hemi, "prot")
            if key_f in band_hemi_info:
                frac_n_particles[b] += band_hemi_info[key_f]["n_particles"]
            if key_p in band_hemi_info:
                prot_n_particles[b] += band_hemi_info[key_p]["n_particles"]

    # Helper to extract damage array at a given timestep (merges all partition readers)
    def _extract_damage_at_timestep(ts_idx):
        """Read damage from all VTK readers at given timestep, return damage array indexed by disc order."""
        gid_damage = {}  # global_id -> damage_val

        def _recurse(data_object):
            if data_object is None:
                return
            if data_object.IsA('vtkMultiBlockDataSet'):
                for i in range(data_object.GetNumberOfBlocks()):
                    _recurse(data_object.GetBlock(i))
            else:
                cell_data = data_object.GetCellData() if hasattr(data_object, 'GetCellData') else None
                if cell_data is None:
                    return

                damage_arr = None
                for name in ["Damage", "damage", "DAMAGE"]:
                    damage_arr = cell_data.GetArray(name)
                    if damage_arr:
                        break
                if damage_arr is None:
                    return

                global_id_arr = cell_data.GetArray("GlobalElementId")
                if global_id_arr is None:
                    return

                for k in range(damage_arr.GetNumberOfTuples()):
                    gid = int(global_id_arr.GetValue(k))
                    dmg = damage_arr.GetValue(k)
                    if gid >= 1:
                        gid_damage[gid] = dmg

        for rd in readers:
            rd.SetTimeStep(ts_idx)
            rd.Update()
            _recurse(rd.GetOutput())

        # Map to disc file order
        damage_by_disc = np.full(n_particles, 0.0)
        for gid, dmg in gid_damage.items():
            disc_idx = gid - 1  # 1-indexed -> 0-indexed
            if 0 <= disc_idx < n_particles:
                damage_by_disc[disc_idx] = dmg

        return damage_by_disc

    # Iterate ALL timesteps
    for ts_idx in range(n_timesteps):
        t_s = time_values[ts_idx] if ts_idx < len(time_values) else 0.0
        t_us = t_s * 1e6

        damage_arr = _extract_damage_at_timestep(ts_idx)

        # For each band, compute crack line continuity in both hemispheres
        for b in range(n_bands):
            # === FRACTURE ZONE ===
            best_frac_cont = 0.0
            best_frac_slices = 0
            for hemi in ["lower", "upper"]:
                key = (b, hemi, "frac")
                if key not in band_hemi_info:
                    continue
                info = band_hemi_info[key]
                idxs = info["indices"]
                cont, n_sl, n_dmg = compute_crack_line_continuity(
                    disc_coords[idxs], damage_arr[idxs],
                    info["z_min"], info["z_max"],
                    damage_threshold, slice_mm,
                )
                if cont > best_frac_cont:
                    best_frac_cont = cont
                    best_frac_slices = n_sl

            # Track max continuity
            if best_frac_cont > frac_max_continuity[b]:
                frac_max_continuity[b] = best_frac_cont
                frac_n_slices[b] = best_frac_slices

            # Track first time reaching 100%
            if best_frac_cont >= 1.0 and frac_time_to_100_us[b] is None:
                frac_time_to_100_us[b] = t_us

            # === PROTECTED ZONE ===
            best_prot_cont = 0.0
            for hemi in ["lower", "upper"]:
                key = (b, hemi, "prot")
                if key not in band_hemi_info:
                    continue
                info = band_hemi_info[key]
                idxs = info["indices"]
                cont, _, _ = compute_crack_line_continuity(
                    disc_coords[idxs], damage_arr[idxs],
                    info["z_min"], info["z_max"],
                    damage_threshold, slice_mm,
                )
                if cont > best_prot_cont:
                    best_prot_cont = cont

            if best_prot_cont > prot_max_continuity[b]:
                prot_max_continuity[b] = best_prot_cont
            if best_prot_cont >= 1.0 and prot_time_to_100_us[b] is None:
                prot_time_to_100_us[b] = t_us

    # Compute time-weighted scores
    frac_scores = np.zeros(n_bands)
    prot_scores = np.ones(n_bands)  # Default: perfect (never broke)

    for b in range(n_bands):
        if frac_n_particles[b] == 0:
            frac_scores[b] = 0.0
            continue
        if frac_time_to_100_us[b] is not None and sim_time_us > 0:
            # Fracture achieved 100% continuity: score = (T_sim - T_100%) / T_sim
            frac_scores[b] = (sim_time_us - frac_time_to_100_us[b]) / sim_time_us
            frac_scores[b] = max(0.0, frac_scores[b])
        else:
            # Never reached 100% continuity
            frac_scores[b] = 0.0

        if prot_time_to_100_us[b] is not None and sim_time_us > 0:
            # Protected zone broke: score = T_100% / T_sim (later = less bad)
            prot_scores[b] = prot_time_to_100_us[b] / sim_time_us
        # else: never broke → score stays 1.0

    # Count full crack bands (used for cone angle adaptation)
    full_crack_bands = int(np.sum(frac_max_continuity >= 1.0))

    # Damage penetration: count consecutive full-crack bands starting from the
    # first band that has fracture particles (pole bands are often empty)
    # Find first band with particles
    first_with_particles = None
    for b in range(n_bands):
        if frac_n_particles[b] > 0:
            first_with_particles = b
            break

    damage_penetration_depth = 0
    n_bands_with_particles = sum(1 for n in frac_n_particles if n > 0)
    if first_with_particles is not None:
        for b in range(first_with_particles, n_bands):
            if frac_max_continuity[b] >= 1.0:
                damage_penetration_depth += 1
            elif frac_n_particles[b] == 0:
                # Empty band in the middle — don't break the chain
                continue
            else:
                break

    penetration_pct = damage_penetration_depth / n_bands_with_particles if n_bands_with_particles > 0 else 0.0

    # Print summary
    print(f"\n[Crack Line] Results ({n_timesteps} timesteps, {sim_time_us:.0f}µs):")
    print(f"  {'Band':>6} | {'FracCont':>8} {'FracT100':>9} {'FracScore':>9} | {'ProtCont':>8} {'ProtScore':>9} | {'#Frac':>6} {'#Prot':>6}")
    print(f"  {'-'*85}")
    for b in range(n_bands):
        th_lo = theta_edges[b]
        th_hi = theta_edges[b + 1]
        ft = f"{frac_time_to_100_us[b]:.1f}µs" if frac_time_to_100_us[b] is not None else "   -   "
        print(f"  {th_lo:.0f}-{th_hi:.0f}° | {frac_max_continuity[b]:>8.2f} {ft:>9} {frac_scores[b]:>9.3f} "
              f"| {prot_max_continuity[b]:>8.2f} {prot_scores[b]:>9.3f} | {frac_n_particles[b]:>6} {prot_n_particles[b]:>6}")

    print(f"\n  Full crack bands: {full_crack_bands}/{n_bands}")
    print(f"  Penetration depth: {damage_penetration_depth} bands = {penetration_pct:.0%}")

    return {
        "fracture_zone": {
            "per_band_continuity": frac_max_continuity.tolist(),
            "per_band_time_to_100_us": frac_time_to_100_us,
            "per_band_score": frac_scores.tolist(),
            "per_band_n_particles": frac_n_particles.tolist(),
            "per_band_n_slices": frac_n_slices.tolist(),
        },
        "protected_zone": {
            "per_band_continuity": prot_max_continuity.tolist(),
            "per_band_time_to_100_us": prot_time_to_100_us,
            "per_band_score": prot_scores.tolist(),
            "per_band_n_particles": prot_n_particles.tolist(),
        },
        "n_timesteps_evaluated": n_timesteps,
        "simulation_time_us": sim_time_us,
        "full_crack_bands": full_crack_bands,
        "damage_penetration_depth": damage_penetration_depth,
        "damage_penetration_pct": penetration_pct,
    }


def _empty_crack_line_result(n_bands: int) -> dict:
    """Return empty crack line metrics result."""
    zeros = [0.0] * n_bands
    int_zeros = [0] * n_bands
    nones = [None] * n_bands
    return {
        "fracture_zone": {
            "per_band_continuity": zeros,
            "per_band_time_to_100_us": nones,
            "per_band_score": zeros,
            "per_band_n_particles": int_zeros,
            "per_band_n_slices": int_zeros,
        },
        "protected_zone": {
            "per_band_continuity": zeros,
            "per_band_time_to_100_us": nones,
            "per_band_score": [1.0] * n_bands,
            "per_band_n_particles": int_zeros,
        },
        "n_timesteps_evaluated": 0,
        "simulation_time_us": 0.0,
        "full_crack_bands": 0,
        "damage_penetration_depth": 0,
        "damage_penetration_pct": 0.0,
    }


def compute_per_band_hemisphere_damage(
    coords_by_block: dict,
    damage_by_block: dict,
    theta_band_edges_deg: list,
    fracture_zone_block: int = 3,
    shell_body_block: int = 2,
    damage_threshold: float = 0.1,
) -> dict:
    """
    Compute per-band, per-hemisphere damage for fracture and protected zones.

    Splits the assembly into upper (wall-side, z > z_mid) and lower (cone-side,
    z < z_mid) hemispheres. For each theta band, computes mean damage in:
    - Fracture zone (block 3): the designed-to-break region
    - Protected shell (block 2): the structural region that should stay intact

    Theta is measured from the nearest pole: 0° = pole, 90° = equator.

    Args:
        coords_by_block: {block_id: [[x,y,z], ...]} reference coordinates
        damage_by_block: {block_id: [damage_values]}
        theta_band_edges_deg: Band edges e.g. [0, 9, 18, ..., 90]
        fracture_zone_block: Block ID for fracture zone (default: 3)
        shell_body_block: Block ID for protected shell (default: 2)
        damage_threshold: Threshold for "significant" damage in penetration depth

    Returns:
        Dict with per-band damage arrays and damage penetration depth
    """
    theta_edges = np.array(theta_band_edges_deg, dtype=float)
    n_bands = len(theta_edges) - 1

    # Collect all PLA particles (blocks 2 and 3)
    all_coords = []
    all_damage = []
    all_is_frac = []

    for bid in [fracture_zone_block, shell_body_block]:
        if bid not in coords_by_block or bid not in damage_by_block:
            continue
        coords = coords_by_block[bid]
        dmg = damage_by_block[bid]
        n = min(len(coords), len(dmg))
        all_coords.extend(coords[:n])
        all_damage.extend(dmg[:n])
        all_is_frac.extend([bid == fracture_zone_block] * n)

    if not all_coords:
        print("[Per-Band] WARNING: No PLA particles found")
        return _empty_per_band_result(n_bands)

    coords_arr = np.array(all_coords)
    damage_arr = np.array(all_damage)
    is_frac = np.array(all_is_frac, dtype=bool)

    z = coords_arr[:, 2]
    z_min, z_max = z.min(), z.max()
    z_mid = (z_min + z_max) / 2.0
    half_len = (z_max - z_min) / 2.0

    if half_len < 1e-10:
        print("[Per-Band] WARNING: Zero assembly length")
        return _empty_per_band_result(n_bands)

    # Hemisphere classification
    is_lower = z < z_mid  # cone-side

    # Theta from nearest pole (0° = pole, 90° = equator)
    dist_from_pole = np.where(is_lower, z_mid - z, z - z_mid) / half_len
    dist_from_pole = np.clip(dist_from_pole, -1, 1)
    theta_deg = np.degrees(np.arccos(dist_from_pole))

    # Bin into bands
    band_idx = np.clip(np.digitize(theta_deg, theta_edges[1:]), 0, n_bands - 1)

    # Compute per-band damage
    lower_frac = np.zeros(n_bands)
    upper_frac = np.zeros(n_bands)
    lower_prot = np.zeros(n_bands)
    upper_prot = np.zeros(n_bands)

    # Track particle counts per band to distinguish "no damage" from "no particles"
    n_frac_particles = np.zeros(n_bands, dtype=int)
    n_prot_particles = np.zeros(n_bands, dtype=int)

    # Volumetric damage fraction: what fraction of particles have damage > threshold
    n_frac_damaged = np.zeros(n_bands, dtype=int)
    n_prot_damaged = np.zeros(n_bands, dtype=int)

    for b in range(n_bands):
        bm = band_idx == b
        lf = bm & is_lower & is_frac
        uf = bm & (~is_lower) & is_frac
        lp = bm & is_lower & (~is_frac)
        up = bm & (~is_lower) & (~is_frac)

        n_frac_particles[b] = int(lf.sum() + uf.sum())
        n_prot_particles[b] = int(lp.sum() + up.sum())

        lower_frac[b] = float(damage_arr[lf].mean()) if lf.sum() > 0 else 0.0
        upper_frac[b] = float(damage_arr[uf].mean()) if uf.sum() > 0 else 0.0
        lower_prot[b] = float(damage_arr[lp].mean()) if lp.sum() > 0 else 0.0
        upper_prot[b] = float(damage_arr[up].mean()) if up.sum() > 0 else 0.0

        # Count particles with damage exceeding threshold
        frac_mask = bm & is_frac
        prot_mask = bm & (~is_frac)
        if frac_mask.sum() > 0:
            n_frac_damaged[b] = int(np.sum(damage_arr[frac_mask] > damage_threshold))
        if prot_mask.sum() > 0:
            n_prot_damaged[b] = int(np.sum(damage_arr[prot_mask] > damage_threshold))

    worst_frac = np.maximum(lower_frac, upper_frac)
    worst_prot = np.maximum(lower_prot, upper_prot)

    # Volumetric damage fraction per band (fraction of particles exceeding threshold)
    frac_damaged_fraction = n_frac_damaged / np.maximum(n_frac_particles, 1)
    prot_damaged_fraction = n_prot_damaged / np.maximum(n_prot_particles, 1)

    # ================================================================
    # FULL POLE-TO-POLE DAMAGE SPAN
    # ================================================================
    # Lay out hemispheres as 2N zones from cone-side pole to wall-side pole:
    #   lower[0], lower[1], ..., lower[9], upper[9], upper[8], ..., upper[0]
    # Count how many consecutive zones from the cone end have significant
    # damage to get a true pole-to-pole penetration percentage.
    # ================================================================
    full_span_frac = np.concatenate([lower_frac, upper_frac[::-1]])
    n_full = len(full_span_frac)  # 2 * n_bands = 20

    # Also track which zones have fracture particles (per-hemisphere)
    n_lower_frac = np.zeros(n_bands, dtype=int)
    n_upper_frac = np.zeros(n_bands, dtype=int)
    for b in range(n_bands):
        bm = band_idx == b
        n_lower_frac[b] = int((bm & is_lower & is_frac).sum())
        n_upper_frac[b] = int((bm & (~is_lower) & is_frac).sum())
    full_span_has_data = np.concatenate([n_lower_frac > 0, (n_upper_frac > 0)[::-1]])

    # Count from cone-side pole: how far does damage reach?
    # Skip zones with no data, count zones with damage > threshold
    damage_reached = 0
    zones_with_data = 0
    for i in range(n_full):
        if not full_span_has_data[i]:
            continue  # no particles here, skip
        zones_with_data += 1
        if full_span_frac[i] > damage_threshold:
            damage_reached = zones_with_data

    penetration_pct = damage_reached / zones_with_data if zones_with_data > 0 else 0.0

    # Also keep the old per-band metric (highest band index with damage)
    frac_has_data = n_frac_particles > 0
    significant_bands = np.where((worst_frac > damage_threshold) & frac_has_data)[0]
    penetration_depth = int(significant_bands.max()) if len(significant_bands) > 0 else 0

    # Print summary
    print(f"[Per-Band] Hemisphere damage:")
    print(f"  {'Band':>6} | {'L.frac':>7} {'U.frac':>7} {'Worst':>7} | {'L.prot':>7} {'U.prot':>7} | {'FracVol':>7} {'ProtVol':>7} | {'#Lfr':>5} {'#Ufr':>5} {'#prot':>5}")
    print(f"  {'-'*100}")
    for b in range(n_bands):
        th_lo = theta_edges[b]
        th_hi = theta_edges[b + 1]
        tag = " (no frac)" if n_frac_particles[b] == 0 else ""
        print(f"  {th_lo:.0f}-{th_hi:.0f}° | {lower_frac[b]:>7.3f} {upper_frac[b]:>7.3f} {worst_frac[b]:>7.3f} "
              f"| {lower_prot[b]:>7.3f} {upper_prot[b]:>7.3f} "
              f"| {frac_damaged_fraction[b]:>6.1%} {prot_damaged_fraction[b]:>6.1%} "
              f"| {n_lower_frac[b]:>5} {n_upper_frac[b]:>5} {n_prot_particles[b]:>5}{tag}")

    print(f"\n  Full pole-to-pole span ({zones_with_data} zones with data):")
    labels = [f"-t{b}" for b in range(n_bands)] + [f"+t{b}" for b in range(n_bands-1, -1, -1)]
    dmg_strs = []
    for i in range(n_full):
        if not full_span_has_data[i]:
            dmg_strs.append(f"{labels[i]}:---")
        elif full_span_frac[i] > damage_threshold:
            dmg_strs.append(f"{labels[i]}:{full_span_frac[i]:.2f}")
        else:
            dmg_strs.append(f"{labels[i]}:0  ")
    print(f"  {' '.join(dmg_strs)}")
    print(f"  Penetration: {damage_reached}/{zones_with_data} zones = {penetration_pct:.0%}")
    print(f"  Deepest band: {penetration_depth} ({theta_edges[penetration_depth]:.0f}-{theta_edges[min(penetration_depth+1, n_bands)]:.0f}°)")

    return {
        "lower_frac_damage": lower_frac.tolist(),
        "upper_frac_damage": upper_frac.tolist(),
        "lower_prot_damage": lower_prot.tolist(),
        "upper_prot_damage": upper_prot.tolist(),
        "worst_frac_damage": worst_frac.tolist(),
        "worst_prot_damage": worst_prot.tolist(),
        "frac_damaged_fraction": frac_damaged_fraction.tolist(),
        "prot_damaged_fraction": prot_damaged_fraction.tolist(),
        "n_frac_damaged": n_frac_damaged.tolist(),
        "n_prot_damaged": n_prot_damaged.tolist(),
        "damage_penetration_depth": penetration_depth,
        "damage_penetration_pct": penetration_pct,
        "n_frac_particles": n_frac_particles.tolist(),
        "n_prot_particles": n_prot_particles.tolist(),
        "n_lower_frac": n_lower_frac.tolist(),
        "n_upper_frac": n_upper_frac.tolist(),
    }


def _empty_per_band_result(n_bands: int) -> dict:
    """Return empty per-band damage result."""
    zeros = [0.0] * n_bands
    int_zeros = [0] * n_bands
    return {
        "lower_frac_damage": zeros,
        "upper_frac_damage": zeros,
        "lower_prot_damage": zeros,
        "upper_prot_damage": zeros,
        "worst_frac_damage": zeros,
        "worst_prot_damage": zeros,
        "frac_damaged_fraction": zeros,
        "prot_damaged_fraction": zeros,
        "n_frac_damaged": int_zeros,
        "n_prot_damaged": int_zeros,
        "damage_penetration_depth": 0,
        "damage_penetration_pct": 0.0,
        "n_frac_particles": int_zeros,
        "n_prot_particles": int_zeros,
        "n_lower_frac": int_zeros,
        "n_upper_frac": int_zeros,
    }


def compute_cone_angle_adaptation(
    per_band_damage: dict,
    current_cone_angle_deg: float,
    config: dict,
) -> dict:
    """
    Compute new cone angle based on damage penetration depth.

    The damage penetration depth (highest theta band with significant fracture
    zone damage) indicates whether the cone is distributing force correctly:
    - Too shallow (near pole only) → cone angle too narrow → widen
    - Too deep (near equator) → cone angle too wide → narrow
    - In sweet spot → keep current angle

    Args:
        per_band_damage: Output from compute_per_band_hemisphere_damage()
        current_cone_angle_deg: Current cone half-angle in degrees
        config: Full optimization config dict

    Returns:
        Dict with new_cone_angle_deg, adjustment_deg, reason
    """
    peridigm_cfg = config.get("peridigm", {})
    adapt_cfg = peridigm_cfg.get("cone_angle_adaptation", {})

    if not adapt_cfg.get("enabled", False):
        return {
            "new_cone_angle_deg": current_cone_angle_deg,
            "adjustment_deg": 0,
            "damage_penetration_depth": per_band_damage.get("damage_penetration_depth", 0),
            "reason": "adaptation_disabled",
        }

    min_angle = adapt_cfg.get("min_angle_deg", 15)
    max_angle = adapt_cfg.get("max_angle_deg", 60)
    step = adapt_cfg.get("step_deg", 5)
    target_min = adapt_cfg.get("target_depth_min", 3)
    target_max = adapt_cfg.get("target_depth_max", 6)

    depth = per_band_damage.get("damage_penetration_depth", 0)
    adjustment = 0
    reason = "in_sweet_spot"

    if depth < target_min:
        adjustment = step
        reason = f"too_shallow (depth={depth} < target_min={target_min})"
    elif depth > target_max:
        adjustment = -step
        reason = f"too_deep (depth={depth} > target_max={target_max})"

    new_angle = float(np.clip(current_cone_angle_deg + adjustment, min_angle, max_angle))

    print(f"[Cone Adapt] depth={depth}, target=[{target_min},{target_max}], "
          f"current={current_cone_angle_deg}°, adjustment={adjustment:+d}°, "
          f"new={new_angle}° ({reason})")

    return {
        "new_cone_angle_deg": new_angle,
        "adjustment_deg": adjustment,
        "damage_penetration_depth": depth,
        "target_depth_range": [target_min, target_max],
        "reason": reason,
    }


def main():
    parser = argparse.ArgumentParser(description="Process Peridigm results")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--peridigm-dir", required=True, help="Peridigm output directory")
    parser.add_argument("--job-id", required=True, help="Peridigm job ID")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    peridigm_dir = Path(args.peridigm_dir)
    job_id = args.job_id
    state_file = output_dir / "optimization_state.json"

    print(f"\n{'=' * 70}")
    print("Processing Peridigm Results")
    print(f"{'=' * 70}")
    print(f"Peridigm directory: {peridigm_dir}")
    print(f"Job ID: {job_id}")

    # Find Exodus output file (handle both serial and parallel formats)
    exodus_files = list(peridigm_dir.glob("*.e")) + list(peridigm_dir.glob("*.exo"))
    serial_exodus = [f for f in exodus_files if not any(c.isdigit() for c in f.suffix)]
    parallel_parts = list(peridigm_dir.glob("*.e.[0-9]*.[0-9]*"))

    # Block ID mapping (from theta_mesh export convention):
    # 1 = Steel core (ignore in damage calc)
    # 2 = PLA protected (shell body)
    # 3 = PLA fracture (fracture zone)
    # 99 = Cone (ignore)
    FRACTURE_ZONE_BLOCK = 3
    SHELL_BODY_BLOCK = 2

    D_fracture = 0.0
    D_shell = 0.0
    damage_stats = {}
    damage_by_block = {}
    fracture_metrics = {}
    shell_metrics = {}

    coords_by_block = {}

    # Collect all GlobalElementId→damage pairs across all Exodus parts
    all_globalid_pairs = []

    # ====================================================================
    # COMPUTE EVALUATION TIME (1st wave reflection)
    # ====================================================================
    # All damage metrics (D_frac, breach, per-band) must use the SAME
    # timestep. Using the final timestep overestimates damage due to
    # compounding from multiple wave reflections.
    # ====================================================================
    config_file = output_dir / "dual_physics_config.yaml"
    eval_config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            eval_config = yaml.safe_load(f)

    peridigm_cfg_early = eval_config.get("peridigm", {})
    wave_speed = peridigm_cfg_early.get("pla_wave_speed_mps", 1420.0)

    # Find discretization file early (needed for reflection time + disc mapping)
    # Prefer the combined file (flyback_split_core_discretization.txt) which matches
    # Peridigm's GlobalElementId ordering. Fall back to split_core_peridigm.txt
    # (same particle order for blocks 1-4, just missing cone/wall blocks 98/99).
    disc_files = list(peridigm_dir.glob("*_discretization.txt"))
    if not disc_files:
        disc_files = list(peridigm_dir.glob("*_peridigm.txt")) + list(peridigm_dir.glob("split_core_peridigm.txt"))
    if not disc_files:
        parent_peridigm = peridigm_dir.parent
        disc_files = list(parent_peridigm.glob("*_discretization.txt")) + list(parent_peridigm.glob("*_peridigm.txt"))

    # Only compute eval time if damage_eval_reflection is explicitly set in config.
    # When absent (v7 cone-cylinder), use final timestep — the wave reflection model
    # doesn't apply to wedge-driven fracture where breach occurs at ~660µs, not ~20µs.
    reflection_cfg = peridigm_cfg_early.get("damage_eval_reflection", None)

    eval_time_s = None
    if reflection_cfg is not None and disc_files:
        reflection_number = float(reflection_cfg)
        try:
            eval_time_s = compute_eval_time(disc_files[0], wave_speed, reflection_number)
            print(f"\n[Eval Time] All damage evaluated at reflection #{reflection_number}: {eval_time_s*1e6:.1f}µs")
        except Exception as e:
            print(f"\n[Eval Time] Could not compute reflection time: {e} — using final timestep")
    else:
        print(f"\n[Eval Time] No damage_eval_reflection in config — using FINAL timestep")

    if serial_exodus:
        exodus_file = max(serial_exodus, key=lambda p: p.stat().st_mtime)
        print(f"Found serial Exodus file: {exodus_file}")
        try:
            damage_by_block, coords_by_block, gid_pairs = read_damage_from_exodus_vtk(
                exodus_file, target_time_s=eval_time_s
            )
            all_globalid_pairs.extend(gid_pairs)
            total_cells = sum(len(v) for v in damage_by_block.values())
            print(f"  Read {total_cells} damage values across {len(damage_by_block)} blocks")
        except Exception as e:
            print(f"  Warning: Could not read {exodus_file}: {e}")

    elif parallel_parts:
        print(f"Found parallel Exodus output ({len(parallel_parts)} parts)")
        print("Reading with VTK ExodusII reader...")

        from collections import defaultdict
        accumulated_damage = defaultdict(list)
        accumulated_coords = defaultdict(list)

        for part_file in sorted(parallel_parts):
            try:
                part_damage, part_coords, gid_pairs = read_damage_from_exodus_vtk(
                    part_file, target_time_s=eval_time_s
                )
                all_globalid_pairs.extend(gid_pairs)
                for block_id, values in part_damage.items():
                    accumulated_damage[block_id].extend(values)
                for block_id, coords in part_coords.items():
                    accumulated_coords[block_id].extend(coords)
                total_in_part = sum(len(v) for v in part_damage.values())
                if total_in_part > 0:
                    print(f"  {part_file.name}: {total_in_part} damage values")
            except Exception as e:
                print(f"  Warning: Could not read {part_file.name}: {e}")

        damage_by_block = dict(accumulated_damage)
        coords_by_block = dict(accumulated_coords)
        total_cells = sum(len(v) for v in damage_by_block.values())
        print(f"Total damage values from {len(parallel_parts)} parts: {total_cells}")

    else:
        print("WARNING: No Exodus output found")
        print(f"  Files in directory: {list(peridigm_dir.iterdir())}")

    # ====================================================================
    # MAP DAMAGE TO ORIGINAL POSITIONS VIA DISCRETIZATION FILE
    # ====================================================================
    # GlobalElementId maps each Exodus particle to its line in the
    # discretization file (1-indexed). This gives us ground-truth original
    # coordinates and block IDs, independent of VTK displacement handling.
    # ====================================================================
    # disc_files already found above (eval time section)
    has_global_ids = any(gid >= 0 for gid, _, _ in all_globalid_pairs) if all_globalid_pairs else False

    if disc_files and has_global_ids:
        disc_file = disc_files[0]
        print(f"\n[Disc Mapping] Using GlobalElementId → discretization file mapping")
        print(f"[Disc Mapping] Disc file: {disc_file.name}")
        try:
            mapped_damage, mapped_coords = map_damage_to_discretization(
                all_globalid_pairs, disc_file
            )
            # Replace VTK-derived data with disc-file-mapped data
            damage_by_block = mapped_damage
            coords_by_block = mapped_coords
            total_cells = sum(len(v) for v in damage_by_block.values())
            print(f"[Disc Mapping] Using disc file positions for {total_cells} particles")
        except Exception as e:
            print(f"[Disc Mapping] Failed: {e} — falling back to VTK reference coords")
            import traceback
            traceback.print_exc()
    elif not has_global_ids and damage_by_block:
        print("\n[Disc Mapping] No GlobalElementId in Exodus — using VTK reference coords")

    # Calculate damage statistics PER BLOCK
    if damage_by_block:
        print(f"\nDamage by block:")
        for block_id in sorted(damage_by_block.keys()):
            block_damage = np.array(damage_by_block[block_id])
            block_mean = float(np.mean(block_damage)) if len(block_damage) > 0 else 0.0
            block_max = float(np.max(block_damage)) if len(block_damage) > 0 else 0.0
            n_damaged = int(np.sum(block_damage > 0.01))
            print(f"  Block {block_id}: {len(block_damage)} cells, mean={block_mean:.4f}, max={block_max:.4f}, damaged={n_damaged}")

        # Calculate D_fracture_zone (Block 3 = fracture zone)
        fracture_metrics = {}
        if FRACTURE_ZONE_BLOCK in damage_by_block:
            fracture_damage = np.array(damage_by_block[FRACTURE_ZONE_BLOCK])
            D_fracture = float(np.mean(fracture_damage))

            # Additional fracture metrics
            n_fracture_cells = len(fracture_damage)
            n_fully_broken = int(np.sum(fracture_damage >= 0.9))  # 90%+ damage = broken
            n_partially_damaged = int(np.sum((fracture_damage > 0.1) & (fracture_damage < 0.9)))
            n_intact = int(np.sum(fracture_damage <= 0.1))

            fracture_completeness = n_fully_broken / n_fracture_cells if n_fracture_cells > 0 else 0.0
            fracture_breach = bool(float(np.max(fracture_damage)) >= 0.99)  # Any cell fully broken?

            fracture_metrics = {
                "mean_damage": D_fracture,
                "max_damage": float(np.max(fracture_damage)),
                "n_cells": n_fracture_cells,
                "n_fully_broken": n_fully_broken,
                "n_partially_damaged": n_partially_damaged,
                "n_intact": n_intact,
                "completeness": fracture_completeness,  # % of fracture zone that's fully broken
                "breach": fracture_breach,  # Did it break through anywhere?
            }

            print(f"\n  D_fracture_zone (Block {FRACTURE_ZONE_BLOCK}): {D_fracture:.4f}")
            print(f"    Fracture completeness: {fracture_completeness*100:.1f}% fully broken ({n_fully_broken}/{n_fracture_cells})")
            print(f"    Fracture breach: {'YES' if fracture_breach else 'NO'} (max damage: {fracture_metrics['max_damage']:.3f})")
        else:
            print(f"\n  WARNING: Fracture zone block {FRACTURE_ZONE_BLOCK} not found!")
            D_fracture = 0.0

        # Calculate D_shell_body (Block 2 = protected shell)
        shell_metrics = {}
        if SHELL_BODY_BLOCK in damage_by_block:
            shell_damage = np.array(damage_by_block[SHELL_BODY_BLOCK])
            D_shell = float(np.mean(shell_damage))

            # Additional shell metrics
            n_shell_cells = len(shell_damage)
            n_shell_damaged = int(np.sum(shell_damage > 0.1))  # Any significant damage
            n_shell_broken = int(np.sum(shell_damage >= 0.9))  # Fully broken
            shell_integrity = (n_shell_cells - n_shell_damaged) / n_shell_cells if n_shell_cells > 0 else 0.0

            shell_metrics = {
                "mean_damage": D_shell,
                "max_damage": float(np.max(shell_damage)),
                "n_cells": n_shell_cells,
                "n_damaged": n_shell_damaged,
                "n_broken": n_shell_broken,
                "integrity": shell_integrity,  # % of shell that's intact
            }

            print(f"  D_shell_body (Block {SHELL_BODY_BLOCK}): {D_shell:.4f}")
            print(f"    Shell integrity: {shell_integrity*100:.1f}% intact ({n_shell_cells - n_shell_damaged}/{n_shell_cells})")
            print(f"    Shell breaches: {n_shell_broken} cells fully broken")
        else:
            print(f"  WARNING: Shell body block {SHELL_BODY_BLOCK} not found!")
            D_shell = 0.0

        # Overall statistics (for backwards compatibility)
        all_damage = []
        for values in damage_by_block.values():
            all_damage.extend(values)
        all_damage = np.array(all_damage)

        damage_stats = {
            "mean": float(np.mean(all_damage)),
            "max": float(np.max(all_damage)),
            "min": float(np.min(all_damage)),
            "std": float(np.std(all_damage)),
            "n_damaged": int(np.sum(all_damage > 0.01)),
            "n_total": len(all_damage),
        }
        print(f"\nOverall damage statistics:")
        print(f"  Mean: {damage_stats['mean']:.4f}")
        print(f"  Max:  {damage_stats['max']:.4f}")
        print(f"  Damaged cells (>0.01): {damage_stats['n_damaged']}/{damage_stats['n_total']}")
    else:
        print("WARNING: No damage data found")

    # ====================================================================
    # PATH-BASED BREACH DETECTION
    # ====================================================================
    # Uses BFS through connected broken bonds to determine if crack
    # propagated from z_max to z_min, and which theta band blocked it.
    # ====================================================================
    breach_info = None

    # Config, disc_files, config_file already loaded above (eval time section)
    # Extract theta band edges, horizon, fracture zone IDs from config
    theta_band_edges = [0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90]
    horizon = 0.009  # default 9mm
    fracture_zone_ids = [3]

    if eval_config:
        mesh_cfg = eval_config.get("mesh", {})
        theta_overrides = mesh_cfg.get("theta_overrides", {})
        mesh_params = theta_overrides.get("mesh", {})
        theta_band_edges = mesh_params.get("theta_band_edges_deg", theta_band_edges)
        horizon = peridigm_cfg_early.get("horizon_mm", 9.0) / 1000.0
        fracture_zone_ids = peridigm_cfg_early.get("fracture_zone_ids", [3])

    if damage_by_block and coords_by_block:
        print(f"\n[Path Breach] Using VTK coordinates (properly aligned with damage)")

        try:
            # Build aligned arrays directly from VTK data (coordinates match damage)
            all_points = []
            all_damage = []
            all_block_ids = []

            for bid in sorted(damage_by_block.keys()):
                if bid not in coords_by_block:
                    continue
                dmg_vals = damage_by_block[bid]
                coord_vals = coords_by_block[bid]
                n = min(len(dmg_vals), len(coord_vals))
                all_points.extend(coord_vals[:n])
                all_damage.extend(dmg_vals[:n])
                all_block_ids.extend([bid] * n)

            all_points = np.array(all_points)
            all_damage = np.array(all_damage)
            all_block_ids = np.array(all_block_ids)

            print(f"[Path Breach] Total points: {len(all_points)}")
            frac_mask = np.isin(all_block_ids, fracture_zone_ids)
            frac_dmg = all_damage[frac_mask]
            frac_z = all_points[frac_mask, 2] * 1000
            n_broken = np.sum(frac_dmg > 0.9)
            if n_broken > 0:
                broken_z = frac_z[frac_dmg > 0.9]
                print(f"[Path Breach] Fracture zone broken: {n_broken}, z=[{broken_z.min():.1f}, {broken_z.max():.1f}] mm")
            else:
                print(f"[Path Breach] Fracture zone broken: 0")

            # Import and run path-based breach detection
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modules"))
            from perifeato.optimization.adapters.peridigm_adapter import compute_path_based_breach

            breach_info = compute_path_based_breach(
                points=all_points,
                damage=all_damage,
                block_ids=all_block_ids,
                fracture_zone_ids=fracture_zone_ids,
                theta_band_edges_deg=theta_band_edges,
                horizon=horizon,
                damage_threshold=0.9,
            )

            if breach_info:
                print(f"[Path Breach] Breach: {'YES' if breach_info['breach'] else 'NO'}")
                print(f"[Path Breach] Crack depth: {breach_info['crack_depth']:.1%}")
                if not breach_info['breach'] and breach_info.get('blocked_at_band') is not None:
                    print(f"[Path Breach] Blocked at band: {breach_info['blocked_at_band']}")
                print(f"[Path Breach] Path length: {breach_info.get('path_length', 0)} points")

                # Print per-band breach scores
                band_scores = breach_info.get('band_breach_scores', [])
                if band_scores:
                    print(f"[Path Breach] Per-band breach scores:")
                    for bi, sc in enumerate(band_scores):
                        status = "FULL_BREACH" if sc >= 0.7 else "BREACH" if sc >= 0.4 else "PARTIAL" if sc > 0 else "INTACT"
                        print(f"  Band {bi}: score={sc:.3f} ({status})")

        except Exception as e:
            print(f"[Path Breach] Error: {e}")
            import traceback
            traceback.print_exc()
    elif not damage_by_block:
        print("\n[Path Breach] No damage data - skipping path breach detection")
    else:
        print("\n[Path Breach] No coordinates from VTK - skipping path breach detection")

    # ====================================================================
    # PER-BAND HEMISPHERE-AWARE DAMAGE
    # ====================================================================
    # All damage was already read at the 1st reflection evaluation time
    # (see eval_time_s above). Compute per-band breakdown directly from
    # the same damage_by_block / coords_by_block used for D_frac etc.
    # ====================================================================
    per_band_damage = None
    cone_angle_info = None
    second_reflection_time = eval_time_s

    # Reuse config loaded earlier for cone adaptation
    config = eval_config
    peridigm_cfg = config.get("peridigm", {})
    adapt_cfg = peridigm_cfg.get("cone_angle_adaptation", {})
    damage_threshold = adapt_cfg.get("damage_threshold", 0.1)

    if damage_by_block and coords_by_block:
        try:
            per_band_damage = compute_per_band_hemisphere_damage(
                coords_by_block=coords_by_block,
                damage_by_block=damage_by_block,
                theta_band_edges_deg=theta_band_edges,
                fracture_zone_block=FRACTURE_ZONE_BLOCK,
                shell_body_block=SHELL_BODY_BLOCK,
                damage_threshold=damage_threshold,
            )
        except Exception as e:
            print(f"[Per-Band Damage] Error: {e}")
            import traceback
            traceback.print_exc()

    # ====================================================================
    # CRACK LINE CONTINUITY (ALL-TIMESTEP EVALUATION)
    # ====================================================================
    # Evaluates ALL timesteps in the Exodus file to measure crack line
    # continuity per band. This replaces single-timestep evaluation and
    # the damage_eval_reflection parameter entirely.
    # ====================================================================
    crack_line_metrics = None
    opt_cfg = config.get("optimization", {}) if config else {}
    use_crack_line = opt_cfg.get("use_crack_line_metrics", True)

    # Determine the Exodus file(s) to use for all-timestep evaluation
    # For parallel output, pass ALL partition files — each gets its own vtkExodusIIReader
    exodus_for_crack_line = None  # Path or list[Path]
    if serial_exodus:
        exodus_for_crack_line = max(serial_exodus, key=lambda p: p.stat().st_mtime)
    elif parallel_parts:
        # Pass all partition files; evaluate_all_timesteps opens one reader per file
        exodus_for_crack_line = sorted(parallel_parts)
        print(f"[Crack Line] Using {len(parallel_parts)} parallel Exodus partitions")

    if use_crack_line and exodus_for_crack_line and disc_files:
        cl_damage_threshold = float(opt_cfg.get("crack_line_damage_threshold", 0.5))
        cl_slice_mm = float(opt_cfg.get("crack_line_slice_mm", 1.0))

        display_name = exodus_for_crack_line.name if isinstance(exodus_for_crack_line, Path) else exodus_for_crack_line[0].name
        print(f"\n[Crack Line] Evaluating all timesteps for crack line continuity...")
        print(f"[Crack Line] Exodus: {display_name}")
        print(f"[Crack Line] Damage threshold: {cl_damage_threshold}, slice: {cl_slice_mm}mm")

        try:
            crack_line_metrics = evaluate_all_timesteps(
                exodus_file=exodus_for_crack_line,
                disc_file=disc_files[0],
                theta_band_edges_deg=theta_band_edges,
                fracture_block=FRACTURE_ZONE_BLOCK,
                protected_block=SHELL_BODY_BLOCK,
                damage_threshold=cl_damage_threshold,
                slice_mm=cl_slice_mm,
            )
        except Exception as e:
            print(f"[Crack Line] Error: {e}")
            import traceback
            traceback.print_exc()
    elif not use_crack_line:
        print("\n[Crack Line] Disabled (use_crack_line_metrics=false)")
    elif not exodus_for_crack_line:
        print("\n[Crack Line] No serial Exodus file found for all-timestep evaluation")
    elif not disc_files:
        print("\n[Crack Line] No discretization file found")

    # ====================================================================
    # FULL-ZONE BFS BREACH DETECTION
    # ====================================================================
    # Definitive structural disconnection check: BFS through all damaged
    # fracture zone particles from z_min to z_max. This confirms whether
    # the structure is actually split, complementing per-band crack line
    # continuity which is used for optimization feedback.
    # ====================================================================
    full_zone_breach_info = None
    if damage_by_block and coords_by_block and FRACTURE_ZONE_BLOCK in damage_by_block:
        try:
            frac_coords = np.array(coords_by_block[FRACTURE_ZONE_BLOCK])
            frac_damage = np.array(damage_by_block[FRACTURE_ZONE_BLOCK])

            # Connectivity radius = 1.5× particle spacing
            particle_spacing_mm = float(opt_cfg.get("particle_spacing_mm",
                                        config.get("peridigm", {}).get("particle_spacing_mm", 1.0)))
            connectivity_radius_m = 1.5 * particle_spacing_mm / 1000.0

            # Damage threshold from config (same as crack line)
            fzb_threshold = float(opt_cfg.get("crack_line_damage_threshold", 0.5))

            print(f"\n[Full Zone Breach] Checking structural disconnection...")
            print(f"[Full Zone Breach] Connectivity radius: {connectivity_radius_m*1000:.1f}mm, threshold: {fzb_threshold}")

            full_zone_breach_info = compute_full_zone_breach(
                coords=frac_coords,
                damage=frac_damage,
                damage_threshold=fzb_threshold,
                connectivity_radius_m=connectivity_radius_m,
            )
        except Exception as e:
            print(f"[Full Zone Breach] Error: {e}")
            import traceback
            traceback.print_exc()
    elif FRACTURE_ZONE_BLOCK not in damage_by_block:
        print("\n[Full Zone Breach] No fracture zone data — skipping")

    # ====================================================================
    # CONE ANGLE ADAPTATION
    # ====================================================================
    # Use crack_line_metrics full_crack_bands as depth if available,
    # otherwise fall back to per_band_damage penetration depth.
    # ====================================================================
    cone_angle_info = None
    if config:
        # Build a per_band_damage-like dict for cone angle adaptation
        # Prefer crack_line_metrics if available
        cone_depth_source = per_band_damage
        if crack_line_metrics is not None:
            # Synthesize a per_band_damage-compatible dict from crack line metrics
            cone_depth_source = {
                "damage_penetration_depth": crack_line_metrics["damage_penetration_depth"],
                "damage_penetration_pct": crack_line_metrics["damage_penetration_pct"],
            }

        if cone_depth_source is not None:
            # Read current cone angle from state
            current_cone_angle = 30.0
            if state_file.exists():
                with open(state_file, "r") as f:
                    _state = json.load(f)
                current_cone_angle = _state.get("cone_half_angle_deg", 30.0)

            cone_angle_info = compute_cone_angle_adaptation(
                per_band_damage=cone_depth_source,
                current_cone_angle_deg=current_cone_angle,
                config=config,
            )

    # Update state
    if not state_file.exists():
        print(f"No existing state file — creating new one: {state_file}")
        state = {"history": [{}], "peridigm_pending": True}
    else:
        with open(state_file, "r") as f:
            state = json.load(f)

    state["peridigm_results"] = {
        "D_fracture_zone": D_fracture,
        "D_shell_body": D_shell,
        "job_id": job_id,
        "damage_stats": damage_stats,
        "fracture_metrics": fracture_metrics,
        "shell_metrics": shell_metrics,
        "path_breach": breach_info,
        "full_zone_breach": full_zone_breach_info,
        "per_band_damage": per_band_damage,
        "crack_line_metrics": crack_line_metrics,
        "cone_angle_adaptation": cone_angle_info,
        "second_reflection_time_s": second_reflection_time,
    }
    state["peridigm_pending"] = False

    # Update cone angle in top-level state (read by SLURM for next iteration)
    if cone_angle_info is not None:
        state["cone_half_angle_deg"] = cone_angle_info["new_cone_angle_deg"]

    # Update history entry with Peridigm results
    if state["history"]:
        state["history"][-1]["D_fracture_zone"] = D_fracture
        state["history"][-1]["D_shell_body"] = D_shell
        state["history"][-1]["peridigm_job_id"] = job_id
        state["history"][-1]["peridigm_damage_stats"] = damage_stats
        state["history"][-1]["fracture_metrics"] = fracture_metrics
        state["history"][-1]["shell_metrics"] = shell_metrics
        # Breach detection: prefer full_zone_breach (definitive BFS), then crack_line, then path breach
        if full_zone_breach_info is not None:
            state["history"][-1]["fracture_breach"] = full_zone_breach_info["breach"]
            state["history"][-1]["full_zone_breach"] = full_zone_breach_info
        elif crack_line_metrics is not None:
            # Fallback: breach = all bands with particles have full crack continuity
            n_bands_with_particles = sum(1 for n in crack_line_metrics["fracture_zone"]["per_band_n_particles"] if n > 0)
            full_bands = crack_line_metrics["full_crack_bands"]
            state["history"][-1]["fracture_breach"] = (full_bands >= n_bands_with_particles) if n_bands_with_particles > 0 else False
        elif breach_info:
            state["history"][-1]["fracture_breach"] = breach_info["breach"]
        else:
            state["history"][-1]["fracture_breach"] = fracture_metrics.get("breach", False) if fracture_metrics else False

        # Blocked band / breach scores (from old path-based detection, still useful)
        if breach_info:
            state["history"][-1]["blocked_at_band"] = breach_info.get("blocked_at_band")
            state["history"][-1]["band_breach_scores"] = breach_info.get("band_breach_scores", [])
        else:
            state["history"][-1]["blocked_at_band"] = None

        # Crack depth: prefer crack_line_metrics, then per_band_damage, then path breach
        if crack_line_metrics is not None:
            state["history"][-1]["crack_depth"] = crack_line_metrics.get("damage_penetration_pct", 0.0)
        elif per_band_damage is not None:
            state["history"][-1]["crack_depth"] = per_band_damage.get("damage_penetration_pct", 0.0)
        elif breach_info:
            state["history"][-1]["crack_depth"] = breach_info["crack_depth"]
        else:
            state["history"][-1]["crack_depth"] = D_fracture
        # Per-band damage and cone angle
        if per_band_damage is not None:
            state["history"][-1]["per_band_damage"] = per_band_damage
        # Crack line metrics (new all-timestep evaluation)
        if crack_line_metrics is not None:
            state["history"][-1]["crack_line_metrics"] = crack_line_metrics
        if cone_angle_info is not None:
            state["history"][-1]["cone_half_angle_deg"] = cone_angle_info["new_cone_angle_deg"]

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    # Also update history.json
    history_file = output_dir / "history.json"
    with open(history_file, "w") as f:
        json.dump(state["history"], f, indent=2)

    print(f"\nPeridigm results processed and saved")
    print(f"D_fracture_zone: {D_fracture:.4f}")
    print(f"D_shell_body: {D_shell:.4f}")
    if fracture_metrics:
        print(f"Fracture completeness: {fracture_metrics.get('completeness', 0)*100:.1f}%")
        print(f"Fracture breach: {'YES - shell split!' if fracture_metrics.get('breach', False) else 'NO'}")
    if shell_metrics:
        print(f"Shell integrity: {shell_metrics.get('integrity', 0)*100:.1f}%")
    if breach_info:
        print(f"Path breach: {'YES' if breach_info['breach'] else 'NO'} (depth: {breach_info['crack_depth']:.1%})")
        if breach_info.get('blocked_at_band') is not None:
            print(f"Blocked at band: {breach_info['blocked_at_band']}")
    if crack_line_metrics:
        frac_scores = crack_line_metrics["fracture_zone"]["per_band_score"]
        full_cracks = crack_line_metrics["full_crack_bands"]
        n_bands = len(frac_scores)
        mean_score = sum(frac_scores) / n_bands if n_bands > 0 else 0
        print(f"Crack line: {full_cracks} full cracks, mean score: {mean_score:.3f}")
    if full_zone_breach_info:
        print(f"Full-zone BFS breach: {'YES' if full_zone_breach_info['breach'] else 'NO'} "
              f"(coverage: {full_zone_breach_info['coverage_pct']:.1%}, "
              f"z-span: {full_zone_breach_info['z_span_mm']:.1f}mm, "
              f"components: {full_zone_breach_info['n_components']})")

    return 0


if __name__ == "__main__":
    exit(main())
