#!/usr/bin/env python3
"""
Split-core mesh generator for flyback simulation.

Creates steel core as two separate halves that can separate during flyback event.
The split plane is along the z-axis (long axis of the egg), creating two "half-pipe" shapes.

This is a wrapper/modifier for the existing ShellGeometryGeneratorRefined that adds
split-core capability.
"""

import numpy as np
from pathlib import Path
from typing import Optional


def generate_split_core_peridigm(
    source_discretization: Path,
    output_dir: Path,
    split_axis: str = "x",  # Split plane normal: "x" or "y"
    gap_m: float = 0.0001,  # Small gap between halves (0.1mm)
    hollow_steel: bool = False,  # Only keep surface layer of steel
    steel_shell_layers: int = 2,  # Number of layers to keep when hollow
    steel_only: bool = False,  # Remove PLA (blocks 2,3) for diagnostic tests
) -> dict:
    """
    Take an existing Peridigm discretization and split the steel core into two halves.

    Args:
        source_discretization: Path to existing peridigm.txt file
        output_dir: Output directory for new files
        split_axis: Axis perpendicular to split plane ("x" or "y")
        gap_m: Gap between the two halves (meters)

    Returns:
        Dictionary with paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading source discretization: {source_discretization}")

    # Read existing discretization
    points = []
    with open(source_discretization, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                block_id = int(parts[3])
                volume = float(parts[4])
                points.append([x, y, z, block_id, volume])

    points = np.array(points)
    print(f"  Total points: {len(points)}")

    # Count blocks
    block_ids = points[:, 3].astype(int)
    unique_blocks, counts = np.unique(block_ids, return_counts=True)
    print(f"  Blocks: {dict(zip(unique_blocks, counts))}")

    # Hollow steel: remove interior steel particles, keep only surface shell
    if hollow_steel:
        steel_mask = block_ids == 1
        pla_mask = (block_ids == 2) | (block_ids == 3)
        n_steel_before = np.sum(steel_mask)

        if n_steel_before > 0 and np.any(pla_mask):
            try:
                from scipy.spatial import cKDTree
                pla_tree = cKDTree(points[pla_mask, :3])
                steel_dists, _ = pla_tree.query(points[steel_mask, :3])

                # Estimate steel spacing from nearest-neighbor distance
                steel_pts = points[steel_mask, :3]
                steel_tree = cKDTree(steel_pts)
                nn_dists, _ = steel_tree.query(steel_pts, k=2)
                steel_spacing = np.median(nn_dists[:, 1])

                threshold = steel_shell_layers * steel_spacing
                keep_steel = steel_dists < threshold

                # Build mask for all points: keep all non-steel + surface steel
                keep_all = ~steel_mask  # keep all PLA
                steel_indices = np.where(steel_mask)[0]
                keep_all[steel_indices[keep_steel]] = True

                n_steel_after = np.sum(keep_steel)
                print(f"  Hollow steel: {n_steel_before} -> {n_steel_after} "
                      f"(removed {n_steel_before - n_steel_after} interior, "
                      f"threshold={threshold*1000:.1f}mm, {steel_shell_layers} layers)")

                points = points[keep_all]
                block_ids = points[:, 3].astype(int)
            except ImportError:
                print("  WARNING: scipy not available, skipping hollow steel")

    # Check if mesh is already pre-split (has block 4)
    has_block_4 = 4 in unique_blocks
    if has_block_4:
        print(f"  Mesh already has block 4 (pre-split) - skipping split step")
        new_block_ids = block_ids.copy()
        new_points = points[:, :3].copy()
    else:
        # Split axis index
        axis_idx = {"x": 0, "y": 1, "z": 2}[split_axis.lower()]

        # Find steel core points (block_id == 1)
        steel_mask = block_ids == 1
        n_steel = np.sum(steel_mask)
        print(f"  Steel core points: {n_steel}")

        if n_steel == 0:
            print("  WARNING: No steel core points found!")
            return None

        # Split steel core based on position relative to axis
        # Points with coord > 0 go to block_1, points with coord <= 0 go to block_4
        steel_coords = points[steel_mask, axis_idx]

        # Create new block IDs
        new_block_ids = block_ids.copy()

        # Steel points on negative side of split axis -> block_4
        steel_negative_mask = steel_mask & (points[:, axis_idx] <= 0)
        new_block_ids[steel_negative_mask] = 4

        # Add small gap by shifting the halves apart
        new_points = points[:, :3].copy()
        if gap_m > 0:
            # Shift block_1 (positive side) in +axis direction
            # Shift block_4 (negative side) in -axis direction
            steel_positive_mask = steel_mask & (points[:, axis_idx] > 0)
            new_points[steel_positive_mask, axis_idx] += gap_m / 2
            new_points[steel_negative_mask, axis_idx] -= gap_m / 2

        # Count new blocks
        unique_new, counts_new = np.unique(new_block_ids, return_counts=True)
        print(f"  New blocks: {dict(zip(unique_new, counts_new))}")

    # Steel-only: remove PLA particles (blocks 2, 3) for diagnostic tests
    if steel_only:
        keep = (new_block_ids != 2) & (new_block_ids != 3)
        n_removed = (~keep).sum()
        new_points = new_points[keep]
        new_block_ids = new_block_ids[keep]
        points = points[keep]
        print(f"  Steel-only: removed {n_removed} PLA particles, {len(points)} remaining")

    # Write new discretization
    disc_path = output_dir / "split_core_peridigm.txt"
    with open(disc_path, 'w') as f:
        for i in range(len(points)):
            x, y, z = new_points[i]
            blk = int(new_block_ids[i])
            vol = points[i, 4]
            f.write(f"{x:.10e} {y:.10e} {z:.10e} {blk} {vol:.10e}\n")

    print(f"  Written: {disc_path}")

    # Write nodesets
    all_ids = np.arange(1, len(points) + 1)

    # Z coordinates and bounds
    z_coords = new_points[:, 2]
    z_max = z_coords.max()
    z_min = z_coords.min()
    far_end_mask = z_coords > (z_max - 0.003)  # 3mm band at z_max

    # Separate nodesets for shell vs steel at far end
    # Shell (blocks 2, 3) at far end - fully fixed
    shell_mask = (new_block_ids == 2) | (new_block_ids == 3)
    shell_far_end_mask = far_end_mask & shell_mask
    shell_far_end_ids = all_ids[shell_far_end_mask]

    shell_fixed_path = output_dir / "nodeset_shell_fixed.txt"
    with open(shell_fixed_path, 'w') as f:
        for nid in shell_far_end_ids:
            f.write(f"{nid}\n")
    print(f"  Written: {shell_fixed_path} ({len(shell_far_end_ids)} shell points)")

    # Steel (blocks 1, 4) at far end - fixed in Z only (free in X/Y for separation)
    steel_all_mask = (new_block_ids == 1) | (new_block_ids == 4)
    steel_far_end_mask = far_end_mask & steel_all_mask
    steel_far_end_ids = all_ids[steel_far_end_mask]

    steel_fixed_z_path = output_dir / "nodeset_steel_fixed_z.txt"
    with open(steel_fixed_z_path, 'w') as f:
        for nid in steel_far_end_ids:
            f.write(f"{nid}\n")
    print(f"  Written: {steel_fixed_z_path} ({len(steel_far_end_ids)} steel points - Z only)")

    # Legacy: combined fixed end for compatibility
    fixed_end_ids = all_ids[far_end_mask]
    fixed_end_path = output_dir / "nodeset_fixed_end.txt"
    with open(fixed_end_path, 'w') as f:
        for nid in fixed_end_ids:
            f.write(f"{nid}\n")
    print(f"  Written: {fixed_end_path} ({len(fixed_end_ids)} points total)")

    # Steel core halves nodesets (for applying contact or constraints)
    steel_1_ids = all_ids[new_block_ids == 1]
    steel_4_ids = all_ids[new_block_ids == 4]

    steel_1_path = output_dir / "nodeset_steel_half_1.txt"
    with open(steel_1_path, 'w') as f:
        for nid in steel_1_ids:
            f.write(f"{nid}\n")
    print(f"  Written: {steel_1_path} ({len(steel_1_ids)} points)")

    steel_4_path = output_dir / "nodeset_steel_half_4.txt"
    with open(steel_4_path, 'w') as f:
        for nid in steel_4_ids:
            f.write(f"{nid}\n")
    print(f"  Written: {steel_4_path} ({len(steel_4_ids)} points)")

    # All shell+steel nodeset (for initial velocity boundary condition)
    shell_steel_mask = (new_block_ids == 1) | (new_block_ids == 2) | (new_block_ids == 3) | (new_block_ids == 4)
    shell_steel_ids = all_ids[shell_steel_mask]

    shell_steel_path = output_dir / "nodeset_shell_steel_all.txt"
    with open(shell_steel_path, 'w') as f:
        for nid in shell_steel_ids:
            f.write(f"{nid}\n")
    print(f"  Written: {shell_steel_path} ({len(shell_steel_ids)} shell+steel points)")

    return {
        "discretization": str(disc_path),
        "nodeset_fixed_end": str(fixed_end_path),
        "nodeset_shell_fixed": str(shell_fixed_path),
        "nodeset_steel_fixed_z": str(steel_fixed_z_path),
        "nodeset_shell_steel_all": str(shell_steel_path),
        "nodeset_steel_half_1": str(steel_1_path),
        "nodeset_steel_half_4": str(steel_4_path),
        "n_points": len(points),
        "n_steel_half_1": len(steel_1_ids),
        "n_steel_half_4": len(steel_4_ids),
        "n_shell_fixed": len(shell_far_end_ids),
        "n_steel_fixed_z": len(steel_far_end_ids),
        "n_shell_steel_all": len(shell_steel_ids),
    }


def generate_flyback_split_core_yaml(
    split_core_files: dict,
    output_dir: Path,
    impact_velocity: float = 250.0,
    final_time: float = 1.0e-4,  # 100 µs - captures full impact + fracture, avoids bouncing artifacts
    cone_position: str = "inside",  # "inside" or "outside"
    state_based: bool = False,  # Use Elastic Correspondence for accurate Poisson ratio
    horizon: float = None,  # Override horizon (auto-selected based on state_based if None)
    pla_critical_stretch: float = 0.011,  # Literature-based for Gc=6000 J/m²
    fracture_critical_stretch: float = 0.007,  # Weaker than shell for controlled fracture
    fix_far_end: bool = False,  # Fix far-end shell+steel nodes (z_max) to prevent free-body explosion
    wall: bool = False,  # Add rigid contact wall at far end instead of clamped nodes
    wall_spacing: float = 0.001,  # Wall particle spacing (m), default 1mm
    particle_spacing: float = None,  # Particle spacing (m); contact_radius = 0.9*spacing (must be < spacing per Peridigm docs)
    cone_initial_velocity: bool = False,  # Use initial velocity instead of prescribed displacement
    cone_mass_kg: float = None,  # Override cone mass (effective density); None = native PLA
    cone_no_damage: bool = False,  # Make cone indestructible (no damage model)
    rigid_cone: bool = False,  # Use steel stiffness for cone (prevents crushing against steel)
    cone_half_angle_deg: float = 30.0,  # Cone half-angle in degrees (15=narrow wedge, 60=wide spreader)
    cone_height_m: float = 0.010,  # Cone height in meters (default 10mm; use 30mm+ for narrow wedge cones)
    cone_cylinder_radius_m: float = None,  # If set, cone tapers to this radius then continues as cylinder
    spring_constant: float = 1.0e12,  # Contact spring constant (N/m); reduce for wider cones to avoid tiny timestep
    steel_stiffness_factor: float = 1.0,  # Multiply steel K,G by this factor for near-rigid behavior (100x → ~0.1% strain, ~10x slower)
    steel_mass_scaling: bool = False,  # Scale steel density with stiffness factor (no timestep penalty, but heavier steel)
    output_frequency: int = 100,  # Exodus output every N steps (10=debug, 100=production; disk_impact uses 350)
    search_frequency: int = 100,  # Contact search every N steps (disk_impact uses 100; 10=conservative)
    debug_output: bool = False,  # If True, output Number_Of_Neighbors, Contact_Force_Density, Proc_Num
    steel_only: bool = False,  # Omit PLA blocks from YAML (diagnostic: steel + cone + wall only)
) -> Path:
    """
    Generate Peridigm YAML for flyback simulation with split core.

    The cone enters through the hole and pushes the steel core halves apart,
    which in turn splits the shell along the fracture zones.
    """
    output_dir = Path(output_dir)
    disc_file = split_core_files["discretization"]

    # Select horizon based on material model
    # State-based correspondence model needs larger horizon to ensure at least 3 neighbors
    # for rotation tensor computation, especially as bonds break during fracture
    if horizon is None:
        if state_based:
            horizon = 0.006  # 6mm - larger for correspondence model stability
        else:
            horizon = 0.004  # 4mm - standard for bond-based
    print(f"Using horizon: {horizon}m ({'state-based' if state_based else 'bond-based'})")
    steel_density = 7800.0 * (steel_stiffness_factor if steel_mass_scaling else 1.0)
    if steel_stiffness_factor != 1.0:
        print(f"Steel stiffness factor: {steel_stiffness_factor}x (K={160e9*steel_stiffness_factor:.2e}, G={80e9*steel_stiffness_factor:.2e})")
        if steel_mass_scaling:
            print(f"Steel mass scaling: density={steel_density:.0f} kg/m³ (no timestep penalty)")

    # Read discretization to get geometry bounds
    points = []
    with open(disc_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                block_id = int(parts[3])
                points.append([x, y, z, block_id])

    points = np.array(points)
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    # Determine steel split geometry for bond filter
    # Blocks 1 and 4 are the two steel halves — we need to prevent bonds
    # from forming across the split plane (gap < horizon → bonds would form
    # across the gap and permanently weld the halves together since steel
    # has no damage model)
    steel_mask = (points[:, 3] == 1) | (points[:, 3] == 4)
    if np.any(steel_mask):
        steel_pts = points[steel_mask, :3]
        # Determine split axis: block 1 has positive coords, block 4 has negative
        blk1_mask = points[:, 3] == 1
        blk4_mask = points[:, 3] == 4
        if np.any(blk1_mask) and np.any(blk4_mask):
            mean1 = points[blk1_mask, :3].mean(axis=0)
            mean4 = points[blk4_mask, :3].mean(axis=0)
            split_axis_idx = int(np.argmax(np.abs(mean1 - mean4)))
            split_axis_names = {0: "x", 1: "y", 2: "z"}
            detected_split_axis = split_axis_names[split_axis_idx]

            # Compute filter rectangle extent on the non-split axes
            # Shrink by 1mm margin to avoid cutting PLA bonds near interface
            non_split = [i for i in range(3) if i != split_axis_idx]
            # Rectangle must cover ALL steel particles plus 1mm margin
            # to catch bonds whose midpoints fall near the edge.
            # Previous bug: subtracting margin left 138 particles unfiltered!
            filter_extent_0 = np.abs(steel_pts[:, non_split[0]]).max() + 0.001
            filter_extent_1 = np.abs(steel_pts[:, non_split[1]]).max() + 0.001

            print(f"Steel split detected: axis={detected_split_axis}, "
                  f"filter extents: {non_split[0]}=±{filter_extent_0*1e3:.1f}mm, "
                  f"{non_split[1]}=±{filter_extent_1*1e3:.1f}mm")
        else:
            detected_split_axis = None
            filter_extent_0 = filter_extent_1 = 0
    else:
        detected_split_axis = None
        filter_extent_0 = filter_extent_1 = 0

    # Cone position: tip starts just outside the assembly at z_min
    # This prevents particle overlap for any cone angle
    gap = 0.0005  # 0.5mm clearance between cone tip and assembly
    cone_height = cone_height_m
    cone_tip_z = z_min - gap
    cone_base_z = cone_tip_z - cone_height

    # Generate cone (or cone+cylinder) points
    spacing = particle_spacing if particle_spacing else 0.001  # match discretization or 1mm default
    cone_half_angle = cone_half_angle_deg
    cone_base_radius = cone_height * np.tan(np.radians(cone_half_angle))

    # Cone-cylinder composite: cone tapers to cylinder_radius, then continues as cylinder
    if cone_cylinder_radius_m is not None and cone_cylinder_radius_m > 0:
        cyl_r = cone_cylinder_radius_m
        # Height where cone reaches cylinder radius (measured from tip)
        cone_taper_height = cyl_r / np.tan(np.radians(cone_half_angle))
        if cone_taper_height >= cone_height:
            # Cone never reaches cylinder radius — pure cone
            cone_taper_height = cone_height
            cylinder_height = 0.0
        else:
            cylinder_height = cone_height - cone_taper_height
        print(f"Cone-cylinder: {cone_half_angle}° taper for {cone_taper_height*1000:.1f}mm "
              f"→ {cyl_r*1000:.1f}mm radius cylinder for {cylinder_height*1000:.1f}mm "
              f"(total {cone_height*1000:.0f}mm)")
    else:
        cyl_r = None
        cone_taper_height = cone_height
        cylinder_height = 0.0
        print(f"Cone: half-angle={cone_half_angle}°, height={cone_height*1000:.0f}mm, "
              f"base_radius={cone_base_radius*1000:.1f}mm")

    print(f"Cone position: tip_z={cone_tip_z*1000:.1f}mm, base_z={cone_base_z*1000:.1f}mm "
          f"(z_min={z_min*1000:.1f}mm, gap={gap*1000:.1f}mm)")

    cone_points = []
    n_layers = int(cone_height / spacing) + 1

    for i in range(n_layers):
        z = cone_base_z + i * spacing
        dist_from_tip = cone_tip_z - z  # distance from tip (tip is at top z)

        if dist_from_tip <= cone_taper_height:
            # In the cone taper section (near tip)
            radius = dist_from_tip * np.tan(np.radians(cone_half_angle))
        else:
            # In the cylinder section (away from tip)
            radius = cyl_r if cyl_r is not None else 0.0

        if radius < spacing / 2:
            cone_points.append((0, 0, z))
        else:
            n_rings = max(1, int(radius / spacing))
            for r_idx in range(n_rings + 1):
                r = r_idx * spacing
                if r > radius:
                    continue
                if r < spacing / 2:
                    cone_points.append((0, 0, z))
                else:
                    n_theta = max(6, int(2 * np.pi * r / spacing))
                    for t_idx in range(n_theta):
                        theta = 2 * np.pi * t_idx / n_theta
                        x = r * np.cos(theta)
                        y = r * np.sin(theta)
                        cone_points.append((x, y, z))

    cone_points = np.array(cone_points)
    print(f"Generated {len(cone_points)} cone points")

    # Compute cone effective density for mass loading
    cone_particle_volume = spacing ** 3  # volume per cone particle
    total_cone_volume = len(cone_points) * cone_particle_volume
    if cone_mass_kg is not None and cone_mass_kg > 0:
        cone_density = cone_mass_kg / total_cone_volume
        native_mass = 1240.0 * total_cone_volume
        print(f"Cone mass loading: {cone_mass_kg*1000:.1f}g target "
              f"(native PLA mass: {native_mass*1000:.2f}g, "
              f"effective density: {cone_density:.0f} kg/m³)")
    else:
        cone_density = 1240.0  # default PLA density

    # Generate wall particles at far end (z_max) if requested
    wall_points = np.empty((0, 3))
    if wall:
        # Find extent of assembly near z_max
        near_top = points[:, 2] > (z_max - 0.005)
        if np.any(near_top):
            r_max = np.sqrt(points[near_top, 0]**2 + points[near_top, 1]**2).max()
        else:
            r_max = 0.015  # fallback 15mm
        r_max += 0.003  # 3mm margin

        wall_z = z_max + wall_spacing  # 1 spacing gap
        ws = wall_spacing
        wall_list = []
        nx = int(2 * r_max / ws) + 1
        for ix in range(nx):
            for iy in range(nx):
                x = -r_max + ix * ws
                y = -r_max + iy * ws
                if x*x + y*y <= r_max * r_max:
                    wall_list.append([x, y, wall_z])
        wall_points = np.array(wall_list)
        print(f"Generated {len(wall_points)} wall particles "
              f"(disc r={r_max*1000:.1f}mm at z={wall_z*1000:.1f}mm, spacing={ws*1000:.1f}mm)")

    # Write combined discretization with cone + wall
    cone_volume = spacing ** 3
    wall_volume = wall_spacing ** 3
    combined_file = output_dir / "flyback_split_core_discretization.txt"

    # Read original points
    with open(disc_file, 'r') as f:
        original_lines = f.readlines()

    n_original = len(original_lines)
    with open(combined_file, 'w') as f:
        # Original points
        for line in original_lines:
            f.write(line)
        # Cone points (block 99)
        for p in cone_points:
            f.write(f"{p[0]:.10e} {p[1]:.10e} {p[2]:.10e} 99 {cone_volume:.10e}\n")
        # Wall points (block 98)
        for p in wall_points:
            f.write(f"{p[0]:.10e} {p[1]:.10e} {p[2]:.10e} 98 {wall_volume:.10e}\n")

    print(f"Written: {combined_file}")

    # Cone nodeset
    cone_start_idx = n_original
    cone_nodeset_file = output_dir / "nodeset_cone.txt"
    with open(cone_nodeset_file, 'w') as f:
        for i in range(len(cone_points)):
            f.write(f"{cone_start_idx + i + 1}\n")
    print(f"Written: {cone_nodeset_file}")

    # Wall nodeset
    wall_nodeset_file = output_dir / "nodeset_wall.txt"
    if wall and len(wall_points) > 0:
        wall_start_idx = n_original + len(cone_points)
        with open(wall_nodeset_file, 'w') as f:
            for i in range(len(wall_points)):
                f.write(f"{wall_start_idx + i + 1}\n")
        print(f"Written: {wall_nodeset_file} ({len(wall_points)} points)")

    # Generate YAML with proper boundary conditions
    shell_fixed_file = split_core_files["nodeset_shell_fixed"]
    steel_fixed_z_file = split_core_files["nodeset_steel_fixed_z"]

    # Build materials section based on state_based flag
    if state_based:
        # State-based correspondence model requires more stabilization for high-velocity impact:
        # - Hourglass Coefficient 0.1 (higher end of 0.02-0.15 range from Littlewood 2024)
        # - Larger horizon (set earlier) ensures adequate neighbors for rotation tensor
        pla_material = """    PLA:
      # State-based: accurate Poisson ratio (nu=0.36)
      Material Model: Elastic Correspondence
      Density: 1240.0
      Young's Modulus: 2.5e9
      Poisson's Ratio: 0.36
      Hourglass Coefficient: 0.1"""
        if rigid_cone:
            cone_material = f"""    Cone PLA:
      Material Model: Elastic
      Density: {cone_density}
      Bulk Modulus: 1.60e11
      Shear Modulus: 8.00e10"""
        else:
            cone_material = f"""    Cone PLA:
      # State-based: accurate Poisson ratio (nu=0.36)
      Material Model: Elastic Correspondence
      Density: {cone_density}
      Young's Modulus: 2.5e9
      Poisson's Ratio: 0.36
      Hourglass Coefficient: 0.1"""
    else:
        pla_material = """    PLA:
      Material Model: Elastic
      Density: 1240.0
      Bulk Modulus: 2.98e9
      Shear Modulus: 0.92e9"""
        if rigid_cone:
            cone_material = f"""    Cone PLA:
      Material Model: Elastic
      Density: {cone_density}
      Bulk Modulus: 1.60e11
      Shear Modulus: 8.00e10"""
        else:
            cone_material = f"""    Cone PLA:
      Material Model: Elastic
      Density: {cone_density}
      Bulk Modulus: 2.98e9
      Shear Modulus: 0.92e9"""

    # Nodeset for shell+steel (initial velocity)
    shell_steel_file = split_core_files.get("nodeset_shell_steel_all", "")

    # Build bond filter YAML (must go INSIDE Discretization section, not top-level)
    bond_filter_yaml = ""
    if detected_split_axis is not None:
        bf_axis_idx = {"x": 0, "y": 1, "z": 2}[detected_split_axis]
        bf_non_split = [i for i in range(3) if i != bf_axis_idx]
        bf_labels = {0: "X", 1: "Y", 2: "Z"}
        bf_normal = [0.0, 0.0, 0.0]
        bf_normal[bf_axis_idx] = 1.0
        bf_bottom = [0.0, 0.0, 0.0]
        bf_bottom[bf_non_split[0]] = 1.0
        bf_bot_len = 2 * filter_extent_0
        bf_side_len = 2 * filter_extent_1
        bf_corner = [0.0, 0.0, 0.0]
        bf_corner[bf_non_split[0]] = -filter_extent_0
        bf_corner[bf_non_split[1]] = -filter_extent_1
        bond_filter_yaml = f"""    Bond Filters:
      Steel Split Plane:
        Type: Rectangular_Plane
        Normal_{bf_labels[bf_axis_idx]}: {bf_normal[bf_axis_idx]}
        Normal_{bf_labels[bf_non_split[0]]}: 0.0
        Normal_{bf_labels[bf_non_split[1]]}: 0.0
        Lower_Left_Corner_{bf_labels[bf_axis_idx]}: 0.0
        Lower_Left_Corner_{bf_labels[bf_non_split[0]]}: {bf_corner[bf_non_split[0]]}
        Lower_Left_Corner_{bf_labels[bf_non_split[1]]}: {bf_corner[bf_non_split[1]]}
        Bottom_Unit_Vector_{bf_labels[bf_axis_idx]}: 0.0
        Bottom_Unit_Vector_{bf_labels[bf_non_split[0]]}: {bf_bottom[bf_non_split[0]]}
        Bottom_Unit_Vector_{bf_labels[bf_non_split[1]]}: 0.0
        Bottom_Length: {bf_bot_len}
        Side_Length: {bf_side_len}"""

    # Build conditional sections for steel_only mode
    if steel_only:
        materials_pla = ""
        damage_pla = ""
        blocks_pla = ""
    else:
        materials_pla = f"\n{pla_material}"
        damage_pla = f"""
    PLA Damage:
      Damage Model: Critical Stretch
      Critical Stretch: {pla_critical_stretch}
    Fracture Damage:
      Damage Model: Critical Stretch
      Critical Stretch: {fracture_critical_stretch}"""
        blocks_pla = f"""
    # PLA protected shell
    PLA Block:
      Block Names: block_2
      Material: PLA
      Horizon: {horizon}
      Damage Model: PLA Damage

    # PLA fracture zones
    PLA Fracture Block:
      Block Names: block_3
      Material: PLA
      Horizon: {horizon}
      Damage Model: Fracture Damage"""

    bc_mode = "initial velocity" if cone_initial_velocity else "prescribed displacement"
    mode_label = "steel-only diagnostic" if steel_only else "v6"
    yaml_content = f"""# Flyback simulation with split steel core ({mode_label})
# Cone impacts {"steel" if steel_only else "shell+steel"} at {impact_velocity} m/s ({bc_mode} mode)
#
# KEY PHYSICS:
# - Cone {"starts at" if cone_initial_velocity else "pulled at"} {impact_velocity} m/s in +z
# - {"Cone decelerates naturally via contact (finite KE)" if cone_initial_velocity else "Cone has infinite momentum (prescribed displacement)"}
# - Cone wedges steel halves apart as it passes through
{"# - NO PLA shell (steel-only diagnostic)" if steel_only else "# - Shell fractures along designed weak zones"}

Peridigm:
  Discretization:
    Type: Text File
    Input Mesh File: {combined_file}
{bond_filter_yaml}

  Materials:{materials_pla}
    Steel:
      Material Model: Elastic
      Density: {steel_density:.1f}
      Bulk Modulus: {160.0e9 * steel_stiffness_factor:.6e}
      Shear Modulus: {80.0e9 * steel_stiffness_factor:.6e}
{cone_material}

  Damage Models:{damage_pla}
    Cone Damage:
      Damage Model: Critical Stretch
      Critical Stretch: 0.015

  Blocks:
    # Steel core half 1 (positive side)
    Steel Block 1:
      Block Names: block_1
      Material: Steel
      Horizon: {horizon}

    # Steel core half 2 (negative side)
    Steel Block 4:
      Block Names: block_4
      Material: Steel
      Horizon: {horizon}
{blocks_pla}

    # Cone
    Cone Block:
      Block Names: block_99
      Material: Cone PLA
      Horizon: {horizon}
"""
    if not cone_no_damage:
        yaml_content += """      Damage Model: Cone Damage
"""

    # Add wall block if requested
    if wall and len(wall_points) > 0:
        yaml_content += f"""
    # Rigid wall at far end (contact boundary)
    Wall Block:
      Block Names: block_98
      Material: Steel
      Horizon: {horizon}
"""

    yaml_content += f"""
  Boundary Conditions:
    # Nodesets
    Node Set Cone: {cone_nodeset_file}
"""

    if cone_initial_velocity:
        # Initial velocity: cone starts at impact_velocity, decelerates via contact
        # Finite kinetic energy = 0.5 * m * v^2
        yaml_content += f"""
    # Cone: initial velocity in +z (finite momentum, decelerates via contact)
    Cone Initial Velocity Z:
      Type: Initial Velocity
      Node Set: Node Set Cone
      Coordinate: z
      Value: "{impact_velocity}"
"""
    else:
        # Prescribed displacement: cone moves at constant velocity (infinite momentum)
        yaml_content += f"""
    # Cone: prescribed displacement in +z (infinite momentum, never slows)
    Cone Pull Z:
      Type: Prescribed Displacement
      Node Set: Node Set Cone
      Coordinate: z
      Value: "{impact_velocity}*t"
"""

    # Add wall boundary conditions (rigid contact wall at far end)
    if wall and len(wall_points) > 0:
        yaml_content += f"""
    # Rigid wall at z_max: fixed in all directions
    # Assembly compresses against this wall via contact (not clamped nodes)
    Node Set Wall: {wall_nodeset_file}

    Wall Fixed X:
      Type: Prescribed Displacement
      Node Set: Node Set Wall
      Coordinate: x
      Value: "0.0"
    Wall Fixed Y:
      Type: Prescribed Displacement
      Node Set: Node Set Wall
      Coordinate: y
      Value: "0.0"
    Wall Fixed Z:
      Type: Prescribed Displacement
      Node Set: Node Set Wall
      Coordinate: z
      Value: "0.0"
"""

    # Add far-end boundary conditions if requested
    if fix_far_end:
        yaml_content += f"""
    # Far-end constraints: fix shell and steel at z_max to prevent free-body explosion
    # This represents the shell being held in a housing/mechanism
    Node Set Shell Fixed: {shell_fixed_file}
    Node Set Steel Fixed Z: {steel_fixed_z_file}

    Shell Fixed X:
      Type: Prescribed Displacement
      Node Set: Node Set Shell Fixed
      Coordinate: x
      Value: "0.0"
    Shell Fixed Y:
      Type: Prescribed Displacement
      Node Set: Node Set Shell Fixed
      Coordinate: y
      Value: "0.0"
    Shell Fixed Z:
      Type: Prescribed Displacement
      Node Set: Node Set Shell Fixed
      Coordinate: z
      Value: "0.0"

    # Steel far end: fixed in Z only (free in X/Y so halves can separate)
    Steel Fixed Z:
      Type: Prescribed Displacement
      Node Set: Node Set Steel Fixed Z
      Coordinate: z
      Value: "0.0"
"""

    debug_vars = ""
    if debug_output:
        debug_vars = """      Number_Of_Neighbors: true
      Contact_Force_Density: true
      Proc_Num: true
"""

    yaml_content += f"""
  Solver:
    Initial Time: 0.0
    Final Time: {final_time}
    Verlet:
      Safety Factor: {0.4 if state_based else 0.7}

  Output:
    Output File Type: ExodusII
    Output Filename: {output_dir / 'flyback_split_core'}
    Output Frequency: {output_frequency}
    Output Variables:
      Displacement: true
      Velocity: true
      Damage: true
{debug_vars}

  Contact:
    # Contact radius per Peridigm maintainer recommendation (Littlewood, GH #123):
    # "Setting it to 0.9 times the grid spacing seems like a good starting point."
    # Contact is all-to-all for non-bonded nodes (self-contact always active).
    # Must be < particle spacing to avoid spurious repulsion in undeformed config.
    Search Radius: {3 * particle_spacing if particle_spacing else 0.5 * horizon}
    Search Frequency: {search_frequency}
    Models:
      My Contact Model:
        Contact Model: Short Range Force
        Contact Radius: {0.9 * particle_spacing if particle_spacing else 0.2 * horizon}
        Spring Constant: {spring_constant:.1e}
    Interactions:
      # Note: Peridigm contact is always all-to-all for non-bonded nodes regardless
      # of block specifications (Littlewood, GH #123). Explicit pairs are documentation only.
      General Contact:
        Contact Model: My Contact Model
"""

    yaml_file = output_dir / "flyback_split_core.yaml"
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    print(f"Written: {yaml_file}")

    return yaml_file


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate split-core flyback simulation")
    parser.add_argument("--source", required=True, help="Source peridigm discretization file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--split-axis", default="x", choices=["x", "y"], help="Split axis")
    parser.add_argument("--velocity", type=float, default=250.0, help="Impact velocity m/s")
    parser.add_argument("--final-time", type=float, default=1.0e-4, help="Simulation time (s) - default 100µs")
    parser.add_argument("--gap", type=float, default=0.0005, help="Gap between core halves (m) - 0.5mm tolerance for release")
    parser.add_argument("--state-based", action="store_true", help="Use state-based Elastic Correspondence material model (accurate Poisson ratio)")
    parser.add_argument("--horizon", type=float, default=None, help="Override horizon (default: 0.006 for state-based, 0.004 for bond-based)")
    parser.add_argument("--pla-critical-stretch", type=float, default=0.011, help="PLA critical stretch (default: 0.011 for Gc=6000 J/m²)")
    parser.add_argument("--fracture-critical-stretch", type=float, default=0.007, help="Fracture zone critical stretch (default: 0.007, weaker than shell)")
    parser.add_argument("--fix-far-end", action="store_true", help="Fix far-end nodes (z_max) to prevent free-body explosion")
    parser.add_argument("--wall", action="store_true", help="Add rigid contact wall at far end (better than clamped nodes)")
    parser.add_argument("--wall-spacing", type=float, default=0.001, help="Wall particle spacing in meters (default: 0.001 = 1mm)")
    parser.add_argument("--hollow-steel", action="store_true", help="Remove interior steel particles (keep surface shell only)")
    parser.add_argument("--steel-shell-layers", type=int, default=2, help="Number of steel layers to keep when hollow (default: 2)")
    parser.add_argument("--particle-spacing", type=float, default=None, help="Particle spacing (m) for contact radius (decoupled from horizon)")
    parser.add_argument("--initial-velocity", action="store_true", help="Use initial velocity instead of prescribed displacement (finite momentum)")
    parser.add_argument("--cone-mass-kg", type=float, default=None, help="Override cone mass in kg (effective density); default = native PLA mass")
    parser.add_argument("--cone-no-damage", action="store_true", help="Make cone indestructible (no damage model)")
    parser.add_argument("--rigid-cone", action="store_true", help="Use steel stiffness for cone (prevents crushing)")
    parser.add_argument("--cone-half-angle", type=float, default=30.0, help="Cone half-angle in degrees (default: 30)")
    parser.add_argument("--cone-height", type=float, default=0.010, help="Cone height in meters (default: 0.010 = 10mm; use 0.030+ for narrow wedge cones)")
    parser.add_argument("--cone-cylinder-radius", type=float, default=None, help="If set, cone tapers to this radius (m) then continues as cylinder (e.g. 0.002 = 2mm)")
    parser.add_argument("--spring-constant", type=float, default=1.0e12, help="Contact spring constant in N/m (default: 1e12; reduce for wider cones)")
    parser.add_argument("--steel-stiffness-factor", type=float, default=1.0, help="Multiply steel K,G by this factor for near-rigid behavior (100=rigid, ~10x slower)")
    parser.add_argument("--steel-mass-scaling", action="store_true", help="Scale steel density with stiffness factor (no timestep penalty)")
    parser.add_argument("--output-frequency", type=int, default=100, help="Exodus output every N steps (10=debug, 100=production; default: 100)")
    parser.add_argument("--search-frequency", type=int, default=100, help="Contact search every N steps (default: 100, matches disk_impact example)")
    parser.add_argument("--debug-output", action="store_true", help="Include Number_Of_Neighbors, Contact_Force_Density, Proc_Num in output")
    parser.add_argument("--steel-only", action="store_true", help="Remove PLA (blocks 2,3) for diagnostic tests (steel + cone + wall only)")
    args = parser.parse_args()

    print("=" * 70)
    print("Split-Core Flyback Simulation Generator")
    print("=" * 70)

    # Generate split core discretization
    split_files = generate_split_core_peridigm(
        source_discretization=Path(args.source),
        output_dir=Path(args.output_dir),
        split_axis=args.split_axis,
        gap_m=args.gap,
        hollow_steel=args.hollow_steel,
        steel_shell_layers=args.steel_shell_layers,
        steel_only=args.steel_only,
    )

    if split_files is None:
        print("ERROR: Failed to generate split core")
        return 1

    # Generate Peridigm YAML
    yaml_file = generate_flyback_split_core_yaml(
        split_core_files=split_files,
        output_dir=Path(args.output_dir),
        impact_velocity=args.velocity,
        final_time=args.final_time,
        state_based=args.state_based,
        horizon=args.horizon,
        pla_critical_stretch=args.pla_critical_stretch,
        fracture_critical_stretch=args.fracture_critical_stretch,
        fix_far_end=args.fix_far_end,
        wall=args.wall,
        wall_spacing=args.wall_spacing,
        particle_spacing=args.particle_spacing,
        cone_initial_velocity=args.initial_velocity,
        cone_mass_kg=args.cone_mass_kg,
        cone_no_damage=args.cone_no_damage,
        rigid_cone=args.rigid_cone,
        cone_half_angle_deg=args.cone_half_angle,
        cone_height_m=args.cone_height,
        cone_cylinder_radius_m=args.cone_cylinder_radius,
        spring_constant=args.spring_constant,
        steel_stiffness_factor=args.steel_stiffness_factor,
        steel_mass_scaling=args.steel_mass_scaling,
        output_frequency=args.output_frequency,
        search_frequency=args.search_frequency,
        debug_output=args.debug_output,
        steel_only=args.steel_only,
    )

    print("\n" + "=" * 70)
    print("Generated files:")
    print(f"  Discretization: {split_files['discretization']}")
    print(f"  YAML: {yaml_file}")
    print(f"\nTo run: mpirun -np 8 Peridigm {yaml_file}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
