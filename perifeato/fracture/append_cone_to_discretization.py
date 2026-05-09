#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path


def _parse_discretization(path: Path) -> tuple[list[tuple[float, float, float, int, float]], list[float]]:
    points: list[tuple[float, float, float, int, float]] = []
    volumes: list[float] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise SystemExit(f"Invalid discretization line (expected 5 columns): {line[:200]}")
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                blk = int(float(parts[3]))
                vol = float(parts[4])
            except Exception as e:
                raise SystemExit(f"Failed parsing line: {line[:200]} ({e})") from e
            points.append((x, y, z, blk, vol))
            volumes.append(vol)
    if not points:
        raise SystemExit(f"No points found in discretization: {path}")
    return points, volumes


def _grid_points_for_cone(
    *,
    center_x: float,
    center_y: float,
    tip_z: float,
    height: float,
    half_angle_deg: float,
    tip_radius: float,
    spacing: float,
    axis: str,
) -> list[tuple[float, float, float]]:
    if spacing <= 0.0:
        raise SystemExit("Spacing must be > 0.")
    if height <= 0.0:
        raise SystemExit("Cone height must be > 0.")
    if tip_radius < 0.0:
        raise SystemExit("Tip radius must be >= 0.")
    if axis not in {"-z", "+z"}:
        raise SystemExit("Axis must be one of: -z, +z")

    half_angle_rad = math.radians(half_angle_deg)
    tan_half = math.tan(half_angle_rad)

    if axis == "-z":
        z_start = tip_z
        z_dir = -1.0
    else:
        z_start = tip_z
        z_dir = 1.0

    num_layers = int(math.ceil(height / spacing)) + 1
    points: list[tuple[float, float, float]] = []

    for i in range(num_layers):
        z = z_start + z_dir * (i * spacing)
        axial = abs(z_start - z)
        if axial > height + 1e-12:
            continue
        radius = tip_radius + axial * tan_half
        if radius <= 1e-12:
            points.append((center_x, center_y, z))
            continue
        n_r = int(math.ceil(radius / spacing))
        for ix in range(-n_r, n_r + 1):
            x = center_x + ix * spacing
            dx = x - center_x
            if abs(dx) > radius + 1e-12:
                continue
            for iy in range(-n_r, n_r + 1):
                y = center_y + iy * spacing
                dy = y - center_y
                if dx * dx + dy * dy <= radius * radius + 1e-12:
                    points.append((x, y, z))
    return points


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Append a cone point cloud to a Peridigm text discretization and write a cone nodeset."
    )
    ap.add_argument("--discretization", required=True, type=Path, help="Input Peridigm discretization (text).")
    ap.add_argument("--out-discretization", required=True, type=Path, help="Output discretization (text).")
    ap.add_argument("--out-nodeset", required=True, type=Path, help="Output nodeset file for cone points (1-based).")
    ap.add_argument("--cone-block-id", type=int, default=99, help="Block id for cone points.")
    ap.add_argument("--half-angle-deg", type=float, default=30.0, help="Cone half-angle in degrees.")
    ap.add_argument("--tip-radius", type=float, default=0.0, help="Cone tip radius (meters).")
    ap.add_argument("--height", type=float, default=0.01, help="Cone height in meters.")
    ap.add_argument("--spacing", type=float, default=None, help="Point spacing (meters); default from median volume.")
    ap.add_argument("--tip-clearance", type=float, default=None, help="Clearance above shell max-z (meters).")
    ap.add_argument("--tip-z", type=float, default=None, help="Explicit cone tip z; overrides clearance.")
    ap.add_argument("--center-x", type=float, default=None, help="Cone center x; default from shell centroid.")
    ap.add_argument("--center-y", type=float, default=None, help="Cone center y; default from shell centroid.")
    ap.add_argument("--axis", type=str, default="+z", help="Cone axis direction: -z or +z.")
    ap.add_argument("--flyback", action="store_true", help="Flyback mode: position cone at z_min (inside hole) instead of z_max.")
    args = ap.parse_args()

    disc = args.discretization
    if not disc.is_file():
        raise SystemExit(f"Missing discretization: {disc}")

    points, volumes = _parse_discretization(disc)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    cx = args.center_x if args.center_x is not None else statistics.fmean(xs)
    cy = args.center_y if args.center_y is not None else statistics.fmean(ys)
    z_max = max(zs)
    z_min = min(zs)

    vol_med = statistics.median(volumes)
    spacing = args.spacing if args.spacing is not None else vol_med ** (1.0 / 3.0)
    clearance = args.tip_clearance if args.tip_clearance is not None else 0.2 * spacing

    if args.tip_z is not None:
        tip_z = args.tip_z
    elif args.flyback:
        # Flyback mode: cone tip inside the hole at z_min
        tip_z = z_min + clearance
    else:
        # Standard mode: cone tip above shell at z_max
        tip_z = z_max + clearance

    cone_points = _grid_points_for_cone(
        center_x=cx,
        center_y=cy,
        tip_z=tip_z,
        height=args.height,
        half_angle_deg=args.half_angle_deg,
        tip_radius=args.tip_radius,
        spacing=spacing,
        axis=args.axis,
    )

    if not cone_points:
        raise SystemExit("Cone generation produced no points. Check spacing/height/angle.")

    out_disc = args.out_discretization
    out_disc.parent.mkdir(parents=True, exist_ok=True)
    out_nodeset = args.out_nodeset
    out_nodeset.parent.mkdir(parents=True, exist_ok=True)

    # Copy original discretization lines verbatim, then append cone points.
    original_lines = disc.read_text().splitlines()
    with out_disc.open("w") as f:
        for line in original_lines:
            if line.strip():
                f.write(line.rstrip() + "\n")
        for (x, y, z) in cone_points:
            f.write(f"{x:.8e} {y:.8e} {z:.8e} {args.cone_block_id} {vol_med:.8e}\n")

    start_id = len(points) + 1
    with out_nodeset.open("w") as f:
        for i in range(len(cone_points)):
            f.write(f"{start_id + i}\n")

    print(f"Wrote: {out_disc}")
    print(f"Wrote: {out_nodeset}")
    print(f"Shell points: {len(points)}")
    print(f"Cone points: {len(cone_points)}")
    print(f"Spacing: {spacing:.6e} m (median volume {vol_med:.6e})")
    print(
        f"Cone tip z: {tip_z:.6e} m, height: {args.height:.6e} m, half-angle: {args.half_angle_deg:.2f} deg, "
        f"tip radius: {args.tip_radius:.6e} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
