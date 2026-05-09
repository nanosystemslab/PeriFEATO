#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _read_nodeset(path: Path) -> np.ndarray:
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ids.append(int(line))
    return np.asarray(ids, dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate Peridigm Text File discretization export.")
    ap.add_argument("discretization", type=Path, help="Path to core_shell_refined_peridigm.txt")
    ap.add_argument(
        "--nodeset",
        type=Path,
        action="append",
        default=[],
        help="Optional path to a nodeset_*.txt to validate (repeatable).",
    )
    args = ap.parse_args()

    disc = args.discretization
    if not disc.is_file():
        raise SystemExit(f"Missing discretization file: {disc}")

    data = np.loadtxt(disc)
    if data.ndim != 2 or data.shape[1] != 5:
        raise SystemExit(f"Expected 5 columns (x y z block_id volume), got shape {data.shape}")

    xyz = data[:, 0:3]
    block = data[:, 3].astype(np.int64)
    vol = data[:, 4]

    if np.any(~np.isfinite(xyz)):
        raise SystemExit("Non-finite coordinates found.")
    if np.any(~np.isfinite(vol)):
        raise SystemExit("Non-finite volumes found.")
    if np.any(vol <= 0):
        raise SystemExit(f"Non-positive volumes found (min={vol.min():.3e}).")

    uniq, counts = np.unique(block, return_counts=True)
    bounds_min = xyz.min(axis=0)
    bounds_max = xyz.max(axis=0)

    print(f"File: {disc}")
    print(f"Points: {len(xyz)}")
    print(f"Blocks: {dict(zip([int(x) for x in uniq], [int(c) for c in counts]))}")
    print(f"Bounds min (m): {bounds_min.tolist()}")
    print(f"Bounds max (m): {bounds_max.tolist()}")
    print(f"Total volume (m^3): {float(vol.sum()):.6e}")

    n = len(xyz)
    for ns in args.nodeset:
        if not ns.is_file():
            raise SystemExit(f"Missing nodeset file: {ns}")
        ids = _read_nodeset(ns)
        if len(ids) == 0:
            raise SystemExit(f"Empty nodeset: {ns}")
        if np.any(ids < 1) or np.any(ids > n):
            raise SystemExit(f"Nodeset {ns} contains ids outside [1, {n}]")
        print(f"Nodeset: {ns.name}  n={len(ids)}  min_id={int(ids.min())}  max_id={int(ids.max())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

