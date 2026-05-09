#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


def _flatten(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten(key, v, out)
        return
    out[prefix] = obj


def _infer_jobid(path: Path) -> str | None:
    # e.g. results_truth_contact_vertical_theta0_base_9926984/vertical_theta0_base_metrics.yaml
    for part in path.parts[::-1]:
        if part.startswith("results_truth_contact_"):
            tail = part.rsplit("_", 1)[-1]
            if tail.isdigit():
                return tail
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("results_from_koa")],
        help="Root directories to search (default: results_from_koa).",
    )
    p.add_argument("--out", type=Path, default=Path("truth_metrics_summary.csv"))
    args = p.parse_args()

    metrics_files: list[Path] = []
    for root in args.roots:
        if not root.exists():
            continue
        metrics_files.extend(sorted(root.rglob("*_metrics.yaml")))
        metrics_files.extend(sorted(root.rglob("*_summary.yaml")))

    # Prefer *_metrics.yaml when present; fall back to *_summary.yaml.
    by_orientation_job: dict[tuple[str, str | None], Path] = {}
    for f in metrics_files:
        if f.name.endswith("_metrics.yaml"):
            try:
                doc = yaml.safe_load(f.read_text())
            except Exception:
                continue
            orientation = str(doc.get("orientation", f.stem.replace("_metrics", "")))
            jobid = str(doc.get("jobid")) if doc.get("jobid") is not None else _infer_jobid(f)
            by_orientation_job[(orientation, jobid)] = f

    for f in metrics_files:
        if not f.name.endswith("_summary.yaml"):
            continue
        try:
            doc = yaml.safe_load(f.read_text())
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        orientation = str(doc.get("orientation", {}).get("name", f.stem.replace("_summary", "")))
        jobid = _infer_jobid(f)
        key = (orientation, jobid)
        if key not in by_orientation_job:
            by_orientation_job[key] = f

    rows: list[dict[str, Any]] = []
    for (orientation, jobid), f in sorted(by_orientation_job.items(), key=lambda t: (t[0][0], t[0][1] or "", str(t[1]))):
        doc = yaml.safe_load(f.read_text())
        flat: dict[str, Any] = {}
        _flatten("", doc, flat)
        flat["__file"] = str(f)
        flat["__orientation"] = orientation
        flat["__jobid"] = jobid or ""
        rows.append(flat)

    if not rows:
        raise SystemExit("No metrics found (expected *_metrics.yaml or *_summary.yaml under the given roots).")

    # Build a stable-ish column set: required identifiers first, then sorted metrics keys.
    preferred = [
        "__orientation",
        "__jobid",
        # Keys from *_metrics.yaml (new)
        "energy_target_j",
        "energy_work_j",
        "energy_work_ratio",
        "energy_stop_reason",
        "energy_steps_used",
        "drop_mass_kg",
        "drop_height_m",
        "neumann_pressure_mpa",
        "max_displacement_norm_m",
        "max_displacement_abs_component_m",
        "min_uz_m",
        "max_uz_m",
        "top_bc_err_max_m",
        "von_mises.pla_all.max_pa",
        "von_mises.pla_protected.max_pa",
        "von_mises.pla_fracture.max_pa",
        # Keys from *_summary.yaml (older; nested under metrics.*)
        "metrics.displacement.max_norm_m",
        "metrics.displacement.max_abs_component_m",
        "metrics.von_mises.pla_all.max_pa",
        "metrics.von_mises.pla_protected.max_pa",
        "metrics.von_mises.pla_fracture.max_pa",
        "__file",
    ]
    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())
    for k in preferred:
        all_keys.discard(k)
    fieldnames = preferred + sorted(all_keys)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote: {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
