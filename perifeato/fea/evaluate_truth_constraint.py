#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import yaml


def _get(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _infer_jobid(path: Path) -> str | None:
    # e.g. results_truth_contact_vertical_theta0_base_9926984/vertical_theta0_base_metrics.yaml
    for part in path.parts[::-1]:
        if part.startswith("results_truth_contact_"):
            tail = part.rsplit("_", 1)[-1]
            if tail.isdigit():
                return tail
    return None


def _infer_orientation(doc: dict, fallback_stem: str) -> str:
    ori = doc.get("orientation")
    if isinstance(ori, str) and ori:
        return ori
    if isinstance(ori, dict):
        name = ori.get("name")
        if isinstance(name, str) and name:
            return name
    return fallback_stem


def _extract_vm_candidates_pa(doc: dict) -> dict[str, float]:
    """
    Return a dict of candidate PLA von Mises maxima in Pa from either:
    - *_metrics.yaml (new): {von_mises: {pla_all: {max_pa: ...}, ...}}
    - *_summary.yaml (new): {metrics: {von_mises: ...}}
    - *_summary.yaml (old): {von_mises_summary: "PLA (all): max=... MPa ..."}
    """
    candidates: dict[str, float] = {}

    # New *_metrics.yaml
    for key in ("pla_all", "pla_protected", "pla_fracture"):
        v = _get(doc, f"von_mises.{key}.max_pa")
        if isinstance(v, (int, float)):
            candidates[key] = float(v)

    # New *_summary.yaml (metrics nested)
    for key in ("pla_all", "pla_protected", "pla_fracture"):
        v = _get(doc, f"metrics.von_mises.{key}.max_pa")
        if isinstance(v, (int, float)):
            candidates.setdefault(key, float(v))

    # Older *_summary.yaml: parse formatted string
    if not candidates:
        s = doc.get("von_mises_summary")
        if isinstance(s, str) and s:
            patterns = {
                "pla_all": r"PLA\s*\(all\)\s*:\s*max=([0-9.+-eE]+)\s*MPa",
                "pla_protected": r"PLA\s*protected.*?:\s*max=([0-9.+-eE]+)\s*MPa",
                "pla_fracture": r"PLA\s*fracture.*?:\s*max=([0-9.+-eE]+)\s*MPa",
            }
            for name, pat in patterns.items():
                m = re.search(pat, s)
                if not m:
                    continue
                try:
                    candidates[name] = float(m.group(1)) * 1e6
                except Exception:
                    continue

    return candidates


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate max von Mises constraint across orientations from *_metrics.yaml or *_summary.yaml."
    )
    p.add_argument("root", type=Path, nargs="?", default=Path("results_from_koa"))
    p.add_argument("--limit-mpa", type=float, default=50.0)
    p.add_argument("--out", type=Path, default=Path("truth_constraint_report.yaml"))
    p.add_argument(
        "--jobids",
        type=str,
        default="",
        help="Optional: comma/space-separated Slurm jobids to include (filters by jobid inferred from path).",
    )
    p.add_argument(
        "--orientations",
        nargs="*",
        default=[],
        help="Optional: only evaluate these orientation names; error if any are missing.",
    )
    args = p.parse_args()

    jobid_filter: set[str] = set()
    if args.jobids.strip():
        for tok in re.split(r"[,\s]+", args.jobids.strip()):
            tok = tok.strip()
            if tok:
                jobid_filter.add(tok)

    files = sorted(args.root.rglob("*_metrics.yaml")) + sorted(args.root.rglob("*_summary.yaml"))
    if not files:
        raise SystemExit(f"No *_metrics.yaml or *_summary.yaml found under {args.root}")

    limit_pa = float(args.limit_mpa) * 1e6
    per_orientation = []
    overall_max = -1.0
    overall_argmax = None
    seen: set[str] = set()

    for f in files:
        try:
            doc = yaml.safe_load(f.read_text())
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue

        inferred_jobid = str(doc.get("jobid") or _infer_jobid(f) or "")
        if jobid_filter and inferred_jobid not in jobid_filter:
            continue

        fallback_stem = f.stem
        for suffix in ("_metrics", "_summary"):
            if fallback_stem.endswith(suffix):
                fallback_stem = fallback_stem[: -len(suffix)]
        orientation = _infer_orientation(doc, fallback_stem)
        if args.orientations and orientation not in args.orientations:
            continue
        seen.add(orientation)

        # Prefer per-region PLA maxima; fall back to whatever is present.
        vm = _extract_vm_candidates_pa(doc)
        candidates = list(vm.values())

        if not candidates:
            # Last resort: accept any overall max.
            v = _get(doc, "von_mises.overall.max_pa")
            if isinstance(v, (int, float)):
                candidates.append(float(v))
            v = _get(doc, "metrics.von_mises.overall.max_pa")
            if isinstance(v, (int, float)):
                candidates.append(float(v))

        if not candidates:
            raise SystemExit(f"{f}: could not find von Mises max metrics")

        max_pa = max(candidates)
        ok = bool(max_pa <= limit_pa)

        per_orientation.append(
            {
                "orientation": orientation,
                "jobid": inferred_jobid,
                "max_von_mises_pa": float(max_pa),
                "max_von_mises_mpa": float(max_pa / 1e6),
                "breakdown_pa": {k: float(v) for k, v in sorted(vm.items())} if vm else None,
                "ok": ok,
                "file": str(f),
            }
        )

        if max_pa > overall_max:
            overall_max = max_pa
            overall_argmax = orientation

    if args.orientations:
        missing = [o for o in args.orientations if o not in seen]
        if missing:
            raise SystemExit(f"Missing orientations under {args.root}: {', '.join(missing)}")

    per_orientation.sort(key=lambda r: r["max_von_mises_pa"], reverse=True)
    all_ok = all(bool(r["ok"]) for r in per_orientation)

    report = {
        "root": str(args.root),
        "limit_mpa": float(args.limit_mpa),
        "pass": bool(all_ok),
        "worst_orientation": overall_argmax,
        "worst_max_von_mises_mpa": float(overall_max / 1e6),
        "orientations": per_orientation,
    }

    args.out.write_text(yaml.safe_dump(report, sort_keys=False))
    print(f"Wrote: {args.out}")
    print(f"PASS={report['pass']} worst={report['worst_orientation']} max_von_mises={report['worst_max_von_mises_mpa']:.2f} MPa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
