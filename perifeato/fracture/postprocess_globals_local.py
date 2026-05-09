#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


def _run_ncdump(args: list[str]) -> str:
    proc = subprocess.run(args, text=True, capture_output=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ncdump failed (exit {proc.returncode}): {msg}")
    return proc.stdout


def _parse_vars_from_header(header: str) -> set[str]:
    vars_section = []
    in_vars = False
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("variables:"):
            in_vars = True
            continue
        if in_vars:
            if line.startswith("//") or line.startswith("data:"):
                break
            vars_section.append(line)
    names = set()
    for line in vars_section:
        parts = line.split()
        if len(parts) < 2:
            continue
        token = parts[1]
        if "(" in token:
            name = token.split("(", 1)[0]
        else:
            name = token
        if name:
            names.add(name)
    return names


def _parse_data(ncdump_text: str) -> dict[str, list[float]]:
    if "data:" not in ncdump_text:
        raise ValueError("ncdump output missing data section")
    data = ncdump_text.split("data:", 1)[1]
    statements: list[str] = []
    buf = ""
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        buf += " " + line
        while ";" in buf:
            head, tail = buf.split(";", 1)
            statements.append(head.strip())
            buf = tail.strip()
    parsed: dict[str, list[float]] = {}
    for stmt in statements:
        if "=" not in stmt:
            continue
        name, values = stmt.split("=", 1)
        name = name.strip()
        vals = []
        for tok in values.replace(",", " ").split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
        if vals:
            parsed[name] = vals
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract Peridigm global history to CSV (local).")
    ap.add_argument("--hist", required=True, type=Path, help="Path to core_shell_from_tets.h")
    ap.add_argument("--out-csv", type=Path, default=None, help="CSV output path")
    ap.add_argument("--ncdump", type=Path, default=None, help="Path to ncdump binary")
    args = ap.parse_args()

    hist = args.hist
    if not hist.is_file():
        raise SystemExit(f"Missing history file: {hist}")

    ncdump_bin: Path | None = None
    if args.ncdump is not None:
        ncdump_bin = args.ncdump
    else:
        for candidate in (Path("/opt/homebrew/bin/ncdump"), Path("ncdump")):
            if candidate.is_absolute():
                if candidate.exists():
                    ncdump_bin = candidate
                    break
            else:
                ncdump_bin = candidate
                break
    if ncdump_bin is None:
        raise SystemExit("ncdump not found. Install netCDF tools locally or pass --ncdump.")

    try:
        header = _run_ncdump([str(ncdump_bin), "-h", str(hist)])
    except FileNotFoundError:
        raise SystemExit("ncdump not found in PATH. Install netCDF tools locally or pass --ncdump.")
    except RuntimeError as e:
        raise SystemExit(str(e))

    names = _parse_vars_from_header(header)
    time_candidates = [
        "time_whole",
        "time",
        "time_step",
        "Time",
        "Time_Step",
    ]
    time_name = None
    for candidate in time_candidates:
        if candidate in names:
            time_name = candidate
            break
    if time_name is None:
        for name in sorted(names):
            if "time" in name.lower():
                time_name = name
                break
    if time_name is None:
        raise SystemExit("Could not find time variable in history file.")

    wanted = ["Global_Kinetic_Energy", "Global_Linear_Momentum", "Global_Angular_Momentum"]
    if "Global_Strain_Energy" in names:
        wanted.insert(1, "Global_Strain_Energy")

    missing = [w for w in wanted if w not in names]
    if missing:
        raise SystemExit(f"Missing expected variables in .h: {', '.join(missing)}")

    dump_vars = [time_name] + wanted
    try:
        data_text = _run_ncdump([str(ncdump_bin), "-v", ",".join(dump_vars), str(hist)])
    except RuntimeError as e:
        raise SystemExit(str(e))
    data = _parse_data(data_text)

    out_csv = args.out_csv or hist.with_suffix(".csv")
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(dump_vars)
        rows = zip(*(data[name] for name in dump_vars))
        writer.writerows(rows)

    print(f"Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
