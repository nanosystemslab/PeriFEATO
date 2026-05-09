#!/bin/bash
# Generate a single refined mesh (theta-banded thickness) for a specific case tag.
#
# Required env vars (set via sbatch --export=ALL,...):
#   RUNS_ROOT   Absolute path to where run dirs should be created
#   MANIFEST    Absolute path to a YAML file describing cases
#   TAG         Case tag to run (matches manifest cases[].tag)
#
# Optional env vars:
#   BASE_CONFIG Absolute path to a base config.yaml (default: $HOME/hpc_deployment/config.yaml)
#   CLEAN_RESULTS_REFINED  Set to 1 to delete results_refined before regenerating (default: 0)
#
# Example submit:
#   sbatch --export=ALL,RUNS_ROOT=/home/user/hpc_deployment/mesh_sweeps/sweep_XXXX,MANIFEST=/home/user/.../manifest.yaml,TAG=theta_uniform_mid \
#     /home/user/Optimization_Framework/modules/theta_mesh/hpc/generate_theta_mesh_single.slurm
#

#SBATCH --job-name=theta_mesh_single
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=theta_mesh_single_%j.out
#SBATCH --error=theta_mesh_single_%j.err

set -euo pipefail

if ! type module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
fi
if type module >/dev/null 2>&1; then
  module load tools/Singularity/latest 2>/dev/null || module load tools/Apptainer/latest 2>/dev/null || true
fi

ROOT="${HOME}/hpc_deployment"
PKG="${ROOT}/hpc_package"
CONTAINER="${ROOT}/fem-drop-test_sandbox"

RUNS_ROOT="${RUNS_ROOT:-}"
MANIFEST="${MANIFEST:-}"
TAG="${TAG:-}"
BASE_CONFIG="${BASE_CONFIG:-${PKG}/config.yaml}"
CLEAN_RESULTS_REFINED="${CLEAN_RESULTS_REFINED:-0}"

if [ -z "${RUNS_ROOT}" ]; then
  echo "ERROR: RUNS_ROOT is required (absolute path)" >&2
  exit 2
fi
if [ -z "${MANIFEST}" ]; then
  echo "ERROR: MANIFEST is required (absolute path)" >&2
  exit 2
fi
if [ -z "${TAG}" ]; then
  echo "ERROR: TAG is required (case tag in manifest)" >&2
  exit 2
fi
if [ ! -f "${MANIFEST}" ]; then
  echo "ERROR: MANIFEST not found: ${MANIFEST}" >&2
  exit 2
fi
if [ ! -f "${BASE_CONFIG}" ]; then
  echo "ERROR: BASE_CONFIG not found: ${BASE_CONFIG}" >&2
  exit 2
fi
if [ ! -d "${CONTAINER}" ]; then
  echo "ERROR: sandbox container not found: ${CONTAINER}" >&2
  exit 2
fi
if [ ! -f "${PKG}/shell_geometry_generator_refined.py" ]; then
  echo "ERROR: missing mesh generator: ${PKG}/shell_geometry_generator_refined.py" >&2
  exit 2
fi

RUNTIME=""
if command -v singularity >/dev/null 2>&1; then
  RUNTIME="singularity"
elif command -v apptainer >/dev/null 2>&1; then
  RUNTIME="apptainer"
else
  echo "ERROR: neither 'singularity' nor 'apptainer' found in PATH after module load" >&2
  exit 2
fi

export RUNS_ROOT
export MANIFEST
export BASE_CONFIG
export TAG

mkdir -p "${RUNS_ROOT}"

echo "=== Theta mesh single case ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID:-}"
echo "Runs root: ${RUNS_ROOT}"
echo "Manifest:  ${MANIFEST}"
echo "Base cfg:  ${BASE_CONFIG}"
echo "Tag:       ${TAG}"
echo "Runtime:   ${RUNTIME}"
echo "Clean results_refined: ${CLEAN_RESULTS_REFINED}"

set -x
TAG_OUT="$("${RUNTIME}" exec "${CONTAINER}" python3 - <<'PY'
import os
from pathlib import Path
import yaml

runs_root = Path(os.environ["RUNS_ROOT"]).expanduser().resolve()
manifest_path = Path(os.environ["MANIFEST"]).expanduser().resolve()
base_config_path = Path(os.environ["BASE_CONFIG"]).expanduser().resolve()
tag = os.environ["TAG"]

manifest = yaml.safe_load(manifest_path.read_text())
cases = manifest.get("cases", [])
if not isinstance(cases, list) or not cases:
    raise SystemExit("MANIFEST must contain a non-empty list at key 'cases'")

case = None
for c in cases:
    if str(c.get("tag", "")) == tag:
        case = c
        break

if case is None:
    raise SystemExit(f"Tag '{tag}' not found in manifest")

theta_edges = case.get("theta_band_edges_deg", None)
protected = case.get("protected_thickness_theta_mm", None)
fracture_factor = case.get("fracture_factor_theta", None)
fracture_thickness = case.get("fracture_thickness_theta_mm", None)

if theta_edges is None or protected is None:
    raise SystemExit("Case must define theta_band_edges_deg and protected_thickness_theta_mm")

cfg = yaml.safe_load(base_config_path.read_text())
cfg.setdefault("mesh", {})["theta_band_edges_deg"] = theta_edges
cfg.setdefault("material", {})["protected_thickness_theta_mm"] = protected

# Ensure Peridigm export is enabled.
cfg.setdefault("output", {})["export_peridigm"] = True

cfg.setdefault("fracture_design", {})
fracture_cfg = cfg["fracture_design"]
fracture_enabled = bool(fracture_cfg.get("fracture_planes_phi_deg")) and float(fracture_cfg.get("zone_width_deg", 0.0) or 0.0) > 0.0
disable_fracture = (fracture_factor is None) and (fracture_thickness is None) and not fracture_enabled

if disable_fracture:
    fracture_cfg.pop("fracture_factor_theta", None)
    fracture_cfg.pop("fracture_thickness_theta_mm", None)
else:
    if fracture_thickness is not None:
        fracture_cfg["fracture_thickness_theta_mm"] = fracture_thickness
        fracture_cfg.pop("fracture_factor_theta", None)
    else:
        if fracture_factor is None:
            factor = float(fracture_cfg.get("initial_thickness_factor", 0.6))
            fracture_factor = [factor] * (len(theta_edges) - 1)
        fracture_cfg["fracture_factor_theta"] = fracture_factor
        fracture_cfg.pop("fracture_thickness_theta_mm", None)

run_dir = runs_root / tag
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

print(tag)
PY
)"
set +x

RUN_DIR="${RUNS_ROOT}/${TAG_OUT}"
if [ ! -d "${RUN_DIR}" ]; then
  echo "ERROR: RUN_DIR not created: ${RUN_DIR}" >&2
  exit 2
fi

echo "Run dir:  ${RUN_DIR}"

if [ "${CLEAN_RESULTS_REFINED}" -eq 1 ]; then
  rm -rf "${RUN_DIR}/results_refined"
fi

cd "${RUN_DIR}"

set -x
"${RUNTIME}" exec "${CONTAINER}" python3 "${PKG}/shell_geometry_generator_refined.py"
set +x

ls -la results_refined || true
test -f results_refined/core_shell_refined.msh
test -f results_refined/core_shell_refined.vtu
test -f results_refined/core_shell_refined.xdmf
test -f results_refined/peridigm/core_shell_refined_peridigm.txt
echo "Wrote: ${RUN_DIR}/results_refined/core_shell_refined.{msh,vtu,xdmf}"
