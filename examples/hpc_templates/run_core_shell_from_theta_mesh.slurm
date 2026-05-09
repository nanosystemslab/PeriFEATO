#!/bin/bash
# Run Peridigm on a discretization exported by the theta_mesh generator (Option A).
#
# Required env vars (set via sbatch --export=ALL,...):
#   THETA_RUN_DIR  Absolute path to a mesh run dir that contains:
#                  results_refined/peridigm/core_shell_refined_peridigm.txt
#                  results_refined/peridigm/nodeset_fixed_end.txt
#                  results_refined/peridigm/nodeset_impact_end.txt
#
# Optional env vars:
#   IMPACT_VZ      Initial velocity applied at impact end (m/s, default: -5.0)
#   FINAL_TIME     Final time in seconds (default: 2e-5)
#   PLA_CRIT_STRETCH    Critical stretch for PLA damage (default: 0.01, set <=0 to disable)
#   FRACTURE_CRIT_STRETCH Critical stretch for fracture block (default: 0.005)
#   CONE_ENABLE    Set to 1 to append a cone impactor + contact (default: 0)
#   CONE_HALF_ANGLE_DEG  Cone half-angle (deg, default: 30)
#   CONE_HEIGHT    Cone height (m, default: 0.01)
#   CONE_BLOCK_ID  Cone block id in discretization (default: 99)
#   CONE_AXIS      Cone axis direction: +z or -z (default: +z)
#   CONE_TIP_RADIUS Cone tip radius (m, default: 0.0)
#   CONE_TIP_CLEARANCE Tip clearance above shell max-z (m, default: 0.2*spacing)
#   CONE_IMPACT_VZ Initial velocity applied to cone (m/s, default: IMPACT_VZ)
#   CONE_HORIZON   Cone horizon (m, default: 0.004)
#   CONE_DENSITY   Cone density (kg/m^3, default: 7800)
#   CONE_BULK_MODULUS  Cone bulk modulus (Pa, default: 160e9)
#   CONE_SHEAR_MODULUS Cone shear modulus (Pa, default: 80e9)
#   CONTACT_SEARCH_RADIUS  Contact search radius (m, default: 0.006)
#   CONTACT_SEARCH_FREQUENCY Contact search frequency (steps, default: 50)
#   CONTACT_RADIUS Contact radius (m, default: 0.003)
#   CONTACT_SPRING_CONSTANT Contact spring constant (N/m, default: 1.0e10)
#   SHELL_SLIDE_Z  Set to 1 to allow shell to slide along +/‑Z (fix X/Y only, apply impact to all nodes)
#   CONE_FIXED     Set to 1 to fix cone in place (no initial velocity)
#   FLYBACK_MODE   Set to 1 for flyback simulation (cone at z_min, moving +z into shell)
#   STATE_BASED    Set to 1 to use state-based material model with accurate Poisson ratio (ν=0.36)
#   PLA_YOUNGS_MOD PLA Young's modulus (Pa, default: 2.5e9)
#   PLA_POISSON    PLA Poisson's ratio (default: 0.36)
#   OUTDIR         Output directory (default: <THETA_RUN_DIR>/results_refined/peridigm/out_core_shell_<jobid>)
#
# Example:
#   sbatch --partition shared --export=ALL,THETA_RUN_DIR=$HOME/hpc_deployment/mesh_sweeps/sweep_*/theta_uniform_thick \
#     hpc/run_core_shell_from_theta_mesh.slurm

#SBATCH --job-name=peridigm_core_shell
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00  # Increased from 20min - flyback needs ~50min
#SBATCH --output=output_logs/peridigm_core_shell_%j.out
#SBATCH --error=output_logs/peridigm_core_shell_%j.err

set -euo pipefail

# Allow override of MPI ranks; defaults to Slurm ntasks.
PERIDIGM_NTASKS="${PERIDIGM_NTASKS:-${SLURM_NTASKS:-1}}"

# Slurm copies the script to /var/spool/... at runtime, so relative-to-script paths break.
# Resolve module root from env or common install locations.
ROOT="${PERIDIGM_ROOT:-${OF_ROOT:-}}"
if [[ -z "${ROOT}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/tools/peridigm_env.sh" ]]; then
    ROOT="${SLURM_SUBMIT_DIR}"
  elif [[ -f "${HOME}/Optimization_Framework/modules/peridigm_fracture/tools/peridigm_env.sh" ]]; then
    ROOT="${HOME}/Optimization_Framework/modules/peridigm_fracture"
  else
    echo "ERROR: cannot locate peridigm_fracture root; set PERIDIGM_ROOT" >&2
    exit 2
  fi
fi
ROOT="$(cd "${ROOT}" && pwd -P)"

mkdir -p "${ROOT}/output_logs"

THETA_RUN_DIR="${THETA_RUN_DIR:-}"
if [[ -z "${THETA_RUN_DIR}" ]]; then
  echo "ERROR: THETA_RUN_DIR is required" >&2
  exit 2
fi
THETA_RUN_DIR="$(cd "${THETA_RUN_DIR}" && pwd -P)"

DISC="${THETA_RUN_DIR}/results_refined/peridigm/core_shell_refined_peridigm.txt"
NS_FIXED="${THETA_RUN_DIR}/results_refined/peridigm/nodeset_fixed_end.txt"
NS_IMPACT="${THETA_RUN_DIR}/results_refined/peridigm/nodeset_impact_end.txt"

if [[ ! -f "${DISC}" ]]; then
  echo "ERROR: missing discretization: ${DISC}" >&2
  exit 2
fi
if [[ ! -f "${NS_FIXED}" ]]; then
  echo "ERROR: missing nodeset_fixed_end: ${NS_FIXED}" >&2
  exit 2
fi
if [[ ! -f "${NS_IMPACT}" ]]; then
  echo "ERROR: missing nodeset_impact_end: ${NS_IMPACT}" >&2
  exit 2
fi

IMPACT_VZ="${IMPACT_VZ:--5.0}"
FINAL_TIME="${FINAL_TIME:-2e-5}"
PLA_CRIT_STRETCH="${PLA_CRIT_STRETCH:-0.01}"
FRACTURE_CRIT_STRETCH="${FRACTURE_CRIT_STRETCH:-0.005}"
CONE_ENABLE="${CONE_ENABLE:-0}"
CONE_HALF_ANGLE_DEG="${CONE_HALF_ANGLE_DEG:-30}"
CONE_HEIGHT="${CONE_HEIGHT:-0.01}"
CONE_BLOCK_ID="${CONE_BLOCK_ID:-99}"
CONE_AXIS="${CONE_AXIS:-+z}"
CONE_TIP_RADIUS="${CONE_TIP_RADIUS:-0.0}"
CONE_TIP_CLEARANCE="${CONE_TIP_CLEARANCE:-}"
CONE_IMPACT_VZ="${CONE_IMPACT_VZ:-}"
CONE_HORIZON="${CONE_HORIZON:-0.004}"
CONE_DENSITY="${CONE_DENSITY:-7800}"
CONE_BULK_MODULUS="${CONE_BULK_MODULUS:-160e9}"
CONE_SHEAR_MODULUS="${CONE_SHEAR_MODULUS:-80e9}"
CONTACT_SEARCH_RADIUS="${CONTACT_SEARCH_RADIUS:-0.006}"
CONTACT_SEARCH_FREQUENCY="${CONTACT_SEARCH_FREQUENCY:-50}"
CONTACT_RADIUS="${CONTACT_RADIUS:-0.003}"
CONTACT_SPRING_CONSTANT="${CONTACT_SPRING_CONSTANT:-1.0e10}"
SHELL_SLIDE_Z="${SHELL_SLIDE_Z:-0}"
CONE_FIXED="${CONE_FIXED:-0}"
FLYBACK_MODE="${FLYBACK_MODE:-0}"
STATE_BASED="${STATE_BASED:-0}"
PLA_YOUNGS_MOD="${PLA_YOUNGS_MOD:-2.5e9}"
PLA_POISSON="${PLA_POISSON:-0.36}"

OUTDIR="${OUTDIR:-${THETA_RUN_DIR}/results_refined/peridigm/out_core_shell_${SLURM_JOB_ID:-job}}"
mkdir -p "${OUTDIR}"

echo "=== Peridigm core_shell (from tetra mesh export) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID:-}"
echo "Tasks: ${PERIDIGM_NTASKS}"
echo "ROOT: ${ROOT}"
echo "Run:  ${THETA_RUN_DIR}"
echo "Disc: ${DISC}"
echo "Out:  ${OUTDIR}"
echo "IMPACT_VZ: ${IMPACT_VZ}"
echo "FINAL_TIME: ${FINAL_TIME}"
echo "PLA_CRIT_STRETCH: ${PLA_CRIT_STRETCH}"
echo "FRACTURE_CRIT_STRETCH: ${FRACTURE_CRIT_STRETCH}"
echo "CONE_ENABLE: ${CONE_ENABLE}"
echo "SHELL_SLIDE_Z: ${SHELL_SLIDE_Z}"
echo "CONE_FIXED: ${CONE_FIXED}"
echo "FLYBACK_MODE: ${FLYBACK_MODE}"
echo "STATE_BASED: ${STATE_BASED}"
if [[ "${STATE_BASED}" -eq 1 ]]; then
  echo "  PLA_YOUNGS_MOD: ${PLA_YOUNGS_MOD}"
  echo "  PLA_POISSON: ${PLA_POISSON}"
fi

source "${ROOT}/tools/peridigm_env.sh"

NS_CONE=""
DISC_BASE="${DISC}"
NS_ALL="${THETA_RUN_DIR}/results_refined/peridigm/nodeset_all.txt"
if [[ ! -f "${NS_ALL}" ]]; then
  NS_ALL=""
fi
if [[ "${CONE_ENABLE}" -eq 1 ]]; then
  CONE_DISC="${OUTDIR}/core_shell_refined_peridigm_with_cone.txt"
  NS_CONE="${OUTDIR}/nodeset_cone.txt"
  CONE_ARGS=(
    --discretization "${DISC_BASE}"
    --out-discretization "${CONE_DISC}"
    --out-nodeset "${NS_CONE}"
    --half-angle-deg "${CONE_HALF_ANGLE_DEG}"
    --tip-radius "${CONE_TIP_RADIUS}"
    --height "${CONE_HEIGHT}"
    --cone-block-id "${CONE_BLOCK_ID}"
    --axis="${CONE_AXIS}"
  )
  if [[ -n "${CONE_TIP_CLEARANCE}" ]]; then
    CONE_ARGS+=(--tip-clearance "${CONE_TIP_CLEARANCE}")
  fi
  if [[ "${FLYBACK_MODE}" -eq 1 ]]; then
    CONE_ARGS+=(--flyback)
  fi
  python3 "${ROOT}/tools/append_cone_to_discretization.py" "${CONE_ARGS[@]}"
  DISC="${CONE_DISC}"
fi

YAML_PATH="${OUTDIR}/core_shell_from_tets.yaml"
YAML_ARGS=(
  --discretization "${DISC}"
  --nodeset-fixed "${NS_FIXED}"
  --nodeset-impact "${NS_IMPACT}"
  --impact-vz "${IMPACT_VZ}"
  --final-time "${FINAL_TIME}"
  --pla-critical-stretch "${PLA_CRIT_STRETCH}"
  --fracture-critical-stretch "${FRACTURE_CRIT_STRETCH}"
  --out-yaml "${YAML_PATH}"
  --out-prefix "${OUTDIR}/core_shell_from_tets"
)
if [[ -n "${NS_ALL}" ]]; then
  YAML_ARGS+=(--nodeset-all "${NS_ALL}")
fi
if [[ "${SHELL_SLIDE_Z}" -eq 1 ]]; then
  YAML_ARGS+=(--fixed-end-slide)
  YAML_ARGS+=(--impact-on-all)
  YAML_ARGS+=(--shell-impact)
fi
if [[ "${CONE_ENABLE}" -eq 1 ]]; then
  YAML_ARGS+=(--nodeset-cone "${NS_CONE}")
  YAML_ARGS+=(--cone-block-id "${CONE_BLOCK_ID}")
  YAML_ARGS+=(--cone-horizon "${CONE_HORIZON}")
  YAML_ARGS+=(--cone-density "${CONE_DENSITY}")
  YAML_ARGS+=(--cone-bulk-modulus "${CONE_BULK_MODULUS}")
  YAML_ARGS+=(--cone-shear-modulus "${CONE_SHEAR_MODULUS}")
  YAML_ARGS+=(--contact-search-radius "${CONTACT_SEARCH_RADIUS}")
  YAML_ARGS+=(--contact-search-frequency "${CONTACT_SEARCH_FREQUENCY}")
  YAML_ARGS+=(--contact-radius "${CONTACT_RADIUS}")
  YAML_ARGS+=(--contact-spring-constant "${CONTACT_SPRING_CONSTANT}")
  if [[ -n "${CONE_IMPACT_VZ}" ]]; then
    YAML_ARGS+=(--cone-impact-vz "${CONE_IMPACT_VZ}")
  fi
  if [[ "${CONE_FIXED}" -eq 1 ]]; then
    YAML_ARGS+=(--cone-fixed)
  fi
fi
if [[ "${STATE_BASED}" -eq 1 ]]; then
  YAML_ARGS+=(--state-based)
  YAML_ARGS+=(--pla-youngs-modulus "${PLA_YOUNGS_MOD}")
  YAML_ARGS+=(--pla-poisson-ratio "${PLA_POISSON}")
fi
python3 "${ROOT}/tools/generate_core_shell_from_tets_yaml.py" \
  "${YAML_ARGS[@]}"

cd "${OUTDIR}"
mpirun -np "${PERIDIGM_NTASKS}" --oversubscribe Peridigm "${YAML_PATH}"

ls -lah "${OUTDIR}"/*.e 2>/dev/null || true
echo "Done: ${OUTDIR}"
