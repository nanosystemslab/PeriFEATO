#!/bin/bash
#SBATCH --job-name=perifeato_thick
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7  # 7 parallel FEA orientations
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/perifeato_thick_%j.log
#SBATCH --error=logs/perifeato_thick_%j.err

# ============================================================================
# PeriFEATO shell thickness optimization (FEA-only) - end-to-end test
# ============================================================================
#
# Submit from the project root:
#     cd ~/PeriFEATO && sbatch examples/hpc_templates/run_perifeato_thickness.slurm
#
# Optional overrides (sbatch --export=ALL,VAR=value ...):
#   CONFIG          Path to YAML config (default: examples/configs/perifeato_thickness_test.yaml)
#   OF_ROOT         Existing Optimization_Framework root (default: ~/Optimization_Framework)
#   CONTAINER_PATH  DOLFINx singularity image (default: $OF_ROOT/containers/dolfinx_v0.9.0_pyvista.sif)
# ============================================================================

set -u

PERIFEATO_ROOT="${PERIFEATO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"

if [[ ! -d "${PERIFEATO_ROOT}/perifeato" ]]; then
    echo "ERROR: ${PERIFEATO_ROOT}/perifeato not found." >&2
    echo "  Submit from the project root: cd ~/PeriFEATO && sbatch examples/hpc_templates/run_perifeato_thickness.slurm" >&2
    exit 2
fi

OF_ROOT="${OF_ROOT:-${HOME}/Optimization_Framework}"
CONTAINER_PATH="${CONTAINER_PATH:-${OF_ROOT}/containers/dolfinx_v0.9.0_pyvista.sif}"
CONFIG="${CONFIG:-${PERIFEATO_ROOT}/examples/configs/perifeato_thickness_test.yaml}"

OUTPUT_DIR="${HOME}/koa_scratch/perifeato_thick_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PERIFEATO_ROOT}/logs"

echo "======================================================================"
echo "PeriFEATO Shell Thickness Optimization - Job ${SLURM_JOB_ID}"
echo "======================================================================"
echo "Started:        $(date)"
echo "Node:           $(hostname)"
echo "PERIFEATO_ROOT: ${PERIFEATO_ROOT}"
echo "OF_ROOT:        ${OF_ROOT}"
echo "Container:      ${CONTAINER_PATH}"
echo "Config:         ${CONFIG}"
echo "Output:         ${OUTPUT_DIR}"
echo ""

module purge
module load tools/Singularity/latest 2>/dev/null || \
module load tools/Apptainer/latest 2>/dev/null
CONTAINER_CMD=$(command -v singularity || command -v apptainer)

# Drop stale bytecode
find "${PERIFEATO_ROOT}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Stamp output dir into the config so results land where we expect.
TEMP_CONFIG="${OUTPUT_DIR}/config.yaml"
cp "${CONFIG}" "${TEMP_CONFIG}"
sed -i "s|dir: .*|dir: \"${OUTPUT_DIR}\"|" "${TEMP_CONFIG}"

START=$(date +%s)

${CONTAINER_CMD} exec \
    --bind "${PERIFEATO_ROOT}:${PERIFEATO_ROOT}" \
    --bind "${OF_ROOT}:${OF_ROOT}" \
    --bind "${OUTPUT_DIR}:${OUTPUT_DIR}" \
    --pwd "${PERIFEATO_ROOT}" \
    "${CONTAINER_PATH}" \
    bash -c "
export OF_ROOT=${OF_ROOT}
export PYTHONPATH=${PERIFEATO_ROOT}:\${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
python3 -u ${PERIFEATO_ROOT}/scripts/run_thickness_optimization.py --config ${TEMP_CONFIG}
"
RC=$?
END=$(date +%s)

echo ""
echo "======================================================================"
echo "Exit code: ${RC}    (took $((END - START))s)"
echo "Output:    ${OUTPUT_DIR}"
echo "Finished:  $(date)"
echo "======================================================================"

if [[ ${RC} -eq 0 && -f "${OUTPUT_DIR}/history.json" ]]; then
    echo ""
    echo "Convergence history:"
    python3 -c "
import json
with open('${OUTPUT_DIR}/history.json') as f:
    history = json.load(f)
for h in history:
    stress = h.get('max_stress_MPa', 0)
    print(f\"  iter {h['iteration']}: stress={stress:.2f} MPa, |dt|={h.get('thickness_change_norm', 0):.3f} mm\")
"
fi

exit ${RC}
