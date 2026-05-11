#!/bin/bash
#SBATCH --job-name=thick_opt
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/thickness_opt_%j.log
#SBATCH --error=logs/thickness_opt_%j.err

# ============================================================================
# Shell Thickness Optimization - HPC Execution
# ============================================================================
#
# Runs FEA-based shell thickness optimization using module framework
#
# Configuration:
# - Target: Safety Factor = 2.0 (22 MPa stress limit)
# - Physics: Transient dynamics (5 orientations, 10 m/s impact)
# - Mesh: Coarse (3mm PLA, proven fast)
# - Expected runtime: ~5 hours (5 iterations)
#
# ============================================================================

set -e
set -u

echo "======================================================================"
echo "Shell Thickness Optimization (Module Framework) - Job ${SLURM_JOB_ID}"
echo "======================================================================"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

OF_ROOT="${HOME}/Optimization_Framework"
CONTAINER_PATH="${OF_ROOT}/containers/dolfinx_v0.9.0_pyvista.sif"
CONFIG_FILE="${OF_ROOT}/modules/optimization/config/thickness_optimization_hpc.yaml"

# Output directory
OUTPUT_DIR="${HOME}/scratch/thickness_opt_$(date +%Y%m%d)_${SLURM_JOB_ID}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OF_ROOT}/logs"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

module purge
module load tools/Singularity/latest 2>/dev/null || \
module load tools/Apptainer/latest 2>/dev/null

CONTAINER_CMD=$(command -v singularity || command -v apptainer)

echo "OF_ROOT: ${OF_ROOT}"
echo "Container: ${CONTAINER_PATH}"
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# ============================================================================
# UPDATE CONFIG WITH OUTPUT DIRECTORY
# ============================================================================

cd "${OF_ROOT}"

# Create temporary config
TEMP_CONFIG="/tmp/thickness_opt_${SLURM_JOB_ID}.yaml"
cp "${CONFIG_FILE}" "${TEMP_CONFIG}"
sed -i "s|dir: .*|dir: \"${OUTPUT_DIR}\"|" "${TEMP_CONFIG}"

# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

echo "======================================================================"
echo "Starting Shell Thickness Optimization"
echo "======================================================================"
echo ""

${CONTAINER_CMD} exec \
    --bind "${OF_ROOT}:${OF_ROOT}" \
    --bind "${OUTPUT_DIR}:${OUTPUT_DIR}" \
    --pwd "${OF_ROOT}" \
    "${CONTAINER_PATH}" \
    bash -c "
export OF_ROOT=${OF_ROOT}
export PYTHONPATH=${OF_ROOT}/modules:\${PYTHONPATH}
export PYTHONUNBUFFERED=1

python3 -u scripts/run_thickness_optimization.py --config ${TEMP_CONFIG}
"

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Optimization Completed"
echo "======================================================================"
echo "Exit code: ${EXIT_CODE}"
echo "Finished at: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo ""

# ============================================================================
# POST-PROCESSING
# ============================================================================

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Optimization succeeded!"

    if [ -f "${OUTPUT_DIR}/history.json" ]; then
        echo ""
        echo "Convergence history:"
        ${CONTAINER_CMD} exec \
            --bind "${OUTPUT_DIR}:${OUTPUT_DIR}" \
            "${CONTAINER_PATH}" \
            python3 -c "
import json
with open('${OUTPUT_DIR}/history.json', 'r') as f:
    history = json.load(f)
for h in history:
    print(f\"  Iteration {h['iteration']}: {h['max_stress_MPa']:.2f} MPa\")
"
    fi

    echo ""
    echo "To retrieve results:"
    echo "  rsync -avz mtdsn@koa.its.hawaii.edu:${OUTPUT_DIR}/ ./results/"
else
    echo "✗ Optimization failed (exit code: ${EXIT_CODE})"
fi

rm -f "${TEMP_CONFIG}"
exit ${EXIT_CODE}
