#!/bin/bash
#SBATCH --job-name=perifeato_smoke
#SBATCH --partition=kill-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/perifeato_smoke_%j.log
#SBATCH --error=logs/perifeato_smoke_%j.err

# ============================================================================
# PeriFEATO smoke test - validates package loads and 7 optimizers dispatch
# ============================================================================
#
# Submit from the project root so SLURM's relative log paths resolve to
# <project>/logs/:
#
#     cd ~/PeriFEATO && sbatch examples/hpc_templates/smoke_test.slurm
#
# Runs scripts/smoke_test.py inside the DOLFINx container. No Peridigm
# submissions, no real FEA - just verifies the import chain and update
# dispatch on synthetic data.
# ============================================================================

set -u

# PERIFEATO_ROOT is the directory you ran `sbatch` from. SLURM copies this
# script to /var/spool/slurm/scripts/ before running it, so BASH_SOURCE
# can't be used to find the original location. SLURM_SUBMIT_DIR is the cwd
# at submission time, which (per the usage note above) is the project root.
PERIFEATO_ROOT="${PERIFEATO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"

if [[ ! -d "${PERIFEATO_ROOT}/perifeato" ]]; then
    echo "ERROR: ${PERIFEATO_ROOT}/perifeato not found." >&2
    echo "  Submit from the project root: cd ~/PeriFEATO && sbatch examples/hpc_templates/smoke_test.slurm" >&2
    echo "  Or set: PERIFEATO_ROOT=/path/to/PeriFEATO sbatch ..." >&2
    exit 2
fi

CONTAINER_PATH="${CONTAINER_PATH:-${HOME}/Optimization_Framework/containers/dolfinx_v0.9.0_pyvista.sif}"

mkdir -p "${PERIFEATO_ROOT}/logs"

echo "======================================================================"
echo "PeriFEATO smoke test - Job ${SLURM_JOB_ID}"
echo "======================================================================"
echo "Started at: $(date)"
echo "Node:       $(hostname)"
echo "Root:       ${PERIFEATO_ROOT}"
echo "Container:  ${CONTAINER_PATH}"
echo ""

module purge
module load tools/Singularity/latest 2>/dev/null || \
module load tools/Apptainer/latest 2>/dev/null

CONTAINER_CMD=$(command -v singularity || command -v apptainer)

# Drop stale bytecode from any previous syncs
find "${PERIFEATO_ROOT}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "${PERIFEATO_ROOT}" -name "*.pyc" -delete 2>/dev/null || true

START=$(date +%s)

${CONTAINER_CMD} exec \
    --bind "${PERIFEATO_ROOT}:${PERIFEATO_ROOT}" \
    --pwd "${PERIFEATO_ROOT}" \
    "${CONTAINER_PATH}" \
    bash -c "
export PYTHONPATH=${PERIFEATO_ROOT}:\${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
python3 -u ${PERIFEATO_ROOT}/scripts/smoke_test.py
"
RC=$?
END=$(date +%s)

echo ""
echo "======================================================================"
echo "Smoke test exit code: ${RC}    (took $((END - START))s)"
echo "Finished at: $(date)"
echo "======================================================================"

exit ${RC}
