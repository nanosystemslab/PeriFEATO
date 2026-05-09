#!/bin/bash
# ============================================================================
# Submit optimization methods (FSD, OC, MMA, BO, PGD, COBYLA, FSD-G)
# ============================================================================
# Each method uses the same validated v7 Peridigm physics and spline geometry.
# Only the optimization_method and method-specific hyperparameters differ.
#
# Usage:
#   bash modules/optimization/hpc/submit_all_methods.sh              # all fresh (P95/50mps, default)
#   bash modules/optimization/hpc/submit_all_methods.sh fsd oc       # specific methods
#   bash modules/optimization/hpc/submit_all_methods.sh --continue   # restart all from latest
#   bash modules/optimization/hpc/submit_all_methods.sh --p99 bo pgd # P99 stress stat
#   bash modules/optimization/hpc/submit_all_methods.sh --25mps --p99 bo pgd  # 25 m/s + P99
#   bash modules/optimization/hpc/submit_all_methods.sh --split      # old mode (sbatch child Peridigm jobs)
# ============================================================================

set -u

OF_ROOT="${HOME}/Optimization_Framework"
CONFIG_DIR="${OF_ROOT}/modules/optimization/config"

# Parse flags
CONTINUE_MODE=false
USE_UNIFIED=true  # Default to unified (single-job) mode
USE_P99=false
USE_25MPS=false
USE_KILL=false
EXCLUDE_NODES=""
OUTPUT_PREFIX="opt"
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --continue) CONTINUE_MODE=true; shift ;;
        --unified)  USE_UNIFIED=true; shift ;;
        --split)    USE_UNIFIED=false; shift ;;  # Use old sbatch-child mode
        --p99)      USE_P99=true; shift ;;
        --25mps)    USE_25MPS=true; shift ;;
        --kill)     USE_KILL=true; shift ;;       # Use kill-shared partition (preemptible, faster start)
        --exclude)  EXCLUDE_NODES="$2"; shift 2 ;; # Exclude bad nodes
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Build output prefix and select base config
if [[ "${USE_25MPS}" == "true" && "${USE_P99}" == "true" ]]; then
    OUTPUT_PREFIX="v25p99"
    BASE_CONFIG="${OF_ROOT}/modules/optimization/config/dual_physics_optimization_hpc_25mps_p99.yaml"
elif [[ "${USE_P99}" == "true" ]]; then
    OUTPUT_PREFIX="p99"
    BASE_CONFIG="${OF_ROOT}/modules/optimization/config/dual_physics_optimization_hpc_p99.yaml"
elif [[ "${USE_25MPS}" == "true" ]]; then
    OUTPUT_PREFIX="v25"
    BASE_CONFIG="${OF_ROOT}/modules/optimization/config/dual_physics_optimization_hpc_25mps.yaml"
else
    OUTPUT_PREFIX="opt"
    BASE_CONFIG="${OF_ROOT}/modules/optimization/config/dual_physics_optimization_hpc.yaml"
fi

if [[ "${USE_UNIFIED}" == "true" ]]; then
    SLURM_SCRIPT="${OF_ROOT}/modules/optimization/hpc/run_dual_physics_unified.slurm"
else
    SLURM_SCRIPT="${OF_ROOT}/modules/optimization/hpc/run_dual_physics_optimization.slurm"
fi

# Methods to run (default: all)
if [[ $# -gt 0 ]]; then
    METHODS=("$@")
else
    METHODS=(fsd fsda fsdb fsdg fsdga oc mma bo pgd cobyla)
fi

echo "======================================================================"
echo "Submitting optimization methods: ${METHODS[*]}"
echo "Continue mode: ${CONTINUE_MODE}"
echo "Unified mode: ${USE_UNIFIED} (single-job, no child sbatch)"
echo "P99 mode: ${USE_P99}, 25mps mode: ${USE_25MPS} (output prefix: ${OUTPUT_PREFIX})"
echo "SLURM script: ${SLURM_SCRIPT}"
echo "Base config: ${BASE_CONFIG}"
echo "======================================================================"
echo ""

for METHOD in "${METHODS[@]}"; do
    METHOD_CONFIG="${CONFIG_DIR}/dual_physics_${OUTPUT_PREFIX}_${METHOD}.yaml"

    echo "----------------------------------------------------------------------"
    echo "Method: ${METHOD} (prefix: ${OUTPUT_PREFIX})"
    echo "----------------------------------------------------------------------"

    # Find previous run directory for --continue mode
    CONTINUE_FROM=""
    if [[ "${CONTINUE_MODE}" == "true" ]]; then
        # Find most recent scratch dir for this method
        LATEST_DIR=$(ls -td ${HOME}/koa_scratch/${OUTPUT_PREFIX}_${METHOD}_* 2>/dev/null | head -1)
        if [[ -n "${LATEST_DIR}" && -f "${LATEST_DIR}/optimization_state.json" ]]; then
            CONTINUE_FROM="${LATEST_DIR}"
            PREV_ITER=$(python3 -c "import json; print(json.load(open('${CONTINUE_FROM}/optimization_state.json'))['iteration'])")
            echo "  Continuing from: ${CONTINUE_FROM}"
            echo "  Previous iteration: ${PREV_ITER}"
        else
            echo "  No previous run found — starting fresh"
        fi
    fi

    # Generate per-method config from base
    python3 -c "
import yaml

with open('${BASE_CONFIG}') as f:
    cfg = yaml.safe_load(f)

# Update version/name
cfg['version'] = 'v7-${METHOD}'
cfg['name'] = 'dual-physics-${METHOD}'

# Set optimization method
cfg['optimization']['optimization_method'] = '${METHOD}'

# Method-specific hyperparameters
method = '${METHOD}'
opt = cfg['optimization']

if method == 'fsd':
    # FSD base: momentum on, secant off
    opt['use_momentum'] = True
    opt['use_secant_method'] = False

elif method == 'fsda':
    # FSD + secant acceleration (with momentum)
    opt['optimization_method'] = 'fsd'
    opt['use_momentum'] = True
    opt['use_secant_method'] = True

elif method == 'fsdb':
    # Secant-only (no momentum)
    opt['optimization_method'] = 'fsd'
    opt['use_momentum'] = False
    opt['use_secant_method'] = True

elif method == 'fsdg':
    # FSD-G: Global FSD with stress diffusion + update smoothing
    opt['optimization_method'] = 'fsd'
    opt['use_momentum'] = True
    opt['use_secant_method'] = False
    opt['fsd_stress_diffusion'] = 0.4
    opt['fsd_update_smoothing'] = 0.3

elif method == 'fsdga':
    # FSD-GA: FSD-G + secant acceleration (best of both)
    opt['optimization_method'] = 'fsd'
    opt['use_momentum'] = True
    opt['use_secant_method'] = True
    opt['fsd_stress_diffusion'] = 0.4
    opt['fsd_update_smoothing'] = 0.3

elif method == 'oc':
    opt['oc_move_limit'] = 0.3       # Max fractional change per iteration
    opt['oc_move_decay'] = 0.95      # Move limit shrinks by 5% per iteration
    opt['oc_move_floor'] = 0.05      # Minimum move limit

elif method == 'mma':
    opt['mma_asymptote_init'] = 0.2  # Initial asymptote distance (0.5 caused thickness explosion)
    opt['mma_oscillation_shrink'] = 0.7
    opt['mma_monotone_expand'] = 1.2

elif method == 'bo':
    opt['bo_length_scale'] = 2.0     # Smoother GP (was 1.0 — overfitting with few obs in 10D)
    opt['bo_noise'] = 1.0e-3         # More noise tolerance (was 1e-4 — too tight)
    opt['bo_exploration_xi'] = 0.05  # More exploration (was 0.01 — too exploitative early on)
    opt['bo_n_random_init'] = 8      # More initial samples (was 3 — need ~n_bands for 10D)

elif method == 'pgd':
    opt['pgd_initial_step'] = 1.0    # Initial step size
    opt['pgd_backtrack_factor'] = 0.5
    opt['pgd_min_step'] = 0.01

elif method == 'cobyla':
    opt['cobyla_rhobeg'] = 2.0       # Initial trust-region radius (mm) — large enough to reach feasible region
    opt['cobyla_rhoend'] = 0.01      # Convergence tolerance
    opt['cobyla_maxiter'] = 2000     # Max inner iterations per outer step

with open('${METHOD_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f'  Config written: ${METHOD_CONFIG}')
"

    # Build export string
    EXPORT_VARS="ALL,CONFIG_FILE=${METHOD_CONFIG},OUTPUT_BASE=${HOME}/koa_scratch/${OUTPUT_PREFIX}_${METHOD}"
    if [[ -n "${CONTINUE_FROM}" ]]; then
        EXPORT_VARS="${EXPORT_VARS},CONTINUE_FROM=${CONTINUE_FROM}"
    fi

    # Build sbatch extra args
    SBATCH_EXTRA=""
    if [[ "${USE_KILL}" == "true" ]]; then
        SBATCH_EXTRA="${SBATCH_EXTRA} --partition=kill-shared"
    fi
    if [[ -n "${EXCLUDE_NODES}" ]]; then
        SBATCH_EXTRA="${SBATCH_EXTRA} --exclude=${EXCLUDE_NODES}"
    fi

    # Submit with method-specific job name
    JOB_ID=$(sbatch --parsable \
        --job-name="${OUTPUT_PREFIX}_${METHOD}" \
        --export="${EXPORT_VARS}" \
        ${SBATCH_EXTRA} \
        "${SLURM_SCRIPT}")

    echo "  Submitted: job ${JOB_ID}"
    echo ""
done

echo "======================================================================"
echo "All methods submitted. Monitor with:"
echo "  squeue -u \${USER}"
echo ""
echo "Pull results with:"
for METHOD in "${METHODS[@]}"; do
    echo "  rsync -avz mtdsn@koa.its.hawaii.edu:~/koa_scratch/${OUTPUT_PREFIX}_${METHOD}_*/ ./results_${OUTPUT_PREFIX}_${METHOD}/"
done
echo "======================================================================"
