#!/bin/bash
#SBATCH --job-name=dual_phys_opt
#SBATCH --partition=kill-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7  # 7 parallel FEA orientations
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dual_physics_opt_%j.log
#SBATCH --error=logs/dual_physics_opt_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=mtdsn@hawaii.edu

# ============================================================================
# Dual-Physics Shell Thickness Optimization - HPC Execution (PARALLEL)
# ============================================================================
#
# Runs FEA and Peridigm IN PARALLEL for maximum efficiency.
#
# Workflow per iteration:
# 1. Generate mesh (inside container) - creates Peridigm discretization
# 2. Submit Peridigm job immediately (native SLURM, runs in background)
# 3. Run FEA (inside container) - runs in parallel with Peridigm
# 4. Wait for Peridigm if still running
# 5. Process combined results and update thickness
# 6. Check convergence and repeat
#
# ============================================================================

set -u
# No set -e: container crashes (Gmsh segfault, malloc) are caught per-phase

echo "======================================================================"
echo "Dual-Physics Shell Thickness Optimization (PARALLEL) - Job ${SLURM_JOB_ID}"
echo "======================================================================"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

OF_ROOT="${HOME}/Optimization_Framework"
CONTAINER_PATH="${OF_ROOT}/containers/dolfinx_v0.9.0_pyvista.sif"
CONFIG_FILE="${CONFIG_FILE:-${OF_ROOT}/modules/optimization/config/dual_physics_optimization_hpc.yaml}"
PERIDIGM_MODULE="${OF_ROOT}/modules/peridigm_fracture"

# Output directory
# OUTPUT_BASE allows overriding output location (e.g., for sensitivity studies)
if [[ -n "${OUTPUT_BASE:-}" ]]; then
    OUTPUT_DIR="${OUTPUT_BASE}_${SLURM_JOB_ID}"
else
    OUTPUT_DIR="${HOME}/koa_scratch/dual_physics_opt_${SLURM_JOB_ID}"
fi

# Continue from previous run (optional)
# Set via: sbatch --export=ALL,CONTINUE_FROM=/path/to/previous/run ...
CONTINUE_FROM="${CONTINUE_FROM:-}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/peridigm"
mkdir -p "${OF_ROOT}/logs"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

module purge
module load tools/Singularity/latest 2>/dev/null || \
module load tools/Apptainer/latest 2>/dev/null

CONTAINER_CMD=$(command -v singularity || command -v apptainer)

# Prevent stale .pyc bytecode cache (rsync timestamps can trick Python into
# using old cached bytecode instead of updated .py source files)
export PYTHONDONTWRITEBYTECODE=1
find "${OF_ROOT}/modules" -name "*.pyc" -delete 2>/dev/null || true
find "${OF_ROOT}/scripts" -name "*.pyc" -delete 2>/dev/null || true

echo "OF_ROOT: ${OF_ROOT}"
echo "Container: ${CONTAINER_PATH}"
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "Peridigm Module: ${PERIDIGM_MODULE}"
echo ""

# ============================================================================
# CODE VERSION & CONFIG MANIFEST
# ============================================================================
# Creates a manifest file with:
# 1. Git commit hash and status
# 2. MD5 hash of key source files (to detect unsynced changes)
# 3. Summary of enabled optimization features
# This helps debug when HPC code is out of sync with local development
# ============================================================================

echo "======================================================================"
echo "Creating Code & Config Manifest"
echo "======================================================================"

MANIFEST_FILE="${OUTPUT_DIR}/run_manifest.yaml"

# Get git info
GIT_COMMIT=$(cd "${OF_ROOT}" && git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(cd "${OF_ROOT}" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(cd "${OF_ROOT}" && git status --porcelain 2>/dev/null | wc -l | tr -d ' ')

# Compute hash of key source files (to detect sync issues)
KEY_FILES=(
    "modules/optimization/src/shell_thickness/thickness_updater.py"
    "modules/optimization/src/shell_thickness/dual_physics_loop.py"
    "modules/optimization/src/shell_thickness/theta_band_params.py"
    "modules/theta_mesh/src/shell_geometry_generator_refined.py"
    "scripts/run_dual_physics_fea_only.py"
    "scripts/run_dual_physics_mesh_only.py"
)

CODE_HASH=""
for f in "${KEY_FILES[@]}"; do
    if [[ -f "${OF_ROOT}/${f}" ]]; then
        FILE_HASH=$(md5sum "${OF_ROOT}/${f}" 2>/dev/null | cut -d' ' -f1 || echo "error")
        CODE_HASH="${CODE_HASH}${FILE_HASH}"
    fi
done
COMBINED_HASH=$(echo -n "${CODE_HASH}" | md5sum | cut -d' ' -f1)

echo "Git commit: ${GIT_COMMIT}"
echo "Git branch: ${GIT_BRANCH}"
echo "Uncommitted changes: ${GIT_DIRTY}"
echo "Code hash: ${COMBINED_HASH}"

# Write manifest header
cat > "${MANIFEST_FILE}" << EOF
# ============================================================================
# Optimization Run Manifest
# Generated at: $(date -Iseconds)
# Job ID: ${SLURM_JOB_ID}
# ============================================================================

version:
  git_commit: "${GIT_COMMIT}"
  git_branch: "${GIT_BRANCH}"
  uncommitted_changes: ${GIT_DIRTY}
  code_hash: "${COMBINED_HASH}"

  # Individual file hashes (for debugging sync issues)
  file_hashes:
EOF

# Add individual file hashes
for f in "${KEY_FILES[@]}"; do
    if [[ -f "${OF_ROOT}/${f}" ]]; then
        FILE_HASH=$(md5sum "${OF_ROOT}/${f}" 2>/dev/null | cut -d' ' -f1 || echo "error")
        echo "    ${f}: \"${FILE_HASH}\"" >> "${MANIFEST_FILE}"
    fi
done

# Extract and display key config settings
echo ""
echo "Optimization Configuration:"

python3 << PYEOF >> "${MANIFEST_FILE}"
import yaml
import sys

with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)

opt = config.get('optimization', {})
fea = config.get('fea', {})
mat = config.get('material', {})

# Key optimization settings
settings = {
    # Core FSD parameters
    'target_safety_factor': opt.get('target_safety_factor', 1.0),
    'thickness_relaxation': opt.get('thickness_relaxation', 0.5),
    'stress_thickness_exponent': opt.get('stress_thickness_exponent', 0.5),
    'thickness_resolution_mm': opt.get('thickness_resolution_mm', 0.05),

    # Acceleration features
    'use_secant_method': opt.get('use_secant_method', False),
    'use_adaptive_relaxation': opt.get('use_adaptive_relaxation', False),
    'use_fracture_bisection': opt.get('use_fracture_bisection', False),
    'use_momentum': opt.get('use_momentum', False),

    # Stress aggregation
    'stress_band_stat': opt.get('stress_band_stat', 'max'),

    # Constraints
    'min_thickness_mm': mat.get('min_thickness_mm', opt.get('min_thickness_mm', 1.0)),
    'max_thickness_mm': mat.get('max_thickness_mm', opt.get('max_thickness_mm', 10.0)),
    'max_thickness_gradient_mm': opt.get('max_thickness_gradient_mm', 0.0),
    'initial_thickness_mm': opt.get('initial_thickness_mm', 1.5),

    # FEA settings
    'fea_mode': fea.get('mode', 'unknown'),
    'fea_orientations': len(fea.get('orientations', [])),

    # Material
    'yield_stress_mpa': mat.get('yield_stress_Pa', 44e6) / 1e6,
}

# Print to stdout for console
print("  target_safety_factor:", settings['target_safety_factor'])
print("  thickness_relaxation:", settings['thickness_relaxation'])
print("  stress_band_stat:", settings['stress_band_stat'])
print("  use_secant_method:", settings['use_secant_method'])
print("  use_adaptive_relaxation:", settings['use_adaptive_relaxation'])
print("  use_momentum:", settings['use_momentum'])
print("  max_thickness_gradient_mm:", settings['max_thickness_gradient_mm'])

# Write to manifest
print("\nconfig:")
for key, val in settings.items():
    if isinstance(val, bool):
        print(f"  {key}: {'true' if val else 'false'}")
    elif isinstance(val, str):
        print(f"  {key}: \"{val}\"")
    else:
        print(f"  {key}: {val}")
PYEOF

echo ""
echo "Manifest saved to: ${MANIFEST_FILE}"
echo ""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

run_in_container() {
    local script="$1"
    # Subshell (...) contains Singularity signal propagation (SIGSEGV from Gmsh etc.)
    (
        ${CONTAINER_CMD} exec \
            --bind "${OF_ROOT}:${OF_ROOT}" \
            --bind "${OUTPUT_DIR}:${OUTPUT_DIR}" \
            --bind "${HOME}/scratch:${HOME}/scratch" \
            --bind "${HOME}/koa_scratch:${HOME}/koa_scratch" \
            --pwd "${OF_ROOT}" \
            "${CONTAINER_PATH}" \
            bash -c "
export OF_ROOT=${OF_ROOT}
export OUTPUT_DIR=${OUTPUT_DIR}
export PYTHONPATH=${OF_ROOT}/modules:\${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
${script}
"
    )
}

wait_for_job() {
    local job_id="$1"
    local timeout="${2:-3600}"  # 1 hour default for RUNNING time (flyback sims need ~50min)
    local check_interval=15
    local running_start_time=""
    local last_status=""

    echo "  Waiting for Peridigm job ${job_id} (timeout: ${timeout}s after RUNNING starts)..."

    while true; do
        local status=$(squeue -j "${job_id}" -h -o "%T" 2>/dev/null || echo "")

        # Job finished (no longer in queue)
        if [[ -z "${status}" ]]; then
            local final_state=$(sacct -j "${job_id}" -n -o State | head -1 | awk '{print $1}')

            if [[ "${final_state}" == "COMPLETED" ]]; then
                echo "  Peridigm job ${job_id} completed successfully"
                return 0
            else
                echo "  WARNING: Peridigm job ${job_id} ended with state: ${final_state}"
                return 1
            fi
        fi

        # Log status changes
        if [[ "${status}" != "${last_status}" ]]; then
            echo "  Job ${job_id} status: ${status}"
            last_status="${status}"
        fi

        # Track when job starts RUNNING (timeout only applies to running time)
        if [[ "${status}" == "RUNNING" ]]; then
            if [[ -z "${running_start_time}" ]]; then
                running_start_time=$(date +%s)
                echo "  Job started RUNNING at $(date)"
            fi

            # Only timeout on RUNNING jobs (not PENDING)
            local running_elapsed=$(($(date +%s) - running_start_time))
            if [[ ${running_elapsed} -gt ${timeout} ]]; then
                echo "  ERROR: Peridigm job ${job_id} exceeded RUNNING timeout (${timeout}s)"
                scancel "${job_id}" 2>/dev/null || true
                return 1
            fi
        fi
        # PENDING jobs wait indefinitely - cluster is just busy

        sleep ${check_interval}
    done
}

# ============================================================================
# UPDATE CONFIG WITH OUTPUT DIRECTORY
# ============================================================================

cd "${OF_ROOT}"

TEMP_CONFIG="${OUTPUT_DIR}/dual_physics_config.yaml"
cp "${CONFIG_FILE}" "${TEMP_CONFIG}"
sed -i "s|dir: .*|dir: \"${OUTPUT_DIR}\"|" "${TEMP_CONFIG}"

# Extract Peridigm HPC settings from config
PERIDIGM_NTASKS=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print(c.get('peridigm',{}).get('hpc',{}).get('ntasks', 4))")
PERIDIGM_TIME=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print(c.get('peridigm',{}).get('hpc',{}).get('time_limit', '01:00:00'))")
PERIDIGM_TIMEOUT=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print(c.get('peridigm',{}).get('hpc',{}).get('timeout', 3600))")

# Extract Peridigm material settings from config
PLA_CRIT_STRETCH=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print(c.get('peridigm',{}).get('pla_critical_stretch', 0.016))")
IMPACT_VELOCITY=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print(c.get('peridigm',{}).get('impact_velocity', 50.0))")
SIM_TIME=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print(c.get('peridigm',{}).get('simulation_time_s', 8e-4))")
PLA_HORIZON=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print(c.get('peridigm',{}).get('horizon_mm', 4.5) / 1000.0)")

INITIAL_CONE_ANGLE=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print(c.get('peridigm',{}).get('cone_half_angle_deg', 10))")

# Extract v7 cone-cylinder parameters from config
CONE_HEIGHT_M=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print(c.get('peridigm',{}).get('cone_height_m', 0.030))")
CONE_CYL_R_M=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print(c.get('peridigm',{}).get('cone_cylinder_radius_m', '') or '')")
CONE_MASS_KG_VAL=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print(c.get('peridigm',{}).get('cone_mass_kg', '') or '')")
USE_INITIAL_VELOCITY=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print('1' if c.get('peridigm',{}).get('cone_initial_velocity', False) else '0')")
USE_RIGID_CONE=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print('1' if c.get('peridigm',{}).get('rigid_cone', False) else '0')")
USE_CONE_NO_DAMAGE=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print('1' if c.get('peridigm',{}).get('cone_no_damage', False) else '0')")
STEEL_GAP_M=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print(c.get('peridigm',{}).get('steel_gap_m', 0.003))")
PARTICLE_SPACING_MM=$(python3 -c "import yaml; c=yaml.safe_load(open('${TEMP_CONFIG}')); print(c.get('peridigm',{}).get('particle_spacing_mm', '') or '')")

# Compute fracture critical stretch from FF and PLA sc: sc_frac = sc × FF^0.5
# Read initial_fracture_factor from config (used until per-band FF is read from state)
FRACTURE_CRIT_STRETCH=$(python3 -c "
import yaml, math
c = yaml.safe_load(open('${TEMP_CONFIG}'))
sc = c.get('peridigm',{}).get('pla_critical_stretch', 0.016)
ff = c.get('optimization',{}).get('initial_fracture_factor', 0.3)
print(f'{sc * math.sqrt(ff):.6f}')
")

echo "Peridigm settings: ntasks=${PERIDIGM_NTASKS}, time=${PERIDIGM_TIME}, timeout=${PERIDIGM_TIMEOUT}s, horizon=${PLA_HORIZON}m"
echo "Peridigm material: PLA_crit=${PLA_CRIT_STRETCH}, fracture_crit=${FRACTURE_CRIT_STRETCH}, velocity=${IMPACT_VELOCITY} m/s"
echo "Peridigm cone: angle=${INITIAL_CONE_ANGLE}°, height=${CONE_HEIGHT_M}m, cyl_r=${CONE_CYL_R_M:-none}m, mass=${CONE_MASS_KG_VAL:-native}kg"
echo "Peridigm flags: initial_vel=${USE_INITIAL_VELOCITY}, rigid=${USE_RIGID_CONE}, no_damage=${USE_CONE_NO_DAMAGE}"

# ============================================================================
# CONTINUE FROM PREVIOUS RUN (OPTIONAL)
# ============================================================================
# Copies state from a previous run to use as starting point.
# Useful for:
#   - Continuing from where a previous run left off
#   - Testing algorithm changes from a known state
#   - Using a known good configuration as starting point
#
# Usage: sbatch --export=ALL,CONTINUE_FROM=/path/to/previous/run ...
# ============================================================================

STARTING_ITERATION=0

if [[ -n "${CONTINUE_FROM}" ]]; then
    echo ""
    echo "======================================================================"
    echo "Continuing from previous run: ${CONTINUE_FROM}"
    echo "======================================================================"

    if [[ -f "${CONTINUE_FROM}/optimization_state.json" ]]; then
        # Copy state file
        cp "${CONTINUE_FROM}/optimization_state.json" "${OUTPUT_DIR}/"
        echo "  Copied optimization_state.json"

        # Get the iteration number from previous run
        PREV_ITERATION=$(python3 -c "
import json
with open('${CONTINUE_FROM}/optimization_state.json', 'r') as f:
    state = json.load(f)
print(state.get('iteration', 0))
")
        STARTING_ITERATION=${PREV_ITERATION}
        echo "  Previous run ended at iteration ${PREV_ITERATION}"

        # Copy history if exists
        if [[ -f "${CONTINUE_FROM}/history.json" ]]; then
            cp "${CONTINUE_FROM}/history.json" "${OUTPUT_DIR}/"
            echo "  Copied history.json"
        fi

        # Show thickness values we're starting from
        python3 -c "
import json
with open('${OUTPUT_DIR}/optimization_state.json', 'r') as f:
    state = json.load(f)
thickness = state.get('thickness_params', {}).get('thickness_mm', [])
print(f'  Starting thickness (mm): {[round(t, 2) for t in thickness]}')
"
    else
        echo "  WARNING: No optimization_state.json found in ${CONTINUE_FROM}"
        echo "  Starting fresh instead."
    fi
    echo ""
fi

# ============================================================================
# MAIN OPTIMIZATION LOOP
# ============================================================================

# Read max_iterations from YAML config (default 25 if not found)
MAX_ITERATIONS=$(python3 -c "
import yaml
with open('${CONFIG_FILE}', 'r') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('optimization', {}).get('max_iterations', 25))
" 2>/dev/null || echo 25)
echo "Max iterations: ${MAX_ITERATIONS} (from config)"
ITERATION=${STARTING_ITERATION}
CONVERGED=false

echo "======================================================================"
echo "Starting Dual-Physics Optimization Loop (PARALLEL FEA + Peridigm)"
echo "======================================================================"

while [[ ${ITERATION} -lt ${MAX_ITERATIONS} ]] && [[ "${CONVERGED}" == "false" ]]; do
    ITERATION=$((ITERATION + 1))
    PERIDIGM_JOB_ID=""

    echo ""
    echo "======================================================================"
    echo "ITERATION ${ITERATION}"
    echo "======================================================================"
    echo "Started at: $(date)"

    # ------------------------------------------------------------------
    # PHASE 1: Generate mesh (inside container)
    # ------------------------------------------------------------------
    echo ""
    echo "[Phase 1] Generating mesh (inside container)..."
    MESH_START=$(date +%s)

    run_in_container "python3 -u ${OF_ROOT}/scripts/run_dual_physics_mesh_only.py --config ${TEMP_CONFIG} --iteration ${ITERATION} --output-dir ${OUTPUT_DIR}"

    MESH_EXIT_CODE=$?
    MESH_END=$(date +%s)
    echo "  Mesh generation took $((MESH_END - MESH_START))s"

    if [[ ${MESH_EXIT_CODE} -ne 0 ]]; then
        # Check if output files were actually produced (PETSc/DOLFINx can crash
        # during cleanup after all outputs are written, yielding exit 134/139)
        MESH_XDMF="${OUTPUT_DIR}/core_shell_refined.xdmf"
        PERIDIGM_TXT="${OUTPUT_DIR}/peridigm/core_shell_refined_peridigm.txt"
        if [[ -f "${MESH_XDMF}" && -f "${PERIDIGM_TXT}" ]]; then
            echo "WARNING: Mesh process exited ${MESH_EXIT_CODE} but output files exist — continuing"
        else
            echo "WARNING: Mesh generation failed (exit code: ${MESH_EXIT_CODE}) — skipping to next iteration"
            continue
        fi
    fi

    # Store fixed cone angle in state (for process_peridigm_results.py to read)
    python3 -c "
import json
sf='${OUTPUT_DIR}/optimization_state.json'
with open(sf) as f: s=json.load(f)
s['cone_half_angle_deg']=${INITIAL_CONE_ANGLE}
with open(sf,'w') as f: json.dump(s,f,indent=2)
"

    # ------------------------------------------------------------------
    # PHASE 2: Submit Peridigm job IMMEDIATELY (runs in background)
    # ------------------------------------------------------------------
    echo ""
    echo "[Phase 2] Submitting Peridigm job (will run in parallel with FEA)..."

    # Read mesh output directory from state
    MESH_OUTPUT_DIR=$(python3 -c "
import json
with open('${OUTPUT_DIR}/optimization_state.json', 'r') as f:
    state = json.load(f)
print(state.get('mesh_output_dir', '${OUTPUT_DIR}'))
")

    # Use the FRESHLY GENERATED Peridigm discretization (not stale iter_N files!)
    # The mesh generator exports to: ${MESH_OUTPUT_DIR}/peridigm/core_shell_refined_peridigm.txt
    PERIDIGM_DISC="${MESH_OUTPUT_DIR}/peridigm/core_shell_refined_peridigm.txt"

    if [[ ! -f "${PERIDIGM_DISC}" ]]; then
        echo "  WARNING: Fresh Peridigm discretization not found at ${PERIDIGM_DISC}"
        echo "  Checking alternate locations (maxdepth 1 to avoid stale iter_N files)..."
        PERIDIGM_DISC=$(find "${OUTPUT_DIR}/peridigm" -maxdepth 1 -name "*_peridigm.txt" -type f 2>/dev/null | head -1)
    fi

    if [[ -z "${PERIDIGM_DISC}" ]]; then
        echo "  WARNING: No Peridigm discretization found. Skipping Peridigm."
    else
        echo "  Found: ${PERIDIGM_DISC}"

        # --- DEDUP CHECK: skip Peridigm if discretization is identical to previous iteration ---
        DISC_HASH=$(md5sum "${PERIDIGM_DISC}" 2>/dev/null | awk '{print $1}')
        PREV_HASH_FILE="${OUTPUT_DIR}/peridigm/.prev_disc_hash"
        PREV_HASH=""
        [[ -f "${PREV_HASH_FILE}" ]] && PREV_HASH=$(cat "${PREV_HASH_FILE}")

        # Always save current hash for next iteration's dedup check
        mkdir -p "$(dirname "${PREV_HASH_FILE}")"
        echo "${DISC_HASH}" > "${PREV_HASH_FILE}"

        if [[ -n "${DISC_HASH}" && "${DISC_HASH}" == "${PREV_HASH}" && ${ITERATION} -gt 1 ]]; then
            echo "  DEDUP: Discretization unchanged (hash=${DISC_HASH:0:12}...)"
            echo "  Reusing previous Peridigm results — skipping submission"

            # Find most recent Peridigm output directory with actual Exodus files
            PREV_ITER=$((ITERATION - 1))
            PREV_PERIDIGM_DIR="${OUTPUT_DIR}/peridigm/iter_${PREV_ITER}"
            while [[ ! -d "${PREV_PERIDIGM_DIR}" || -z "$(ls ${PREV_PERIDIGM_DIR}/*.e* 2>/dev/null)" ]] && [[ ${PREV_ITER} -gt 0 ]]; do
                PREV_ITER=$((PREV_ITER - 1))
                PREV_PERIDIGM_DIR="${OUTPUT_DIR}/peridigm/iter_${PREV_ITER}"
            done

            if [[ -d "${PREV_PERIDIGM_DIR}" ]]; then
                PERIDIGM_OUTDIR="${OUTPUT_DIR}/peridigm/iter_${ITERATION}"
                mkdir -p "${PERIDIGM_OUTDIR}"
                # Symlink Exodus files and disc files from previous run
                for f in "${PREV_PERIDIGM_DIR}"/*; do
                    ln -sf "$f" "${PERIDIGM_OUTDIR}/$(basename $f)" 2>/dev/null || true
                done
                echo "  Linked results from iter_${PREV_ITER} → iter_${ITERATION}"
                PERIDIGM_JOB_ID="reused_iter_${PREV_ITER}"
                PERIDIGM_EXIT_CODE=0
            else
                echo "  WARNING: No previous Peridigm output found — running fresh"
            fi
        fi

        # Only submit new Peridigm job if not deduped
        if [[ -z "${PERIDIGM_JOB_ID}" || "${PERIDIGM_JOB_ID}" != reused_* ]]; then

        # The Peridigm SLURM script expects: ${THETA_RUN_DIR}/results_refined/peridigm/
        # But theta_mesh exports to: ${OUTPUT_DIR}/peridigm/
        # Fix: Create symlink to match expected structure
        PERIDIGM_PARENT=$(dirname "${PERIDIGM_DISC}")
        THETA_RUN_DIR=$(dirname "${PERIDIGM_PARENT}")

        # ALWAYS recreate symlink to ensure it points to FRESH discretization
        # (Previous bug: stale symlink pointed to iter_N subdirectory)
        mkdir -p "${THETA_RUN_DIR}/results_refined"
        rm -f "${THETA_RUN_DIR}/results_refined/peridigm"  # Remove stale symlink
        ln -sf "${PERIDIGM_PARENT}" "${THETA_RUN_DIR}/results_refined/peridigm"
        echo "  Created symlink: results_refined/peridigm -> $(basename ${PERIDIGM_PARENT})"

        PERIDIGM_OUTDIR="${OUTPUT_DIR}/peridigm/iter_${ITERATION}"
        mkdir -p "${PERIDIGM_OUTDIR}"

        # Set Peridigm environment variables for FLYBACK simulation
        # Cone starts inside hole at z_min, moves +z into shell at impact velocity
        export THETA_RUN_DIR="${THETA_RUN_DIR}"
        export IMPACT_VZ="${IMPACT_VELOCITY}"    # From config (m/s)
        export FINAL_TIME="${SIM_TIME}"          # From config (s)
        export PLA_CRIT_STRETCH="${PLA_CRIT_STRETCH}"           # From config
        export FRACTURE_CRIT_STRETCH="${FRACTURE_CRIT_STRETCH}" # From config (sc × FF^0.5)
        export CONE_ENABLE="1"
        # Cone angle is fixed for v7 geometry (adaptation disabled)
        export CONE_HALF_ANGLE_DEG="${INITIAL_CONE_ANGLE}"
        echo "  Cone half-angle: ${CONE_HALF_ANGLE_DEG}° (fixed v7)"
        export CONE_HEIGHT="${CONE_HEIGHT_M}"
        export CONE_AXIS="-z"            # Cone opens backward (base behind tip)
        export FLYBACK_MODE="1"          # Position cone at z_min (inside hole)
        export OUTDIR="${PERIDIGM_OUTDIR}"

        # v7 cone-cylinder parameters
        [[ -n "${CONE_CYL_R_M}" ]] && export CONE_CYL_R="${CONE_CYL_R_M}"
        [[ -n "${CONE_MASS_KG_VAL}" ]] && export CONE_MASS_KG="${CONE_MASS_KG_VAL}"
        [[ -n "${PARTICLE_SPACING_MM}" ]] && export PARTICLE_SPACING="$(python3 -c "print(${PARTICLE_SPACING_MM}/1000.0)")"
        export INITIAL_VELOCITY="${USE_INITIAL_VELOCITY}"
        export RIGID_CONE="${USE_RIGID_CONE}"
        export CONE_NO_DAMAGE="${USE_CONE_NO_DAMAGE}"

        # Flyback split core parameters (for run_flyback_split_core.slurm)
        export SOURCE_DISC="${PERIDIGM_DISC}"
        export SPLIT_AXIS="x"            # Split steel core along X axis
        export GAP="${STEEL_GAP_M}"      # From config
        export PLA_HORIZON="${PLA_HORIZON}"

        # Submit Peridigm using FLYBACK SPLIT CORE (constant velocity cone)
        # This uses Prescribed Displacement so cone pushes through at constant velocity
        # NOT Initial Velocity which would bounce off
        PERIDIGM_JOB_ID=$(sbatch --parsable \
            --partition=${SLURM_JOB_PARTITION:-shared} \
            --nodes=1 \
            --ntasks=${PERIDIGM_NTASKS} \
            --time=${PERIDIGM_TIME} \
            --output="${PERIDIGM_OUTDIR}/peridigm_%j.log" \
            --error="${PERIDIGM_OUTDIR}/peridigm_%j.err" \
            --export=ALL \
            "${PERIDIGM_MODULE}/hpc/run_flyback_split_core.slurm")

        echo "  Submitted Peridigm job: ${PERIDIGM_JOB_ID} (running in background)"

        fi  # end of dedup guard (only submit if not reused)
    fi

    # ------------------------------------------------------------------
    # PHASE 3: Run FEA (inside container) - PARALLEL with Peridigm
    # ------------------------------------------------------------------
    echo ""
    echo "[Phase 3] Running FEA (inside container, parallel with Peridigm)..."
    FEA_START=$(date +%s)

    run_in_container "python3 -u ${OF_ROOT}/scripts/run_dual_physics_fea_only.py --config ${TEMP_CONFIG} --iteration ${ITERATION} --output-dir ${OUTPUT_DIR}"

    FEA_EXIT_CODE=$?
    FEA_END=$(date +%s)
    echo "  FEA took $((FEA_END - FEA_START))s"

    if [[ ${FEA_EXIT_CODE} -ne 0 ]]; then
        # Check if FEA actually completed (PETSc/DOLFINx can crash during
        # cleanup after all results are written, yielding exit 134/139)
        FEA_COMPLETED=$(python3 -c "
import json
try:
    s=json.load(open('${OUTPUT_DIR}/optimization_state.json'))
    print('yes' if s.get('fea_completed') and s.get('iteration',0)==${ITERATION} else 'no')
except: print('no')
" 2>/dev/null)
        if [[ "${FEA_COMPLETED}" == "yes" ]]; then
            echo "WARNING: FEA process exited ${FEA_EXIT_CODE} but state shows fea_completed — continuing"
        else
            echo "WARNING: FEA phase failed (exit code: ${FEA_EXIT_CODE}) — skipping to next iteration"
            # Cancel Peridigm job if running (skip if deduped)
            [[ -n "${PERIDIGM_JOB_ID}" && "${PERIDIGM_JOB_ID}" != reused_* ]] && scancel "${PERIDIGM_JOB_ID}" 2>/dev/null || true
            continue
        fi
    fi

    # ------------------------------------------------------------------
    # PHASE 4: Wait for Peridigm (if still running)
    # ------------------------------------------------------------------
    if [[ -n "${PERIDIGM_JOB_ID}" ]]; then
        if [[ "${PERIDIGM_JOB_ID}" == reused_* ]]; then
            # Deduped — results already linked, skip wait
            echo ""
            echo "[Phase 4] Peridigm deduped (${PERIDIGM_JOB_ID}) — using cached results"
            PERIDIGM_EXIT_CODE=0
        else
            # Real job — wait for completion
            PERIDIGM_STATUS=$(squeue -j "${PERIDIGM_JOB_ID}" -h -o "%T" 2>/dev/null || echo "")

            if [[ -n "${PERIDIGM_STATUS}" ]]; then
                echo ""
                echo "[Phase 4] Waiting for Peridigm job ${PERIDIGM_JOB_ID} to complete..."
                wait_for_job "${PERIDIGM_JOB_ID}" "${PERIDIGM_TIMEOUT}"
                PERIDIGM_EXIT_CODE=$?
            else
                # Already finished - check final state
                FINAL_STATE=$(sacct -j "${PERIDIGM_JOB_ID}" -n -o State | head -1 | awk '{print $1}')
                echo ""
                echo "[Phase 4] Peridigm already completed (state: ${FINAL_STATE})"
                [[ "${FINAL_STATE}" == "COMPLETED" ]] && PERIDIGM_EXIT_CODE=0 || PERIDIGM_EXIT_CODE=1
            fi
        fi

        # ------------------------------------------------------------------
        # PHASE 5: Process Peridigm results (inside container)
        # ------------------------------------------------------------------
        if [[ ${PERIDIGM_EXIT_CODE} -eq 0 ]]; then
            echo ""
            echo "[Phase 5] Processing Peridigm results..."
            run_in_container "python3 -u ${OF_ROOT}/scripts/process_peridigm_results.py --output-dir ${OUTPUT_DIR} --peridigm-dir ${PERIDIGM_OUTDIR} --job-id ${PERIDIGM_JOB_ID}"
        else
            echo "  WARNING: Peridigm failed, continuing with FEA-only results"
        fi
    fi

    # ------------------------------------------------------------------
    # Check convergence
    # ------------------------------------------------------------------
    CONVERGED=$(python3 -c "
import json
with open('${OUTPUT_DIR}/optimization_state.json', 'r') as f:
    state = json.load(f)
print('true' if state.get('converged', False) else 'false')
")

    ITER_END=$(date +%s)
    echo ""
    echo "Iteration ${ITERATION} completed in $((ITER_END - MESH_START))s"
    echo "  (Mesh: $((MESH_END - MESH_START))s, FEA: $((FEA_END - FEA_START))s)"

    if [[ "${CONVERGED}" == "true" ]]; then
        echo ""
        echo "*** Optimization converged! ***"
        break
    fi
done

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "======================================================================"
echo "Dual-Physics Optimization Complete"
echo "======================================================================"
echo "Total iterations: ${ITERATION}"
echo "Converged: ${CONVERGED}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Finished at: $(date)"

if [[ -f "${OUTPUT_DIR}/history.json" ]]; then
    echo ""
    echo "Convergence history:"
    python3 -c "
import json
with open('${OUTPUT_DIR}/history.json', 'r') as f:
    history = json.load(f)
for h in history:
    stress = h.get('max_stress_MPa', 0)
    D_frac = h.get('D_fracture_zone', 'N/A')
    if isinstance(D_frac, float):
        D_frac = f'{D_frac:.3f}'
    print(f\"  Iter {h['iteration']}: stress={stress:.2f} MPa, D_fracture={D_frac}\")
"
fi

echo ""
echo "To retrieve results:"
echo "  rsync -avz mtdsn@koa.its.hawaii.edu:${OUTPUT_DIR}/ ./results/"

exit 0
