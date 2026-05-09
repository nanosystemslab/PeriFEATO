#!/bin/bash
# Build DOLFINx container with PyVista support

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CONTAINER="${BASE_CONTAINER:-${HOME}/Optimization_Framework/containers/dolfinx_v0.9.0.sif}"
OUTPUT_CONTAINER="${OUTPUT_CONTAINER:-${HOME}/Optimization_Framework/containers/dolfinx_v0.9.0_pyvista.sif}"
DEF_FILE="${SCRIPT_DIR}/dolfinx_v0.9.0_pyvista.def"

echo "=== Building DOLFINx Container with PyVista ==="
echo "Base container:   ${BASE_CONTAINER}"
echo "Output container: ${OUTPUT_CONTAINER}"
echo "Definition file:  ${DEF_FILE}"
echo ""

# Check if base container exists
if [ ! -f "${BASE_CONTAINER}" ]; then
    echo "ERROR: Base container not found: ${BASE_CONTAINER}"
    echo ""
    echo "Please ensure the base DOLFINx container exists, or set BASE_CONTAINER environment variable."
    exit 1
fi

# Check if definition file exists
if [ ! -f "${DEF_FILE}" ]; then
    echo "ERROR: Definition file not found: ${DEF_FILE}"
    exit 1
fi

# Check if output already exists
if [ -f "${OUTPUT_CONTAINER}" ]; then
    echo "WARNING: Output container already exists: ${OUTPUT_CONTAINER}"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    rm -f "${OUTPUT_CONTAINER}"
fi

# Build container
echo "Building container (this may take 5-10 minutes)..."
echo ""

# Singularity build requires root or --fakeroot
# On HPC, use --fakeroot if available
if singularity help build | grep -q "\-\-fakeroot"; then
    FAKEROOT_FLAG="--fakeroot"
else
    FAKEROOT_FLAG=""
fi

singularity build ${FAKEROOT_FLAG} "${OUTPUT_CONTAINER}" "${DEF_FILE}"

echo ""
echo "=== Build Complete ==="
echo "Container: ${OUTPUT_CONTAINER}"
echo ""

# Test the new container
echo "=== Testing PyVista Installation ==="
singularity exec "${OUTPUT_CONTAINER}" python3 << 'EOF'
import sys
print(f"Python: {sys.version}")
print()

try:
    import pyvista as pv
    print(f"✓ PyVista {pv.__version__} available")
except ImportError as e:
    print(f"✗ PyVista not available: {e}")
    sys.exit(1)

try:
    import dolfinx
    print(f"✓ DOLFINx {dolfinx.__version__} available")
except ImportError as e:
    print(f"✗ DOLFINx not available: {e}")
    sys.exit(1)

try:
    import meshio
    print(f"✓ meshio {meshio.__version__} available")
except ImportError:
    print("✗ meshio not available")

print()
print("Container is ready to use!")
EOF

echo ""
echo "=== Usage ==="
echo "Update your SLURM script to use the new container:"
echo "  CONTAINER=\"${OUTPUT_CONTAINER}\""
echo ""
echo "Or update the default in run_thickness_optimization_hpc.slurm"
