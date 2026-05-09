#!/usr/bin/env bash
set -euo pipefail

# Load Peridigm runtime environment on KOA.
#
# Customize the install root if your KOA install lives elsewhere:
#   export PERIDIGM_INSTALL_ROOT="$HOME/KOA_Peridigm_Install/software/install"
#
# After sourcing, `Peridigm` should be in PATH.

if ! type module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
fi

module purge 2>/dev/null || true
module load compiler/GCC/13.2.0 2>/dev/null || true
module load mpi/OpenMPI/4.1.6-GCC-13.2.0 2>/dev/null || true
module load devel/Boost/1.83.0-GCC-13.2.0 2>/dev/null || true
module load numlib/OpenBLAS/0.3.24-GCC-13.2.0 2>/dev/null || true

PERIDIGM_INSTALL_ROOT="${PERIDIGM_INSTALL_ROOT:-${HOME}/KOA_Peridigm_Install/software/install}"

export PERIDIGM_HOME="${PERIDIGM_INSTALL_ROOT}/peridigm"
export TRILINOS_HOME="${PERIDIGM_INSTALL_ROOT}/trilinos"
export NETCDF_ROOT="${PERIDIGM_INSTALL_ROOT}"
export HDF5_ROOT="${PERIDIGM_INSTALL_ROOT}"

export PATH="${PERIDIGM_HOME}/bin:${PERIDIGM_INSTALL_ROOT}/bin:${PATH}"

# Prefer module-provided libstdc++ first.
if [[ -n "${EBROOTGCC:-}" ]]; then
  export LD_LIBRARY_PATH="${EBROOTGCC}/lib64:${EBROOTGCC}/lib:${LD_LIBRARY_PATH:-}"
fi
export LD_LIBRARY_PATH="${PERIDIGM_HOME}/lib:${TRILINOS_HOME}/lib:${NETCDF_ROOT}/lib:${HDF5_ROOT}/lib:${PERIDIGM_INSTALL_ROOT}/lib:${LD_LIBRARY_PATH:-}"

# MPI settings seen to help on KOA.
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_btl=^openib
export PMIX_MCA_gds=hash

command -v Peridigm >/dev/null 2>&1 || {
  echo "ERROR: Peridigm not found in PATH after env setup" >&2
  echo "PERIDIGM_INSTALL_ROOT=${PERIDIGM_INSTALL_ROOT}" >&2
  return 2
}
