# Installation Guide

PeriFEATO sits on top of two external solvers (DOLFINx for FEA, Peridigm for
peridynamics) and a parametric mesher (Gmsh). The Python package itself is
lightweight; most of the install effort goes into the underlying solvers.

## 1. Python package

```bash
git clone https://github.com/mattnakamura/PeriFEATO.git
cd PeriFEATO
pip install -e .
```

For the optional FEA dependencies (DOLFINx + PyVista) and optimization extras (SciPy):

```bash
pip install -e ".[all]"
```

## 2. DOLFINx (FEA)

Use the prebuilt container:

```bash
docker pull mattnakamura/dolfinx:v0.9.0-pyvista
```

For HPC (Singularity / Apptainer):

```bash
singularity build dolfinx_v0.9.0_pyvista.sif docker://mattnakamura/dolfinx:v0.9.0-pyvista
```

Or build from source — see `containers/Dockerfile` and
`containers/dolfinx_v0.9.0_pyvista.def`.

## 3. Peridigm (peridynamics)

PeriFEATO requires a **specific fork** of Peridigm with dependency
modifications for compatibility with modern compilers and HDF5/Trilinos
versions:

- **Fork (required):** https://github.com/mattnakamura/peridigm

The upstream Sandia Peridigm will not work as a drop-in replacement -- the
fork patches deprecated C++ standard library headers and link-time issues
introduced by newer dependency stacks.

### On UH KOA HPC

A complete automated build pipeline for KOA is published as a separate repo:

- **Build pipeline:** https://github.com/mattnakamura/KOA_Peridigm_Install

The pipeline manages SLURM job dependencies to compile the full stack
(HDF5 1.14.3 → NetCDF 4.9.2 → Trilinos 13.4.1 → Peridigm) in roughly 15 hours
on a typical KOA node.

After installation, set the Peridigm binary path in your environment:

```bash
export PERIDIGM_BIN=$HOME/KOA_Peridigm_Install/software/install/Peridigm/bin/Peridigm
```

The example HPC templates (`examples/hpc_templates/peridigm_env.sh`) load the
correct module stack and library paths.

### On other clusters

Adapt `KOA_Peridigm_Install` to your cluster's module system, or build manually
following the dependency order:

1. HDF5 1.14.3 (with parallel + C++ support)
2. NetCDF-C 4.9.2 + NetCDF-CXX
3. Trilinos 13.4.1 (with SEACAS / mesh I/O)
4. Peridigm from `mattnakamura/peridigm`

## 4. Gmsh (mesh generation)

Install via pip:

```bash
pip install gmsh
```

Or use the same DOLFINx container above -- it includes Gmsh.

## Verifying the install

A quick local test (uses mock Peridigm, no HPC required):

```bash
python -m perifeato.cli.run_optimization --config examples/configs/dual_physics_local_test.yaml
```

If this completes a few iterations without error, the Python side is wired up
correctly. For a real Peridigm run, submit one of the HPC templates in
`examples/hpc_templates/`.
