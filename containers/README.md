# DOLFINx Container with PyVista

This directory contains Singularity definition files for building custom DOLFINx containers.

## Building the PyVista-Enhanced Container

The `dolfinx_v0.9.0_pyvista.sif` container extends the base DOLFINx 0.9.0 container with PyVista for robust VTK file reading.

### On KOA HPC

```bash
# 1. Sync the container definition files
cd ~/Optimization_Framework/containers

# 2. Build the container (requires fakeroot)
bash build_pyvista_container.sh
```

**Build time:** ~5-10 minutes

### Manual Build (if script fails)

```bash
cd ~/Optimization_Framework/containers

# Build with fakeroot (if available on your system)
singularity build --fakeroot dolfinx_v0.9.0_pyvista.sif dolfinx_v0.9.0_pyvista.def

# OR: Build without fakeroot (may require root/sudo)
singularity build dolfinx_v0.9.0_pyvista.sif dolfinx_v0.9.0_pyvista.def
```

### Verify Installation

```bash
singularity exec dolfinx_v0.9.0_pyvista.sif python3 -c "import pyvista; print(pyvista.__version__)"
```

## Using the New Container

Update your SLURM script to use the new container:

```bash
# In run_thickness_optimization_hpc.slurm
CONTAINER="${CONTAINER:-${HOME}/Optimization_Framework/containers/dolfinx_v0.9.0_pyvista.sif}"
```

## What Gets Installed

- **PyVista** - Modern VTK interface with better file format support
- **VTK Python bindings** - Installed as PyVista dependency
- All existing DOLFINx 0.9.0 functionality

## Why PyVista?

The base container's `meshio` library doesn't support VTU XML version 2.2 files. PyVista provides:

- Support for all VTU versions
- Native PVTU (parallel VTU) reading
- Better error handling
- Faster I/O

## Container Size

- Base container: ~1.5 GB
- With PyVista: ~1.7 GB (+200 MB)

## Troubleshooting

### "fakeroot not available"

If `--fakeroot` is not available, you have two options:

1. **Request root access** from your HPC admin to build containers
2. **Use overlay directory** (if supported):
   ```bash
   singularity build --sandbox dolfinx_pyvista_sandbox/ dolfinx_v0.9.0_pyvista.def
   singularity build dolfinx_v0.9.0_pyvista.sif dolfinx_pyvista_sandbox/
   ```

3. **Build locally** and transfer:
   ```bash
   # On your local machine (if you have Singularity/Docker)
   singularity build dolfinx_v0.9.0_pyvista.sif dolfinx_v0.9.0_pyvista.def

   # Transfer to HPC
   rsync -avz dolfinx_v0.9.0_pyvista.sif user@koa.its.hawaii.edu:~/Optimization_Framework/containers/
   ```

### Build fails with "FATAL: Unable to pull docker://..."

This means the definition file is trying to pull from Docker. Use the `localimage` bootstrap method instead (already configured in `dolfinx_v0.9.0_pyvista.def`).

### "Permission denied" errors

Ensure you have write access to the containers directory and sufficient disk space (~2 GB free).

## Alternative: Use Direct XML Parser

If building a new container is not possible, the VTK reader code includes a fallback direct XML parser that works without PyVista. See `to_do/local_mvp/src/adapters/fea_vtk_reader.py`.
