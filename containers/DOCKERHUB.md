# mattnakamura/dolfinx

DOLFINx 0.9.0 with PyVista for finite element analysis and visualization.
Built as the runtime container for [PeriFEATO](https://github.com/nanosystemslab/PeriFEATO) — a topology
optimization framework coupling FEA and peridynamic fracture simulation —
but usable as a standalone DOLFINx environment.

## What's inside

- **DOLFINx 0.9.0** — finite element solver from the FEniCS project
- **PyVista** — VTK-based visualization with off-screen rendering
- **Gmsh** — parametric mesh generator
- **MPI** (OpenMPI) — for parallel FEA
- Standard scientific Python stack: NumPy, SciPy, meshio, h5py, PyYAML

## Quick start

### Docker

```bash
docker pull mattnakamura/dolfinx:v0.9.0-pyvista
docker run --rm -it mattnakamura/dolfinx:v0.9.0-pyvista \
    python3 -c "import dolfinx; print(dolfinx.__version__)"
```

### Singularity / Apptainer (HPC)

```bash
singularity build dolfinx_v0.9.0_pyvista.sif docker://mattnakamura/dolfinx:v0.9.0-pyvista
singularity exec dolfinx_v0.9.0_pyvista.sif \
    python3 -c "import dolfinx; print(dolfinx.__version__)"
```

## Tags

| Tag | DOLFINx | PyVista | Notes |
|-----|---------|---------|-------|
| `v0.9.0-pyvista` | 0.9.0 | latest | Tagged release used by PeriFEATO |
| `latest` | 0.9.0 | latest | Alias for the current stable build |

## Use with PeriFEATO

This image is the FEA runtime that drives parallel multi-orientation
transient impact analysis inside PeriFEATO's optimization loop. To bind a
PeriFEATO checkout and run an end-to-end optimization on HPC:

```bash
singularity exec \
    --bind ~/PeriFEATO:/work \
    --pwd /work \
    dolfinx_v0.9.0_pyvista.sif \
    python3 scripts/run_thickness_optimization.py --config examples/configs/perifeato_thickness_test.yaml
```

See the [PeriFEATO repository](https://github.com/nanosystemslab/PeriFEATO) for
the full pipeline (mesh generation → FEA → Peridigm fracture → optimization
update) including SLURM templates and configuration examples for all 7
supported optimization algorithms.

## Source

The Dockerfile and Singularity definition live in PeriFEATO under
[`containers/`](https://github.com/nanosystemslab/PeriFEATO/tree/main/containers).
Pull requests and issues for the image itself are tracked in that
repository.

## License

The container packages DOLFINx (LGPL-3.0) and PyVista (MIT). The
PeriFEATO project itself is GPL-3.0 — see
[LICENSE](https://github.com/nanosystemslab/PeriFEATO/blob/main/LICENSE).
