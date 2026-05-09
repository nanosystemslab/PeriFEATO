# PeriFEATO

[![status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/nanosystemslab/PeriFEATO)
[![python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)
[![DOLFINx](https://img.shields.io/badge/DOLFINx-0.9.0-blue)](https://github.com/FEniCS/dolfinx)
[![Peridigm](https://img.shields.io/badge/Peridigm-fork-blue)](https://github.com/mattnakamura/peridigm)

[![Physics](https://img.shields.io/badge/physics-FEA%20%2B%20Peridynamics-blueviolet)](#overview)
[![Optimizers](https://img.shields.io/badge/optimizers-7-green)](#supported-optimization-algorithms)
[![Mesh](https://img.shields.io/badge/mesh-Gmsh%20parametric-yellow)](perifeato/mesh/)
[![HPC](https://img.shields.io/badge/HPC-SLURM%20%2B%20Singularity-lightgrey)](examples/hpc_templates/)
[![Container](https://img.shields.io/badge/docker-mattnakamura%2Fdolfinx-2496ED?logo=docker)](https://hub.docker.com/r/mattnakamura/dolfinx)

**Peri**digm + **FEA** + **T**opology **O**ptimization -- or as the Italians would say, *perfetto*.

A dual-physics shell thickness optimization framework coupling finite element analysis (FEA) with peridynamic fracture simulation (Peridigm). PeriFEATO iteratively designs shell structures that satisfy both stress constraints and fracture performance targets.

## Overview

PeriFEATO provides a complete pipeline for impact-driven topology optimization of core-shell structures:

1. **Parametric mesh generation** -- Gmsh-based 3D core-shell geometry with theta-banded thickness control
2. **Multi-orientation FEA** -- DOLFINx transient dynamics with Newmark-beta integration and penalty contact
3. **Peridynamic fracture analysis** -- Peridigm state-based peridynamics for high-velocity impact simulation
4. **Optimization loop** -- Iterative thickness updates with 7 optimization algorithms

### Supported Optimization Algorithms

| Algorithm | Key | Description |
|-----------|-----|-------------|
| FSD-G     | `fsd` | Fully Stressed Design with stress diffusion |
| FSD-GA    | `fsd` + secant | FSD with gradient-adjoint (secant) acceleration |
| OC        | `oc` | Optimality Criteria with adaptive move limits |
| MMA       | `mma` | Method of Moving Asymptotes (Svanberg 1987) |
| BO        | `bo` | Bayesian Optimization with GP surrogate + Expected Improvement |
| PGD       | `pgd` | Projected Gradient Descent with Barzilai-Borwein step size |
| COBYLA    | `cobyla` | Constrained Optimization by Linear Approximations |

## Installation

```bash
git clone https://github.com/mattnakamura/PeriFEATO.git
cd PeriFEATO
pip install -e .
```

### Prerequisites

PeriFEATO requires:

- **Python 3.10+**
- **DOLFINx 0.9.0** -- for finite element analysis (see [container setup](#container-setup))
- **Peridigm** -- for peridynamic fracture simulation

#### Peridigm Installation

PeriFEATO requires a specific fork of Peridigm with dependency modifications for compatibility:

- **Peridigm fork**: [mattnakamura/peridigm](https://github.com/mattnakamura/peridigm)
- **Build pipeline for UH KOA HPC**: [mattnakamura/KOA_Peridigm_Install](https://github.com/mattnakamura/KOA_Peridigm_Install)

Follow the instructions in [KOA_Peridigm_Install](https://github.com/mattnakamura/KOA_Peridigm_Install) to build Peridigm and its dependencies (HDF5, NetCDF, Trilinos) on the UH KOA cluster.

#### Container Setup

A pre-built DOLFINx container with PyVista is available on Docker Hub:

```bash
docker pull mattnakamura/dolfinx:v0.9.0-pyvista
```

For HPC (Singularity/Apptainer):

```bash
singularity build dolfinx_v0.9.0_pyvista.sif docker://mattnakamura/dolfinx:v0.9.0-pyvista
```

See `containers/` for Dockerfile and Singularity definition files.

## Quick Start

```python
from perifeato.optimization import run_dual_physics_optimization

# Run optimization with a YAML config
thickness_params, history = run_dual_physics_optimization("configs/dual_physics_opt_fsd.yaml")
```

Or from the command line:

```bash
python -m perifeato.cli.run_optimization --config configs/dual_physics_opt_fsd.yaml
```

## Project Structure

```
PeriFEATO/
├── perifeato/                # Installable Python package
│   ├── mesh/                 # Parametric mesh generation (Gmsh)
│   ├── fea/                  # FEA solver (DOLFINx)
│   ├── fracture/             # Peridigm fracture wrapper
│   └── optimization/         # Optimization algorithms + loop
├── examples/                 # Example configs and HPC templates
│   ├── configs/              # YAML configs for each algorithm
│   └── hpc_templates/        # SLURM job scripts
├── containers/               # Docker + Singularity definitions
├── docs/                     # Pipeline documentation
└── scripts/                  # Utility scripts
```

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [mattnakamura/peridigm](https://github.com/mattnakamura/peridigm) | Fork of Peridigm with KOA-specific dependency patches |
| [mattnakamura/KOA_Peridigm_Install](https://github.com/mattnakamura/KOA_Peridigm_Install) | Automated build pipeline for Peridigm on UH KOA HPC |

## License

This project is licensed under the GNU General Public License v3.0 -- see the [LICENSE](LICENSE) file for details.
