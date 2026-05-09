# PeriFEATO Examples

## Configs (`configs/`)

YAML configurations for the dual-physics optimization, one per algorithm:

| Config | Algorithm |
|--------|-----------|
| `dual_physics_opt_fsdg.yaml`   | FSD-G (Fully Stressed Design) |
| `dual_physics_opt_fsdga.yaml`  | FSD-GA (FSD with secant acceleration) |
| `dual_physics_opt_oc.yaml`     | Optimality Criteria |
| `dual_physics_opt_mma.yaml`    | Method of Moving Asymptotes |
| `dual_physics_opt_bo.yaml`     | Bayesian Optimization |
| `dual_physics_opt_pgd.yaml`    | Projected Gradient Descent |
| `dual_physics_opt_cobyla.yaml` | COBYLA |
| `dual_physics_local_test.yaml` | Quick local test (mock Peridigm) |

The mesh and FEA solver have separate base configs:

- `mesh_base_config.yaml` -- Geometry and mesh sizing parameters
- `fea_base_config.yaml`  -- Material properties, load cases, solver options

## HPC Templates (`hpc_templates/`)

SLURM job scripts for HPC execution. These are templates -- adapt paths and
account information for your cluster.

- `run_dual_physics_optimization.slurm` -- Main dual-physics (FEA + Peridigm) optimization
- `run_thickness_optimization.slurm`    -- FEA-only thickness optimization
- `submit_all_methods.sh`               -- Helper to submit all 7 algorithms in sequence
- `generate_theta_mesh_single.slurm`    -- Standalone mesh generation
- `run_core_shell_from_theta_mesh.slurm` -- Standalone Peridigm fracture run
- `peridigm_env.sh`                     -- Environment loader for Peridigm runtime

## Running an Example

Local (with mock Peridigm):

```bash
python -m perifeato.cli.run_optimization --config examples/configs/dual_physics_local_test.yaml
```

HPC (UH KOA):

```bash
sbatch examples/hpc_templates/run_dual_physics_optimization.slurm
```
