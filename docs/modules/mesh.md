# theta_mesh module

Self-contained copy of the theta-banded (t(theta)) mesh generation pipeline.

## Contents
- `src/shell_geometry_generator_refined.py`: Gmsh OCC generator; exports `.msh/.vtu/.xdmf` (+ Peridigm discretization)
- `config/base_config.yaml`: baseline parameters (mesh sizing, occ_heal=false, etc.)
- `manifests/theta_mesh_manifest.example.yaml`: sweep case examples
- `hpc/start_theta_mesh_sweep.sh`: creates a sweep dir + submits Slurm array
- `hpc/generate_theta_mesh_sweep_array.slurm`: Slurm array job to generate meshes
- `tools/local_try_mesh_brep.py`: local debug for `mesh_fail.brep`
- `docs/theta_mesh_module.md`: usage notes

## KOA usage (typical)
On KOA, copy this module into the folder you run jobs from, then run:

```bash
bash hpc/start_theta_mesh_sweep.sh --partition shared
```

Then inspect outputs in the created `mesh_sweeps/sweep_<timestamp>/...` directory.

## Peridigm export
By default the generator also writes a Peridigm “Text File” discretization derived from the same tetra mesh:

- `results_refined/peridigm/core_shell_refined_peridigm.txt` (columns: `x y z block_id volume`)
- `results_refined/peridigm/nodeset_*.txt` (1-based point ids for BCs)
