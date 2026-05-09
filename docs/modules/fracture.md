# peridigm_fracture module

Wrappers + tests for running **Peridigm** fracture/impact simulations on KOA.

This module does **not** build Peridigm; it assumes you already have a working install on KOA.
Default expected install prefix:
- `$HOME/KOA_Peridigm_Install/software/install`

Override with:
- `export PERIDIGM_INSTALL_ROOT=/path/to/software/install`

## Contents
- `tools/peridigm_env.sh`: loads KOA modules + sets PATH/LD_LIBRARY_PATH
- `hpc/run_smoke_text.slurm`: 10-minute smoke test (tiny text mesh)
- `hpc/run_shell_cone_contact.slurm`: cone impacting an ellipsoid shell (text mesh)
- `hpc/run_core_shell_from_theta_mesh.slurm`: runs Peridigm on the theta_mesh Peridigm export (tet-centroid discretization)
- `tests/shell_cone_contact/src/mesh_generator.py`: generates the text mesh + node sets

## KOA quickstart
On KOA, from the module directory:

```bash
sbatch hpc/run_smoke_text.slurm
sbatch hpc/run_shell_cone_contact.slurm
```

## Run on a theta_mesh output (Option A)
After generating a mesh with `modules/theta_mesh`, each mesh run contains:
`results_refined/peridigm/core_shell_refined_peridigm.txt` and nodesets.

Run Peridigm directly on that exported discretization:

```bash
sbatch --partition shared --export=ALL,THETA_RUN_DIR=/abs/path/to/theta_run_dir \
  hpc/run_core_shell_from_theta_mesh.slurm
```

Damage model knobs (optional):

```bash
sbatch --partition shared --export=ALL,THETA_RUN_DIR=/abs/path/to/theta_run_dir,PLA_CRIT_STRETCH=0.01,FRACTURE_CRIT_STRETCH=0.005 \
  hpc/run_core_shell_from_theta_mesh.slurm
```

## Notes on the mirrored KOA_Peridigm_Install tests
From your mirror:
- `test/fragmenting_cylinder` and `test/fragmenting_ellipseoid_shell` ran successfully.
- `test/shell_cone_contact` failed due to missing node set registration (fixed here by adding `Cone Nodes: models/cone_nodes.txt`).
- `fragmenting_ellipsoid_contact_impact.yaml` failed because Peridigm's YAML parser choked on non-ASCII characters in comments (remove Unicode like `³` and emojis).
