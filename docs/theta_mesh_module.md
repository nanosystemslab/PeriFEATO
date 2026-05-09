# Theta Mesh Module (t(theta) sweep)

This module generates **theta-banded** (t(θ)) core+shell meshes on KOA using a Slurm array job.

## What it does

- Uses `shell_geometry_generator_refined.py` to build a steel core + PLA shell with **piecewise-constant thickness vs θ**
  (θ measured from +z; 0°=pole, 90°=equator).
- Produces per-case outputs in a sweep directory:
  - `results_refined/core_shell_refined.msh`
  - `results_refined/core_shell_refined.vtu`
  - `results_refined/core_shell_refined.xdmf`
  - `results_refined/peridigm/core_shell_refined_peridigm.txt` (tet centroids + volumes)
  - `results_refined/peridigm/nodeset_*.txt` (1-based point ids for Peridigm BCs)
- Writes diagnostic cell-data for visualization:
  - `material_id` (1=steel, 2=PLA)
  - `region_id` (Gmsh physical group id)
  - `thickness_mm` (diagnostic only)

## Key files

- Entry script (KOA): `hpc_deployment/start_theta_mesh_sweep.sh`
- Array job: `hpc_deployment/generate_theta_mesh_sweep_array.slurm`
- Example manifest: `hpc_deployment/manifests/theta_mesh_manifest.example.yaml`
- Local debug helper: `hpc_deployment/local_try_mesh_brep.py`

## Run (KOA)

```bash
cd ~/hpc_deployment
bash start_theta_mesh_sweep.sh --partition shared
```

This creates `~/hpc_deployment/mesh_sweeps/sweep_<timestamp>/` and submits an array job.

## Edit cases

Edit the generated `manifest.yaml` in the sweep directory, then re-submit:

```bash
SWEEP=~/hpc_deployment/mesh_sweeps/sweep_YYYYMMDD_HHMMSS
${EDITOR:-vi} "$SWEEP/manifest.yaml"
```

## Common failure mode: “no dim=3 volumes”

If `mesh_fail.brep` opens with only surfaces (no volumes), tetra meshing cannot succeed.
This can happen when OpenCASCADE healing turns solids into shells.

Default is now:
- `mesh.occ_heal: false`

## Local debug

If KOA writes `mesh_fail.brep`:

```bash
python3 hpc_deployment/local_try_mesh_brep.py path/to/mesh_fail.brep
```
