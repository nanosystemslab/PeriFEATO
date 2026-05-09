# fea_truth_contact module

Reusable, self-contained copy of the **truth-contact FEA** workflow (DOLFINx) used on KOA.

## Contents
- `src/truth_quasistatic_contact.py`: solver + outputs (`.bp`/VTX or `.pvd`/VTK)
- `config/base_config.yaml`: baseline solver config (materials, defaults)
- `hpc/truth_sandbox_mpi_job.slurm`: Slurm job script (MPI inside container)
- `hpc/submit_truth_jobs.sh`: submit one job per orientation
- `hpc/submit_truth_for_sweep.sh`: submit one job per sweep mesh (runs default orientations)
- `tools/collect_truth_metrics.py`: aggregate `*_metrics.yaml` into a CSV
- `tools/evaluate_truth_constraint.py`: check max von Mises constraint and write a report
- `tools/pull_sweep_from_koa.sh`: pull sweep logs+meshes+truth outputs to laptop
- `tools/render_max_vm_sweep.sh`: batch screenshots via ParaView `pvpython`

## KOA quickstart
On KOA, from the directory where you want results:

```bash
bash hpc/submit_truth_jobs.sh --partition shared --ranks 8 --output-format vtx \
  vertical_theta0_base \
  tipped_theta45_phi90_protected \
  horizontal_theta90_phi90_protected \
  horizontal_theta90_phi0_fracture \
  horizontal_theta90_phi10_edge
```

For a theta-mesh sweep dir (one job per mesh case):

```bash
bash hpc/submit_truth_for_sweep.sh --sweep-dir /path/to/sweep_YYYYMMDD_HHMMSS \
  --partition shared --ranks 8 --output-format vtx --orientations default
```

## Laptop visualization
After pulling results locally:

```bash
bash tools/render_max_vm_sweep.sh --root results_from_koa/sweep_YYYYMMDD_HHMMSS
```
