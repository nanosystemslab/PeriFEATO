# Optimization Module

Shell thickness optimization framework combining FEA and Peridigm physics.

## Module Structure

```
modules/optimization/
├── __init__.py                # Module entry point
├── README.md                  # This file
├── src/
│   ├── __init__.py
│   ├── adapters/              # Unified interfaces for physics modules
│   │   ├── __init__.py
│   │   ├── fea_adapter.py     # FEA backend (simple, truth_contact)
│   │   ├── peridigm_adapter.py # Peridigm backend (mock, real)
│   │   ├── mesh_adapter.py    # Mesh generation (theta_mesh)
│   │   ├── fea_vtk_reader.py  # VTK utilities
│   │   └── fea_vtk_screenshot.py
│   └── shell_thickness/       # Thickness optimization algorithms
│       ├── __init__.py
│       ├── theta_band_params.py     # Thickness parameterization
│       ├── thickness_updater.py     # Update logic
│       ├── thickness_loop.py        # FEA-based optimization loop
│       └── dual_physics_loop.py     # FEA + Peridigm optimization loop
├── config/                    # Configuration files
│   ├── thickness_optimization_hpc.yaml    # FEA-only (HPC)
│   ├── dual_physics_hpc.yaml             # FEA + Peridigm (HPC)
│   └── dual_physics_local_test.yaml      # Local testing
├── hpc/                       # SLURM scripts for HPC execution
│   ├── run_thickness_optimization.slurm
│   └── run_dual_physics_optimization.slurm
└── tests/                     # Unit and integration tests
```

## Entry Points

All optimizations are run through the `scripts/` directory at the repository root:

### FEA-Based Thickness Optimization

```bash
# Local execution
python scripts/run_thickness_optimization.py \
    --config modules/optimization/config/thickness_optimization_hpc.yaml

# HPC execution
sbatch modules/optimization/hpc/run_thickness_optimization.slurm
```

### Dual-Physics Optimization (FEA + Peridigm)

```bash
# Local execution (mock Peridigm)
python scripts/run_dual_physics_optimization.py \
    --config modules/optimization/config/dual_physics_local_test.yaml

# HPC execution (real Peridigm)
sbatch modules/optimization/hpc/run_dual_physics_optimization.slurm
```

## Configuration

Configuration files are YAML-based and located in `config/`.

### Key Configuration Sections

#### 1. Mesh Generation

```yaml
mesh:
  backend: "theta_mesh"
  theta_config_path: "${HOME}/Optimization_Framework/modules/theta_mesh/config/base_config.yaml"
  output_dir: "${HOME}/scratch/meshes"
  theta_overrides:
    geometry:
      inner_radius_r_mm: 13.25
      inner_radius_z_mm: 34.0
    mesh:
      target_element_size_mm: 3.0
    fracture_design:
      fracture_planes_phi_deg: [0, 180]
      zone_width_deg: 10.0
      fracture_factor_theta: [0.6, 0.6, 0.6]
```

#### 2. FEA Configuration

```yaml
fea:
  backend: "truth_contact"  # or "simple"
  mode: "local"             # or "hpc"
  simulation_mode: "transient"
  use_worst_case_stress: true  # Test all 5 orientations

  # Transient dynamics
  time_end_s: 0.005
  timestep_s: 2.0e-5
  newmark_beta: 0.25
  newmark_gamma: 0.5
  damping_ratio: 0.03

  # Drop parameters
  drop_mass_kg: 0.045
  drop_height_m: 5.1

  # Stress constraint
  stress_limit_Pa: 44.0e6
```

#### 3. Peridigm Configuration

```yaml
peridigm:
  backend: "mock"  # or "real"
  mode: "local"    # or "hpc"

  # Impact parameters
  impact_velocity_m_s: 250.0

  # Material parameters
  pla_critical_stretch: 0.01
  fracture_zone_critical_stretch: 0.005

  # Damage thresholds
  fracture_damage_target: 0.8
  shell_damage_threshold: 0.2
```

#### 4. Optimization Parameters

```yaml
optimization:
  n_theta_bands: 3
  initial_thickness_mm: 1.0
  max_iterations: 5

  # Update parameters
  thickness_update_rate: 0.1
  safety_margin: 0.5  # Target SF=2.0

  # Dual-physics weights
  dual_physics_weights:
    stress: 1.0
    fracture_miss: 1.0
    shell_damage: 2.0
```

## Module Integration

### Adapters

The optimization module uses adapters to interface with physics modules:

- **FEA Adapter** (`adapters/fea_adapter.py`):
  - Routes to `modules/fea_truth_contact/` for production FEA
  - Supports local and HPC execution
  - Handles multi-orientation testing

- **Peridigm Adapter** (`adapters/peridigm_adapter.py`):
  - Routes to `modules/peridigm_fracture/` for production Peridigm
  - Mock backend for local testing
  - Real backend submits SLURM jobs on HPC

- **Mesh Adapter** (`adapters/mesh_adapter.py`):
  - Routes to `modules/theta_mesh/` for mesh generation
  - Regenerates mesh each iteration with updated thickness

### Dependencies

This module depends on:
- `modules/fea_truth_contact/` - Transient FEA with contact mechanics
- `modules/peridigm_fracture/` - Peridigm fracture simulation (HPC)
- `modules/theta_mesh/` - Shell geometry generation

## Workflow

### FEA-Only Optimization

1. Initialize thickness parameters (uniform or from previous run)
2. For each iteration:
   - Generate mesh with current thickness
   - Run FEA for all 5 drop orientations
   - Find worst-case stress
   - Update thickness to target SF=2.0 (22 MPa stress limit)
   - Check convergence
3. Save optimized thickness and history

### Dual-Physics Optimization

1. Initialize thickness parameters
2. For each iteration:
   - Generate mesh with current thickness
   - **Parallel execution**:
     - Run FEA for all 5 drop orientations (stress-based)
     - Run Peridigm high-velocity fracture test (damage-based)
   - Compute weighted fitness:
     - Stress violation (minimize)
     - Fracture damage miss (minimize - want high damage in fracture zones)
     - Shell damage excess (minimize - want low damage in main body)
   - Update thickness based on combined objectives
   - Check convergence
3. Save optimized thickness and history

## Output

All results are saved to the directory specified in the config (`output.dir`):

```
output_dir/
├── history.json              # Convergence history
├── thickness_final.json      # Final optimized thickness
├── thickness_iter001.json    # Iteration snapshots
├── thickness_iter002.json
└── ...
```

### History Format

```json
[
  {
    "iteration": 1,
    "thickness_mm": [1.0, 1.0, 1.0],
    "max_stress_MPa": 25.2,
    "worst_case_orientation": "vertical_theta0_base",
    "D_fracture_zone": 0.75,
    "D_shell_body": 0.12,
    "fitness_total": 0.234,
    "converged": false
  },
  ...
]
```

## Testing

Run integration tests before HPC deployment:

```bash
python modules/optimization/tests/test_integration.py
```

## HPC Deployment

1. Sync code to HPC:
   ```bash
   rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
       modules/ mtdsn@koa.its.hawaii.edu:~/Optimization_Framework/modules/
   rsync -avz scripts/ mtdsn@koa.its.hawaii.edu:~/Optimization_Framework/scripts/
   ```

2. SSH to HPC:
   ```bash
   ssh mtdsn@koa.its.hawaii.edu
   cd ~/Optimization_Framework
   ```

3. Submit job:
   ```bash
   sbatch modules/optimization/hpc/run_thickness_optimization.slurm
   # or
   sbatch modules/optimization/hpc/run_dual_physics_optimization.slurm
   ```

4. Monitor:
   ```bash
   squeue -u mtdsn
   tail -f logs/thickness_opt_<jobid>.log
   ```

5. Retrieve results:
   ```bash
   # On local machine
   rsync -avz mtdsn@koa.its.hawaii.edu:~/scratch/thickness_opt_<date>_<jobid>/ ./results/
   ```

## Architecture Notes

### Why Module Framework?

- **Separation of concerns**: Each module handles one physics type
- **Reusability**: Adapters allow mixing different physics backends
- **Testability**: Can test optimization loop with mock physics
- **Scalability**: Easy to add new physics modules (e.g., thermal, manufacturing)

### Adapter Pattern Benefits

- Configuration-driven backend selection
- No code changes to swap physics implementations
- Fallback mechanisms (mock → real)
- Consistent interfaces across modules

## Version History

- **1.0.0** (2026-01-20): Initial module framework release
  - FEA-based thickness optimization
  - Dual-physics (FEA + Peridigm) optimization
  - Adapter pattern for module integration
  - HPC support via SLURM

## Contact

Questions or issues? Check:
- Documentation: `docs/` directory at repository root
- Issues: GitHub issues or contact maintainer
