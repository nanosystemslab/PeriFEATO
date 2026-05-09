# Stress Extraction Pipeline: From Mesh to Theta-Band Stress Data

## Overview

The stress extraction pipeline transforms finite element analysis (FEA) results into theta-band-specific stress values suitable for thickness optimization. The process preserves spatial correspondence between mesh cells and their geometric location on the shell, enabling targeted thickness adjustments in the optimization loop.

## 1. Mesh Generation with Theta-Band Tagging

The mesh generation phase creates a tetrahedral finite element mesh of the protective shell geometry while simultaneously computing and storing geometric metadata for each cell.

### Theta Band Computation

For each mesh cell, the centroid coordinates $(x_c, y_c, z_c)$ are computed. The theta angle $\theta$ is calculated as the angle from the z-axis (pole) to the radial direction:

$$\theta = \arctan\left(\frac{\sqrt{x_c^2 + y_c^2}}{|z_c|}\right)$$

The shell is discretized into 10 latitude bands spanning from $\theta = 9°$ (just below the pole hole) to $\theta = 90°$ (equator). Band edges are uniformly distributed:

$$\theta_{\text{edges}} = [9°, 17.1°, 25.2°, 33.3°, 41.4°, 49.5°, 57.6°, 65.7°, 73.8°, 81.9°, 90°]$$

Each cell is assigned a band index $i \in \{0, 1, ..., 9\}$ based on which interval contains its theta angle. Band $t_0$ is nearest the pole; band $t_9$ is at the equator.

### Hemisphere Tagging

To distinguish impact response on opposite sides of the shell, each cell is also tagged with a hemisphere indicator:

$$h = \begin{cases} 1 & \text{if } z_c \geq 0 \text{ (upper hemisphere)} \\ 0 & \text{if } z_c < 0 \text{ (lower hemisphere)} \end{cases}$$

### Output Format

The mesh is exported in XDMF/HDF5 format with cell data arrays:
- `theta_band`: Integer band index (0-9) for each cell
- `hemisphere`: Integer hemisphere indicator (0 or 1) for each cell
- Material properties and boundary condition markers

## 2. Finite Element Analysis

The FEA solver reads the tagged mesh and performs transient dynamic analysis of drop impact. For each impact orientation (vertical, tipped, horizontal), the solver:

1. Rotates the mesh to align the impact direction with the ground plane
2. Applies initial velocity conditions representing the drop
3. Solves the equations of motion using Newmark-beta time integration
4. Computes stress fields at each timestep

### Impact Orientation Convention

The orientation naming convention specifies which part of the shell contacts the ground. The theta angle ($\theta$) measures latitude from the +z pole, and phi ($\phi$) measures azimuth around the z-axis:

| Orientation | $\theta$ | $\phi$ | Impact Location |
|-------------|----------|--------|-----------------|
| `vertical_theta180_pole` | 180° | 0° | Pole (hole rim) contacts ground |
| `tipped_theta45_phi90_protected` | 45° | 90° | Mid-latitude protected zone |
| `tipped_theta45_phi0_fracture` | 45° | 0° | Mid-latitude fracture zone |
| `horizontal_theta90_phi90_protected` | 90° | 90° | Equator protected zone |
| `horizontal_theta90_phi0_fracture` | 90° | 0° | Equator fracture zone |

**Vertical Orientation Change:** The vertical load case was updated from `vertical_theta0_base` ($\theta=0°$) to `vertical_theta180_pole` ($\theta=180°$). This change reflects the physical scenario more accurately:

- **Previous definition** ($\theta=0°$): The shell's +z axis pointed upward, meaning the equator (base) contacted the ground. This was inconsistent with the "pole impact" intent.
- **Current definition** ($\theta=180°$): The shell is rotated so the pole (with the through-hole) faces downward toward the ground. The hole rim contacts the rigid ground plane, representing a pole-first vertical drop.

This convention ensures naming consistency: the orientation name indicates which geometric feature impacts the ground (pole, protected zone, or fracture zone)

### Stress Field Computation

At each timestep, the von Mises stress $\sigma_{vM}$ is computed for each cell from the Cauchy stress tensor:

$$\sigma_{vM} = \sqrt{\frac{1}{2}\left[(\sigma_1 - \sigma_2)^2 + (\sigma_2 - \sigma_3)^2 + (\sigma_3 - \sigma_1)^2\right]}$$

where $\sigma_1, \sigma_2, \sigma_3$ are the principal stresses.

### Maximum Stress Tracking

The solver tracks the maximum von Mises stress experienced by each cell across all timesteps:

$$\sigma_{\max,k} = \max_{t \in [0, T]} \sigma_{vM,k}(t)$$

where $k$ indexes mesh cells and $T$ is the simulation duration.

### Output Format

Results are written to VTK files containing:
- `max_stress`: Maximum von Mises stress per cell (Pa)
- `theta_band`: Band index (passed through from mesh)
- `hemisphere`: Hemisphere indicator (passed through from mesh)

## 3. Stress Extraction by Theta Band

The post-processing phase reads FEA results and aggregates stress values by theta band for use in the optimization loop.

### Reading Results

The VTK reader extracts three parallel arrays from the results file:
- Stress values: $\{\sigma_{\max,k}\}$ for all cells $k$
- Band indices: $\{b_k\}$ where $b_k \in \{0, ..., 9\}$
- Hemisphere flags: $\{h_k\}$ where $h_k \in \{0, 1\}$

### Band-wise Aggregation

For each theta band $i$ and hemisphere $h$, the stress values are aggregated:

$$S_{i,h} = \{\sigma_{\max,k} : b_k = i \text{ and } h_k = h\}$$

Two statistics are computed for each band-hemisphere pair:

**Maximum stress:**
$$\sigma_{\max}^{(i,h)} = \max(S_{i,h})$$

**95th percentile stress (P95):**
$$\sigma_{P95}^{(i,h)} = \text{percentile}(S_{i,h}, 95)$$

The P95 statistic is preferred for optimization because it filters out stress singularities at contact points while still capturing the representative stress state in each band.

### Multi-Orientation Aggregation

For worst-case design, stress values are aggregated across all impact orientations:

$$\sigma_{\text{design}}^{(i,h)} = \max_{o \in \text{orientations}} \sigma_{P95}^{(i,h,o)}$$

This ensures the optimized design satisfies stress constraints for all tested impact scenarios.

## 4. Integration with Thickness Optimization

The extracted band-wise stress values feed directly into the Fully Stressed Design (FSD) update rule:

$$t_i^{(n+1)} = t_i^{(n)} \cdot \left(\frac{\sigma_{\text{design}}^{(i)}}{\sigma_{\text{target}}}\right)^\alpha$$

where:
- $t_i^{(n)}$ is the thickness of band $i$ at iteration $n$
- $\sigma_{\text{target}} = \sigma_{\text{yield}} / \text{SF}$ is the target stress (yield stress divided by safety factor)
- $\alpha$ is the stress-thickness exponent (0.5 for bending-dominated shells)

The optimization loop iterates between FEA simulation and thickness updates until convergence, defined as:

$$\max_i |t_i^{(n+1)} - t_i^{(n)}| < \epsilon$$

where $\epsilon$ is the convergence tolerance (typically 0.05 mm).

## Data Flow Summary

```
┌─────────────────┐
│  Mesh Generator │
│  (theta_mesh)   │
└────────┬────────┘
         │ XDMF/H5 mesh with theta_band, hemisphere
         ▼
┌─────────────────┐
│   FEA Solver    │
│ (DOLFINx)       │
└────────┬────────┘
         │ VTK with max_stress, theta_band, hemisphere
         ▼
┌─────────────────┐
│  VTK Reader     │
│ (PyVista)       │
└────────┬────────┘
         │ Arrays: stress[], band[], hemisphere[]
         ▼
┌─────────────────┐
│ Band Aggregator │
│ (fea_adapter)   │
└────────┬────────┘
         │ stress_by_band_MPa[10], stress_p95_by_band_MPa[10]
         ▼
┌─────────────────┐
│   Thickness     │
│   Updater       │
└─────────────────┘
```

## Key Implementation Files

| Component | File | Function |
|-----------|------|----------|
| Mesh generation | `shell_geometry_generator_refined.py` | Computes theta_band, hemisphere per cell |
| FEA solver | `truth_quasistatic_contact.py` | Passes through band/hemisphere to output |
| VTK reader | `fea_vtk_reader.py` | Extracts stress and metadata arrays |
| Band aggregation | `fea_adapter.py` | `compute_stress_by_theta_band()` |
| Thickness update | `thickness_updater.py` | FSD update rule |
