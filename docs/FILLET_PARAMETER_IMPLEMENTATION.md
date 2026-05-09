# Hole Fillet Parameter Implementation

## Goal

Add an optimizable fillet radius parameter to the hole rim to reduce stress concentration at the pole during vertical impact.

## Background

The optimization found that pole stress (~120 MPa P95) cannot be reduced by increasing thickness alone. The stress concentration is geometric - caused by the sharp edge where the cylindrical hole intersects the ellipsoidal shell surface. A fillet (rounded edge) would distribute stress over a larger area.

## Current Implementation

**File:** `modules/theta_mesh/src/shell_geometry_generator_refined.py`

The hole is created at line 725:
```python
hole = occ.addCylinder(0.0, 0.0, -z_extent, 0.0, 0.0, 2.0 * z_extent, self.hole_radius)
```

Then boolean cuts are performed (lines 756-816):
```python
steel_out, _ = occ.cut([(3, core_for_steel)], [(3, hole)], ...)
prot_cut, _ = occ.cut([(3, v) for v in prot_vols_pre_hole], [(3, hole)], ...)
frac_cut, _ = occ.cut([(3, v) for v in frac_vols_pre_hole], [(3, hole)], ...)
```

## Implementation Steps

### 1. Add Configuration Parameter

In `modules/theta_mesh/config/base_config.yaml`:
```yaml
geometry:
  hole_radius_mm: 2.0
  hole_fillet_radius_mm: 0.5  # NEW: fillet radius at hole rim (0 = sharp edge)
```

### 2. Read Parameter in Generator

In `shell_geometry_generator_refined.py` `__init__`:
```python
self.hole_fillet_radius = geom.get('hole_fillet_radius_mm', 0.0) / 1000  # m
```

### 3. Apply Fillet After Boolean Operations

After the boolean cuts that create the hole, apply fillet to the rim edges. Add after line ~816:

```python
# Apply fillet to hole rim edges if specified
if self.hole_fillet_radius > 0:
    occ.synchronize()

    # Find edges at the hole rim
    # These are circular edges at radius ~= hole_radius near z = c_core (pole)
    hole_rim_edges = []

    for vol_tag in prot_final + frac_final + steel_vols:
        # Get all edges of this volume
        edges = gmsh.model.getBoundary([(3, vol_tag)], combined=False, oriented=False, recursive=True)
        edges = [(d, t) for d, t in edges if d == 1]  # Only edges (dim=1)

        for dim, edge_tag in edges:
            # Get edge bounding box
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, edge_tag)

            # Check if edge is near the pole (z ~ c_core) and at hole radius
            z_center = (zmin + zmax) / 2
            r_min = np.sqrt(xmin**2 + ymin**2)
            r_max = np.sqrt(xmax**2 + ymax**2)

            # Edge is at hole rim if:
            # 1. Near the pole (z close to c_core)
            # 2. At approximately hole_radius
            if abs(z_center - self.c_core) < 0.005:  # within 5mm of pole
                if abs(r_min - self.hole_radius) < 0.002 or abs(r_max - self.hole_radius) < 0.002:
                    hole_rim_edges.append(edge_tag)

    # Remove duplicates
    hole_rim_edges = list(set(hole_rim_edges))

    if hole_rim_edges:
        print(f"  Applying fillet (r={self.hole_fillet_radius*1000:.2f}mm) to {len(hole_rim_edges)} hole rim edges")
        try:
            occ.fillet(hole_rim_edges, [self.hole_fillet_radius] * len(hole_rim_edges))
        except Exception as e:
            print(f"  Warning: Fillet failed - {e}. Continuing with sharp edges.")

    occ.synchronize()
```

### 4. Add to Optimization Parameters

In `modules/optimization/config/dual_physics_optimization_hpc.yaml`:
```yaml
mesh:
  theta_overrides:
    geometry:
      hole_fillet_radius_mm: 0.5  # Can be optimized
```

To make it optimizable, add to `ThetaBandParams` class or create a new `GeometryParams` class.

## Challenges

1. **Edge Identification**: After boolean operations, edge tags change. Need robust filtering by geometry (position/radius) not tags.

2. **Fillet Failure**: GMSH's fillet can fail if radius is too large relative to geometry. Need error handling and constraints.

3. **Mesh Quality**: Large fillets may require finer mesh at the fillet region to capture curvature.

4. **Multiple Materials**: The fillet affects both PLA shell and steel core at the hole. Both need consistent fillet.

## Testing

1. Generate mesh with `hole_fillet_radius_mm: 0.5`
2. Visualize in ParaView - check for smooth transition at hole rim
3. Run single FEA at vertical orientation
4. Compare P95 stress at pole bands with/without fillet

## Expected Impact

A 0.5mm fillet on a 2mm radius hole should reduce stress concentration factor from ~3-4 (sharp edge) to ~1.5-2 (rounded). This could reduce pole stress from 120 MPa to 60-80 MPa, making SF=2.0 achievable with reasonable thickness.

## Alternative: Chamfer

If fillet is too complex, a chamfer (angled cut) is simpler:
```python
# Create chamfer tool - a cone that cuts the sharp edge
chamfer_cone = occ.addCone(0, 0, z_pole - chamfer_depth,
                            0, 0, chamfer_depth,
                            hole_radius + chamfer_depth, hole_radius)
occ.cut(shell_volumes, [(3, chamfer_cone)], ...)
```

This creates a 45° chamfer which also reduces stress concentration, though not as effectively as a fillet.
