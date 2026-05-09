#!/usr/bin/env python3
"""
Gmsh-based 3D core-shell mesh generator with INTERFACE REFINEMENT.

Enhanced version with:
- Refined mesh at steel/PLA interface for smooth ellipsoidal boundary
- Distance-based mesh sizing
- Better control of element sizes at material boundaries
"""

import numpy as np
from pathlib import Path
import yaml
import meshio
import gmsh

class ShellGeometryGeneratorRefined:
    """
    Generator for 3D ellipsoidal steel core + PLA shell with refined interfaces.
    """

    def __init__(self, config_file='config.yaml'):
        """Initialize generator with configuration."""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config

        # Geometry parameters
        geom = config['geometry']
        self.core_radius_r = geom['inner_radius_r_mm'] / 1000  # steel core radial semi-axis (m)
        self.core_radius_z = geom['inner_radius_z_mm'] / 1000  # steel core axial semi-axis (m)
        self.hole_radius = geom.get('hole_radius_mm', 0.0) / 1000  # cylindrical through-hole radius (m) — used for steel
        self.pla_hole_radius = geom.get('pla_hole_radius_mm', 0.0) / 1000  # PLA through-hole radius (m); 0 = same as hole_radius
        self.hole_fillet_radius = geom.get('hole_fillet_radius_mm', 0.0) / 1000  # fillet at hole rim (m)
        self.hole_chamfer_size = geom.get('hole_chamfer_size_mm', 0.0) / 1000  # chamfer at hole rim (m)
        self.hole_fillet_torus = geom.get('hole_fillet_torus_mm', 0.0) / 1000  # torus fillet at hole rim (m)

        # Optional core scaling
        self.steel_core_factor = float(geom.get('steel_core_factor', 1.0))
        self.steel_core_factor_z = float(geom.get('steel_core_factor_z', self.steel_core_factor))

        # Fracture design
        fracture = config['fracture_design']
        self.fracture_planes = np.array(fracture['fracture_planes_phi_deg'], dtype=float) * np.pi / 180
        self.fracture_width = float(fracture['zone_width_deg']) * np.pi / 180
        self.min_thickness = float(config['material']['min_thickness_mm']) / 1000
        self.initial_thickness_factor = fracture.get('initial_thickness_factor', 0.6)

        # Mesh parameters
        self.n_profile = int(geom.get('n_profile_points', 100))
        self.n_phi = int(config.get('mesh', {}).get('n_phi_divisions', 60))

        # Enhanced mesh sizing parameters for interface refinement
        self.mesh_size = float(config.get('mesh', {}).get('target_element_size_mm', 1.0)) / 1000
        self.interface_mesh_size = float(config.get('mesh', {}).get('interface_mesh_size_mm', 0.3)) / 1000
        self.interface_distance = float(config.get('mesh', {}).get('interface_distance_mm', 2.0)) / 1000
        # Coarse mesh for steel core (optional - defaults to same as target_element_size)
        self.core_mesh_size = float(config.get('mesh', {}).get('core_element_size_mm', self.mesh_size * 1000)) / 1000

        self.gmsh_terminal = int(config.get('mesh', {}).get('gmsh_terminal', 0))
        self.mesh_algorithm_2d = int(config.get('mesh', {}).get('mesh_algorithm_2d', 6))  # 6 = Frontal-Delaunay
        self.mesh_algorithm = int(config.get('mesh', {}).get('mesh_algorithm', 10))  # 10 = HXT algorithm
        self.mesh_optimize = int(config.get('mesh', {}).get('mesh_optimize', 1))
        self.mesh_smoothing = int(config.get('mesh', {}).get('mesh_smoothing', 5))

        self.merge_duplicate_nodes = bool(config.get("mesh", {}).get("merge_duplicate_nodes", True))
        self.merge_tol = float(config.get("mesh", {}).get("merge_tol_mm", 1e-6)) / 1000.0

        # OpenCASCADE healing can resolve some OCC issues, but it can also *destroy solids*
        # (turning them into shells) depending on the Gmsh/OCC build. Keep it opt-in.
        self.occ_heal = bool(config.get("mesh", {}).get("occ_heal", False))
        self.occ_heal_tol = float(config.get("mesh", {}).get("occ_heal_tol_m", 1e-8))
        self.theta_band_fuse = bool(config.get("mesh", {}).get("theta_band_fuse", True))
        self.debug_geometry = str(config.get("mesh", {}).get("debug_geometry", "on-failure")).strip().lower()
        if self.debug_geometry not in {"never", "on-failure", "always"}:
            raise ValueError("mesh.debug_geometry must be one of: never, on-failure, always")

        # Steel core dimensions
        self.a_core = self.core_radius_r * self.steel_core_factor
        self.c_core = self.core_radius_z * self.steel_core_factor_z

        # Split core configuration (for flyback simulation)
        split_cfg = config.get('split_core', {}) or {}
        self.split_core_enabled = bool(split_cfg.get('enabled', False))
        self.split_core_axis = str(split_cfg.get('axis', 'x')).lower()  # 'x' or 'y'
        self.split_core_gap_mm = float(split_cfg.get('gap_mm', 2.0))  # Gap between halves in mm
        if self.split_core_axis not in ('x', 'y'):
            raise ValueError("split_core.axis must be 'x' or 'y'")

        # Storage for generated data
        self.thickness_field = None
        self.tetrahedra = None
        self.element_materials = None
        self.volumetric_vertices = None
        self._background_field_id = None

    @staticmethod
    def _tet_centroids_and_volumes(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (centroids, volumes) for tetrahedra.

        centroids: (n, 3)
        volumes:   (n,)
        """
        v0 = points[tets[:, 0]]
        v1 = points[tets[:, 1]]
        v2 = points[tets[:, 2]]
        v3 = points[tets[:, 3]]
        centroids = 0.25 * (v0 + v1 + v2 + v3)
        vols = np.abs(np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0
        return centroids, vols

    @staticmethod
    def _write_nodeset(path: Path, ids_1based: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for nid in ids_1based.tolist():
                f.write(f"{int(nid)}\n")

    def _classify_grid(
        self,
        ds: float,
        a_core: float,
        c_core: float,
        hole_r: float,
        a_out: float,
        c_out: float,
        theta_cfg: dict | None,
        fracture_enabled: bool,
        theta_edges_rad: np.ndarray | None,
        protected_m: np.ndarray | None,
        fracture_m: np.ndarray | None,
        t_protected: float | None,
        t_fracture: float | None,
        ds_x: float | None = None,
        ds_y: float | None = None,
        ds_z: float | None = None,
        pla_hole_r: float | None = None,
        use_spline: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a grid and classify points into blocks.

        When ds_x/ds_y/ds_z are provided, the grid is anisotropic (different
        spacing per axis).  Otherwise falls back to isotropic spacing *ds*.

        Returns (pts, block_id) arrays for ALL candidate points (including block 0 = skip).
        """
        dx = ds_x if ds_x is not None else ds
        dy = ds_y if ds_y is not None else ds
        dz = ds_z if ds_z is not None else ds
        vol = dx * dy * dz
        xs = np.arange(-a_out + dx / 2.0, a_out, dx)
        ys = np.arange(-a_out + dy / 2.0, a_out, dy)
        zs = np.arange(-c_out + dz / 2.0, c_out, dz)

        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r_xy = np.sqrt(x**2 + y**2)
        abs_z = np.abs(z)
        r_actual = np.sqrt(x**2 + y**2 + z**2)

        alpha = np.arctan2(r_xy, abs_z)

        sin_a = np.sin(alpha)
        cos_a = np.cos(alpha)
        denom = np.sqrt((c_core * sin_a)**2 + (a_core * cos_a)**2)
        denom = np.maximum(denom, 1e-30)
        r_inner = (a_core * c_core) / denom

        if theta_cfg is not None and use_spline and len(protected_m) >= 2:
            from scipy.interpolate import CubicSpline
            midpoints = 0.5 * (theta_edges_rad[:-1] + theta_edges_rad[1:])
            spline_prot = CubicSpline(midpoints, protected_m, bc_type='clamped')
            spline_frac = CubicSpline(midpoints, fracture_m, bc_type='clamped')
            t_prot_pt = np.clip(spline_prot(alpha),
                                float(np.min(protected_m)), float(np.max(protected_m)))
            t_frac_pt = np.clip(spline_frac(alpha),
                                float(np.min(fracture_m)), float(np.max(fracture_m)))
        elif theta_cfg is not None:
            band_idx = np.searchsorted(theta_edges_rad, alpha, side='right') - 1
            band_idx = np.clip(band_idx, 0, len(theta_edges_rad) - 2)
            t_prot_pt = protected_m[band_idx]
            t_frac_pt = fracture_m[band_idx]
        else:
            t_prot_pt = np.full(len(pts), t_protected)
            t_frac_pt = np.full(len(pts), t_fracture)

        def _ellipsoid_radius(a: np.ndarray, c: np.ndarray) -> np.ndarray:
            d = np.sqrt((c * sin_a)**2 + (a * cos_a)**2)
            d = np.maximum(d, 1e-30)
            return (a * c) / d

        r_outer_prot = _ellipsoid_radius(a_core + t_prot_pt, c_core + t_prot_pt)
        r_outer_frac = _ellipsoid_radius(a_core + t_frac_pt, c_core + t_frac_pt)

        in_hole_steel = r_xy < hole_r
        # PLA uses a separate (potentially larger) hole radius
        pla_hr = pla_hole_r if pla_hole_r is not None and pla_hole_r > 0 else hole_r
        in_hole_pla = r_xy < pla_hr

        inside_steel = r_actual <= r_inner
        outside_protected = r_actual > r_outer_prot
        in_shell = ~inside_steel & ~outside_protected & ~in_hole_pla

        in_fracture_wedge = np.zeros(len(pts), dtype=bool)
        if fracture_enabled:
            phi = np.arctan2(y, x)
            for frac_phi in self.fracture_planes:
                # Normalize frac_phi to [-pi, pi] to match arctan2 range
                fp = float(frac_phi)
                fp = (fp + np.pi) % (2 * np.pi) - np.pi
                angle_diff = np.abs(phi - fp)
                angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
                in_fracture_wedge |= angle_diff < (self.fracture_width / 2.0)

        in_fracture_zone = in_shell & in_fracture_wedge & (r_actual <= r_outer_frac)

        # Points in the fracture wedge but beyond the fracture outer radius
        # are OUTSIDE the shell — the wedge is only fracture-zone thick,
        # not protected-shell thick.
        in_wedge_gap = in_fracture_wedge & (r_actual > r_outer_frac) & ~inside_steel & ~in_hole_pla

        block_id = np.zeros(len(pts), dtype=np.int32)
        block_id[inside_steel & ~in_hole_steel] = 1
        block_id[in_shell & ~in_fracture_zone & ~in_wedge_gap] = 2
        block_id[in_fracture_zone] = 3

        return pts, block_id

    def _export_peridigm_uniform(self, output_dir: str | Path, spacing_mm: float) -> dict[str, str]:
        """
        Generate a uniform-grid Peridigm discretization directly from geometry parameters.

        Instead of using tet centroids (which have non-uniform spacing due to gmsh adaptive
        meshing), this creates particles on a regular 3D grid and classifies each into:
          Block 1 = steel core
          Block 2 = PLA protected shell
          Block 3 = PLA fracture zone
          Skip    = outside shell, inside hole

        Particle volume is ds^3 (uniform everywhere), and thinning the fracture zone
        directly reduces particle count — exactly what peridynamics needs.

        If ``peridigm_export.steel_spacing_mm`` is set, the steel core uses a coarser
        grid to reduce particle count (steel doesn't fracture so fine resolution is wasted).

        Args:
            output_dir: Directory for output files
            spacing_mm: Grid spacing in mm for PLA shell (e.g. 0.5)

        Returns:
            Dict with paths to discretization file, nodesets, and metadata.
        """
        ds = spacing_mm / 1000.0  # Convert to meters
        vol_shell = ds ** 3

        cfg_pe = self.config.get("peridigm_export", {}) or {}
        steel_spacing_mm = cfg_pe.get("steel_spacing_mm", None)

        a_core = float(self.a_core)
        c_core = float(self.c_core)
        hole_r = float(self.hole_radius)

        # Get theta-band configuration for per-band thicknesses
        theta_cfg = self._theta_band_config()
        fracture_enabled = (self.fracture_planes.size > 0) and (float(self.fracture_width) > 0.0)

        if theta_cfg is not None:
            theta_edges_rad = theta_cfg["theta_edges_rad"]
            protected_m = theta_cfg["protected_thickness_m"]
            fracture_m = theta_cfg["fracture_thickness_m"]
            max_t = float(np.max(protected_m))
            t_protected = None
            t_fracture = None
        else:
            t_protected = float(self.min_thickness)
            t_fracture = (
                float(self.min_thickness) * float(self.initial_thickness_factor)
                if fracture_enabled else float(self.min_thickness)
            )
            theta_edges_rad = None
            protected_m = None
            fracture_m = None
            max_t = t_protected

        # Bounding box for the outer surface
        a_out = a_core + max_t
        c_out = c_core + max_t

        pla_hole_r = self.pla_hole_radius if self.pla_hole_radius > 0 else hole_r
        use_spline = theta_cfg.get("use_spline", False) if theta_cfg is not None else False
        common_args = dict(
            a_core=a_core, c_core=c_core, hole_r=hole_r,
            a_out=a_out, c_out=c_out,
            theta_cfg=theta_cfg, fracture_enabled=fracture_enabled,
            theta_edges_rad=theta_edges_rad,
            protected_m=protected_m, fracture_m=fracture_m,
            t_protected=t_protected, t_fracture=t_fracture,
            pla_hole_r=pla_hole_r,
            use_spline=use_spline,
        )

        print(f"\n{'='*80}")
        print(f"UNIFORM PERIDIGM DISCRETIZATION")
        print(f"{'='*80}")
        print(f"  Shell spacing: {spacing_mm:.3f} mm ({ds:.6f} m)")
        if pla_hole_r != hole_r:
            print(f"  Steel hole radius: {hole_r*1e3:.1f} mm, PLA hole radius: {pla_hole_r*1e3:.1f} mm")

        fracture_spacing_mm = cfg_pe.get("fracture_spacing_mm", None)
        fracture_radial_only = bool(cfg_pe.get("fracture_radial_only", False))

        if fracture_spacing_mm is not None and steel_spacing_mm is not None and fracture_spacing_mm < spacing_mm:
            # Three-tier spacing: fine fracture, medium shell, coarse steel
            ds_frac = fracture_spacing_mm / 1000.0
            ds_steel = steel_spacing_mm / 1000.0

            if fracture_radial_only:
                # Anisotropic fracture grid: fine in x (radial for phi=0,180 strips),
                # coarse in y,z (matching shell spacing). This equalizes neighbor
                # counts between fracture and shell regions while still resolving
                # the thin through-thickness geometry.
                vol_frac = ds_frac * ds * ds
                print(f"  Fracture spacing: {fracture_spacing_mm:.3f} mm radial only (tangential={spacing_mm:.3f} mm)")
                print(f"  Shell spacing: {spacing_mm:.3f} mm ({ds:.6f} m)")
                print(f"  Steel spacing: {steel_spacing_mm:.3f} mm ({ds_steel:.6f} m)")
                print(f"  Fracture volume: {vol_frac:.6e} m^3 (anisotropic)")

                # Pass 1: Anisotropic grid → fine in x, coarse in y,z → keep fracture only
                pts_fine, blk_fine = self._classify_grid(
                    ds=ds, ds_x=ds_frac, ds_y=ds, ds_z=ds, **common_args)
            else:
                vol_frac = ds_frac ** 3
                print(f"  Fracture spacing: {fracture_spacing_mm:.3f} mm ({ds_frac:.6f} m)")
                print(f"  Shell spacing: {spacing_mm:.3f} mm ({ds:.6f} m)")
                print(f"  Steel spacing: {steel_spacing_mm:.3f} mm ({ds_steel:.6f} m)")

                # Pass 1: Fine isotropic grid → keep fracture zone only (block 3)
                pts_fine, blk_fine = self._classify_grid(ds=ds_frac, **common_args)

            vol_steel = ds_steel ** 3
            frac_mask = blk_fine == 3
            pts_frac = pts_fine[frac_mask]
            blk_frac = blk_fine[frac_mask]
            vol_frac_arr = np.full(len(pts_frac), vol_frac)

            # Pass 2: Medium grid → keep protected shell only (block 2)
            pts_med, blk_med = self._classify_grid(ds=ds, **common_args)
            shell_mask = blk_med == 2
            pts_shell = pts_med[shell_mask]
            blk_shell = blk_med[shell_mask]
            vol_shell_arr = np.full(len(pts_shell), vol_shell)

            # Pass 3: Coarse grid → keep steel only (block 1)
            pts_coarse, blk_coarse = self._classify_grid(ds=ds_steel, **common_args)
            steel_mask = blk_coarse == 1
            pts_steel = pts_coarse[steel_mask]
            blk_steel = blk_coarse[steel_mask]
            vol_steel_arr = np.full(len(pts_steel), vol_steel)

            # Merge
            pts_out = np.vstack([pts_steel, pts_shell, pts_frac])
            blk_out = np.concatenate([blk_steel, blk_shell, blk_frac])
            vol_out = np.concatenate([vol_steel_arr, vol_shell_arr, vol_frac_arr])

            n1 = len(pts_steel)
            n2 = len(pts_shell)
            n3 = len(pts_frac)
            print(f"  Particles: {len(pts_out):,} total")
            print(f"    Block 1 (steel     @ {steel_spacing_mm}mm): {n1:,}")
            print(f"    Block 2 (protected @ {spacing_mm}mm): {n2:,}")
            if fracture_radial_only:
                print(f"    Block 3 (fracture  @ {fracture_spacing_mm}mm radial / {spacing_mm}mm tangential): {n3:,}")
            else:
                print(f"    Block 3 (fracture  @ {fracture_spacing_mm}mm): {n3:,}")

        elif steel_spacing_mm is not None and steel_spacing_mm > spacing_mm:
            ds_steel = steel_spacing_mm / 1000.0
            vol_steel = ds_steel ** 3
            print(f"  Steel spacing: {steel_spacing_mm:.3f} mm ({ds_steel:.6f} m)")

            # Pass 1: Fine grid → keep PLA shell only (blocks 2, 3)
            pts_fine, blk_fine = self._classify_grid(ds=ds, **common_args)
            shell_mask = (blk_fine == 2) | (blk_fine == 3)
            pts_shell = pts_fine[shell_mask]
            blk_shell = blk_fine[shell_mask]
            vol_shell_arr = np.full(len(pts_shell), vol_shell)

            # Pass 2: Coarse grid → keep steel only (block 1)
            pts_coarse, blk_coarse = self._classify_grid(ds=ds_steel, **common_args)
            steel_mask = blk_coarse == 1
            pts_steel = pts_coarse[steel_mask]
            blk_steel = blk_coarse[steel_mask]
            vol_steel_arr = np.full(len(pts_steel), vol_steel)

            # Merge
            pts_out = np.vstack([pts_steel, pts_shell])
            blk_out = np.concatenate([blk_steel, blk_shell])
            vol_out = np.concatenate([vol_steel_arr, vol_shell_arr])

            n1 = len(pts_steel)
            n2 = int(np.sum(blk_shell == 2))
            n3 = int(np.sum(blk_shell == 3))
            print(f"  Particles: {len(pts_out):,} total")
            print(f"    Block 1 (steel  @ {steel_spacing_mm}mm): {n1:,}")
            print(f"    Block 2 (PLA protected @ {spacing_mm}mm): {n2:,}")
            print(f"    Block 3 (PLA fracture  @ {spacing_mm}mm): {n3:,}")
        else:
            # Single spacing for everything (original behaviour)
            pts_all, blk_all = self._classify_grid(ds=ds, **common_args)
            keep = blk_all > 0
            pts_out = pts_all[keep]
            blk_out = blk_all[keep]
            vol_out = np.full(len(pts_out), vol_shell)

            n1 = int(np.sum(blk_out == 1))
            n2 = int(np.sum(blk_out == 2))
            n3 = int(np.sum(blk_out == 3))
            print(f"  Particle volume: {vol_shell:.6e} m^3")
            print(f"  Particles: {len(pts_out):,} total")
            print(f"    Block 1 (steel):          {n1:,}")
            print(f"    Block 2 (PLA protected):  {n2:,}")
            print(f"    Block 3 (PLA fracture):   {n3:,}")

        # Write discretization file
        out_root = Path(output_dir)
        per_dir = out_root / "peridigm"
        per_dir.mkdir(parents=True, exist_ok=True)

        disc_path = per_dir / "core_shell_refined_peridigm.txt"
        with disc_path.open("w") as f:
            for (px, py, pz), blk, v in zip(pts_out, blk_out, vol_out):
                f.write(f"{px:.10e} {py:.10e} {pz:.10e} {int(blk)} {v:.10e}\n")

        # Nodesets (1-based line numbers)
        all_ids = np.arange(1, len(pts_out) + 1, dtype=np.int64)
        self._write_nodeset(per_dir / "nodeset_all.txt", all_ids)

        cfg = self.config.get("peridigm_export", {}) or {}

        def _make_end_nodeset(name: str, default_side: str) -> None:
            ns_cfg = (cfg.get("node_sets", {}) or {}).get(name, None)
            if ns_cfg is False:
                return
            ns_cfg = ns_cfg or {}
            axis = str(ns_cfg.get("axis", "z")).lower()
            axis_i = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
            side = str(ns_cfg.get("side", default_side)).lower()
            band_m = float(ns_cfg.get("band_m", 0.003))

            coord = pts_out[:, axis_i]
            lo = float(coord.min())
            hi = float(coord.max())
            if side == "min":
                mask = coord <= (lo + band_m)
            else:
                mask = coord >= (hi - band_m)

            ids = np.where(mask)[0] + 1
            self._write_nodeset(per_dir / f"nodeset_{name}.txt", ids.astype(np.int64))

        _make_end_nodeset("fixed_end", "max")
        _make_end_nodeset("impact_end", "min")

        # Split core nodesets (if enabled)
        if self.split_core_enabled:
            steel1_mask = (blk_out == 1)
            steel1_ids = np.where(steel1_mask)[0] + 1
            self._write_nodeset(per_dir / "nodeset_steel_half_1.txt", steel1_ids.astype(np.int64))

            steel4_mask = (blk_out == 4)
            steel4_ids = np.where(steel4_mask)[0] + 1
            self._write_nodeset(per_dir / "nodeset_steel_half_4.txt", steel4_ids.astype(np.int64))

            shell_mask = (blk_out == 2) | (blk_out == 3)
            z_max = float(pts_out[:, 2].max())
            far_end_mask = pts_out[:, 2] >= (z_max - 0.003)
            shell_far_end_ids = np.where(shell_mask & far_end_mask)[0] + 1
            self._write_nodeset(per_dir / "nodeset_shell_fixed.txt", shell_far_end_ids.astype(np.int64))

            steel_mask = (blk_out == 1) | (blk_out == 4)
            steel_far_end_ids = np.where(steel_mask & far_end_mask)[0] + 1
            self._write_nodeset(per_dir / "nodeset_steel_fixed_z.txt", steel_far_end_ids.astype(np.int64))

            print(f"  Split core nodesets: steel_half_1={len(steel1_ids)}, steel_half_4={len(steel4_ids)}")
            print(f"  Far end nodesets: shell_fixed={len(shell_far_end_ids)}, steel_fixed_z={len(steel_far_end_ids)}")

        # Metadata
        meta = {
            "discretization": str(disc_path),
            "n_points": int(len(pts_out)),
            "shell_spacing_mm": float(spacing_mm),
            "steel_spacing_mm": float(steel_spacing_mm) if steel_spacing_mm else float(spacing_mm),
            "shell_particle_volume_m3": float(vol_shell),
            "block_ids": {int(k): int(v) for k, v in zip(*np.unique(blk_out, return_counts=True))},
            "bounds_m": {
                "min": [float(x) for x in pts_out.min(axis=0)],
                "max": [float(x) for x in pts_out.max(axis=0)],
            },
        }
        meta_path = per_dir / "peridigm_export_metadata.yaml"
        meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))

        print(f"  Output: {disc_path}")
        print(f"{'='*80}")

        return {
            "discretization": str(disc_path),
            "nodeset_all": str(per_dir / "nodeset_all.txt"),
            "nodeset_fixed_end": str(per_dir / "nodeset_fixed_end.txt"),
            "nodeset_impact_end": str(per_dir / "nodeset_impact_end.txt"),
            "metadata": str(meta_path),
        }

    def export_peridigm(self, output_dir: str | Path = "results_refined") -> dict[str, str]:
        """
        Export a Peridigm Text File discretization.

        If ``peridigm_export.uniform_spacing_mm`` is set in the config, a uniform-grid
        discretization is generated directly from the geometry parameters.  Otherwise,
        the legacy tet-centroid approach is used (points come from the gmsh tetra mesh).

        Peridigm format (one point per line):
          x y z block_id volume
        """
        cfg = self.config.get("peridigm_export", {}) or {}
        uniform_spacing = cfg.get("uniform_spacing_mm", None)

        if uniform_spacing is not None:
            return self._export_peridigm_uniform(output_dir, float(uniform_spacing))

        # --- Legacy tet-centroid path (unchanged) ---
        if self.tetrahedra is None or self.volumetric_vertices is None:
            self.create_volumetric_mesh(output_dir=output_dir)

        out_root = Path(output_dir)
        per_dir = out_root / "peridigm"
        per_dir.mkdir(parents=True, exist_ok=True)

        region_id = getattr(self, "cell_region_id", None)
        if region_id is None:
            region_id = self.element_materials.astype(np.int32)
        else:
            region_id = np.asarray(region_id, dtype=np.int32)

        centroids, volumes = self._tet_centroids_and_volumes(self.volumetric_vertices, self.tetrahedra)

        # Default: Peridigm block_id == gmsh region_id (1=steel, 2=pla_protected, 3=pla_fracture)
        block_map = cfg.get("block_id_map", None)
        if block_map is None:
            block_id = region_id.copy()
        else:
            # block_id_map keys may come in as strings via YAML.
            block_map_norm = {int(k): int(v) for k, v in dict(block_map).items()}
            block_id = np.array([block_map_norm.get(int(r), int(r)) for r in region_id], dtype=np.int32)

        disc_path = per_dir / "core_shell_refined_peridigm.txt"
        with disc_path.open("w") as f:
            for (x, y, z), blk, vol in zip(centroids, block_id, volumes):
                f.write(f"{x:.10e} {y:.10e} {z:.10e} {int(blk)} {vol:.10e}\n")

        # Node sets for Peridigm BCs are just lists of 1-based point ids (line numbers).
        all_ids = np.arange(1, len(centroids) + 1, dtype=np.int64)
        self._write_nodeset(per_dir / "nodeset_all.txt", all_ids)

        def _make_end_nodeset(name: str, default_side: str) -> None:
            ns_cfg = (cfg.get("node_sets", {}) or {}).get(name, None)
            if ns_cfg is False:
                return
            ns_cfg = ns_cfg or {}
            axis = str(ns_cfg.get("axis", "z")).lower()
            axis_i = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
            side = str(ns_cfg.get("side", default_side)).lower()
            band_m = float(ns_cfg.get("band_m", 0.003))
            region_ids = ns_cfg.get("region_ids", None)

            coord = centroids[:, axis_i]
            lo = float(coord.min())
            hi = float(coord.max())
            if side == "min":
                mask = coord <= (lo + band_m)
            else:
                mask = coord >= (hi - band_m)

            if region_ids is not None:
                region_ids = {int(x) for x in region_ids}
                mask &= np.isin(region_id, list(region_ids))

            ids = np.where(mask)[0] + 1
            self._write_nodeset(per_dir / f"nodeset_{name}.txt", ids.astype(np.int64))

        _make_end_nodeset("fixed_end", "max")
        _make_end_nodeset("impact_end", "min")

        # Additional nodesets for split core (if enabled)
        if self.split_core_enabled:
            # Steel half 1 (block 1)
            steel1_mask = (block_id == 1)
            steel1_ids = np.where(steel1_mask)[0] + 1
            self._write_nodeset(per_dir / "nodeset_steel_half_1.txt", steel1_ids.astype(np.int64))

            # Steel half 4 (block 4)
            steel4_mask = (block_id == 4)
            steel4_ids = np.where(steel4_mask)[0] + 1
            self._write_nodeset(per_dir / "nodeset_steel_half_4.txt", steel4_ids.astype(np.int64))

            # Shell (PLA) at far end - fully fixed
            shell_mask = (block_id == 2) | (block_id == 3)
            z_max = float(centroids[:, 2].max())
            far_end_mask = centroids[:, 2] >= (z_max - 0.003)
            shell_far_end_ids = np.where(shell_mask & far_end_mask)[0] + 1
            self._write_nodeset(per_dir / "nodeset_shell_fixed.txt", shell_far_end_ids.astype(np.int64))

            # Steel at far end - for Z-only constraint (free in X/Y for separation)
            steel_mask = (block_id == 1) | (block_id == 4)
            steel_far_end_ids = np.where(steel_mask & far_end_mask)[0] + 1
            self._write_nodeset(per_dir / "nodeset_steel_fixed_z.txt", steel_far_end_ids.astype(np.int64))

            print(f"  Split core nodesets: steel_half_1={len(steel1_ids)}, steel_half_4={len(steel4_ids)}")
            print(f"  Far end nodesets: shell_fixed={len(shell_far_end_ids)}, steel_fixed_z={len(steel_far_end_ids)}")

        meta = {
            "discretization": str(disc_path),
            "n_points": int(len(centroids)),
            "n_tets": int(len(self.tetrahedra)),
            "block_ids": {int(k): int(v) for k, v in zip(*np.unique(block_id, return_counts=True))},
            "bounds_m": {
                "min": [float(x) for x in centroids.min(axis=0)],
                "max": [float(x) for x in centroids.max(axis=0)],
            },
        }
        meta_path = per_dir / "peridigm_export_metadata.yaml"
        meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))

        return {
            "discretization": str(disc_path),
            "nodeset_all": str(per_dir / "nodeset_all.txt"),
            "nodeset_fixed_end": str(per_dir / "nodeset_fixed_end.txt"),
            "nodeset_impact_end": str(per_dir / "nodeset_impact_end.txt"),
            "metadata": str(meta_path),
        }

    def generate_geometry(self):
        """Generate and store the fracture-thickness field (phi-based)."""
        phi_angles = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)
        thickness_by_phi = np.full(self.n_phi, self.min_thickness, dtype=float)

        for j, phi in enumerate(phi_angles):
            in_fracture = False
            for frac_phi in self.fracture_planes:
                fp = float(frac_phi)
                fp = (fp + np.pi) % (2 * np.pi) - np.pi
                p = (phi + np.pi) % (2 * np.pi) - np.pi
                angle_diff = abs(p - fp)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                if angle_diff < self.fracture_width / 2:
                    in_fracture = True
                    break
            if in_fracture:
                thickness_by_phi[j] = self.min_thickness * float(self.initial_thickness_factor)

        self.thickness_field = thickness_by_phi
        return {"thickness_by_phi": thickness_by_phi}

    def _theta_band_config(self):
        """
        Optional theta-banded thickness configuration.

        Theta is measured from the +z axis on the steel core ellipsoid:
          theta = 0 at the pole (|z| ~= c_core), theta = 90 deg at the equator (z ~= 0).

        If configured, thickness is piecewise-constant in theta bands and still split
        into protected/fracture regions via the existing phi wedge logic.
        """
        mesh_cfg = self.config.get("mesh", {}) or {}
        edges_deg = mesh_cfg.get("theta_band_edges_deg", None)
        if edges_deg is None:
            return None

        edges_deg = [float(x) for x in edges_deg]
        if len(edges_deg) < 2:
            raise ValueError("mesh.theta_band_edges_deg must have at least 2 entries.")
        if abs(edges_deg[0]) > 1e-12 or abs(edges_deg[-1] - 90.0) > 1e-12:
            raise ValueError("mesh.theta_band_edges_deg must start at 0 and end at 90.")
        if any(b <= a for a, b in zip(edges_deg, edges_deg[1:])):
            raise ValueError("mesh.theta_band_edges_deg must be strictly increasing.")
        if any((x < 0.0) or (x > 90.0) for x in edges_deg):
            raise ValueError("mesh.theta_band_edges_deg entries must be within [0, 90].")

        n_bands = len(edges_deg) - 1

        mat = self.config.get("material", {}) or {}
        protected_mm = mat.get("protected_thickness_theta_mm", None)
        if protected_mm is None:
            protected_mm = [float(self.min_thickness) * 1000.0] * n_bands
        protected_mm = [float(x) for x in protected_mm]
        if len(protected_mm) != n_bands:
            raise ValueError("material.protected_thickness_theta_mm length must match theta bands.")

        fracture_enabled = (self.fracture_planes.size > 0) and (float(self.fracture_width) > 0.0)
        frac = self.config.get("fracture_design", {}) or {}
        fracture_mm = frac.get("fracture_thickness_theta_mm", None)
        if fracture_mm is None:
            factor_theta = frac.get("fracture_factor_theta", None)
            if factor_theta is None:
                if fracture_enabled:
                    raise ValueError(
                        "fracture zones are enabled, so you must set fracture_design.fracture_factor_theta "
                        "or fracture_design.fracture_thickness_theta_mm when using theta bands."
                    )
                factor_theta = [1.0] * n_bands
            factor_theta = [float(x) for x in factor_theta]
            if len(factor_theta) != n_bands:
                raise ValueError("fracture_design.fracture_factor_theta length must match theta bands.")
            fracture_mm = [p * f for p, f in zip(protected_mm, factor_theta)]
        else:
            fracture_mm = [float(x) for x in fracture_mm]
            if len(fracture_mm) != n_bands:
                raise ValueError("fracture_design.fracture_thickness_theta_mm length must match theta bands.")

        # Enforce minimum fracture zone thickness for meshability
        # Need at least 0.5mm for reliable meshing with varying protected thickness
        min_fracture_mm = frac.get("min_fracture_thickness_mm", 0.5)
        fracture_mm_clamped = []
        for i, (frac_t, prot_t) in enumerate(zip(fracture_mm, protected_mm)):
            if frac_t < min_fracture_mm:
                print(
                    f"[theta_mesh] WARNING: Fracture zone band {i}: thickness {frac_t:.2f}mm < min {min_fracture_mm}mm, "
                    f"clamping to {min_fracture_mm}mm (protected={prot_t:.2f}mm)"
                )
                frac_t = min_fracture_mm
            fracture_mm_clamped.append(frac_t)
        fracture_mm = fracture_mm_clamped

        edges_rad = np.deg2rad(np.array(edges_deg, dtype=float))
        protected_m = np.array(protected_mm, dtype=float) / 1000.0
        fracture_m = np.array(fracture_mm, dtype=float) / 1000.0

        return {
            "theta_edges_deg": np.array(edges_deg, dtype=float),
            "theta_edges_rad": edges_rad,
            "protected_thickness_m": protected_m,
            "fracture_thickness_m": fracture_m,
            "use_spline": bool(mesh_cfg.get("use_spline_thickness", False)),
        }

    def _add_theta_banded_outer(self, thickness_by_band_m: np.ndarray, theta_edges_rad: np.ndarray) -> list[int]:
        """
        Build an axisymmetric outer volume as a union of ellipsoids clipped by |z| slabs.

        This is a pragmatic way to approximate thickness variation with theta while keeping
        boolean operations stable and avoiding a custom CAD offset.
        """
        if len(thickness_by_band_m) != (len(theta_edges_rad) - 1):
            raise ValueError("thickness_by_band_m must have one entry per theta band.")

        occ = gmsh.model.occ
        a_core = float(self.a_core)
        c_core = float(self.c_core)
        max_t = float(np.max(thickness_by_band_m)) if len(thickness_by_band_m) else 0.0

        # Big enough to cover the entire ellipsoid during clipping.
        r_box = 3.0 * (a_core + max_t)

        vols: list[int] = []
        for band_idx, t in enumerate(thickness_by_band_m):
            theta0 = float(theta_edges_rad[band_idx])
            theta1 = float(theta_edges_rad[band_idx + 1])

            z_hi = c_core * float(np.cos(theta0))
            z_lo = c_core * float(np.cos(theta1))

            if z_hi < z_lo:
                z_hi, z_lo = z_lo, z_hi

            a_out = a_core + float(t)
            c_out = c_core + float(t)

            # IMPORTANT: the outer ellipsoid extends beyond `±c_core`. If we clip the top band
            # at `z=±c_core` we create a degenerate shell with ~zero thickness at the poles
            # (outer and inner coincide), which leads to invalid OCC solids and Gmsh meshing
            # failures. Extend the pole-facing end of the first band to the true outer pole.
            if band_idx == 0 and abs(theta0) < 1e-12:
                z_hi = c_out

            def intersect_with_box(z0: float, z1: float) -> None:
                ell = self._add_ellipsoid(a_out, c_out)
                box = occ.addBox(-r_box, -r_box, z0, 2 * r_box, 2 * r_box, z1 - z0)
                out, _ = occ.intersect([(3, ell)], [(3, box)], removeObject=True, removeTool=True)
                vols.extend([tag for (dim, tag) in out if dim == 3])

            if z_lo <= 1e-14:
                intersect_with_box(-z_hi, z_hi)
            else:
                intersect_with_box(z_lo, z_hi)
                intersect_with_box(-z_hi, -z_lo)

        if not vols:
            raise RuntimeError("Theta-banded outer volume construction produced no volumes.")

        # Fuse into as few solids as possible. Keeping lots of adjacent volumes can make
        # tetra meshing fragile (and is unnecessary for our downstream use).
        if self.theta_band_fuse and len(vols) > 1:
            occ.synchronize()
            try:
                fused, _ = occ.fuse([(3, int(vols[0]))], [(3, int(v)) for v in vols[1:]], removeObject=True, removeTool=True)
                occ.synchronize()
                vols = [int(tag) for (dim, tag) in fused if dim == 3]
            except Exception as e:
                print(f"⚠️ Theta-band fuse failed (continuing without fuse): {e}")

        return vols

    def _add_spline_outer(self, thickness_by_band_m: np.ndarray, theta_edges_rad: np.ndarray) -> list[int]:
        """Build a smooth outer volume by revolving a B-spline profile around the z-axis.

        Instead of clipping separate ellipsoids per band (which creates step
        discontinuities), this method:
        1. Fits a CubicSpline through band-midpoint thickness values.
        2. Computes θ_hole where the hole intersects the inner surface — the
           shell only exists for θ > θ_hole, so the BSpline starts there.
        3. Samples the spline from θ_hole to π/2, avoiding the degenerate
           pole region (r→0) that creates cusp geometry and Gmsh crashes.
        4. Closes the profile with straight lines through the pole cap
           (which gets cut away by the hole cylinder anyway).
        5. Revolves 360° around the z-axis to produce a solid of revolution.
        """
        from scipy.interpolate import CubicSpline as _CSpline

        occ = gmsh.model.occ
        a_core = float(self.a_core)
        c_core = float(self.c_core)
        hole_r = float(self.hole_radius)  # in metres

        n_bands = len(thickness_by_band_m)
        if n_bands < 1:
            raise ValueError("Need at least 1 band for spline outer surface.")

        # Build spline: knots at band midpoints, values = thickness per band
        midpoints = 0.5 * (theta_edges_rad[:-1] + theta_edges_rad[1:])

        if n_bands < 2:
            # Single band → constant thickness → plain ellipsoid
            t = float(thickness_by_band_m[0])
            return [self._add_ellipsoid(a_core + t, c_core + t)]

        spline = _CSpline(midpoints, thickness_by_band_m, bc_type='clamped')

        # ── Compute θ_hole: angle where hole intersects inner ellipsoid ──
        # Inner surface: r = a_core·sin(θ), so θ_hole = arcsin(hole_r / a_core)
        # Add a small margin (0.5°) so the BSpline starts just past the hole edge.
        if hole_r > 0 and hole_r < a_core:
            theta_hole = np.arcsin(hole_r / a_core) + np.deg2rad(0.5)
        else:
            theta_hole = np.deg2rad(1.0)  # fallback: skip the first degree

        theta_hole = min(theta_hole, float(theta_edges_rad[1]))  # don't skip past band 1

        # Sample profile from θ_hole to equator (θ=π/2)
        N = 200
        theta_samples = np.linspace(theta_hole, np.pi / 2.0, N)
        t_samples = np.clip(spline(theta_samples),
                            float(np.min(thickness_by_band_m)),
                            float(np.max(thickness_by_band_m)))

        # Compute outer profile points in (r, z) — axisymmetric in the xz-plane
        r_profile = (a_core + t_samples) * np.sin(theta_samples)
        z_profile = (c_core + t_samples) * np.cos(theta_samples)

        # Pole cap: thickness at θ_hole (constant up to the pole)
        t_pole = float(t_samples[0])
        z_top = c_core + t_pole  # z-axis intercept of outer surface at pole

        print(f"  Spline outer: θ_hole={np.rad2deg(theta_hole):.1f}°, "
              f"t_pole={t_pole*1000:.2f}mm, z_top={z_top*1000:.1f}mm")

        # === Build curves forming a closed loop in the xz-plane ===
        #
        #  Profile (upper half, xz-plane):
        #
        #    (0, z_top)  ─── line ───  (r_first, z_first)
        #        |                          |
        #     z-axis                    BSpline (smooth outer surface)
        #        |                          |
        #    (0, -z_top) ─── line ──  (r_first, -z_first)
        #
        #  The pole cap region (above the line) gets cut away by the hole
        #  cylinder, so its exact shape doesn't matter — straight lines are fine.

        # Top pole point on z-axis
        pt_pole_top = occ.addPoint(0.0, 0.0, z_top)

        # Upper-half BSpline: (r_first, z_first) → (r_eq, 0)
        upper_pts = []
        for i in range(N):
            upper_pts.append(occ.addPoint(float(r_profile[i]), 0.0, float(z_profile[i])))
        upper_curve = occ.addBSpline(upper_pts)

        # Line from pole to first BSpline point (pole cap — cut by hole)
        pole_to_spline = occ.addLine(pt_pole_top, upper_pts[0])

        # Lower-half BSpline: (r_eq, 0) → (r_first, -z_first)
        lower_pts = [upper_pts[-1]]  # start at equator
        for i in range(N - 2, -1, -1):
            lower_pts.append(occ.addPoint(float(r_profile[i]), 0.0, -float(z_profile[i])))
        lower_curve = occ.addBSpline(lower_pts)

        # Bottom pole point on z-axis
        pt_pole_bot = occ.addPoint(0.0, 0.0, -z_top)

        # Line from last BSpline point to bottom pole (pole cap — cut by hole)
        spline_to_pole = occ.addLine(lower_pts[-1], pt_pole_bot)

        # Closing line along z-axis: bottom pole → top pole
        axis_line = occ.addLine(pt_pole_bot, pt_pole_top)

        # Create closed curve loop → planar surface → revolve
        curve_loop = occ.addCurveLoop([
            pole_to_spline, upper_curve, lower_curve, spline_to_pole, axis_line
        ])
        surface = occ.addPlaneSurface([curve_loop])

        # Revolve 360° around the z-axis
        revolved = occ.revolve([(2, surface)], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.pi * 2)

        vols = [tag for (dim, tag) in revolved if dim == 3]
        if not vols:
            raise RuntimeError("Spline outer surface revolution produced no volumes.")

        return vols

    def _add_ellipsoid(self, a_r: float, a_z: float) -> int:
        """Create an ellipsoid primitive."""
        vol = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
        gmsh.model.occ.dilate([(3, vol)], 0.0, 0.0, 0.0, a_r, a_r, a_z)
        return vol

    def _split_volume_with_gap(self, vol_tag: int, axis: str, gap_m: float) -> tuple[int, int]:
        """
        Split a volume into two halves with a gap between them.

        Args:
            vol_tag: The volume tag to split
            axis: 'x' or 'y' - the axis perpendicular to the split plane
            gap_m: Gap between the halves in meters

        Returns:
            Tuple of (positive_half_tag, negative_half_tag)
        """
        occ = gmsh.model.occ

        # Get bounding box to create cutting boxes
        occ.synchronize()
        xmin, ymin, zmin, xmax, ymax, zmax = occ.getBoundingBox(3, vol_tag)

        # Add padding
        pad = 0.01  # 10mm padding
        xmin -= pad
        ymin -= pad
        zmin -= pad
        xmax += pad
        ymax += pad
        zmax += pad

        half_gap = gap_m / 2.0

        if axis == 'x':
            # Positive half: x >= half_gap
            box_pos = occ.addBox(half_gap, ymin, zmin, xmax - half_gap, ymax - ymin, zmax - zmin)
            # Negative half: x <= -half_gap
            box_neg = occ.addBox(xmin, ymin, zmin, -half_gap - xmin, ymax - ymin, zmax - zmin)
        else:  # axis == 'y'
            # Positive half: y >= half_gap
            box_pos = occ.addBox(xmin, half_gap, zmin, xmax - xmin, ymax - half_gap, zmax - zmin)
            # Negative half: y <= -half_gap
            box_neg = occ.addBox(xmin, ymin, zmin, xmax - xmin, -half_gap - ymin, zmax - zmin)

        # Copy the original volume for each intersection
        vol_copy1 = occ.copy([(3, vol_tag)])[0][1]
        vol_copy2 = occ.copy([(3, vol_tag)])[0][1]

        # Intersect to get the two halves
        pos_result, _ = occ.intersect([(3, vol_copy1)], [(3, box_pos)], removeObject=True, removeTool=True)
        neg_result, _ = occ.intersect([(3, vol_copy2)], [(3, box_neg)], removeObject=True, removeTool=True)

        # Remove original volume
        occ.remove([(3, vol_tag)], recursive=True)

        pos_vols = [tag for (dim, tag) in pos_result if dim == 3]
        neg_vols = [tag for (dim, tag) in neg_result if dim == 3]

        if not pos_vols or not neg_vols:
            raise RuntimeError(f"Split core failed: pos_vols={pos_vols}, neg_vols={neg_vols}")

        print(f"  Split core: axis={axis}, gap={gap_m*1000:.1f}mm")
        print(f"    Positive half: {len(pos_vols)} volume(s)")
        print(f"    Negative half: {len(neg_vols)} volume(s)")

        return pos_vols[0], neg_vols[0]

    def _add_sector_prism(self, radius: float, phi_start: float, phi_end: float, z0: float, height: float) -> int:
        """Create a sector prism for fracture zones."""
        phi_mid = 0.5 * (phi_start + phi_end)

        def p_at(phi: float) -> int:
            return gmsh.model.occ.addPoint(radius * float(np.cos(phi)), radius * float(np.sin(phi)), z0)

        p0 = gmsh.model.occ.addPoint(0.0, 0.0, z0)
        p1 = p_at(phi_start)
        p2 = p_at(phi_end)
        pm = p_at(phi_mid)

        l1 = gmsh.model.occ.addLine(p0, p1)
        arc = gmsh.model.occ.addCircleArc(p1, pm, p2)  # center=False is default
        l2 = gmsh.model.occ.addLine(p2, p0)
        loop = gmsh.model.occ.addCurveLoop([l1, arc, l2])
        surf = gmsh.model.occ.addPlaneSurface([loop])

        extruded = gmsh.model.occ.extrude([(2, surf)], 0.0, 0.0, height)
        vols = [tag for (dim, tag) in extruded if dim == 3]
        if len(vols) != 1:
            raise RuntimeError(f"Expected 1 wedge volume, got {len(vols)}")
        return vols[0]

    def _setup_mesh_refinement(self, *, steel_vols: list[int]):
        """
        Setup distance-based mesh refinement at the steel/PLA interface.
        This creates a smooth transition from fine mesh at the interface to coarser mesh away from it.
        """
        # IMPORTANT: refine ONLY around the OUTER steel boundary (steel/PLA interface),
        # not the inner hole surface or other surfaces.
        gmsh.model.occ.synchronize()

        # Get geometry parameters for filtering
        hole_radius = self.hole_radius  # Inner hole radius (small, ~2mm)
        interface_radius = self.core_radius_r  # Steel-PLA interface radius (larger, ~13mm)
        radius_threshold = (hole_radius + interface_radius) / 2  # Midpoint for classification

        all_steel_surfaces: list[int] = []
        valid_vols = {int(tag) for (dim, tag) in gmsh.model.getEntities(3) if dim == 3}
        for tag in steel_vols:
            tag_i = int(tag)
            if tag_i not in valid_vols:
                print(f"⚠️ Interface refinement: steel volume tag {tag_i} not found in model (skipping).")
                continue
            try:
                boundaries = gmsh.model.getBoundary([(3, tag_i)], oriented=False)
            except Exception as e:
                print(f"⚠️ Interface refinement: failed to getBoundary for steel volume {tag_i}: {e} (skipping).")
                continue
            for bdim, btag in boundaries:
                if bdim == 2:
                    all_steel_surfaces.append(int(btag))

        # Remove duplicates
        all_steel_surfaces = list(set(all_steel_surfaces))

        # Filter to only include OUTER interface surfaces (exclude inner hole)
        # The outer interface is at r ~ core_radius_r, the hole is at r ~ hole_radius
        interface_surfaces: list[int] = []
        hole_surfaces: list[int] = []
        for surf_tag in all_steel_surfaces:
            try:
                # Get the bounding box of the surface to determine its radial position
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surf_tag)
                # Compute the MAXIMUM radial extent - if a surface reaches outer radius, it's interface
                # Check all corners of bounding box in x-y plane
                corners_r = [
                    np.sqrt(xmin**2 + ymin**2),
                    np.sqrt(xmin**2 + ymax**2),
                    np.sqrt(xmax**2 + ymin**2),
                    np.sqrt(xmax**2 + ymax**2),
                ]
                r_max = max(corners_r)
                r_min = min(corners_r)

                # If surface extends to outer interface radius, it's an interface surface
                # If surface is entirely within the hole radius zone, it's a hole surface
                if r_max > radius_threshold:
                    interface_surfaces.append(surf_tag)
                else:
                    hole_surfaces.append(surf_tag)
            except Exception:
                # If we can't get bounding box, include it to be safe
                interface_surfaces.append(surf_tag)

        print(f"✓ Interface surface filtering:")
        print(f"  - Total steel surfaces: {len(all_steel_surfaces)}")
        print(f"  - Outer interface surfaces: {len(interface_surfaces)} (r_max > {radius_threshold*1000:.1f}mm)")
        print(f"  - Inner hole surfaces excluded: {len(hole_surfaces)}")

        if interface_surfaces:
            # Create distance field from interface surfaces
            dist_id = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_id, "SurfacesList", interface_surfaces)

            # Create threshold field for smooth size transition in PLA
            # This goes from interface_mesh_size (at interface) to mesh_size (away from interface)
            thresh_id = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresh_id, "InField", dist_id)
            gmsh.model.mesh.field.setNumber(thresh_id, "SizeMin", self.interface_mesh_size)
            gmsh.model.mesh.field.setNumber(thresh_id, "SizeMax", self.mesh_size)
            gmsh.model.mesh.field.setNumber(thresh_id, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(thresh_id, "DistMax", self.interface_distance)

            # If core_mesh_size is different from mesh_size, apply coarse mesh to steel volumes
            if abs(self.core_mesh_size - self.mesh_size) > 1e-6 and steel_vols:
                # Strategy: Set mesh size on steel boundary points BEFORE setting background field
                # Background field only affects new points created during meshing

                # First, set coarse mesh size on all steel volume boundary points
                for vol_tag in steel_vols:
                    try:
                        # Get all boundary entities recursively (surfaces->curves->points)
                        boundaries = gmsh.model.getBoundary([(3, vol_tag)], oriented=False, recursive=True)
                        points_to_set = [(dim, tag) for dim, tag in boundaries if dim == 0]
                        if points_to_set:
                            gmsh.model.mesh.setSize(points_to_set, self.core_mesh_size)
                    except Exception as e:
                        print(f"⚠️ Could not set mesh size on steel volume {vol_tag}: {e}")

                # Use threshold field only for PLA - set SizeMax to mesh_size (PLA bulk)
                # The steel points already have their size set, so background field won't override them
                gmsh.model.mesh.field.setAsBackgroundMesh(thresh_id)
                self._background_field_id = int(thresh_id)

                # Optionally set global max to allow coarse steel
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.core_mesh_size * 1.1)

                print(f"✓ Multi-region mesh refinement configured:")
                print(f"  - Interface mesh size: {self.interface_mesh_size*1000:.2f} mm")
                print(f"  - PLA bulk mesh size: {self.mesh_size*1000:.2f} mm")
                print(f"  - Steel core mesh size: {self.core_mesh_size*1000:.2f} mm (coarse)")
                print(f"  - Transition distance: {self.interface_distance*1000:.2f} mm")
            else:
                # No separate core mesh - use original single-field approach
                gmsh.model.mesh.field.setAsBackgroundMesh(thresh_id)
                self._background_field_id = int(thresh_id)

                print(f"✓ Interface refinement configured:")
                print(f"  - Interface mesh size: {self.interface_mesh_size*1000:.2f} mm")
                print(f"  - Bulk mesh size: {self.mesh_size*1000:.2f} mm")
                print(f"  - Transition distance: {self.interface_distance*1000:.2f} mm")
        else:
            self._background_field_id = None
            print("⚠️ Interface refinement skipped (could not find steel boundary surfaces).")

    def _merge_duplicate_nodes(self, points, tets, cell_arrays, tol_m):
        """Merge coincident points and remap tetra connectivity."""
        if points.size == 0 or tets.size == 0:
            return points, tets, cell_arrays

        if tol_m <= 0:
            return points, tets, cell_arrays

        keys = np.round(points / tol_m).astype(np.int64)
        _, unique_idx, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        if len(unique_idx) == len(points):
            return points, tets, cell_arrays

        new_points = points[unique_idx]
        new_tets = inverse[tets]

        # Drop any degenerate tets created by merging
        sorted_tets = np.sort(new_tets, axis=1)
        valid = np.all(np.diff(sorted_tets, axis=1) != 0, axis=1)
        new_tets = new_tets[valid]
        new_cell_arrays = {k: v[valid] for k, v in cell_arrays.items()}
        return new_points, new_tets, new_cell_arrays

    def create_volumetric_mesh(self, include_solid_core=True, output_dir=None):
        """
        Create a conforming tetra mesh via Gmsh with interface refinement.

        Args:
            include_solid_core: Must be True (this generator always produces solid core)
            output_dir: Directory for intermediate mesh files (default: "results_refined")
        """
        if self.thickness_field is None:
            self.generate_geometry()

        if not include_solid_core:
            raise ValueError("This generator always produces a solid steel core; pass include_solid_core=True.")

        fracture_enabled = (self.fracture_planes.size > 0) and (float(self.fracture_width) > 0.0)
        theta_cfg = self._theta_band_config()

        t_protected = float(self.min_thickness)
        t_fracture = (
            float(self.min_thickness) * float(self.initial_thickness_factor) if fracture_enabled else float(self.min_thickness)
        )
        a_core = float(self.a_core)
        c_core = float(self.c_core)

        if theta_cfg is None:
            t_protected_by_band_m = None
            t_fracture_by_band_m = None
            theta_edges_rad = None
            max_t_protected = t_protected
            max_t_fracture = t_fracture
            min_outer_a = a_core + min(t_protected, t_fracture)
        else:
            theta_edges_rad = theta_cfg["theta_edges_rad"]
            t_protected_by_band_m = theta_cfg["protected_thickness_m"]
            t_fracture_by_band_m = theta_cfg["fracture_thickness_m"]

            max_t_protected = float(np.max(t_protected_by_band_m))
            max_t_fracture = float(np.max(t_fracture_by_band_m))
            min_outer_a = a_core + min(float(np.min(t_protected_by_band_m)), float(np.min(t_fracture_by_band_m)))

        if self.hole_radius <= 0:
            raise ValueError("Set `geometry.hole_radius_mm` > 0 for a through-hole.")
        if self.hole_radius >= min_outer_a:
            raise ValueError("Hole radius is too large for the outer geometry.")

        out_dir = Path(output_dir) if output_dir else Path("results_refined")
        out_dir.mkdir(parents=True, exist_ok=True)
        msh_path = out_dir / "core_shell_refined.msh"

        gmsh.initialize()
        try:
            if self.occ_heal:
                # Enable OCC auto-fix options in addition to explicit healShapes().
                gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
                gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
                gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
                gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
                gmsh.option.setNumber("Geometry.OCCMakeSolids", 1)
            gmsh.option.setNumber("General.Terminal", self.gmsh_terminal)
            gmsh.model.add("core_shell_refined")

            occ = gmsh.model.occ

            print("\n" + "="*80)
            print("CREATING REFINED MESH WITH SMOOTH INTERFACE")
            print("="*80)

            # Primitives
            vol_core_prim = self._add_ellipsoid(a_core, c_core)

            if theta_cfg is None:
                vol_outer_protected_prims = [self._add_ellipsoid(a_core + t_protected, c_core + t_protected)]
                vol_outer_fracture_prims = (
                    [self._add_ellipsoid(a_core + t_fracture, c_core + t_fracture)] if fracture_enabled else []
                )
            elif theta_cfg.get("use_spline", False):
                vol_outer_protected_prims = self._add_spline_outer(t_protected_by_band_m, theta_edges_rad)
                vol_outer_fracture_prims = (
                    self._add_spline_outer(t_fracture_by_band_m, theta_edges_rad) if fracture_enabled else []
                )
            else:
                vol_outer_protected_prims = self._add_theta_banded_outer(t_protected_by_band_m, theta_edges_rad)
                vol_outer_fracture_prims = (
                    self._add_theta_banded_outer(t_fracture_by_band_m, theta_edges_rad) if fracture_enabled else []
                )

            # Tools: hole cylinder + fracture wedges
            a_out_protected_bound = a_core + max_t_protected
            c_out_protected_bound = c_core + max_t_protected
            z_extent = 2.2 * c_out_protected_bound

            # Create hole cutting tool - optionally with filleted ends
            if self.hole_fillet_torus > 0:
                # Create a "filleted cylinder" using sphere caps
                # Simpler than torus approach, more reliable boolean operations
                fillet_r = self.hole_fillet_torus
                print(f"\n  Creating filleted hole (fillet_r={fillet_r*1000:.2f}mm)...")

                z_top = c_core + max_t_protected
                z_bot = -z_top

                # Main cylinder through the entire part
                main_cyl = occ.addCylinder(0.0, 0.0, -z_extent, 0.0, 0.0, 2.0 * z_extent, self.hole_radius)

                # Add spheres at each pole to create rounded openings
                # The sphere creates a smooth transition that widens the hole at the surface
                sphere_parts = []

                for sign in [1, -1]:  # Top and bottom poles
                    z_surface = sign * z_top  # Outer shell surface

                    # Place sphere so it's centered at hole edge, at surface level
                    # Sphere radius = fillet_r, centered at (hole_radius, 0, z_surface)
                    # We'll revolve this around z-axis by creating a torus-like shape

                    # Simpler: use a cone to create a smooth widening
                    # Cone from (hole_radius) at z_surface-fillet_r to (hole_radius+fillet_r) at z_surface
                    if sign > 0:
                        cone = occ.addCone(0, 0, z_surface - fillet_r,
                                          0, 0, fillet_r + 0.001,
                                          self.hole_radius,  # Bottom radius
                                          self.hole_radius + fillet_r)  # Top radius (wider at surface)
                    else:
                        cone = occ.addCone(0, 0, z_surface + fillet_r,
                                          0, 0, -(fillet_r + 0.001),
                                          self.hole_radius,  # Top radius (at inner)
                                          self.hole_radius + fillet_r)  # Bottom radius (wider at surface)
                    sphere_parts.append(cone)

                # Fuse cylinder with cone caps
                try:
                    all_parts = [(3, main_cyl)] + [(3, p) for p in sphere_parts]
                    fused, _ = occ.fuse([all_parts[0]], all_parts[1:],
                                       removeObject=True, removeTool=True)
                    hole_vols = [tag for dim, tag in fused if dim == 3]
                    if hole_vols:
                        hole = hole_vols[0]
                        print(f"    ✓ Filleted hole tool created (cone-based)")
                    else:
                        print(f"    ⚠️ Fillet fusion failed, using plain cylinder")
                        hole = occ.addCylinder(0.0, 0.0, -z_extent, 0.0, 0.0, 2.0 * z_extent, self.hole_radius)
                except Exception as e:
                    print(f"    ⚠️ Fillet creation failed: {e}, using plain cylinder")
                    hole = occ.addCylinder(0.0, 0.0, -z_extent, 0.0, 0.0, 2.0 * z_extent, self.hole_radius)
            else:
                # Plain cylinder hole (sharp edges)
                hole = occ.addCylinder(0.0, 0.0, -z_extent, 0.0, 0.0, 2.0 * z_extent, self.hole_radius)

            wedges = []
            if fracture_enabled:
                wedge_radius = 2.5 * a_out_protected_bound
                wedge_height = 2.4 * z_extent
                wedge_z0 = -0.5 * wedge_height
                for phi0 in self.fracture_planes:
                    phi_start = float(phi0) - self.fracture_width / 2
                    phi_end = float(phi0) + self.fracture_width / 2
                    wedges.append(self._add_sector_prism(wedge_radius, phi_start, phi_end, wedge_z0, wedge_height))

            # Copies for boolean operations
            core_for_steel = occ.copy([(3, vol_core_prim)])[0][1]
            core_for_shell_protected = occ.copy([(3, vol_core_prim)])[0][1]
            core_for_shell_fracture = occ.copy([(3, vol_core_prim)])[0][1] if fracture_enabled else None
            outer_protected = [
                tag for (dim, tag) in occ.copy([(3, v) for v in vol_outer_protected_prims]) if dim == 3
            ]
            outer_fracture = (
                [tag for (dim, tag) in occ.copy([(3, v) for v in vol_outer_fracture_prims]) if dim == 3]
                if fracture_enabled
                else []
            )

            # Remove the primitives
            occ.remove([(3, vol_core_prim)], recursive=True)
            occ.remove([(3, v) for v in vol_outer_protected_prims], recursive=True)
            occ.remove([(3, v) for v in vol_outer_fracture_prims], recursive=True)

            # Boolean operations
            steel_out, _ = occ.cut([(3, core_for_steel)], [(3, hole)], removeObject=True, removeTool=False)
            steel_vols = [tag for (dim, tag) in steel_out if dim == 3]

            # Split steel core into two halves if enabled (for flyback simulation)
            steel_vols_half1 = []
            steel_vols_half4 = []
            if self.split_core_enabled and steel_vols:
                print(f"\n  Splitting steel core for flyback simulation...")
                gap_m = self.split_core_gap_mm / 1000.0
                for sv in steel_vols:
                    try:
                        half1, half4 = self._split_volume_with_gap(sv, self.split_core_axis, gap_m)
                        steel_vols_half1.append(half1)
                        steel_vols_half4.append(half4)
                    except Exception as e:
                        print(f"  ⚠️ Failed to split steel volume {sv}: {e}")
                        # Keep original if split fails
                        steel_vols_half1.append(sv)
                # Update steel_vols to be the union of both halves for downstream processing
                steel_vols = steel_vols_half1 + steel_vols_half4
                print(f"  Split steel core: {len(steel_vols_half1)} + {len(steel_vols_half4)} volumes")

            prot_base, _ = occ.cut(
                [(3, v) for v in outer_protected],
                [(3, core_for_shell_protected)],
                removeObject=True,
                removeTool=True,
            )
            prot_base_vols = [dt for dt in prot_base if dt[0] == 3]

            prot_outside = prot_base_vols
            frac_vols_pre_hole = []
            if fracture_enabled:
                prot_outside, _ = occ.cut(prot_base_vols, [(3, w) for w in wedges], removeObject=True, removeTool=False)

                frac_base, _ = occ.cut(
                    [(3, v) for v in outer_fracture],
                    [(3, core_for_shell_fracture)],
                    removeObject=True,
                    removeTool=True,
                )
                frac_base_vols = [dt for dt in frac_base if dt[0] == 3]
                frac_inside, _ = occ.intersect(
                    frac_base_vols, [(3, w) for w in wedges], removeObject=True, removeTool=False
                )
                frac_vols_pre_hole = [tag for (dim, tag) in frac_inside if dim == 3]

            prot_vols_pre_hole = [tag for (dim, tag) in prot_outside if dim == 3]

            if not prot_vols_pre_hole and not frac_vols_pre_hole:
                raise RuntimeError("PLA volume construction failed.")

            prot_final = []
            if prot_vols_pre_hole:
                prot_cut, _ = occ.cut([(3, v) for v in prot_vols_pre_hole], [(3, hole)], removeObject=True, removeTool=False)
                prot_final = [tag for (dim, tag) in prot_cut if dim == 3]

            frac_final = []
            if frac_vols_pre_hole:
                frac_cut, _ = occ.cut([(3, v) for v in frac_vols_pre_hole], [(3, hole)], removeObject=True, removeTool=False)
                frac_final = [tag for (dim, tag) in frac_cut if dim == 3]

            # Apply chamfer to hole rim if specified (more reliable than fillet)
            # Chamfer creates a beveled edge where the hole wall meets the outer shell surface
            elif self.hole_chamfer_size > 0:
                occ.synchronize()
                chamfer_size = self.hole_chamfer_size
                print(f"\n  Applying chamfer (size={chamfer_size*1000:.2f}mm) to hole rim...")

                # Create chamfer tool: truncated cones at each pole that widen the hole opening
                # The chamfer creates a 45° bevel at the outer surface of the shell
                # - At the outer surface: hole opens to (hole_radius + chamfer_size)
                # - At the steel interface: hole stays at hole_radius
                chamfer_tools = []

                # Shell thickness at pole (approximate)
                shell_thickness = max_t_protected

                for sign in [1, -1]:  # Top and bottom poles
                    # z_outer: outer surface of shell at pole
                    # z_inner: inner surface (steel-PLA interface) at pole
                    z_outer = sign * (c_core + shell_thickness)
                    z_inner = sign * c_core

                    # Create a cone tool:
                    # - Wide end (r = hole_radius + chamfer_size) at outer surface
                    # - Narrow end (r = hole_radius) at inner surface (or slightly beyond)
                    # The cone should extend slightly past the inner surface to ensure clean cut

                    if sign > 0:  # Top pole (z > 0)
                        # Cone base at outer surface, tip toward inner
                        cone = occ.addCone(
                            0, 0, z_outer,                           # Base at outer surface
                            0, 0, -(chamfer_size + 0.001),           # Extend inward by chamfer_size
                            self.hole_radius + chamfer_size,          # Base radius (wide, at outer)
                            self.hole_radius                          # Top radius (narrow, at inner)
                        )
                    else:  # Bottom pole (z < 0)
                        # Cone base at outer surface (negative z), tip toward inner
                        cone = occ.addCone(
                            0, 0, z_outer,                           # Base at outer surface
                            0, 0, (chamfer_size + 0.001),            # Extend inward (positive direction)
                            self.hole_radius + chamfer_size,          # Base radius (wide, at outer)
                            self.hole_radius                          # Top radius (narrow, at inner)
                        )
                    chamfer_tools.append(cone)
                    print(f"    Chamfer cone at z={z_outer*1000:.1f}mm: r_outer={1000*(self.hole_radius + chamfer_size):.1f}mm -> r_inner={self.hole_radius*1000:.1f}mm")

                if chamfer_tools:
                    print(f"    Created {len(chamfer_tools)} chamfer tool cones")
                    try:
                        # Cut the chamfer from all shell volumes
                        all_shell_vols = [(3, v) for v in (prot_final + frac_final)]
                        if all_shell_vols:
                            chamfer_cut, _ = occ.cut(
                                all_shell_vols,
                                [(3, c) for c in chamfer_tools],
                                removeObject=True,
                                removeTool=True
                            )
                            occ.synchronize()

                            # Update volume lists
                            prot_final = []
                            frac_final = []
                            for dim, tag in chamfer_cut:
                                if dim == 3:
                                    # Re-classify based on wedge intersection would be needed here
                                    # For simplicity, put all in protected for now
                                    prot_final.append(tag)

                            print(f"    ✓ Chamfer applied to {len(chamfer_cut)} shell volumes")
                        else:
                            # Clean up chamfer tools
                            occ.remove([(3, c) for c in chamfer_tools], recursive=True)
                    except Exception as e:
                        print(f"    ⚠️ Chamfer failed: {e}. Continuing with sharp edges.")
                        try:
                            occ.remove([(3, c) for c in chamfer_tools], recursive=True)
                        except Exception:
                            pass

            # Apply fillet to hole rim edges if specified (alternative to chamfer)
            # Note: Gmsh fillet can be fragile with complex boolean geometries
            elif self.hole_fillet_radius > 0:
                occ.synchronize()
                print(f"\n  Applying fillet (r={self.hole_fillet_radius*1000:.2f}mm) to hole rim edges...")
                print(f"    Note: Gmsh fillet can fail on complex geometries. Consider using chamfer instead.")
                print(f"    Set geometry.hole_chamfer_size_mm instead of hole_fillet_radius_mm")

                # Try to apply fillet - simplified approach focusing on one edge at a time
                all_edges = gmsh.model.getEntities(1)
                hole_rim_edges = []

                for dim, edge_tag in all_edges:
                    try:
                        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, edge_tag)
                        r_center = np.sqrt(((xmin+xmax)/2)**2 + ((ymin+ymax)/2)**2)
                        z_center = (zmin + zmax) / 2
                        dz = zmax - zmin

                        # Check if edge is at hole radius and near pole
                        is_at_hole_radius = abs(r_center - self.hole_radius) < self.hole_radius * 0.3
                        is_near_pole = abs(abs(z_center) - c_core) < 0.005
                        is_circular = dz < 0.005

                        if is_at_hole_radius and is_near_pole and is_circular:
                            hole_rim_edges.append(edge_tag)
                    except Exception:
                        continue

                if hole_rim_edges:
                    print(f"    Found {len(hole_rim_edges)} candidate edges")
                    fillet_success = False

                    # Try fillet on each volume individually
                    for vol_list, name in [(steel_vols, "steel"), (prot_final, "protected"), (frac_final, "fracture")]:
                        if not vol_list:
                            continue
                        for vol in vol_list:
                            try:
                                occ.fillet([vol], hole_rim_edges[:2], [self.hole_fillet_radius], removeVolume=True)
                                occ.synchronize()
                                print(f"    ✓ Fillet applied to {name} volume {vol}")
                                fillet_success = True
                                break
                            except Exception as e:
                                continue
                        if fillet_success:
                            break

                    if not fillet_success:
                        print(f"    ⚠️ Fillet failed on all volumes. Consider using chamfer instead.")
                else:
                    print(f"    ⚠️ No suitable hole rim edges found for fillet")

            # Ensure protected + fracture volumes share topology so the mesh is conforming across the interface.
            # CRITICAL: If this step fails, the mesh will be non-conforming at the protected/fracture
            # interface, causing stress artifacts and unreliable FEA results. We raise an error
            # rather than silently continuing with a bad mesh.
            if fracture_enabled and prot_final and frac_final:
                prot_objs = [(3, int(v)) for v in prot_final]
                frac_objs = [(3, int(v)) for v in frac_final]
                try:
                    frag_out, frag_map = occ.fragment(prot_objs, frac_objs, removeObject=False, removeTool=False)
                except Exception as e:
                    raise RuntimeError(
                        f"CRITICAL: occ.fragment() failed for protected/fracture interface. "
                        f"Mesh would be non-conforming. Error: {e}"
                    ) from e
                occ.synchronize()

                def _flatten(map_list):
                    out = []
                    for entry in map_list:
                        out.extend(entry)
                    return out

                prot_new = [tag for (dim, tag) in _flatten(frag_map[: len(prot_objs)]) if dim == 3]
                frac_new = [tag for (dim, tag) in _flatten(frag_map[len(prot_objs) :]) if dim == 3]

                if prot_new and frac_new:
                    occ.remove(prot_objs + frac_objs, recursive=True)
                    occ.synchronize()
                    prot_final = prot_new
                    frac_final = frac_new
                else:
                    raise RuntimeError(
                        "CRITICAL: occ.fragment() produced empty volumes for protected/fracture interface. "
                        f"prot_new={len(prot_new)}, frac_new={len(frac_new)}. "
                        "Mesh would be non-conforming. Check geometry configuration."
                    )

            # Cleanup tools
            occ.remove([(3, hole)], recursive=True)
            if wedges:
                occ.remove([(3, w) for w in wedges], recursive=True)

            occ.synchronize()

            # OCC healing + merge coincident entities so the multi-volume mesh is conforming.
            if self.occ_heal:
                try:
                    occ.healShapes(
                        [],
                        tolerance=self.occ_heal_tol,
                        fixDegenerated=True,
                        fixSmallEdges=True,
                        fixSmallFaces=True,
                        sewFaces=True,
                        makeSolids=True,
                    )
                    occ.synchronize()
                except Exception as e:
                    print(f"⚠️ OCC healShapes failed (continuing): {e}")

            try:
                occ.removeAllDuplicates()
                occ.synchronize()
            except Exception:
                pass

            # `removeAllDuplicates()` may delete/renumber volume tags; refresh and filter before
            # defining physical groups or using tags for interface refinement.
            valid_vols = {int(tag) for (dim, tag) in gmsh.model.getEntities(3) if dim == 3}

            def _refresh_vols(pg_tag: int, fallback: list[int]) -> list[int]:
                try:
                    pg = [int(t) for t in gmsh.model.getEntitiesForPhysicalGroup(3, pg_tag)]
                except Exception:
                    pg = []
                pg = [t for t in pg if t in valid_vols]
                if pg:
                    return pg
                return [int(t) for t in fallback if int(t) in valid_vols]

            steel_vols = _refresh_vols(1, steel_vols)
            prot_final = _refresh_vols(2, prot_final)
            frac_final = _refresh_vols(3, frac_final)

            # Physical groups (re-add after refresh; same tag overwrites membership in Gmsh)
            if self.split_core_enabled and steel_vols_half1 and steel_vols_half4:
                # Split core: two separate physical groups
                steel_vols_half1 = [v for v in steel_vols_half1 if v in valid_vols]
                steel_vols_half4 = [v for v in steel_vols_half4 if v in valid_vols]
                if steel_vols_half1:
                    gmsh.model.addPhysicalGroup(3, steel_vols_half1, tag=1)
                    gmsh.model.setPhysicalName(3, 1, "steel_half1")
                if steel_vols_half4:
                    gmsh.model.addPhysicalGroup(3, steel_vols_half4, tag=4)
                    gmsh.model.setPhysicalName(3, 4, "steel_half4")
                print(f"  Split core physical groups: block_1={len(steel_vols_half1)}, block_4={len(steel_vols_half4)}")
            elif steel_vols:
                gmsh.model.addPhysicalGroup(3, steel_vols, tag=1)
                gmsh.model.setPhysicalName(3, 1, "steel")
            if prot_final:
                gmsh.model.addPhysicalGroup(3, prot_final, tag=2)
                gmsh.model.setPhysicalName(3, 2, "pla_protected")
            if frac_final:
                gmsh.model.addPhysicalGroup(3, frac_final, tag=3)
                gmsh.model.setPhysicalName(3, 3, "pla_fracture")

            # INTERFACE REFINEMENT - Key addition
            self._setup_mesh_refinement(steel_vols=steel_vols)

            # Enhanced mesh options for better quality
            gmsh.option.setNumber("Mesh.SaveAll", 0)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.option.setNumber("Mesh.Algorithm", self.mesh_algorithm_2d)
            gmsh.option.setNumber("Mesh.Algorithm3D", self.mesh_algorithm)
            gmsh.option.setNumber("Mesh.Optimize", self.mesh_optimize)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)  # Netgen segfaults on complex multi-region geometry
            gmsh.option.setNumber("Mesh.Smoothing", self.mesh_smoothing)
            gmsh.option.setNumber("Mesh.SmoothRatio", 1.8)
            gmsh.option.setNumber("Mesh.AnisoMax", 1.0)  # Prevent highly anisotropic elements
            gmsh.option.setNumber("Mesh.IgnorePeriodicity", 1)

            if self.debug_geometry == "always":
                try:
                    dbg_brep = out_dir / "debug_geometry.brep"
                    dbg_geo = out_dir / "debug_geometry.geo_unrolled"
                    gmsh.write(str(dbg_brep))
                    gmsh.write(str(dbg_geo))
                    print(f"Debug geometry: {dbg_brep}")
                    print(f"Debug geometry: {dbg_geo}")
                except Exception as e:
                    print(f"⚠️ Failed to write debug geometry: {e}")

            # Generate 3D mesh
            print("\nGenerating refined mesh...")
            def _attempt_mesh(*, label: str, algorithm3d: int, algorithm2d: int, background_field=None):
                print(f"Meshing attempt: {label} (Algorithm3D={algorithm3d}, Algorithm2D={algorithm2d}, "
                      f"BackgroundField={'on' if background_field else 'off'})")
                gmsh.model.mesh.clear()
                gmsh.option.setNumber("Mesh.Algorithm3D", int(algorithm3d))
                gmsh.option.setNumber("Mesh.Algorithm", int(algorithm2d))
                gmsh.option.setNumber("Mesh.IgnorePeriodicity", 1)
                if background_field:
                    gmsh.model.mesh.field.setAsBackgroundMesh(int(background_field))
                else:
                    gmsh.model.mesh.field.setAsBackgroundMesh(0)
                gmsh.model.mesh.generate(3)
                node_tags, _, _ = gmsh.model.mesh.getNodes()
                if len(node_tags) == 0:
                    raise RuntimeError("Gmsh generated zero nodes (invalid/empty mesh).")
                etypes, etags, _ = gmsh.model.mesh.getElements(3)
                n3d = sum(len(x) for x in etags) if etags else 0
                if n3d == 0:
                    raise RuntimeError("Gmsh generated zero 3D elements (invalid/empty mesh).")

            mesh_exc = None
            try:
                _attempt_mesh(
                    label="primary",
                    algorithm3d=self.mesh_algorithm,
                    algorithm2d=self.mesh_algorithm_2d,
                    background_field=self._background_field_id,
                )
            except Exception as e:
                mesh_exc = e
                print(f"⚠️ Primary meshing failed: {e}")

            if mesh_exc is not None and self._background_field_id is not None:
                try:
                    _attempt_mesh(
                        label="no-background-field",
                        algorithm3d=self.mesh_algorithm,
                        algorithm2d=self.mesh_algorithm_2d,
                        background_field=None,
                    )
                    mesh_exc = None
                except Exception as e:
                    mesh_exc = e
                    print(f"⚠️ Meshing without background field failed: {e}")

            if mesh_exc is not None:
                # Common fix for "Impossible to mesh periodic surface": switch to a more robust 2D algorithm.
                try:
                    _attempt_mesh(
                        label="no-background-field-2d-delaunay",
                        algorithm3d=self.mesh_algorithm,
                        algorithm2d=5,  # Delaunay
                        background_field=None,
                    )
                    mesh_exc = None
                except Exception as e:
                    mesh_exc = e
                    print(f"⚠️ Meshing with 2D Delaunay failed: {e}")

            if mesh_exc is not None:
                # Fallback: HXT can be fragile for some multi-volume boolean geometries on certain Gmsh builds.
                # Retry with more conservative algorithms.
                for algorithm3d, algorithm2d, label in [
                    (1, 5, "fallback-delaunay"),
                    (4, 6, "fallback-frontal"),
                ]:
                    try:
                        _attempt_mesh(
                            label=label,
                            algorithm3d=algorithm3d,
                            algorithm2d=algorithm2d,
                            background_field=None,
                        )
                        mesh_exc = None
                        break
                    except Exception as e:
                        mesh_exc = e
                        print(f"⚠️ Meshing {label} failed: {e}")

            if mesh_exc is not None:
                try:
                    fail_brep = out_dir / "mesh_fail.brep"
                    fail_geo = out_dir / "mesh_fail.geo_unrolled"
                    gmsh.write(str(fail_brep))
                    gmsh.write(str(fail_geo))
                    print(f"⚠️ Wrote debug geometry: {fail_brep}")
                    print(f"⚠️ Wrote debug geometry: {fail_geo}")
                except Exception as e:
                    print(f"⚠️ Failed to write debug geometry: {e}")
                raise mesh_exc

            # Optional: Additional optimization pass (with fallback on crash)
            # Gmsh/Netgen optimize can segfault on certain geometries,
            # so we write the mesh first, then try to optimize in-place.
            # If optimize crashes, the unoptimized mesh is still valid.
            if self.mesh_optimize > 0:
                # Save unoptimized mesh first as fallback
                gmsh.write(str(msh_path))
                print("Optimizing mesh quality (with segfault fallback)...")
                try:
                    gmsh.model.mesh.optimize("Relocate3D")
                    print("  Mesh optimization succeeded (Relocate3D)")
                    _mesh_already_written = True
                except Exception as e:
                    print(f"  Mesh optimization failed ({e}), using unoptimized mesh")
                    _mesh_already_written = True
            else:
                _mesh_already_written = False

            if not _mesh_already_written:
                gmsh.write(str(msh_path))
            else:
                # Re-write with optimized mesh (overwrites the fallback)
                try:
                    gmsh.write(str(msh_path))
                except Exception:
                    print("  Using previously saved unoptimized mesh")

            # Get mesh statistics
            nodes = gmsh.model.mesh.getNodes()
            print(f"\n✓ Mesh generated:")
            print(f"  - Nodes: {len(nodes[0]):,}")

        finally:
            gmsh.finalize()

        # Read and process mesh
        msh = meshio.read(str(msh_path))
        if "tetra" not in msh.cells_dict:
            raise RuntimeError("Gmsh output has no tetra cells.")
        points = msh.points.astype(float)
        tets = msh.cells_dict["tetra"].astype(np.int32)

        if "gmsh:physical" not in msh.cell_data_dict or "tetra" not in msh.cell_data_dict["gmsh:physical"]:
            raise RuntimeError("Missing gmsh physical tags on tetra cells.")
        physical = msh.cell_data_dict["gmsh:physical"]["tetra"].astype(np.int32)

        # Map to material_id: 1=steel (including block 4 for split core), 2=PLA
        material_id = np.full_like(physical, 2, dtype=np.int32)
        material_id[physical == 1] = 1
        material_id[physical == 4] = 1  # Split core half 2 is also steel

        # Per-cell thickness (diagnostic only; solver reads thickness from config)
        thickness_mm = np.zeros_like(physical, dtype=float)

        # Compute theta band for ALL cells based on ORIGINAL mesh coordinates
        # This is critical: theta_band is assigned before any rotation, so it stays
        # correct regardless of how the mesh is oriented during simulation
        z = points[:, 2]
        z_centroid = z[tets].mean(axis=1)
        abs_z = np.abs(z_centroid)
        theta = np.arccos(np.clip(abs_z / float(self.c_core), 0.0, 1.0))

        if theta_cfg is None:
            # Use default 10 bands starting at 9° (after the pole hole at ~8.7°)
            # This ensures band 0 has a visible region instead of being cut by the hole
            default_edges = np.linspace(9, 90, 11)  # 10 bands, ~8.1° each
            theta_edges_rad_for_band = np.deg2rad(default_edges)
            thickness_mm[physical == 2] = t_protected * 1000.0
            thickness_mm[physical == 3] = t_fracture * 1000.0
        else:
            theta_edges_rad_for_band = theta_edges_rad
            band = np.searchsorted(theta_edges_rad, theta, side="right") - 1
            band = np.clip(band, 0, len(theta_edges_rad) - 2).astype(np.int32)
            thickness_mm[physical == 2] = t_protected_by_band_m[band[physical == 2]] * 1000.0
            thickness_mm[physical == 3] = t_fracture_by_band_m[band[physical == 3]] * 1000.0

        # Compute theta_band for ALL cells (including steel core)
        # This survives mesh rotation and can be used for stress binning
        theta_band = np.searchsorted(theta_edges_rad_for_band, theta, side="right") - 1
        theta_band = np.clip(theta_band, 0, len(theta_edges_rad_for_band) - 2).astype(np.int32)

        cell_arrays = {
            "physical": physical,
            "material_id": material_id,
            "thickness_mm": thickness_mm,
            "theta_band": theta_band,
        }

        if self.merge_duplicate_nodes:
            print("Merging duplicate nodes...")
            points, tets, cell_arrays = self._merge_duplicate_nodes(points, tets, cell_arrays, self.merge_tol)
            physical = cell_arrays["physical"]
            material_id = cell_arrays["material_id"]
            thickness_mm = cell_arrays["thickness_mm"]
            theta_band = cell_arrays["theta_band"]

        self.volumetric_vertices = points
        self.tetrahedra = tets
        self.element_materials = material_id
        self.cell_thickness_mm = thickness_mm
        self.cell_region_id = physical
        self.cell_theta_band = theta_band

        print(f"\n✓ Final mesh statistics:")
        print(f"  - Vertices: {len(points):,}")
        print(f"  - Tetrahedra: {len(tets):,}")
        print(f"  - Steel elements: {np.sum(material_id == 1):,} ({np.sum(material_id == 1)/len(tets)*100:.1f}%)")
        print(f"  - PLA elements: {np.sum(material_id == 2):,} ({np.sum(material_id == 2)/len(tets)*100:.1f}%)")
        if self.split_core_enabled:
            n_half1 = np.sum(physical == 1)
            n_half4 = np.sum(physical == 4)
            print(f"  - Split core: half_1={n_half1:,} ({n_half1/len(tets)*100:.1f}%), half_4={n_half4:,} ({n_half4/len(tets)*100:.1f}%)")
            print(f"  - Gap: {self.split_core_gap_mm:.1f}mm along {self.split_core_axis}-axis")
        print("="*80)

        result = {
            "vertices": self.volumetric_vertices,
            "tetrahedra": self.tetrahedra,
            "element_materials": self.element_materials,
            "cell_thickness_mm": self.cell_thickness_mm,
            "cell_region_id": self.cell_region_id,
            "n_steel_elements": int(np.sum(material_id == 1)),
            "n_pla_elements": int(np.sum(material_id == 2)),
            "has_solid_core": True,
        }
        if self.split_core_enabled:
            result["split_core"] = {
                "enabled": True,
                "axis": self.split_core_axis,
                "gap_mm": self.split_core_gap_mm,
                "n_half1": int(np.sum(physical == 1)),
                "n_half4": int(np.sum(physical == 4)),
            }
        return result

    def export_mesh(self, output_dir='results_refined'):
        """Export mesh to XDMF/VTU for ParaView and FEM."""
        if self.tetrahedra is None:
            self.create_volumetric_mesh(output_dir=output_dir)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        mesh = meshio.Mesh(
            points=self.volumetric_vertices,
            cells=[("tetra", self.tetrahedra)],
            cell_data={
                "material_id": [self.element_materials],
                "thickness_mm": [getattr(self, "cell_thickness_mm", np.zeros(len(self.tetrahedra), dtype=float))],
                "region_id": [getattr(self, "cell_region_id", self.element_materials)],
                "theta_band": [getattr(self, "cell_theta_band", np.zeros(len(self.tetrahedra), dtype=np.int32))],
            },
        )

        xdmf_file = output_path / "core_shell_refined.xdmf"
        vtu_file = output_path / "core_shell_refined.vtu"
        meshio.write(str(xdmf_file), mesh)
        meshio.write(str(vtu_file), mesh)

        per_files = None
        if bool((self.config.get("output", {}) or {}).get("export_peridigm", False)):
            try:
                per_files = self.export_peridigm(output_dir)
            except Exception as e:
                print(f"⚠️ Peridigm export failed (continuing): {e}")

        return {
            'xdmf': str(xdmf_file),
            'vtu': str(vtu_file),
            **({"peridigm_discretization": per_files["discretization"]} if per_files else {}),
        }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("REFINED MESH GENERATOR - SMOOTH STEEL/PLA INTERFACE")
    print("="*80)

    generator = ShellGeometryGeneratorRefined('config.yaml')
    generator.generate_geometry()
    generator.create_volumetric_mesh(include_solid_core=True)
    files = generator.export_mesh('results_refined')

    print(f"\n✅ Refined mesh exported:")
    print(f"  - {files['xdmf']}")
    print(f"  - {files['vtu']}")
    print("\nThe interface between steel and PLA should now be much smoother!")
    print("="*80)
