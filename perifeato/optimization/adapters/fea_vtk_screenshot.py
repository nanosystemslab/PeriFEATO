"""
Generate PNG screenshots from VTK output using PyVista.

This module provides functionality to automatically generate visualization
images from FEA results without needing to transfer large VTK files.
"""

from pathlib import Path
import numpy as np


def generate_stress_screenshots(vtk_path: Path, output_dir: Path, orientation_name: str):
    """
    Generate PNG screenshots of stress field from VTK output.

    Args:
        vtk_path: Path to .pvd file (ParaView data file)
        output_dir: Directory to save PNG screenshots
        orientation_name: Name of orientation (for filename)

    Creates:
        - {orientation}_stress_overview.png: Overall stress distribution
        - {orientation}_stress_max.png: Zoomed view of maximum stress region
        - {orientation}_displacement.png: Displacement magnitude
    """
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True  # Headless rendering on HPC
    except ImportError:
        print(f"[Screenshot] PyVista not available, skipping PNG generation")
        return

    if not vtk_path.exists():
        print(f"[Screenshot] VTK file not found: {vtk_path}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Read the VTK data (final timestep)
        reader = pv.get_reader(str(vtk_path))
        data = reader.read()[-1]  # Get last timestep

        # Handle multi-block datasets (parallel VTU files)
        if isinstance(data, pv.MultiBlock):
            print(f"[Screenshot] Multi-block dataset detected, merging blocks...")
            mesh = data.combine()  # Merge all blocks into single mesh
        else:
            mesh = data

        if mesh.n_points == 0 or mesh.n_cells == 0:
            print(f"[Screenshot] Empty mesh, skipping")
            return

        # Debug: Print available fields
        print(f"[Screenshot] Available point data: {list(mesh.point_data.keys())}")
        print(f"[Screenshot] Available cell data: {list(mesh.cell_data.keys())}")

        # Try to find stress field (multiple possible names)
        stress_field_name = None
        possible_names = ['von_mises', 'stress', 'von_mises_stress', 'sigma_vm', 'VonMises']

        for name in possible_names:
            if name in mesh.point_data or name in mesh.cell_data:
                stress_field_name = name
                print(f"[Screenshot] Found stress field: {name}")
                break

        if stress_field_name is None:
            print(f"[Screenshot] No stress field found in VTK")
            print(f"[Screenshot] Tried: {possible_names}")
            print(f"[Screenshot] Available: point_data={list(mesh.point_data.keys())}, cell_data={list(mesh.cell_data.keys())}")
            return

        # Get stress field (try cell data first, then point data)
        if stress_field_name in mesh.cell_data:
            stress_field = mesh.cell_data[stress_field_name]
            mesh = mesh.cell_data_to_point_data()  # Convert for better visualization
        else:
            stress_field = mesh.point_data[stress_field_name]

        max_stress = np.max(stress_field) if len(stress_field) > 0 else 1.0
        max_stress_mpa = max_stress / 1e6

        print(f"[Screenshot] Generating images for {orientation_name} (max stress: {max_stress_mpa:.2f} MPa)")

        # === 1. Overview with stress ===
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        plotter.add_mesh(
            mesh,
            scalars=stress_field_name,
            cmap='turbo',
            clim=[0, max_stress],
            show_edges=False,
            scalar_bar_args={
                'title': 'Von Mises Stress (Pa)',
                'title_font_size': 20,
                'label_font_size': 16,
                'n_labels': 5,
                'fmt': '%.2e',
            }
        )
        plotter.add_text(
            f"{orientation_name}\nMax Stress: {max_stress_mpa:.2f} MPa",
            position='upper_left',
            font_size=14,
            color='black'
        )
        plotter.camera_position = 'iso'
        plotter.background_color = 'white'

        screenshot_path = output_dir / f"{orientation_name}_stress_overview.png"
        plotter.screenshot(str(screenshot_path))
        plotter.close()
        print(f"[Screenshot]   → {screenshot_path.name}")

        # === 2. Top view with stress ===
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        plotter.add_mesh(
            mesh,
            scalars=stress_field_name,
            cmap='turbo',
            clim=[0, max_stress],
            show_edges=False,
            scalar_bar_args={
                'title': 'Von Mises Stress (Pa)',
                'title_font_size': 20,
                'label_font_size': 16,
                'n_labels': 5,
                'fmt': '%.2e',
            }
        )
        plotter.add_text(
            f"{orientation_name} - Top View\nMax Stress: {max_stress_mpa:.2f} MPa",
            position='upper_left',
            font_size=14,
            color='black'
        )
        plotter.camera_position = 'xy'  # Top view
        plotter.background_color = 'white'

        screenshot_path = output_dir / f"{orientation_name}_stress_topview.png"
        plotter.screenshot(str(screenshot_path))
        plotter.close()
        print(f"[Screenshot]   → {screenshot_path.name}")

        # === 3. Side view with stress ===
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        plotter.add_mesh(
            mesh,
            scalars=stress_field_name,
            cmap='turbo',
            clim=[0, max_stress],
            show_edges=False,
            scalar_bar_args={
                'title': 'Von Mises Stress (Pa)',
                'title_font_size': 20,
                'label_font_size': 16,
                'n_labels': 5,
                'fmt': '%.2e',
            }
        )
        plotter.add_text(
            f"{orientation_name} - Side View\nMax Stress: {max_stress_mpa:.2f} MPa",
            position='upper_left',
            font_size=14,
            color='black'
        )
        plotter.camera_position = 'xz'  # Side view
        plotter.background_color = 'white'

        screenshot_path = output_dir / f"{orientation_name}_stress_sideview.png"
        plotter.screenshot(str(screenshot_path))
        plotter.close()
        print(f"[Screenshot]   → {screenshot_path.name}")

        # === 4. Displacement (if available) ===
        if 'displacement' in mesh.array_names or 'u' in mesh.array_names:
            disp_name = 'displacement' if 'displacement' in mesh.array_names else 'u'

            plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
            plotter.add_mesh(
                mesh,
                scalars=disp_name,
                cmap='viridis',
                show_edges=False,
                scalar_bar_args={
                    'title': 'Displacement Magnitude (m)',
                    'title_font_size': 20,
                    'label_font_size': 16,
                    'n_labels': 5,
                    'fmt': '%.2e',
                }
            )
            plotter.add_text(
                f"{orientation_name}\nDisplacement",
                position='upper_left',
                font_size=14,
                color='black'
            )
            plotter.camera_position = 'iso'
            plotter.background_color = 'white'

            screenshot_path = output_dir / f"{orientation_name}_displacement.png"
            plotter.screenshot(str(screenshot_path))
            plotter.close()
            print(f"[Screenshot]   → {screenshot_path.name}")

        print(f"[Screenshot] ✓ Generated {orientation_name} screenshots")

    except Exception as e:
        print(f"[Screenshot] Error generating screenshots: {e}")
        import traceback
        traceback.print_exc()


def generate_multi_orientation_comparison(all_vtk_paths: dict, output_dir: Path):
    """
    Generate comparison plot showing all orientations side-by-side.

    Args:
        all_vtk_paths: Dict mapping orientation_name -> vtk_path
        output_dir: Directory to save comparison image
    """
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
    except ImportError:
        return

    if not all_vtk_paths:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create subplot for each orientation
        n_orientations = len(all_vtk_paths)
        plotter = pv.Plotter(
            off_screen=True,
            shape=(1, min(n_orientations, 5)),  # Max 5 columns
            window_size=[1920 * min(n_orientations, 3), 1080]
        )

        max_stress_global = 0.0
        meshes = {}
        stress_field_name = None
        possible_names = ['von_mises', 'stress', 'von_mises_stress', 'sigma_vm', 'VonMises']

        # Load all meshes and find global max stress
        for orient_name, vtk_path in all_vtk_paths.items():
            if not Path(vtk_path).exists():
                continue

            reader = pv.get_reader(str(vtk_path))
            data = reader.read()[-1]  # Last timestep

            # Handle multi-block datasets
            if isinstance(data, pv.MultiBlock):
                mesh = data.combine()
            else:
                mesh = data

            # Find stress field name
            if stress_field_name is None:
                for name in possible_names:
                    if name in mesh.point_data or name in mesh.cell_data:
                        stress_field_name = name
                        break

            if stress_field_name is None:
                continue  # Skip if no stress field

            # Convert cell data to point data if needed
            if stress_field_name in mesh.cell_data:
                mesh = mesh.cell_data_to_point_data()

            if stress_field_name in mesh.array_names:
                max_stress = np.max(mesh[stress_field_name])
                max_stress_global = max(max_stress_global, max_stress)
                meshes[orient_name] = mesh

        if not meshes or stress_field_name is None:
            return  # No valid meshes to plot

        # Plot each orientation with same color scale
        for idx, (orient_name, mesh) in enumerate(meshes.items()):
            plotter.subplot(0, idx)
            plotter.add_mesh(
                mesh,
                scalars=stress_field_name,
                cmap='turbo',
                clim=[0, max_stress_global],
                show_edges=False,
            )
            max_stress_mpa = np.max(mesh[stress_field_name]) / 1e6
            plotter.add_text(
                f"{orient_name}\n{max_stress_mpa:.2f} MPa",
                font_size=10,
                color='black'
            )
            plotter.camera_position = 'iso'
            plotter.background_color = 'white'

        screenshot_path = output_dir / "all_orientations_comparison.png"
        plotter.screenshot(str(screenshot_path))
        plotter.close()

        print(f"[Screenshot] ✓ Generated comparison: {screenshot_path.name}")

    except Exception as e:
        print(f"[Screenshot] Error generating comparison: {e}")
