"""
VTK stress field reader with PyVista fallback for PVTU support.
"""

from pathlib import Path
import os
import numpy as np
import xml.etree.ElementTree as ET

_STRESS_ARRAY_NAMES = ("max_stress", "von_mises", "von_mises_stress", "vm", "VonMises")
_MATERIAL_ARRAY_NAMES = ("material_id", "material", "mat_id", "MaterialID")
_THETA_BAND_ARRAY_NAMES = ("theta_band", "ThetaBand", "theta_band_id")
_HEMISPHERE_ARRAY_NAMES = ("hemisphere", "Hemisphere", "hemisphere_id")


def read_stress_field_from_vtk(
    vtk_path: Path, expected_n_cells: int, return_cell_centers: bool = False, return_theta_band: bool = False, return_hemisphere: bool = False
) -> np.ndarray | tuple[np.ndarray, ...]:
    """
    Read von Mises stress field from VTK output (.pvd/.vtu/.pvtu).

    **FILTERS OUT STEEL STRESS**: Sets stress to 0.0 for steel cells (material_id=1)
    to ensure optimization only considers PLA shell stress, not steel boundary condition.

    Args:
        vtk_path: Path to .pvd file (e.g., {orientation}_cell.pvd)
        expected_n_cells: Expected number of cells for validation
        return_cell_centers: If True, also return cell centers from VTK (for z-binning)
        return_theta_band: If True, also return theta_band from VTK (survives rotation)

    Returns:
        Based on flags:
        - Default: stress_field
        - return_cell_centers: (stress_field, cell_centers)
        - return_theta_band: (stress_field, theta_band)
        - Both: (stress_field, cell_centers, theta_band)
    """
    def _make_none_result():
        # Build result tuple based on requested fields
        n_fields = 1  # stress_field
        if return_cell_centers:
            n_fields += 1
        if return_theta_band:
            n_fields += 1
        if return_hemisphere:
            n_fields += 1
        if n_fields == 1:
            return None
        return tuple([None] * n_fields)

    _none_result = _make_none_result()

    if not vtk_path.exists():
        print(f"[VTK Reader] Warning: File not found: {vtk_path}")
        return _none_result

    try:
        # Parse PVD to find the actual VTU/PVTU file
        tree = ET.parse(vtk_path)
        root = tree.getroot()
        datasets = root.findall(".//DataSet")
        if not datasets:
            print(f"[VTK Reader] Warning: No DataSet found in {vtk_path}")
            return _none_result

        dataset = _select_dataset_with_array(datasets, vtk_path.parent, _STRESS_ARRAY_NAMES)
        if dataset is None:
            dataset = _select_latest_dataset(datasets)

        vtu_filename = dataset.get("file")
        if vtu_filename is None:
            print(f"[VTK Reader] Warning: No file attribute in DataSet")
            return _none_result

        vtu_path = vtk_path.parent / vtu_filename
        if not vtu_path.exists():
            print(f"[VTK Reader] Warning: VTU/PVTU file not found: {vtu_path}")
            return _none_result

        debug = os.environ.get("VTK_READER_DEBUG") == "1"

        stress_field = None
        material_id = None
        cell_centers = None
        theta_band = None
        hemisphere = None

        def _finalize(stress_field, material_id, cell_centers, theta_band, hemisphere):
            if stress_field is None:
                return _make_none_result()
            if material_id is None:
                material_id = _read_material_id_from_pvd(datasets, vtk_path.parent, debug)
            if material_id is not None and len(material_id) != len(stress_field):
                print(
                    f"[VTK Reader] Warning: material_id size mismatch: {len(material_id)} != {len(stress_field)}"
                )
                material_id = None
            filtered = _filter_steel_stress(stress_field, material_id)
            # Build result tuple based on requested fields
            result = [filtered]
            if return_cell_centers:
                result.append(cell_centers)
            if return_theta_band:
                result.append(theta_band)
            if return_hemisphere:
                result.append(hemisphere)
            if len(result) == 1:
                return result[0]
            return tuple(result)

        # Try PyVista first (supports all VTK versions including PVTU)
        try:
            import pyvista as pv
            if debug:
                print("[VTK Reader] backend=pyvista")
            stress_field, material_id, cell_centers, theta_band, hemisphere = _read_with_pyvista(
                vtu_path, expected_n_cells, return_cell_centers
            )
            result = _finalize(stress_field, material_id, cell_centers, theta_band, hemisphere)
            if return_cell_centers or return_theta_band or return_hemisphere:
                if isinstance(result, tuple) and result[0] is not None:
                    return result
            elif result is not None:
                return result
        except ImportError:
            pass

        # Try VTK Python bindings (available with DOLFINx)
        try:
            import vtk
            from vtk.util import numpy_support
            if debug:
                print("[VTK Reader] backend=vtk")
            stress_field, material_id, cell_centers = _read_with_vtk(
                vtu_path, expected_n_cells, return_cell_centers
            )
            # VTK backend doesn't support theta_band or hemisphere yet
            theta_band = None
            hemisphere = None
            result = _finalize(stress_field, material_id, cell_centers, theta_band, hemisphere)
            if return_cell_centers or return_theta_band or return_hemisphere:
                if isinstance(result, tuple) and result[0] is not None:
                    return result
            elif result is not None:
                return result
        except ImportError:
            pass

        # Fallback to meshio (limited VTK version support)
        try:
            import meshio
            if debug:
                print("[VTK Reader] backend=meshio")
            stress_field, material_id = _read_with_meshio(vtu_path, expected_n_cells)
            # meshio doesn't easily provide cell centers, theta_band, or hemisphere
            result = _finalize(stress_field, material_id, None, None, None)
            if return_cell_centers or return_theta_band or return_hemisphere:
                if isinstance(result, tuple) and result[0] is not None:
                    return result
            elif result is not None:
                return result
        except Exception as meshio_error:
            # If meshio fails (e.g., VTU version 2.2), try direct XML parsing
            print(f"[VTK Reader] meshio failed ({meshio_error}), trying direct XML parsing...")
            try:
                if debug:
                    print("[VTK Reader] backend=xml")
                stress_field, material_id = _read_with_xml(vtu_path, expected_n_cells)
                # XML parsing doesn't provide cell centers, theta_band, or hemisphere
                result = _finalize(stress_field, material_id, None, None, None)
                if return_cell_centers or return_theta_band or return_hemisphere:
                    if isinstance(result, tuple) and result[0] is not None:
                        return result
                elif result is not None:
                    return result
            except Exception as xml_error:
                print(f"[VTK Reader] Error: All readers failed. meshio: {meshio_error}, xml: {xml_error}")
                return _none_result

        return _none_result

    except Exception as e:
        print(f"[VTK Reader] Error reading {vtk_path}: {e}")
        return _none_result


def _read_with_pyvista(
    vtu_path: Path, expected_n_cells: int, return_cell_centers: bool = False
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Read VTU/PVTU using PyVista (handles all formats)."""
    import pyvista as pv

    # PyVista can read both VTU and PVTU directly
    mesh = pv.read(str(vtu_path))

    # Extract stress field
    stress_field = None
    for name in _STRESS_ARRAY_NAMES:
        if name in mesh.cell_data:
            stress_field = mesh.cell_data[name]
            break

    if stress_field is None:
        print(f"[VTK Reader] Warning: No von Mises stress in {vtu_path}")
        print(f"[VTK Reader] Available: {list(mesh.cell_data.keys())}")
        return None, None, None, None, None

    # Extract material_id (may live in a different dataset)
    material_id = None
    for name in _MATERIAL_ARRAY_NAMES:
        if name in mesh.cell_data:
            material_id = mesh.cell_data[name]
            break

    # Extract theta_band (assigned during mesh generation, survives rotation)
    theta_band = None
    for name in _THETA_BAND_ARRAY_NAMES:
        if name in mesh.cell_data:
            theta_band = mesh.cell_data[name]
            print(f"[VTK Reader] Found theta_band in VTK (correct for rotated meshes)")
            break

    # Extract hemisphere (1=upper/+z, 0=lower/-z, assigned during mesh generation)
    hemisphere = None
    for name in _HEMISPHERE_ARRAY_NAMES:
        if name in mesh.cell_data:
            hemisphere = mesh.cell_data[name]
            print(f"[VTK Reader] Found hemisphere in VTK (correct for rotated meshes)")
            break

    # If theta_band or hemisphere not in VTK, try reading from companion XDMF file
    # DOLFINx VTKFile.write_function with a list doesn't reliably write all functions,
    # so we write theta_band/hemisphere to a separate XDMF file.
    if theta_band is None or hemisphere is None:
        xdmf_path = _find_companion_xdmf(vtu_path)
        if xdmf_path is not None:
            xdmf_theta_band, xdmf_hemisphere = _read_cell_data_from_xdmf(xdmf_path)
            if theta_band is None and xdmf_theta_band is not None:
                theta_band = xdmf_theta_band
                print(f"[VTK Reader] Found theta_band in companion XDMF: {xdmf_path.name}")
            if hemisphere is None and xdmf_hemisphere is not None:
                hemisphere = xdmf_hemisphere
                print(f"[VTK Reader] Found hemisphere in companion XDMF: {xdmf_path.name}")

    # Ensure arrays are 1D
    stress_field = np.asarray(stress_field).flatten()
    if material_id is not None:
        material_id = np.asarray(material_id).flatten()
    if theta_band is not None:
        theta_band = np.asarray(theta_band).flatten().astype(np.int32)
    if hemisphere is not None:
        hemisphere = np.asarray(hemisphere).flatten().astype(np.int32)

    # Extract cell centers if requested (critical for z-binning!)
    cell_centers = None
    if return_cell_centers:
        cell_centers = np.asarray(mesh.cell_centers().points)

    # Validate size
    if len(stress_field) != expected_n_cells:
        print(f"[VTK Reader] Warning: Size mismatch: {len(stress_field)} != {expected_n_cells}")

    return stress_field, material_id, cell_centers, theta_band, hemisphere


def _find_companion_xdmf(vtu_path: Path) -> Path | None:
    """Find the companion XDMF file containing theta_band and hemisphere.

    The FEA code writes theta_band and hemisphere to {orientation}_cell_data.xdmf
    alongside the VTK files.
    """
    # Get the orientation name from the VTU path
    # e.g., vertical_theta0_base_max_stress_p0_000001.vtu -> vertical_theta0_base
    vtu_name = vtu_path.stem
    # Remove the suffix like _max_stress_p0_000001 or _cell_p0_000001
    parts = vtu_name.split('_')
    # Find where the suffix starts (usually at "max" or "cell" or "p0")
    suffix_starts = ['max', 'cell', 'p0']
    orientation_parts = []
    for part in parts:
        if part in suffix_starts:
            break
        orientation_parts.append(part)
    orientation_name = '_'.join(orientation_parts) if orientation_parts else None

    if orientation_name is None:
        return None

    # Look for the XDMF file
    xdmf_path = vtu_path.parent / f"{orientation_name}_cell_data.xdmf"
    if xdmf_path.exists():
        return xdmf_path

    return None


def _read_cell_data_from_xdmf(xdmf_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read theta_band and hemisphere from XDMF file."""
    try:
        import h5py

        # Parse XDMF to find the HDF5 file and dataset paths
        tree = ET.parse(xdmf_path)
        root = tree.getroot()

        theta_band = None
        hemisphere = None

        # Find all Attribute elements with cell data
        for attr in root.findall(".//{*}Attribute"):
            name = attr.get("Name", "")
            center = attr.get("Center", "")
            if center != "Cell":
                continue

            data_item = attr.find("{*}DataItem")
            if data_item is None:
                continue

            # Parse HDF5 reference: "filename.h5:/path/to/data"
            text = data_item.text.strip() if data_item.text else ""
            if ":" not in text:
                continue

            h5_file, h5_path = text.split(":", 1)
            h5_path_full = xdmf_path.parent / h5_file

            if not h5_path_full.exists():
                continue

            try:
                with h5py.File(h5_path_full, "r") as f:
                    if h5_path in f:
                        data = f[h5_path][:]
                        if name.lower() == "theta_band":
                            theta_band = data.flatten().astype(np.int32)
                        elif name.lower() == "hemisphere":
                            hemisphere = data.flatten().astype(np.int32)
            except Exception as e:
                print(f"[VTK Reader] Warning: Error reading {h5_path} from {h5_path_full}: {e}")
                continue

        return theta_band, hemisphere

    except ImportError:
        print(f"[VTK Reader] Warning: h5py not available, cannot read XDMF")
        return None, None
    except Exception as e:
        print(f"[VTK Reader] Warning: Error reading XDMF {xdmf_path}: {e}")
        return None, None


def _select_latest_dataset(datasets) -> ET.Element:
    """Return the DataSet element for the latest timestep (or the last entry)."""
    fallback = datasets[-1]
    best = None
    best_ts = None

    for ds in datasets:
        ts_raw = ds.get("timestep")
        if ts_raw is None:
            continue
        try:
            ts = float(ts_raw)
        except ValueError:
            continue
        if best_ts is None or ts >= best_ts:
            best_ts = ts
            best = ds

    return best if best is not None else fallback


def _select_dataset_with_array(datasets, base_dir: Path, array_names) -> ET.Element | None:
    """Pick the last dataset whose file advertises a target cell array."""
    for ds in reversed(datasets):
        vtu_filename = ds.get("file")
        if not vtu_filename:
            continue
        vtu_path = base_dir / vtu_filename
        if not vtu_path.exists():
            continue
        if _file_has_cell_array(vtu_path, array_names):
            return ds
    return None


def _file_has_cell_array(vtu_path: Path, array_names) -> bool:
    """Check if a VTU/PVTU advertises any of the target cell arrays."""
    try:
        tree = ET.parse(vtu_path)
    except Exception:
        return False

    root = tree.getroot()
    cell_data = root.find(".//PCellData")
    if cell_data is None:
        cell_data = root.find(".//CellData")
    if cell_data is None:
        return False

    data_arrays = cell_data.findall("PDataArray")
    if not data_arrays:
        data_arrays = cell_data.findall("DataArray")

    for data_array in data_arrays:
        if data_array.get("Name") in array_names:
            return True

    return False


def _read_material_id_from_pvd(
    datasets, base_dir: Path, debug: bool
) -> np.ndarray | None:
    """Read material_id from any dataset in the PVD (may be a separate PV(T)U)."""
    dataset = _select_dataset_with_array(datasets, base_dir, _MATERIAL_ARRAY_NAMES)
    if dataset is None:
        return None

    vtu_filename = dataset.get("file")
    if not vtu_filename:
        return None

    vtu_path = base_dir / vtu_filename
    if not vtu_path.exists():
        return None

    material_id = _read_cell_array_from_file(vtu_path, _MATERIAL_ARRAY_NAMES, debug)
    if material_id is not None and debug:
        print(f"[VTK Reader] material_id source={vtu_path}")
    return material_id


def _read_cell_array_from_file(
    vtu_path: Path, array_names, debug: bool
) -> np.ndarray | None:
    """Read a single cell array from a VTU/PVTU using best available backend."""
    try:
        import pyvista as pv
        if debug:
            print("[VTK Reader] backend=pyvista (array lookup)")
        arr = _read_cell_array_with_pyvista(vtu_path, array_names)
        if arr is not None:
            return arr
    except ImportError:
        pass

    try:
        import vtk
        from vtk.util import numpy_support
        if debug:
            print("[VTK Reader] backend=vtk (array lookup)")
        arr = _read_cell_array_with_vtk(vtu_path, array_names)
        if arr is not None:
            return arr
    except ImportError:
        pass

    try:
        import meshio
        if debug:
            print("[VTK Reader] backend=meshio (array lookup)")
        arr = _read_cell_array_with_meshio(vtu_path, array_names)
        if arr is not None:
            return arr
    except Exception:
        pass

    return _read_cell_array_with_xml(vtu_path, array_names)


def _read_cell_array_with_pyvista(vtu_path: Path, array_names) -> np.ndarray | None:
    import pyvista as pv

    mesh = pv.read(str(vtu_path))
    for name in array_names:
        if name in mesh.cell_data:
            return np.asarray(mesh.cell_data[name]).flatten()
    return None


def _read_cell_array_with_vtk(vtu_path: Path, array_names) -> np.ndarray | None:
    import vtk
    from vtk.util import numpy_support

    if str(vtu_path).endswith(".pvtu"):
        reader = vtk.vtkXMLPUnstructuredGridReader()
    else:
        reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()

    mesh = reader.GetOutput()
    cell_data = mesh.GetCellData()
    for name in array_names:
        arr = cell_data.GetArray(name)
        if arr is not None:
            return numpy_support.vtk_to_numpy(arr)
    return None


def _read_cell_array_with_meshio(vtu_path: Path, array_names) -> np.ndarray | None:
    import meshio

    if str(vtu_path).endswith(".pvtu"):
        return None

    mesh = meshio.read(vtu_path)
    for name in array_names:
        if name in mesh.cell_data:
            arr = mesh.cell_data[name]
            if isinstance(arr, list):
                return np.concatenate([b for b in arr if len(b) > 0])
            if isinstance(arr, dict):
                return np.concatenate([b for b in arr.values() if len(b) > 0])
            return arr
    return None


def _read_cell_array_with_xml(vtu_path: Path, array_names) -> np.ndarray | None:
    if str(vtu_path).endswith(".pvtu"):
        return None

    try:
        tree = ET.parse(vtu_path)
        root = tree.getroot()
        piece = root.find(".//Piece")
        if piece is None:
            return None
        cell_data = piece.find("CellData")
        if cell_data is None:
            return None
        for name in array_names:
            for data_array in cell_data.findall("DataArray"):
                if data_array.get("Name") == name:
                    return _parse_data_array(data_array)
    except Exception:
        return None

    return None


def _read_with_meshio(vtu_path: Path, expected_n_cells: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Read VTU using meshio (limited format support)."""
    import meshio

    # meshio doesn't support PVTU - only works for serial VTU
    if str(vtu_path).endswith('.pvtu'):
        print(f"[VTK Reader] Warning: meshio doesn't support PVTU format")
        return None, None

    mesh = meshio.read(vtu_path)

    # Extract stress
    stress_field = None
    for name in _STRESS_ARRAY_NAMES:
        if name in mesh.cell_data:
            stress_field = mesh.cell_data[name]
            break

    if stress_field is None:
        print(f"[VTK Reader] Warning: No von Mises stress in {vtu_path}")
        return None, None

    # Extract material_id
    material_id = None
    for name in _MATERIAL_ARRAY_NAMES:
        if name in mesh.cell_data:
            material_id = mesh.cell_data[name]
            break

    # Flatten (meshio may return per-cell-type blocks)
    if isinstance(stress_field, (list, dict)):
        if isinstance(stress_field, list):
            stress_field = np.concatenate([b for b in stress_field if len(b) > 0])
        else:
            stress_field = np.concatenate([b for b in stress_field.values() if len(b) > 0])

    if material_id is not None and isinstance(material_id, (list, dict)):
        if isinstance(material_id, list):
            material_id = np.concatenate([b for b in material_id if len(b) > 0])
        else:
            material_id = np.concatenate([b for b in material_id.values() if len(b) > 0])

    # Validate size
    if len(stress_field) != expected_n_cells:
        print(f"[VTK Reader] Warning: Size mismatch: {len(stress_field)} != {expected_n_cells}")

    return stress_field, material_id


def _read_with_vtk(
    vtu_path: Path, expected_n_cells: int, return_cell_centers: bool = False
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read VTU/PVTU using VTK Python bindings (available with DOLFINx)."""
    import vtk
    from vtk.util import numpy_support

    # Choose reader based on file extension
    if str(vtu_path).endswith('.pvtu'):
        reader = vtk.vtkXMLPUnstructuredGridReader()
    else:
        reader = vtk.vtkXMLUnstructuredGridReader()

    reader.SetFileName(str(vtu_path))
    reader.Update()

    mesh = reader.GetOutput()
    cell_data = mesh.GetCellData()

    # Extract stress field
    stress_array = None
    for name in _STRESS_ARRAY_NAMES:
        stress_array = cell_data.GetArray(name)
        if stress_array is not None:
            break

    if stress_array is None:
        print(f"[VTK Reader] Warning: No von Mises stress in {vtu_path}")
        available = [cell_data.GetArrayName(i) for i in range(cell_data.GetNumberOfArrays())]
        print(f"[VTK Reader] Available: {available}")
        return None, None, None

    stress_field = numpy_support.vtk_to_numpy(stress_array)

    # Extract material_id
    material_id = None
    for name in _MATERIAL_ARRAY_NAMES:
        mat_array = cell_data.GetArray(name)
        if mat_array is not None:
            material_id = numpy_support.vtk_to_numpy(mat_array)
            break

    # Extract cell centers if requested (critical for z-binning!)
    cell_centers = None
    if return_cell_centers:
        cell_centers_filter = vtk.vtkCellCenters()
        cell_centers_filter.SetInputData(mesh)
        cell_centers_filter.Update()
        points = cell_centers_filter.GetOutput().GetPoints()
        cell_centers = numpy_support.vtk_to_numpy(points.GetData())

    # Validate size
    if len(stress_field) != expected_n_cells:
        print(f"[VTK Reader] Warning: Size mismatch: {len(stress_field)} != {expected_n_cells}")

    return stress_field, material_id, cell_centers


def _read_with_xml(vtu_path: Path, expected_n_cells: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Read VTU using direct XML parsing (works for any VTU version including 2.2)."""
    import base64
    import struct

    tree = ET.parse(vtu_path)
    root = tree.getroot()

    # Find Piece element
    piece = root.find(".//Piece")
    if piece is None:
        print(f"[VTK Reader] Error: No Piece element found in {vtu_path}")
        return None, None

    # Find CellData
    cell_data = piece.find("CellData")
    if cell_data is None:
        print(f"[VTK Reader] Error: No CellData found in {vtu_path}")
        return None, None

    # Extract stress array
    stress_field = None
    for name in _STRESS_ARRAY_NAMES:
        for data_array in cell_data.findall("DataArray"):
            if data_array.get("Name") == name:
                stress_field = _parse_data_array(data_array)
                break
        if stress_field is not None:
            break

    if stress_field is None:
        available = [da.get("Name") for da in cell_data.findall("DataArray")]
        print(f"[VTK Reader] Error: No von Mises stress found in {vtu_path}")
        print(f"[VTK Reader] Available: {available}")
        return None, None

    # Extract material_id
    material_id = None
    for name in _MATERIAL_ARRAY_NAMES:
        for data_array in cell_data.findall("DataArray"):
            if data_array.get("Name") == name:
                material_id = _parse_data_array(data_array)
                break
        if material_id is not None:
            break

    # Validate size
    if len(stress_field) != expected_n_cells:
        print(f"[VTK Reader] Warning: Size mismatch: {len(stress_field)} != {expected_n_cells}")

    print(f"[VTK Reader] Successfully parsed VTU with direct XML (version {root.get('version', 'unknown')})")

    return stress_field, material_id


def _parse_data_array(data_array_elem) -> np.ndarray:
    """Parse a VTK DataArray element (handles base64 encoding)."""
    import base64
    import struct

    dtype_map = {
        "Float32": np.float32,
        "Float64": np.float64,
        "Int8": np.int8,
        "Int16": np.int16,
        "Int32": np.int32,
        "Int64": np.int64,
        "UInt8": np.uint8,
        "UInt16": np.uint16,
        "UInt32": np.uint32,
        "UInt64": np.uint64,
    }

    vtype = data_array_elem.get("type")
    if vtype not in dtype_map:
        raise ValueError(f"Unsupported data type: {vtype}")

    dtype = dtype_map[vtype]
    encoding = data_array_elem.get("format", "ascii")

    # Get the text content (base64 or ASCII)
    text = data_array_elem.text
    if text is None:
        raise ValueError("DataArray has no text content")

    text = text.strip()

    if encoding == "binary":
        # Base64 encoded binary data
        # VTK binary format: [header_uint32][data...]
        decoded = base64.b64decode(text)

        # First 4 bytes are the data size in bytes (little-endian uint32)
        if len(decoded) < 4:
            raise ValueError("Binary data too short (no header)")

        data_size = struct.unpack('<I', decoded[:4])[0]
        binary_data = decoded[4:4+data_size]

        # Convert to numpy array
        arr = np.frombuffer(binary_data, dtype=dtype)
        return arr

    elif encoding == "ascii":
        # ASCII space-separated values
        values = [dtype(x) for x in text.split()]
        return np.array(values, dtype=dtype)

    else:
        raise ValueError(f"Unsupported encoding: {encoding}")


def _filter_steel_stress(stress_field: np.ndarray, material_id: np.ndarray) -> np.ndarray:
    """Zero out steel stress (material_id=1), keep only PLA stress (material_id=2)."""
    if material_id is None or len(material_id) != len(stress_field):
        print(f"[VTK Reader] WARNING: No material_id - using unfiltered stress (may include steel!)")
        return stress_field

    steel_mask = (material_id == 1)
    pla_mask = (material_id == 2)
    n_steel = np.sum(steel_mask)
    n_pla = np.sum(pla_mask)

    # Get stress statistics BEFORE filtering
    max_stress_steel = np.max(stress_field[steel_mask]) if n_steel > 0 else 0.0
    max_stress_pla = np.max(stress_field[pla_mask]) if n_pla > 0 else 0.0

    # Zero out steel stress
    stress_field_filtered = stress_field.copy()
    stress_field_filtered[steel_mask] = 0.0

    print(f"[VTK Reader] Filtered steel stress: {n_steel} steel cells (max={max_stress_steel/1e6:.1f} MPa), "
          f"{n_pla} PLA cells (max={max_stress_pla/1e6:.1f} MPa)")

    return stress_field_filtered
