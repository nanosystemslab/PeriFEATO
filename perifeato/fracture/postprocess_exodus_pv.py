#!/usr/bin/env pvpython
"""
Post-process a Peridigm Exodus file with ParaView to extract basic fracture metrics.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from paraview.simple import GetAnimationScene, MergeBlocks, OpenDataFile, UpdatePipeline
from paraview.servermanager import Fetch
from vtk.util.numpy_support import vtk_to_numpy


def _iter_datasets(data_object):
    if data_object is None:
        return
    if data_object.IsA("vtkPartitionedDataSetCollection"):
        try:
            count = data_object.GetNumberOfPartitionedDataSets()
        except Exception:
            count = 0
        for i in range(count):
            pds = data_object.GetPartitionedDataSet(i)
            if pds is not None:
                yield from _iter_datasets(pds)
        return
    if data_object.IsA("vtkPartitionedDataSet"):
        try:
            count = data_object.GetNumberOfPartitions()
        except Exception:
            count = 0
        for i in range(count):
            part = data_object.GetPartition(i)
            if part is not None:
                yield from _iter_datasets(part)
        return
    if data_object.IsA("vtkDataSet"):
        yield data_object
        return
    if data_object.IsA("vtkCompositeDataSet"):
        it = data_object.NewIterator()
        it.UnRegister(None)
        it.InitTraversal()
        while not it.IsDoneWithTraversal():
            current = it.GetCurrentDataObject()
            if current is not None:
                if current.IsA("vtkDataSet"):
                    yield current
                else:
                    yield from _iter_datasets(current)
            it.GoToNextItem()
        return
    # Fallback: treat as single dataset-like object.
    yield data_object


def _collect_array(dataset, name, location):
    if location == "cell":
        data = dataset.GetCellData()
    else:
        data = dataset.GetPointData()
    arr = data.GetArray(name) if data is not None else None
    if arr is None:
        return None
    return vtk_to_numpy(arr)


def _collect_numeric_from_datasets(datasets, name, location):
    arrays = []
    for ds in datasets:
        arr = _collect_array(ds, name, location)
        if arr is not None and arr.size:
            arrays.append(arr)
    if not arrays:
        return None
    return arrays


def _collect_first_available_from_datasets(datasets, names, location):
    for name in names:
        arrays = _collect_numeric_from_datasets(datasets, name, location)
        if arrays is not None:
            return name, arrays
    return None, None


def _fetch_datasets(reader, time_val):
    if math.isnan(time_val):
        UpdatePipeline(reader)
    else:
        UpdatePipeline(float(time_val), reader)
    data = Fetch(reader)
    datasets = list(_iter_datasets(data))
    if datasets:
        return datasets, data, "direct"

    merged = MergeBlocks(Input=reader)
    if math.isnan(time_val):
        UpdatePipeline(merged)
    else:
        UpdatePipeline(float(time_val), merged)
    data = Fetch(merged)
    datasets = list(_iter_datasets(data))
    return datasets, data, "merged"


def _enable_reader_arrays(reader):
    for prop_name in ("PointVariables", "ElementVariables", "PointArrayStatus", "CellArrayStatus"):
        prop = getattr(reader, prop_name, None)
        if prop is None:
            continue
        available = None
        try:
            available = list(prop.Available)
        except Exception:
            available = None
        if available:
            try:
                setattr(reader, prop_name, available)
            except Exception:
                continue


def _mag_max(arrays):
    max_val = None
    for arr in arrays:
        if arr.ndim == 1:
            local_max = float(arr.max())
        else:
            local_max = float((arr * arr).sum(axis=1).max() ** 0.5)
        max_val = local_max if max_val is None else max(max_val, local_max)
    return max_val if max_val is not None else math.nan


def _scalar_max(arrays):
    max_val = None
    for arr in arrays:
        local_max = float(arr.max())
        max_val = local_max if max_val is None else max(max_val, local_max)
    return max_val if max_val is not None else math.nan


def _damage_metrics(damage_arrays, vol_arrays, threshold):
    max_damage = _scalar_max(damage_arrays)
    damage_mask_count = 0
    total_count = 0
    for arr in damage_arrays:
        total_count += arr.size
        damage_mask_count += int((arr >= threshold).sum())

    damage_fraction = (
        float(damage_mask_count) / float(total_count) if total_count else math.nan
    )

    if not vol_arrays:
        return max_damage, damage_fraction, math.nan, math.nan

    total_vol = 0.0
    damaged_vol = 0.0
    for darr, varr in zip(damage_arrays, vol_arrays):
        if darr.size != varr.size:
            continue
        total_vol += float(varr.sum())
        damaged_vol += float(varr[darr >= threshold].sum())

    damage_vol_frac = damaged_vol / total_vol if total_vol else math.nan
    return max_damage, damage_fraction, damaged_vol, damage_vol_frac


def main():
    parser = argparse.ArgumentParser(
        description="Extract basic fracture metrics from Peridigm Exodus output."
    )
    parser.add_argument("--input", required=True, help="Path to Exodus file (.e.*)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--damage-threshold", type=float, default=0.3, help="Damage threshold"
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=-1,
        help="Time index to sample when not using --time-series (default: last)",
    )
    parser.add_argument(
        "--time-series",
        action="store_true",
        help="Write one row per available timestep",
    )
    parser.add_argument(
        "--split-damage-fraction",
        type=float,
        default=0.2,
        help="Damage fraction threshold for split heuristic",
    )
    parser.add_argument(
        "--split-max-damage",
        type=float,
        default=0.95,
        help="Max damage threshold for split heuristic",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print dataset array diagnostics to stdout",
    )
    args = parser.parse_args()

    reader = OpenDataFile(args.input)
    # Prime available arrays before enabling all.
    try:
        UpdatePipeline(reader)
    except Exception:
        pass
    _enable_reader_arrays(reader)
    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    times = list(getattr(reader, "TimestepValues", None) or [])

    if not times:
        times = [math.nan]

    if not args.time_series:
        idx = args.time_index
        if idx < 0:
            idx = len(times) - 1
        idx = max(0, min(idx, len(times) - 1))
        times = [times[idx]]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time",
                "damage_threshold",
                "max_damage",
                "damage_fraction",
                "damaged_volume",
                "damaged_volume_fraction",
                "max_force_density",
                "max_contact_force",
                "max_displacement",
                "max_velocity",
                "split_flag",
                "damage_basis",
                "damage_field",
                "volume_field",
                "contact_field",
            ]
        )

        for step_idx, time_val in enumerate(times):
            datasets, data, fetch_mode = _fetch_datasets(reader, time_val)
            if data is None or not datasets:
                raise SystemExit("ERROR: failed to fetch datasets from reader.")

            if args.debug and step_idx == 0:
                print("=== Debug: reader arrays ===")
                try:
                    print("Reader PointData keys:", list(reader.PointData.keys()))
                    print("Reader CellData keys:", list(reader.CellData.keys()))
                except Exception:
                    print("Reader arrays: unavailable")

                print("=== Debug: fetched datasets ===")
                print("Fetch mode:", fetch_mode)
                try:
                    print("Fetched data class:", data.GetClassName())
                except Exception:
                    print("Fetched data class: unknown")
                for ds in datasets:
                    try:
                        print("Dataset:", ds.GetClassName())
                        pnames = []
                        cnames = []
                        if ds.GetPointData() is not None:
                            pnames = [
                                ds.GetPointData().GetArrayName(i)
                                for i in range(ds.GetPointData().GetNumberOfArrays())
                            ]
                        if ds.GetCellData() is not None:
                            cnames = [
                                ds.GetCellData().GetArrayName(i)
                                for i in range(ds.GetCellData().GetNumberOfArrays())
                            ]
                        print("  Point arrays:", pnames)
                        print("  Cell arrays:", cnames)
                    except Exception as exc:
                        print("  Dataset inspect failed:", exc)

            damage_name, damage_arrays = _collect_first_available_from_datasets(
                datasets, ["Damage"], "cell"
            )
            damage_basis = "cell"
            if damage_arrays is None:
                damage_name, damage_arrays = _collect_first_available_from_datasets(
                    datasets, ["Damage"], "point"
                )
                damage_basis = "point"
            if damage_arrays is None:
                damage_basis = "none"

            vol_names = ["Weighted_Volume", "Volume", "Cell_Volume"]
            vol_name, vol_arrays = (None, None)
            if damage_basis == "cell":
                vol_name, vol_arrays = _collect_first_available_from_datasets(
                    datasets, vol_names, "cell"
                )
            elif damage_basis == "point":
                vol_name, vol_arrays = _collect_first_available_from_datasets(
                    datasets, vol_names, "point"
                )

            force_name, force_arrays = _collect_first_available_from_datasets(
                datasets, ["Force_Density"], "point"
            )
            contact_name, contact_force_arrays = _collect_first_available_from_datasets(
                datasets, ["Contact_Force", "Contact_Force_Density"], "point"
            )
            disp_name, disp_arrays = _collect_first_available_from_datasets(
                datasets, ["Displacement"], "point"
            )
            vel_name, vel_arrays = _collect_first_available_from_datasets(
                datasets, ["Velocity"], "point"
            )

            max_damage = damage_fraction = damaged_vol = damaged_vol_frac = math.nan
            if damage_arrays:
                (
                    max_damage,
                    damage_fraction,
                    damaged_vol,
                    damaged_vol_frac,
                ) = _damage_metrics(damage_arrays, vol_arrays, args.damage_threshold)

            max_force_density = _mag_max(force_arrays) if force_arrays else math.nan
            max_contact_force = (
                _mag_max(contact_force_arrays) if contact_force_arrays else math.nan
            )
            max_displacement = _mag_max(disp_arrays) if disp_arrays else math.nan
            max_velocity = _mag_max(vel_arrays) if vel_arrays else math.nan

            split_flag = 0
            if (
                not math.isnan(damage_fraction)
                and not math.isnan(max_damage)
                and damage_fraction >= args.split_damage_fraction
                and max_damage >= args.split_max_damage
            ):
                split_flag = 1

            writer.writerow(
                [
                    time_val,
                    args.damage_threshold,
                    max_damage,
                    damage_fraction,
                    damaged_vol,
                    damaged_vol_frac,
                    max_force_density,
                    max_contact_force,
                    max_displacement,
                    max_velocity,
                    split_flag,
                    damage_basis,
                    damage_name or "",
                    vol_name or "",
                    contact_name or "",
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
