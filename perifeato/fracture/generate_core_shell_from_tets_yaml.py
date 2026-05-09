#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _detect_block_ids(discretization: Path) -> list[int]:
    """
    Return sorted unique block ids found in the Peridigm text discretization.

    Expected format per line: x y z block_id volume
    """
    blocks: set[int] = set()
    with discretization.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise SystemExit(f"Invalid discretization line (expected 5 columns): {line[:200]}")
            try:
                blk = int(float(parts[3]))
            except Exception as e:
                raise SystemExit(f"Failed parsing block_id from line: {line[:200]} ({e})") from e
            blocks.add(blk)
    if not blocks:
        raise SystemExit("No block ids found in discretization.")
    return sorted(blocks)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a Peridigm YAML that runs on a text discretization exported from a tetra mesh."
    )
    ap.add_argument("--discretization", required=True, type=Path, help="Peridigm text discretization file (x y z block v)")
    ap.add_argument("--nodeset-fixed", required=True, type=Path, help="Nodeset file (1-based ids) for fixed end")
    ap.add_argument("--nodeset-impact", required=True, type=Path, help="Nodeset file (1-based ids) for impact end")
    ap.add_argument("--nodeset-cone", type=Path, default=None, help="Nodeset file (1-based ids) for cone impactor")
    ap.add_argument("--nodeset-all", type=Path, default=None, help="Nodeset file (1-based ids) for all shell points")
    ap.add_argument("--out-yaml", required=True, type=Path, help="Where to write the generated YAML")
    ap.add_argument("--out-prefix", default="out/core_shell_from_tets", help="Output filename prefix (no extension)")
    ap.add_argument("--impact-vz", type=float, default=-5.0, help="Initial impact velocity in -z (m/s)")
    ap.add_argument("--cone-impact-vz", type=float, default=None, help="Cone impact velocity in -z (m/s).")
    ap.add_argument("--final-time", type=float, default=2.0e-5, help="Final time (s)")
    ap.add_argument("--output-frequency", type=int, default=5, help="Exodus output frequency (steps)")
    ap.add_argument(
        "--pla-critical-stretch",
        type=float,
        default=0.01,
        help="Critical stretch for PLA damage model (dimensionless). Set <=0 to disable damage.",
    )
    ap.add_argument(
        "--fracture-critical-stretch",
        type=float,
        default=0.005,
        help="Critical stretch for fracture-strip block (dimensionless). Ignored if block_3 absent.",
    )
    ap.add_argument("--cone-block-id", type=int, default=99, help="Block id for cone impactor points.")
    ap.add_argument("--cone-horizon", type=float, default=0.004, help="Horizon for cone block (m).")
    ap.add_argument("--cone-density", type=float, default=7800.0, help="Cone density (kg/m^3).")
    ap.add_argument("--cone-bulk-modulus", type=float, default=160.0e9, help="Cone bulk modulus (Pa).")
    ap.add_argument("--cone-shear-modulus", type=float, default=80.0e9, help="Cone shear modulus (Pa).")
    ap.add_argument("--contact-search-radius", type=float, default=0.006, help="Contact search radius (m).")
    ap.add_argument("--contact-search-frequency", type=int, default=50, help="Contact search frequency (steps).")
    ap.add_argument("--contact-radius", type=float, default=0.003, help="Short range contact radius (m).")
    ap.add_argument("--contact-spring-constant", type=float, default=1.0e10, help="Short range contact spring (N/m).")
    ap.add_argument("--impact-on-all", action="store_true", help="Apply shell impact velocity to Node Set All if provided")
    ap.add_argument("--shell-impact", action="store_true", help="Apply shell impact velocity even when cone is enabled")
    ap.add_argument("--fixed-end-slide", action="store_true", help="Constrain fixed end in X/Y only (allow Z sliding)")
    ap.add_argument("--cone-fixed", action="store_true", help="Fix cone nodes in place (no initial velocity)")
    ap.add_argument(
        "--cone-constant-velocity",
        action="store_true",
        help="Use Prescribed Displacement for cone (constant velocity, pushes through shell). "
             "Without this flag, Initial Velocity is used (cone decelerates on contact)."
    )
    ap.add_argument(
        "--global-strain-energy",
        action="store_true",
        help="Include Global_Strain_Energy in history output (if supported by Peridigm build).",
    )
    ap.add_argument(
        "--state-based",
        action="store_true",
        help="Use state-based 'Elastic Correspondence' material model for accurate Poisson ratio (ν=0.36). "
             "Default is bond-based 'Elastic' which has fixed ν≈0.25 in 3D.",
    )
    ap.add_argument(
        "--pla-youngs-modulus",
        type=float,
        default=2.5e9,
        help="PLA Young's modulus (Pa). Default: 2.5e9 from literature.",
    )
    ap.add_argument(
        "--pla-poisson-ratio",
        type=float,
        default=0.36,
        help="PLA Poisson's ratio. Default: 0.36 from literature.",
    )
    ap.add_argument(
        "--pla-horizon",
        type=float,
        default=0.009,
        help="PLA horizon (m). Should be ~3× mesh element size. Default: 0.009 (9mm for 3mm mesh).",
    )
    args = ap.parse_args()

    disc = args.discretization
    if not disc.is_file():
        raise SystemExit(f"Missing discretization: {disc}")
    ns_fixed = args.nodeset_fixed
    if not ns_fixed.is_file():
        raise SystemExit(f"Missing nodeset-fixed: {ns_fixed}")
    ns_impact = args.nodeset_impact
    if not ns_impact.is_file():
        raise SystemExit(f"Missing nodeset-impact: {ns_impact}")
    ns_cone = args.nodeset_cone
    if ns_cone is not None and not ns_cone.is_file():
        raise SystemExit(f"Missing nodeset-cone: {ns_cone}")
    ns_all = args.nodeset_all
    if ns_all is not None and not ns_all.is_file():
        raise SystemExit(f"Missing nodeset-all: {ns_all}")

    out_yaml = args.out_yaml
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    block_ids = _detect_block_ids(disc)

    # Our export convention: gmsh region_id == block_id
    #  1 = steel
    #  2 = pla_protected
    #  3 = pla_fracture (only present when fracture wedges are enabled)
    blocks_cfg: dict[str, dict] = {}
    damage_cfg: dict[str, dict] = {}
    use_cone = ns_cone is not None

    use_pla_damage = args.pla_critical_stretch is not None and args.pla_critical_stretch > 0.0
    use_frac_damage = args.fracture_critical_stretch is not None and args.fracture_critical_stretch > 0.0
    if use_pla_damage:
        damage_cfg["PLA Damage"] = {
            "Damage Model": "Critical Stretch",
            "Critical Stretch": float(args.pla_critical_stretch),
        }
    if use_frac_damage:
        damage_cfg["Fracture Damage"] = {
            "Damage Model": "Critical Stretch",
            "Critical Stretch": float(args.fracture_critical_stretch),
        }

    pla_horizon = float(args.pla_horizon)
    for blk in block_ids:
        if blk == 1:
            # Steel uses smaller horizon (it's stiffer, doesn't need as much nonlocal averaging)
            blocks_cfg["Steel Block"] = {"Block Names": "block_1", "Material": "Steel", "Horizon": 0.004}
        elif blk == 2:
            entry = {"Block Names": "block_2", "Material": "PLA", "Horizon": pla_horizon}
            if use_pla_damage:
                entry["Damage Model"] = "PLA Damage"
            blocks_cfg["PLA Block"] = entry
        elif use_cone and blk == args.cone_block_id:
            blocks_cfg["Cone Block"] = {
                "Block Names": f"block_{args.cone_block_id}",
                "Material": "Cone Material",
                "Horizon": float(args.cone_horizon),
            }
        elif blk == 3:
            entry = {"Block Names": "block_3", "Material": "PLA", "Horizon": pla_horizon}
            if use_frac_damage:
                entry["Damage Model"] = "Fracture Damage"
            elif use_pla_damage:
                entry["Damage Model"] = "PLA Damage"
            blocks_cfg["PLA Fracture Block"] = entry
        else:
            # Default: treat as PLA unless user changes this script later.
            blocks_cfg[f"Block {blk}"] = {"Block Names": f"block_{blk}", "Material": "PLA", "Horizon": pla_horizon}

    # Note: For Peridigm "Text File" discretizations, node sets must be declared under Boundary Conditions
    # with keys beginning with "Node Set". The declared key string is then referenced by BC entries.
    cfg = {
        "Peridigm": {
            "Discretization": {"Type": "Text File", "Input Mesh File": str(disc)},
            "Materials": {
                # PLA properties from literature (Gao2022, Ramirez2024, Farah2016)
                # E = 2.5 GPa, nu = 0.36 -> K = 2.98 GPa, G = 0.92 GPa
                "PLA": (
                    # State-based: accurate Poisson ratio (Bobaru & Hu 2012)
                    # Requires Hourglass Coefficient for stabilization
                    {
                        "Material Model": "Elastic Correspondence",
                        "Density": 1240.0,
                        "Young's Modulus": float(args.pla_youngs_modulus),
                        "Poisson's Ratio": float(args.pla_poisson_ratio),
                        "Hourglass Coefficient": 0.02,  # Peridigm default for correspondence models
                    }
                    if args.state_based
                    else
                    # Bond-based: fixed nu ~ 0.25 in 3D, but faster
                    {
                        "Material Model": "Elastic",
                        "Density": 1240.0,
                        "Bulk Modulus": 2.98e9,
                        "Shear Modulus": 0.92e9,
                    }
                ),
                "Steel": {
                    "Material Model": "Elastic",
                    "Density": 7800.0,
                    "Bulk Modulus": 160.0e9,
                    "Shear Modulus": 80.0e9,
                },
            },
            **({"Damage Models": damage_cfg} if damage_cfg else {}),
            # Ensure blocks declared in YAML exactly match what's in the discretization file.
            "Blocks": blocks_cfg,
            "Boundary Conditions": {
                "Node Set Fixed End": str(ns_fixed),
                "Node Set Impact End": str(ns_impact),
                "Fix Fixed End X": {
                    "Type": "Prescribed Displacement",
                    "Node Set": "Node Set Fixed End",
                    "Coordinate": "x",
                    "Value": "0.0",
                },
                "Fix Fixed End Y": {
                    "Type": "Prescribed Displacement",
                    "Node Set": "Node Set Fixed End",
                    "Coordinate": "y",
                    "Value": "0.0",
                },
            },
            "Solver": {
                "Initial Time": 0.0,
                "Final Time": float(args.final_time),
                "Verlet": {"Safety Factor": 0.7},
            },
            "Output_1": {
                "Output File Type": "ExodusII",
                "Output Filename": str(args.out_prefix),
                "Output Frequency": int(args.output_frequency),
                "Output Variables": {
                    "Displacement": True,
                    "Velocity": True,
                    "Proc_Num": True,
                    "Element_Id": True,
                    "Damage": True,
                    "Number_Of_Neighbors": True,
                },
            },
            "Output_2": {
                "Output File Type": "ExodusII",
                "Output Filename": str(args.out_prefix),
                "Output Frequency": int(args.output_frequency),
                "Output Variables": {
                    "Global_Kinetic_Energy": True,
                    "Global_Linear_Momentum": True,
                    "Global_Angular_Momentum": True,
                },
            },
        }
    }

    if args.global_strain_energy:
        cfg["Peridigm"]["Output_2"]["Output Variables"]["Global_Strain_Energy"] = True

    if ns_all is not None:
        cfg["Peridigm"]["Boundary Conditions"]["Node Set All"] = str(ns_all)

    if not args.fixed_end_slide:
        cfg["Peridigm"]["Boundary Conditions"]["Fix Fixed End Z"] = {
            "Type": "Prescribed Displacement",
            "Node Set": "Node Set Fixed End",
            "Coordinate": "z",
            "Value": "0.0",
        }

    if use_cone:
        shell_blocks = [f"block_{bid}" for bid in block_ids if bid != args.cone_block_id]
        cfg["Peridigm"]["Materials"]["Cone Material"] = {
            "Material Model": "Elastic",
            "Density": float(args.cone_density),
            "Bulk Modulus": float(args.cone_bulk_modulus),
            "Shear Modulus": float(args.cone_shear_modulus),
        }
        cfg["Peridigm"]["Boundary Conditions"]["Node Set Cone"] = str(ns_cone)
        if args.cone_fixed:
            for coord in ("x", "y", "z"):
                cfg["Peridigm"]["Boundary Conditions"][f"Fix Cone {coord.upper()}"] = {
                    "Type": "Prescribed Displacement",
                    "Node Set": "Node Set Cone",
                    "Coordinate": coord,
                    "Value": "0.0",
                }
        else:
            cone_vz = args.cone_impact_vz if args.cone_impact_vz is not None else args.impact_vz
            # Use Initial Velocity so cone has momentum but can slow down on contact
            # This simulates fishing line recoil: line breaks, cone has initial velocity,
            # but contact forces with core/shell will naturally decelerate it
            cfg["Peridigm"]["Boundary Conditions"]["Cone Initial Velocity Z"] = {
                "Type": "Initial Velocity",
                "Node Set": "Node Set Cone",
                "Coordinate": "z",
                "Value": f"{cone_vz}",
            }
        cfg["Peridigm"]["Contact"] = {
            "Search Radius": float(args.contact_search_radius),
            "Search Frequency": int(args.contact_search_frequency),
            "Models": {
                "Cone Shell Contact": {
                    "Contact Model": "Short Range Force",
                    "Contact Radius": float(args.contact_radius),
                    "Spring Constant": float(args.contact_spring_constant),
                }
            },
            "Interactions": {
                f"Interaction Cone with {blk}": {
                    "First Block": f"block_{args.cone_block_id}",
                    "Second Block": blk,
                    "Contact Model": "Cone Shell Contact",
                }
                for blk in shell_blocks
            },
        }
        cfg["Peridigm"]["Output_1"]["Output Variables"]["Contact_Force_Density"] = True
    apply_shell_impact = args.shell_impact or not use_cone
    if apply_shell_impact:
        impact_nodeset_name = "Node Set Impact End"
        if args.impact_on_all and ns_all is not None:
            impact_nodeset_name = "Node Set All"
        cfg["Peridigm"]["Boundary Conditions"]["Impact Initial Velocity Z"] = {
            "Type": "Initial Velocity",
            "Node Set": impact_nodeset_name,
            "Coordinate": "z",
            "Value": f"{args.impact_vz}",
        }

    out_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Wrote: {out_yaml}")
    print(f"Detected blocks: {block_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
