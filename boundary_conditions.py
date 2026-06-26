#!/usr/bin/env python3
"""Pre-processor layer 3.2: boundary-condition mapping (plan section 3.2).

Reads the NEUTRAL geometric face labels from out/beam.msh and maps them, via
PARAMS.bc, into the FIXED and LOAD node sets the solver input needs. This is the
ONLY place where geometric labels acquire boundary-condition meaning: moving
where the beam is fixed or loaded touches this module only, never mesh.py.

Output is an in-memory BC definition (plain data) consumed by build_inp.py; no
file is written (the plan's out/ layout lists no BC artifact). Node ids are read
from the same .msh build_inp.py will read for *NODE, so the numbering matches.

Naming contract (do not change): node sets are FIXED and LOAD.
Unit system: SI-mm (force in N).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gmsh
import numpy as np

from params import PARAMS

# Project-anchored paths (work regardless of the caller's CWD).
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
MSH_PATH = OUT_DIR / "beam.msh"


def _read_face_node_sets(msh_path, verbose: bool = False) -> dict:
    """Return {face_label: sorted node-id list} for every surface physical group.

    Self-contained gmsh session: it only initializes/finalizes if no gmsh session
    is already open, so it is safe to call standalone or from another module.
    """
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    try:
        gmsh.open(str(msh_path))
        face_nodes: dict = {}
        for dim, tag in gmsh.model.getPhysicalGroups(2):  # dim 2 = surfaces only
            name = gmsh.model.getPhysicalName(dim, tag)
            ntags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
            face_nodes[name] = sorted(int(t) for t in ntags)
    finally:
        if own:
            gmsh.finalize()
    return face_nodes


def _unit_vector(vec) -> list:
    """Normalize a direction vector. A zero vector returns zeros (the unit-vector
    acceptance gate then flags it as invalid rather than dividing by zero)."""
    v = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return [0.0, 0.0, 0.0]
    return (v / norm).tolist()


def build_bc(params: dict, msh_path=MSH_PATH, verbose: bool = False) -> dict:
    """Map PARAMS.bc + mesh face labels into FIXED / LOAD node sets.

    Returns a plain-data BC definition consumed by build_inp.py. Missing face
    references are recorded (not raised) so the acceptance report can flag them.
    """
    if not Path(msh_path).exists():
        raise FileNotFoundError(f"{msh_path} not found; run mesh.py first")

    bc = params["bc"]
    face_nodes = _read_face_node_sets(msh_path, verbose=verbose)
    available = sorted(face_nodes.keys())

    fixed_faces = list(bc["fixed"])
    referenced = set(fixed_faces) | {entry["face"] for entry in bc["loads"]}
    missing = sorted(f for f in referenced if f not in face_nodes)

    # FIXED node set = union over the present fixed faces (DOF 1-3 clamped later).
    fixed_ids = sorted(set().union(
        *[set(face_nodes[f]) for f in fixed_faces if f in face_nodes]
    )) if fixed_faces else []

    # LOAD: per-load metadata + the union LOAD node set.
    loads = []
    load_union: set = set()
    for entry in bc["loads"]:
        face = entry["face"]
        nids = face_nodes.get(face, [])
        loads.append({
            "face": face,
            "dir_unit": _unit_vector(entry["dir"]),
            "magnitude": float(entry["magnitude"]),  # N, total over this load's nodes
            "node_ids": list(nids),
            "n_nodes": len(nids),
        })
        load_union.update(nids)
    load_ids = sorted(load_union)

    return {
        "msh_path": str(msh_path),
        "available_faces": available,
        "missing_faces": missing,
        "fixed": {"name": "FIXED", "faces": fixed_faces,
                  "node_ids": fixed_ids, "n_nodes": len(fixed_ids)},
        "load": {"name": "LOAD", "node_ids": load_ids,
                 "n_nodes": len(load_ids), "loads": loads},
    }


def report_and_check(bc_def: dict) -> bool:
    """Print the acceptance report (plan section 3.2) and return PASS (True)/FAIL."""
    fixed, load = bc_def["fixed"], bc_def["load"]
    print("=" * 64)
    print("boundary_conditions.py acceptance report")
    print("=" * 64)
    print(f"mesh           : {bc_def['msh_path']}")
    print(f"available faces: {', '.join(bc_def['available_faces'])}")
    print(f"FIXED          : faces={fixed['faces']} -> {fixed['n_nodes']} nodes (DOF 1-3 clamped)")
    print(f"LOAD           : {load['n_nodes']} nodes (union of load faces)")
    for i, ld in enumerate(load["loads"]):
        unit_norm = float(np.linalg.norm(ld["dir_unit"]))
        print(f"  load[{i}]      : face={ld['face']}  dir_unit={ld['dir_unit']} "
              f"(|dir|={unit_norm:.4f})  |F|={ld['magnitude']} N over {ld['n_nodes']} nodes")
    if bc_def["missing_faces"]:
        print(f"  !! missing referenced faces: {bc_def['missing_faces']}")

    # Deterministic acceptance gates (plan section 3.2).
    dirs_unit = (len(load["loads"]) > 0) and all(
        abs(float(np.linalg.norm(ld["dir_unit"])) - 1.0) < 1e-9 for ld in load["loads"]
    )
    checks = [
        ("referenced face labels all exist in mesh", len(bc_def["missing_faces"]) == 0),
        ("FIXED node set non-empty", fixed["n_nodes"] > 0),
        ("LOAD node set non-empty", load["n_nodes"] > 0),
        ("every load direction is a unit vector", dirs_unit),
    ]
    passed = all(ok for _, ok in checks)
    print("-" * 64)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 64)
    print(f"ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    print("=" * 64)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Map BC face labels -> FIXED/LOAD (plan layer 3.2).")
    parser.add_argument("-v", "--verbose", action="store_true", help="show Gmsh log output")
    args = parser.parse_args()
    bc_def = build_bc(PARAMS, verbose=args.verbose)
    return 0 if report_and_check(bc_def) else 1


if __name__ == "__main__":
    sys.exit(main())
