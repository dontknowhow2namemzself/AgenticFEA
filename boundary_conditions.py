#!/usr/bin/env python3
"""Pre-processor layer 3.2: boundary-condition mapping (plan section 3.2,
plate plan M3).

Reads the NEUTRAL geometric boundary labels from out/beam.msh (faces of the 3D
beam, edge curves of the 2D plate) and maps them, via PARAMS.bc, into the
constraint table and LOAD node sets the solver input needs. This is the ONLY
place where geometric labels acquire boundary-condition meaning: moving where
the model is fixed or loaded touches this module only, never mesh.py.

M3 upgrade: `bc.fixed` entries may be
  * a plain string label            -> all translational DOFs on that entity
  * {"face", "dofs"}                -> the listed DOFs on every node of it
  * {"face", "dofs", "at": "mid"}   -> the listed DOFs on the SINGLE node
                                       nearest the entity midpoint (rigid-body
                                       anchor; ties broken by lowest node id)
The output carries both a per-(node, dof) constraint table (what *BOUNDARY
needs for the plate) and the FIXED node-id union (reaction summing, and the
cantilever's clamp path, unchanged).

New acceptance gate: a RIGID-BODY MODE CHECK. Every in-plane (2D: Tx, Ty, Rz)
or spatial (3D: Tx..Rz) rigid mode must be eliminated by the constraints --
checked exactly via the rank of the constraint rows evaluated on the rigid
modes, not by heuristics. ccx tolerates an under-constrained deck (exits 0
with garbage), so this must be caught here, before the solver.

Output is an in-memory BC definition (plain data) consumed by build_inp.py.
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
from schema import fixed_entry_at, fixed_entry_dofs, fixed_entry_face

# Project-anchored paths (work regardless of the caller's CWD).
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
MSH_PATH = OUT_DIR / "beam.msh"

RIGID_MODES_2D = ("Tx", "Ty", "Rz")
RIGID_MODES_3D = ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz")


def _bc_entity_dim(geom_type: str) -> int:
    """Dimension of the boundary entities BC labels refer to (2D plate -> curves)."""
    return 1 if geom_type == "plate_with_hole" else 2


def _read_boundary_node_sets(msh_path, dim: int, verbose: bool = False) -> dict:
    """{label: {"node_ids": sorted list, "xyz": {nid: coords}}} for physical
    groups of the given dimension.

    Self-contained gmsh session: it only initializes/finalizes if no gmsh
    session is already open, so it is safe standalone or from another module.
    """
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    try:
        gmsh.open(str(msh_path))
        sets: dict = {}
        for d, tag in gmsh.model.getPhysicalGroups(dim):
            name = gmsh.model.getPhysicalName(d, tag)
            ntags, coords = gmsh.model.mesh.getNodesForPhysicalGroup(d, tag)
            xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
            ids = [int(t) for t in ntags]
            sets[name] = {
                "node_ids": sorted(ids),
                "xyz": {nid: xyz[k] for k, nid in enumerate(ids)},
            }
    finally:
        if own:
            gmsh.finalize()
    return sets


def _unit_vector(vec) -> list:
    """Normalize a direction vector. A zero vector returns zeros (the unit-vector
    acceptance gate then flags it as invalid rather than dividing by zero)."""
    v = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return [0.0, 0.0, 0.0]
    return (v / norm).tolist()


def _resolve_fixed_entry(entry, sets: dict) -> dict:
    """Resolve one bc.fixed entry against the mesh label sets (plain data)."""
    face, dofs, at = fixed_entry_face(entry), fixed_entry_dofs(entry), fixed_entry_at(entry)
    info = sets.get(face)
    if info is None:
        return {"face": face, "dofs": sorted(dofs), "at": at,
                "node_ids": [], "n_nodes": 0, "mid": None}
    if at == "mid":
        # Deterministic anchor: node nearest the entity midpoint (center of the
        # label's node bounding box); ties broken by lowest node id.
        nids = info["node_ids"]
        xyz = info["xyz"]
        pts = np.array([xyz[n] for n in nids])
        mid = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        chosen = min(nids, key=lambda n: (float(np.linalg.norm(xyz[n] - mid)), n))
        sel = [chosen]
        mid_info = {"midpoint": mid.tolist(),
                    "chosen_xyz": xyz[chosen].tolist(),
                    "dist": float(np.linalg.norm(xyz[chosen] - mid))}
    else:
        sel = list(info["node_ids"])
        mid_info = None
    return {"face": face, "dofs": sorted(dofs), "at": at,
            "node_ids": sel, "n_nodes": len(sel), "mid": mid_info}


def _rigid_body_check(constraints: list, node_xyz: dict, two_d: bool) -> dict:
    """Exact rigid-body elimination check via the rank of the constraint rows.

    Each constraint (node, dof) contributes one row: the displacement that each
    rigid mode produces at that node's dof. All modes are eliminated iff the
    row matrix has full column rank (rank == number of rigid modes). Columns
    are max-normalized so translation (O(1)) and rotation (O(coordinate)) modes
    are compared on equal footing.
    """
    modes = RIGID_MODES_2D if two_d else RIGID_MODES_3D
    m = len(modes)
    rows = []
    for node, dof in constraints:
        x, y, z = (float(c) for c in node_xyz[node])
        if two_d:
            if dof == 3:      # out-of-plane dof carries no in-plane rigid mode
                continue
            row = {1: [1.0, 0.0, -y], 2: [0.0, 1.0, x]}[dof]
        else:
            row = {1: [1.0, 0.0, 0.0, 0.0, z, -y],
                   2: [0.0, 1.0, 0.0, -z, 0.0, x],
                   3: [0.0, 0.0, 1.0, y, -x, 0.0]}[dof]
        rows.append(row)
    if not rows:
        return {"mode_names": modes, "n_modes": m, "rank": 0, "ok": False,
                "free_modes": list(modes)}
    G = np.asarray(rows, dtype=float)
    scale = np.max(np.abs(G), axis=0)
    scale[scale == 0.0] = 1.0
    Gn = G / scale
    rank = int(np.linalg.matrix_rank(Gn))
    free = []
    if rank < m:
        # Null-space directions = rigid motions the constraints do not see.
        _, s, vh = np.linalg.svd(Gn)
        sv = np.zeros(m)
        sv[:len(s)] = s
        tol = max(Gn.shape) * np.finfo(float).eps * (float(s.max()) if len(s) else 1.0)
        for k in range(m):
            if sv[k] <= tol:
                v = vh[k]
                parts = [f"{v[i]:+.2f}*{modes[i]}" for i in range(m) if abs(v[i]) > 0.3]
                free.append(" ".join(parts) if parts else "(mixed)")
    return {"mode_names": modes, "n_modes": m, "rank": rank,
            "ok": rank == m, "free_modes": free}


def build_bc(params: dict, msh_path=MSH_PATH, verbose: bool = False) -> dict:
    """Map PARAMS.bc + mesh boundary labels into the constraint table and the
    FIXED / LOAD node sets.

    Returns a plain-data BC definition consumed by build_inp.py. Missing label
    references are recorded (not raised) so the acceptance report can flag them.
    """
    if not Path(msh_path).exists():
        raise FileNotFoundError(f"{msh_path} not found; run mesh.py first")

    bc = params["bc"]
    geom_type = params["geometry"]["type"]
    two_d = geom_type == "plate_with_hole"
    sets = _read_boundary_node_sets(msh_path, _bc_entity_dim(geom_type), verbose=verbose)
    available = sorted(sets.keys())

    fixed_raw = list(bc["fixed"])
    referenced = ([fixed_entry_face(e) for e in fixed_raw]
                  + [entry["face"] for entry in bc["loads"]])
    missing = sorted({f for f in referenced if f not in sets})

    # Constraint table: per-entry resolution -> unique (node, dof) pairs.
    entries = [_resolve_fixed_entry(e, sets) for e in fixed_raw]
    constraints_set = set()
    for e in entries:
        for nid in e["node_ids"]:
            for dof in e["dofs"]:
                constraints_set.add((nid, dof))
    constraints = sorted(constraints_set)
    fixed_ids = sorted({nid for nid, _ in constraints})

    # Coordinates of every constrained node (for the rigid-body check).
    node_xyz: dict = {}
    for info in sets.values():
        node_xyz.update(info["xyz"])
    rigid = _rigid_body_check(constraints, node_xyz, two_d)

    # LOAD: per-load metadata + the union LOAD node set.
    loads = []
    load_union: set = set()
    for entry in bc["loads"]:
        face = entry["face"]
        nids = sets.get(face, {}).get("node_ids", [])
        loads.append({
            "face": face,
            "dir_unit": _unit_vector(entry["dir"]),
            "magnitude": float(entry["magnitude"]),  # N, total over this load's entity
            "node_ids": list(nids),
            "n_nodes": len(nids),
        })
        load_union.update(nids)
    load_ids = sorted(load_union)

    return {
        "msh_path": str(msh_path),
        "geometry_type": geom_type,
        "available_faces": available,
        "missing_faces": missing,
        "fixed": {"name": "FIXED",
                  "faces": [e["face"] for e in entries],
                  "entries": entries,
                  "constraints": constraints,
                  "n_constraints": len(constraints),
                  "node_ids": fixed_ids, "n_nodes": len(fixed_ids)},
        "rigid": rigid,
        "load": {"name": "LOAD", "node_ids": load_ids,
                 "n_nodes": len(load_ids), "loads": loads},
    }


def report_and_check(bc_def: dict) -> bool:
    """Print the acceptance report (plan 3.2 / plate plan M3); PASS (True)/FAIL."""
    fixed, load, rigid = bc_def["fixed"], bc_def["load"], bc_def["rigid"]
    print("=" * 64)
    print("boundary_conditions.py acceptance report")
    print("=" * 64)
    print(f"mesh           : {bc_def['msh_path']}  (geometry: {bc_def['geometry_type']})")
    print(f"available      : {', '.join(bc_def['available_faces'])}")
    print(f"FIXED          : {fixed['n_constraints']} (node, dof) constraints "
          f"over {fixed['n_nodes']} nodes")
    for i, e in enumerate(fixed["entries"]):
        line = (f"  fixed[{i}]     : face={e['face']}  at={e['at']}  "
                f"dofs={e['dofs']}  -> {e['n_nodes']} node(s)")
        if e["mid"] is not None and e["node_ids"]:
            cx = ", ".join(f"{v:.6g}" for v in e["mid"]["chosen_xyz"])
            line += (f"  [node {e['node_ids'][0]} at ({cx}), "
                     f"midpoint dist {e['mid']['dist']:.3g} mm]")
        print(line)
    mode_str = "/".join(rigid["mode_names"])
    if rigid["ok"]:
        print(f"rigid-body     : rank {rigid['rank']}/{rigid['n_modes']} -> "
              f"all modes ({mode_str}) eliminated")
    else:
        print(f"rigid-body     : rank {rigid['rank']}/{rigid['n_modes']} -> "
              f"FREE MODES REMAIN: {rigid['free_modes']}")
    print(f"LOAD           : {load['n_nodes']} nodes (union of load entities)")
    for i, ld in enumerate(load["loads"]):
        unit_norm = float(np.linalg.norm(ld["dir_unit"]))
        print(f"  load[{i}]      : face={ld['face']}  dir_unit={ld['dir_unit']} "
              f"(|dir|={unit_norm:.4f})  |F|={ld['magnitude']} N over {ld['n_nodes']} nodes")
    if bc_def["missing_faces"]:
        print(f"  !! missing referenced labels: {bc_def['missing_faces']}")

    # Deterministic acceptance gates (plan 3.2 + plate plan M3).
    dirs_unit = (len(load["loads"]) > 0) and all(
        abs(float(np.linalg.norm(ld["dir_unit"])) - 1.0) < 1e-9 for ld in load["loads"]
    )
    mids_ok = all(e["at"] != "mid" or e["n_nodes"] == 1 for e in fixed["entries"])
    checks = [
        ("referenced boundary labels all exist in mesh", len(bc_def["missing_faces"]) == 0),
        ("constraint table non-empty", fixed["n_constraints"] > 0),
        ("FIXED node set non-empty", fixed["n_nodes"] > 0),
        ("every 'mid' entry resolved to exactly 1 node", mids_ok),
        (f"rigid-body modes eliminated (rank {rigid['n_modes']}/{rigid['n_modes']})",
         rigid["ok"]),
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
    parser = argparse.ArgumentParser(
        description="Map BC labels -> constraints + FIXED/LOAD (plan 3.2 / plate M3).")
    parser.add_argument("-v", "--verbose", action="store_true", help="show Gmsh log output")
    args = parser.parse_args()
    bc_def = build_bc(PARAMS, verbose=args.verbose)
    return 0 if report_and_check(bc_def) else 1


if __name__ == "__main__":
    sys.exit(main())
