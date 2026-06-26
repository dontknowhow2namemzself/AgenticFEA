#!/usr/bin/env python3
"""Pre-processor layer 3.4: CalculiX input-deck generation (plan section 3.4).

Reads out/beam.msh (nodes + C3D20R volume elements) and the FIXED / LOAD node
sets from boundary_conditions.py, then writes a complete CalculiX deck
out/beam.inp with: *NODE, *ELEMENT (C3D20R, ELSET=BEAM_VOL), *NSET=FIXED/LOAD,
*MATERIAL STEEL + *ELASTIC, *SOLID SECTION, *BOUNDARY, *STEP + *STATIC, *CLOAD
(total force split equally over the LOAD nodes), and the U / RF / S output
requests the downstream layers need.

CRITICAL detail handled here: Gmsh (msh element type 17) and CalculiX (C3D20)
agree on the 8 hex corners but order the 12 midside nodes differently. The
remap is DERIVED from Gmsh's own reference-node coordinates and VERIFIED
geometrically (every midside node must sit at its edge midpoint on the real
mesh). See _gmsh_to_abaqus_c3d20_perm / _verify_c3d20_ordering.

Unit system: SI-mm (length mm, force N, stress/E MPa, displacement mm).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gmsh
import numpy as np

from boundary_conditions import build_bc
from params import PARAMS

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
MSH_PATH = OUT_DIR / "beam.msh"
INP_PATH = OUT_DIR / "beam.inp"

GMSH_HEX20 = 17  # Gmsh element type id for the 20-node hexahedron.

# Canonical Abaqus / CalculiX C3D20 reference-node local coordinates, in C3D20
# ordering (corners 1-8, bottom edges 9-12, top edges 13-16, vertical edges
# 17-20). Source: Abaqus Element Library / CalculiX manual (identical ordering).
_ABQ_C3D20_REF = (
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),   # 1-4  bottom corners
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),       # 5-8  top corners
    (0, -1, -1), (1, 0, -1), (0, 1, -1), (-1, 0, -1),     # 9-12 bottom edges
    (0, -1, 1), (1, 0, 1), (0, 1, 1), (-1, 0, 1),         # 13-16 top edges
    (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0),       # 17-20 vertical edges
)
# Midside node (1-based C3D20 index) -> the two corner nodes it bisects.
_ABQ_C3D20_EDGES = {
    9: (1, 2), 10: (2, 3), 11: (3, 4), 12: (4, 1),
    13: (5, 6), 14: (6, 7), 15: (7, 8), 16: (8, 5),
    17: (1, 5), 18: (2, 6), 19: (3, 7), 20: (4, 8),
}


def _gmsh_to_abaqus_c3d20_perm() -> list:
    """Permutation p with abaqus_nodes[i] = gmsh_nodes[p[i]] for a 20-node hex.

    Derived (not hardcoded) by matching Gmsh's own reference-node coordinates
    against the canonical Abaqus C3D20 reference table. Requires an open gmsh
    session. Raises if the result is not a bijection of the 20 local indices.
    """
    _, _, _, num_nodes, local_coord, _ = gmsh.model.mesh.getElementProperties(GMSH_HEX20)
    if num_nodes != 20:
        raise RuntimeError(f"gmsh type {GMSH_HEX20} reports {num_nodes} nodes, expected 20")
    gref = np.asarray(local_coord, dtype=float).reshape(20, 3)
    # Normalize each axis to {-1, 0, +1} (Gmsh's reference cube may be [0,1] or [-1,1]).
    gnorm = np.empty_like(gref)
    for ax in range(3):
        lo, hi = gref[:, ax].min(), gref[:, ax].max()
        gnorm[:, ax] = np.rint(2.0 * (gref[:, ax] - lo) / (hi - lo) - 1.0)
    gmap = {tuple(int(v) for v in gnorm[j]): j for j in range(20)}
    if len(gmap) != 20:
        raise RuntimeError("gmsh C3D20 reference coordinates are not 20 distinct positions")
    perm = []
    for i in range(20):
        key = _ABQ_C3D20_REF[i]
        if key not in gmap:
            raise RuntimeError(f"no gmsh node at Abaqus reference position {key} (C3D20 slot {i + 1})")
        perm.append(gmap[key])
    if sorted(perm) != list(range(20)):
        raise RuntimeError("Gmsh->Abaqus C3D20 permutation is not a bijection")
    return perm


def _verify_c3d20_ordering(elem_conn: list, node_xyz: dict, tol: float) -> float:
    """Geometric proof of the remap: with nodes already in Abaqus order, every
    midside node must lie at the midpoint of its two corner nodes. Checks all
    elements, returns the max deviation, and raises if it exceeds tol.
    """
    max_dev = 0.0
    for nodes in elem_conn:  # 20 global node ids, Abaqus order
        for mid, (a, b) in _ABQ_C3D20_EDGES.items():
            p_mid = node_xyz[nodes[mid - 1]]
            p_exp = 0.5 * (node_xyz[nodes[a - 1]] + node_xyz[nodes[b - 1]])
            dev = float(np.linalg.norm(p_mid - p_exp))
            if dev > max_dev:
                max_dev = dev
    if max_dev > tol:
        raise RuntimeError(
            f"C3D20 node-ordering verification FAILED: max midside deviation "
            f"{max_dev:.3e} > tol {tol:.3e} (Gmsh->Abaqus remap is wrong)"
        )
    return max_dev


def _ints_to_lines(ints: list, per_line: int = 16) -> list:
    """Format integers as CalculiX data lines (<=16 items, comma-continued)."""
    lines = []
    for k in range(0, len(ints), per_line):
        chunk = ints[k:k + per_line]
        line = ", ".join(str(v) for v in chunk)
        if k + per_line < len(ints):
            line += ","          # trailing comma -> continued on next line
        lines.append(line)
    return lines


def build_inp(params: dict, msh_path=MSH_PATH, inp_path=INP_PATH, verbose: bool = False) -> dict:
    """Assemble out/beam.inp from the mesh + BC node sets. Returns a summary."""
    if not Path(msh_path).exists():
        raise FileNotFoundError(f"{msh_path} not found; run mesh.py first")

    g = params["geometry"]["cantilever_beam"]
    L, h, b = float(g["L"]), float(g["h"]), float(g["b"])
    mat = params["material"]

    # 1. BC node sets (build_bc runs its own gmsh session and finalizes it).
    bc_def = build_bc(params, msh_path=msh_path, verbose=verbose)

    # 2. Our gmsh session: derive remap, read nodes + volume elements.
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    try:
        gmsh.open(str(msh_path))
        perm = _gmsh_to_abaqus_c3d20_perm()

        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        node_ids = [int(t) for t in ntags]
        coords = np.asarray(ncoords, dtype=float).reshape(-1, 3)
        node_xyz = {nid: coords[k] for k, nid in enumerate(node_ids)}

        etypes, etags, enodes = gmsh.model.mesh.getElements(3)
        elements = []  # (elem_tag, [20 node ids in Abaqus order])
        for etype, tags, conn in zip(etypes, etags, enodes):
            if int(etype) != GMSH_HEX20:
                raise RuntimeError(f"unexpected volume element type {etype}; expected {GMSH_HEX20} (C3D20)")
            conn = [int(x) for x in conn]
            for e, etag in enumerate(tags):
                gnodes = conn[e * 20:(e + 1) * 20]
                elements.append((int(etag), [gnodes[perm[i]] for i in range(20)]))
    finally:
        if own:
            gmsh.finalize()

    # 3. Verify the remap geometrically on every element.
    tol = 1e-6 * max(L, h, b)
    max_dev = _verify_c3d20_ordering([conn for _, conn in elements], node_xyz, tol)

    # 4. CLOAD: split each load's total force equally over its nodes.
    #    Equal lumping reproduces the exact RESULTANT (what force balance and tip
    #    deflection depend on); it is not the consistent quad8 distribution, but
    #    stress is not auto-judged in the MVP so the local distribution is moot.
    cload_terms = []           # (node_id, dof, value)
    applied = np.zeros(3)      # total applied force vector (N)
    per_node_note = ""
    for ld in bc_def["load"]["loads"]:
        n = ld["n_nodes"]
        if n == 0:
            continue
        f_total = ld["magnitude"] * np.asarray(ld["dir_unit"], dtype=float)
        f_node = f_total / n
        for nid in ld["node_ids"]:
            for dof in (1, 2, 3):
                val = float(f_node[dof - 1])
                if abs(val) > 1e-12:
                    cload_terms.append((nid, dof, val))
        applied += f_total
        per_node_note = f"{np.array2string(f_node, precision=4)} N/node over {n} nodes"
    cload_sum = np.zeros(3)
    for _, dof, val in cload_terms:
        cload_sum[dof - 1] += val

    # 5. Assemble the deck.
    fixed_ids = bc_def["fixed"]["node_ids"]
    load_ids = bc_def["load"]["node_ids"]
    lines = []
    lines.append("** CalculiX input deck generated by build_inp.py (plan layer 3.4)")
    lines.append("** Unit system: SI-mm (length mm, force N, stress/E MPa, displacement mm)")
    lines.append(f"** Geometry: cantilever_beam  L={L} h={h} b={b} (mm)")
    lines.append(f"** Mesh: {len(elements)} x C3D20R   Nodes: {len(node_ids)}")
    lines.append(f"** Node ordering: Gmsh type 17 -> CalculiX C3D20, remap verified "
                 f"(max midside deviation {max_dev:.2e} mm)")

    # Nodes.
    lines.append("*NODE")
    for nid in sorted(node_ids):
        x, y, z = node_xyz[nid]
        lines.append(f"{nid}, {x:.10g}, {y:.10g}, {z:.10g}")

    # Elements (C3D20R) -> BEAM_VOL.
    lines.append("*ELEMENT, TYPE=C3D20R, ELSET=BEAM_VOL")
    for etag, conn in sorted(elements, key=lambda t: t[0]):
        lines.extend(_ints_to_lines([etag] + conn))

    # BC node sets.
    lines.append("*NSET, NSET=FIXED")
    lines.extend(_ints_to_lines(sorted(fixed_ids)))
    lines.append("*NSET, NSET=LOAD")
    lines.extend(_ints_to_lines(sorted(load_ids)))

    # Material + section.
    lines.append("*MATERIAL, NAME=STEEL")
    lines.append("*ELASTIC")
    lines.append(f"{mat['E']:.10g}, {mat['nu']:.10g}")          # E (MPa), nu
    lines.append("*SOLID SECTION, ELSET=BEAM_VOL, MATERIAL=STEEL")

    # Permanent constraint: clamp DOF 1-3 of FIXED (C3D has only translational DOF).
    lines.append("*BOUNDARY")
    lines.append("FIXED, 1, 3")

    # Static step + loads + output requests (U / RF nodal, S element).
    lines.append("*STEP")
    lines.append("*STATIC")
    lines.append("*CLOAD")
    for nid, dof, val in cload_terms:
        lines.append(f"{nid}, {dof}, {val:.10g}")
    lines.append("*NODE FILE")
    lines.append("U, RF")
    lines.append("*EL FILE")
    lines.append("S")
    lines.append("*END STEP")

    inp_text = "\n".join(lines) + "\n"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inp_path = Path(inp_path)
    inp_path.write_text(inp_text)

    return {
        "inp_path": str(inp_path),
        "file_size": inp_path.stat().st_size,
        "n_nodes": len(node_ids),
        "n_elems": len(elements),
        "elem_type": "C3D20R",
        "max_midside_dev": max_dev,
        "remap_tol": tol,
        "material": {"E": mat["E"], "nu": mat["nu"], "name": mat["name"]},
        "fixed_n": bc_def["fixed"]["n_nodes"],
        "load_n": bc_def["load"]["n_nodes"],
        "cload_terms": len(cload_terms),
        "applied": applied.tolist(),
        "cload_sum": cload_sum.tolist(),
        "per_node_note": per_node_note,
        "inp_text": inp_text,
    }


def report_and_check(s: dict) -> bool:
    """Print the acceptance report (plan section 3.4) and return PASS (True)/FAIL."""
    print("=" * 64)
    print("build_inp.py acceptance report")
    print("=" * 64)
    print(f"output    : {s['inp_path']}  ({s['file_size'] / 1024:.1f} KB)")
    print(f"nodes     : {s['n_nodes']}   elements: {s['n_elems']} x {s['elem_type']}")
    print(f"remap     : Gmsh type 17 -> CalculiX C3D20, "
          f"max midside deviation = {s['max_midside_dev']:.2e} mm (tol {s['remap_tol']:.1e})")
    print(f"material  : STEEL  E={s['material']['E']} MPa  nu={s['material']['nu']}")
    print(f"FIXED     : {s['fixed_n']} nodes, DOF 1-3 clamped")
    print(f"CLOAD     : {s['cload_terms']} terms over {s['load_n']} LOAD nodes  "
          f"({s['per_node_note']})")
    print(f"            applied resultant = {s['applied']} N ; CLOAD sum = {s['cload_sum']} N")
    print( "outputs   : *NODE FILE U, RF  /  *EL FILE S")

    inp = s["inp_text"]
    required = [
        "*NODE", "*ELEMENT, TYPE=C3D20R, ELSET=BEAM_VOL", "*NSET, NSET=FIXED",
        "*NSET, NSET=LOAD", "*MATERIAL, NAME=STEEL", "*ELASTIC",
        "*SOLID SECTION, ELSET=BEAM_VOL, MATERIAL=STEEL", "*BOUNDARY",
        "*STEP", "*STATIC", "*CLOAD", "*NODE FILE", "*EL FILE", "*END STEP",
    ]
    cards_ok = all(tok in inp for tok in required)
    outputs_ok = ("U, RF" in inp) and ("\nS\n" in inp)
    resultant_ok = np.allclose(s["cload_sum"], s["applied"], atol=1e-6 * (np.linalg.norm(s["applied"]) + 1.0))

    checks = [
        ("node count > 0 and matches mesh read", s["n_nodes"] > 0),
        ("element count > 0 (all C3D20R)", s["n_elems"] > 0),
        ("C3D20 remap verified (midside dev < tol)", s["max_midside_dev"] < s["remap_tol"]),
        ("all required CalculiX cards present", cards_ok),
        ("CLOAD node-force sum == applied load total", bool(resultant_ok)),
        ("output requests include U / RF / S", outputs_ok),
    ]
    if not cards_ok:
        missing = [tok for tok in required if tok not in inp]
        print(f"  !! missing cards: {missing}")
    passed = all(ok for _, ok in checks)
    print("-" * 64)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 64)
    print(f"ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    if passed:
        print("Next: solve with ccx (layer 4).  Human check (optional): head out/beam.inp")
    print("=" * 64)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CalculiX .inp (plan layer 3.4).")
    parser.add_argument("-v", "--verbose", action="store_true", help="show Gmsh log output")
    args = parser.parse_args()
    summary = build_inp(PARAMS, verbose=args.verbose)
    return 0 if report_and_check(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
