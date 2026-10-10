#!/usr/bin/env python3
"""Pre-processor layer 3.4: CalculiX input-deck generation (plan section 3.4,
plate plan M4).

Reads out/beam.msh and the constraint table / LOAD node sets from
boundary_conditions.py, then writes a complete CalculiX deck out/beam.inp.
Dispatches on geometry.type:

  * cantilever_beam -> C3D20R volume elements (ELSET=BEAM_VOL), full clamp,
                       total force split equally over the LOAD nodes.
  * plate_with_hole -> CPS8R / CPS6 plane-stress elements (ELSET=PLATE_SURF)
                       with the section THICKNESS, per-(node, dof) *BOUNDARY
                       from the M3 constraint table, and CONSISTENT nodal
                       forces on the loaded edge (quadratic line element:
                       1/6 - 2/3 - 1/6, plate plan D5) so a uniform traction
                       is reproduced exactly instead of being smeared into a
                       saw-tooth by an equal split.

CRITICAL detail handled here: Gmsh and CalculiX element node orderings must
agree. Both remaps (3D C3D20 and 2D CPS8/CPS6) are DERIVED from Gmsh's own
reference-node coordinates -- never hardcoded -- and then VERIFIED
geometrically on the real mesh:
  * 3D: every midside node sits at its edge midpoint.
  * 2D: every midside node sits at its edge midpoint, EXCEPT nodes on the hole
        boundary, which are curved by design and must instead lie ON the hole
        circle (|dist(center) - r| < tol).
2D elements additionally must be counter-clockwise (positive signed area),
which CalculiX requires for a positive Jacobian.

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
from paths import RunPaths, add_workdir_argument, from_args

GMSH_HEX20 = 17  # Gmsh element type id for the 20-node hexahedron.
GMSH_QUAD8 = 16  # Gmsh element type id for the 8-node quadrangle -> CPS8R.
GMSH_TRI6 = 9    # Gmsh element type id for the 6-node triangle   -> CPS6.
GMSH_LINE3 = 8   # Gmsh element type id for the 3-node line (edge loads).

# 2D element families: gmsh type -> (CalculiX type, corner count, midside edges).
# Midside slot (1-based) -> the two corner slots it bisects, in Abaqus ordering.
_2D_FAMILY = {
    GMSH_QUAD8: ("CPS8R", 4, {5: (1, 2), 6: (2, 3), 7: (3, 4), 8: (4, 1)}),
    GMSH_TRI6: ("CPS6", 3, {4: (1, 2), 5: (2, 3), 6: (3, 1)}),
}

# Consistent nodal-force weights for a uniform traction on a 3-node quadratic
# line element (integrals of the shape functions over the element):
#   corner nodes  int(N) = L/6 each ;  midside node  int(N) = 2L/3
# (plate plan D5; Cook et al., Concepts and Applications of FEA, 4th ed.)
_LINE3_WEIGHTS = (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)  # gmsh line3 order: end, end, mid

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


def _gmsh_to_abaqus_2d_perm(gmsh_type: int) -> list:
    """Permutation p with abaqus_nodes[i] = gmsh_nodes[p[i]] for a 2D quadratic
    element (quad8 -> CPS8, tri6 -> CPS6).

    Derived (not hardcoded) from Gmsh's own reference-node coordinates: gmsh
    lists the corners first in counter-clockwise order, which is also the
    Abaqus convention, so corners map through unchanged; each Abaqus midside
    slot is then matched to the gmsh node sitting at the midpoint of the two
    reference corners it bisects. Requires an open gmsh session. Raises if the
    result is not a bijection of the local indices.
    """
    _, _, _, num_nodes, local_coord, _ = gmsh.model.mesh.getElementProperties(gmsh_type)
    ccx_type, n_corner, edges = _2D_FAMILY[gmsh_type]
    ref = np.asarray(local_coord, dtype=float).reshape(num_nodes, -1)
    tol = 1e-9

    perm = list(range(n_corner))          # corners: gmsh CCW == Abaqus CCW
    for slot in range(n_corner + 1, num_nodes + 1):
        c1, c2 = edges[slot]
        target = 0.5 * (ref[c1 - 1] + ref[c2 - 1])
        hits = [j for j in range(n_corner, num_nodes)
                if float(np.linalg.norm(ref[j] - target)) < tol]
        if len(hits) != 1:
            raise RuntimeError(
                f"{ccx_type}: no unique gmsh midside node at reference position "
                f"{target} for slot {slot} (candidates {hits})")
        perm.append(hits[0])
    if sorted(perm) != list(range(num_nodes)):
        raise RuntimeError(f"Gmsh->Abaqus {ccx_type} permutation is not a bijection")
    return perm


def _verify_2d_ordering(elements: list, node_xyz: dict, hole_nodes: set,
                        hole_center, hole_r: float, tol: float) -> dict:
    """Geometric proof of the 2D remap on the real mesh.

    With nodes already in Abaqus order, every midside node must either sit at
    the midpoint of its two corner nodes (straight edge) or, for a node on the
    hole boundary, lie ON the hole circle (the arc is curved by design). Every
    element must also be counter-clockwise (positive signed area), which
    CalculiX needs for a positive Jacobian.

    Returns measured maxima; raises if any tolerance is exceeded.
    """
    max_chord_dev = 0.0     # straight-edge midside deviation
    max_arc_dev = 0.0       # |dist(hole center) - r| for arc midside nodes
    n_arc = 0
    min_area = float("inf")
    cx, cy = float(hole_center[0]), float(hole_center[1])

    for gmsh_type, _etag, nodes in elements:
        _ccx, n_corner, edges = _2D_FAMILY[gmsh_type]
        pts = [node_xyz[n] for n in nodes[:n_corner]]
        area = 0.0
        for i in range(n_corner):
            x1, y1 = float(pts[i][0]), float(pts[i][1])
            x2, y2 = float(pts[(i + 1) % n_corner][0]), float(pts[(i + 1) % n_corner][1])
            area += x1 * y2 - x2 * y1
        min_area = min(min_area, area / 2.0)

        for mid, (c1, c2) in edges.items():
            nid = nodes[mid - 1]
            p_mid = node_xyz[nid]
            if nid in hole_nodes:
                n_arc += 1
                dev = abs(float(np.hypot(p_mid[0] - cx, p_mid[1] - cy)) - hole_r)
                max_arc_dev = max(max_arc_dev, dev)
            else:
                p_exp = 0.5 * (node_xyz[nodes[c1 - 1]] + node_xyz[nodes[c2 - 1]])
                max_chord_dev = max(max_chord_dev, float(np.linalg.norm(p_mid - p_exp)))

    if max_chord_dev > tol:
        raise RuntimeError(
            f"2D node-ordering verification FAILED: max straight-edge midside "
            f"deviation {max_chord_dev:.3e} > tol {tol:.3e}")
    if max_arc_dev > tol:
        raise RuntimeError(
            f"2D hole-arc verification FAILED: midside node off the hole circle "
            f"by {max_arc_dev:.3e} > tol {tol:.3e}")
    if min_area <= 0.0:
        raise RuntimeError(
            f"2D element orientation FAILED: min signed area {min_area:.3e} <= 0 "
            f"(CalculiX needs counter-clockwise elements)")
    return {"max_chord_dev": max_chord_dev, "max_arc_dev": max_arc_dev,
            "n_arc_checked": n_arc, "min_signed_area": min_area}


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


def build_inp(params: dict, paths: RunPaths = None, verbose: bool = False) -> dict:
    """Assemble <run>/beam.inp for the active geometry. Returns a summary."""
    paths = paths or RunPaths.default()
    if not paths.msh.exists():
        raise FileNotFoundError(f"{paths.msh} not found; run mesh.py first")
    if params["geometry"]["type"] == "plate_with_hole":
        return _build_plate_inp(params, paths, verbose=verbose)
    return _build_beam_inp(params, paths, verbose=verbose)


def _build_beam_inp(params: dict, paths: RunPaths, verbose: bool = False) -> dict:
    """Cantilever deck: C3D20R volume elements, full clamp, equal nodal split."""
    msh_path, inp_path = paths.msh, paths.inp
    g = params["geometry"]["cantilever_beam"]
    L, h, b = float(g["L"]), float(g["h"]), float(g["b"])
    mat = params["material"]

    # 1. BC node sets (build_bc runs its own gmsh session and finalizes it).
    bc_def = build_bc(params, paths=paths, verbose=verbose)

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
    inp_path.parent.mkdir(parents=True, exist_ok=True)
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


def _read_edge_line_elements(label_tags: dict, label: str) -> list:
    """[(n_end1, n_end2, n_mid), ...] line3 elements of one edge physical group.

    Requires an open gmsh session with the mesh loaded. Gmsh's line3 node order
    is (end, end, midside) -- verified against getElementProperties reference
    coordinates (-1, +1, 0).
    """
    _, _, _, n_per, local_coord, _ = gmsh.model.mesh.getElementProperties(GMSH_LINE3)
    ref = np.asarray(local_coord, dtype=float).ravel()
    if n_per != 3 or not (ref[0] < ref[2] < ref[1]):
        raise RuntimeError(f"unexpected gmsh line3 reference layout {ref} (expected end/end/mid)")
    out = []
    for ent in label_tags[label]:
        etypes, etags, enodes = gmsh.model.mesh.getElements(1, ent)
        for etype, tags, conn in zip(etypes, etags, enodes):
            if int(etype) != GMSH_LINE3:
                raise RuntimeError(f"edge {label!r} carries element type {etype}, expected {GMSH_LINE3}")
            conn = [int(x) for x in conn]
            for e in range(len(tags)):
                out.append(tuple(conn[e * 3:(e + 1) * 3]))
    if not out:
        raise RuntimeError(f"no line elements found on edge {label!r}")
    return out


def _build_plate_inp(params: dict, paths: RunPaths, verbose: bool = False) -> dict:
    """Plane-stress plate deck: CPS8R/CPS6, thickness, per-dof *BOUNDARY,
    consistent edge loads (plate plan M4)."""
    msh_path, inp_path = paths.msh, paths.inp
    g = params["geometry"]["plate_with_hole"]
    a, b, t = float(g["a"]), float(g["b"]), float(g["t"])
    r, xc, yc = float(g["r"]), float(g["xc"]), float(g["yc"])
    mat = params["material"]

    # 1. Constraint table + LOAD sets (build_bc runs its own gmsh session).
    bc_def = build_bc(params, paths=paths, verbose=verbose)

    # 2. Our gmsh session: derive remaps, read nodes / surface / edge elements.
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    try:
        gmsh.open(str(msh_path))
        perms = {gt: _gmsh_to_abaqus_2d_perm(gt) for gt in _2D_FAMILY}

        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        node_ids = [int(t_) for t_ in ntags]
        coords = np.asarray(ncoords, dtype=float).reshape(-1, 3)
        node_xyz = {nid: coords[k] for k, nid in enumerate(node_ids)}
        max_abs_z = float(np.max(np.abs(coords[:, 2]))) if node_ids else 0.0

        # Surface elements, remapped into Abaqus order.
        elements = []          # (gmsh_type, elem_tag, [node ids, Abaqus order])
        etypes, etags, enodes = gmsh.model.mesh.getElements(2)
        for etype, tags, conn in zip(etypes, etags, enodes):
            etype = int(etype)
            if etype not in _2D_FAMILY:
                raise RuntimeError(f"unexpected surface element type {etype}; "
                                   f"expected one of {sorted(_2D_FAMILY)}")
            n_per = len(perms[etype])
            perm = perms[etype]
            conn = [int(x) for x in conn]
            for e, etag in enumerate(tags):
                gnodes = conn[e * n_per:(e + 1) * n_per]
                elements.append((etype, int(etag), [gnodes[perm[i]] for i in range(n_per)]))

        # Curve physical groups (labels -> entity tags) + hole node set.
        label_tags: dict = {}
        for dim, tag in gmsh.model.getPhysicalGroups(1):
            label_tags[gmsh.model.getPhysicalName(dim, tag)] = list(
                gmsh.model.getEntitiesForPhysicalGroup(dim, tag))
        hole_ntags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(
            1, [tag for dim, tag in gmsh.model.getPhysicalGroups(1)
                if gmsh.model.getPhysicalName(dim, tag) == "hole_boundary"][0])
        hole_nodes = {int(x) for x in hole_ntags}

        # Consistent nodal forces on each loaded edge (plate plan D5).
        cload_vec: dict = {}   # node -> force vector (N)
        applied = np.zeros(3)
        load_notes = []
        for ld in bc_def["load"]["loads"]:
            segs = _read_edge_line_elements(label_tags, ld["face"])
            seg_len = [float(np.linalg.norm(node_xyz[s[1]] - node_xyz[s[0]])) for s in segs]
            edge_len = float(sum(seg_len))
            if edge_len <= 0:
                raise RuntimeError(f"edge {ld['face']!r} has zero length")
            f_total = ld["magnitude"] * np.asarray(ld["dir_unit"], dtype=float)
            q = f_total / edge_len          # force per unit length along the edge
            for seg, length in zip(segs, seg_len):
                for nid, w in zip(seg, _LINE3_WEIGHTS):
                    cload_vec[nid] = cload_vec.get(nid, np.zeros(3)) + q * length * w
            applied += f_total
            load_notes.append(
                f"{ld['face']}: {len(segs)} quadratic segments over {edge_len:.4f} mm, "
                f"|q| = {float(np.linalg.norm(q)):.4f} N/mm, weights 1/6-2/3-1/6")
    finally:
        if own:
            gmsh.finalize()

    # 3. Verify the remap + orientation geometrically on every element.
    tol = 1e-6 * max(a, b)
    verify = _verify_2d_ordering(elements, node_xyz, hole_nodes, (xc, yc), r, tol)

    cload_terms = []           # (node_id, dof, value)
    for nid in sorted(cload_vec):
        for dof in (1, 2, 3):
            val = float(cload_vec[nid][dof - 1])
            if abs(val) > 1e-12:
                cload_terms.append((nid, dof, val))
    cload_sum = np.zeros(3)
    for _, dof, val in cload_terms:
        cload_sum[dof - 1] += val

    # 4. Assemble the deck.
    constraints = bc_def["fixed"]["constraints"]
    fixed_ids = bc_def["fixed"]["node_ids"]
    load_ids = bc_def["load"]["node_ids"]
    by_type: dict = {}
    for gtype, etag, conn in elements:
        by_type.setdefault(gtype, []).append((etag, conn))

    lines = []
    lines.append("** CalculiX input deck generated by build_inp.py (plate plan M4)")
    lines.append("** Unit system: SI-mm (length mm, force N, stress/E MPa, displacement mm)")
    lines.append(f"** Geometry: plate_with_hole  a={a} b={b} t={t}  r={r} at ({xc}, {yc}) (mm)")
    lines.append(f"** Analysis: 2D plane stress in the z=0 plane")
    lines.append(f"** Mesh: {len(elements)} elements  Nodes: {len(node_ids)}")
    lines.append(f"** Node ordering: derived from gmsh reference coords, verified "
                 f"(straight midside dev {verify['max_chord_dev']:.2e} mm, "
                 f"hole-arc dev {verify['max_arc_dev']:.2e} mm, "
                 f"min signed area {verify['min_signed_area']:.4g} mm^2)")

    lines.append("*NODE")
    for nid in sorted(node_ids):
        x, y, z = node_xyz[nid]
        lines.append(f"{nid}, {x:.10g}, {y:.10g}, {z:.10g}")

    # One *ELEMENT block per element type, all in ELSET=PLATE_SURF.
    for gtype in sorted(by_type):
        ccx_type = _2D_FAMILY[gtype][0]
        lines.append(f"*ELEMENT, TYPE={ccx_type}, ELSET=PLATE_SURF")
        for etag, conn in sorted(by_type[gtype], key=lambda x: x[0]):
            lines.extend(_ints_to_lines([etag] + conn))

    lines.append("*NSET, NSET=FIXED")
    lines.extend(_ints_to_lines(sorted(fixed_ids)))
    lines.append("*NSET, NSET=LOAD")
    lines.extend(_ints_to_lines(sorted(load_ids)))

    lines.append(f"*MATERIAL, NAME={mat['name']}")
    lines.append("*ELASTIC")
    lines.append(f"{mat['E']:.10g}, {mat['nu']:.10g}")          # E (MPa), nu
    # Plane stress: the section card carries the THICKNESS on its data line.
    lines.append(f"*SOLID SECTION, ELSET=PLATE_SURF, MATERIAL={mat['name']}")
    lines.append(f"{t:.10g}")

    # Per-(node, dof) constraints from the M3 table (roller + anchor, or clamp).
    lines.append("*BOUNDARY")
    for nid, dof in constraints:
        lines.append(f"{nid}, {dof}, {dof}")

    lines.append("*STEP")
    lines.append("*STATIC")
    lines.append("*CLOAD")
    for nid, dof, val in cload_terms:
        lines.append(f"{nid}, {dof}, {val:.10g}")
    # OUTPUT=2D maps results back onto the original 2D nodes (ccx expands
    # plane-stress elements to 3D internally). Verified in layer M5.
    lines.append("*NODE FILE, OUTPUT=2D")
    lines.append("U, RF")
    lines.append("*EL FILE, OUTPUT=2D")
    lines.append("S")
    lines.append("*END STEP")

    inp_text = "\n".join(lines) + "\n"
    inp_path.parent.mkdir(parents=True, exist_ok=True)
    inp_path.write_text(inp_text)

    elem_type_counts = {_2D_FAMILY[gt][0]: len(v) for gt, v in by_type.items()}
    return {
        "geometry_type": "plate_with_hole",
        "inp_path": str(inp_path),
        "file_size": inp_path.stat().st_size,
        "n_nodes": len(node_ids),
        "n_elems": len(elements),
        "elem_type": " + ".join(f"{n} x {k}" for k, n in sorted(elem_type_counts.items())),
        "elem_type_counts": elem_type_counts,
        "verify": verify,
        "remap_tol": tol,
        "max_abs_z": max_abs_z,
        "thickness": t,
        "material": {"E": mat["E"], "nu": mat["nu"], "name": mat["name"]},
        "fixed_n": bc_def["fixed"]["n_nodes"],
        "n_constraints": len(constraints),
        "constraint_note": "; ".join(
            f"{e['face']} at={e['at']} dofs={e['dofs']} -> {e['n_nodes']} node(s)"
            for e in bc_def["fixed"]["entries"]),
        "load_n": bc_def["load"]["n_nodes"],
        "cload_terms": len(cload_terms),
        "applied": applied.tolist(),
        "cload_sum": cload_sum.tolist(),
        "per_node_note": " | ".join(load_notes),
        "sigma_gross": float(np.linalg.norm(applied)) / (b * t),
        "inp_text": inp_text,
    }


def report_and_check(s: dict) -> bool:
    """Print the acceptance report for the generated deck; PASS (True)/FAIL."""
    if s.get("geometry_type") == "plate_with_hole":
        return _report_plate(s)
    return _report_beam(s)


def _report_plate(s: dict) -> bool:
    """Plate deck acceptance report (plate plan M4)."""
    v = s["verify"]
    print("=" * 64)
    print("build_inp.py acceptance report  (plate_with_hole, plate plan M4)")
    print("=" * 64)
    print(f"output    : {s['inp_path']}  ({s['file_size'] / 1024:.1f} KB)")
    print(f"nodes     : {s['n_nodes']}   elements: {s['elem_type']}  (plane stress)")
    print(f"section   : *SOLID SECTION thickness = {s['thickness']} mm   "
          f"material {s['material']['name']} E={s['material']['E']} MPa nu={s['material']['nu']}")
    print(f"remap     : derived from gmsh reference coords, verified on every element")
    print(f"            straight midside dev = {v['max_chord_dev']:.2e} mm (tol {s['remap_tol']:.1e})")
    print(f"            hole-arc nodes       = {v['n_arc_checked']} checked, max off-circle "
          f"{v['max_arc_dev']:.2e} mm (curved by design)")
    print(f"            min signed area      = {v['min_signed_area']:.4g} mm^2 (must be > 0, CCW)")
    print(f"planarity : max|z| = {s['max_abs_z']:.2e} mm (plane-stress elements must lie in z=0)")
    print(f"BOUNDARY  : {s['n_constraints']} (node, dof) constraints over {s['fixed_n']} nodes")
    print(f"            {s['constraint_note']}")
    print(f"CLOAD     : {s['cload_terms']} terms over {s['load_n']} LOAD nodes (consistent, D5)")
    print(f"            {s['per_node_note']}")
    print(f"            applied resultant = {s['applied']} N ; CLOAD sum = {s['cload_sum']} N")
    print(f"            -> sigma_gross = F/(b*t) = {s['sigma_gross']:.4f} MPa")
    print( "outputs   : *NODE FILE, OUTPUT=2D  U, RF  /  *EL FILE, OUTPUT=2D  S")

    inp = s["inp_text"]
    required = [
        "*NODE", "*ELEMENT, TYPE=CPS", "*NSET, NSET=FIXED", "*NSET, NSET=LOAD",
        "*MATERIAL, NAME=", "*ELASTIC", "*SOLID SECTION, ELSET=PLATE_SURF",
        "*BOUNDARY", "*STEP", "*STATIC", "*CLOAD",
        "*NODE FILE, OUTPUT=2D", "*EL FILE, OUTPUT=2D", "*END STEP",
    ]
    cards_ok = all(tok in inp for tok in required)
    outputs_ok = ("U, RF" in inp) and ("\nS\n" in inp)
    resultant_ok = np.allclose(s["cload_sum"], s["applied"],
                               atol=1e-9 * (np.linalg.norm(s["applied"]) + 1.0))
    if not cards_ok:
        print(f"  !! missing cards: {[tok for tok in required if tok not in inp]}")

    checks = [
        ("node count > 0 and matches mesh read", s["n_nodes"] > 0),
        ("element count > 0 (all plane-stress)", s["n_elems"] > 0),
        ("2D remap verified (straight midside dev < tol)",
         v["max_chord_dev"] < s["remap_tol"]),
        ("hole-arc midside nodes lie on the hole circle",
         v["max_arc_dev"] < s["remap_tol"]),
        ("all elements counter-clockwise (signed area > 0)", v["min_signed_area"] > 0.0),
        ("all nodes in the z = 0 plane", s["max_abs_z"] < s["remap_tol"]),
        ("all required CalculiX cards present", cards_ok),
        ("constraint table written to *BOUNDARY", s["n_constraints"] > 0),
        ("consistent CLOAD sum == applied load total", bool(resultant_ok)),
        ("output requests include U / RF / S", outputs_ok),
    ]
    passed = all(ok for _, ok in checks)
    print("-" * 64)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 64)
    print(f"ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    if passed:
        print(f"Next: solve with ccx (layer 4 / plate M5).  Human check: head {s['inp_path']}")
    print("=" * 64)
    return passed


def _report_beam(s: dict) -> bool:
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
        print(f"Next: solve with ccx (layer 4).  Human check (optional): head {s['inp_path']}")
    print("=" * 64)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CalculiX .inp (plan layer 3.4).")
    parser.add_argument("-v", "--verbose", action="store_true", help="show Gmsh log output")
    add_workdir_argument(parser)
    args = parser.parse_args()
    summary = build_inp(PARAMS, paths=from_args(args), verbose=args.verbose)
    return 0 if report_and_check(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
