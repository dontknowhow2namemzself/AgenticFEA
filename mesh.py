#!/usr/bin/env python3
"""Pre-processor layer 3.1: geometry + mesh generation (plan section 3.1,
plate plan M2).

Dispatches on geometry.type:
  * cantilever_beam  -> box solid, BEAM_VOL + face_x_min..face_z_max labels,
                        second-order structured hex mesh (C3D20R downstream)
  * plate_with_hole  -> 2D rectangle-minus-disk in the z = 0 plane, PLATE_SURF
                        + edge/hole curve labels, second-order quad-dominant
                        (or all-tri) mesh with local refinement at the hole
                        (CPS8R/CPS6 downstream)

Both write out/beam.msh in MSH 2.2 ASCII (artifact basename kept per plate
plan D8).

Scope boundary (do not cross): this module describes ONLY what the geometry
looks like and where its boundary entities are. It carries NO boundary-
condition meaning. Mapping geometric labels to FIXED / LOAD is the job of
boundary_conditions.py.

Naming contract (shared across layers, do not change):
    cantilever: volume group BEAM_VOL; face groups face_x_min ... face_z_max
    plate:      surface group PLATE_SURF; curve groups edge_x_min / edge_x_max
                / edge_y_min / edge_y_max / hole_boundary (imported from
                schema.PLATE_EDGE_LABELS so the contract lives in one place)

Unit system: SI-mm (see params.py).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import gmsh
import numpy as np

from params import PARAMS
from schema import PLATE_EDGE_LABELS

# Project-anchored paths (work regardless of the caller's CWD).
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
MSH_PATH = OUT_DIR / "beam.msh"

# Fixed naming contract for the six box faces (neutral geometry, no BC meaning).
FACE_LABELS = (
    "face_x_min", "face_x_max",
    "face_y_min", "face_y_max",
    "face_z_min", "face_z_max",
)

# Gmsh element type id -> (human name, CalculiX element it maps to downstream).
_GMSH_ELEM_TYPES = {
    5: ("8-node hexahedron", "C3D8"),
    17: ("20-node hexahedron", "C3D20R"),
    12: ("27-node hexahedron", "C3D27 (unused)"),
    4: ("4-node tetrahedron", "C3D4"),
    11: ("10-node tetrahedron", "C3D10"),
    2: ("3-node triangle", "CPS3"),
    3: ("4-node quadrangle", "CPS4R"),
    9: ("6-node triangle", "CPS6"),
    16: ("8-node quadrangle", "CPS8R"),
}
# Corner-node count per 2D gmsh element type (gmsh orders corners first).
_CORNERS_2D = {2: 3, 3: 4, 9: 3, 16: 4}

# Plate meshing policy (plate plan D7). The sizing target 28 leaves headroom so
# the acceptance gate (>= 24 segments around the hole) holds robustly.
HOLE_TARGET_SEGMENTS = 28
HOLE_MIN_SEGMENTS = 24
# Elements demanded across the THINNEST ligament. Resolving the hole's
# circumference is not enough on its own: a hole sitting close to an edge
# concentrates its gradient across the narrow strip of material left there, and
# the 20-case pilot found a case whose ligament spanned only 4.1 elements and
# whose free-surface traction residual came out at 12.2% (vs ~6% for a centred
# hole at the same setting). Refining it -- 4.1 / 8.2 / 16.5 elements across the
# ligament -> 12.21% / 4.09% / 1.43% -- confirmed resolution, not physics.
LIGAMENT_MIN_ELEMS = 6
# Local refinement may go at most this much finer than the far-field size. The
# rules above are ratios, and a ratio has no floor: a hole left 0.001 mm from
# the plate edge asks for 0.00017 mm elements, which is not a failure the
# mesher reports -- it is 4e11 elements and a run that never returns. Bounding
# the ratio turns "cannot mesh this" into an error instead of a hang, and it
# defines what the schema should refuse in the first place.
MAX_REFINE_RATIO = 20
REFINE_DIST_MIN_R = 0.3    # uniform fine band within 0.3*r of the hole edge
REFINE_DIST_MAX_R = 2.0    # linear growth back to elem_size by 2*r away
PLATE_AREA_TOL = 0.005     # mesh-area gate vs a*b - pi*r^2 (0.5%; includes the
                           # chord-vs-arc residual of corner-polygon areas)


def _divisions(length: float, elem_size: float) -> int:
    """Number of element divisions along an edge of the given length."""
    return max(1, int(round(length / elem_size)))


def _classify_faces(tol: float) -> dict:
    """Map each box-face label to its CAD surface tag, purely by geometry.

    An axis-aligned box face is flat along exactly one axis (min == max on that
    axis), and that flat coordinate equals the global min or max of the model.
    The resulting labels are neutral orientation tags -> they carry no BC meaning.
    """
    gxmin, gymin, gzmin, gxmax, gymax, gzmax = gmsh.model.getBoundingBox(-1, -1)
    label_to_tag: dict = {}
    for dim, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        flat_x = abs(xmax - xmin) < tol
        flat_y = abs(ymax - ymin) < tol
        flat_z = abs(zmax - zmin) < tol
        label = None
        if flat_x and abs(xmin - gxmin) < tol:
            label = "face_x_min"
        elif flat_x and abs(xmax - gxmax) < tol:
            label = "face_x_max"
        elif flat_y and abs(ymin - gymin) < tol:
            label = "face_y_min"
        elif flat_y and abs(ymax - gymax) < tol:
            label = "face_y_max"
        elif flat_z and abs(zmin - gzmin) < tol:
            label = "face_z_min"
        elif flat_z and abs(zmax - gzmax) < tol:
            label = "face_z_max"
        if label is None:
            raise RuntimeError(f"surface tag {tag} did not match any box-face label")
        if label in label_to_tag:
            raise RuntimeError(f"duplicate face label {label!r} (geometry is not a box?)")
        label_to_tag[label] = tag
    missing = [lbl for lbl in FACE_LABELS if lbl not in label_to_tag]
    if missing:
        raise RuntimeError(f"missing face labels after classification: {missing}")
    return label_to_tag


def _set_structured_hex(nx: int, ny: int, nz: int) -> None:
    """Make the box mesh a structured all-hex grid (transfinite + recombine)."""
    # Transfinite node count per curve, chosen by the curve's axis direction.
    # For an axis-aligned box edge exactly one extent is the edge length.
    for dim, tag in gmsh.model.getEntities(1):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        lx, ly, lz = xmax - xmin, ymax - ymin, zmax - zmin
        if lx >= ly and lx >= lz:
            n = nx
        elif ly >= lx and ly >= lz:
            n = ny
        else:
            n = nz
        gmsh.model.mesh.setTransfiniteCurve(tag, n + 1)
    for dim, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(dim, tag)  # quads -> hexes after generate(3)
    for dim, tag in gmsh.model.getEntities(3):
        gmsh.model.mesh.setTransfiniteVolume(tag)


def build_mesh(params: dict, verbose: bool = False) -> dict:
    """Build the mesh for the active geometry, write out/beam.msh, and return
    an acceptance summary. Dispatches on geometry.type (the extension point)."""
    geom_type = params["geometry"]["type"]
    if geom_type == "cantilever_beam":
        return _build_cantilever_mesh(params, verbose=verbose)
    if geom_type == "plate_with_hole":
        return _build_plate_mesh(params, verbose=verbose)
    raise NotImplementedError(f"geometry.type={geom_type!r} is not supported")


def _build_cantilever_mesh(params: dict, verbose: bool = False) -> dict:
    """Structured second-order hex mesh for the cantilever box (plan 3.1)."""
    g = params["geometry"]["cantilever_beam"]
    L, h, b = float(g["L"]), float(g["h"]), float(g["b"])

    m = params["mesh"]
    elem_size = float(m["elem_size"])
    elem_order = int(m["elem_order"])
    elem_shape = str(m["elem_shape"])
    if elem_shape != "hex":
        # Reserved extension point: "tet" (free meshing -> C3D10) is needed for
        # future non-box geometries that cannot be structured-meshed. It is not
        # part of the MVP verification milestone.
        raise NotImplementedError(
            f"elem_shape={elem_shape!r} is reserved; the MVP implements 'hex' only"
        )

    # Edge divisions: b -> y, h -> z (axis convention, see params.py).
    nx = _divisions(L, elem_size)
    ny = _divisions(b, elem_size)
    nz = _divisions(h, elem_size)
    tol = 1e-6 * max(L, h, b)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("cantilever_beam")

        # Geometry: box with the cross-section centered on the x-axis so the
        # neutral axis sits at y = z = 0 (clean tip-deflection sign in z).
        #   origin (x0, y0, z0) = (0, -b/2, -h/2);  extents (dx, dy, dz) = (L, b, h)
        gmsh.model.occ.addBox(0.0, -b / 2.0, -h / 2.0, L, b, h)
        gmsh.model.occ.synchronize()

        # Whole solid -> BEAM_VOL.
        vol_tags = [tag for _, tag in gmsh.model.getEntities(3)]
        gmsh.model.addPhysicalGroup(3, vol_tags, name="BEAM_VOL")

        # Neutral geometric face labels (NO BC meaning is assigned here).
        label_to_tag = _classify_faces(tol)
        face_phys = {}
        for label in FACE_LABELS:
            face_phys[label] = gmsh.model.addPhysicalGroup(2, [label_to_tag[label]], name=label)

        # Structured second-order hexahedral mesh (-> C3D20R).
        _set_structured_hex(nx, ny, nz)
        gmsh.option.setNumber("Mesh.ElementOrder", elem_order)
        if elem_order == 2:
            # 20-node serendipity hex (C3D20R), NOT the 27-node full hex.
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
        gmsh.model.mesh.generate(3)

        # ---- gather acceptance data ----
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)

        elem_counts: dict = {}          # gmsh type id -> count
        vol_elem_tags: list = []
        etypes, etags, _ = gmsh.model.mesh.getElements(3)
        for etype, tags in zip(etypes, etags):
            elem_counts[int(etype)] = elem_counts.get(int(etype), 0) + len(tags)
            vol_elem_tags.extend(int(t) for t in tags)

        # Min element quality: minSICN <= 0 means an inverted / negative-Jacobian element.
        qual = gmsh.model.mesh.getElementQualities(vol_elem_tags, "minSICN")
        min_quality = float(np.min(qual)) if len(qual) else float("nan")

        # Per-face node counts (each must be > 0).
        face_node_counts = {}
        for label in FACE_LABELS:
            ntags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, face_phys[label])
            face_node_counts[label] = len(ntags)

        # Write MSH 2.2 ASCII; SaveAll=0 -> only physical-group entities are written.
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(str(MSH_PATH))
    finally:
        gmsh.finalize()

    file_size = MSH_PATH.stat().st_size if MSH_PATH.exists() else 0
    return {
        "geometry": {"type": "cantilever_beam", "L": L, "h": h, "b": b},
        "mesh": {
            "elem_shape": elem_shape, "elem_order": elem_order,
            "elem_size": elem_size, "nx": nx, "ny": ny, "nz": nz,
        },
        "n_nodes": n_nodes,
        "n_vol_elems": sum(elem_counts.values()),
        "elem_counts": elem_counts,
        "min_quality": min_quality,
        "face_node_counts": face_node_counts,
        "msh_path": str(MSH_PATH),
        "file_size": file_size,
    }


def _shoelace(pts) -> float:
    """Unsigned polygon area from an ordered list of (x, y, z) corner points."""
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = float(pts[i][0]), float(pts[i][1])
        x2, y2 = float(pts[(i + 1) % n][0]), float(pts[(i + 1) % n][1])
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _classify_plate_curves(a: float, b: float, xc: float, yc: float, r: float,
                           tol: float) -> dict:
    """Map every dim-1 CAD curve to a plate edge label, purely by geometry.

    The four outer edges are axis-aligned straight lines at x=0/a or y=0/b
    (flat along one axis). Every remaining curve must lie inside the hole's
    bounding box -> hole_boundary (OCC may split the circle into several arcs,
    so that label may carry more than one curve tag). Labels are neutral
    orientation tags -> they carry no BC meaning.
    """
    label_to_tags: dict = {lbl: [] for lbl in PLATE_EDGE_LABELS}
    for dim, tag in gmsh.model.getEntities(1):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
        flat_x = abs(xmax - xmin) < tol
        flat_y = abs(ymax - ymin) < tol
        if flat_x and abs(xmin - 0.0) < tol:
            label_to_tags["edge_x_min"].append(tag)
        elif flat_x and abs(xmin - a) < tol:
            label_to_tags["edge_x_max"].append(tag)
        elif flat_y and abs(ymin - 0.0) < tol:
            label_to_tags["edge_y_min"].append(tag)
        elif flat_y and abs(ymin - b) < tol:
            label_to_tags["edge_y_max"].append(tag)
        else:
            in_hole_bbox = (xmin >= xc - r - tol and xmax <= xc + r + tol
                            and ymin >= yc - r - tol and ymax <= yc + r + tol)
            if not in_hole_bbox:
                raise RuntimeError(
                    f"curve tag {tag} (bbox x[{xmin:.3g},{xmax:.3g}] "
                    f"y[{ymin:.3g},{ymax:.3g}]) matches no plate edge label")
            label_to_tags["hole_boundary"].append(tag)
    for lbl in ("edge_x_min", "edge_x_max", "edge_y_min", "edge_y_max"):
        if len(label_to_tags[lbl]) != 1:
            raise RuntimeError(f"expected exactly 1 curve for {lbl!r}, "
                               f"got {label_to_tags[lbl]}")
    if not label_to_tags["hole_boundary"]:
        raise RuntimeError("no hole_boundary curve found (hole not cut?)")
    return label_to_tags


def _build_plate_mesh(params: dict, verbose: bool = False) -> dict:
    """2D plane-stress plate-with-hole mesh (plate plan M2).

    Rectangle minus disk in the z = 0 plane, second-order elements, local
    refinement at the hole via a Distance/Threshold size field (policy D7):
    local size h_hole = min(elem_size, 2*pi*r / HOLE_TARGET_SEGMENTS) held
    within REFINE_DIST_MIN_R * r of the hole edge, growing linearly back to
    elem_size by REFINE_DIST_MAX_R * r.
    """
    g = params["geometry"]["plate_with_hole"]
    a, b, t = float(g["a"]), float(g["b"]), float(g["t"])
    r, xc, yc = float(g["r"]), float(g["xc"]), float(g["yc"])

    m = params["mesh"]
    elem_size = float(m["elem_size"])
    elem_order = int(m["elem_order"])
    elem_shape = str(m["elem_shape"])
    if elem_shape not in ("quad", "tri"):
        raise NotImplementedError(
            f"elem_shape={elem_shape!r} is not a 2D strategy (plate: quad | tri)")

    min_ligament = min(xc - r, a - xc - r, yc - r, b - yc - r)
    h_floor = elem_size / MAX_REFINE_RATIO
    h_hole = min(elem_size,
                 2.0 * math.pi * r / HOLE_TARGET_SEGMENTS,
                 min_ligament / LIGAMENT_MIN_ELEMS)
    if h_hole < h_floor:
        raise ValueError(
            f"geometry cannot be meshed at a bounded cost: the thinnest ligament is "
            f"{min_ligament:.4g} mm, so resolving it with {LIGAMENT_MIN_ELEMS} elements "
            f"would need {h_hole:.3g} mm elements -- finer than the floor "
            f"{h_floor:.3g} mm (elem_size / {MAX_REFINE_RATIO}). Move the hole away "
            f"from the edge or make it smaller."
        )
    tol = 1e-6 * max(a, b)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("plate_with_hole")

        # Geometry: rectangle [0,a]x[0,b] minus the disk at (xc, yc), z = 0.
        rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, a, b)
        disk = gmsh.model.occ.addDisk(xc, yc, 0.0, r, r)
        gmsh.model.occ.cut([(2, rect)], [(2, disk)])
        gmsh.model.occ.synchronize()

        surfs = [tag for _, tag in gmsh.model.getEntities(2)]
        if len(surfs) != 1:
            raise RuntimeError(f"expected exactly 1 surface after cut, got {surfs}")
        gmsh.model.addPhysicalGroup(2, surfs, name="PLATE_SURF")

        # Neutral geometric edge labels (NO BC meaning is assigned here).
        label_to_tags = _classify_plate_curves(a, b, xc, yc, r, tol)
        edge_phys = {}
        for lbl in PLATE_EDGE_LABELS:
            edge_phys[lbl] = gmsh.model.addPhysicalGroup(1, label_to_tags[lbl], name=lbl)

        # Hole refinement: Distance from the hole curves -> Threshold size.
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "CurvesList",
                                         [float(c) for c in label_to_tags["hole_boundary"]])
        gmsh.model.mesh.field.setNumber(dist, "Sampling", 200)
        thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thr, "InField", dist)
        gmsh.model.mesh.field.setNumber(thr, "SizeMin", h_hole)
        gmsh.model.mesh.field.setNumber(thr, "SizeMax", elem_size)
        gmsh.model.mesh.field.setNumber(thr, "DistMin", REFINE_DIST_MIN_R * r)
        gmsh.model.mesh.field.setNumber(thr, "DistMax", REFINE_DIST_MAX_R * r)
        gmsh.model.mesh.field.setAsBackgroundMesh(thr)
        # The background field is the ONLY size source (gmsh recipe, tutorial t10).
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        if elem_shape == "quad":
            gmsh.option.setNumber("Mesh.Algorithm", 8)   # Frontal-Delaunay for quads
            gmsh.model.mesh.setRecombine(2, surfs[0])    # quad-dominant (CPS6 leftovers ok)
        gmsh.option.setNumber("Mesh.ElementOrder", elem_order)
        if elem_order == 2:
            # Serendipity elements: 8-node quads / 6-node tris (no face nodes).
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
        gmsh.model.mesh.generate(2)

        # ---- gather acceptance data ----
        node_tags, ncoords, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)
        coords = np.asarray(ncoords, dtype=float).reshape(-1, 3)
        node_xyz = {int(tid): coords[k] for k, tid in enumerate(node_tags)}
        max_abs_z = float(np.max(np.abs(coords[:, 2]))) if n_nodes else float("nan")

        # Surface elements: counts + corner-polygon (chord) area sum.
        elem_counts: dict = {}
        surf_elem_tags: list = []
        mesh_area = 0.0
        etypes, etags, enodes = gmsh.model.mesh.getElements(2)
        for etype, tags, conn in zip(etypes, etags, enodes):
            etype = int(etype)
            if etype not in _CORNERS_2D:
                raise RuntimeError(f"unexpected surface element type {etype}")
            n_per = int(gmsh.model.mesh.getElementProperties(etype)[3])
            n_corner = _CORNERS_2D[etype]
            conn = [int(x) for x in conn]
            elem_counts[etype] = elem_counts.get(etype, 0) + len(tags)
            surf_elem_tags.extend(int(x) for x in tags)
            for e in range(len(tags)):
                nodes = conn[e * n_per:(e + 1) * n_per]
                mesh_area += _shoelace([node_xyz[nid] for nid in nodes[:n_corner]])
        ref_area = a * b - math.pi * r * r
        area_rel_err = abs(mesh_area - ref_area) / ref_area

        # Min element quality: minSICN <= 0 means an inverted element.
        qual = gmsh.model.mesh.getElementQualities(surf_elem_tags, "minSICN")
        min_quality = float(np.min(qual)) if len(qual) else float("nan")

        # Per-edge node counts + hole segment (line element) count.
        edge_node_counts = {}
        for lbl in PLATE_EDGE_LABELS:
            ntags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, edge_phys[lbl])
            edge_node_counts[lbl] = len(ntags)
        n_hole_segments = 0
        for ctag in label_to_tags["hole_boundary"]:
            _, ltags, _ = gmsh.model.mesh.getElements(1, ctag)
            n_hole_segments += sum(len(x) for x in ltags)

        # Write MSH 2.2 ASCII; SaveAll=0 -> only physical-group entities are
        # written (the surface elements AND the edge line elements M3/M4 need).
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(str(MSH_PATH))
    finally:
        gmsh.finalize()

    file_size = MSH_PATH.stat().st_size if MSH_PATH.exists() else 0
    return {
        "geometry": {"type": "plate_with_hole",
                     "a": a, "b": b, "t": t, "r": r, "xc": xc, "yc": yc},
        "mesh": {
            "elem_shape": elem_shape, "elem_order": elem_order,
            "elem_size": elem_size, "h_hole": h_hole,
            "min_ligament": min_ligament,
            "ligament_elems": min_ligament / h_hole,
            "dist_band": [REFINE_DIST_MIN_R * r, REFINE_DIST_MAX_R * r],
        },
        "n_nodes": n_nodes,
        "n_surf_elems": sum(elem_counts.values()),
        "elem_counts": elem_counts,
        "min_quality": min_quality,
        "edge_node_counts": edge_node_counts,
        "n_hole_segments": n_hole_segments,
        "mesh_area": mesh_area,
        "ref_area": ref_area,
        "area_rel_err": area_rel_err,
        "max_abs_z": max_abs_z,
        "z_tol": tol,
        "msh_path": str(MSH_PATH),
        "file_size": file_size,
    }


def _elem_type_str(elem_counts: dict) -> str:
    parts = []
    for etype, count in elem_counts.items():
        name, ccx = _GMSH_ELEM_TYPES.get(etype, (f"gmsh-type-{etype}", "?"))
        parts.append(f"{count} x {name} (Gmsh type {etype} -> CalculiX {ccx})")
    return "; ".join(parts) if parts else "(none)"


def report_and_check(s: dict) -> bool:
    """Print the acceptance report for the built mesh and return PASS/FAIL."""
    if s["geometry"]["type"] == "plate_with_hole":
        return _report_plate(s)
    return _report_beam(s)


def _report_plate(s: dict) -> bool:
    """Plate acceptance report (plate plan M2). Returns PASS (True)/FAIL."""
    g, m = s["geometry"], s["mesh"]
    enc = s["edge_node_counts"]
    print("=" * 64)
    print("mesh.py acceptance report  (plate_with_hole, plate plan M2)")
    print("=" * 64)
    print(f"geometry : plate_with_hole  a={g['a']} b={g['b']} t={g['t']}  "
          f"r={g['r']} at (xc, yc)=({g['xc']}, {g['yc']})  (SI-mm, z=0 plane)")
    print(f"mesh     : elem_shape={m['elem_shape']} elem_order={m['elem_order']}  "
          f"elem_size={m['elem_size']} far-field, h_hole={m['h_hole']:.3g} at the hole")
    print(f"           refine band: {m['dist_band'][0]:.3g} -> {m['dist_band'][1]:.3g} mm "
          f"from the hole edge (policy D7)")
    print(f"           thinnest ligament {m['min_ligament']:.2f} mm spans "
          f"{m['ligament_elems']:.1f} elements (policy >= {LIGAMENT_MIN_ELEMS})")
    print(f"elements : {_elem_type_str(s['elem_counts'])}")
    print(f"nodes    : {s['n_nodes']} total")
    print( "edges    : neutral geometric labels (each node count must be > 0)")
    for lbl in PLATE_EDGE_LABELS:
        print(f"           {lbl:<13}: {enc.get(lbl, 0)} nodes")
    print(f"hole     : {s['n_hole_segments']} boundary segments "
          f"(policy target >= {HOLE_MIN_SEGMENTS})")
    print(f"area     : mesh {s['mesh_area']:.2f} vs a*b - pi*r^2 = {s['ref_area']:.2f} mm^2 "
          f"-> rel err {s['area_rel_err'] * 100:.4f}% (< {PLATE_AREA_TOL * 100:.1f}%)")
    print(f"quality  : minSICN = {s['min_quality']:.4f}  (must be > 0)")
    print(f"planarity: max|z| = {s['max_abs_z']:.2e} mm (must be < {s['z_tol']:.1e})")
    print(f"output   : {s['msh_path']}  (MSH 2.2 ASCII, {s['file_size'] / 1024:.1f} KB)")

    checks = [
        ("msh non-empty", s["file_size"] > 0),
        ("surface elements present", s["n_surf_elems"] > 0),
        ("all 5 edge labels present",
         all(lbl in enc for lbl in PLATE_EDGE_LABELS)),
        ("every edge label has > 0 nodes",
         all(enc.get(lbl, 0) > 0 for lbl in PLATE_EDGE_LABELS)),
        (f"hole circumference >= {HOLE_MIN_SEGMENTS} segments",
         s["n_hole_segments"] >= HOLE_MIN_SEGMENTS),
        ("no negative Jacobian (minSICN > 0)", s["min_quality"] > 0.0),
        (f"mesh area matches plate minus hole (< {PLATE_AREA_TOL * 100:.1f}%)",
         s["area_rel_err"] < PLATE_AREA_TOL),
        ("all nodes in the z = 0 plane", s["max_abs_z"] < s["z_tol"]),
    ]
    passed = all(ok for _, ok in checks)
    print("-" * 64)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 64)
    print(f"ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    if passed:
        print("Human check (optional): gmsh out/beam.msh")
    print("=" * 64)
    return passed


def _report_beam(s: dict) -> bool:
    """Cantilever acceptance report (plan section 3.1). Returns PASS (True)/FAIL."""
    g, m = s["geometry"], s["mesh"]
    print("=" * 64)
    print("mesh.py acceptance report")
    print("=" * 64)
    print(f"geometry : {g['type']}  L={g['L']} h={g['h']} b={g['b']}  (SI-mm)")
    print( "           x=axis  y=b/width  z=h/height & load direction")
    print(f"mesh     : elem_shape={m['elem_shape']} elem_order={m['elem_order']} "
          f"elem_size={m['elem_size']}  ->  nx={m['nx']} ny={m['ny']} nz={m['nz']}")
    print(f"elements : {_elem_type_str(s['elem_counts'])}")
    print(f"nodes    : {s['n_nodes']} total")
    print( "faces    : neutral geometric labels (each node count must be > 0)")
    for label in FACE_LABELS:
        print(f"           {label:<12}: {s['face_node_counts'].get(label, 0)} nodes")
    print(f"quality  : minSICN = {s['min_quality']:.4f}  (must be > 0; <= 0 == negative Jacobian)")
    print(f"output   : {s['msh_path']}  (MSH 2.2 ASCII, {s['file_size'] / 1024:.1f} KB)")

    # Deterministic acceptance gates (plan section 3.1).
    checks = [
        ("msh non-empty", s["file_size"] > 0),
        ("volume elements present", s["n_vol_elems"] > 0),
        ("all 6 face labels present",
         all(lbl in s["face_node_counts"] for lbl in FACE_LABELS)),
        ("every face has > 0 nodes",
         all(s["face_node_counts"].get(lbl, 0) > 0 for lbl in FACE_LABELS)),
        ("no negative Jacobian (minSICN > 0)", s["min_quality"] > 0.0),
    ]
    passed = all(ok for _, ok in checks)
    print("-" * 64)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 64)
    print(f"ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    if passed:
        print("Human check (optional): gmsh out/beam.msh")
    print("=" * 64)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the active geometry's mesh (plan layer 3.1 / plate plan M2).")
    parser.add_argument("-v", "--verbose", action="store_true", help="show Gmsh log output")
    args = parser.parse_args()
    summary = build_mesh(PARAMS, verbose=args.verbose)
    return 0 if report_and_check(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
