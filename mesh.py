#!/usr/bin/env python3
"""Pre-processor layer 3.1: geometry + mesh generation (plan section 3.1).

Builds the cantilever-beam solid with the Gmsh OpenCASCADE kernel, tags the
volume as BEAM_VOL and every outer face with a NEUTRAL geometric label
(face_x_min ... face_z_max), generates a second-order structured hexahedral
mesh, and writes out/beam.msh in MSH 2.2 ASCII.

Scope boundary (do not cross): this module describes ONLY what the geometry
looks like and where the faces are. It carries NO boundary-condition meaning.
Mapping geometric labels to FIXED / LOAD is the job of boundary_conditions.py.

Naming contract (shared across layers, do not change):
    volume physical group : BEAM_VOL
    face physical groups   : face_x_min / face_x_max / face_y_min /
                             face_y_max / face_z_min / face_z_max

Unit system: SI-mm (see params.py).
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
}


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
    """Build the mesh, write out/beam.msh, and return an acceptance summary."""
    geom_type = params["geometry"]["type"]
    if geom_type != "cantilever_beam":
        raise NotImplementedError(f"geometry.type={geom_type!r} is not supported in the MVP")
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
        "geometry": {"type": geom_type, "L": L, "h": h, "b": b},
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


def _elem_type_str(elem_counts: dict) -> str:
    parts = []
    for etype, count in elem_counts.items():
        name, ccx = _GMSH_ELEM_TYPES.get(etype, (f"gmsh-type-{etype}", "?"))
        parts.append(f"{count} x {name} (Gmsh type {etype} -> CalculiX {ccx})")
    return "; ".join(parts) if parts else "(none)"


def report_and_check(s: dict) -> bool:
    """Print the acceptance report (plan section 3.1) and return PASS (True)/FAIL."""
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
    parser = argparse.ArgumentParser(description="Build cantilever-beam mesh (plan layer 3.1).")
    parser.add_argument("-v", "--verbose", action="store_true", help="show Gmsh log output")
    args = parser.parse_args()
    summary = build_mesh(PARAMS, verbose=args.verbose)
    return 0 if report_and_check(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
