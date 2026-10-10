#!/usr/bin/env python3
"""Render the mesh straight from beam.inp to out/mesh.png for the dashboard's
Pre-Process artifact.

Reads the CalculiX deck (*NODE, *ELEMENT, *NSET FIXED/LOAD, *BOUNDARY, *CLOAD)
and rebuilds the elements from their CORNER nodes only, which gives clean
straight-edged cells instead of the triangulated quadratic surface VTK would
draw from beam.vtu. Boundary conditions come from the same deck, so the picture
always shows what was actually solved.

  3D decks (C3D20R)  : isometric hex view; the FIXED face as a translucent wall
                       and one arrow per loaded node, fixed end to the left.
  2D decks (CPS8R /   : top (x-y) view of the plate; roller symbols on the
  CPS6, plate plan M9)  DOF-1 constrained edge, a distinct marker on the single
                       anchor node, and load arrows along the loaded edge. This
                       is the picture that shows the roller + anchor support
                       model (plate plan D2) at a glance.

PyVista off-screen -- same stack as postprocess.py, no extra dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from paths import RunPaths, add_workdir_argument, from_args

pv.OFF_SCREEN = True

# CalculiX element type -> (nodes per element, corner count, VTK cell type).
# Only corners are drawn, so the midside nodes are read and discarded.
ELEM_KINDS = {
    "C3D20": (20, 8, pv.CellType.HEXAHEDRON),
    "C3D20R": (20, 8, pv.CellType.HEXAHEDRON),
    "CPS8": (8, 4, pv.CellType.QUAD),
    "CPS8R": (8, 4, pv.CellType.QUAD),
    "CPS6": (6, 3, pv.CellType.TRIANGLE),
}


def parse_inp(path):
    """Pull nodes, per-type element blocks, node sets, constraints and nodal
    loads from a deck. Returns (nodes, blocks, nsets, cload, boundary) where
    blocks is {elem_type: [[eid, n1, n2, ...], ...]}."""
    nodes, nsets, cload, boundary = {}, {}, [], []
    blocks: dict = {}
    etok: list = []
    etype, mode, cur = None, None, None

    def flush():
        """Slice the accumulated token stream of one *ELEMENT block."""
        if etype and etok:
            stride = ELEM_KINDS[etype][0] + 1        # eid + connectivity
            blocks.setdefault(etype, []).extend(
                etok[i:i + stride] for i in range(0, len(etok), stride))

    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("**"):
            continue
        if s.startswith("*"):
            kw = s.split(",")[0].strip().upper()
            up = s.upper()
            if mode == "elem":
                flush()
                etok = []
            if kw == "*NODE":
                mode = "node"
            elif kw == "*ELEMENT":
                mode = "elem"
                etype = up.split("TYPE=")[1].split(",")[0].strip() if "TYPE=" in up else None
                if etype not in ELEM_KINDS:
                    raise ValueError(f"unsupported element type {etype!r} in {path}")
            elif kw == "*NSET":
                mode, cur = "nset", up.split("NSET=")[1].split(",")[0].strip()
                nsets[cur] = []
            elif kw == "*CLOAD":
                mode = "cload"
            elif kw == "*BOUNDARY":
                mode = "boundary"
            else:
                mode = None
            continue
        if mode == "node":
            p = [t for t in s.split(",") if t.strip()]
            nodes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
        elif mode == "elem":
            etok += [int(t) for t in s.split(",") if t.strip()]
        elif mode == "nset":
            nsets[cur] += [int(t) for t in s.split(",") if t.strip()]
        elif mode == "cload":
            p = [t for t in s.split(",") if t.strip()]
            cload.append((int(p[0]), int(p[1]), float(p[2])))
        elif mode == "boundary":
            p = [t for t in s.split(",") if t.strip()]
            if len(p) >= 2 and p[0].lstrip("-").isdigit():
                lo = int(p[1])
                hi = int(p[2]) if len(p) > 2 else lo
                boundary += [(int(p[0]), d) for d in range(lo, hi + 1)]
    if mode == "elem":
        flush()
    return nodes, blocks, nsets, cload, boundary


def _grid_from_blocks(nodes, blocks):
    """Corner-only PyVista grid spanning every element block in the deck."""
    ids = sorted(nodes)
    idx = {n: i for i, n in enumerate(ids)}
    pts = np.array([nodes[n] for n in ids], float)
    cells, ctypes = [], []
    for etype, elems in blocks.items():
        _n_per, n_corner, vtk_type = ELEM_KINDS[etype]
        for e in elems:
            cells += [n_corner] + [idx[n] for n in e[1:1 + n_corner]]
            ctypes.append(vtk_type)
    return pv.UnstructuredGrid(np.array(cells, np.int64),
                               np.array(ctypes, np.uint8), pts)


def _render_plate(nodes, blocks, nsets, cload, boundary, png) -> Path:
    """Top view of a 2D plate deck with its roller / anchor / load symbols."""
    grid = _grid_from_blocks(nodes, blocks)
    n_elem = sum(len(v) for v in blocks.values())
    etype_txt = " + ".join(f"{len(v)}x{k}" for k, v in sorted(blocks.items()))
    xmn, xmx, ymn, ymx = grid.bounds[0:4]
    span = max(xmx - xmn, ymx - ymn)

    # Canvas is deliberately taller than the part's 2:1 aspect: the plate fills
    # the width and leaves dark bands top and bottom for the title and legend,
    # so no text ends up sitting on the mesh.
    pl = pv.Plotter(off_screen=True, window_size=(1200, 800))
    pl.background_color = "#11161f"
    pl.enable_parallel_projection()
    pl.add_mesh(grid, color="#9fb2c8", show_edges=True, edge_color="#56708c", line_width=1)

    # Constraints, split by DOF: DOF 1 == the roller edge, DOF 2 == the anchor.
    # Legend entries are pixel-positioned text, not 3D point labels: a tight
    # camera on a wide part clips anything placed outside the plate outline.
    by_dof: dict = {}
    for nid, dof in boundary:
        by_dof.setdefault(dof, []).append(nid)
    legend = []
    roller = [n for n in by_dof.get(1, []) if n in nodes]
    if roller:
        pts = np.array([nodes[n] for n in roller])
        pl.add_mesh(pv.PolyData(pts).glyph(geom=pv.Sphere(radius=span * 0.006),
                                           scale=False, orient=False),
                    color="#3fb1d0")
        legend.append((f"ROLLER   u_x = 0  ({len(roller)} nodes)", "#3fb1d0"))
    anchor = [n for n in by_dof.get(2, []) if n in nodes]
    if anchor:
        # Nudged just outside the edge: the anchor node is also a roller node,
        # so drawn in place its marker would sit under the roller glyph.
        pts = np.array([nodes[n] for n in anchor]) - np.array([span * 0.012, 0, 0])
        pl.add_mesh(pv.PolyData(pts).glyph(geom=pv.Sphere(radius=span * 0.014),
                                           scale=False, orient=False),
                    color="#e05c5c")
        legend.append((f"ANCHOR   u_y = 0  ({len(anchor)} node, mid-edge)", "#e05c5c"))
    # Loads: one arrow per loaded node, tip landing on the node.
    if cload:
        fx: dict = {}
        for nid, dof, val in cload:
            if dof == 1:
                fx[nid] = fx.get(nid, 0.0) + val
        pts = np.array([nodes[n] for n in sorted(fx) if n in nodes])
        if len(pts):
            total = sum(fx.values())
            d = np.array([1.0 if total > 0 else -1.0, 0.0, 0.0])
            alen = span * 0.075
            cloud = pv.PolyData(pts - d * alen)
            cloud["v"] = np.tile(d, (len(pts), 1))
            slim = pv.Arrow(tip_length=0.3, tip_radius=0.09, shaft_radius=0.028)
            pl.add_mesh(cloud.glyph(orient="v", scale=False, factor=alen, geom=slim),
                        color="#e8a13a")
            legend.append((f"LOAD     {len(pts)} consistent nodal forces, "
                           f"total {round(total)} N (+X)", "#e8a13a"))

    pl.add_text(f"{etype_txt} plane stress · {n_elem} elem · SI-mm",
                position=(14, 766), font_size=11, color="#8893a3")
    for i, (text, colour) in enumerate(legend):
        pl.add_text(text, position=(14, 22 + 24 * (len(legend) - 1 - i)),
                    font_size=11, color=colour)
    pl.view_xy()
    # adjust_render_window=False: tight() otherwise RESIZES the canvas to the
    # part's aspect ratio, which would swallow the bands reserved for the text.
    pl.camera.tight(padding=0.10, adjust_render_window=False)
    pl.screenshot(str(png))
    pl.close()
    return Path(png)


def render_mesh(paths: RunPaths = None) -> Path:
    paths = paths or RunPaths.default()
    inp, png = paths.inp, paths.mesh_png
    nodes, blocks, nsets, cload, boundary = parse_inp(inp)
    if any(k.startswith("CPS") for k in blocks):
        return _render_plate(nodes, blocks, nsets, cload, boundary, png)
    etype = next(iter(blocks))
    elems = blocks[etype]
    ids = sorted(nodes)
    idx = {n: i for i, n in enumerate(ids)}
    pts = np.array([nodes[n] for n in ids], float)
    # C3D20 -> linear hex: the first 8 nodes are the corners, ordered like VTK_HEXAHEDRON
    cells = np.hstack([[8, *[idx[n] for n in e[1:9]]] for e in elems]).astype(np.int64)
    grid = pv.UnstructuredGrid(cells, np.full(len(elems), pv.CellType.HEXAHEDRON, np.uint8), pts)
    ymn, ymx, zmn, zmx = grid.bounds[2:6]

    pl = pv.Plotter(off_screen=True, window_size=(1200, 600))
    pl.background_color = "#11161f"          # dashboard panel colour (control-room dark)
    pl.enable_parallel_projection()
    pl.add_mesh(grid, color="#9fb2c8", show_edges=True, edge_color="#56708c", line_width=1)

    lab_xyz, lab_txt = [], []
    # FIXED: a translucent wall over the constrained cross-section
    if nsets.get("FIXED"):
        fc = np.array([nodes[n] for n in nsets["FIXED"]]).mean(0)
        wall = pv.Plane(center=(fc[0] - 0.3, (ymn + ymx) / 2, (zmn + zmx) / 2), direction=(1, 0, 0),
                        i_size=(zmx - zmn) * 1.25, j_size=(ymx - ymn) * 1.25)
        pl.add_mesh(wall, color="#3fb1d0", opacity=0.5)
        lab_xyz.append(fc + [0, 0, 6.5])
        lab_txt.append("FIXED  (encastre, u=0)")
    # LOAD: one arrow per loaded node (faithful to the nodal *CLOAD); tip lands on the node
    if nsets.get("LOAD") and cload:
        load_pts = np.array([nodes[n] for n in nsets["LOAD"]])
        dof = cload[0][1]
        sign = -1.0 if cload[0][2] < 0 else 1.0
        d = np.zeros(3)
        d[dof - 1] = sign
        total = sum(abs(c[2]) for c in cload)
        # one slim arrow per loaded node, tip landing on the node; thin glyph keeps
        # the 96-arrow field legible even on a short, head-on beam
        arrow_len = 4.0
        cloud = pv.PolyData(load_pts - d * arrow_len)
        cloud["v"] = np.tile(d, (len(load_pts), 1))
        slim = pv.Arrow(tip_length=0.28, tip_radius=0.08, shaft_radius=0.025)
        pl.add_mesh(cloud.glyph(orient="v", scale=False, factor=arrow_len, geom=slim),
                    color="#e8a13a")
        lc = load_pts.mean(0)
        axis = f"{'+' if sign > 0 else '-'}{'XYZ'[dof - 1]}"
        lab_xyz.append(lc + [0, 0, 12])
        lab_txt.append(f"LOAD  {len(load_pts)} nodal forces · total {round(total)} N ({axis})")

    if lab_xyz:
        pl.add_point_labels(np.array(lab_xyz), lab_txt, font_size=15, text_color="#dbe1e8",
                            show_points=False, shape=None, always_visible=True)
    pl.add_text(f"{etype} hex · {len(elems)} elem · SI-mm", position="upper_left",
                font_size=11, color="#8893a3")
    pl.view_isometric()
    pl.camera.Azimuth(-90)            # fixed end to the left -- conventional cantilever layout
    pl.reset_camera()
    pl.camera.zoom(1.22)
    pl.add_axes()
    pl.screenshot(str(png))
    pl.close()
    return Path(png)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render the annotated mesh figure from the deck.")
    add_workdir_argument(ap)
    print("wrote", render_mesh(from_args(ap.parse_args())))
    sys.exit(0)
