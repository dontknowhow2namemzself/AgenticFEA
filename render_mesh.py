#!/usr/bin/env python3
"""Render the mesh straight from beam.inp to out/mesh.png for the dashboard's
Pre-Process artifact.

Reads the CalculiX deck (*NODE, *ELEMENT C3D20, *NSET FIXED/LOAD, *CLOAD) and
rebuilds the elements as linear hexahedra -- the 8 corner nodes of each C3D20, which
gives clean quad faces instead of the triangulated quadratic surface VTK would draw
from beam.vtu. Boundary conditions come from the same deck: the FIXED face as a
translucent wall, and the load as one arrow per loaded node (faithful to the nodal
*CLOAD) with the resultant in the label. Isometric + parallel projection, fixed end
to the left. PyVista off-screen -- same stack as postprocess.py, no extra dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
pv.OFF_SCREEN = True


def parse_inp(path):
    """Pull nodes, C3D20 connectivity, named node sets and nodal loads from a deck."""
    nodes, etok, nsets, cload = {}, [], {}, []
    etype, mode, cur = "C3D20", None, None
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("**"):
            continue
        if s.startswith("*"):
            kw = s.split(",")[0].strip().upper()
            up = s.upper()
            if kw == "*NODE":
                mode = "node"
            elif kw == "*ELEMENT":
                mode = "elem"
                if "TYPE=" in up:
                    etype = up.split("TYPE=")[1].split(",")[0].strip()
            elif kw == "*NSET":
                mode, cur = "nset", up.split("NSET=")[1].split(",")[0].strip()
                nsets[cur] = []
            elif kw == "*CLOAD":
                mode = "cload"
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
    elems = [etok[i:i + 21] for i in range(0, len(etok), 21)]   # eid + 20 nodes per C3D20
    return nodes, elems, nsets, cload, etype


def render_mesh(inp=OUT / "beam.inp", png=OUT / "mesh.png") -> Path:
    nodes, elems, nsets, cload, etype = parse_inp(inp)
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
    print("wrote", render_mesh())
    sys.exit(0)
