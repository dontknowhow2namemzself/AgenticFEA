#!/usr/bin/env python3
"""How a run's results are drawn: the interactive 3D view and the fixed PNG views.

Two halves, kept apart on purpose:

  build_payload(mesh) -- pipeline side (layer 6). Reads the solver result once
                         and keeps only what a viewer needs: the outer surface,
                         its nodal results and the element edges. Saved as
                         <run>/viewer.npz, a few hundred KB.
  figure(payload)     -- dashboard side. Builds a Plotly figure from that file
                         alone, so showing a run needs neither PyVista nor the
                         solver output, and a replayed demo carries its own.
                         The field and the deformed shape are switched by menus
                         inside the figure, so turning to a view and then
                         changing the field keeps the view.
  render_png(payload) -- pipeline side again: the fixed views (PNG) drawn off
                         screen by PyVista FROM THE SAME PAYLOAD, with the same
                         edges, colours, range and bar labels. The 3D view is
                         for looking around; the PNGs are the views worth
                         keeping, and a report can use them as they are.

Each half imports its heavy library (pyvista / plotly) only when called, so the
dashboard never loads the solver-side stack just to draw a figure.

What is drawn
  * Surface: the boundary faces of the mesh (for the plate, every element). A
    browser can only shade triangles, so each face is split -- an 8-node quad
    into 6 triangles that use all 8 nodes, a 6-node triangle into 4 -- and the
    colour is interpolated between the real nodal values, midside nodes
    included.
  * Edges: the ELEMENT edges, never the triangles'. A quadratic edge is drawn
    through its midside node as the parabola the element itself interpolates,
    so an edge that is curved in the element (around the hole) is curved here.
  * Colour: a rainbow fringe, the convention of CAE post-processors. Each
    field runs from its own min to its own max, and the bar is labelled at
    even steps with the exact min and max at its ends -- the same numbers on
    the 3D view and on every PNG of that field.
  * Lighting is nearly flat, so a colour reads as the value it stands for
    rather than as the value plus a shadow.

Unit system: SI-mm (coordinates and displacement mm, stress MPa).

Usage:  python viewer.py [--workdir DIR]   -> <run>/viewer.npz + viewer.html
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import numpy as np

from paths import RunPaths, add_workdir_argument, from_args

# ---- element tables (VTK numbering, as ccx2paraview writes the .vtu) --------
# Each cell type maps to its faces as (corners, midside nodes or None). For a
# 2D cell the cell is its own single face. Face winding is fixed up from the
# geometry afterwards, so these tables only have to be right about which nodes
# belong together, not about orientation.
_HEX_FACES = ((0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7))
_HEX20_EDGE_MID = {(0, 1): 8, (1, 2): 9, (2, 3): 10, (3, 0): 11, (4, 5): 12, (5, 6): 13,
                   (6, 7): 14, (7, 4): 15, (0, 4): 16, (1, 5): 17, (2, 6): 18, (3, 7): 19}
_HEX20_EDGE_MID.update({(b, a): m for (a, b), m in list(_HEX20_EDGE_MID.items())})


def _ring_mids(corners, edge_mid):
    return tuple(edge_mid[(corners[i], corners[(i + 1) % len(corners)])]
                 for i in range(len(corners)))


VTK_CELLS = {
    # type: (name, spatial dim, ((corners, mids), ...))
    12: ("hexahedron", 3, tuple((f, None) for f in _HEX_FACES)),
    25: ("quadratic hexahedron", 3,
         tuple((f, _ring_mids(f, _HEX20_EDGE_MID)) for f in _HEX_FACES)),
    9: ("quad", 2, (((0, 1, 2, 3), None),)),
    23: ("quadratic quad", 2, (((0, 1, 2, 3), (4, 5, 6, 7)),)),
    5: ("triangle", 2, (((0, 1, 2), None),)),
    22: ("quadratic triangle", 2, (((0, 1, 2), (3, 4, 5)),)),
}

# Triangles of one face, in local slots: corners are 0..n-1, midside nodes
# n..2n-1 (the midside node of edge i sits between corners i and i+1).
_TRIANGLES = {
    (4, False): ((0, 1, 2), (0, 2, 3)),
    (4, True): ((0, 4, 7), (4, 1, 5), (5, 2, 6), (6, 3, 7), (4, 5, 6), (4, 6, 7)),
    (3, False): ((0, 1, 2),),
    (3, True): ((0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)),
}

# ---- fields ---------------------------------------------------------------
# key -> (label, unit, kind). A "signed" field gets both its extremes labelled
# (where tension and compression peak), a magnitude only its maximum. S is
# stored in ccx2paraview's component order xx, yy, zz, xy, yz, xz.
FIELDS = {
    "vm": ("von Mises", "MPa", "magnitude"),
    "u": ("U magnitude", "mm", "magnitude"),    # not "|U|": VTK hands any "|" to another text engine
    "sxx": ("σxx", "MPa", "signed"),
    "syy": ("σyy", "MPa", "signed"),
    "szz": ("σzz", "MPa", "signed"),
    "sxy": ("σxy", "MPa", "signed"),
    "syz": ("σyz", "MPa", "signed"),
    "sxz": ("σxz", "MPa", "signed"),
}
S_COMPONENT = {"sxx": 0, "syy": 1, "szz": 2, "sxy": 3, "syz": 4, "sxz": 5}
PLANE_FIELDS = ("vm", "u", "sxx", "syy", "sxy")     # plane stress: the rest are zero

# ---- colours: the dashboard's dark panel ----------------------------------
PANEL = "#11161f"      # = --panel in app.py
TEXT = "#c9d1d9"       # = --text
MUTED = "#6e7681"      # = --muted
GRID = "#232a35"       # = --line
# Dark edges: light ones would lighten every element they cross and make it
# read as a higher value.
EDGE_RGB, EDGE_ALPHA = "#0a0e14", 0.55
EDGE = f"rgba(10, 14, 20, {EDGE_ALPHA})"
# Rainbow fringe: matplotlib's "turbo" -- jet's blue-to-red hue sequence with a
# smooth lightness ramp, so it reads like the CAE tools' contour plots without
# jet's false bright bands at cyan and yellow. One list of stops feeds both
# renderers, so a colour means the same value in the 3D view and in the PNGs.
RAINBOW = ["#30123b", "#4040a2", "#466be3", "#4294ff", "#28bceb", "#18ddc2", "#32f298",
           "#6dfe62", "#a4fc3c", "#cdec34", "#eecf3a", "#fdac34", "#fb7e21", "#eb500e",
           "#d02f05", "#a91601", "#7a0403"]
N_TICKS = 5           # fringe-bar labels, the exact min and max at the two ends
TICK_FMT = "%.4g"     # printf style: PyVista takes it as is, Plotly gets it applied

# Named cameras, shared by the 3D view's opening camera and the PNGs: direction
# from the model's centre towards the eye, the up vector, parallel projection.
VIEWS = {
    "iso": ((-1.25, -0.9, 0.6), (0, 0, 1), False),   # from the clamped end, along the length
    "top": ((0, 0, 1), (0, 1, 0), True),
    "side": ((0, -1, 0), (0, 0, 1), True),
    "end": ((-1, 0, 0), (0, 0, 1), True),             # the clamped end's face
}

MENU_BG = "#8b949e"     # in-figure menus; see figure() for why they are light

# The toolbar's "default camera" is Plotly's generic view, not this figure's
# framing; "reset camera to last save" (kept) returns to the framing.
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["resetCameraDefault3d"],
                 "toImageButtonOptions": {"format": "png", "filename": "agentic_fea_view",
                                          "scale": 2}}
EDGE_SAMPLES = 7       # points per quadratic edge; a parabola needs few
EDGE_LIFT = 0.1        # edges sit this fraction of an element edge off the surface
CURVED_EDGE = 0.005    # midside node off the chord by > 0.5% of the edge: draw it curved
WARP_FRACTION = 0.05   # deformed view: peak |U| drawn at 5% of the model diagonal


# ============================================================================
# pipeline side
# ============================================================================
def _faces(mesh) -> list:
    """Every boundary face as (corners (n,k), mids (n,k) or None, cell centroids (n,3)).

    For 3D cells a face is on the boundary when no other cell shares it, i.e.
    its corner set occurs exactly once. For 2D cells every cell is a face.
    """
    pts = np.asarray(mesh.points, dtype=float)
    groups = {}   # (k corners, quadratic) -> lists
    for ctype, conn in mesh.cells_dict.items():
        if ctype not in VTK_CELLS:
            names = ", ".join(f"{t} ({v[0]})" for t, v in VTK_CELLS.items())
            raise ValueError(f"VTK cell type {ctype} is not supported by the viewer; "
                             f"supported: {names}")
        _, dim, faces = VTK_CELLS[ctype]
        conn = np.asarray(conn)
        centroid = pts[conn].mean(axis=1)
        for corners, mids in faces:
            key = (len(corners), mids is not None)
            g = groups.setdefault(key, {"c": [], "m": [], "cen": [], "dim": dim})
            g["c"].append(conn[:, corners])
            g["m"].append(conn[:, mids] if mids is not None else None)
            g["cen"].append(centroid)

    out = []
    for (k, quadratic), g in groups.items():
        c = np.concatenate(g["c"])
        m = np.concatenate(g["m"]) if quadratic else None
        cen = np.concatenate(g["cen"])
        if g["dim"] == 3:
            _, inverse, counts = np.unique(np.sort(c, axis=1), axis=0,
                                           return_inverse=True, return_counts=True)
            keep = counts[inverse.ravel()] == 1
            c, cen = c[keep], cen[keep]
            m = m[keep] if quadratic else None
        out.append((c, m, cen, g["dim"]))
    return out


def _orient(pts, c, m, cen, dim):
    """Wind every face counter-clockwise seen from outside (2D: from +z).

    Shading uses the triangle normals, and a face wound the wrong way would be
    lit as if seen from behind. The VTK tables are not trusted for this; the
    geometry decides.
    """
    p = pts[c]
    k = c.shape[1]
    # Face normal from the two diagonals (k=4) or two sides (k=3).
    a = p[:, 2] - p[:, 0]
    b = (p[:, 3] - p[:, 1]) if k == 4 else (p[:, 1] - p[:, 0])
    n = np.cross(a, b) if k == 4 else np.cross(b, a)
    outward = (p.mean(axis=1) - cen) if dim == 3 else np.array([0.0, 0.0, 1.0])
    flip = np.einsum("ij,ij->i", n, np.broadcast_to(outward, n.shape)) < 0
    if flip.any():
        # Reverse the ring, keeping corner 0: corners c0,c3,c2,c1 and their
        # edge midside nodes m30,m23,m12,m01.
        c = c.copy()
        c[flip] = np.concatenate([c[flip][:, :1], c[flip][:, :0:-1]], axis=1)
        if m is not None:
            m = m.copy()
            m[flip] = m[flip][:, ::-1]
    return c, m


def build_payload(mesh) -> dict:
    """The viewer's data from a solved mesh (a pyvista UnstructuredGrid of the .vtu)."""
    pts = np.asarray(mesh.points, dtype=float)
    tris, edges, dims, n_faces = [], [], set(), 0
    for c, m, cen, dim in _faces(mesh):
        dims.add(dim)
        n_faces += len(c)
        c, m = _orient(pts, c, m, cen, dim)
        k = c.shape[1]
        slots = np.concatenate([c, m], axis=1) if m is not None else c
        tris.append(np.concatenate([slots[:, list(t)] for t in _TRIANGLES[(k, m is not None)]]))
        for i in range(k):
            a, b = c[:, i], c[:, (i + 1) % k]
            mid = m[:, i] if m is not None else np.full_like(a, -1)
            edges.append(np.stack([a, mid, b], axis=1))
    tri = np.concatenate(tris)
    edge = np.concatenate(edges)
    # An edge between two boundary faces appears twice; draw it once.
    _, first = np.unique(np.sort(edge[:, [0, 2]], axis=1), axis=0, return_index=True)
    edge = edge[np.sort(first)]

    # Keep only the nodes the surface uses, renumbered compactly.
    used = np.unique(np.concatenate([tri.ravel(), edge[:, [0, 2]].ravel(), edge[edge[:, 1] >= 0, 1]]))
    renum = np.full(len(pts), -1, dtype=np.int64)
    renum[used] = np.arange(len(used))
    edge = np.where(edge >= 0, renum[np.maximum(edge, 0)], -1)

    pd = mesh.point_data
    return {
        "xyz": pts[used].astype(np.float32),
        "U": np.asarray(pd["U"])[used].astype(np.float32),
        "S": np.asarray(pd["S"])[used].astype(np.float32),
        "vm": np.asarray(pd["S_Mises"]).ravel()[used].astype(np.float32),
        "tri": renum[tri].astype(np.int32),
        "edges": edge.astype(np.int32),
        "dim": np.int32(max(dims)),
        "n_faces": np.int32(n_faces),      # element faces on the surface (2D: the elements)
    }


def save_payload(mesh, npz_path: Path) -> dict:
    """Build and write <run>/viewer.npz; returns a summary for the stage record."""
    p = build_payload(mesh)
    np.savez_compressed(npz_path, **p)
    return {"path": str(npz_path), "n_nodes": int(len(p["xyz"])), "n_faces": int(p["n_faces"]),
            "n_triangles": int(len(p["tri"])), "n_edges": int(len(p["edges"])),
            "kbytes": Path(npz_path).stat().st_size / 1024}


# ============================================================================
# dashboard side
# ============================================================================
def load_payload(npz_path) -> dict:
    with np.load(npz_path) as z:
        return {k: z[k] for k in z.files}


def fields_for(payload: dict) -> tuple:
    """Field keys worth offering: a plane-stress model has no out-of-plane stress."""
    return PLANE_FIELDS if int(payload["dim"]) == 2 else tuple(FIELDS)


def field_label(key: str) -> str:
    label, unit, _ = FIELDS[key]
    return f"{label} [{unit}]"


def field_values(payload: dict, key: str) -> np.ndarray:
    if key == "vm":
        return payload["vm"].astype(float)
    if key == "u":
        return np.linalg.norm(payload["U"].astype(float), axis=1)
    return payload["S"][:, S_COMPONENT[key]].astype(float)


def auto_warp_factor(payload: dict) -> float:
    """Exaggeration that draws the peak displacement at 5% of the model diagonal,
    but never below 1.

    A fixed factor cannot serve a 0.01 mm plate stretch and a 2 mm cantilever
    tip at once, so the factor follows the result: the 100 x 10 x 10 demo beam
    (diagonal 101 mm, peak 1.91 mm) gets 0.05 x 101 / 1.91 = x2.64. A
    deformation already larger than that is drawn at true scale (x1) rather
    than shrunk -- shrinking it would hide exactly what is worth seeing.
    """
    xyz = payload["xyz"].astype(float)
    diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    peak = float(np.linalg.norm(payload["U"].astype(float), axis=1).max())
    return max(1.0, WARP_FRACTION * diag / peak) if peak > 0 and diag > 0 else 1.0


def _b64(a) -> dict:
    """A float array in Plotly's binary form, {dtype, bdata}.

    plotly.py writes trace data this way but arrays inside menu arguments as
    JSON text, about four times larger; plotly.js decodes this form wherever it
    reads trace data, menus included.
    """
    a = np.ascontiguousarray(a, dtype="<f4")
    return {"dtype": "f4", "bdata": base64.b64encode(a.tobytes()).decode("ascii")}


def _edge_polylines(configs: list, edges: np.ndarray, tri: np.ndarray) -> list:
    """Element edges as polylines, one array per configuration, NaN rows between edges.

    A quadratic edge whose midside node sits off its chord (around the hole) is
    sampled along the parabola through its three nodes -- the element's own
    shape functions on that edge -- so it is drawn curved, not as two chords. A
    straight one is a single segment: the same picture with a fraction of the
    points. An edge curved in ANY of the configurations is sampled in all of
    them, so they share one layout. Curvature is judged on the true geometry.

    Every edge is then lifted a hair off the surface (EDGE_LIFT of an element
    edge), because a line lying exactly in it flickers in and out of it -- the
    depth test cannot tell which is in front. The lift is ONE vector per edge,
    a rigid shift. Moving each node along its own normal instead bends an edge
    wherever the normals differ -- at the part's own edges and corners, where
    they turn -- and draws curves that are not in the mesh; scaling the lift
    with the whole model instead of the elements made that a 2 mm bend on a
    1000 mm beam's 10 mm section.
    """
    quad = np.flatnonzero(edges[:, 1] >= 0)
    curved = np.zeros(len(edges), dtype=bool)
    for p in configs:
        a, m, b = p[edges[quad, 0]], p[edges[quad, 1]], p[edges[quad, 2]]
        sag = np.linalg.norm(m - 0.5 * (a + b), axis=1)
        curved[quad] |= sag > CURVED_EDGE * np.linalg.norm(b - a, axis=1)
    ref = configs[0]
    lift = EDGE_LIFT * float(np.median(np.linalg.norm(ref[edges[:, 2]] - ref[edges[:, 0]], axis=1)))
    s = np.linspace(-1.0, 1.0, EDGE_SAMPLES)[None, :, None]
    out = []
    for p in configs:
        n = _vertex_normals(p, tri)
        shift = n[edges[:, 0]] + n[edges[:, 2]] + np.where(edges[:, 1:2] >= 0, n[np.maximum(edges[:, 1], 0)], 0.0)
        shift *= lift / np.maximum(np.linalg.norm(shift, axis=1, keepdims=True), 1e-12)
        parts = []
        if curved.any():
            e, d = edges[curved], shift[curved][:, None]
            a, m, b = p[e[:, 0]][:, None], p[e[:, 1]][:, None], p[e[:, 2]][:, None]
            curve = 0.5 * s * (s - 1) * a + (1 - s ** 2) * m + 0.5 * s * (s + 1) * b + d
            parts.append(np.concatenate([curve, np.full((len(e), 1, 3), np.nan)], axis=1).reshape(-1, 3))
        if (~curved).any():
            e, d = edges[~curved], shift[~curved]
            seg = np.stack([p[e[:, 0]] + d, p[e[:, 2]] + d, np.full((len(e), 3), np.nan)], axis=1)
            parts.append(seg.reshape(-1, 3))
        out.append(np.concatenate(parts))
    return out


def _vertex_normals(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Unit normal at each node: the area-weighted mean of its triangles' normals."""
    n = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
    vn = np.zeros_like(pts)
    for k in range(3):
        np.add.at(vn, tri[:, k], n)
    length = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.where(length > 0, length, 1.0)


def _axis(title: str, span=None) -> dict:
    axis = dict(title=dict(text=title, font=dict(color=MUTED, size=11)), color=MUTED,
                gridcolor=GRID, zeroline=False, showbackground=False, showspikes=False,
                tickfont=dict(size=10))
    if span is not None:
        axis.update(range=[float(span[0]), float(span[1])], autorange=False)
    return axis


def fringe(val: np.ndarray) -> dict:
    """Colour range and bar labels of one field, the same for the 3D view and the PNGs.

    The range is the field's own min to max over the drawn surface, labelled at
    N_TICKS even steps with the exact min and max at the ends -- so the bar of
    every view of a field carries the same numbers.
    """
    lo, hi = float(val.min()), float(val.max())
    # Round-off at a fixed support (|U| ~ 1e-21) is zero: label it 0, not 1.9e-21.
    span = max(abs(lo), abs(hi))
    lo, hi = (0.0 if abs(lo) < 1e-9 * span else lo), (0.0 if abs(hi) < 1e-9 * span else hi)
    if hi - lo <= 1e-12 * max(1.0, abs(hi)):          # a constant field still needs a range
        lo, hi = lo - 1.0, hi + 1.0
    ticks = np.linspace(lo, hi, N_TICKS)
    return {"cmin": lo, "cmax": hi, "tickvals": ticks.tolist(),
            "ticktext": [TICK_FMT % v for v in ticks]}


def _screen_axes(view: str) -> tuple:
    """(right, up) unit vectors of the screen for a named view."""
    direction, up, _ = VIEWS[view]
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    u = np.asarray(up, float) - np.dot(up, d) * d
    u /= np.linalg.norm(u)
    return np.cross(-d, u), u


def extremes(payload: dict, key: str, val: np.ndarray = None, view: str = "iso") -> list:
    """The labelled extreme(s) of a field: (name, node, value, text position).

    A magnitude gets its peak, a signed component both its peaks. Max sits
    above its point and min below, so mirrored pairs (the plate's shear lobes)
    never share a line. The text reads away from the nearer side edge of the
    view -- rightward from a peak at the clamped end, leftward from one at the
    free tip -- so it is never cut off by the frame.
    """
    val = field_values(payload, key) if val is None else val
    xyz = payload["xyz"].astype(float)
    s = (xyz - 0.5 * (xyz.min(axis=0) + xyz.max(axis=0))) @ _screen_axes(view)[0]
    s = s / max(float(np.abs(s).max()), 1e-12)            # -1 left edge .. +1 right edge
    i_max = int(np.argmax(val))
    nodes = [("max", i_max)]
    if FIELDS[key][2] == "signed":
        nodes.append(("min", int(np.argmin(val))))
    side = {i: "left" if s[i] > 0.3 else "right" for _, i in nodes}
    if len(nodes) == 2 and all(abs(s[i]) <= 0.3 for _, i in nodes):
        # Both peaks mid-view (the plate's shear lobes either side of the hole
        # top): read away from each other.
        lo, hi = sorted((i for _, i in nodes), key=lambda i: s[i])
        side = {lo: "left", hi: "right"}
    return [(name, i, float(val[i]), f"{'top' if name == 'max' else 'bottom'} {side[i]}")
            for name, i in nodes]


def figure(payload: dict, height: int = 540):
    """The Plotly figure: shaded field on the surface, element edges, labelled extremes.

    The field and the deformed shape are switched by Plotly's own menus inside
    the figure, not by dashboard widgets. A widget change re-runs the Streamlit
    script, which rebuilds the chart from scratch and throws away wherever the
    user had turned the camera; a menu only restyles the figure in the browser.
    So every field and both shapes travel with the figure, packed as float32.
    """
    import plotly.graph_objects as go

    xyz = payload["xyz"].astype(float)
    tri = payload["tri"]
    plane = int(payload["dim"]) == 2
    factor = auto_warp_factor(payload)
    shapes = {"undeformed": xyz, f"deformed ×{factor:.3g}": xyz + factor * payload["U"].astype(float)}
    keys = fields_for(payload)
    vals = {k: field_values(payload, k) for k in keys}

    # Every field's extreme(s) in one marker trace; a field shows its own
    # markers and hides the others' (size 0, no text).
    opening = "top" if plane else "iso"
    picks = [(k,) + e for k in keys for e in extremes(payload, k, vals[k], opening)]
    nodes = np.array([p[2] for p in picks])

    def marks(key):
        on = [p[0] == key for p in picks]
        unit = FIELDS[key][1]
        return dict(size=[5 if o else 0 for o in on],
                    text=[f"{p[1]} {p[3]:.4g} {unit}" if o else "" for o, p in zip(on, picks)])

    where = "x %{customdata[0]:.2f}  y %{customdata[1]:.2f}" + ("" if plane else "  z %{customdata[2]:.2f}")

    def hover(key):
        label, unit, _ = FIELDS[key]
        return f"{where} mm<br>{label} %{{intensity:.4g}} {unit}<extra></extra>"

    lines = _edge_polylines(list(shapes.values()), payload["edges"], tri)

    # Sent as float32 / the smallest index type, like the menus: the figure
    # travels to the browser on every dashboard rerun.
    f4 = lambda a: np.asarray(a, dtype=np.float32)
    idx = tri.astype(np.uint16 if len(xyz) <= np.iinfo(np.uint16).max else np.uint32)
    first = keys[0]
    m0, fr0 = marks(first), fringe(vals[first])
    surface = go.Mesh3d(
        x=f4(xyz[:, 0]), y=f4(xyz[:, 1]), z=f4(xyz[:, 2]), i=idx[:, 0], j=idx[:, 1], k=idx[:, 2],
        intensity=f4(vals[first]), intensitymode="vertex",
        colorscale=[[i / (len(RAINBOW) - 1), c] for i, c in enumerate(RAINBOW)],
        cmin=fr0["cmin"], cmax=fr0["cmax"],
        # Horizontal under the model, as on the PNGs.
        colorbar=dict(orientation="h", x=0.5, xanchor="center", y=0.0, yanchor="bottom",
                      len=0.85, thickness=12, outlinewidth=0,
                      # No bar title: the field menu names the field, as the
                      # title does on the PNGs.
                      tickvals=fr0["tickvals"], ticktext=fr0["ticktext"],
                      tickfont=dict(color=TEXT, size=10)),
        lighting=dict(ambient=0.85, diffuse=0.2, specular=0.0, roughness=1.0, fresnel=0.0),
        flatshading=False, customdata=f4(xyz), hovertemplate=hover(first),
        showlegend=False,
    )
    wire = go.Scatter3d(x=f4(lines[0][:, 0]), y=f4(lines[0][:, 1]), z=f4(lines[0][:, 2]), mode="lines",
                        line=dict(color=EDGE, width=1.5), hoverinfo="skip", showlegend=False)
    ext = go.Scatter3d(
        x=xyz[nodes, 0], y=xyz[nodes, 1], z=xyz[nodes, 2], mode="markers+text",
        marker=dict(size=m0["size"], color=TEXT, line=dict(color=PANEL, width=1)),
        text=m0["text"], textposition=[p[4] for p in picks], textfont=dict(color=TEXT, size=11),
        hoverinfo="skip", showlegend=False)

    # Menus restyle only the attributes they own -- the field menu the colours
    # (trace 0) and the marker labels (trace 2), the shape menu the coordinates
    # -- so either can be used in any state of the other. null leaves a trace
    # an attribute does not apply to untouched.
    field_menu = []
    for k in keys:
        fr, m = fringe(vals[k]), marks(k)
        field_menu.append(dict(label=field_label(k), method="restyle", args=[{
            "intensity": [_b64(vals[k]), None], "cmin": [fr["cmin"], None], "cmax": [fr["cmax"], None],
            "colorbar.tickvals": [fr["tickvals"], None], "colorbar.ticktext": [fr["ticktext"], None],
            "hovertemplate": [hover(k), None],
            "marker.size": [None, m["size"]], "text": [None, m["text"]],
        }, [0, 2]]))
    shape_menu = []
    for (name, p), line in zip(shapes.items(), lines):
        shape_menu.append(dict(label=name, method="restyle", args=[{
            ax: [_b64(p[:, d]), _b64(line[:, d]), _b64(p[nodes, d])] for d, ax in enumerate("xyz")
        }, [0, 1, 2]]))
    # Plotly paints the active and hovered menu item near-white and offers no
    # setting for it, so the menus are light with dark text: legible in every
    # state, where light text on a dark menu would vanish when selected.
    menu = dict(bgcolor=MENU_BG, bordercolor=PANEL, borderwidth=1, showactive=True, active=0,
                font=dict(color=PANEL, size=11), xanchor="left", yanchor="top", y=1.0,
                pad=dict(l=2, r=2, t=2, b=2))
    updatemenus = [dict(type="dropdown", direction="down", x=0.0, buttons=field_menu, **menu),
                   dict(type="buttons", direction="right", x=0.4, buttons=shape_menu, **menu)]

    # True proportions, scaled so the longest side is 1. Plotly's own "data"
    # mode scales to a unit VOLUME instead, which stretches a 10:1 beam to 4.6
    # units and puts any fixed camera inside it.
    # The axis ranges are fixed too, to a box holding BOTH shapes. Left to
    # Plotly, a range follows the data shown while the box keeps the
    # proportions set here: in the deformed shape the cantilever's tip goes
    # 5 mm down, z spanned 15 mm in a box sized for 10, and the 10 x 10
    # section was drawn 10 x 6.7.
    both = np.vstack(list(shapes.values()))
    pad = 0.03 * float((both.max(axis=0) - both.min(axis=0)).max())
    lo, hi = both.min(axis=0) - pad, both.max(axis=0) + pad
    ratio = (hi - lo) / (hi - lo).max()
    direction, up, parallel = VIEWS[opening]
    camera = dict(up=dict(zip("xyz", up)),
                  projection=dict(type="orthographic" if parallel else "perspective"))
    if plane:
        # Straight down, x to the right, y up. Orthographic, so the camera
        # distance does not zoom; the box size does (1.4 fills a 2:1 plate to
        # about three quarters of the panel width).
        ratio = ratio * 1.4
        ratio[2] = 0.07                                  # a flat part has no depth
        camera["eye"] = dict(x=0.0, y=0.0, z=1.6)
        zaxis = dict(_axis("", (lo[2], hi[2])), visible=False)
    else:
        # Looking along the length: the long axis recedes into depth, so the
        # part stays in frame however narrow the panel is.
        camera["eye"] = dict(zip("xyz", direction))
        zaxis = _axis("z [mm]", (lo[2], hi[2]))

    fig = go.Figure([surface, wire, ext])
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family='ui-monospace, "SF Mono", Menlo, monospace', color=TEXT),
        updatemenus=updatemenus,
        scene=dict(xaxis=_axis("x [mm]", (lo[0], hi[0])), yaxis=_axis("y [mm]", (lo[1], hi[1])), zaxis=zaxis,
                   domain=dict(x=[0, 1], y=[0.16, 1]),          # the bar lives below
                   aspectmode="manual", aspectratio=dict(x=ratio[0], y=ratio[1], z=ratio[2]),
                   bgcolor=PANEL, camera=camera,
                   # Turntable keeps z up, which is right for the solid but
                   # leaves a straight-down view of the plate free to spin.
                   dragmode="orbit" if plane else "turntable"),
    )
    return fig


# A monospace font with Greek, as in the dashboard: VTK's built-in Courier has
# no σ and drops "|". Without either file the PNG text falls back to ASCII.
MONO_FONTS = ("/System/Library/Fonts/Menlo.ttc",                        # macOS
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")    # Debian / Ubuntu


def _mono_font():
    return next((f for f in MONO_FONTS if Path(f).exists()), None)


def _polyline_cells(line: np.ndarray):
    """NaN-separated polylines -> (points, VTK line connectivity)."""
    ok = np.flatnonzero(~np.isnan(line[:, 0]))
    breaks = np.flatnonzero(np.diff(ok) > 1) + 1
    cells = []
    for s, e in zip(np.r_[0, breaks], np.r_[breaks, len(ok)]):
        cells += [e - s, *range(s, e)]
    return line[ok], np.asarray(cells)


def _place_camera(pl, pts: np.ndarray, view: str, size: tuple) -> None:
    """Point the camera as VIEWS says and fit the model above the fringe bar.

    Parallel views are fitted exactly: the model's extent across and up the
    screen is measured and the parallel scale set so it fills the band between
    the title and the bar. The perspective view keeps VTK's fit, zoomed in.
    """
    direction, _, parallel = VIEWS[view]
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    right, u = _screen_axes(view)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = 0.5 * (lo + hi)
    dist = 4.0 * float(np.linalg.norm(hi - lo))
    pl.camera.up = tuple(u)
    if not parallel:
        pl.camera.focal_point = tuple(centre)
        pl.camera.position = tuple(centre + d * dist)
        pl.reset_camera()
        pl.camera.zoom(1.35)
        return
    pl.enable_parallel_projection()
    x, y = (pts - centre) @ right, (pts - centre) @ u
    bottom, top, side = 0.17, 0.08, 0.06       # bands kept clear: bar, title, margins
    band = 1.0 - bottom - top
    aspect = size[0] / size[1]
    scale = max((y.max() - y.min()) / 2 / band, (x.max() - x.min()) / 2 / (aspect * (1 - 2 * side)))
    # Centre the model in the band, not in the window.
    focal = centre + right * 0.5 * (x.max() + x.min()) + u * 0.5 * (y.max() + y.min())
    focal = focal - u * (bottom + band / 2 - 0.5) * 2 * scale
    pl.camera.focal_point = tuple(focal)
    pl.camera.position = tuple(focal + d * dist)
    pl.camera.parallel_scale = scale


def render_png(payload: dict, key: str, png_path, view: str = "iso", warp: float = 0.0,
               size: tuple = (1280, 800)) -> None:
    """One fixed view of one field as a PNG, drawn exactly like the 3D view.

    Same surface, element edges, colours, range and bar labels as the
    interactive figure, from the same payload -- so a PNG and the 3D view of a
    field can be put side by side, and the PNGs can go into a report as they
    are. `warp` > 0 draws the deformed shape at that exaggeration.
    """
    import pyvista as pv
    from matplotlib.colors import LinearSegmentedColormap
    from vtkmodules.vtkCommonCore import VTK_FONT_FILE

    xyz = payload["xyz"].astype(float)
    tri = payload["tri"].astype(np.int64)
    pts = xyz + warp * payload["U"].astype(float)
    unit = FIELDS[key][1]
    val = field_values(payload, key)
    fr = fringe(val)

    surf = pv.PolyData(pts, np.hstack([np.full((len(tri), 1), 3), tri]).ravel())
    surf.point_data["value"] = val
    line = _edge_polylines([pts], payload["edges"], tri)[0]
    lp, lc = _polyline_cells(line)
    wire = pv.PolyData(lp, lines=lc)

    font = _mono_font()
    text = (lambda s: s) if font else (lambda s: s.replace("σ", "s").replace("×", "x"))

    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.background_color = PANEL
    pl.add_mesh(surf, scalars="value", cmap=LinearSegmentedColormap.from_list("fringe", RAINBOW),
                clim=(fr["cmin"], fr["cmax"]), ambient=0.85, diffuse=0.2, specular=0.0,
                show_scalar_bar=False)
    # No bar title: VTK centres it over the middle label. The field and unit are
    # in the view's title, as they are in the 3D view's field menu.
    bar = pl.add_scalar_bar(title="", n_labels=N_TICKS, fmt=TICK_FMT,
                            color=TEXT, vertical=False, position_x=0.075, position_y=0.02,
                            width=0.85, height=0.125, title_font_size=18, label_font_size=16)
    if font:
        for prop in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
            prop.SetFontFamily(VTK_FONT_FILE)
            prop.SetFontFile(font)
    pl.add_mesh(wire, color=EDGE_RGB, opacity=EDGE_ALPHA, line_width=1)
    for name, i, v, pos in extremes(payload, key, val, view):
        leftward = pos.endswith("left")
        # A shadow keeps the label legible on the fringe's brightest colours.
        pl.add_point_labels(pts[i:i + 1], [text(f" {name} {v:.4g} {unit} ")], font_size=17,
                            font_file=font, text_color=TEXT, point_color=TEXT, point_size=10,
                            render_points_as_spheres=True, shape=None, shadow=True,
                            always_visible=True,
                            justification_horizontal="right" if leftward else "left",
                            justification_vertical="bottom" if name == "max" else "top")
    _place_camera(pl, pts, view, size)
    title = f"{field_label(key)} · {view} view" + (f" · deformed ×{warp:.3g}" if warp else "")
    pl.add_text(text(title), position="upper_left", font_file=font, color=TEXT, font_size=12)
    pl.screenshot(str(png_path))
    pl.close()


# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the interactive 3D view of a run (viewer.npz + a standalone viewer.html).")
    add_workdir_argument(parser)
    args = parser.parse_args()
    paths = from_args(args)

    import pyvista as pv
    if not paths.vtu.exists():
        print(f"ERROR: {paths.vtu} not found; run postprocess.py first")
        return 1
    info = save_payload(pv.read(str(paths.vtu)), paths.viewer_npz)
    payload = load_payload(paths.viewer_npz)
    html = paths.root / "viewer.html"
    figure(payload).write_html(html, include_plotlyjs="cdn", config=PLOTLY_CONFIG)
    print(f"viewer.npz : {info['path']}  ({info['kbytes']:.0f} KB; {info['n_nodes']} surface nodes, "
          f"{info['n_triangles']} triangles, {info['n_edges']} element edges)")
    print(f"viewer.html: {html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
