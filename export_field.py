#!/usr/bin/env python3
"""Field export (plate plan M11): FE solution -> regular-grid tensors for the
stress-field surrogate.

Turns one solved case into the arrays a CNN consumes:

  inputs   mask  (H, W) uint8    1 = material, 0 = hole or outside the plate
           sdf   (H, W) float32  signed distance to the HOLE boundary, in mm
                                 (positive in material, negative inside the hole)
  targets  sxx, syy, sxy (H, W) float32, each divided by sigma_gross = F/(b*t)

Why an SDF and not just the mask: a binary mask is flat everywhere except at
the two edges, so a pixel far from the hole looks identical to one close to it
and the network can only recover "how far am I from the hole" by growing a
receptive field big enough to see it. Stress concentration decays with exactly
that distance, so the SDF hands the single most important variable to the
network for free. The mask is kept as well -- it is nearly free and it is what
masks the loss.

Why three stress components and not von Mises: von Mises is a nonlinear
function of them, so predicting the components lets us derive it AND check the
prediction against equilibrium; predicting von Mises directly throws that away.
Dividing by sigma_gross makes the target dimensionless, so a trained model
transfers to any F and t (sigma ~ F/t is a free rescaling).

The canvas is a FIXED physical window at a fixed mm/pixel, not a box stretched
to fit each plate. Stretching would distort the geometry (a circular hole would
become an ellipse whenever the plate aspect ratio changed) and would hide a
change of plate size from the network entirely -- which would make the
aspect-ratio out-of-distribution test meaningless.

Interpolation uses VTK's cell-based sampling on the real quadratic elements,
i.e. the element shape functions themselves, not a scattered-point fit.

Unit system: SI-mm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gmsh
import numpy as np
import pyvista as pv

from paths import RunPaths, add_workdir_argument, from_args
from physics_validation import _von_mises, parse_frd

# Canvas: a fixed physical window, so every case lands on the same pixel grid.
CANVAS_MM = (200.0, 100.0)     # x, y extent covered by the grid
# Resolution chosen by measurement, not by picking a round number. Two separate
# quantities were measured across hole sizes r = 8 / 14 / 20 / 30 mm:
#
#   grid        field nRMSE (round-trip)   peak lost at the hole   505 cases
#   256 x 128        0.19 - 0.39 %                 ~5.0 %            201 MB
#   384 x 192        0.13 - 0.26 %                 ~2.6 %            462 MB
#   512 x 256        0.09 - 0.19 %                 ~2.6 % (noisy)    831 MB
#
# The FIELD is already well captured at 256 x 128; what improves from 256 to
# 384 is the PEAK, and the two are not the same thing. The peak sits exactly ON
# the hole boundary, where no pixel centre can ever sit, so a fixed fraction of
# it is lost by any pixel grid -- refining past 384 stops helping and starts
# jittering with grid alignment. 384 x 192 halves the peak floor for 2.2x the
# storage; 512 doubles the storage again for nothing.
#
# The remaining ~2.6% peak loss is a REPRESENTATION FLOOR, not a solver error.
# It applies equally to the label and to any grid-based prediction, so it does
# not distort a CNN's measured accuracy -- but it does bound what the surrogate
# can say about the true peak. The true FE peak is therefore stored alongside
# each case (vm_max_fe / kt_fe) so the floor stays visible and quantified.
DEFAULT_SHAPE = (384, 192)     # nx, ny  -> 0.521 mm/pixel

# gmsh 2D element type -> (nodes per element, VTK cell type). The gmsh node
# order matches VTK's quadratic cells exactly (verified in plate plan M4).
_VTK_2D = {
    16: (8, pv.CellType.QUADRATIC_QUAD),
    9: (6, pv.CellType.QUADRATIC_TRIANGLE),
}


def fe_grid(paths: RunPaths = None) -> pv.UnstructuredGrid:
    """The solved mesh as a PyVista grid carrying the nodal stress tensor."""
    paths = paths or RunPaths.default()
    msh_path, frd_path = paths.msh, paths.frd
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.open(str(msh_path))
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        ids = [int(t) for t in ntags]
        pts = np.asarray(ncoords, dtype=float).reshape(-1, 3)
        idx = {nid: k for k, nid in enumerate(ids)}
        cells, ctypes = [], []
        for etype, tags, conn in zip(*gmsh.model.mesh.getElements(2)):
            etype = int(etype)
            if etype not in _VTK_2D:
                raise RuntimeError(f"unsupported 2D element type {etype}")
            n_per, vtk_type = _VTK_2D[etype]
            conn = [int(x) for x in conn]
            for e in range(len(tags)):
                nodes = conn[e * n_per:(e + 1) * n_per]
                cells += [n_per] + [idx[n] for n in nodes]
                ctypes.append(vtk_type)
    finally:
        if own:
            gmsh.finalize()

    grid = pv.UnstructuredGrid(np.array(cells, np.int64), np.array(ctypes, np.uint8), pts)
    frd = parse_frd(Path(frd_path))
    S = np.zeros((len(ids), 6))
    for k, nid in enumerate(ids):
        if nid in frd["STRESS"]:
            S[k] = frd["STRESS"][nid]
    grid.point_data["S"] = S
    grid.point_data["S_Mises"] = np.array([_von_mises(s) for s in S])
    return grid


def export_case(params: dict, shape=DEFAULT_SHAPE, paths: RunPaths = None,
                npz_path=None) -> dict:
    """Sample one solved case onto the regular grid. Returns the arrays + meta."""
    g = params["geometry"]["plate_with_hole"]
    a, b, t = float(g["a"]), float(g["b"]), float(g["t"])
    r, xc, yc = float(g["r"]), float(g["xc"]), float(g["yc"])
    F = float(params["bc"]["loads"][0]["magnitude"])
    sigma_gross = F / (b * t)
    nx, ny = shape

    # Pixel centres of the fixed canvas.
    dx, dy = CANVAS_MM[0] / nx, CANVAS_MM[1] / ny
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys, indexing="xy")           # (ny, nx)

    # Geometry channels, analytic (the hole is a circle -- no need to estimate).
    sdf = np.hypot(X - xc, Y - yc) - r                  # + in material, - in the hole
    inside_plate = (X >= 0) & (X <= a) & (Y >= 0) & (Y <= b)
    mask = ((sdf > 0) & inside_plate).astype(np.uint8)

    # Stress channels: VTK cell-based sampling on the quadratic elements.
    grid = fe_grid(paths)
    probe = pv.ImageData(dimensions=(nx, ny, 1), spacing=(dx, dy, 1.0),
                         origin=(0.5 * dx, 0.5 * dy, 0.0))
    sampled = probe.sample(grid)
    valid = np.asarray(sampled.point_data["vtkValidPointMask"]).reshape(ny, nx).astype(bool)
    S = np.asarray(sampled.point_data["S"]).reshape(ny, nx, 6)

    # A pixel the sampler could not place in any element is outside the material.
    mask = (mask.astype(bool) & valid).astype(np.uint8)
    m = mask.astype(bool)
    chans = {}
    for name, comp in (("sxx", 0), ("syy", 1), ("sxy", 3)):
        f = np.where(m, S[:, :, comp] / sigma_gross, 0.0)
        chans[name] = f.astype(np.float32)

    # Peak recovered FROM THE GRID, and the true FE peak. Both are stored: the
    # gap between them is the representation floor of this grid, and keeping the
    # true value means the surrogate's peak can be judged against ground truth
    # rather than against an already-smoothed label.
    vm_grid = np.sqrt(np.where(m, chans["sxx"] ** 2 - chans["sxx"] * chans["syy"]
                               + chans["syy"] ** 2 + 3 * chans["sxy"] ** 2, 0.0))
    vm_max_grid = float(vm_grid.max() * sigma_gross)
    vm_max_fe = float(np.max(grid.point_data["S_Mises"]))
    net_width = b - 2.0 * r
    sigma_net = F / (net_width * t) if net_width > 0 else float("nan")

    out = {
        "mask": mask,
        "sdf": sdf.astype(np.float32),
        **chans,
        # Self-describing scalars: a consumer never has to know a column order.
        "a": np.float32(a), "b": np.float32(b), "t": np.float32(t),
        "r": np.float32(r), "xc": np.float32(xc), "yc": np.float32(yc),
        "F": np.float32(F),
        "sigma_gross": np.float32(sigma_gross), "sigma_net": np.float32(sigma_net),
        "vm_max_fe": np.float32(vm_max_fe),        # ground-truth peak, grid-independent
        "vm_max_grid": np.float32(vm_max_grid),    # what this grid preserves of it
        "kt_fe": np.float32(vm_max_fe / sigma_net if sigma_net else np.nan),
        "mm_per_px": np.float32(dx),
    }
    if npz_path is not None:
        np.savez_compressed(npz_path, **out)
    return {"arrays": out, "shape": (ny, nx), "sigma_gross": sigma_gross,
            "n_material_px": int(mask.sum()), "vm_max_grid": vm_max_grid,
            "vm_max_fe": vm_max_fe,
            "peak_loss_pct": 100.0 * (vm_max_fe - vm_max_grid) / vm_max_fe,
            "npz_path": str(npz_path) if npz_path else ""}


def resolution_study(params: dict, shapes=((128, 64), (256, 128), (384, 192), (512, 256)),
                     paths: RunPaths = None) -> list:
    """How much of the peak does each grid resolution keep?

    The export is an interpolation, and interpolation smooths peaks. The peak IS
    the engineering quantity, so the resolution has to be chosen by measuring
    what it costs -- not by picking a round number.
    """
    g = params["geometry"]["plate_with_hole"]
    rows = []
    for shape in shapes:
        res = export_case(params, shape=shape, paths=paths)
        px = CANVAS_MM[0] / shape[0]
        rows.append({
            "shape": shape, "mm_per_px": px,
            "hole_px": 2 * float(g["r"]) / px,
            "vm_max_grid": res["vm_max_grid"], "vm_max_fe": res["vm_max_fe"],
            "peak_loss_pct": 100 * (res["vm_max_fe"] - res["vm_max_grid"]) / res["vm_max_fe"],
            "n_material_px": res["n_material_px"],
            "mb": res["arrays"]["mask"].size * 5 * 4 / 1e6,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Export the solved case to grid tensors (plate M11).")
    ap.add_argument("--npz", default=None, help="write the .npz here")
    ap.add_argument("--nx", type=int, default=DEFAULT_SHAPE[0])
    ap.add_argument("--ny", type=int, default=DEFAULT_SHAPE[1])
    ap.add_argument("--resolution-study", action="store_true",
                    help="measure how much peak stress each grid resolution keeps")
    ap.add_argument("--case", default=None,
                    help="use a prepared case from params.CASES instead of the active PARAMS")
    add_workdir_argument(ap)
    args = ap.parse_args()
    paths = from_args(args)
    from params import CASES, PARAMS

    PARAMS = CASES[args.case] if args.case else PARAMS
    if PARAMS["geometry"]["type"] != "plate_with_hole":
        ap.error("export_field only handles plate_with_hole; pass --case plate_with_hole")

    if args.resolution_study:
        print(f"{'grid':>12} {'mm/px':>7} {'hole px':>8} {'vM grid':>9} {'vM FE':>9} "
              f"{'peak lost':>10} {'material px':>12} {'MB/case':>8}")
        for row in resolution_study(PARAMS, paths=paths):
            print(f"{str(row['shape']):>12} {row['mm_per_px']:7.3f} {row['hole_px']:8.1f} "
                  f"{row['vm_max_grid']:9.3f} {row['vm_max_fe']:9.3f} "
                  f"{row['peak_loss_pct']:9.2f}% {row['n_material_px']:12d} {row['mb']:8.2f}")
        return 0

    res = export_case(PARAMS, shape=(args.nx, args.ny), paths=paths, npz_path=args.npz)
    print(f"grid {res['shape']}  material pixels {res['n_material_px']}  "
          f"peak vM grid {res['vm_max_grid']:.3f} vs FE {res['vm_max_fe']:.3f} MPa")
    if args.npz:
        print(f"wrote {res['npz_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
