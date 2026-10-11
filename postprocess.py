#!/usr/bin/env python3
"""Postprocess layer 6 (plan section 6, plate plan M8): .frd -> .vtu -> 3D view + PNGs.

Runs ONLY after physics validation (layer 5) passes; the orchestrator skips it
on a Critical FAIL. Three steps:
  (a) convert  : ccx2paraview out/beam.frd vtu   (PyVista cannot read .frd)
  (b) export   : viewer.npz -- the outer surface, its nodal results and the
                 element edges -- which the dashboard shows as an interactive
                 3D view (viewer.py)
  (c) render   : the fixed views as PNGs, drawn by viewer.render_png FROM THE
                 SAME viewer.npz: same element edges, rainbow fringe, range and
                 bar labels as the 3D view. The 3D view is for looking around;
                 the PNGs are the views worth keeping.

  cantilever_beam : von Mises looked at along each axis (stress_x/y/z.png: the
                    clamped end's face, the side, the top) + the deformed shape
                    from the side (displacement.png).
  plate_with_hole : the model is flat, so three directions would be pointless.
                    Instead the top view of the four stress fields that
                    describe a plane-stress state -- sxx / syy / sxy / von
                    Mises -- plus the deformed shape.

Max displacement and max von Mises are RECORDED for the summary, and cross-
checked against the .frd to confirm the conversion is lossless. For the
cantilever, stress is information only and never auto-judged; for the plate the
SCF is judged in layer 7 (plate plan D3) -- these renders are still the human's
view of the field either way.
Fallback: if headless rendering fails, open out/beam.vtu in the ParaView GUI.
The deformed views use the 3D view's exaggeration rule (peak |U| drawn at 5% of
the model diagonal), so a PNG and the 3D view show the same deformed shape.

Unit system: SI-mm (displacement mm, stress MPa).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from params import PARAMS
from paths import RunPaths, add_workdir_argument, from_args
from physics_validation import _von_mises, parse_frd
import viewer

# Plate renders (top view): PNG key -> viewer field. The filenames come from
# RunPaths.plate_png(key).
PLATE_PNG = {"stress_xx": "sxx", "stress_yy": "syy", "stress_xy": "sxy", "stress_vm": "vm"}

# Beam PNGs are named by the axis the camera looks ALONG -> the viewer's view.
BEAM_VIEWS = {"x": "end", "y": "side", "z": "top"}
MIN_PNG_BYTES = 5000  # a real 3D render is tens of KB; a blank image is ~1-2 KB


def convert_frd_to_vtu(frd_path: Path) -> Path:
    """Convert .frd to .vtu via the ccx2paraview CLI. Returns the .vtu path."""
    if shutil.which("ccx2paraview") is None:
        raise RuntimeError("ccx2paraview not on PATH (activate the fea env)")
    if not Path(frd_path).exists():
        raise FileNotFoundError(f"{frd_path} not found; run solve.sh first")
    proc = subprocess.run(["ccx2paraview", str(frd_path), "vtu"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ccx2paraview failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    produced = sorted(Path(frd_path).parent.glob(Path(frd_path).stem + "*.vtu"))
    if not produced:
        raise RuntimeError("ccx2paraview produced no .vtu file")
    return produced[0]


def _export_viewer(mesh, paths: RunPaths) -> tuple:
    """viewer.npz for the dashboard's interactive 3D view: (summary, error).

    Non-fatal like the renders: a failure is reported in the acceptance check,
    and none of the recorded numbers depend on it.
    """
    try:
        return viewer.save_payload(mesh, paths.viewer_npz), ""
    except Exception as exc:  # noqa: BLE001
        return {}, f"{exc.__class__.__name__}: {exc}"


def _render_views(views: list, warp_factor, paths: RunPaths) -> tuple:
    """Draw the fixed views from viewer.npz: (warp used, error or "").

    `views` is [(field, view, png path, deformed?)]. The deformed views use the
    3D view's exaggeration unless the caller gave one.
    """
    payload = viewer.load_payload(paths.viewer_npz)
    warp = viewer.auto_warp_factor(payload) if warp_factor is None else warp_factor
    try:
        for field, view, png, deformed in views:
            viewer.render_png(payload, field, png, view, warp=warp if deformed else 0.0)
    except Exception as exc:  # noqa: BLE001
        return warp, f"{exc.__class__.__name__}: {exc}"
    return warp, ""


def _postprocess_plate(mesh, frd, vtu_path, warp_factor, paths: RunPaths) -> dict:
    """Plate: 3D view data + top views of sxx / syy / sxy / von Mises + deformed (plate M8)."""
    disp = np.asarray(mesh.point_data["U"])
    u_mag = np.linalg.norm(disp, axis=1)
    S = np.asarray(mesh.point_data["S"])
    if "S_Mises" in mesh.point_data:
        vm = np.asarray(mesh.point_data["S_Mises"]).ravel()
    else:
        vm = np.array([_von_mises(s) for s in S])
        mesh.point_data["S_Mises"] = vm
    max_disp, max_vm = float(u_mag.max()), float(vm.max())

    ref_disp = max(float(np.linalg.norm(v)) for v in frd["DISP"].values())
    ref_vm = max(_von_mises(s) for s in frd["STRESS"].values())

    view, render_error = _export_viewer(mesh, paths)
    if view:
        warp_factor, render_error = _render_views(
            [(field, "top", paths.plate_png(key), False) for key, field in PLATE_PNG.items()]
            + [("u", "top", paths.displacement_png, True)], warp_factor, paths)

    pngs = {k: paths.plate_png(k) for k in PLATE_PNG}
    pngs["displacement"] = paths.displacement_png
    png_sizes = {name: (p.stat().st_size if p.exists() else 0) for name, p in pngs.items()}

    return {
        "geometry_type": "plate_with_hole",
        "vtu_path": str(vtu_path),
        "n_points": int(mesh.n_points), "n_cells": int(mesh.n_cells),
        "fields": list(mesh.point_data.keys()),
        "max_disp": max_disp, "ref_disp": ref_disp,
        "disp_ok": bool(np.isclose(max_disp, ref_disp, rtol=1e-3)),
        "max_vm": max_vm, "ref_vm": ref_vm,
        "vm_ok": bool(np.isclose(max_vm, ref_vm, rtol=1e-3)),
        "warp_factor": warp_factor,
        "stress_ranges": {
            "sigma_xx": (float(S[:, 0].min()), float(S[:, 0].max())),
            "sigma_yy": (float(S[:, 1].min()), float(S[:, 1].max())),
            "sigma_xy": (float(S[:, 3].min()), float(S[:, 3].max())),
        },
        "png_sizes": png_sizes, "render_error": render_error,
        "viewer": view, "viewer_error": render_error if not view else "",
    }


def run_postprocess(warp_factor=None, params=None, paths: RunPaths = None) -> dict:
    """Convert, render, and cross-check. Returns a summary dict.

    warp_factor=None means "choose one": the 3D view's rule, peak |U| drawn at
    5% of the model diagonal (viewer.auto_warp_factor), for both geometries.
    """
    params = params if params is not None else PARAMS
    paths = paths or RunPaths.default()
    vtu_path = convert_frd_to_vtu(paths.frd)
    if params["geometry"]["type"] == "plate_with_hole":
        return _postprocess_plate(pv.read(str(vtu_path)), parse_frd(paths.frd),
                                  vtu_path, warp_factor, paths)
    return _postprocess_beam(vtu_path, paths.frd, warp_factor, paths)


def _postprocess_beam(vtu_path, frd_path, warp_factor, paths: RunPaths) -> dict:
    """Cantilever: 3D view data + von Mises along each axis + the deformed side view."""
    mesh = pv.read(str(vtu_path))
    frd_path = Path(frd_path)

    # Fields from the .vtu (ccx2paraview precomputes S_Mises; fall back if absent).
    disp = np.asarray(mesh.point_data["U"])
    u_mag = np.linalg.norm(disp, axis=1)
    if "S_Mises" in mesh.point_data:
        vm = np.asarray(mesh.point_data["S_Mises"]).ravel()
    else:
        vm = np.array([_von_mises(s) for s in np.asarray(mesh.point_data["S"])])
        mesh.point_data["S_Mises"] = vm
    max_disp = float(u_mag.max())
    max_vm = float(vm.max())

    # Lossless-conversion cross-check against the .frd (plan layer-6 acceptance).
    frd = parse_frd(Path(frd_path))
    ref_disp = max(float(np.linalg.norm(v)) for v in frd["DISP"].values())
    ref_vm = max(_von_mises(s) for s in frd["STRESS"].values())
    disp_ok = bool(np.isclose(max_disp, ref_disp, rtol=1e-3))
    vm_ok = bool(np.isclose(max_vm, ref_vm, rtol=1e-3))

    # 3D view data, then the fixed views drawn from it. Non-fatal on failure;
    # the ParaView GUI on the .vtu is the documented human fallback.
    view, render_error = _export_viewer(mesh, paths)
    if view:
        warp_factor, render_error = _render_views(
            [("vm", v, paths.beam_stress_png(axis), False) for axis, v in BEAM_VIEWS.items()]
            + [("u", "side", paths.displacement_png, True)], warp_factor, paths)

    pngs = {**{f"stress_{a}": paths.beam_stress_png(a) for a in BEAM_VIEWS},
            "displacement": paths.displacement_png}
    png_sizes = {name: (p.stat().st_size if p.exists() else 0) for name, p in pngs.items()}

    return {
        "geometry_type": "cantilever_beam",
        "vtu_path": str(vtu_path),
        "n_points": int(mesh.n_points), "n_cells": int(mesh.n_cells),
        "fields": list(mesh.point_data.keys()),
        "max_disp": max_disp, "ref_disp": ref_disp, "disp_ok": disp_ok,
        "max_vm": max_vm, "ref_vm": ref_vm, "vm_ok": vm_ok, "warp_factor": warp_factor,
        "png_sizes": png_sizes, "render_error": render_error,
        "viewer": view, "viewer_error": render_error if not view else "",
    }


def report_and_check(s: dict) -> bool:
    """Print the acceptance report (plan section 6 / plate M8); PASS (True)/FAIL."""
    is_plate = s.get("geometry_type") == "plate_with_hole"
    print("=" * 64)
    print("postprocess.py acceptance report"
          + ("  (plate_with_hole, plate plan M8)" if is_plate else ""))
    print("=" * 64)
    print(f"vtu      : {s['vtu_path']}  ({s['n_points']} points, {s['n_cells']} cells)")
    print(f"fields   : {', '.join(s['fields'])}")
    print(f"max |U|  : {s['max_disp']:.6f} mm   (frd: {s['ref_disp']:.6f} mm)")
    vm_note = "judged as the SCF in layer 7" if is_plate else "recorded only, not judged"
    print(f"max vM   : {s['max_vm']:.2f} MPa   (frd: {s['ref_vm']:.2f} MPa)  [{vm_note}]")
    if is_plate:
        print( "components (plane stress; sigma_zz == 0 by definition):")
        for name, (lo, hi) in s["stress_ranges"].items():
            print(f"           {name:<9}: [{lo:9.3f}, {hi:9.3f}] MPa")
    if s.get("warp_factor") is not None:
        print(f"warp     : x{s['warp_factor']:.3g} (peak |U| drawn at 5% of the model diagonal,"
              f" never below x1, as in the 3D view)")
    print( "renders  :")
    for name, size in s["png_sizes"].items():
        print(f"           {name:<14}: {size / 1024:.1f} KB")
    if s["render_error"]:
        print(f"  !! render error: {s['render_error']}")
        print(f"     fallback: open {s['vtu_path']} in the ParaView GUI (human path).")
    v = s["viewer"]
    if v:
        print(f"3D view  : {Path(v['path']).name}  ({v['kbytes']:.0f} KB; {v['n_faces']} element "
              f"faces, {v['n_edges']} element edges; a browser draws the faces as "
              f"{v['n_triangles']} triangles)")
    else:
        print(f"  !! 3D view export error: {s['viewer_error']}")

    pngs_ok = all(size >= MIN_PNG_BYTES for size in s["png_sizes"].values())
    n_png = len(s["png_sizes"])
    checks = [
        ("vtu loads in PyVista (points > 0)", s["n_points"] > 0),
        ("max |U| matches frd (lossless conversion)", s["disp_ok"]),
        ("max von Mises matches frd (lossless conversion)", s["vm_ok"]),
        (f"all {n_png} PNGs exist and are non-empty", pngs_ok),
        ("3D view data exported (viewer.npz)", bool(s["viewer"])),
    ]
    passed = all(ok for _, ok in checks)
    print("-" * 64)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 64)
    print(f"ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    if passed:
        if is_plate:
            print(f"Human check (optional): open {s['vtu_path']} in ParaView; confirm the hole")
            print("  peak sits on the transverse diameter and the field is smooth.")
            print("Next: numerical verification (layer 7) - SCF vs Howland, displacement N/A.")
        else:
            print(f"Human check (optional): open {s['vtu_path']} in ParaView; "
                  f"confirm bend & stress pattern.")
            print("Next: numerical verification (layer 7) - displacement vs analytical, stress N/A.")
    print("=" * 64)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Postprocess frd -> vtu -> PNG (plan layer 6 / plate M8).")
    parser.add_argument("--warp", type=float, default=None,
                        help="displacement exaggeration factor "
                             "(default: the 3D view's rule -- peak |U| drawn at 5%% of the model "
                             "diagonal, never below 1)")
    add_workdir_argument(parser)
    args = parser.parse_args()
    summary = run_postprocess(warp_factor=args.warp, paths=from_args(args))
    return 0 if report_and_check(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
