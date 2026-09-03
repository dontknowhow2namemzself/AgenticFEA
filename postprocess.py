#!/usr/bin/env python3
"""Postprocess layer 6 (plan section 6, plate plan M8): .frd -> .vtu -> PNGs.

Runs ONLY after physics validation (layer 5) passes; the orchestrator skips it
on a Critical FAIL. Two stages:
  (a) convert  : ccx2paraview out/beam.frd vtu   (PyVista cannot read .frd)
  (b) render   : PyVista off-screen plots, chosen per geometry.

  cantilever_beam : von Mises from three camera directions (stress_x/y/z.png,
                    named by the axis looked along) + a deformed-shape
                    displacement plot (displacement.png).
  plate_with_hole : the model is flat, so three directions would be pointless.
                    Instead the top (x-y) view of the four stress fields that
                    describe a plane-stress state -- sxx / syy / sxy / von
                    Mises -- plus the displacement magnitude. Stress component
                    plots use a symmetric diverging map centred on zero so
                    compression around the hole reads as a different colour
                    than tension, not merely a darker shade of it.

Max displacement and max von Mises are RECORDED for the summary, and cross-
checked against the .frd to confirm the conversion is lossless. For the
cantilever, stress is information only and never auto-judged; for the plate the
SCF is judged in layer 7 (plate plan D3) -- these renders are still the human's
view of the field either way.
Fallback: if headless rendering fails, open out/beam.vtu in the ParaView GUI.

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
from physics_validation import _von_mises, parse_frd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
FRD_PATH = OUT_DIR / "beam.frd"
VTU_PATH = OUT_DIR / "beam.vtu"
STRESS_PNG = {"x": OUT_DIR / "stress_x.png", "y": OUT_DIR / "stress_y.png", "z": OUT_DIR / "stress_z.png"}
DISP_PNG = OUT_DIR / "displacement.png"

# Plate renders (top view). Keys are the out/<key>.png basenames.
PLATE_PNG = {
    "stress_xx": (OUT_DIR / "stress_xx.png", "sigma_xx [MPa]", 0, True),
    "stress_yy": (OUT_DIR / "stress_yy.png", "sigma_yy [MPa]", 1, True),
    "stress_xy": (OUT_DIR / "stress_xy.png", "sigma_xy [MPa]", 3, True),
    "stress_vm": (OUT_DIR / "stress_vm.png", "von Mises [MPa]", None, False),
}

# PNG is named by the axis the camera looks ALONG; PyVista helper renders the
# orthogonal plane (look along X -> see the Y-Z plane via view_yz, etc.).
VIEW_METHOD = {"x": "view_yz", "y": "view_xz", "z": "view_xy"}
MIN_PNG_BYTES = 5000  # a real 3D render is tens of KB; a blank image is ~1-2 KB

pv.OFF_SCREEN = True


def convert_frd_to_vtu(frd_path=FRD_PATH) -> Path:
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


def _render(mesh, scalars, title, view_method, out_png, cmap, warp_factor=None) -> None:
    """Off-screen render of `scalars` on `mesh` from one camera direction."""
    plot_mesh = mesh.warp_by_vector("U", factor=warp_factor) if warp_factor else mesh
    pl = pv.Plotter(off_screen=True, window_size=(1280, 800))
    pl.background_color = "#11161f"          # match the dashboard panel (control-room dark)
    pl.add_mesh(plot_mesh, scalars=scalars, cmap=cmap, show_edges=False,
                scalar_bar_args={"title": title, "color": "#dbe1e8", "n_labels": 5})
    getattr(pl, view_method)()
    pl.add_axes()
    pl.add_text(f"{title}  [{view_method}]", font_size=10, color="#dbe1e8")
    pl.screenshot(str(out_png))
    pl.close()


def _render_plate(mesh, scalars, title, out_png, cmap, clim=None,
                  warp_factor=None) -> None:
    """Top (x-y) view render of a flat plate field."""
    plot_mesh = mesh.warp_by_vector("U", factor=warp_factor) if warp_factor else mesh
    pl = pv.Plotter(off_screen=True, window_size=(1280, 720))
    pl.background_color = "#11161f"
    pl.add_mesh(plot_mesh, scalars=scalars, cmap=cmap, clim=clim, show_edges=False,
                scalar_bar_args={"title": title, "color": "#dbe1e8", "n_labels": 5})
    pl.view_xy()
    pl.enable_parallel_projection()   # a flat part reads wrong in perspective
    pl.camera.tight(padding=0.08)
    pl.add_text(title, font_size=10, color="#dbe1e8")
    pl.screenshot(str(out_png))
    pl.close()


def _auto_warp_factor(mesh, u_mag, target_fraction: float = 0.05) -> float:
    """Exaggeration that makes the peak displacement `target_fraction` of the
    model's diagonal -- readable but still recognisably the same part.

    A fixed factor cannot serve the batch: the plate's in-plane deflection is
    ~0.01 mm on a 200 mm part and it changes with the hole size, so any magic
    number is wrong for most cases.
    """
    b = mesh.bounds
    diag = float(np.hypot(b[1] - b[0], b[3] - b[2]))
    peak = float(u_mag.max())
    if peak <= 0 or diag <= 0:
        return 1.0
    return target_fraction * diag / peak


def _postprocess_plate(mesh, frd, vtu_path, warp_factor) -> dict:
    """Plate renders: top view of sxx / syy / sxy / von Mises + |U| (plate M8)."""
    disp = np.asarray(mesh.point_data["U"])
    u_mag = np.linalg.norm(disp, axis=1)
    mesh.point_data["U_magnitude"] = u_mag
    if warp_factor is None:
        warp_factor = _auto_warp_factor(mesh, u_mag)
    S = np.asarray(mesh.point_data["S"])
    if "S_Mises" in mesh.point_data:
        vm = np.asarray(mesh.point_data["S_Mises"]).ravel()
    else:
        vm = np.array([_von_mises(s) for s in S])
        mesh.point_data["S_Mises"] = vm
    max_disp, max_vm = float(u_mag.max()), float(vm.max())

    ref_disp = max(float(np.linalg.norm(v)) for v in frd["DISP"].values())
    ref_vm = max(_von_mises(s) for s in frd["STRESS"].values())

    render_error = ""
    try:
        for key, (path, title, comp, diverging) in PLATE_PNG.items():
            if comp is None:
                scal, cmap, clim = "S_Mises", "jet", None
            else:
                name = f"S_{key.split('_')[1]}"
                mesh.point_data[name] = S[:, comp]
                scal, cmap = name, "coolwarm"
                # Symmetric range about zero: compression and tension must be
                # visually distinct, not two shades of the same colour.
                lim = float(np.abs(S[:, comp]).max())
                clim = (-lim, lim) if diverging and lim > 0 else None
            _render_plate(mesh, scal, title, path, cmap=cmap, clim=clim)
        _render_plate(mesh, "U_magnitude", f"|U| [mm] (deformed x{warp_factor:.0f})",
                      DISP_PNG, cmap="viridis", warp_factor=warp_factor)
    except Exception as exc:  # noqa: BLE001
        render_error = f"{exc.__class__.__name__}: {exc}"

    pngs = {k: v[0] for k, v in PLATE_PNG.items()}
    pngs["displacement"] = DISP_PNG
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
    }


def run_postprocess(frd_path=FRD_PATH, warp_factor=None, params=None) -> dict:
    """Convert, render, and cross-check. Returns a summary dict.

    warp_factor=None means "choose one": the plate scales itself (see
    _auto_warp_factor), the cantilever keeps its verified x5 default.
    """
    params = params if params is not None else PARAMS
    vtu_path = convert_frd_to_vtu(frd_path)
    if params["geometry"]["type"] == "plate_with_hole":
        return _postprocess_plate(pv.read(str(vtu_path)), parse_frd(Path(frd_path)),
                                  vtu_path, warp_factor)
    return _postprocess_beam(vtu_path, frd_path,
                             5.0 if warp_factor is None else warp_factor)


def _postprocess_beam(vtu_path, frd_path, warp_factor) -> dict:
    """Cantilever renders: von Mises from three directions + deformed shape."""
    mesh = pv.read(str(vtu_path))
    frd_path = Path(frd_path)

    # Fields from the .vtu (ccx2paraview precomputes S_Mises; fall back if absent).
    disp = np.asarray(mesh.point_data["U"])
    u_mag = np.linalg.norm(disp, axis=1)
    mesh.point_data["U_magnitude"] = u_mag
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

    # Renders (von Mises x3 views + deformed displacement). Non-fatal on failure;
    # the ParaView GUI on the .vtu is the documented human fallback.
    render_error = ""
    try:
        for axis, view in VIEW_METHOD.items():
            _render(mesh, "S_Mises", "von Mises [MPa]", view, STRESS_PNG[axis], cmap="jet")
        _render(mesh, "U_magnitude", f"|U| [mm] (deformed x{warp_factor:g})",
                "view_xz", DISP_PNG, cmap="viridis", warp_factor=warp_factor)
    except Exception as exc:  # noqa: BLE001
        render_error = f"{exc.__class__.__name__}: {exc}"

    pngs = {**{f"stress_{a}": p for a, p in STRESS_PNG.items()}, "displacement": DISP_PNG}
    png_sizes = {name: (p.stat().st_size if p.exists() else 0) for name, p in pngs.items()}

    return {
        "geometry_type": "cantilever_beam",
        "vtu_path": str(vtu_path),
        "n_points": int(mesh.n_points), "n_cells": int(mesh.n_cells),
        "fields": list(mesh.point_data.keys()),
        "max_disp": max_disp, "ref_disp": ref_disp, "disp_ok": disp_ok,
        "max_vm": max_vm, "ref_vm": ref_vm, "vm_ok": vm_ok,
        "png_sizes": png_sizes, "render_error": render_error,
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
        print(f"warp     : x{s['warp_factor']:.0f} (auto: peak |U| drawn at 5% of the "
              f"model diagonal)")
    print( "renders  :")
    for name, size in s["png_sizes"].items():
        print(f"           {name:<14}: {size / 1024:.1f} KB")
    if s["render_error"]:
        print(f"  !! render error: {s['render_error']}")
        print("     fallback: open out/beam.vtu in the ParaView GUI (human path).")

    pngs_ok = all(size >= MIN_PNG_BYTES for size in s["png_sizes"].values())
    n_png = len(s["png_sizes"])
    checks = [
        ("vtu loads in PyVista (points > 0)", s["n_points"] > 0),
        ("max |U| matches frd (lossless conversion)", s["disp_ok"]),
        ("max von Mises matches frd (lossless conversion)", s["vm_ok"]),
        (f"all {n_png} PNGs exist and are non-empty", pngs_ok),
    ]
    passed = all(ok for _, ok in checks)
    print("-" * 64)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 64)
    print(f"ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    if passed:
        if is_plate:
            print("Human check (optional): open out/beam.vtu in ParaView; confirm the hole")
            print("  peak sits on the transverse diameter and the field is smooth.")
            print("Next: numerical verification (layer 7) - SCF vs Howland, displacement N/A.")
        else:
            print("Human check (optional): open out/beam.vtu in ParaView; confirm bend & stress pattern.")
            print("Next: numerical verification (layer 7) - displacement vs analytical, stress N/A.")
    print("=" * 64)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Postprocess frd -> vtu -> PNG (plan layer 6 / plate M8).")
    parser.add_argument("--warp", type=float, default=None,
                        help="displacement exaggeration factor "
                             "(default: auto for the plate, x5 for the beam)")
    args = parser.parse_args()
    summary = run_postprocess(warp_factor=args.warp)
    return 0 if report_and_check(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
