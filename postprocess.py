#!/usr/bin/env python3
"""Postprocess layer 6 (plan section 6): .frd -> .vtu -> PNG renders.

Runs ONLY after physics validation (layer 5) passes; the orchestrator skips it
on a Critical FAIL. Two stages:
  (a) convert  : ccx2paraview out/beam.frd vtu   (PyVista cannot read .frd)
  (b) render   : PyVista off-screen von Mises contours from three camera
                 directions (stress_x/y/z.png, named by the axis looked along)
                 plus a deformed-shape displacement plot (displacement.png)

Max displacement and max von Mises are RECORDED for the summary, and cross-
checked against the .frd to confirm the conversion is lossless. Stress is
information only -- it is never auto-judged (iron law; see plan section 7).
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

from physics_validation import _von_mises, parse_frd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
FRD_PATH = OUT_DIR / "beam.frd"
VTU_PATH = OUT_DIR / "beam.vtu"
STRESS_PNG = {"x": OUT_DIR / "stress_x.png", "y": OUT_DIR / "stress_y.png", "z": OUT_DIR / "stress_z.png"}
DISP_PNG = OUT_DIR / "displacement.png"

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


def run_postprocess(frd_path=FRD_PATH, warp_factor=5.0) -> dict:
    """Convert, render, and cross-check. Returns a summary dict."""
    vtu_path = convert_frd_to_vtu(frd_path)
    mesh = pv.read(str(vtu_path))

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
        "vtu_path": str(vtu_path),
        "n_points": int(mesh.n_points), "n_cells": int(mesh.n_cells),
        "fields": list(mesh.point_data.keys()),
        "max_disp": max_disp, "ref_disp": ref_disp, "disp_ok": disp_ok,
        "max_vm": max_vm, "ref_vm": ref_vm, "vm_ok": vm_ok,
        "png_sizes": png_sizes, "render_error": render_error,
    }


def report_and_check(s: dict) -> bool:
    """Print the acceptance report (plan section 6) and return PASS (True)/FAIL."""
    print("=" * 64)
    print("postprocess.py acceptance report")
    print("=" * 64)
    print(f"vtu      : {s['vtu_path']}  ({s['n_points']} points, {s['n_cells']} cells)")
    print(f"fields   : {', '.join(s['fields'])}")
    print(f"max |U|  : {s['max_disp']:.4f} mm   (frd: {s['ref_disp']:.4f} mm)")
    print(f"max vM   : {s['max_vm']:.2f} MPa   (frd: {s['ref_vm']:.2f} MPa)  [recorded only, not judged]")
    print( "renders  :")
    for name, size in s["png_sizes"].items():
        print(f"           {name:<14}: {size / 1024:.1f} KB")
    if s["render_error"]:
        print(f"  !! render error: {s['render_error']}")
        print("     fallback: open out/beam.vtu in the ParaView GUI (human path).")

    pngs_ok = all(size >= MIN_PNG_BYTES for size in s["png_sizes"].values())
    checks = [
        ("vtu loads in PyVista (points > 0)", s["n_points"] > 0),
        ("max |U| matches frd (lossless conversion)", s["disp_ok"]),
        ("max von Mises matches frd (lossless conversion)", s["vm_ok"]),
        ("all 4 PNGs exist and are non-empty", pngs_ok),
    ]
    passed = all(ok for _, ok in checks)
    print("-" * 64)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 64)
    print(f"ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    if passed:
        print("Human check (optional): open out/beam.vtu in ParaView; confirm bend & stress pattern.")
        print("Next: numerical verification (layer 7) - displacement vs analytical, stress N/A.")
    print("=" * 64)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Postprocess frd -> vtu -> PNG (plan layer 6).")
    parser.add_argument("--warp", type=float, default=5.0, help="displacement exaggeration factor")
    args = parser.parse_args()
    summary = run_postprocess(warp_factor=args.warp)
    return 0 if report_and_check(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
