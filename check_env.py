#!/usr/bin/env python3
"""One-time environment check for the cantilever FEA MVP toolchain.

Verifies every dependency in the deterministic chain:
    mesh -> inp -> solve -> physics validation -> postprocess -> verification

This is an environment gate, not a pipeline module. Every check is
independent and non-fatal: a missing item is reported, never raised, so the
script always runs to completion and prints a full summary.

Usage:
    python3 check_env.py
"""

import glob
import importlib
import platform
import shutil
import subprocess
import sys

PASS = "PASS"
MISSING = "MISSING"
WARN = "WARN"

results = []


def record(name, status, detail=""):
    results.append((name, status, detail))
    print(f"[{status:>7}] {name:<18} {detail}")


def check_pkg(modname, label=None):
    """Import a Python package and report its version."""
    label = label or modname
    try:
        module = importlib.import_module(modname)
        version = getattr(module, "__version__", "?")
        record(label, PASS, f"version {version}")
        return module
    except Exception as exc:  # noqa: BLE001 - report any import failure
        record(label, MISSING, f"pip install {label}  ({exc.__class__.__name__})")
        return None


print("=" * 70)
print("Cantilever FEA MVP - environment check")
print("=" * 70)

# 1. Platform / Python interpreter (arm64 native expected on Apple silicon).
arch = platform.machine()
record(
    "platform",
    PASS if arch == "arm64" else WARN,
    f"{arch} / Python {platform.python_version()} / {sys.executable}",
)

# 2. Python packages used across the chain.
check_pkg("numpy")
gmsh = check_pkg("gmsh")
check_pkg("pyvista")
check_pkg("ccx2paraview")
check_pkg("pydantic")  # parameterization layer (used later, checked now)

# 3. Gmsh runtime: initialize/finalize must succeed, not just import.
if gmsh is not None:
    try:
        gmsh.initialize()
        version = gmsh.option.getString("General.Version")
        gmsh.finalize()
        record("gmsh runtime", PASS, f"initialize() ok, Gmsh {version}")
    except Exception as exc:  # noqa: BLE001
        record("gmsh runtime", MISSING, f"{exc.__class__.__name__}: {exc}")

# 4. CalculiX ccx solver on PATH.
ccx_path = shutil.which("ccx")
if ccx_path:
    version_line = "(found; version banner only printed during a solve)"
    try:
        proc = subprocess.run(
            [ccx_path], input="", capture_output=True, text=True, timeout=5
        )
        for line in (proc.stdout + proc.stderr).splitlines():
            if "ersion" in line:  # matches "Version" / "version"
                version_line = line.strip()
                break
    except Exception:  # noqa: BLE001 - probe is best-effort only
        pass
    record("ccx", PASS, f"{ccx_path}  {version_line}")
else:
    record(
        "ccx",
        MISSING,
        "conda install -c conda-forge calculix  OR  "
        "brew install costerwi/homebrew-calculix/calculix-ccx",
    )

# 5. ccx2paraview command-line entry point (module import already checked above).
ccx2pv_cli = shutil.which("ccx2paraview")
record(
    "ccx2paraview cli",
    PASS if ccx2pv_cli else WARN,
    ccx2pv_cli or "CLI not on PATH (the Python module may still be importable)",
)

# 6. PyVista off-screen render - de-risks the postprocess PNG stage early.
try:
    import pyvista as pv

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv.Sphere())
    image = plotter.screenshot(return_img=True)
    plotter.close()
    record("pyvista offscreen", PASS, f"headless render ok, image {image.shape}")
except Exception as exc:  # noqa: BLE001
    record("pyvista offscreen", WARN, f"headless render failed: {exc.__class__.__name__}: {exc}")

# 7. ParaView GUI app - optional human postprocess path.
paraview_app = glob.glob("/Applications/ParaView*.app")
record(
    "ParaView.app",
    PASS if paraview_app else WARN,
    paraview_app[0] if paraview_app else "not in /Applications (optional human path)",
)

# Summary: only true MISSING items block the chain; WARN items are non-blocking.
print("-" * 70)
missing = [name for name, status, _ in results if status == MISSING]
if missing:
    print(f"RESULT: blocking dependencies MISSING -> {', '.join(missing)}")
else:
    print("RESULT: all blocking dependencies present (WARN items are optional)")
print("=" * 70)
