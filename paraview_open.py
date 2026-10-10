#!/usr/bin/env python3
"""Open a run in ParaView, set up the way the dashboard's 3D view shows it.

The browser view is for looking around; ParaView is for what a browser view
cannot do -- slice the solid, probe a point, look at the interior or at any
field the solver wrote. This module writes a small ParaView script for one run
and starts the ParaView GUI on it, so the model opens already showing what the
dashboard showed: von Mises on the real elements (quadratic edges curved, only
element edges drawn), the same rainbow and the same range on the bar, the same
opening camera, and the deformed shape at the same exaggeration one click away
in the Pipeline Browser.

Local only, on purpose: the button starts a program on the machine the
dashboard runs on, which only makes sense for the person sitting at it. The
dashboard therefore offers it only when
  * the browser is on this machine (loopback client),
  * ParaView is installed here, and
  * the run has its .vtu -- a replayed demo carries the 3D view data only.
The client check is a convenience, not a security boundary (a reverse proxy on
the same machine makes every client look local); the boundary is the loopback
bind in .streamlit/config.toml.

Usage:  python paraview_open.py [--workdir DIR]      open a run in ParaView
        python paraview_open.py --script-only ...    only write <run>/pv_view.py
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import viewer
from paths import RunPaths, add_workdir_argument, from_args

# ParaView's built-in "Turbo" is the colormap the dashboard samples (viewer.RAINBOW).
PRESET = "Turbo"


def find_paraview() -> Path | None:
    """The ParaView GUI executable: $PARAVIEW, then PATH, then the newest macOS app."""
    if os.environ.get("PARAVIEW"):
        exe = Path(os.environ["PARAVIEW"])
        return exe if exe.exists() else None
    if shutil.which("paraview"):
        return Path(shutil.which("paraview"))
    apps = sorted(Path("/Applications").glob("ParaView*.app"),
                  key=lambda a: [int(n) for n in re.findall(r"\d+", a.name)])
    exes = [a / "Contents" / "MacOS" / "paraview" for a in apps]
    return next((e for e in reversed(exes) if e.exists()), None)


def paraview_name(exe: Path) -> str:
    """'ParaView 6.1.1' from the app bundle's name, else just 'ParaView'."""
    m = re.search(r"ParaView-([\d.]+)\.app", str(exe))
    return f"ParaView {m.group(1)}" if m else "ParaView"


def is_local_client(ip: str | None) -> bool:
    """Streamlit reports no address for a localhost client, and loopback otherwise."""
    if ip is None:
        return True
    try:
        return ipaddress.ip_address(ip).is_loopback
    except ValueError:
        return False


def available(run_dir: Path, client_ip: str | None) -> Path | None:
    """The ParaView executable if the button makes sense for this run and client, else None."""
    paths = RunPaths(Path(run_dir))
    if not (is_local_client(client_ip) and paths.vtu.exists() and paths.viewer_npz.exists()):
        return None
    return find_paraview()


_SCRIPT = '''\
# Written by paraview_open.py for this run; ParaView runs it on start-up.
# The scene matches the dashboard's 3D view: same field, colours, range, camera.
from paraview.simple import *

VTU = {vtu!r}
WARP = {warp!r}               # the dashboard's deformed-shape exaggeration
VM_RANGE = {vm_range!r}       # von Mises min..max, as on the dashboard's fringe bar
DIRECTION, UP, PARALLEL = {direction!r}, {up!r}, {parallel!r}
EDGE_RGB, BACKGROUND = {edge!r}, {background!r}

view = GetActiveViewOrCreate("RenderView")
model = XMLUnstructuredGridReader(registrationName={name!r}, FileName=[VTU])
deformed = WarpByVector(registrationName="deformed x%.3g" % WARP, Input=model)
deformed.Vectors = ["POINTS", "U"]
deformed.ScaleFactor = WARP

for source in (model, deformed):
    shown = Show(source, view)
    shown.SetRepresentationType("Surface With Edges")
    shown.NonlinearSubdivisionLevel = 1    # quadratic cells drawn curved, real edges only
    shown.EdgeColor = EDGE_RGB
    ColorBy(shown, ("POINTS", "S_Mises"))
Hide(deformed, view)                       # one eye-click away in the Pipeline Browser
SetActiveSource(model)

lut = GetColorTransferFunction("S_Mises")
lut.ApplyPreset({preset!r}, True)
lut.AutomaticRescaleRangeMode = "Never"    # keep the dashboard's range
lut.RescaleTransferFunction(*VM_RANGE)
bar = GetScalarBar(lut, view)
bar.Title, bar.ComponentTitle = "von Mises", "[MPa]"
# ParaView 6.1 moved label formats from printf to std::format.
pv_version = (servermanager.vtkSMProxyManager.GetVersionMajor(),
              servermanager.vtkSMProxyManager.GetVersionMinor())
bar.RangeLabelFormat = "{{:.4g}}" if pv_version >= (6, 1) else "%.4g"
bar.Visibility = 1

view.UseColorPaletteForBackground = 0
view.BackgroundColorMode = "Single Color"
view.Background = BACKGROUND
view.OrientationAxesVisibility = 1
view.CameraParallelProjection = int(PARALLEL)
b = model.GetDataInformation().GetBounds()
centre = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
view.CameraFocalPoint = centre
view.CameraPosition = [c + d for c, d in zip(centre, DIRECTION)]
view.CameraViewUp = UP
view.ResetCamera(False)
Render()
'''


def write_script(paths: RunPaths) -> Path:
    """<run>/pv_view.py: the ParaView scene for this run, from its viewer.npz."""
    payload = viewer.load_payload(paths.viewer_npz)
    fr = viewer.fringe(viewer.field_values(payload, "vm"))
    direction, up, parallel = viewer.VIEWS["top" if int(payload["dim"]) == 2 else "iso"]
    rgb = lambda h: [round(int(h[i:i + 2], 16) / 255, 4) for i in (1, 3, 5)]
    paths.paraview_script.write_text(_SCRIPT.format(
        vtu=str(paths.vtu), warp=round(viewer.auto_warp_factor(payload), 6),
        vm_range=[fr["cmin"], fr["cmax"]], direction=list(direction), up=list(up),
        parallel=parallel, edge=rgb(viewer.EDGE_RGB), background=rgb(viewer.PANEL),
        name=paths.vtu.name, preset=PRESET))
    return paths.paraview_script


def launch(paths: RunPaths, exe: Path = None) -> subprocess.Popen:
    """Start the ParaView GUI on this run, detached from the dashboard.

    Its own session, so closing the dashboard does not close ParaView; its
    output goes to <run>/paraview.log. The command is a fixed list built here,
    never a shell string, and nothing in it comes from the browser.
    """
    exe = exe or find_paraview()
    if exe is None:
        raise FileNotFoundError("ParaView not found (set $PARAVIEW to its executable)")
    script = write_script(paths)
    with open(paths.paraview_log, "w") as log:
        return subprocess.Popen([str(exe), f"--script={script}"], cwd=paths.root,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a run in ParaView, set up like the dashboard's 3D view.")
    parser.add_argument("--script-only", action="store_true", help="write <run>/pv_view.py, do not start ParaView")
    add_workdir_argument(parser)
    args = parser.parse_args()
    paths = from_args(args)
    if not (paths.vtu.exists() and paths.viewer_npz.exists()):
        print(f"ERROR: {paths.root} has no beam.vtu / viewer.npz; run the pipeline first")
        return 1
    if args.script_only:
        print(f"script: {write_script(paths)}")
        return 0
    exe = find_paraview()
    proc = launch(paths, exe)
    print(f"{paraview_name(exe)} started (pid {proc.pid}) on {paths.vtu}; log: {paths.paraview_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
