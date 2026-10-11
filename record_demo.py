#!/usr/bin/env python3
"""Record example runs into demo/ so the deployed app can run in DEMO_MODE with
no API key and no solver -- it replays these real, pre-computed runs.

Each example runs the full pipeline once (needs the fea env + API key), then its
run record (run.json) and artifact images are snapshotted into demo/<name>/.
Run order leaves out/ holding the standard passing cantilever.

Usage:  python record_demo.py            (re-record all; needs the API key)
        python record_demo.py --refresh  (re-run the existing snapshots' FE analysis:
                                          renders, 3D view, stage times; no key, no LLM)
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import pipeline_run
import report
from paths import RunPaths

HERE = Path(__file__).resolve().parent
DEMO = HERE / "demo"

# Recorded last-wins for out/: keep the passing cantilever last.
EXAMPLES = {
    "stubby_fail": ("A steel cantilever beam 20 mm long, 10x10 mm square section, "
                    "fixed at one end, with a 1000 N downward load at the free end."),
    # Phrased the way someone would actually say it: a diameter rather than a
    # radius, kN rather than N, a material spelling the library has to
    # canonicalise, and no thickness or support description at all. The run
    # therefore demonstrates not only that the plate solves, but that the
    # language layer read it correctly and disclosed what it assumed.
    "plate_pass": ("An aluminium plate 240 by 120 mm with a 36 mm diameter hole "
                   "right in the middle, pulled with 4.5 kN."),
    "cantilever_pass": ("A steel cantilever beam 100 mm long, 10x10 mm square section, "
                        "fixed at one end, with a 1000 N downward load at the free end."),
}
def _artifact_names(rec: dict) -> list:
    """The image files THIS run actually referenced.

    Copying everything that happens to sit in out/ would drag the previous
    example's renders into this one's folder -- the two geometries write
    different filenames, so nothing overwrites and the strays just accumulate.
    Reading the names off the record keeps each snapshot to its own run.
    """
    names = []
    for stage in rec["stages"]:
        for a in stage.get("artifacts", []):
            if "path" in a:
                names.append(Path(a["path"]).name)
            names += [Path(v).name for v in a.get("views", {}).values()]
    return sorted(set(names))


def _rewrite_paths(rec: dict, name: str) -> None:
    """Point artifact paths at the demo snapshot copies (relative).

    The run's working directory is rewritten too: it is an absolute path on the
    recording machine (user name included), and demo/*/run.json is committed.
    """
    rec["workdir"] = f"demo/{name}"
    for s in rec["stages"]:
        for a in s.get("artifacts", []):
            if "path" in a:
                a["path"] = f"demo/{name}/{Path(a['path']).name}"
            if "views" in a:
                a["views"] = {k: f"demo/{name}/{Path(v).name}" for k, v in a["views"].items()}


def _scrub(obj):
    """Make every absolute path in the record relative, in every string.

    The stage reports print the files they read (`frd: /Users/<name>/.../beam.frd`),
    and demo/*/run.json is committed -- so without this the recording machine's
    user name and folder layout end up in the public repo. Paths under the repo
    become repo-relative (`out/runs/<id>/beam.frd`, still true to where the run
    wrote); anything else under the home directory falls back to `~/`.
    """
    if isinstance(obj, str):
        for old, new in ((f"{HERE}/", ""), (str(HERE), "."), (f"{Path.home()}/", "~/")):
            obj = obj.replace(old, new)
        return obj
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    return obj


def snapshot(name: str, description: str) -> None:
    rec = pipeline_run.run_pipeline(description, run_id=name)
    dst = DEMO / name
    if dst.exists():
        for stale in dst.glob("*.png"):      # a re-record must not keep old renders
            stale.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    run_dir = Path(rec["workdir"])          # wherever this run actually wrote
    for f in _artifact_names(rec):
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, dst / f)
    _rewrite_paths(rec, name)
    rec = _scrub(rec)
    (dst / "run.json").write_text(json.dumps(rec, indent=2, default=str))
    print(f"recorded {name:<16} status={rec['status']:<9} verdict={rec['verdict']:<5} "
          f"stages={[s['status'] for s in rec['stages']]}")


def refresh(name: str) -> None:
    """Re-run an existing snapshot's FE analysis without re-recording it.

    Re-recording would call the language layers again and replace the recorded
    conversation. Instead the deterministic chain (pre-process -> solve ->
    physics -> post-process -> verify, no LLM) is re-run on the snapshot's own
    validated parameters, and the result is accepted only if it reproduces the
    recorded maxima and verdict -- so the new stages are shown to be of the
    same solution. Those five stages then replace the recorded ones, with their
    renders, the 3D view data and the time each took on this machine. The
    conversation, the two language-model stages and a failed run's diagnosis
    stay as recorded.
    """
    dst = DEMO / name
    rec = json.loads((dst / "run.json").read_text())
    old = {s["id"]: s for s in rec["stages"]}
    with tempfile.TemporaryDirectory() as tmp:
        paths = RunPaths(Path(tmp))
        new = pipeline_run.run_pipeline(rec["description"], run_id=name, params=report.recorded_params(rec),
                                        paths=paths, diagnose=False)
        fresh = {s["id"]: s for s in new["stages"] if s["id"] in report.CHAIN}
        for key in ("max_disp", "max_vm"):
            old_v, new_v = old["postprocess"]["data"][key], fresh["postprocess"]["data"][key]
            if not np.isclose(new_v, old_v, rtol=1e-6, atol=0):
                raise SystemExit(f"{name}: {key} {new_v!r} does not reproduce the recorded "
                                 f"{old_v!r}; snapshot left unchanged")
        if any(a["status"] != "pass" for a in fresh["postprocess"]["actions"]):
            raise SystemExit(f"{name}: rendering failed; snapshot left unchanged")
        if new["verdict"] != rec["verdict"]:
            raise SystemExit(f"{name}: verdict {new['verdict']} does not reproduce the recorded "
                             f"{rec['verdict']}; snapshot left unchanged")
        files = _artifact_names({"stages": list(fresh.values())})
        for stale in dst.glob("*.png"):
            stale.unlink()
        for f in files:
            shutil.copy(paths.root / f, dst / f)
        # The new reports name the throwaway run directory; a recorded run
        # wrote to out/, so say that.
        fresh = json.loads(json.dumps(fresh).replace(str(paths.root), "out"))
    rec["stages"] = [fresh.get(s["id"], s) for s in rec["stages"]]
    for key in ("title", "warnings", "meta"):
        rec[key] = new[key]
    _rewrite_paths(rec, name)
    (dst / "run.json").write_text(json.dumps(_scrub(rec), indent=2, default=str))
    times = " · ".join(f"{s['id']} {s['seconds']:.2f} s" for s in fresh.values())
    print(f"refreshed {name:<16} {len(files)} files · maxima and verdict reproduce · {times}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Record (or refresh) the dashboard's demo snapshots.")
    ap.add_argument("--refresh", action="store_true",
                    help="re-run the existing snapshots' FE analysis: renders, 3D view data, "
                         "stage times (no LLM call)")
    args = ap.parse_args()
    DEMO.mkdir(exist_ok=True)
    if args.refresh:
        for name in EXAMPLES:
            refresh(name)
        return 0
    for name, description in EXAMPLES.items():
        snapshot(name, description)
    print(f"\nsnapshots in {DEMO}/ — set DEMO_MODE=1 to serve them without a key or solver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
