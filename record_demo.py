#!/usr/bin/env python3
"""Record example runs into demo/ so the deployed app can run in DEMO_MODE with
no API key and no solver -- it replays these real, pre-computed runs.

Each example runs the full pipeline once (needs the fea env + API key), then its
run record (run.json) and artifact images are snapshotted into demo/<name>/.
Run order leaves out/ holding the standard passing cantilever.

Usage:  python record_demo.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pipeline_run

HERE = Path(__file__).resolve().parent
DEMO = HERE / "demo"
OUT = HERE / "out"

# Recorded last-wins for out/: keep the passing cantilever last.
EXAMPLES = {
    "stubby_fail": ("A steel cantilever beam 20 mm long, 10x10 mm square section, "
                    "fixed at one end, with a 1000 N downward load at the free end."),
    "cantilever_pass": ("A steel cantilever beam 100 mm long, 10x10 mm square section, "
                        "fixed at one end, with a 1000 N downward load at the free end."),
}
ARTIFACT_FILES = ["mesh.png", "stress_x.png", "stress_y.png", "stress_z.png", "displacement.png"]


def _rewrite_paths(rec: dict, name: str) -> None:
    """Point artifact paths at the demo snapshot copies (relative)."""
    for s in rec["stages"]:
        for a in s.get("artifacts", []):
            if "path" in a:
                a["path"] = f"demo/{name}/{Path(a['path']).name}"
            if "views" in a:
                a["views"] = {k: f"demo/{name}/{Path(v).name}" for k, v in a["views"].items()}


def snapshot(name: str, description: str) -> None:
    rec = pipeline_run.run_pipeline(description, run_id=name)
    dst = DEMO / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ARTIFACT_FILES:
        if (OUT / f).exists():
            shutil.copy(OUT / f, dst / f)
    _rewrite_paths(rec, name)
    (dst / "run.json").write_text(json.dumps(rec, indent=2, default=str))
    print(f"recorded {name:<16} status={rec['status']:<9} verdict={rec['verdict']:<5} "
          f"stages={[s['status'] for s in rec['stages']]}")


def main() -> int:
    DEMO.mkdir(exist_ok=True)
    for name, description in EXAMPLES.items():
        snapshot(name, description)
    print(f"\nsnapshots in {DEMO}/ — set DEMO_MODE=1 to serve them without a key or solver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
