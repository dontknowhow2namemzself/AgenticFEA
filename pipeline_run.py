#!/usr/bin/env python3
"""Run the full agentic pipeline in-process and return a structured run record.

The dashboard (app.py) and the demo recorder (record_demo.py) both consume the
same `run_record`. Each layer's STRUCTURED RETURN (not parsed text) feeds:
  - actions   -> the left-column agent-style sub-step flow
  - data      -> right-column key facts
  - artifacts -> right-column visuals / files (mesh image, PNG triplet, .inp, ...)
  - report    -> the verbatim acceptance report, tucked into an expander

Stages map to the eight layers (diagnostics appears on failure). The number is
always produced by deterministic code; the LLM only fills slots / selects / explains.
"""

from __future__ import annotations

import pprint
import subprocess
import time
from pathlib import Path

import build_inp
import diagnostics
import mesh
import parameterize
import physics_validation
import postprocess
import render_mesh
import task_understanding
import verification
from boundary_conditions import build_bc
from paths import RunPaths

HERE = Path(__file__).resolve().parent


def _read(p) -> str:
    p = Path(p)
    return p.read_text() if p.exists() else ""


def _inp_excerpt(paths: RunPaths) -> str:
    lines = _read(paths.inp).splitlines()
    if not lines:
        return ""
    out = lines[:7]                                   # header + *NODE + first nodes
    for i, line in enumerate(lines):
        if line.startswith("*ELEMENT"):
            out += ["   ..."] + lines[i:i + 3]
            break
    for i, line in enumerate(lines):
        if line.startswith("*MATERIAL"):
            out += ["   ..."] + lines[i:i + 14]       # material / section / step / loads / outputs
            break
    return "\n".join(out)


def _diagnose(stage: str, params: dict, paths: RunPaths) -> dict:
    try:
        d = diagnostics.diagnose(diagnostics.gather_context(stage, params=params, paths=paths))
        out = d.model_dump()
        out["report"] = diagnostics.format_report(stage, d)
        return out
    except Exception as exc:  # noqa: BLE001
        return {"failure_summary": f"diagnosis unavailable: {exc.__class__.__name__}",
                "failure_class": "unknown", "root_cause": "", "confidence": "low",
                "suggested_fix": "", "recommended_next_action": "", "report": ""}


STAGE_ORDER = ["understand", "parameterize", "preprocess", "solve", "physics", "postprocess", "verify"]


def run_pipeline(description: str, run_id: str = "run", warp: float = None, emit=None,
                 params: dict = None, paths: RunPaths = None) -> dict:
    """Run the pipeline and return the dashboard's run record.

    `params` skips layers 1-2 and runs the deterministic chain on a prepared
    parameter set. Two callers need that seam: a geometry whose natural-language
    front end is not taught yet (the plate -- see plate plan M13), and the batch
    runner (M10), which samples parameters directly and must never spend an LLM
    call per case.

    `paths` is the run's own directory. The dashboard passes a fresh one per run
    so that two runs at once cannot overwrite each other; left out, the run uses
    the shared out/ directory exactly as before.
    """
    paths = (paths or RunPaths.default()).ensure()
    emit = emit or (lambda *a: None)   # emit(stage_id, phase) for live progress
    rec = {"run_id": run_id, "description": description, "status": "complete", "verdict": "—",
           "stages": [], "diagnosis": None, "workdir": str(paths.root),
           "meta": {"solver": "CalculiX ccx · SPOOLES", "units": "SI-mm", "host": "M5 / 32GB arm64"}}

    def stage(**kw):
        kw.setdefault("actions", []); kw.setdefault("data", {})
        kw.setdefault("artifacts", []); kw.setdefault("report", "")
        rec["stages"].append(kw)
        emit(kw["id"], kw["status"])
        if kw["status"] == "pass":          # light up the next stage as "running"
            try:
                emit(STAGE_ORDER[STAGE_ORDER.index(kw["id"]) + 1], "running")
            except (ValueError, IndexError):
                pass

    emit("understand", "running")

    # 0 -- prepared-parameter path: layers 1-2 are supplied, not inferred.
    if params is not None:
        gtype = params["geometry"]["type"]
        for sid, label in (("understand", "Understand"), ("parameterize", "Parameterize")):
            stage(id=sid, label=label, status="pass", detail="supplied",
                  actions=[{"text": "Parameters supplied directly (no LLM call)",
                            "data": f"prepared case: {gtype}", "status": "pass"}],
                  data={"source": "prepared"},
                  artifacts=[{"kind": "code", "title": "params (prepared)",
                              "text": "PARAMS = " + pprint.pformat(params, sort_dicts=False,
                                                                   width=78)}] if sid == "parameterize" else [])
        return _run_deterministic_chain(params, rec, stage, warp, paths)

    # 1 -- task understanding
    t = time.time()
    und = task_understanding.run_task_understanding(description, auto=True)
    dt = round(time.time() - t, 1)
    if und["status"] != "complete":
        stage(id="understand", label="Understand", status="fail",
              detail=und.get("reason", "incomplete"),
              actions=[{"text": "Rejected: off-topic or incomplete",
                        "data": und.get("reason") or und.get("message", ""), "status": "fail"}])
        rec["status"], rec["verdict"] = und["status"], "—"
        return rec
    stage(id="understand", label="Understand", status="pass", detail=f"{dt}s",
          actions=[{"text": "Parsed the natural-language request", "data": "", "status": "pass"},
                   {"text": "Completeness check — must-haves present",
                    "data": "geometry · material · fixed · load", "status": "pass"},
                   {"text": f"Applied {len(und['assumptions'])} disclosed defaults",
                    "data": "; ".join(und["assumptions"][:2]), "status": "pass"}],
          data={"assumptions": und["assumptions"]},
          artifacts=[{"kind": "panel", "title": "completed definition", "text": und["completed_description"]},
                     {"kind": "list", "title": "assumptions applied", "items": und["assumptions"]}])

    # 2 -- parameterization
    t = time.time()
    pres = parameterize.run_set_params(und["completed_description"])
    params = pres["params"]; mat = params["material"]
    stage(id="parameterize", label="Parameterize", status="pass", detail="valid",
          actions=[{"text": "LLM filled the set_params tool", "data": f"valid on attempt {pres['attempts']}", "status": "pass"},
                   {"text": f"Material '{mat['name']}' → SI-mm library",
                    "data": f"E={mat['E']:g} · ν={mat['nu']} · ρ={mat['rho']:g}", "status": "pass"},
                   {"text": "Validated against schema (Pydantic)",
                    "data": "all slots in range" if not pres["warnings"] else f"{len(pres['warnings'])} warning(s)", "status": "pass"}],
          data={"material": mat["name"], "warnings": pres["warnings"]},
          artifacts=[{"kind": "code", "title": "params.py (validated)",
                      "text": "PARAMS = " + pprint.pformat(params, sort_dicts=False, width=78)}])

    return _run_deterministic_chain(params, rec, stage, warp, paths)


def _preprocess_actions(params, msh, bc, inp, n_elem, n_nodes) -> tuple:
    """Per-geometry wording + data for the Pre-Process stage card."""
    elem_size = params["mesh"]["elem_size"]
    if params["geometry"]["type"] == "plate_with_hole":
        v = inp["verify"]
        actions = [
            {"text": "Cut rectangle − disk · tagged 5 neutral edges",
             "data": "edge_x_min … hole_boundary", "status": "pass"},
            {"text": f"Meshed {inp['elem_type']} · {n_nodes} nodes",
             "data": f"elem_size {elem_size} mm · {msh['n_hole_segments']} segments around the hole",
             "status": "pass"},
            {"text": f"Quality minSICN = {msh['min_quality']:.3f}",
             "data": f"area matches a·b−πr² to {msh['area_rel_err'] * 100:.3f}%", "status": "pass"},
            {"text": f"BC mapped · {bc['fixed']['n_constraints']} (node,dof) · LOAD {bc['load']['n_nodes']}",
             "data": "roller + mid-edge anchor · rigid-body modes eliminated", "status": "pass"},
            {"text": "Gmsh→CalculiX 2D remap verified",
             "data": (f"midside dev {v['max_chord_dev']:.1e} mm · hole arc {v['max_arc_dev']:.1e} mm · "
                      f"all cells CCW"), "status": "pass"},
            {"text": "Wrote beam.inp (consistent edge loads)",
             "data": f"{inp['file_size'] // 1024} KB · thickness {inp['thickness']} mm · U/RF/S",
             "status": "pass"},
        ]
        data = {"nodes": n_nodes, "elements": n_elem, "type": inp["elem_type"],
                "minSICN": round(msh["min_quality"], 4),
                "hole_segments": msh["n_hole_segments"],
                "constraints": bc["fixed"]["n_constraints"], "load_nodes": bc["load"]["n_nodes"]}
        return actions, data
    actions = [
        {"text": "Built box · tagged 6 neutral faces", "data": "face_x_min … face_z_max", "status": "pass"},
        {"text": f"Meshed {n_elem}×C3D20R · {n_nodes} nodes", "data": f"elem_size {elem_size} mm", "status": "pass"},
        {"text": f"Quality minSICN = {msh['min_quality']:.3f}", "data": "no negative Jacobian", "status": "pass"},
        {"text": f"BC mapped · FIXED {bc['fixed']['n_nodes']} · LOAD {bc['load']['n_nodes']}",
         "data": f"load dir {bc['load']['loads'][0]['dir_unit']}", "status": "pass"},
        {"text": "Gmsh→CalculiX C3D20 remap verified",
         "data": f"max midside dev {inp['max_midside_dev']:.1e} mm", "status": "pass"},
        {"text": "Wrote beam.inp", "data": f"{inp['file_size'] // 1024} KB · U/RF/S requested", "status": "pass"},
    ]
    data = {"nodes": n_nodes, "elements": n_elem, "type": "C3D20R",
            "minSICN": round(msh["min_quality"], 4),
            "fixed_nodes": bc["fixed"]["n_nodes"], "load_nodes": bc["load"]["n_nodes"]}
    return actions, data


def _run_deterministic_chain(params: dict, rec: dict, stage, warp, paths: RunPaths) -> dict:
    """Layers 3-7: pre-process -> solve -> physics -> post -> verify, all inside
    one run directory."""
    is_plate = params["geometry"]["type"] == "plate_with_hole"

    # 3 -- pre-processing (mesh + BC + inp)
    msh = mesh.build_mesh(params, paths=paths)
    bc = build_bc(params, paths=paths)
    inp = build_inp.build_inp(params, paths=paths)
    n_elem, n_nodes = inp["n_elems"], inp["n_nodes"]
    rec["meta"]["n_elem"], rec["meta"]["n_nodes"] = n_elem, n_nodes
    try:
        render_mesh.render_mesh(paths)   # beam.inp now exists -> mesh.png (decoupled from solve)
    except Exception:  # noqa: BLE001
        pass
    pre_actions, pre_data = _preprocess_actions(params, msh, bc, inp, n_elem, n_nodes)
    stage(id="preprocess", label="Pre-Process", status="pass", detail=f"{n_elem} elem",
          actions=pre_actions, data=pre_data,
          artifacts=[{"kind": "image", "title": "mesh", "path": str(paths.mesh_png)},
                     {"kind": "code", "title": "beam.inp (excerpt)", "text": _inp_excerpt(paths)}])

    # 4 -- solve
    t = time.time()
    sp = subprocess.run(["bash", str(HERE / "solve.sh"), str(paths.root)],
                        capture_output=True, text=True)
    dt = round(time.time() - t, 1)
    log = _read(paths.ccx_log)
    solved = sp.returncode == 0 and "Job finished" in log
    stage(id="solve", label="Solve", status="pass" if solved else "fail", detail=f"{dt}s",
          actions=[{"text": "Ran CalculiX ccx", "data": "OMP_NUM_THREADS=4 (M5 P-cores)", "status": "pass" if solved else "fail"},
                   {"text": "Job finished · no *ERROR", "data": f"exit {sp.returncode}", "status": "pass" if solved else "fail"},
                   {"text": "Wrote beam.frd", "data": "U · RF · S blocks", "status": "pass" if solved else "fail"}],
          data={"threads": 4, "exit": sp.returncode, "seconds": dt},
          artifacts=[{"kind": "code", "title": "ccx.log (tail)", "text": "\n".join(log.splitlines()[-20:])}],
          report=log)
    if not solved:
        rec["status"], rec["verdict"] = "failed", "FAIL"
        rec["diagnosis"] = _diagnose("solve", params, paths)
        return rec

    # 5 -- physics validation (fast-fail gate)
    ppass, checks, pinfo = physics_validation.run_physics_validation(params, paths=paths)
    phys_text = physics_validation._format_report(ppass, checks, pinfo)
    paths.physics_report.write_text(phys_text)   # fresh report for diagnosis + expander
    n_pass = sum(1 for c in checks if c.status == "PASS")
    stage(id="physics", label="Physics", status="pass" if ppass else "fail", detail=f"{n_pass}/{len(checks)}",
          actions=[{"text": c.name.replace("_", " "), "data": c.message,
                    "status": c.status.lower(), "critical": c.level == "CRITICAL"} for c in checks],
          data={"passed": n_pass, "total": len(checks)},
          artifacts=[],     # left action-flow already lists the checks; full report
                            # is the right-panel expander -> no left/right duplication
          report=phys_text)
    if not ppass:
        rec["status"], rec["verdict"] = "failed", "FAIL"
        rec["diagnosis"] = _diagnose("physics", params, paths)
        return rec

    # 6 -- post-processing
    post = postprocess.run_postprocess(warp_factor=warp, params=params, paths=paths)
    if is_plate:
        views = {"vM": str(paths.plate_png("stress_vm")), "σxx": str(paths.plate_png("stress_xx")),
                 "σyy": str(paths.plate_png("stress_yy")), "σxy": str(paths.plate_png("stress_xy")),
                 "disp": str(paths.displacement_png)}
        render_note = ("Rendered σxx / σyy / σxy / vM (top view)",
                       f"diverging scale about zero · warp ×{post['warp_factor']:.0f} auto")
        detail = "5 fields"
    else:
        views = {"z": str(paths.beam_stress_png("z")), "y": str(paths.beam_stress_png("y")),
                 "x": str(paths.beam_stress_png("x")), "disp": str(paths.displacement_png)}
        render_note = ("Rendered von Mises X / Y / Z", "PyVista off-screen")
        detail = "3 views"
    stage(id="postprocess", label="Post-Process", status="pass", detail=detail,
          actions=[{"text": "ccx2paraview · beam.frd → beam.vtu", "data": f"{post['n_points']} points", "status": "pass"},
                   {"text": "Extracted field maxima",
                    "data": f"|U|_max = {post['max_disp']:.4f} mm · σ_vM = {post['max_vm']:.1f} MPa", "status": "pass"},
                   {"text": render_note[0], "data": render_note[1], "status": "pass"},
                   {"text": "Lossless check vs frd", "data": "max values match", "status": "pass"}],
          data={"max_disp": post["max_disp"], "max_vm": post["max_vm"], "fields": post["fields"]},
          artifacts=[{"kind": "views", "title": "field renders", "views": views}])

    # 7 -- numerical verification. Which quantity is judged depends on the
    # geometry (plate plan D3): the beam judges displacement, the plate judges
    # the stress concentration factor.
    vok, judged, recorded, meta = verification.run_verification(params, paths=paths)
    verify_text = verification._format_report(vok, judged, recorded, meta)
    paths.verification_report.write_text(verify_text)   # fresh report for diagnosis + expander
    verdict = judged["status"]; rec["verdict"] = verdict
    kb = judged["case"].id if judged.get("case") else None
    if is_plate:
        vdata = {"quantity": "Kt (stress concentration factor)",
                 "fem": judged.get("kt_fem"), "ref": judged.get("kt_ref"),
                 "max_vm": recorded["max_vm_global"],
                 "elongation": recorded["mean_elongation"]}
        judge_title = "stress-concentration verdict"
        default_tol = 0.03
    else:
        vdata = {"quantity": "tip deflection [mm]",
                 "fem": judged.get("delta_fem"), "ref": judged.get("delta_ref"),
                 "max_vm": recorded["max_vm"]}
        judge_title = "displacement verdict"
        default_tol = 0.05
    # delta_fem / delta_ref stay in the record: the dashboard's judge widget and
    # the recorded demo snapshots read those keys.
    vdata.update({
        "delta_fem": judged.get("delta_fem"), "delta_ref": judged.get("delta_ref"),
        "error_pct": round(judged["error"] * 100, 4) if judged.get("error") is not None else None,
        "tol_pct": round(judged.get("tol", default_tol) * 100, 2), "verdict": verdict,
        "kb_ref": kb, "provenance": judged.get("provenance")})
    stage(id="verify", label="Verify", status="pass" if verdict in ("PASS", "N/A") else "fail",
          detail=f'{vdata["error_pct"]}%' if vdata["error_pct"] is not None else verdict,
          actions=[{"text": "RAG retrieved reference candidates", "data": "ChromaDB · kind=reference", "status": "pass"},
                   {"text": f"applies() gate → {kb or 'no applicable reference'}",
                    "data": "deterministic match" if kb else "verdict N/A, nothing fabricated",
                    "status": "pass"},
                   {"text": "Computed error (deterministic code)", "data": "the LLM never invents the formula", "status": "pass"},
                   {"text": f"Judged {vdata['quantity']}: {verdict}",
                    "data": f'threshold ≤ {vdata["tol_pct"]}%',
                    "status": verdict.lower() if verdict != "N/A" else "pass"}],
          data=vdata,
          artifacts=[{"kind": "judge", "title": judge_title, "verify": vdata}],
          report=verify_text)
    if verdict == "FAIL":
        rec["status"] = "failed"
        rec["diagnosis"] = _diagnose("verification", params, paths)
    else:
        rec["status"] = "complete"
    return rec


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Run the pipeline; optionally write the run record as JSON.")
    ap.add_argument("description", nargs="?", default=(
        "A steel cantilever beam 100 mm long, 10x10 mm, fixed at one end, "
        "1000 N downward load at the free end."))
    ap.add_argument("--out", default=None, help="write the run-record JSON to this path (for the dashboard)")
    ap.add_argument("--run-id", default="live")
    ap.add_argument("--stream", action="store_true", help="print '@@ <stage> <phase>' progress markers")
    ap.add_argument("--case", default=None,
                    help="run a prepared case from params.CASES (skips the LLM layers), "
                         "e.g. --case plate_with_hole")
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--workdir", default=None, metavar="DIR",
                       help="run directory to write into (default: out/)")
    where.add_argument("--new-run", action="store_true",
                       help="create a fresh isolated directory under out/runs/ for this run")
    args = ap.parse_args()
    run_paths = RunPaths.new_run() if args.new_run else RunPaths.at(args.workdir)

    prepared = None
    if args.case:
        from params import CASES
        if args.case not in CASES:
            ap.error(f"unknown case {args.case!r}; available: {list(CASES)}")
        prepared = CASES[args.case]

    emit = (lambda sid, ph: print(f"@@ {sid} {ph}", flush=True)) if args.stream else None
    r = run_pipeline(args.description, run_id=args.run_id, emit=emit, params=prepared,
                     paths=run_paths)
    if args.out:
        Path(args.out).write_text(json.dumps(r, default=str))
        print(f"wrote {args.out}  (status={r['status']} verdict={r['verdict']})")
    else:
        print(f"status={r['status']} verdict={r['verdict']}   workdir={r['workdir']}")
        for s in r["stages"]:
            print(f"  [{s['status']:>5}] {s['label']:<13} {s['detail']}")
        if r["diagnosis"]:
            print("  diagnosis:", r["diagnosis"].get("failure_class"))
