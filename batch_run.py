#!/usr/bin/env python3
"""Batch runner (plate plan M10): sample a design space, run the deterministic
chain per case, and record every gate verdict in one manifest.

The pipeline's job here is to be a TRUSTWORTHY DATA GENERATOR for the stress-
field surrogate (M11/M12). Nothing about the iron laws changes: each case still
has to clear the same mesh / physics / verification gates it would clear in an
interactive run, and the manifest records what each gate said so a later export
can select on it rather than trusting the batch blindly.

Two deliberate omissions relative to an interactive run, both measured:
  * RAG retrieval is OFF (`use_rag=False`). Hundreds of cases of one geometry
    produce at most a couple of distinct retrieval queries, so paying an
    embedding API call each (1.4 s + one call) buys nothing; the deterministic
    applies() gate picks the formula either way. Verdicts are identical.
  * PNG rendering is OFF by default (--render turns it on). It is ~65% of the
    per-case wall time and the surrogate consumes fields, not pictures. Keep it
    on for a small pilot, where a human actually looks at every case.

Sampling (plate plan M10, settled 2026-08-28): the plate size is FIXED, so the
free parameters are the hole's radius and position. Dimensional analysis is why
that is not a loss: normalising stress by sigma_gross and coordinates by the
plate leaves the field dependent only on the ratios r/b, xc/a, yc/b and the
aspect ratio a/b -- fixing a x b freezes exactly one of those. F and t are free
rescalings (sigma ~ F/t), so they are not sampled either.

CONDITIONAL Latin-hypercube sampling, not box sampling: the feasible region is
a wedge, since a bigger hole has less room to move (the ligament constraint
shrinks the allowed xc/yc band as r grows). Sampling a bounding box and
rejecting whatever falls outside would throw away a third of the budget AND
bias the survivors toward small holes -- exactly the samples that matter least.
Instead r is stratified directly, and (xc, yc) are stratified in NORMALISED
[0,1] coordinates that are then mapped into that r's own feasible box.

Unit system: SI-mm.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import build_inp
import export_field
import mesh
import physics_validation
import postprocess
import verification
from boundary_conditions import build_bc
from params import CASES

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DEFAULT_ROOT = OUT / "cases"

# Design space (plate plan M10). The plate itself is fixed; only the hole moves.
R_RANGE = (8.0, 30.0)          # mm, hole radius
LIGAMENT_MARGIN_FRAC = 0.08    # of the plate width b
LIGAMENT_MARGIN_MESH = 3.0     # x elem_size -- the mesh must resolve the ligament

# Train / validation / test / out-of-distribution assignment (plate plan M12).
# The split is decided HERE, at generation time, and written into the manifest,
# because an out-of-distribution set that has been used to tune anything is not
# out of distribution any more. A training script may read this column; it must
# never compute its own split.
#
# Two kinds of held-out data, testing two different abilities:
#   ood_large_r  -- IN-family extrapolation. Every hole with r above the cut is
#                   removed from training, so the model must extrapolate in the
#                   one parameter it did see varying.
#   ood_aspect   -- OUT-of-family distribution shift, produced by a separate run
#                   with --plate: a shorter plate the model has never seen. This
#                   probes the single ratio (a/b) that dimensional analysis let
#                   us freeze, so it measures what freezing it actually cost.
OOD_LARGE_R = 26.0
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Files worth keeping per case. The .frd/.msh pair is what the M11 field
# exporter reads, so a pilot keeps them and a large production run can drop
# them once the export happens in the same pass.
RAW_ARTIFACTS = ("beam.msh", "beam.inp", "beam.frd")
FIELDS_NPZ = "fields.npz"
RENDER_ARTIFACTS = ("mesh.png", "stress_xx.png", "stress_yy.png", "stress_xy.png",
                    "stress_vm.png", "displacement.png")


def ligament_margin(b: float, elem_size: float) -> float:
    """Smallest material width allowed between the hole and any plate edge."""
    return max(LIGAMENT_MARGIN_FRAC * b, LIGAMENT_MARGIN_MESH * elem_size)


def feasible_box(r: float, a: float, b: float, margin: float) -> tuple:
    """The (xc, yc) rectangle a hole of radius r may sit in."""
    return ((r + margin, a - r - margin), (r + margin, b - r - margin))


def _lhs(n: int, rng) -> np.ndarray:
    """One stratified draw per bin of [0,1], shuffled (Latin hypercube, 1D)."""
    edges = np.linspace(0.0, 1.0, n + 1)
    pts = edges[:-1] + rng.random(n) * (edges[1:] - edges[:-1])
    return rng.permutation(pts)


def anchor_design(n: int, a: float, b: float) -> list:
    """Deliberately CENTRED holes, radii spread evenly over R_RANGE.

    Control samples for the batch. A randomly sampled hole is almost never
    centred, so the SCF reference does not apply to it and its verdict is an
    honest N/A -- which leaves a production batch with no analytical
    cross-check at all. Anchors are the cases that DO have one: if their Kt
    stays within tolerance of Howland, the mesh policy, load distribution and
    solver settings that produced the whole batch are still sound; if an anchor
    drifts, something regressed and the rest of the batch is suspect.
    """
    return [{"r": float(r), "xc": a / 2.0, "yc": b / 2.0, "is_anchor": True}
            for r in np.linspace(R_RANGE[0], R_RANGE[1], n)]


def sample_design(n: int, seed: int, a: float, b: float, elem_size: float) -> list:
    """Conditional LHS over (r, xc, yc). Every sample is feasible by construction."""
    rng = np.random.default_rng(seed)
    margin = ligament_margin(b, elem_size)
    r = R_RANGE[0] + _lhs(n, rng) * (R_RANGE[1] - R_RANGE[0])
    u, v = _lhs(n, rng), _lhs(n, rng)          # normalised position inside the wedge
    out = []
    for i in range(n):
        (xlo, xhi), (ylo, yhi) = feasible_box(r[i], a, b, margin)
        if xhi <= xlo or yhi <= ylo:
            raise ValueError(f"r={r[i]:.2f} mm leaves no feasible hole position "
                             f"(margin {margin:.1f} mm); narrow R_RANGE")
        out.append({"r": float(r[i]),
                    "xc": float(xlo + u[i] * (xhi - xlo)),
                    "yc": float(ylo + v[i] * (yhi - ylo)),
                    "is_anchor": False})
    return out


def make_params(sample: dict, plate: tuple = None) -> dict:
    """A full PARAMS dict for one sampled hole, on the prepared plate case."""
    p = copy.deepcopy(CASES["plate_with_hole"])
    if plate is not None:
        p["geometry"]["plate_with_hole"].update(a=float(plate[0]), b=float(plate[1]))
    p["geometry"]["plate_with_hole"].update(
        r=sample["r"], xc=sample["xc"], yc=sample["yc"])
    return p


def assign_splits(design: list, seed: int, ood_tag: str = None) -> None:
    """Write a `split` onto every sample, in place (see SPLIT_FRACTIONS)."""
    if ood_tag:                                  # a whole run that is held out
        for s in design:
            s["split"] = ood_tag
        return
    rng = np.random.default_rng(seed + 1)         # independent of the design draw
    pool = []
    for i, s in enumerate(design):
        if s["r"] > OOD_LARGE_R:
            s["split"] = "ood_large_r"
        else:
            pool.append(i)
    order = rng.permutation(pool)
    n_train = int(round(SPLIT_FRACTIONS["train"] * len(order)))
    n_val = int(round(SPLIT_FRACTIONS["val"] * len(order)))
    for k, i in enumerate(order):
        design[i]["split"] = ("train" if k < n_train
                              else "val" if k < n_train + n_val else "test")


def _quiet(fn, *args) -> tuple:
    """Call a report_and_check-style function, capturing its printed report."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = fn(*args)
    return bool(ok), buf.getvalue()


def run_case(params: dict, case_dir: Path, render: bool = False,
             keep_raw: bool = True, is_anchor: bool = False,
             split: str = "", export: bool = False) -> dict:
    """Run one case through the deterministic chain. Never raises: a broken case
    is DATA (that is the point of a pilot), so failures are recorded and the
    batch continues."""
    g = params["geometry"]["plate_with_hole"]
    row = {"is_anchor": is_anchor, "split": split,
           "r": round(g["r"], 4), "xc": round(g["xc"], 4), "yc": round(g["yc"], 4),
           "a": g["a"], "b": g["b"], "t": g["t"],
           "F": params["bc"]["loads"][0]["magnitude"],
           "lam_2r_over_b": round(2 * g["r"] / g["b"], 4),
           "min_ligament": round(min(g["xc"] - g["r"], g["a"] - g["xc"] - g["r"],
                                     g["yc"] - g["r"], g["b"] - g["yc"] - g["r"]), 3),
           "status": "ok", "failed_stage": "", "error": ""}
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))
    t0 = time.time()

    try:
        # --- 3.1 mesh ---
        msh = mesh.build_mesh(params)
        ok, rep = _quiet(mesh.report_and_check, msh)
        (case_dir / "mesh_report.txt").write_text(rep)
        row.update(n_nodes=msh["n_nodes"], n_elems=msh["n_surf_elems"],
                   hole_segments=msh["n_hole_segments"],
                   min_quality=round(msh["min_quality"], 4),
                   area_err_pct=round(msh["area_rel_err"] * 100, 4), mesh_ok=ok)
        if not ok:
            raise RuntimeError("mesh gates failed")

        # --- 3.2 boundary conditions ---
        bc = build_bc(params)
        ok, rep = _quiet(__import__("boundary_conditions").report_and_check, bc)
        (case_dir / "bc_report.txt").write_text(rep)
        row.update(n_constraints=bc["fixed"]["n_constraints"],
                   rigid_ok=bc["rigid"]["ok"], bc_ok=ok)
        if not ok:
            raise RuntimeError("bc gates failed")

        # --- 3.4 deck ---
        inp = build_inp.build_inp(params)
        ok, rep = _quiet(build_inp.report_and_check, inp)
        (case_dir / "inp_report.txt").write_text(rep)
        v = inp["verify"]
        row.update(max_chord_dev=f"{v['max_chord_dev']:.2e}",
                   max_arc_dev=f"{v['max_arc_dev']:.2e}",
                   min_signed_area=round(v["min_signed_area"], 4), inp_ok=ok)
        if not ok:
            raise RuntimeError("inp gates failed")

        # --- 4 solve ---
        sp = subprocess.run(["bash", str(HERE / "solve.sh")],
                            capture_output=True, text=True, cwd=str(HERE))
        row["ccx_rc"] = sp.returncode
        row["solve_ok"] = sp.returncode == 0
        if sp.returncode != 0:
            (case_dir / "solve_log.txt").write_text(sp.stdout + sp.stderr)
            raise RuntimeError("ccx failed")

        # --- 5 physics validation (the fast-fail gate) ---
        pok, checks, pinfo = physics_validation.run_physics_validation(params)
        (case_dir / "physics_report.txt").write_text(
            physics_validation._format_report(pok, checks, pinfo))
        row.update(
            physics_ok=pok,
            # Recorded metric values, not just verdicts: thresholds stay
            # re-tunable from the manifest without re-solving anything.
            hole_traction_rel=round(pinfo.get("hole_traction_rel", float("nan")), 5),
            hole_traction_abs=round(pinfo.get("hole_traction_abs", float("nan")), 5),
            sigma_peak_hole=round(pinfo.get("sigma_peak_hole", float("nan")), 4),
            sigma_net=round(pinfo.get("sigma_net", float("nan")), 4),
            sigma_gross=round(pinfo.get("sigma_gross", float("nan")), 4),
            physics_pass=sum(1 for c in checks if c.status == "PASS"),
            physics_total=len(checks),
            physics_critical_fail=";".join(c.name for c in checks
                                           if c.level == "CRITICAL" and c.status == "FAIL"),
            physics_warnings=";".join(c.name for c in checks if c.status == "WARNING"))
        if not pok:
            raise RuntimeError("physics gates failed")

        # --- 6 postprocess (optional: pictures are for humans, not the surrogate) ---
        if render:
            post = postprocess.run_postprocess(params=params)
            ok, rep = _quiet(postprocess.report_and_check, post)
            (case_dir / "postprocess_report.txt").write_text(rep)
            row["postprocess_ok"] = ok

        # --- 7 verification (retrieval off; the applies() gate is unchanged) ---
        vok, judged, recorded, meta = verification.run_verification(params, use_rag=False)
        (case_dir / "verification_report.txt").write_text(
            verification._format_report(vok, judged, recorded, meta))
        row.update(
            verify_verdict=judged["status"],
            kt_fem=round(judged["kt_fem"], 5),
            kt_ref=round(judged["kt_ref"], 5) if judged.get("kt_ref") is not None else "",
            kt_err_pct=round(judged["error"] * 100, 4) if judged.get("error") is not None else "",
            max_vm=round(recorded["max_vm_global"], 4),
            mean_elongation=round(recorded["mean_elongation"], 6),
            theta_peak=round(judged["theta_peak"], 1))
        if judged["status"] == "FAIL":
            raise RuntimeError("verification FAILED (Kt outside tolerance)")

        # --- field export: the surrogate's actual input. Runs in the SAME pass
        # as the solve, so no raw .frd has to be hoarded for 500 cases.
        if export:
            ex = export_field.export_case(params, npz_path=case_dir / FIELDS_NPZ)
            row.update(npz=FIELDS_NPZ,
                       grid_ny=ex["shape"][0], grid_nx=ex["shape"][1],
                       material_px=ex["n_material_px"],
                       vm_max_grid=round(ex["vm_max_grid"], 4),
                       peak_loss_pct=round(ex["peak_loss_pct"], 3))

    except Exception as exc:  # noqa: BLE001 - a failed case is data, not a crash
        row["status"] = "failed"
        row["error"] = f"{exc.__class__.__name__}: {exc}"
        row["failed_stage"] = _last_stage(row)

    row["seconds"] = round(time.time() - t0, 2)
    # A case is usable training data only if every gate it reached said PASS.
    row["gates_all_pass"] = (row["status"] == "ok")
    for name in (RAW_ARTIFACTS if keep_raw else ()) + (RENDER_ARTIFACTS if render else ()):
        src = OUT / name
        if src.exists():
            shutil.copy(src, case_dir / name)
    return row


def _last_stage(row: dict) -> str:
    """Name the first gate that did not pass (or the stage that raised)."""
    for key, stage in (("mesh_ok", "mesh"), ("bc_ok", "bc"), ("inp_ok", "inp"),
                       ("solve_ok", "solve"), ("physics_ok", "physics")):
        if key in row and not row[key]:
            return stage
        if key not in row:
            return stage
    if row.get("verify_verdict") == "FAIL":
        return "verification"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch the plate design space (plate plan M10).")
    ap.add_argument("-n", type=int, default=20, help="number of cases (default 20 = the pilot)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible sampling)")
    ap.add_argument("--anchors", type=int, default=5,
                    help="centred control cases prepended to the batch; these are the only "
                         "ones the SCF reference applies to, so they audit the whole run")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="output root for per-case folders")
    ap.add_argument("--render", action="store_true",
                    help="also render the PNGs (~65%% of the per-case time; use for a pilot)")
    ap.add_argument("--no-keep-raw", action="store_true",
                    help="do not copy beam.msh/inp/frd into each case folder")
    ap.add_argument("--export", action="store_true",
                    help="also write fields.npz per case (the surrogate's tensors)")
    ap.add_argument("--plate", default=None, metavar="AxB",
                    help="override the plate size, e.g. 150x100. Marks the whole run as "
                         "an out-of-distribution set (a different aspect ratio the model "
                         "never trains on)")
    ap.add_argument("--dry-run", action="store_true", help="print the sampled design only")
    args = ap.parse_args()

    base = CASES["plate_with_hole"]
    g, elem_size = base["geometry"]["plate_with_hole"], base["mesh"]["elem_size"]
    plate = None
    if args.plate:
        plate = tuple(float(v) for v in args.plate.lower().split("x"))
        if len(plate) != 2:
            ap.error("--plate wants AxB, e.g. 150x100")
    pa, pb = (plate if plate else (g["a"], g["b"]))
    design = (anchor_design(args.anchors, pa, pb)
              + sample_design(args.n, args.seed, pa, pb, elem_size))
    assign_splits(design, args.seed, ood_tag=("ood_aspect" if plate else None))

    print("=" * 72)
    print(f"batch_run.py -- {args.n} cases, seed {args.seed}")
    print(f"  plate fixed at a={pa} b={pb} t={g['t']} mm, F={base['bc']['loads'][0]['magnitude']} N"
          + ("   [OOD: aspect ratio the model never trains on]" if plate else ""))
    print(f"  sampled: r in {R_RANGE} mm, hole centre by conditional LHS")
    print(f"  ligament margin: {ligament_margin(g['b'], elem_size):.1f} mm "
          f"(max of {LIGAMENT_MARGIN_FRAC:g}*b and {LIGAMENT_MARGIN_MESH:g}*elem_size)")
    print(f"  {args.anchors} anchor cases (centred, Kt auto-judged) + {args.n} sampled cases")
    print(f"  RAG: off   PNG renders: {'on' if args.render else 'off'}   "
          f"field export: {'on' if args.export else 'off'}")
    counts = {}
    for sm in design:
        counts[sm["split"]] = counts.get(sm["split"], 0) + 1
    print(f"  splits: {', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}")
    print("=" * 72)
    if args.dry_run:
        print(f"{'case':>6} {'kind':>7} {'split':>13} {'r':>7} {'xc':>8} {'yc':>8} {'2r/b':>6}")
        for i, s in enumerate(design, 1):
            kind = "anchor" if s["is_anchor"] else "sample"
            print(f"{i:6d} {kind:>7} {s['split']:>13} {s['r']:7.2f} {s['xc']:8.2f} "
                  f"{s['yc']:8.2f} {2*s['r']/pb:6.2f}")
        return 0

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, s in enumerate(design, 1):
        params = make_params(s, plate=plate)
        case_id = f"case_{i:04d}"
        row = {"case_id": case_id}
        row.update(run_case(params, root / case_id, render=args.render,
                            keep_raw=not args.no_keep_raw, is_anchor=s["is_anchor"],
                            split=s["split"], export=args.export))
        rows.append(row)
        flag = "ok " if row["status"] == "ok" else "FAIL"
        if row["status"] != "ok":
            extra = f"{row['failed_stage']}: {row['error'][:44]}"
        elif row.get("verify_verdict") == "PASS":
            extra = f"Kt {row['kt_fem']} vs ref {row['kt_ref']}  ({row['kt_err_pct']}%)  PASS"
        else:
            extra = f"Kt {row.get('kt_fem', '-')}  (N/A)"
        kind = "A" if s["is_anchor"] else " "
        print(f"[{i:3d}/{len(design)}]{kind} {case_id}  r={s['r']:5.2f} "
              f"({s['xc']:6.1f},{s['yc']:5.1f})  {flag}  {row['seconds']:5.2f}s  {extra}")

    cols = sorted({k for r in rows for k in r}, key=lambda k: (k != "case_id", k))
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_na = sum(1 for r in rows if r.get("verify_verdict") == "N/A")
    anchors = [r for r in rows if r["is_anchor"]]
    print("=" * 72)
    print(f"{n_ok}/{len(rows)} cases passed every gate   ({time.time() - t0:.1f}s total, "
          f"{(time.time() - t0) / max(len(rows), 1):.2f}s per case)")
    print(f"  Kt auto-judged: {len(rows) - n_na} case(s) ; N/A (no applicable reference): {n_na}")
    if anchors:
        errs = [float(r["kt_err_pct"]) for r in anchors if r.get("kt_err_pct") != ""]
        bad = [r for r in anchors if r.get("verify_verdict") != "PASS"]
        print(f"  ANCHOR AUDIT: {len(anchors) - len(bad)}/{len(anchors)} within tolerance"
              + (f", worst Kt error {max(errs):.2f}%" if errs else ""))
        for r in anchors:
            print(f"    {r['case_id']}  r={r['r']:5.2f}  2r/b={r['lam_2r_over_b']:.2f}  "
                  f"Kt {r.get('kt_fem','-')} vs {r.get('kt_ref','-')}  "
                  f"{r.get('kt_err_pct','-')}%  {r.get('verify_verdict','-')}")
    if n_ok < len(rows):
        print("  failures:")
        for r in rows:
            if r["status"] != "ok":
                print(f"    {r['case_id']}  stage={r['failed_stage']}  {r['error'][:70]}")
    print(f"manifest: {manifest}")
    print("=" * 72)
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
