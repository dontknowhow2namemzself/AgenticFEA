#!/usr/bin/env python3
"""Numerical verification layer 7 (plan section 7, plate plan M7): the
knowledge-base-driven judge.

Measures numerical ACCURACY (not physical plausibility): is the FEM result close
to a reference solution? WHICH quantity is auto-judged is decided per geometry,
by what that configuration's physics actually makes judgeable:

  cantilever_beam (plan section 7)
  * DISPLACEMENT is auto-judged against delta = P L^3 / (3 E I), tol_disp (5%):
    a global integral quantity, insensitive to local singularities.
  * STRESS is NOT auto-judged -- the fixed-end concentration DIVERGES under mesh
    refinement and the free-end point load is an artificial singularity, so
    there is no converged value to compare against. Recorded for the human.

  plate_with_hole (plate plan D3 -- deliberately the reverse)
  * STRESS is auto-judged, as the net-section stress concentration factor
    Kt = max von Mises on the hole / sigma_net, against the Howland/Peterson
    reference, tolerance tol_scf (8%). The hole-edge peak is smooth and
    CONVERGES (measured: 2.2395 / 2.2441 / 2.2471 at 32 / 64 / 126 hole
    segments), which is exactly what makes it judgeable.
  * DISPLACEMENT is NOT auto-judged -- a perforated plate's elongation has no
    closed form. Recorded for the human.

  Using von Mises for Kt needs no coordinate transform: a free surface in plane
  stress is uniaxial (sigma_rr = sigma_r_theta = 0), so von Mises = |sigma_theta|
  there.

IRON LAW: the formula comes from the knowledge base (an agent only SELECTS it);
the error and threshold are computed by deterministic code here. Selection is
RAG retrieval NARROWING candidates, then the deterministic applies() gate.

Unit system: SI-mm; stress concentration factors are dimensionless.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from boundary_conditions import build_bc
from knowledge.reference_solutions import select_reference_case
from params import PARAMS
from paths import RunPaths, add_workdir_argument, from_args
from physics_validation import _von_mises, parse_frd

STRESS_EXIT_REASONS = (
    "fixed-end stress concentration diverges under mesh refinement (not the section value Mc/I)",
    "free-end point-load application is an artificial numerical singularity",
    "stress error is not a meaningful convergence quantity in this configuration",
)


def _tip_deflection(frd: dict, load_nodes: list, load_dir_unit: np.ndarray) -> float:
    """FEM tip deflection = max |displacement . load_direction| over the free-end
    LOAD face (the deflection in the load direction; for a -z load this is |Uz|)."""
    vals = [abs(float(np.dot(frd["DISP"][n], load_dir_unit)))
            for n in load_nodes if n in frd["DISP"]]
    if not vals:
        raise RuntimeError("no displacement found at LOAD nodes")
    return max(vals)


def _reference_query(params: dict) -> str:
    """Plain-language description of the problem, for semantic retrieval (RAG)."""
    gtype = params["geometry"]["type"]
    if gtype == "plate_with_hole":
        g = params["geometry"]["plate_with_hole"]
        centered = (abs(float(g["xc"]) - float(g["a"]) / 2) <= 0.02 * float(g["a"])
                    and abs(float(g["yc"]) - float(g["b"]) / 2) <= 0.02 * float(g["b"]))
        where = "central" if centered else "eccentric"
        return (f"flat plate of finite width with a {where} circular hole, "
                f"uniaxial tension along its length, linear elastic plane stress, "
                f"peak stress at the hole edge, stress concentration factor")
    n_fixed = len(params["bc"]["fixed"])
    n_loads = len(params["bc"]["loads"])
    return (f"{gtype.replace('_', ' ')}, fixed at {n_fixed} end, {n_loads} transverse "
            f"concentrated point load at the free end, linear elastic small strain")


def select_reference_via_rag(params: dict, k: int = 3) -> tuple:
    """RAG selection: retrieve candidate references by semantic similarity, then
    the DETERMINISTIC applies() gate decides which candidate is actually valid.
    Degrades gracefully: a full deterministic scan backs it up if retrieval is
    unavailable (no chromadb / no key / empty index) or if the top-k yields no
    applicable case. Returns (case_or_None, provenance). The LLM/embedding only
    narrows candidates; the formula is always chosen by deterministic code."""
    query = _reference_query(params)
    try:
        from knowledge.reference_solutions import get_case
        from knowledge.vector_store import retrieve
        hits = retrieve(query, k=k, kind="reference")
    except Exception as exc:  # noqa: BLE001 - retrieval unavailable -> deterministic fallback
        return select_reference_case(params), f"deterministic scan (RAG unavailable: {exc.__class__.__name__})"

    for rank, hit in enumerate(hits, 1):
        case = get_case(hit["id"])
        if case is not None and case.applies(params):
            score = hit.get("score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            return case, (f"RAG retrieval (k={len(hits)}) then applies() gate -> "
                          f"'{hit['id']}' at rank {rank} (score {score_str})")

    case = select_reference_case(params)
    if case is not None:
        return case, f"deterministic backstop (RAG top-{len(hits)} had no applicable case)"
    return None, f"no applicable reference (RAG candidates: {[h['id'] for h in hits]})"


def _select_reference(params: dict, use_rag: bool = True) -> tuple:
    """Reference selection, with retrieval optionally switched off.

    A batch run re-verifies hundreds of cases of the SAME geometry, so the
    retrieval query takes only a couple of distinct values -- paying an
    embedding API call per case would be pure waste (measured: 1.4 s and one
    API call each, vs 0.025 ms for the deterministic scan). Skipping retrieval
    changes nothing about the verdict: the deterministic applies() gate is what
    picks the formula either way (the iron law), retrieval only ever narrowed
    the candidate list.
    """
    if use_rag:
        return select_reference_via_rag(params)
    return select_reference_case(params), "deterministic scan (retrieval disabled for batch)"


def run_verification(params: dict, paths: RunPaths = None, use_rag: bool = True) -> tuple:
    """Run the geometry's verification. Returns (overall_ok, judged, recorded, meta)."""
    paths = paths or RunPaths.default()
    if params["geometry"]["type"] == "plate_with_hole":
        return _verify_plate(params, paths, use_rag=use_rag)
    return _verify_beam(params, paths, use_rag=use_rag)


def _verify_plate(params: dict, paths: RunPaths, use_rag: bool = True) -> tuple:
    """Plate: judge the stress-concentration factor, record displacement."""
    from boundary_conditions import _read_boundary_node_sets

    frd = parse_frd(paths.frd)
    g = params["geometry"]["plate_with_hole"]
    a, b, t = float(g["a"]), float(g["b"]), float(g["t"])
    r, xc, yc = float(g["r"]), float(g["xc"]), float(g["yc"])
    F = float(params["bc"]["loads"][0]["magnitude"])
    tol = float(params["acceptance"].get("tol_scf", 0.08))

    # Hole nodes come from the MESH physical group: .frd coordinates are written
    # at E12.5 precision, so re-deriving "on the circle" from them would miss
    # most of the boundary nodes.
    edge_sets = _read_boundary_node_sets(paths.msh, 1)
    hole = edge_sets.get("hole_boundary", {})
    hole_ids = hole.get("node_ids", [])
    hole_xyz = hole.get("xyz", {})
    vm_hole = {n: _von_mises(frd["STRESS"][n]) for n in hole_ids if n in frd["STRESS"]}
    if not vm_hole:
        raise RuntimeError("no stress found at hole_boundary nodes")
    n_peak = max(vm_hole, key=vm_hole.get)
    sigma_max = vm_hole[n_peak]
    p_peak = hole_xyz[n_peak]
    theta_peak = float(np.degrees(np.arctan2(p_peak[1] - yc, p_peak[0] - xc)))

    net_width = b - 2.0 * r
    sigma_net = F / (net_width * t)
    sigma_gross = F / (b * t)
    kt_fem = sigma_max / sigma_net
    lam = 2.0 * r / b

    case, provenance = _select_reference(params, use_rag)
    if case is None:
        judged = {"status": "N/A", "kt_fem": kt_fem, "case": None, "provenance": provenance}
    else:
        kt_ref = float(case.formula(params))
        error = abs(kt_fem - kt_ref) / kt_ref if kt_ref else float("inf")
        judged = {
            "status": "PASS" if error < tol else "FAIL",
            "kt_fem": kt_fem, "kt_ref": kt_ref, "error": error, "tol": tol,
            "case": case, "provenance": provenance,
        }
    judged.update({"sigma_max": sigma_max, "sigma_net": sigma_net,
                   "sigma_gross": sigma_gross, "lam": lam,
                   "peak_node": n_peak, "theta_peak": theta_peak,
                   "n_hole_nodes": len(vm_hole)})

    # Recorded, NOT judged: the plate's elongation has no closed form.
    load_ids = build_bc(params, paths=paths)["load"]["node_ids"]
    ux = [float(frd["DISP"][n][0]) for n in load_ids if n in frd["DISP"]]
    recorded = {"mean_elongation": float(np.mean(ux)) if ux else float("nan"),
                "max_vm_global": max(_von_mises(s) for s in frd["STRESS"].values())}
    meta = {"frd": str(paths.frd), "geom": "plate_with_hole",
            "model_line": (f"plate_with_hole a={a} b={b} t={t} mm, hole r={r} at "
                           f"({xc}, {yc}) mm | 2D plane stress | F={F} N")}
    overall_ok = judged["status"] != "FAIL"
    return overall_ok, judged, recorded, meta


def _verify_beam(params: dict, paths: RunPaths, use_rag: bool = True) -> tuple:
    """Cantilever: judge the tip deflection, record stress (plan section 7)."""
    frd = parse_frd(paths.frd)
    bc_def = build_bc(params, paths=paths)
    load_nodes = bc_def["load"]["node_ids"]
    load_dir = np.asarray(bc_def["load"]["loads"][0]["dir_unit"], dtype=float)

    delta_fem = _tip_deflection(frd, load_nodes, load_dir)
    max_vm = max(_von_mises(s) for s in frd["STRESS"].values())

    g = params["geometry"]["cantilever_beam"]
    L, h, b = float(g["L"]), float(g["h"]), float(g["b"])
    E = float(params["material"]["E"])
    P = float(params["bc"]["loads"][0]["magnitude"])
    tol = float(params["acceptance"]["tol_disp"])

    case, provenance = _select_reference(params, use_rag)
    if case is None:
        disp = {"status": "N/A", "delta_fem": delta_fem, "case": None, "provenance": provenance}
    else:
        delta_ref = float(case.formula(params))
        error = abs(delta_fem - delta_ref) / delta_ref if delta_ref else float("inf")
        disp = {
            "status": "PASS" if error < tol else "FAIL",
            "delta_fem": delta_fem, "delta_ref": delta_ref, "error": error, "tol": tol,
            "case": case, "I": b * h ** 3 / 12.0, "P": P, "L": L, "E": E, "L_over_h": L / h,
            "provenance": provenance,
        }

    stress = {"max_vm": max_vm}
    meta = {"frd": str(paths.frd), "geom": "cantilever_beam", "L": L, "h": h, "b": b,
            "model_line": f"cantilever_beam | static | linear elastic | L={L} h={h} b={b} mm"}
    overall_ok = disp["status"] != "FAIL"   # PASS or N/A -> chain ok; FAIL -> diagnostics
    return overall_ok, disp, stress, meta


def _format_plate_report(overall_ok: bool, judged: dict, recorded: dict, meta: dict) -> str:
    """Verification report for the plate (plate plan M7)."""
    lines = []
    lines.append("Numerical Verification Report (plan layer 7 / plate plan M7)")
    lines.append("=" * 64)
    lines.append(f"problem : {meta['model_line']}")
    lines.append(f"frd     : {meta['frd']}")
    lines.append("")
    lines.append("reference selection (RAG retrieval -> deterministic applies() gate):")
    lines.append(f"  {judged.get('provenance', '')}")
    case = judged["case"]
    if case is None:
        lines.append("  -> no applicable analytical judge; SCF verdict = N/A")
        lines.append("  (the measured Kt is recorded below and handed to the human;")
        lines.append("   no verdict is fabricated for a configuration the reference "
                     "does not cover)")
    else:
        lines.append(f"  MATCHED -> {case.name}  [id: {case.id}]")
        lines.append(f"    applies_to : {case.applies_to}")
        lines.append(f"    scope_note : {case.scope_note}")
        lines.append(f"    source     : {case.source}")
    lines.append("")
    lines.append("STRESS CONCENTRATION  (auto-judged):")
    lines.append(f"  peak    = {judged['sigma_max']:.4f} MPa (max von Mises over "
                 f"{judged['n_hole_nodes']} hole-boundary nodes)")
    lines.append(f"            at node {judged['peak_node']}, theta = {judged['theta_peak']:+.1f} deg "
                 f"(Kirsch/Howland peak at +-90 deg)")
    lines.append(f"  scales  = sigma_net {judged['sigma_net']:.4f} MPa "
                 f"(= F/((b-2r)*t)) ; sigma_gross {judged['sigma_gross']:.4f} MPa "
                 f"(= F/(b*t)) ; 2r/b = {judged['lam']:.3f}")
    if case is None:
        lines.append(f"  Kt_fem  = {judged['kt_fem']:.4f}  -> N/A (no reference)")
    else:
        lines.append(f"  Kt_fem  = {judged['kt_fem']:.4f}   (peak / sigma_net)")
        lines.append(f"  Kt_ref  = {judged['kt_ref']:.4f}   (Howland/Peterson fit at "
                     f"2r/b = {judged['lam']:.3f})")
        lines.append(f"  rel error = {judged['error'] * 100:.4f} %    "
                     f"tolerance = {judged['tol'] * 100:.4f} %")
        lines.append(f"  -> {judged['status']}")
    lines.append("")
    lines.append("DISPLACEMENT  (NOT auto-judged -- recorded for human review):")
    lines.append(f"  mean elongation of the loaded edge = {recorded['mean_elongation']:.6f} mm")
    lines.append("  A perforated plate's elongation has no closed-form reference, so no")
    lines.append("  verdict is issued on it. (The reverse of the cantilever, where")
    lines.append("  displacement is judged and stress is not -- each geometry is judged")
    lines.append("  on the quantity its physics actually makes judgeable.)")
    lines.append(f"  max von Mises over the whole model = {recorded['max_vm_global']:.4f} MPa")
    lines.append("")
    lines.append("mesh convergence: measured during M6 -- Kt_fem = 2.2395 / 2.2441 / 2.2471")
    lines.append("  at 32 / 64 / 126 hole segments; the hole-edge peak converges to a finite")
    lines.append("  value (unlike the cantilever's fixed-end corner), which is what makes")
    lines.append("  the stress auto-judgeable here.")
    lines.append("-" * 64)
    if case is None:
        lines.append("VERDICT: N/A  (no applicable reference; SCF not auto-judged)")
    elif judged["status"] == "PASS":
        lines.append("VERDICT: PASS  (Kt within tolerance of the Howland/Peterson reference)")
    else:
        lines.append("VERDICT: FAIL  (Kt outside tolerance) -> go to diagnostics (layer 8)")
    lines.append("=" * 64)
    return "\n".join(lines) + "\n"


def _format_report(overall_ok: bool, disp: dict, stress: dict, meta: dict) -> str:
    if meta.get("geom") == "plate_with_hole":
        return _format_plate_report(overall_ok, disp, stress, meta)
    lines = []
    lines.append("Numerical Verification Report (plan layer 7)")
    lines.append("=" * 64)
    lines.append(f"problem : {meta['geom']} | static | linear elastic | "
                 f"L={meta['L']} h={meta['h']} b={meta['b']} mm")
    lines.append(f"frd     : {meta['frd']}")
    lines.append("")
    lines.append("reference selection (RAG retrieval -> deterministic applies() gate):")
    lines.append(f"  {disp.get('provenance', '')}")
    case = disp["case"]
    if case is None:
        lines.append("  -> no applicable analytical judge; displacement verdict = N/A")
        lines.append("  (FEM results handed to the human; no verdict is fabricated)")
    else:
        lines.append(f"  MATCHED -> {case.name}  [id: {case.id}]")
        lines.append(f"    applies_to : {case.applies_to}")
        lines.append(f"    scope_note : {case.scope_note}")
        lines.append(f"    source     : {case.source}")
        lines.append(f"    note       : this configuration has L/h = {disp['L_over_h']:.1f}")
    lines.append("")
    lines.append("DISPLACEMENT  (auto-judged):")
    if case is None:
        lines.append(f"  delta_fem = {disp['delta_fem']:.5f} mm  -> N/A (no reference)")
    else:
        lines.append(f"  delta_fem = {disp['delta_fem']:.5f} mm   "
                     f"(max |U . load_dir| over the free-end LOAD face)")
        lines.append(f"  delta_ref = {disp['delta_ref']:.5f} mm   "
                     f"(P L^3 / 3EI; P={disp['P']:.0f} N, L={disp['L']:.0f} mm, "
                     f"E={disp['E']:.0f} MPa, I={disp['I']:.2f} mm^4)")
        lines.append(f"  rel error = {disp['error'] * 100:.4f} %    "
                     f"tolerance = {disp['tol'] * 100:.4f} %")
        lines.append(f"  -> {disp['status']}")
    lines.append("")
    lines.append("STRESS  (NOT auto-judged -- recorded for human review):")
    lines.append(f"  max von Mises = {stress['max_vm']:.2f} MPa")
    lines.append("  Stress exits auto-judgment in the MVP because:")
    for i, reason in enumerate(STRESS_EXIT_REASONS, 1):
        lines.append(f"    {i}. {reason}")
    lines.append("  -> N/A. Judge stress visually from out/stress_x.png / stress_y.png / stress_z.png.")
    lines.append("")
    lines.append("mesh convergence study : not performed in the MVP (out of scope).")
    lines.append("-" * 64)
    if case is None:
        lines.append("VERDICT: N/A  (no applicable reference; displacement not auto-judged)")
    elif disp["status"] == "PASS":
        lines.append("VERDICT: PASS  (displacement within tolerance of the analytical solution)")
    else:
        lines.append("VERDICT: FAIL  (displacement outside tolerance) -> go to diagnostics (layer 8)")
    lines.append("=" * 64)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Numerical verification (plan layer 7).")
    add_workdir_argument(ap)
    paths = from_args(ap.parse_args())
    overall_ok, disp, stress, meta = run_verification(PARAMS, paths=paths)
    report = _format_report(overall_ok, disp, stress, meta)
    paths.verification_report.write_text(report)
    print(report, end="")
    print(f"(written to {paths.verification_report})")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
