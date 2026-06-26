#!/usr/bin/env python3
"""Numerical verification layer 7 (plan section 7): knowledge-base-driven judge.

Measures numerical ACCURACY (not physical plausibility): is the FEM result close
to a reference solution? MVP scope decision (plan section 7):

  * DISPLACEMENT is auto-judged against the analytical tip deflection
    delta = P L^3 / (3 E I), tolerance tol_disp (5%). It is a global integral
    quantity -- insensitive to local singularities, fast to converge.
  * STRESS is NOT auto-judged -- only recorded, with the exit reason, and handed
    to the human to read from the PNGs. (Fixed-end stress concentration diverges
    under refinement; the free-end point load is an artificial singularity; so
    stress error is not a meaningful convergence quantity here.)

IRON LAW: the formula comes from the knowledge base (an agent only SELECTS it);
the error and threshold are computed by deterministic code here. For this
milestone the selection is the deterministic knowledge.select_reference_case;
the LLM agent swaps in at that seam only.

Unit system: SI-mm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from boundary_conditions import build_bc
from knowledge.reference_solutions import select_reference_case
from params import PARAMS
from physics_validation import _von_mises, parse_frd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
FRD_PATH = OUT_DIR / "beam.frd"
MSH_PATH = OUT_DIR / "beam.msh"
REPORT_PATH = OUT_DIR / "verification_report.txt"

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
    gtype = params["geometry"]["type"].replace("_", " ")
    n_fixed = len(params["bc"]["fixed"])
    n_loads = len(params["bc"]["loads"])
    return (f"{gtype}, fixed at {n_fixed} end, {n_loads} transverse concentrated "
            f"point load at the free end, linear elastic small strain")


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


def run_verification(params: dict, frd_path=FRD_PATH, msh_path=MSH_PATH) -> tuple:
    """Run the verification. Returns (overall_ok, disp, stress, meta)."""
    frd = parse_frd(Path(frd_path))
    bc_def = build_bc(params, msh_path=msh_path)
    load_nodes = bc_def["load"]["node_ids"]
    load_dir = np.asarray(bc_def["load"]["loads"][0]["dir_unit"], dtype=float)

    delta_fem = _tip_deflection(frd, load_nodes, load_dir)
    max_vm = max(_von_mises(s) for s in frd["STRESS"].values())

    g = params["geometry"]["cantilever_beam"]
    L, h, b = float(g["L"]), float(g["h"]), float(g["b"])
    E = float(params["material"]["E"])
    P = float(params["bc"]["loads"][0]["magnitude"])
    tol = float(params["acceptance"]["tol_disp"])

    case, provenance = select_reference_via_rag(params)
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
    meta = {"frd": str(frd_path), "geom": "cantilever_beam", "L": L, "h": h, "b": b}
    overall_ok = disp["status"] != "FAIL"   # PASS or N/A -> chain ok; FAIL -> diagnostics
    return overall_ok, disp, stress, meta


def _format_report(overall_ok: bool, disp: dict, stress: dict, meta: dict) -> str:
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
    overall_ok, disp, stress, meta = run_verification(PARAMS)
    report = _format_report(overall_ok, disp, stress, meta)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(report, end="")
    print(f"(written to {REPORT_PATH})")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
