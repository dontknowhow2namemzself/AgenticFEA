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

VALIDITY of the linear-elastic model, checked beside the verdict. Agreement
with the reference shows that the solver and the theory agree with each other;
both assume a linear-elastic material and small deflections, so a load that
breaks those assumptions can still "pass" -- a 1000 x 20 x 10 mm steel
cantilever under 1500 N agreed to 0.06 % at 2250 MPa nominal stress and a tip
deflection of 36 % of its length. Two deterministic checks say so:
  * stress below the yield strength (knowledge.materials.YIELD, with its grade
    and source; a material without one is reported as not checked);
  * small deflection, delta / L <= SMALL_DEFLECTION (the beam).
They do not change the verdict on the reference; they say whether the
verified model is the right model.

Unit system: SI-mm; stress concentration factors are dimensionless.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from boundary_conditions import build_bc
from knowledge.materials import yield_strength
from knowledge.reference_solutions import select_reference_case
from params import PARAMS
from paths import RunPaths, add_workdir_argument, from_args
from physics_validation import _von_mises, parse_frd

# Small-deflection limit for the beam, delta / L. Up to here the linear theory
# that both the solver and the reference use differs from the exact large-
# deflection solution (the elastica of a tip-loaded cantilever) by 1.0 %, the
# tip turning 8.5 deg; at 0.20 the difference is 4 %, at 0.36 it is 12 %.
SMALL_DEFLECTION = 0.10

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


def _validity(params: dict, sigma: float, sigma_what: str, thickness: float,
              delta: float = None, length: float = None, not_checked: str = "") -> list:
    """Is the linear-elastic, small-deflection model valid for this result?

    `thickness` is the governing thickness [mm] that picks the yield strength's
    band. Returns [{"name", "status" (OK | EXCEEDED | NOT CHECKED), "detail"}, ...];
    the yield check also carries its "basis": which fy, and where it comes from.
    `not_checked` says why `sigma` is not this result's stress: the yield check
    is then not made.
    """
    checks = []
    material = params["material"]["name"]
    fy = yield_strength(material, thickness)
    if fy is None:
        basis = f"no yield strength for {material} in the materials library"
        checks.append({"name": "stress below yield", "status": "NOT CHECKED",
                       "detail": basis, "basis": basis})
    elif fy[0] is None:
        checks.append({"name": "stress below yield", "status": "NOT CHECKED",
                       "detail": fy[1], "basis": fy[1]})
    elif not_checked:
        checks.append({"name": "stress below yield", "status": "NOT CHECKED",
                       "detail": not_checked, "basis": fy[1]})
    else:
        fy_value, source = fy
        ok = sigma <= fy_value
        checks.append({"name": "stress below yield", "status": "OK" if ok else "EXCEEDED",
                       "detail": (f"{sigma_what} = {sigma:.1f} MPa {'<=' if ok else '>'} "
                                  f"fy = {fy_value:g} MPa ({sigma / fy_value:.2f} x fy); {source}"),
                       "basis": source})
    if delta is not None:
        ratio = delta / length
        ok = ratio <= SMALL_DEFLECTION
        checks.append({"name": "small deflection", "status": "OK" if ok else "EXCEEDED",
                       "detail": (f"delta/L = {delta:.4g}/{length:g} = {ratio:.3f} "
                                  f"{'<=' if ok else '>'} {SMALL_DEFLECTION:g} (beyond it the linear "
                                  f"theory of solver and reference no longer holds)")})
    return checks


def _validity_lines(checks: list) -> list:
    lines = ["VALIDITY OF THE LINEAR-ELASTIC MODEL  (checked beside the verdict, not part of it):"]
    for c in checks:
        lines.append(f"  [{c['status']:<11}] {c['name']:<18} {c['detail']}")
    return lines


def _verdict_suffix(checks: list) -> str:
    broken = [c["name"] for c in checks if c["status"] == "EXCEEDED"]
    if not broken:
        return ""
    return (" -- BUT outside the linear-elastic model's validity (" + "; ".join(broken)
            + "): the solver and the reference agree under assumptions this load breaks")


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
                           f"({xc}, {yc}) mm | 2D plane stress | F={F} N"),
            # The hole peak converges (unlike the beam's clamped corner), so it is
            # a fair stress to hold against the yield strength.
            "validity": _validity(params, sigma_max, "peak von Mises on the hole", thickness=t)}
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
    # Against yield: the nominal bending stress at the clamp, M c / I = 6 P L / (b h^2).
    # The FE peak sits at the clamped corner, a singularity that grows with mesh
    # refinement, so it would trip the check for any load on a fine enough mesh.
    sigma_nominal = 6.0 * P * L / (b * h ** 2)
    # That is the bending of a load along the height h (z) only; for a load with
    # a sideways or axial part it is not this beam's stress, so no check at all.
    along_h = abs(load_dir[0]) < 1e-9 and abs(load_dir[1]) < 1e-9
    meta = {"frd": str(paths.frd), "geom": "cantilever_beam", "L": L, "h": h, "b": b,
            "model_line": f"cantilever_beam | static | linear elastic | L={L} h={h} b={b} mm",
            # A solid rectangular bar's governing thickness is its smaller side.
            "validity": _validity(params, sigma_nominal, "nominal bending stress at the clamp, "
                                  "M c / I", thickness=min(h, b), delta=delta_fem, length=L,
                                  not_checked="" if along_h else (
                                      "the load is not along the height h, so M c / I with "
                                      "I = b h^3 / 12 is not this beam's bending stress"))}
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
    lines.append("")
    lines += _validity_lines(meta["validity"])
    lines.append("-" * 64)
    suffix = _verdict_suffix(meta["validity"])
    if case is None:
        lines.append("VERDICT: N/A  (no applicable reference; SCF not auto-judged)" + suffix)
    elif judged["status"] == "PASS":
        lines.append("VERDICT: PASS  (Kt within tolerance of the Howland/Peterson reference)" + suffix)
    else:
        lines.append("VERDICT: FAIL  (Kt outside tolerance) -> go to diagnostics (layer 8)" + suffix)
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
    lines.append("")
    lines += _validity_lines(meta["validity"])
    lines.append("-" * 64)
    suffix = _verdict_suffix(meta["validity"])
    if case is None:
        lines.append("VERDICT: N/A  (no applicable reference; displacement not auto-judged)" + suffix)
    elif disp["status"] == "PASS":
        lines.append("VERDICT: PASS  (displacement within tolerance of the analytical solution)" + suffix)
    else:
        lines.append("VERDICT: FAIL  (displacement outside tolerance) -> go to diagnostics (layer 8)" + suffix)
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
