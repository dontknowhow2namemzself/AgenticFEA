#!/usr/bin/env python3
"""Physics validation layer 5 (plan section 5, plate plan M6): fast-fail gate.

Parses out/beam.frd directly (its own fixed-width .frd reader -- no dependency
on the postprocess .vtu, which is exactly why this gate can run BEFORE
postprocessing) and runs an engineering sanity-check checklist. Every verdict
is DETERMINISTIC code (the iron law: the LLM never judges numbers such as
sum(F)=0).

Decision rule (plan section 5): any CRITICAL (*) check FAIL -> overall FAIL ->
the orchestrator skips the expensive postprocess and goes to diagnostics.
NORMAL / INFO checks are reported but never veto. There is no numeric score.

The .frd parser and the von Mises helper are geometry-agnostic and shared; only
the CHECKLIST dispatches on geometry.type, because what counts as physically
sane differs per configuration.

cantilever_beam checklist:
  1* numerical_validity      no NaN/Inf; disp/stress/forc fields non-empty
  2* displacement_direction  tip Uz follows the load (downward load -> Uz < 0)
  3* force_balance           |sum(reaction) + sum(applied)| / |applied| < 1%
  4* strain_energy_positive  U > 0  (external work 1/2 * F . u, linear elastic)
  5  moment_balance          |M_reaction + M_applied| / |M_applied| < 2%
  6  max_stress_location     argmax(von Mises) near the fixed end (not mid-span)
  7  stress_magnitude        max von Mises within order of magnitude of Mc/I

plate_with_hole checklist (plate plan M6):
  1* numerical_validity      as above
  2* displacement_direction  mean U at the loaded edge follows the load (+x)
  3* force_balance           as above (reactions summed over FIXED only; the
                             FORC block also carries the applied loads at the
                             loaded nodes, so a whole-model sum is ~0 by
                             construction and would prove nothing)
  4* strain_energy_positive  U > 0, from the deck's actual consistent *CLOAD
  5  hole_traction_free      |sigma . n| ~ 0 on the hole (a free surface)
  6  max_stress_location     argmax(von Mises) on the hole boundary; on a
                             constrained edge instead -> WARNING (BC artifact)
  7  stress_magnitude        max von Mises within an order-of-magnitude band
                             of the Howland estimate Ktn * sigma_net

Note on the plate's applied forces: they are read back from the *CLOAD cards of
the deck that was actually solved (out/beam.inp), not recomputed from PARAMS.
The consistent 1/6-2/3-1/6 distribution lives in exactly one place (build_inp),
and this layer then checks the solved deck against its own results.

Unit system: SI-mm (length mm, force N, stress MPa, displacement mm).
"""

from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

from boundary_conditions import _read_boundary_node_sets, build_bc
from params import PARAMS

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
MSH_PATH = OUT_DIR / "beam.msh"
FRD_PATH = OUT_DIR / "beam.frd"
INP_PATH = OUT_DIR / "beam.inp"
REPORT_PATH = OUT_DIR / "physics_report.txt"

# .frd fixed-width layout (verified against this CalculiX 2.23 output):
#   record key  = columns [0:3]   (" -1", " -4", ...)
#   node number = columns [3:13]  (width 10)
#   each value  = width 12        (E12.5), starting at column 13
_FRD_NODE = slice(3, 13)
_FRD_VAL_W = 12
_FRD_VAL_0 = 13
_FRD_NCOMP = {"COORDS": 3, "DISP": 3, "STRESS": 6, "FORC": 3}

Check = namedtuple("Check", "name level status message")  # level: CRITICAL|NORMAL|INFO

# Hole traction-free tolerance, as a fraction of the PEAK STRESS ON THE HOLE.
# The hole is a free surface, so sigma . n vanishes in the exact solution; what
# a FEM run shows is the nodal extrapolation error, which converges away with
# the mesh (measured on the default case: 6.19 / 2.68 / 0.66 % at 32 / 64 / 126
# hole segments).
#
# The normalisation is the peak von Mises ON THE HOLE BOUNDARY, not sigma_net.
# "Is this traction negligible?" is a question about the stress actually
# present on that surface, and sigma_net is the wrong yardstick for it: as the
# hole shrinks, sigma_net -> sigma_gross while the absolute extrapolation error
# stays ~1-1.5 MPa, so small holes were penalised for their size rather than
# their resolution. The 500-case batch made that concrete -- all 15 flagged
# cases were small holes (r median 10.4 vs 19.0 overall), and refining two of
# the worst moved Kt by only 0.4 % while the residual fell by 4x: the labels
# were fine, the metric was not comparable across hole sizes. Measured on the
# same three cases, dividing by the local peak gives 2.8 / 4.3 / 4.0 % where
# sigma_net gave 6.2 / 11.8 / 10.7 %.
#
# The value is recorded per case (info["hole_traction_rel"]) so this threshold
# is a POST-HOC decision, and it was in fact set that way: over the 505-case
# production batch the new metric lands at p50 2.48 %, p90 3.46 %, p99 3.93 %,
# max 4.28 % -- tight and comparable across hole sizes, where the old sigma_net
# normalisation spread the same population from 2 % to 12 %. 6 % is ~1.4x the
# worst case actually observed on a correctly resolved mesh, so it stays quiet
# on good runs while a genuinely unresolved hole (or a load/constraint wrongly
# assigned to it, which lands at order 100 %) still trips it.
HOLE_TRACTION_TOL = 0.06


def _ffloat(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")  # unparseable field -> NaN, caught by numerical_validity


def parse_frd(path: Path) -> dict:
    """Read a CalculiX ASCII .frd into {coords, DISP, STRESS, FORC, version}.

    Fixed-width parsing (fields can touch, e.g. '0.0E+00-5.0E+00'). Tracks the
    active block so element-connectivity lines are not mistaken for results.
    """
    coords: dict = {}
    fields = {"DISP": {}, "STRESS": {}, "FORC": {}}
    version = ""
    block = None  # None | "COORDS" | "ELEM" | "DISP" | "STRESS" | "FORC" | other

    for line in path.read_text().splitlines():
        key = line[:3]
        if key == " -4":                       # result block header
            block = line[3:].split()[0]
            continue
        if key in (" -5", " -2"):              # component def / connectivity continuation
            continue
        if key == " -3":                       # end of block
            block = None
            continue
        if key == " -1":                       # data line
            if block is None or block == "ELEM":
                continue
            if block != "COORDS" and block not in fields:
                continue
            node = int(line[_FRD_NODE])
            ncomp = _FRD_NCOMP.get(block, (len(line) - _FRD_VAL_0) // _FRD_VAL_W)
            vals = np.array([
                _ffloat(line[_FRD_VAL_0 + _FRD_VAL_W * i: _FRD_VAL_0 + _FRD_VAL_W * (i + 1)])
                for i in range(ncomp)
            ])
            if block == "COORDS":
                coords[node] = vals
            else:
                fields[block][node] = vals
            continue
        # non "-x" lines: block markers (2C nodes / 3C elements) and metadata
        stripped = line.strip()
        if not stripped:
            continue
        tok = stripped.split()[0]
        if tok == "2C":
            block = "COORDS"
        elif tok == "3C":
            block = "ELEM"
        elif tok == "1C":
            block = None
        elif "VERSION" in line:
            version = line.split("VERSION", 1)[1].strip()

    return {"coords": coords, "DISP": fields["DISP"], "STRESS": fields["STRESS"],
            "FORC": fields["FORC"], "version": version}


def _von_mises(s: np.ndarray) -> float:
    """von Mises from [SXX, SYY, SZZ, SXY, SYZ, SZX]."""
    sxx, syy, szz, sxy, syz, szx = s
    return float(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                         + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)))


def _read_cload(inp_path: Path) -> dict:
    """{node_id: force vector (N)} from the *CLOAD cards of the solved deck."""
    loads: dict = {}
    active = False
    for raw in Path(inp_path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            active = line.split(",")[0].strip().upper() == "*CLOAD"
            continue
        if not active:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        nid, dof, val = int(parts[0]), int(parts[1]), float(parts[2])
        vec = loads.setdefault(nid, np.zeros(3))
        vec[dof - 1] += val
    return loads


def _add_common_checks(guarded, disp, stress, forc, fixed_ids, applied_total,
                       applied_by_node, load_ids, disp_dir_fn):
    """Register the four CRITICAL checks, in report order.

    numerical_validity / force_balance / strain_energy_positive are identical
    for every geometry; the displacement-direction check is geometry-specific
    (what "follows the load" looks like differs), so it is passed in.
    """

    def chk_numeric():
        if not disp or not stress or not forc:
            return "FAIL", (f"empty field(s): disp={len(disp)} stress={len(stress)} "
                            f"forc={len(forc)}")
        allvals = np.concatenate([np.concatenate(list(d.values()))
                                  for d in (disp, stress, forc)])
        if not np.all(np.isfinite(allvals)):
            return "FAIL", f"non-finite values present ({np.sum(~np.isfinite(allvals))})"
        return "PASS", (f"finite & non-empty: disp={len(disp)} stress={len(stress)} "
                        f"forc={len(forc)} nodes")

    def chk_force_balance():
        # Reactions are summed over the CONSTRAINED nodes only: the .frd FORC
        # block also carries the applied nodal forces at the loaded nodes, so a
        # whole-model sum vanishes by construction and would prove nothing.
        reaction = np.sum([forc[n] for n in fixed_ids if n in forc], axis=0)
        residual = reaction + applied_total
        denom = np.linalg.norm(applied_total)
        if denom == 0:
            return "PASS", "no applied load; balance check N/A"
        ratio = float(np.linalg.norm(residual) / denom)
        msg = (f"reaction={np.array2string(reaction, precision=2)} "
               f"applied={np.array2string(applied_total, precision=2)} N, "
               f"residual={ratio * 100:.3f}% (< 1%)")
        return ("PASS" if ratio < 0.01 else "FAIL"), msg

    def chk_strain_energy():
        u = 0.5 * float(np.sum([np.dot(applied_by_node[n], disp[n])
                                for n in load_ids if n in disp and n in applied_by_node]))
        return ("PASS" if u > 0 else "FAIL"), f"U = 1/2 * F.u = {u:.4f} N.mm (> 0)"

    guarded("numerical_validity", "CRITICAL", chk_numeric)
    guarded("displacement_direction", "CRITICAL", disp_dir_fn)
    guarded("force_balance", "CRITICAL", chk_force_balance)
    guarded("strain_energy_positive", "CRITICAL", chk_strain_energy)


def run_physics_validation(params: dict, msh_path=MSH_PATH, frd_path=FRD_PATH,
                           inp_path=INP_PATH) -> tuple:
    """Run the geometry's checklist. Returns (overall_pass, checks, info)."""
    if params["geometry"]["type"] == "plate_with_hole":
        return _run_plate_validation(params, msh_path=msh_path, frd_path=frd_path,
                                     inp_path=inp_path)
    return _run_beam_validation(params, msh_path=msh_path, frd_path=frd_path)


def _run_plate_validation(params: dict, msh_path=MSH_PATH, frd_path=FRD_PATH,
                          inp_path=INP_PATH) -> tuple:
    """Plane-stress plate checklist (plate plan M6)."""
    g = params["geometry"]["plate_with_hole"]
    a, b, t = float(g["a"]), float(g["b"]), float(g["t"])
    r, xc, yc = float(g["r"]), float(g["xc"]), float(g["yc"])

    frd = parse_frd(Path(frd_path))
    coords, disp, stress, forc = frd["coords"], frd["DISP"], frd["STRESS"], frd["FORC"]

    bc_def = build_bc(params, msh_path=msh_path)
    fixed_ids = bc_def["fixed"]["node_ids"]
    load_ids = bc_def["load"]["node_ids"]
    load_dir_unit = np.asarray(bc_def["load"]["loads"][0]["dir_unit"], dtype=float)

    # Hole nodes come from the MESH physical group: the .frd coordinates are
    # written at E12.5 precision (+-5e-4 mm), so re-deriving "on the circle"
    # from them would miss most of the boundary nodes.
    edge_sets = _read_boundary_node_sets(msh_path, 1)
    hole_ids = edge_sets.get("hole_boundary", {}).get("node_ids", [])
    hole_xyz = edge_sets.get("hole_boundary", {}).get("xyz", {})

    # Applied nodal forces = the deck's consistent *CLOAD (single source of truth).
    applied_by_node = _read_cload(Path(inp_path)) if Path(inp_path).exists() else {}
    applied_total = (np.sum(list(applied_by_node.values()), axis=0)
                     if applied_by_node else np.zeros(3))

    sigma_gross = float(np.linalg.norm(applied_total)) / (b * t)
    net_width = b - 2.0 * r
    sigma_net = (float(np.linalg.norm(applied_total)) / (net_width * t)
                 if net_width > 0 else float("nan"))
    lam = 2.0 * r / b
    ktn = 3.00 - 3.14 * lam + 3.667 * lam ** 2 - 1.527 * lam ** 3   # Howland fit

    checks: list = []

    def add(name, level, status, message):
        checks.append(Check(name, level, status, message))

    def guarded(name, level, fn):
        try:
            status, message = fn()
        except Exception as exc:  # noqa: BLE001 - a data problem becomes a clean FAIL
            status, message = "FAIL", f"error: {exc.__class__.__name__}: {exc}"
        add(name, level, status, message)

    # The loaded edge must move along the load direction (mean over the edge:
    # individual nodes near the anchor barely move, which is correct).
    def chk_disp_dir():
        vals = np.array([float(np.dot(disp[n], load_dir_unit))
                         for n in load_ids if n in disp])
        if vals.size == 0:
            return "FAIL", "no displacement at LOAD nodes"
        mean_v = float(vals.mean())
        msg = (f"mean U.load_dir over the loaded edge = {mean_v:+.5f} mm "
               f"(range [{vals.min():+.5f}, {vals.max():+.5f}]), must follow the load")
        return ("PASS" if mean_v > 0 else "FAIL"), msg

    _add_common_checks(guarded, disp, stress, forc, fixed_ids, applied_total,
                       applied_by_node, load_ids, chk_disp_dir)

    vm_all = {n: _von_mises(s) for n, s in stress.items()} if stress else {}
    vm_hole = {n: vm_all[n] for n in hole_ids if n in vm_all}

    # 5. The hole is a free surface: its traction sigma . n must vanish.
    #    Computed up front so the number can be recorded, not just judged.
    traction_abs = 0.0
    for n in hole_ids:
        if n not in stress:
            continue
        pt = hole_xyz[n]
        nx, ny = float(pt[0]) - xc, float(pt[1]) - yc
        norm = float(np.hypot(nx, ny))
        if norm == 0:
            continue
        nx, ny = nx / norm, ny / norm
        sxx, syy, _szz, sxy, _syz, _szx = stress[n]
        tx = sxx * nx + sxy * ny            # traction vector on the hole surface
        ty = sxy * nx + syy * ny
        traction_abs = max(traction_abs, float(np.hypot(tx, ty)))
    sigma_peak = max(vm_hole.values()) if vm_hole else float("nan")
    traction_rel = (traction_abs / sigma_peak
                    if sigma_peak and np.isfinite(sigma_peak) else float("inf"))

    def chk_hole_traction():
        if not hole_ids:
            return "WARNING", "no hole_boundary nodes found in the mesh"
        msg = (f"max |sigma.n| on the hole = {traction_abs:.4f} MPa = "
               f"{traction_rel * 100:.2f}% of the peak stress there "
               f"({sigma_peak:.3f} MPa), tol {HOLE_TRACTION_TOL * 100:.0f}%; "
               f"this residual is nodal extrapolation error and converges with "
               f"the mesh (6.2% / 2.7% / 0.7% of sigma_net at 32 / 64 / 126 "
               f"hole segments)")
        return ("PASS" if traction_rel < HOLE_TRACTION_TOL else "WARNING"), msg

    # 6. The stress peak belongs on the hole, not on a constrained edge.
    def chk_stress_location():
        if not vm_all:
            return "FAIL", "no stress field"
        nmax = max(vm_all, key=vm_all.get)
        on_hole = nmax in set(hole_ids)
        on_fixed = nmax in set(fixed_ids)
        where = "hole boundary" if on_hole else (
            "CONSTRAINED edge (BC artifact -- check the support model)" if on_fixed
            else "plate interior / free edge")
        pos = coords.get(nmax)
        pos_str = f"({pos[0]:.2f}, {pos[1]:.2f})" if pos is not None else "?"
        theta = ""
        if on_hole and nmax in hole_xyz:
            p = hole_xyz[nmax]
            theta = (f", theta = {np.degrees(np.arctan2(p[1] - yc, p[0] - xc)):+.1f} deg "
                     f"(Kirsch/Howland peak at +-90)")
        msg = f"max vM = {vm_all[nmax]:.2f} MPa at node {nmax} {pos_str}, on the {where}{theta}"
        return ("PASS" if on_hole else "WARNING"), msg

    # 7. Order-of-magnitude band against the Howland estimate.
    def chk_stress_magnitude():
        if not vm_all:
            return "FAIL", "no stress field"
        vm_max = max(vm_all.values())
        expected = ktn * sigma_net
        if not np.isfinite(expected) or expected <= 0:
            return "WARNING", f"max vM = {vm_max:.2f} MPa; no valid reference estimate"
        ratio = vm_max / expected
        msg = (f"max vM = {vm_max:.2f} MPa vs Ktn*sigma_net = {expected:.2f} MPa "
               f"(Ktn={ktn:.3f} at 2r/b={lam:.2f}); ratio {ratio:.3f} "
               f"(order-of-magnitude band [0.5, 2.0]; layer 7 judges it precisely)")
        return ("PASS" if 0.5 <= ratio <= 2.0 else "WARNING"), msg

    guarded("hole_traction_free", "INFO", chk_hole_traction)
    guarded("max_stress_location", "INFO", chk_stress_location)
    guarded("stress_magnitude", "INFO", chk_stress_magnitude)

    overall_pass = all(c.status != "FAIL" for c in checks if c.level == "CRITICAL")
    info = {
        "version": frd["version"], "frd": str(frd_path),
        "model_line": (f"plate_with_hole a={a} b={b} t={t} mm, hole r={r} at "
                       f"({xc}, {yc}) mm | 2D plane stress"),
        "applied": applied_total.tolist(),
        "extra": (f"sigma_gross = F/(b*t) = {sigma_gross:.4f} MPa ; "
                  f"sigma_net = F/((b-2r)*t) = {sigma_net:.4f} MPa ; "
                  f"hole boundary nodes = {len(vm_hole)}"),
        "n_disp": len(disp), "n_stress": len(stress), "n_forc": len(forc),
        # Recorded metrics (numbers, not verdicts) so a batch manifest can carry
        # them and thresholds stay re-tunable without re-solving.
        "hole_traction_abs": traction_abs,
        "hole_traction_rel": traction_rel,
        "sigma_peak_hole": sigma_peak,
        "sigma_net": sigma_net, "sigma_gross": sigma_gross,
    }
    return overall_pass, checks, info


def _run_beam_validation(params: dict, msh_path=MSH_PATH, frd_path=FRD_PATH) -> tuple:
    """Cantilever checklist (plan section 5)."""
    g = params["geometry"]["cantilever_beam"]
    L, h, b = float(g["L"]), float(g["h"]), float(g["b"])

    frd = parse_frd(Path(frd_path))
    coords, disp, stress, forc = frd["coords"], frd["DISP"], frd["STRESS"], frd["FORC"]

    bc_def = build_bc(params, msh_path=msh_path)
    fixed_ids = bc_def["fixed"]["node_ids"]
    load_ids = bc_def["load"]["node_ids"]

    # Applied force per load node + total (deterministic, from PARAMS/CLOAD model).
    applied_by_node: dict = {}
    applied_total = np.zeros(3)
    for ld in bc_def["load"]["loads"]:
        n = ld["n_nodes"]
        if n == 0:
            continue
        f_total = ld["magnitude"] * np.asarray(ld["dir_unit"], dtype=float)
        f_node = f_total / n
        for nid in ld["node_ids"]:
            applied_by_node[nid] = applied_by_node.get(nid, np.zeros(3)) + f_node
        applied_total += f_total

    checks: list = []

    def add(name, level, status, message):
        checks.append(Check(name, level, status, message))

    def guarded(name, level, fn):
        try:
            status, message = fn()
        except Exception as exc:  # noqa: BLE001 - a data problem becomes a clean FAIL
            status, message = "FAIL", f"error: {exc.__class__.__name__}: {exc}"
        add(name, level, status, message)

    # Displacement direction: the tip follows the load (beam-specific wording).
    def chk_disp_dir():
        tip_uz = np.array([disp[n][2] for n in load_ids if n in disp])
        if tip_uz.size == 0:
            return "FAIL", "no displacement at LOAD nodes"
        max_abs = float(np.max(np.abs(tip_uz)))
        if applied_total[2] < 0:
            ok = float(np.max(tip_uz)) < 1e-9
            direction = "downward (Uz<0)"
        elif applied_total[2] > 0:
            ok = float(np.min(tip_uz)) > -1e-9
            direction = "upward (Uz>0)"
        else:
            return "PASS", "no transverse (z) load; direction check N/A"
        msg = (f"tip max|Uz|={max_abs:.4f} mm, range [{tip_uz.min():.4f}, "
               f"{tip_uz.max():.4f}] mm, expected {direction}")
        return ("PASS" if ok else "FAIL"), msg

    # Moment balance about the fixed-face centroid (origin).
    def chk_moment_balance():
        m_react = np.sum([np.cross(coords[n], forc[n]) for n in fixed_ids
                          if n in forc and n in coords], axis=0)
        m_appl = np.sum([np.cross(coords[n], applied_by_node[n])
                         for n in load_ids if n in coords and n in applied_by_node], axis=0)
        denom = np.linalg.norm(m_appl)
        if denom == 0:
            return "PASS", "no applied moment; check N/A"
        ratio = float(np.linalg.norm(m_react + m_appl) / denom)
        msg = (f"|M_react|={np.linalg.norm(m_react):.1f}, |M_applied|={denom:.1f} N.mm "
               f"(~P*L={float(np.linalg.norm(applied_total)) * L:.0f}), residual={ratio * 100:.3f}% (< 2%)")
        return ("PASS" if ratio < 0.02 else "FAIL"), msg

    # Max von Mises location (INFO: near fixed end vs mid-span).
    def chk_stress_location():
        vm = {n: _von_mises(s) for n, s in stress.items()}
        nmax = max(vm, key=vm.get)
        x = float(coords[nmax][0]) if nmax in coords else float("nan")
        near_fixed = x < 0.15 * L
        where = "near fixed end" if near_fixed else (
            "near free-end load point (point-load singularity)" if x > 0.85 * L else "mid-span")
        msg = f"max vM={vm[nmax]:.1f} MPa at node {nmax} (x={x:.2f} mm, {where})"
        return ("PASS" if near_fixed else "WARNING"), msg

    # 7. stress magnitude order vs nominal bending stress Mc/I = 6*P*L/(b*h^2).
    def chk_stress_magnitude():
        vm_max = max(_von_mises(s) for s in stress.values())
        p = float(np.linalg.norm(applied_total))
        nominal = 6.0 * p * L / (b * h * h)
        if nominal == 0:
            return "PASS", f"max vM={vm_max:.1f} MPa; no nominal stress reference"
        ratio = vm_max / nominal
        ok = 0.5 <= ratio <= 20.0
        msg = (f"max vM={vm_max:.1f} MPa vs nominal Mc/I={nominal:.1f} MPa "
               f"(ratio {ratio:.2f}; FEM max exceeds nominal at the singular corner)")
        return ("PASS" if ok else "WARNING"), msg

    _add_common_checks(guarded, disp, stress, forc, fixed_ids, applied_total,
                       applied_by_node, load_ids, chk_disp_dir)
    guarded("moment_balance", "NORMAL", chk_moment_balance)
    guarded("max_stress_location", "INFO", chk_stress_location)
    guarded("stress_magnitude", "INFO", chk_stress_magnitude)

    overall_pass = all(c.status != "FAIL" for c in checks if c.level == "CRITICAL")
    info = {
        "version": frd["version"], "frd": str(frd_path),
        "model_line": f"cantilever_beam L={L} h={h} b={b} mm",
        "applied": applied_total.tolist(),
        "n_disp": len(disp), "n_stress": len(stress), "n_forc": len(forc),
    }
    return overall_pass, checks, info


def _format_report(overall_pass: bool, checks: list, info: dict) -> str:
    lines = []
    lines.append("Physics Validation Report (plan layer 5)")
    lines.append("=" * 64)
    lines.append(f"solver : CalculiX {info['version']}    frd: {info['frd']}")
    lines.append(f"model  : {info['model_line']}    applied load = {info['applied']} N")
    if info.get("extra"):
        lines.append(f"scales : {info['extra']}")
    lines.append(f"parsed : disp={info['n_disp']}  stress={info['n_stress']}  forc={info['n_forc']} nodes")
    lines.append("-" * 64)
    lines.append("checklist  ( * = Critical, vetoes on FAIL )")
    for c in checks:
        mark = "*" if c.level == "CRITICAL" else " "
        lines.append(f"[{c.status:<7}]{mark} {c.name:<24} {c.message}")
    lines.append("-" * 64)
    if overall_pass:
        lines.append("VERDICT: PASS  (all Critical checks PASS) -> proceed to postprocess (layer 6)")
    else:
        failed = [c.name for c in checks if c.level == "CRITICAL" and c.status == "FAIL"]
        lines.append(f"VERDICT: FAIL  (Critical FAIL: {', '.join(failed)}) "
                     f"-> SKIP postprocess, go to diagnostics (layer 8)")
    lines.append("=" * 64)
    return "\n".join(lines) + "\n"


def main() -> int:
    overall_pass, checks, info = run_physics_validation(PARAMS)
    report = _format_report(overall_pass, checks, info)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(report, end="")
    print(f"(written to {REPORT_PATH})")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
