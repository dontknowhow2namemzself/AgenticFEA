#!/usr/bin/env python3
"""Physics validation layer 5 (plan section 5): cheap fast-fail gate.

Parses out/beam.frd directly (its own fixed-width .frd reader -- no dependency
on the postprocess .vtu, which is exactly why this gate can run BEFORE
postprocessing), assembles a SimulationResult, and runs an engineering
sanity-check checklist. Every verdict is DETERMINISTIC code (the iron law: the
LLM never judges numbers such as sum(F)=0); the LLM-written explanation of a
FAIL is deferred until after the deterministic chain is verified.

Decision rule (plan section 5): any CRITICAL (*) check FAIL -> overall FAIL ->
the orchestrator skips the expensive postprocess and goes to diagnostics.
NORMAL / INFO checks are reported but never veto. There is no numeric score.

Checklist:
  1* displacement_direction  tip Uz follows the load (downward load -> Uz < 0)
  2* force_balance           |sum(reaction) + sum(applied)| / |applied| < 1%
  3  moment_balance          |M_reaction + M_applied| / |M_applied| < 2%
  4* strain_energy_positive  U > 0  (external work 1/2 * F . u, linear elastic)
  5* numerical_validity      no NaN/Inf; disp/stress/forc fields non-empty
  6  max_stress_location     argmax(von Mises) near the fixed end (not mid-span)
  7  stress_magnitude        max von Mises within order of magnitude of Mc/I

Unit system: SI-mm (length mm, force N, stress MPa, displacement mm).
"""

from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

from boundary_conditions import build_bc
from params import PARAMS

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
MSH_PATH = OUT_DIR / "beam.msh"
FRD_PATH = OUT_DIR / "beam.frd"
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


def run_physics_validation(params: dict, msh_path=MSH_PATH, frd_path=FRD_PATH) -> tuple:
    """Run the checklist. Returns (overall_pass, checks, info)."""
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

    # 5. numerical validity (run first: other checks rely on the fields existing).
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

    # 1. displacement direction (tip follows the load).
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

    # 2. force balance.
    def chk_force_balance():
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

    # 3. moment balance about the fixed-face centroid (origin).
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

    # 4. strain energy > 0 (external work = strain energy for linear elasticity).
    def chk_strain_energy():
        u = 0.5 * float(np.sum([np.dot(applied_by_node[n], disp[n])
                                for n in load_ids if n in disp and n in applied_by_node]))
        return ("PASS" if u > 0 else "FAIL"), f"U = 1/2 * F.u = {u:.3f} N.mm (> 0)"

    # 6. max von Mises location (INFO: near fixed end vs mid-span).
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

    guarded("numerical_validity", "CRITICAL", chk_numeric)
    guarded("displacement_direction", "CRITICAL", chk_disp_dir)
    guarded("force_balance", "CRITICAL", chk_force_balance)
    guarded("strain_energy_positive", "CRITICAL", chk_strain_energy)
    guarded("moment_balance", "NORMAL", chk_moment_balance)
    guarded("max_stress_location", "INFO", chk_stress_location)
    guarded("stress_magnitude", "INFO", chk_stress_magnitude)

    overall_pass = all(c.status != "FAIL" for c in checks if c.level == "CRITICAL")
    info = {
        "version": frd["version"], "frd": str(frd_path),
        "L": L, "h": h, "b": b, "applied": applied_total.tolist(),
        "n_disp": len(disp), "n_stress": len(stress), "n_forc": len(forc),
    }
    return overall_pass, checks, info


def _format_report(overall_pass: bool, checks: list, info: dict) -> str:
    lines = []
    lines.append("Physics Validation Report (plan layer 5)")
    lines.append("=" * 64)
    lines.append(f"solver : CalculiX {info['version']}    frd: {info['frd']}")
    lines.append(f"model  : cantilever_beam L={info['L']} h={info['h']} b={info['b']} mm    "
                 f"applied load = {info['applied']} N")
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
