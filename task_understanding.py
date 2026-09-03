#!/usr/bin/env python3
"""Task understanding layer 1 (plan section 1): natural language -> a complete,
unambiguous problem definition for the parameterization layer.

Responsibilities (plan section 1):
  - Off-topic guard: reject anything that is not an FEA task.
  - Completeness: the LLM judges, per dimension, only whether it is SPECIFIED
    (semantic) -- it does NOT invent default values.
  - Missing MUST-HAVE dimensions (geometry, material, fixed end, load):
      interactive mode -> ask the user (ONE consolidated question per round),
                          fold the answer back, re-assess;
      --auto mode      -> stop and report what is missing (never fabricate).
    Material is also validated against the SI-mm materials library: a name that
    is not in the library is treated as a gap (re-ask), and a hit is rewritten
    to its canonical name so layer 2 resolves it deterministically.
  - Missing DEFAULTABLE dimensions (mesh size, element order/shape, analysis
    type, tolerance): a DETERMINISTIC code policy fills them and every default
    is disclosed in an "assumptions applied" list.

Iron law: the LLM does semantic judgement (is this an FEA task? is dimension X
specified?); deterministic code owns the default VALUES, the material lookup,
and the routing. The LLM never fills E/nu/rho.

Output: a completed problem statement (+ assumptions) handed to parameterize.py.
Unit system: SI-mm.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from pydantic import BaseModel

import llm_config
from knowledge.materials import material_names, resolve_material

# Deterministic default policy for the DEFAULTABLE dimensions (code, not LLM).
DEFAULT_ELEM_ORDER = 2
DEFAULT_ELEM_SHAPE = "hex"
DEFAULT_ANALYSIS_TYPE = "static"
DEFAULT_NONLINEAR = False
DEFAULT_TOL_DISP = 0.05
MESH_FRACTION = 5.0  # elem_size = smallest cross-section dimension / 5


class TaskAssessment(BaseModel):
    """What the LLM reports about a description. *_specified = "did the user
    state this?" (semantic); the values are extracted only when specified."""
    is_fea_task: bool
    off_topic_reason: Optional[str] = None
    in_scope: bool = True
    scope_note: Optional[str] = None
    # Which pipeline geometry the description is about, when it is recognisably
    # one of them. This separates "the pipeline cannot model that" from "the
    # pipeline models that, but this chat front end cannot set it up yet" --
    # the plate is supported by every deterministic layer, only its
    # natural-language entry is still missing (plate plan M13).
    geometry_type: Optional[str] = None
    geometry_specified: bool = False
    L: Optional[float] = None
    h: Optional[float] = None
    b: Optional[float] = None
    material_specified: bool = False
    material_name: Optional[str] = None
    fixed_specified: bool = False
    fixed_end: Optional[str] = None
    load_specified: bool = False
    load_magnitude: Optional[float] = None
    load_end: Optional[str] = None
    load_direction: Optional[str] = None
    mesh_specified: bool = False
    mesh_elem_size: Optional[float] = None
    analysis_specified: bool = False
    analysis_type: Optional[str] = None
    nonlinear: Optional[bool] = None
    tol_specified: bool = False
    tol_disp: Optional[float] = None


SYSTEM_PROMPT = """You analyze a description of a structural FEA problem for a
cantilever-beam pipeline and call report_understanding. Rules:

- Decide is_fea_task: false only for things that are not solid-mechanics / FEA
  problems at all (greetings, chit-chat, unrelated questions). If false, give a
  short off_topic_reason and leave the rest empty.
- Set geometry_type when the description is recognisably one of the pipeline's
  geometries: "cantilever_beam" for a beam clamped at one end, or
  "plate_with_hole" for a flat plate/sheet/strip with a circular hole in it
  under tension. A plate with a hole IS a real FEA problem this pipeline
  solves, so is_fea_task stays true for it; set in_scope=false with the
  scope_note "plate_with_hole has no natural-language entry yet" and leave the
  beam dimensions empty, because only the beam can be set up from chat today.
- SUPPORTED SCOPE for THIS chat entry: exactly one cantilever beam with ONE end fully fixed
  (clamped) and a SINGLE TRANSVERSE POINT LOAD at the opposite FREE END. NOT
  supported: a load anywhere but the free end (e.g. mid-span / in the middle),
  more than one fixed end (both ends fixed / clamped-clamped / simply supported),
  distributed loads, multiple loads, axial-only loads, or non-beam geometry. If
  the request clearly needs an unsupported feature, set in_scope=false with a
  one-line scope_note naming it; otherwise set in_scope=true.
- For each dimension set *_specified to whether the USER ACTUALLY STATED it.
  Do NOT invent or assume values. Only fill a value when the user gave it (a
  strongly implied value, e.g. a downward load on a horizontal cantilever, may
  be marked specified). If unsure, mark it not specified.
- Dimensions: geometry (L length, h height, b width, in mm); material (name
  only); fixed (which end is clamped); load (magnitude in N, which end,
  direction); mesh (element size mm); analysis (static/fatigue, linear/
  nonlinear); tolerance (displacement tolerance).
- Convert any units to SI-mm when you extract a value.
"""


def _tool_def() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "report_understanding",
            "description": "Report which FEA problem dimensions the description specifies.",
            "parameters": TaskAssessment.model_json_schema(),
        },
    }


def assess_task(description: str) -> TaskAssessment:
    """One LLM call: semantic completeness assessment of the description."""
    client = llm_config.make_client()
    resp = client.chat.completions.create(
        model=llm_config.OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        tools=[_tool_def()],
        tool_choice={"type": "function", "function": {"name": "report_understanding"}},
    )
    msg = resp.choices[0].message
    if not getattr(msg, "tool_calls", None):
        raise RuntimeError(f"model did not report understanding: {msg.content!r}")
    return TaskAssessment.model_validate(json.loads(msg.tool_calls[0].function.arguments))


def _material_ok(a: TaskAssessment) -> bool:
    """Material is satisfied only if it is specified AND resolvable in the library."""
    return bool(a.material_specified and resolve_material(a.material_name)[1] is not None)


def must_have_gaps(a: TaskAssessment) -> list:
    """The MUST-HAVE dimensions the description does not (validly) specify."""
    gaps = []
    if not (a.geometry_specified and a.L and a.h and a.b):
        gaps.append("geometry")
    if not _material_ok(a):
        gaps.append("material")
    if not a.fixed_specified:
        gaps.append("fixed")
    if not a.load_specified:
        gaps.append("load")
    return gaps


def _gap_line(gap: str, a: TaskAssessment) -> str:
    if gap == "geometry":
        return "geometry: length L, height h, width b in mm (e.g. 100 x 10 x 10)"
    if gap == "material":
        known = ", ".join(material_names())
        if a.material_specified and a.material_name:
            return (f"material: '{a.material_name}' is not in the library "
                    f"(known: {known}; or add it to knowledge/materials.py)")
        return f"material: name (known: {known}; others must be added to the library)"
    if gap == "fixed":
        return "fixed: which end is clamped"
    if gap == "load":
        return "load: magnitude in N, which end, and direction (e.g. 1000 N down at the free end)"
    return gap


def consolidated_question(gaps: list, a: TaskAssessment) -> str:
    """One prompt listing every missing must-have (option C: ask once per round)."""
    lines = ["I still need the following before I can proceed:"]
    lines += [f"  - {_gap_line(g, a)}" for g in gaps]
    lines.append("Please provide them in a single reply: ")
    return "\n".join(lines)


def apply_defaults(a: TaskAssessment) -> tuple:
    """Fill DEFAULTABLE dimensions from the deterministic policy; return
    (resolved, assumptions). Assumes the must-haves are present and valid."""
    assumptions = []
    canonical, _props = resolve_material(a.material_name)  # valid here (not a gap)
    resolved = {
        "L": a.L, "h": a.h, "b": a.b,
        "material_name": canonical or a.material_name,
        "fixed_end": a.fixed_end or "one end",
        "load_magnitude": a.load_magnitude,
        "load_end": a.load_end or "the free end",
        "load_direction": a.load_direction or "down (-z)",
    }
    if not a.load_direction:
        assumptions.append("load direction = down (-z) [cantilever convention]")

    if a.mesh_specified and a.mesh_elem_size:
        resolved["elem_size"] = float(a.mesh_elem_size)
    else:
        section = [v for v in (a.h, a.b) if v]
        elem_size = round(min(section) / MESH_FRACTION, 6) if section else None
        resolved["elem_size"] = elem_size
        assumptions.append(f"mesh elem_size = {elem_size} mm [policy: smallest section dim / {MESH_FRACTION:g}]")

    resolved["elem_order"] = DEFAULT_ELEM_ORDER
    resolved["elem_shape"] = DEFAULT_ELEM_SHAPE
    assumptions.append(f"element order = {DEFAULT_ELEM_ORDER}, shape = {DEFAULT_ELEM_SHAPE} [pipeline convention]")

    if a.analysis_specified:
        resolved["analysis_type"] = a.analysis_type or DEFAULT_ANALYSIS_TYPE
        resolved["nonlinear"] = bool(a.nonlinear)
    else:
        resolved["analysis_type"] = DEFAULT_ANALYSIS_TYPE
        resolved["nonlinear"] = DEFAULT_NONLINEAR
        assumptions.append(f"analysis = {DEFAULT_ANALYSIS_TYPE}, linear [default]")

    if a.tol_specified and a.tol_disp:
        resolved["tol_disp"] = float(a.tol_disp)
    else:
        resolved["tol_disp"] = DEFAULT_TOL_DISP
        assumptions.append(f"displacement tolerance = {DEFAULT_TOL_DISP} [default]")

    return resolved, assumptions


def build_description(r: dict) -> str:
    """A complete, explicit problem statement for parameterize.py (layer 2)."""
    lin = "nonlinear" if r["nonlinear"] else "linear"
    return (
        f"A {r['material_name']} cantilever beam, length L={r['L']} mm, "
        f"height h={r['h']} mm (along z, the load direction), width b={r['b']} mm (along y). "
        f"Fixed at {r['fixed_end']}; a {r['load_magnitude']} N load at {r['load_end']} "
        f"acting {r['load_direction']}. "
        f"Mesh element size {r['elem_size']} mm, order {r['elem_order']}, {r['elem_shape']}. "
        f"Analysis: {r['analysis_type']}, {lin}. Displacement tolerance {r['tol_disp']}."
    )


def run_task_understanding(description: str, auto: bool = False, max_rounds: int = 3,
                           input_fn=input, verbose: bool = True) -> dict:
    """Assess -> (ask once per round | stop | default) -> completed definition."""
    assessment = assess_task(description)
    if not assessment.is_fea_task:
        return {"status": "rejected", "reason": assessment.off_topic_reason or "not an FEA task"}

    cur_desc = description
    for _round in range(max_rounds + 1):
        if not assessment.in_scope:
            # A geometry the pipeline HAS is reported as such, with the way to
            # run it -- claiming it is unsupported would be untrue.
            if assessment.geometry_type and assessment.geometry_type != "cantilever_beam":
                return {"status": "no_nl_entry", "geometry_type": assessment.geometry_type,
                        "reason": (f"{assessment.geometry_type} is supported by the pipeline but "
                                   f"not yet through natural-language entry"),
                        "how_to_run": f"python pipeline_run.py --case {assessment.geometry_type}"}
            return {"status": "out_of_scope",
                    "reason": assessment.scope_note or "outside the supported scope"}
        gaps = must_have_gaps(assessment)
        if not gaps:
            break
        if auto:
            return {"status": "incomplete", "gaps": gaps,
                    "message": "missing must-have info; cannot proceed in --auto mode (no fabrication)"}
        if _round == max_rounds:
            return {"status": "incomplete", "gaps": gaps,
                    "message": f"still missing after {max_rounds} rounds"}
        answer = input_fn(consolidated_question(gaps, assessment) + "\n   your answer: ")
        cur_desc = cur_desc + "\nAdditional details -> " + answer
        assessment = assess_task(cur_desc)

    resolved, assumptions = apply_defaults(assessment)
    return {
        "status": "complete",
        "resolved": resolved,
        "assumptions": assumptions,
        "completed_description": build_description(resolved),
        "assessment": assessment.model_dump(),
    }


def _print_result(result: dict) -> int:
    print("-" * 64)
    status = result["status"]
    if status == "rejected":
        print(f"REJECTED (off-topic): {result['reason']}")
        print("=" * 64)
        return 3
    if status == "out_of_scope":
        print(f"OUT OF SCOPE: {result['reason']}")
        print("supported: a cantilever — one fixed end, a transverse point load at the free end.")
        print("=" * 64)
        return 4
    if status == "incomplete":
        print(f"INCOMPLETE: {result['message']}")
        print(f"missing must-have: {', '.join(result['gaps'])}")
        print("=" * 64)
        return 2
    print("assumptions applied (deterministic defaults, disclosed):")
    for a in result["assumptions"]:
        print(f"  - {a}")
    print("-" * 64)
    print("completed problem definition (-> parameterize.py):")
    print(f"  {result['completed_description']}")
    print("-" * 64)
    print("ACCEPTANCE: PASS (definition complete, no leftover ambiguity)")
    print("=" * 64)
    return 0


def _selftest() -> int:
    """Deterministic self-test of the policy/routing/material checks (no API)."""
    print("task_understanding.py self-test (deterministic parts, no API)")
    print("=" * 64)
    ok = True

    complete = TaskAssessment(is_fea_task=True, geometry_specified=True, L=100, h=10, b=10,
                              material_specified=True, material_name="STEEL",
                              fixed_specified=True, fixed_end="one end",
                              load_specified=True, load_magnitude=1000, load_end="free end")
    print(f"1) complete -> gaps {must_have_gaps(complete)} (expect [])")
    ok = ok and must_have_gaps(complete) == []
    resolved, assumptions = apply_defaults(complete)
    print(f"   mesh default = {resolved['elem_size']} mm (expect 2.0)")
    ok = ok and resolved["elem_size"] == 2.0
    for a in assumptions:
        print(f"     - {a}")
    print(f"   completed: {build_description(resolved)}")

    miss = TaskAssessment(is_fea_task=True, geometry_specified=True, L=100, h=10, b=10,
                          material_specified=False, fixed_specified=True, load_specified=False)
    print(f"2) missing material+load -> gaps {must_have_gaps(miss)} (expect material, load)")
    ok = ok and set(must_have_gaps(miss)) == {"material", "load"}

    off = TaskAssessment(is_fea_task=False, off_topic_reason="greeting")
    print(f"3) off-topic is_fea_task = {off.is_fea_task} (expect False)")
    ok = ok and off.is_fea_task is False

    alu = TaskAssessment(is_fea_task=True, geometry_specified=True, L=100, h=10, b=10,
                         material_specified=True, material_name="Alu",
                         fixed_specified=True, load_specified=True, load_magnitude=1000)
    g_alu = must_have_gaps(alu)
    r_alu, _ = apply_defaults(alu)
    print(f"4) material 'Alu' -> gaps {g_alu} (expect []); canonical -> {r_alu['material_name']} (expect ALUMINUM)")
    ok = ok and g_alu == [] and r_alu["material_name"] == "ALUMINUM"

    unk = TaskAssessment(is_fea_task=True, geometry_specified=True, L=100, h=10, b=10,
                         material_specified=True, material_name="INCONEL",
                         fixed_specified=True, load_specified=True, load_magnitude=1000)
    print(f"5) material 'INCONEL' (not in library) -> gaps {must_have_gaps(unk)} (expect ['material'])")
    print(f"   would ask: {_gap_line('material', unk)}")
    ok = ok and must_have_gaps(unk) == ["material"]

    print("-" * 64)
    print(f"ACCEPTANCE: {'PASS' if ok else 'FAIL'}")
    print("=" * 64)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Task understanding (plan layer 1).")
    ap.add_argument("description", nargs="?", default=None, help="natural-language problem")
    ap.add_argument("--auto", action="store_true", help="full-auto: stop on missing must-haves, do not ask")
    ap.add_argument("--parameterize", action="store_true", help="chain into layer 2 (set_params) on success")
    ap.add_argument("--selftest", action="store_true", help="run the no-API deterministic self-test")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.description:
        ap.error("a description is required (or use --selftest)")

    print("=" * 64)
    print("task_understanding.py -- layer 1")
    print("=" * 64)
    print(f"mode        : {'auto' if args.auto else 'interactive'}")
    print(f"description : {args.description}")
    result = run_task_understanding(args.description, auto=args.auto)
    rc = _print_result(result)
    if rc == 0 and args.parameterize:
        import pprint

        import parameterize
        print("\n>>> handing the completed definition to parameterize.py (layer 2) ...\n")
        params_result = parameterize.run_set_params(result["completed_description"])
        print(pprint.pformat(params_result["params"], sort_dicts=False, width=88))
    return rc


if __name__ == "__main__":
    sys.exit(main())
