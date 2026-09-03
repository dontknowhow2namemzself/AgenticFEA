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
from schema import min_usable_ligament

# Deterministic default policy for the DEFAULTABLE dimensions (code, not LLM).
DEFAULT_ELEM_ORDER = 2
DEFAULT_ELEM_SHAPE = "hex"
DEFAULT_ANALYSIS_TYPE = "static"
DEFAULT_NONLINEAR = False
DEFAULT_TOL_DISP = 0.05
MESH_FRACTION = 5.0        # beam: elem_size = smallest cross-section dimension / 5
# Plate: elem_size is the FAR-FIELD size only -- the mesh layer already refines
# locally by its own rules (>= 28 segments around the hole, >= 6 elements across
# the thinnest ligament), so this default does not need to chase the hole. One
# twenty-fifth of the shorter plate dimension gives 4 mm on a 200x100 plate,
# which is the setting the whole 545-case dataset was generated at.
PLATE_MESH_FRACTION = 25.0
DEFAULT_PLATE_THICKNESS = 1.0   # mm; plane stress wants t << a, b and sigma ~ F/t
                                # is a free linear rescaling, so 1 mm is the
                                # natural unit choice rather than a guess
DEFAULT_PLATE_ELEM_SHAPE = "quad"


class TaskAssessment(BaseModel):
    """What the LLM reports about a description. *_specified = "did the user
    state this?" (semantic); the values are extracted only when specified."""
    is_fea_task: bool
    off_topic_reason: Optional[str] = None
    in_scope: bool = True
    scope_note: Optional[str] = None
    # Which pipeline geometry the description is about. It is chosen FIRST,
    # because it decides which slots are must-haves, what the clarifying
    # question should ask for, and which defaults apply (plate plan M13).
    geometry_type: Optional[str] = None
    geometry_specified: bool = False
    # cantilever_beam slots
    L: Optional[float] = None
    h: Optional[float] = None
    b: Optional[float] = None
    # plate_with_hole slots. The hole centre is optional by design: "a plate
    # with a hole in the middle" is the common phrasing and centring is the
    # textbook configuration, so an unstated centre is a disclosed default
    # rather than a missing must-have.
    plate_a: Optional[float] = None
    plate_b: Optional[float] = None
    plate_t: Optional[float] = None
    hole_r: Optional[float] = None
    hole_xc: Optional[float] = None
    hole_yc: Optional[float] = None
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
- FIRST set geometry_type, which decides which slots matter:
    "cantilever_beam"  -- a beam/bar clamped at one end with a transverse load
                          at the free end.
    "plate_with_hole"  -- a flat plate / sheet / strip / panel with a circular
                          hole (or bore / opening / cutout) in it, pulled in
                          tension.
  Both are fully supported; set in_scope=true for either.
- cantilever_beam SCOPE: exactly ONE end fully fixed (clamped) and a SINGLE
  TRANSVERSE POINT LOAD at the opposite FREE END. NOT supported: a load
  anywhere but the free end (e.g. mid-span), more than one fixed end
  (clamped-clamped / simply supported), distributed loads, multiple loads,
  axial-only loads.
  Slots: L (length), h (height, the load direction), b (width), all in mm.
- plate_with_hole SCOPE: ONE circular hole, uniaxial TENSION along the plate's
  length, linear elastic plane stress. NOT supported: several holes, non-circular
  cutouts, compression, biaxial or shear loading, bending of the plate.
  Slots: plate_a (length along the pull direction), plate_b (width across it),
  plate_t (thickness), hole_r (hole RADIUS), hole_xc / hole_yc (hole centre
  measured from the LOWER-LEFT corner of the plate), all in mm.
  * If the user gives a DIAMETER, halve it and put the radius in hole_r.
  * Leave hole_xc / hole_yc null when the hole is described as centred / "in the
    middle", or when its position is not mentioned at all. Only fill them when a
    genuine off-centre position is given, and convert it to lower-left-corner
    coordinates first.
  * Leave plate_t null when no thickness is given.
  * The plate's supports are NOT a user choice (the pipeline always uses a
    roller plus a mid-edge anchor and pulls the far edge), so do not treat a
    missing support description as a gap: set fixed_specified=true whenever the
    description makes clear the plate is being pulled or held.
  * load_magnitude is the TOTAL tensile force in N. Convert a stated stress to a
    force only if the user gives one (force = stress x b x t); otherwise leave
    load_specified=false.
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


def is_plate(a: TaskAssessment) -> bool:
    return a.geometry_type == "plate_with_hole"


def geometry_impossible(a: TaskAssessment) -> Optional[str]:
    """Why the described geometry cannot exist, or None if it can.

    This belongs HERE, before parameterization, and the reason is specific.
    Layer 2 asks the model to fill slots and re-prompts it with any validation
    error, so a geometry that fails validation invites the model to make the
    error go away -- and it does: asked again about a 50 mm hole in a 100 mm
    plate, it returned r = 49.999, silently editing the number the user gave
    to slip past the gate. That is precisely the failure this project exists
    to prevent, and no wording of the retry prompt can be relied on to stop
    it. Catching it before layer 2 means the user is told their problem is
    impossible instead of being handed a different problem's answer.
    """
    if not is_plate(a) or not (a.plate_a and a.plate_b and a.hole_r):
        return None
    pa, pb, r = float(a.plate_a), float(a.plate_b), float(a.hole_r)
    xc = float(a.hole_xc) if a.hole_xc is not None else pa / 2.0
    yc = float(a.hole_yc) if a.hole_yc is not None else pb / 2.0
    lig = min(xc - r, pa - xc - r, yc - r, pb - yc - r)
    if lig <= 0:
        return (f"a hole of radius {r:g} mm (diameter {2 * r:g} mm) does not fit inside "
                f"a {pa:g} x {pb:g} mm plate centred at ({xc:g}, {yc:g}) mm -- it would "
                f"cut through the edge. The hole needs to be smaller, or the plate bigger.")
    floor = min_usable_ligament(pa, pb)
    if lig < floor:
        return (f"a hole of radius {r:g} mm at ({xc:g}, {yc:g}) mm leaves only "
                f"{lig:.3g} mm of material between the hole and the nearest edge of a "
                f"{pa:g} x {pb:g} mm plate. Below about {floor:.3g} mm that strip cannot "
                f"be meshed or analysed meaningfully. Reduce the radius or move the hole "
                f"further from the edge.")
    return None


def must_have_gaps(a: TaskAssessment) -> list:
    """The MUST-HAVE dimensions the description does not (validly) specify.

    What counts as a must-have differs per geometry, because what the pipeline
    can honestly default differs. The beam needs to be told which end is
    clamped and where the load acts; the plate's support model is not a user
    choice at all (roller plus a mid-edge anchor, pulled on the far edge is the
    only configuration this geometry implements), so asking for it would be
    theatre. Thickness and hole centre are defaulted and disclosed instead.
    """
    gaps = []
    if is_plate(a):
        if not (a.geometry_specified and a.plate_a and a.plate_b and a.hole_r):
            gaps.append("geometry")
    elif not (a.geometry_specified and a.L and a.h and a.b):
        gaps.append("geometry")
    if not _material_ok(a):
        gaps.append("material")
    if not is_plate(a) and not a.fixed_specified:
        gaps.append("fixed")
    if not a.load_specified:
        gaps.append("load")
    return gaps


def _gap_line(gap: str, a: TaskAssessment) -> str:
    if gap == "geometry":
        if is_plate(a):
            return ("geometry: plate length a and width b, and hole radius r, in mm "
                    "(e.g. 200 x 100 plate with a 20 mm radius hole)")
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
        if is_plate(a):
            return "load: total tensile force in N (e.g. 1000 N pulling the plate)"
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
    if is_plate(a):
        return _apply_plate_defaults(a)
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


def _apply_plate_defaults(a: TaskAssessment) -> tuple:
    """Plate policy. Everything assumed is returned in `assumptions` and shown
    to the user -- an assumption the user never sees is a fabrication."""
    assumptions = []
    canonical, _props = resolve_material(a.material_name)
    pa, pb = float(a.plate_a), float(a.plate_b)
    resolved = {"geometry_type": "plate_with_hole", "a": pa, "b": pb,
                "r": float(a.hole_r),
                "material_name": canonical or a.material_name,
                "load_magnitude": a.load_magnitude}

    if a.plate_t:
        resolved["t"] = float(a.plate_t)
    else:
        resolved["t"] = DEFAULT_PLATE_THICKNESS
        assumptions.append(f"plate thickness t = {DEFAULT_PLATE_THICKNESS} mm "
                           f"[plane stress; stress scales as F/t, so t is a free unit choice]")
    if a.hole_xc is not None and a.hole_yc is not None:
        resolved["xc"], resolved["yc"] = float(a.hole_xc), float(a.hole_yc)
    else:
        resolved["xc"], resolved["yc"] = pa / 2.0, pb / 2.0
        assumptions.append(f"hole centred at ({pa/2:g}, {pb/2:g}) mm "
                           f"[the Kirsch/Howland reference configuration]")
    # Not a user choice: this geometry implements exactly one support model.
    assumptions.append("supports = roller on the left edge (u_x = 0) plus one "
                       "mid-edge anchor (u_y = 0); load = uniform tension on the "
                       "right edge [the only configuration this geometry implements]")

    if a.mesh_specified and a.mesh_elem_size:
        resolved["elem_size"] = float(a.mesh_elem_size)
    else:
        elem_size = round(min(pa, pb) / PLATE_MESH_FRACTION, 6)
        resolved["elem_size"] = elem_size
        assumptions.append(f"far-field mesh elem_size = {elem_size} mm "
                           f"[policy: min(a, b) / {PLATE_MESH_FRACTION:g}; the mesh layer "
                           f"refines the hole and the ligament by its own rules]")
    resolved["elem_order"] = DEFAULT_ELEM_ORDER
    resolved["elem_shape"] = DEFAULT_PLATE_ELEM_SHAPE
    assumptions.append(f"element order = {DEFAULT_ELEM_ORDER}, shape = "
                       f"{DEFAULT_PLATE_ELEM_SHAPE} [2D plane stress convention]")

    if a.analysis_specified:
        resolved["analysis_type"] = a.analysis_type or DEFAULT_ANALYSIS_TYPE
        resolved["nonlinear"] = bool(a.nonlinear)
    else:
        resolved["analysis_type"] = DEFAULT_ANALYSIS_TYPE
        resolved["nonlinear"] = DEFAULT_NONLINEAR
        assumptions.append(f"analysis = {DEFAULT_ANALYSIS_TYPE}, linear [default]")
    resolved["tol_disp"] = DEFAULT_TOL_DISP
    return resolved, assumptions


def build_description(r: dict) -> str:
    """A complete, explicit problem statement for parameterize.py (layer 2)."""
    if r.get("geometry_type") == "plate_with_hole":
        lin = "nonlinear" if r["nonlinear"] else "linear"
        return (
            f"A {r['material_name']} flat plate in 2D plane stress, "
            f"a={r['a']} mm along x by b={r['b']} mm along y, thickness t={r['t']} mm, "
            f"with a circular hole of radius r={r['r']} mm centred at "
            f"(xc={r['xc']}, yc={r['yc']}) mm measured from the lower-left corner. "
            f"Supported by a roller on edge_x_min (u_x = 0) plus a single mid-edge "
            f"anchor node (u_y = 0); a total tensile force of {r['load_magnitude']} N "
            f"applied on edge_x_max in the +x direction. "
            f"Mesh element size {r['elem_size']} mm, order {r['elem_order']}, "
            f"{r['elem_shape']}. Analysis: {r['analysis_type']}, {lin}."
        )
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
    impossible = geometry_impossible(assessment)
    if impossible:
        return {"status": "impossible_geometry", "reason": impossible}

    cur_desc = description
    for _round in range(max_rounds + 1):
        if not assessment.in_scope:
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
    print()
    plate_ok = True
    full = TaskAssessment(is_fea_task=True, geometry_type="plate_with_hole",
                          geometry_specified=True, plate_a=200, plate_b=100, hole_r=20,
                          material_specified=True, material_name="STEEL",
                          fixed_specified=True, load_specified=True, load_magnitude=1000)
    gaps = must_have_gaps(full)
    print(f"6) plate, centred hole, no thickness given -> gaps {gaps} (expect [])")
    plate_ok &= (gaps == [])
    resolved, assumptions = apply_defaults(full)
    print(f"   defaults: t={resolved['t']} centre=({resolved['xc']}, {resolved['yc']}) "
          f"elem_size={resolved['elem_size']} shape={resolved['elem_shape']}")
    plate_ok &= (resolved["t"] == 1.0 and resolved["xc"] == 100 and resolved["yc"] == 50
                 and resolved["elem_size"] == 4.0 and resolved["elem_shape"] == "quad")
    print(f"   every assumption disclosed: {len(assumptions)} lines")
    plate_ok &= any("thickness" in a for a in assumptions) and any("centred" in a for a in assumptions)
    print(f"   description: {build_description(resolved)[:88]}...")

    thin = TaskAssessment(is_fea_task=True, geometry_type="plate_with_hole",
                          geometry_specified=True, plate_a=200, plate_b=100,
                          material_specified=True, material_name="STEEL",
                          fixed_specified=True, load_specified=True, load_magnitude=1000)
    g2 = must_have_gaps(thin)
    print(f"7) plate WITHOUT a hole radius -> gaps {g2} (expect ['geometry'])")
    print(f"   would ask: {_gap_line('geometry', thin)}")
    plate_ok &= (g2 == ["geometry"]) and "radius" in _gap_line("geometry", thin)

    nosup = TaskAssessment(is_fea_task=True, geometry_type="plate_with_hole",
                           geometry_specified=True, plate_a=200, plate_b=100, hole_r=20,
                           material_specified=True, material_name="STEEL",
                           fixed_specified=False, load_specified=True, load_magnitude=1000)
    g3 = must_have_gaps(nosup)
    print(f"8) plate with no support description -> gaps {g3} (expect []; the plate's "
          f"support model is not a user choice)")
    plate_ok &= (g3 == [])
    ok = ok and plate_ok

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
