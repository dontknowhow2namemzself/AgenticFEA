#!/usr/bin/env python3
"""Task understanding layer 1 (plan section 1): natural language -> a complete,
unambiguous problem definition for the parameterization layer.

Responsibilities (plan section 1):
  - Off-topic guard: reject anything that is not an FEA task.
  - Scope: decline a configuration the pipeline does not model -- but a request
    that is only VAGUE ("a beam calculation") is not out of scope; it is asked
    which of the two geometries it is, and the conversation goes on.
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

Iron law: the LLM does semantic judgement (is this an FEA task? which geometry?
where did the user state each quantity?); deterministic code owns the values,
the defaults, the material lookup and the routing. The LLM never fills E/nu/rho.

Values: mentions, not slots. The LLM does not fill "the length" -- it lists
every place the user STATED a must-have quantity, with the user's exact words
and the turn they came from. Code then decides: a quoted phrase that is not in
that user turn is discarded (it was not the user's -- an example in our own
question, say, or an invention); one distinct value is the value; two
different values are a contradiction the user is asked to settle, quoting
their own words; and the reply to that question settles it. A single slot per
quantity left the model no way to report a contradiction, so it picked a value
silently -- measured: 15 calls on one conversation that gave the length twice,
none flagged it, and the length it chose depended on the model and on how the
earlier turns were worded.

Output: a completed problem statement (+ assumptions) handed to parameterize.py.
Unit system: SI-mm.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional

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
    """The understood problem. Built by CODE: the LLM's report supplies the
    semantic fields, the must-have values come from resolve(). *_specified =
    "did the user state this?"."""
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


GEOMETRIES = ("cantilever_beam", "plate_with_hole")

# The must-have quantities whose values come from mentions:
# key -> (TaskAssessment field, how a question names it, unit).
QUANTITIES = {
    "L": ("L", "the length L", "mm"),
    "h": ("h", "the height h", "mm"),
    "b": ("b", "the width b", "mm"),
    "plate_a": ("plate_a", "the plate length a", "mm"),
    "plate_b": ("plate_b", "the plate width b", "mm"),
    "plate_t": ("plate_t", "the plate thickness t", "mm"),
    "hole_r": ("hole_r", "the hole radius r", "mm"),
    "load": ("load_magnitude", "the load", "N"),
    "material": ("material_name", "the material", ""),
}
_GEOMETRY_KEYS = ("L", "h", "b", "plate_a", "plate_b", "plate_t", "hole_r")


class Mention(BaseModel):
    """One place where the user stated a must-have quantity."""
    quantity: Literal["L", "h", "b", "plate_a", "plate_b", "plate_t", "hole_r", "load", "material"]
    turn: int                         # the [user N] turn it was stated in
    quote: str                        # the user's words, copied exactly
    value: Optional[float] = None     # in SI-mm / N; not for the material
    material: Optional[str] = None    # the material name, for quantity == "material"


class LLMReport(BaseModel):
    """What the LLM returns: semantic judgements plus mentions -- never the
    must-have values themselves (code derives those from the mentions)."""
    is_fea_task: bool
    off_topic_reason: Optional[str] = None
    in_scope: bool = True
    scope_note: Optional[str] = None
    geometry_type: Optional[str] = None
    mentions: list[Mention] = []
    hole_xc: Optional[float] = None
    hole_yc: Optional[float] = None
    fixed_specified: bool = False
    fixed_end: Optional[str] = None
    load_specified: bool = False
    load_end: Optional[str] = None
    load_direction: Optional[str] = None
    mesh_specified: bool = False
    mesh_elem_size: Optional[float] = None
    analysis_specified: bool = False
    analysis_type: Optional[str] = None
    nonlinear: Optional[bool] = None
    tol_specified: bool = False
    tol_disp: Optional[float] = None


SYSTEM_PROMPT = """You read a conversation about a structural FEA problem for a
two-geometry pipeline and call report_understanding. The conversation is given
as numbered user turns, "[user 1]", "[user 2]", ..., with the assistant's
questions in between as "[assistant]". Rules:

- Decide is_fea_task: false only for things that are not solid-mechanics / FEA
  problems at all (greetings, chit-chat, unrelated questions). If false, give a
  short off_topic_reason and leave the rest empty.
- geometry_type, which decides which quantities matter:
    "cantilever_beam"  -- a beam/bar clamped at one end with a transverse load
                          at the free end.
    "plate_with_hole"  -- a flat plate / sheet / strip / panel with a circular
                          hole (or bore / opening / cutout) in it, pulled in
                          tension.
  If the conversation does not yet say which of the two it is ("a beam
  calculation", "can you help me with an FEA?"), leave geometry_type null and
  keep in_scope=true: a request that is only vague is NOT out of scope.
- A question about what this pipeline can model or calculate ("What can you
  do?", "What kind of problems can you calculate?") IS an FEA question:
  is_fea_task=true, geometry_type null, in_scope=true. The reply lists both
  geometries.
- in_scope=false ONLY when the user describes a configuration the pipeline does
  not model; give a short scope_note.
  cantilever_beam supports exactly ONE end fully fixed (clamped) and a SINGLE
  TRANSVERSE POINT LOAD at the opposite FREE END. NOT supported: a load anywhere
  but the free end (e.g. mid-span), more than one fixed end (clamped-clamped /
  simply supported), distributed loads, multiple loads, axial-only loads.
  plate_with_hole supports ONE circular hole, uniaxial TENSION along the plate's
  length, linear elastic plane stress. NOT supported: several holes,
  non-circular cutouts, compression, biaxial or shear loading, plate bending.

- mentions: record EVERY place a user turn states one of these quantities, one
  mention per statement -- also when a quantity is stated again, and ESPECIALLY
  when it is stated again with a different value. Never choose between two
  values yourself and never drop one; the pipeline asks the user.
    L, h, b      beam length, height (the load direction), width, in mm
    plate_a      plate length along the pull direction, mm
    plate_b      plate width across it, mm
    plate_t      plate thickness, mm
    hole_r       hole RADIUS, mm (for a stated diameter, the value is half of it)
    load         the force in N (for the plate: the TOTAL tensile force;
                 convert a stated stress only if the user gives one:
                 force = stress x b x t)
    material     the material name, in "material"
  For each mention:
    turn   the number N of the [user N] turn the statement is in
    quote  the user's exact words for it, copied character for character from
           that turn (e.g. "1000mm long", "200 x 20 x10", "4.5 kN"). Several
           quantities may share one quote, like L, h and b in "200 x 20 x10".
    value  converted to mm / N.
  Only the USER's words count. Never take a value from an [assistant] turn --
  its examples ("e.g. 100 x 10 x 10") are not the user's.
  Read an answer in the light of the question it answers: after the assistant
  asks for "length L, height h, width b", the reply "200 x 20 x 10" means
  L=200, h=20, b=10.

- Plate hole position: leave hole_xc / hole_yc null when the hole is centred /
  "in the middle", or when its position is not mentioned at all. Only fill
  them for a genuinely off-centre position, measured from the LOWER-LEFT corner.
- The plate's supports are NOT a user choice (the pipeline always uses a roller
  plus a mid-edge anchor and pulls the far edge): set fixed_specified=true
  whenever the plate is described as being pulled or held.
- For the other dimensions set *_specified to whether the USER ACTUALLY STATED
  it; do not invent or assume values (a strongly implied value, e.g. a
  downward load on a horizontal cantilever, may be marked specified). fixed:
  which end is clamped. load: which end and the direction (its size is a
  mention). mesh: element size in mm. analysis: static/fatigue, linear/
  nonlinear. tolerance: the displacement tolerance.
"""


def _inline_refs(schema: dict) -> dict:
    """The tool schema with $defs inlined -- not every model resolves $ref."""
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(schema)


def _tool_def() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "report_understanding",
            "description": "Report the geometry, the scope, and where the user stated each quantity.",
            "parameters": _inline_refs(LLMReport.model_json_schema()),
        },
    }


# ---- the conversation ------------------------------------------------------
# A conversation is a list of (role, text) or (role, text, meta) entries, role
# "user" or "agent". An agent entry's meta {"asks": [quantity, ...]} marks a
# question that asked the user to settle those quantities.

def _meta(entry) -> dict:
    return (entry[2] or {}) if len(entry) > 2 else {}


def _user_turns(conversation) -> list:
    return [e[1] for e in conversation if e[0] == "user"]


def _transcript(conversation) -> str:
    lines, n = [], 0
    for entry in conversation:
        if entry[0] == "user":
            n += 1
            lines.append(f"[user {n}] {entry[1]}")
        else:
            lines.append(f"[assistant] {entry[1]}")
    return "\n".join(lines)


def assess(conversation) -> LLMReport:
    """One LLM call: the semantic report on the whole conversation."""
    client = llm_config.make_client()
    resp = client.chat.completions.create(
        model=llm_config.OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _transcript(conversation)},
        ],
        tools=[_tool_def()],
        tool_choice={"type": "function", "function": {"name": "report_understanding"}},
    )
    msg = resp.choices[0].message
    if not getattr(msg, "tool_calls", None):
        raise RuntimeError(f"model did not report understanding: {msg.content!r}")
    return LLMReport.model_validate(json.loads(msg.tool_calls[0].function.arguments))


# ---- values: decided by code from the mentions ------------------------------
def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _canonical_material(name: str) -> str:
    canonical, _props = resolve_material(name)
    return canonical or name.strip().upper()


def _distinct(quantity: str, found: list) -> list:
    """[(value, quote), ...] with one entry per different value, first quote kept."""
    out = []
    for value, quote, *_ in found:
        same = (lambda v: v == value) if quantity == "material" else \
               (lambda v: math.isclose(v, value, rel_tol=1e-9, abs_tol=1e-12))
        if not any(same(v) for v, _q in out):
            out.append((value, quote))
    return out


def resolve(report: LLMReport, conversation) -> tuple:
    """(values, conflicts, dropped) from the report's mentions, decided by code.

    values    : quantity -> the one value the user gave
    conflicts : quantity -> [(value, quote), ...] when the user gave several
    dropped   : mentions whose quote is not in the turn they name -- not the
                user's words, so not the user's value
    A reply to a question that asked to settle a quantity (meta "asks")
    settles it: earlier statements of that quantity no longer count.
    """
    users = _user_turns(conversation)
    found, dropped = [], []          # (quantity, value, quote, turn)
    for m in report.mentions:
        value = _canonical_material(m.material) if m.quantity == "material" and m.material \
            else m.value
        ok = (1 <= m.turn <= len(users) and m.quote.strip()
              and _squash(m.quote) in _squash(users[m.turn - 1]) and value is not None)
        (found if ok else dropped).append((m.quantity, value, m.quote, m.turn))

    settled, pending, turn = {}, [], 0
    for entry in conversation:
        if entry[0] == "user":
            turn += 1
            for q in pending:
                answer = _distinct(q, [(v, quote) for qq, v, quote, t in found if qq == q and t == turn])
                if len(answer) == 1:
                    settled[q] = (answer[0][0], answer[0][1], turn)
            pending = []
        else:
            pending = list(_meta(entry).get("asks", []))

    values, conflicts = {}, {}
    for q in QUANTITIES:
        stated = [(v, quote, t) for qq, v, quote, t in found if qq == q]
        if q in settled:
            v0, quote0, t0 = settled[q]
            stated = [(v0, quote0, t0)] + [s for s in stated if s[2] > t0]
        options = _distinct(q, stated)
        if len(options) == 1:
            values[q] = options[0][0]
        elif len(options) > 1:
            conflicts[q] = options
    return values, conflicts, dropped


def _assessment(report: LLMReport, values: dict, conflicts: dict) -> TaskAssessment:
    slots = {QUANTITIES[q][0]: v for q, v in values.items()}
    return TaskAssessment(
        **report.model_dump(exclude={"mentions"}), **slots,
        geometry_specified=any(q in values or q in conflicts for q in _GEOMETRY_KEYS),
        material_specified="material" in values or "material" in conflicts)


# ---- one step of the conversation -------------------------------------------
@dataclass
class Step:
    """The pipeline's next move in the conversation, and the reply that says so."""
    kind: str                          # off_topic | out_of_scope | impossible | ask | ready
    message: str                       # the reply to show, deterministic text
    reason: str = ""                   # short cause, for logs and the run record
    asks: list = field(default_factory=list)       # quantities this reply asks to settle
    gaps: list = field(default_factory=list)
    conflicts: dict = field(default_factory=dict)
    dropped: list = field(default_factory=list)
    resolved: Optional[dict] = None
    assumptions: list = field(default_factory=list)
    completed: Optional[str] = None
    assessment: Optional[TaskAssessment] = None
    # A sketch the dashboard shows with the reply: "loadcases" (both supported
    # load cases) when the user does not yet know what to give or asked for
    # something outside them; "" otherwise. Chosen here by code, never by the model.
    figure: str = ""


SUPPORTED = ("I handle two geometries: a cantilever beam (one end fixed, a single transverse "
             "point load at the free end), and a flat plate with one circular hole pulled in "
             "tension.")


def _fmt(value, unit: str) -> str:
    return f"{value:g} {unit}".strip() if isinstance(value, float) else str(value)


def conflict_question(conflicts: dict, gaps: list, a: TaskAssessment) -> str:
    """Ask the user to settle each contradiction, in their own words."""
    lines = []
    for q, options in conflicts.items():
        _field, name, unit = QUANTITIES[q]
        said = " and ".join(f"'{quote}' ({_fmt(v, unit)})" for v, quote in options)
        pick = " or ".join(_fmt(v, unit) for v, _quote in options)
        lines.append(f"You gave {name} more than once, with different values: {said}. "
                     f"Which is it: {pick}?")
    if gaps:
        lines += ["", consolidated_question(gaps, a)]
    return "\n".join(lines)


def which_geometry_question() -> str:
    return ("Which problem is it? I can model two:\n"
            "  - a cantilever beam: one end clamped, one point load at the free end\n"
            "  - a flat plate with one circular hole, pulled in tension\n"
            "Tell me which one, with its dimensions, material and load if you have them.")


def understand(conversation, report: Optional[LLMReport] = None) -> Step:
    """The whole layer-1 decision for a conversation so far: one LLM call, then code.

    `report` skips the LLM call (the self-test feeds a prepared one).
    """
    report = report or assess(conversation)
    if not report.is_fea_task:
        reason = report.off_topic_reason or "off-topic"
        return Step("off_topic", reason=reason, message=(
            f"That doesn't look like a structural FEA problem ({reason}). "
            "Try e.g. 'a steel cantilever 100 mm, 10x10 mm, fixed one end, 1000 N down at the tip'."))
    if not report.in_scope:
        reason = report.scope_note or "unsupported configuration"
        return Step("out_of_scope", reason=reason, figure="loadcases", message=(
            f"That's beyond what I can model yet — {reason}. {SUPPORTED} Want to try one of those?"))

    values, conflicts, dropped = resolve(report, conversation)
    a = _assessment(report, values, conflicts)
    if a.geometry_type not in GEOMETRIES:
        return Step("ask", which_geometry_question(), reason="geometry type not stated",
                    gaps=["geometry_type"], dropped=dropped, assessment=a, figure="loadcases")

    # A contradicted quantity is asked about as a contradiction, not as missing.
    stand_in = a.model_copy(update={QUANTITIES[q][0]: options[0][0] for q, options in conflicts.items()})
    gaps = must_have_gaps(stand_in)
    if conflicts:
        return Step("ask", conflict_question(conflicts, gaps, a),
                    reason="contradictory values: " + ", ".join(conflicts),
                    asks=list(conflicts), gaps=gaps, conflicts=conflicts, dropped=dropped,
                    assessment=a)
    impossible = geometry_impossible(a)
    if impossible:
        # Reported, never repaired. Adjusting the user's numbers to make the
        # geometry valid would answer a different question than the one asked;
        # the reply with corrected numbers settles them.
        return Step("impossible", reason=impossible, asks=["plate_a", "plate_b", "hole_r"],
                    dropped=dropped, assessment=a, message=(
                        f"That geometry cannot exist as described — {impossible}\n\n"
                        "Tell me corrected numbers and I will run it."))
    if gaps:
        return Step("ask", consolidated_question(gaps, a), reason="missing: " + ", ".join(gaps),
                    gaps=gaps, dropped=dropped, assessment=a)
    resolved, assumptions = apply_defaults(a)
    note = "; ".join(assumptions[:2]) + ("…" if len(assumptions) > 2 else "")
    return Step("ready", f"✓ Understood — assumed {note}. Running the pipeline…",
                resolved=resolved, assumptions=assumptions, dropped=dropped,
                completed=build_description(resolved), assessment=a)


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
    if a.geometry_type not in GEOMETRIES:
        return ["geometry_type"]
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
    # The beam's load end is fixed by the supported scope (the free end) and its
    # direction has a disclosed default, so the load is there once its size is.
    if not a.load_magnitude:
        gaps.append("load")
    return gaps


def _gap_line(gap: str, a: TaskAssessment) -> str:
    if gap == "geometry_type":
        return "which problem: a cantilever beam, or a flat plate with a circular hole"
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
    lines.append("Please give them in one reply.")
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
    """Understand -> (ask once per round | stop | default) -> completed definition.

    The same understand() the dashboard calls. Here the conversation is the
    terminal -- or, in --auto mode, a single description that must stand alone.
    """
    conversation = [("user", description)]
    for round_ in range(max_rounds + 1):
        step = understand(conversation)
        if step.kind != "ask" or auto or round_ == max_rounds:
            break
        answer = input_fn(step.message + "\n   your answer: ")
        conversation += [("agent", step.message, {"asks": step.asks}), ("user", answer)]
    if step.kind == "off_topic":
        return {"status": "rejected", "reason": step.reason}
    if step.kind == "out_of_scope":
        return {"status": "out_of_scope", "reason": step.reason}
    if step.kind == "impossible":
        return {"status": "impossible_geometry", "reason": step.reason}
    if step.kind == "ask":
        return {"status": "incomplete", "gaps": step.gaps, "conflicts": step.conflicts,
                "reason": step.reason,
                "message": ("missing or contradictory must-have info; cannot proceed in --auto mode "
                            "(no fabrication, no guessing)") if auto
                           else f"still unresolved after {max_rounds} rounds"}
    return {
        "status": "complete",
        "resolved": step.resolved,
        "assumptions": step.assumptions,
        "completed_description": step.completed,
        "assessment": step.assessment.model_dump(),
    }


def _print_result(result: dict) -> int:
    print()


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
        print(f"missing must-have: {', '.join(result['gaps']) or '-'}")
        for q, options in result.get("conflicts", {}).items():
            print(f"contradictory {q}: " + " vs ".join(f"'{quote}' = {v}" for v, quote in options))
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

    complete = TaskAssessment(is_fea_task=True, geometry_type="cantilever_beam",
                              geometry_specified=True, L=100, h=10, b=10,
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

    miss = TaskAssessment(is_fea_task=True, geometry_type="cantilever_beam",
                              geometry_specified=True, L=100, h=10, b=10,
                          material_specified=False, fixed_specified=True, load_specified=False)
    print(f"2) missing material+load -> gaps {must_have_gaps(miss)} (expect material, load)")
    ok = ok and set(must_have_gaps(miss)) == {"material", "load"}

    off = TaskAssessment(is_fea_task=False, off_topic_reason="greeting")
    print(f"3) off-topic is_fea_task = {off.is_fea_task} (expect False)")
    ok = ok and off.is_fea_task is False

    alu = TaskAssessment(is_fea_task=True, geometry_type="cantilever_beam",
                              geometry_specified=True, L=100, h=10, b=10,
                         material_specified=True, material_name="Alu",
                         fixed_specified=True, load_specified=True, load_magnitude=1000)
    g_alu = must_have_gaps(alu)
    r_alu, _ = apply_defaults(alu)
    print(f"4) material 'Alu' -> gaps {g_alu} (expect []); canonical -> {r_alu['material_name']} (expect ALUMINUM)")
    ok = ok and g_alu == [] and r_alu["material_name"] == "ALUMINUM"

    unk = TaskAssessment(is_fea_task=True, geometry_type="cantilever_beam",
                              geometry_specified=True, L=100, h=10, b=10,
                         material_specified=True, material_name="INCONEL",
                         fixed_specified=True, load_specified=True, load_magnitude=1000)
    print(f"5) material 'INCONEL' (not in library) -> gaps {must_have_gaps(unk)} (expect ['material'])")
    print(f"   would ask: {_gap_line('material', unk)}")
    ok = ok and must_have_gaps(unk) == ["material"]

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

    # Mentions -> values, decided by code (no API: prepared reports).
    conv = [("user", "cantilever_beam 1000mm long"),
            ("agent", "geometry: length L, height h, width b in mm (e.g. 100 x 10 x 10)", {"asks": []}),
            ("user", "200 x 20 x10, Steel, left side fixed, load on the other side 1500N")]
    def report(*mentions, **kw):
        return LLMReport(is_fea_task=True, geometry_type="cantilever_beam", fixed_specified=True,
                         load_specified=True,
                         mentions=[Mention(quantity=q, value=v if q != "material" else None,
                                           material=v if q == "material" else None,
                                           quote=quote, turn=t) for q, v, quote, t in mentions], **kw)
    stated = [("L", 1000.0, "1000mm long", 1), ("L", 200.0, "200 x 20 x10", 2),
              ("h", 20.0, "200 x 20 x10", 2), ("b", 10.0, "200 x 20 x10", 2),
              ("material", "Steel", "Steel", 2), ("load", 1500.0, "1500N", 2)]
    step = understand(conv, report(*stated))
    print(f"9) length stated twice -> {step.kind}, asks {step.asks}")
    print(f"   {step.message.splitlines()[0]}")
    mention_ok = (step.kind == "ask" and step.asks == ["L"] and "'1000mm long'" in step.message
                  and "'200 x 20 x10'" in step.message and step.gaps == [])

    conv2 = conv + [("agent", step.message, {"asks": step.asks}), ("user", "200 mm")]
    step2 = understand(conv2, report(*stated, ("L", 200.0, "200 mm", 3)))
    r2 = step2.resolved or {}
    print(f"10) answered '200 mm' -> {step2.kind}, L={r2.get('L')} h={r2.get('h')} b={r2.get('b')} (expect ready 200/20/10)")
    mention_ok &= step2.kind == "ready" and (r2.get("L"), r2.get("h"), r2.get("b")) == (200.0, 20.0, 10.0)

    step3 = understand(conv, report(("L", 100.0, "100 x 10 x 10", 2), ("h", 10.0, "100 x 10 x 10", 2),
                                    ("b", 10.0, "100 x 10 x 10", 2), ("material", "Steel", "Steel", 2),
                                    ("load", 1500.0, "1500N", 2)))
    print(f"11) a value quoted from the assistant's example -> dropped {len(step3.dropped)}, "
          f"{step3.kind} {step3.gaps} (expect 3 dropped, ask ['geometry'])")
    mention_ok &= len(step3.dropped) == 3 and step3.kind == "ask" and step3.gaps == ["geometry"]

    step4 = understand([("user", "an alu cantilever 100 x 10 x 10, aluminium, 1 kN at the tip")],
                       report(("material", "alu", "alu", 1), ("material", "aluminium", "aluminium", 1),
                              ("L", 100.0, "100 x 10 x 10", 1), ("h", 10.0, "100 x 10 x 10", 1),
                              ("b", 10.0, "100 x 10 x 10", 1), ("load", 1000.0, "1 kN", 1)))
    print(f"12) 'alu' and 'aluminium' -> {step4.kind}, material {(step4.resolved or {}).get('material_name')} "
          f"(expect ready ALUMINUM: one material, two spellings)")
    mention_ok &= step4.kind == "ready" and step4.resolved["material_name"] == "ALUMINUM"

    step5 = understand([("user", "a beam calculation")],
                       LLMReport(is_fea_task=True, in_scope=True, geometry_type=None))
    print(f"13) 'a beam calculation' -> {step5.kind} {step5.gaps} (expect ask ['geometry_type'], not out of scope)")
    mention_ok &= step5.kind == "ask" and step5.gaps == ["geometry_type"]
    ok = ok and mention_ok

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
