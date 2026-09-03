"""Knowledge base of reference / analytical solutions (plan layer 7).

Each case carries:
  - `embed_text`  : a plain-language description used for VECTOR RETRIEVAL (RAG);
  - `applies`     : a DETERMINISTIC predicate that decides whether the case is
                    actually valid for the given params;
  - `formula`     : a code-callable that computes the reference value;
  - `quantity`    : WHAT that value is, so the verification layer knows which
                    FEM measurement to compare it against (tip deflection for
                    the beam cases, a stress-concentration factor for the plate).

RAG (see vector_store.py) retrieves CANDIDATE cases by embed_text similarity;
`applies` is the deterministic gate that decides which candidate is really used
(see verification.py). Retrieval narrows the field (semantic, fuzzy); `applies`
makes the final call (exact, reads the real params). The LLM/embedding can never
force a wrong formula to be used -- that is how RAG stays inside the iron law.

Two cases are reachable by today's pipeline: the standard cantilever point-load
deflection, and the centered-hole plate SCF. The remaining beam cases are
retrieval DISTRACTORS + future extensions: their `applies` returns False for
current params, so the gate correctly rejects them.

IRON LAW: formulas live ONLY here; the verification layer never invents one.
Unit system: SI-mm (length mm, force N, E in MPa, displacement mm); stress
concentration factors are dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ReferenceCase:
    id: str
    name: str
    applies_to: str        # human-readable applicability summary (for the report)
    applies: Callable      # predicate(params) -> bool  (deterministic selection gate)
    formula: Callable      # function(params) -> reference value [displacement only]
    quantity: str          # what `formula` returns, e.g. "tip_deflection_mm"
    scope_note: str        # explicit validity limits (always printed)
    source: str
    expected_error: str
    embed_text: str = ""   # plain-language description for vector retrieval (RAG)


_OPPOSITE = {
    "face_x_min": "face_x_max", "face_x_max": "face_x_min",
    "face_y_min": "face_y_max", "face_y_max": "face_y_min",
    "face_z_min": "face_z_max", "face_z_max": "face_z_min",
}


def _is_static_linear_cantilever(params: dict) -> bool:
    ana = params["analysis"]
    return (params["geometry"]["type"] == "cantilever_beam"
            and ana["type"] == "static" and not ana.get("nonlinear", False))


# ----- standard cantilever, free-end transverse point load (the REACHABLE case) -----
def _cantilever_point_applies(params: dict) -> bool:
    if not _is_static_linear_cantilever(params):
        return False
    bc = params["bc"]
    if len(bc["fixed"]) != 1 or len(bc["loads"]) != 1:
        return False
    fixed_face = bc["fixed"][0]
    load = bc["loads"][0]
    if fixed_face not in ("face_x_min", "face_x_max"):
        return False
    if load["face"] != _OPPOSITE[fixed_face]:
        return False
    if abs(float(load["dir"][0])) > 1e-9:           # transverse to the beam axis (no axial x)
        return False
    if load.get("type", "point") != "point":        # a point/concentrated load
        return False
    return True


def _cantilever_point_deflection(params: dict) -> float:
    g = params["geometry"]["cantilever_beam"]
    L, h, b = float(g["L"]), float(g["h"]), float(g["b"])
    E = float(params["material"]["E"])
    P = float(params["bc"]["loads"][0]["magnitude"])
    I = b * h ** 3 / 12.0
    return P * L ** 3 / (3.0 * E * I)


# ----- distractors / future extensions: applies() == False for today's params -----
def _cantilever_udl_applies(params: dict) -> bool:
    # Needs a distributed-load representation the schema does not have yet.
    if not _is_static_linear_cantilever(params):
        return False
    return any(ld.get("type") == "distributed" for ld in params["bc"]["loads"])


def _cantilever_udl_deflection(params: dict) -> float:
    g = params["geometry"]["cantilever_beam"]
    L, h, b = float(g["L"]), float(g["h"]), float(g["b"])
    E = float(params["material"]["E"])
    w = float(params["bc"]["loads"][0]["magnitude"]) / L   # load per unit length (future field)
    I = b * h ** 3 / 12.0
    return w * L ** 4 / (8.0 * E * I)


def _two_support_center_applies(params: dict) -> bool:
    # Simply-supported / fixed-fixed need two supports; cantilevers have one.
    return (params["geometry"]["type"] == "cantilever_beam"
            and len(params["bc"]["fixed"]) >= 2)


def _simply_supported_center_deflection(params: dict) -> float:
    g = params["geometry"]["cantilever_beam"]
    L, h, b = float(g["L"]), float(g["h"]), float(g["b"])
    E = float(params["material"]["E"])
    P = float(params["bc"]["loads"][0]["magnitude"])
    I = b * h ** 3 / 12.0
    return P * L ** 3 / (48.0 * E * I)


def _fixed_fixed_center_deflection(params: dict) -> float:
    g = params["geometry"]["cantilever_beam"]
    L, h, b = float(g["L"]), float(g["h"]), float(g["b"])
    E = float(params["material"]["E"])
    P = float(params["bc"]["loads"][0]["magnitude"])
    I = b * h ** 3 / 12.0
    return P * L ** 3 / (192.0 * E * I)


STANDARD_CANTILEVER = ReferenceCase(
    id="cantilever_tip_load_eb",
    name="standard cantilever, free-end transverse point load",
    applies_to=("cantilever_beam; one end fully fixed; free-end transverse point "
                "load; linear elastic; small strain"),
    applies=_cantilever_point_applies,
    formula=_cantilever_point_deflection,
    quantity="tip_deflection_mm",
    scope_note=("small strain, linear elastic, slender beam L/h >> 1 "
                "(Euler-Bernoulli; transverse shear deformation neglected)"),
    source="Euler-Bernoulli beam theory: delta = P L^3 / (3 E I), I = b h^3 / 12",
    expected_error=("< ~1% vs 3D FEA for L/h >= 10 (fixed-face clamp stiffening "
                    "roughly cancels shear softening)"),
    embed_text=("Cantilever beam, one end fully fixed (clamped), a transverse "
                "concentrated point load at the free end. Linear elastic, small "
                "strain, slender beam. Tip deflection by Euler-Bernoulli theory."),
)

CANTILEVER_UDL = ReferenceCase(
    id="cantilever_udl_eb",
    name="cantilever, uniformly distributed transverse load",
    applies_to="cantilever_beam; one end fully fixed; uniformly distributed transverse load; linear elastic",
    applies=_cantilever_udl_applies,
    formula=_cantilever_udl_deflection,
    quantity="tip_deflection_mm",
    scope_note="small strain, linear elastic, slender beam; uniformly distributed load w per unit length",
    source="Euler-Bernoulli: delta = w L^4 / (8 E I)",
    expected_error="future / not reachable by the current point-load pipeline",
    embed_text=("Cantilever beam, one end fully fixed, a uniformly distributed "
                "transverse load along its whole length. Linear elastic, small "
                "strain, slender beam. Tip deflection w L^4 / 8 E I."),
)

SIMPLY_SUPPORTED_CENTER = ReferenceCase(
    id="simply_supported_center_eb",
    name="simply supported beam, center point load",
    applies_to="beam on two simple supports; concentrated load at mid-span; linear elastic",
    applies=_two_support_center_applies,
    formula=_simply_supported_center_deflection,
    quantity="center_deflection_mm",
    scope_note="small strain, linear elastic, slender beam; supported at both ends",
    source="Euler-Bernoulli: delta = P L^3 / (48 E I)",
    expected_error="future / not reachable by the current single-fixed-end pipeline",
    embed_text=("Simply supported beam resting on two supports, a concentrated "
                "point load at the center of the span. Linear elastic, small "
                "strain, slender beam. Mid-span deflection P L^3 / 48 E I."),
)

FIXED_FIXED_CENTER = ReferenceCase(
    id="fixed_fixed_center_eb",
    name="clamped-clamped beam, center point load",
    applies_to="beam fully fixed at both ends; concentrated load at mid-span; linear elastic",
    applies=_two_support_center_applies,
    formula=_fixed_fixed_center_deflection,
    quantity="center_deflection_mm",
    scope_note="small strain, linear elastic, slender beam; clamped at both ends",
    source="Euler-Bernoulli: delta = P L^3 / (192 E I)",
    expected_error="future / not reachable by the current single-fixed-end pipeline",
    embed_text=("Beam fully fixed (clamped) at both ends, a concentrated point "
                "load at the center. Linear elastic, small strain, slender beam. "
                "Mid-span deflection P L^3 / 192 E I."),
)

# ----- finite-width plate with a central circular hole, uniaxial tension -----
# Peterson's Chart 4.1 polynomial fit to Howland's series solution. Returns the
# NET-section factor Ktn: sigma_max = Ktn * sigma_net, sigma_net = F/((b-2r)*t).
# The net-section definition is what makes Ktn -> 2 as the hole fills the width
# (and -> 3, the Kirsch infinite-plate value, as the hole becomes small).
PLATE_HOLE_LAMBDA_MAX = 0.6      # validity limit of the polynomial fit (2r/b)
PLATE_HOLE_CENTER_TOL = 0.02     # |xc - a/2| <= tol*a  -> "centered" for the fit
PLATE_HOLE_END_CLEARANCE = 0.7   # hole must sit >= 0.7*b from the loaded ends


def plate_hole_ktn(lam: float) -> float:
    """Howland/Peterson net-section SCF for a central hole, uniaxial tension."""
    return 3.00 - 3.14 * lam + 3.667 * lam ** 2 - 1.527 * lam ** 3


def _plate_hole_scf_applies(params: dict) -> bool:
    if params["geometry"]["type"] != "plate_with_hole":
        return False
    ana = params["analysis"]
    if ana["type"] != "static" or ana.get("nonlinear", False):
        return False
    g = params["geometry"].get("plate_with_hole")
    if not g:
        return False
    a, b, r = float(g["a"]), float(g["b"]), float(g["r"])
    xc, yc = float(g["xc"]), float(g["yc"])

    # 1. The fit is for a hole on the strip's centerline, midway along it.
    if abs(xc - a / 2.0) > PLATE_HOLE_CENTER_TOL * a:
        return False
    if abs(yc - b / 2.0) > PLATE_HOLE_CENTER_TOL * b:
        return False
    # 2. Polynomial validity range.
    if 2.0 * r / b > PLATE_HOLE_LAMBDA_MAX:
        return False
    # 3. Howland's strip is infinitely long: the hole must be far enough from
    #    the loaded / supported ends for their disturbance to have died out.
    if min(xc, a - xc) - r < PLATE_HOLE_END_CLEARANCE * b:
        return False
    # 4. A single uniaxial TENSION load along +x on the far edge, supported at
    #    the near edge -- any other configuration is a different problem.
    bc = params["bc"]
    if len(bc["loads"]) != 1:
        return False
    ld = bc["loads"][0]
    if ld["face"] != "edge_x_max":
        return False
    d = [float(c) for c in ld["dir"]]
    if d[0] <= 0 or abs(d[1]) > 1e-9 or abs(d[2]) > 1e-9:
        return False
    fixed_faces = {f if isinstance(f, str) else f["face"] for f in bc["fixed"]}
    if fixed_faces != {"edge_x_min"}:
        return False
    return True


def _plate_hole_scf(params: dict) -> float:
    g = params["geometry"]["plate_with_hole"]
    return plate_hole_ktn(2.0 * float(g["r"]) / float(g["b"]))


PLATE_HOLE_TENSION_SCF = ReferenceCase(
    id="plate_central_hole_tension_scf",
    name="finite-width plate with a central circular hole, uniaxial tension",
    applies_to=("plate_with_hole; hole centered in a strip under uniaxial "
                "tension; 2r/b <= 0.6; hole >= 0.7*b clear of the loaded ends; "
                "linear elastic plane stress"),
    applies=_plate_hole_scf_applies,
    formula=_plate_hole_scf,
    quantity="stress_concentration_factor_Ktn",
    scope_note=("net-section SCF: sigma_max = Ktn * sigma_net with "
                "sigma_net = F/((b-2r)*t); central hole only, 2r/b <= 0.6; "
                "Howland's strip is infinitely long, so end effects must have "
                "decayed at the hole; small strain, linear elastic plane stress"),
    source=("Peterson's Stress Concentration Factors, 3rd ed. (Pilkey & Pilkey, "
            "Wiley 2008), Ch. 4 'Holes', Chart 4.1: "
            "Ktn = 3.00 - 3.14*lam + 3.667*lam^2 - 1.527*lam^3, lam = 2r/b -- "
            "a fit to Howland (1930), Phil. Trans. R. Soc. A 229, 49-86; "
            "lam -> 0 recovers the Kirsch (1898) infinite-plate value Kt = 3"),
    expected_error=("fit itself ~1% for lam <= 0.6; a converged plane-stress FEA "
                    "sits ~0.6-0.9% above the fit for lam = 0.4"),
    embed_text=("Rectangular flat plate or strip of finite width with a circular "
                "hole at its center, pulled in uniaxial tension along its length. "
                "Linear elastic plane stress. Peak tangential stress at the hole "
                "edge on the transverse diameter; stress concentration factor "
                "relative to the net section through the hole."),
)

KNOWLEDGE_BASE = [STANDARD_CANTILEVER, CANTILEVER_UDL, SIMPLY_SUPPORTED_CENTER,
                  FIXED_FIXED_CENTER, PLATE_HOLE_TENSION_SCF]
_BY_ID = {c.id: c for c in KNOWLEDGE_BASE}


def get_case(case_id: str):
    """Look up a ReferenceCase by id (the bridge from a retrieved id to its code)."""
    return _BY_ID.get(case_id)


def select_reference_case(params: dict, knowledge_base=KNOWLEDGE_BASE):
    """Deterministic fallback selection: the first case whose predicate matches,
    else None. (vector_store / verification use RAG retrieval + this same gate.)"""
    for case in knowledge_base:
        try:
            if case.applies(params):
                return case
        except Exception:  # noqa: BLE001
            continue
    return None
