#!/usr/bin/env python3
"""Parameterization layer 2 (plan section 2): Pydantic validation of PARAMS.

This is the DETERMINISTIC half of the parameterization layer -- no LLM, no API.
It defines the schema that constrains what may go into params.py and validates a
PARAMS dict: types, signs (E>0, L>0, ...), ranges (nu in (0,0.5)), allowed
enums/labels, and structural rules (geometry.type must carry its own parameter
subset). `extra="forbid"` everywhere enforces "predefined slots only" so a typo
or an invented field is rejected rather than silently ignored.

It is the foundation for the `set_params` Tool: the LLM (via OpenRouter, an
OpenAI-compatible endpoint) will only be able to fill these slots, and code --
not the LLM -- decides whether the values are admissible. Numerical correctness
stays with deterministic code (the iron law).

Soft "sensibility" issues that should not block (e.g. a stubby L/h ratio) are
returned as WARNINGS, not errors.

Unit system: SI-mm.
"""

from __future__ import annotations

import sys
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Neutral geometry boundary labels (naming contract, cantilever plan section 0.2
# and plate plan section 0.2). BC references must point at a label belonging to
# the ACTIVE geometry type; checked in the root model so cross-geometry typos
# (e.g. a 3D face label on a 2D plate) fail before meshing.
VALID_FACES = (
    "face_x_min", "face_x_max",
    "face_y_min", "face_y_max",
    "face_z_min", "face_z_max",
)
PLATE_EDGE_LABELS = (
    "edge_x_min", "edge_x_max",
    "edge_y_min", "edge_y_max",
    "hole_boundary",
)
BOUNDARY_LABELS = {
    "cantilever_beam": VALID_FACES,       # 3D solid: boundary entities are faces
    "plate_with_hole": PLATE_EDGE_LABELS, # 2D plane stress: boundary entities are edges
}
# Meshing strategies admissible per geometry (mesh.py enforces what is implemented).
GEOMETRY_ELEM_SHAPES = {
    "cantilever_beam": ("hex", "tet"),
    "plate_with_hole": ("quad", "tri"),
}
SUPPORTED_GEOMETRIES = tuple(BOUNDARY_LABELS)  # extend by adding a subset model + label set

# Thinnest strip of material the pipeline will accept between a hole and a plate
# edge. Derived, not chosen: the mesh layer puts LIGAMENT_MIN_ELEMS across the
# ligament and will not refine finer than elem_size / MAX_REFINE_RATIO, so
# anything thinner cannot be meshed at a bounded cost. Expressed against the
# default far-field size so validation does not have to know the mesh settings.
_DEFAULT_ELEM_SIZE_FRACTION = 25.0   # far-field elem_size ~ min(a, b) / 25
_LIGAMENT_MIN_ELEMS = 6
_MAX_REFINE_RATIO = 20


def min_usable_ligament(a: float, b: float) -> float:
    """Smallest hole-to-edge material width the mesh layer can resolve."""
    elem_size = min(a, b) / _DEFAULT_ELEM_SIZE_FRACTION
    return _LIGAMENT_MIN_ELEMS * elem_size / _MAX_REFINE_RATIO

# Plausible SI-mm magnitude bands. Out-of-band values are WARNINGS (not hard
# limits): they flag a likely unit error for a human to judge, without blocking.
# (With the materials library the LLM no longer fills these -- this is a safety
# net for the library itself and for hand-edited params.)
E_BAND_MPA = (1.0e3, 1.0e6)             # structural materials, MPa  (steel ~ 2.1e5)
RHO_BAND_TONNE_MM3 = (5.0e-10, 5.0e-8)  # solids, tonne/mm^3         (steel ~ 7.85e-9)


class AnalysisParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["static", "fatigue"] = "static"
    nonlinear: bool = False


class CantileverBeamGeom(BaseModel):
    model_config = ConfigDict(extra="forbid")
    L: float = Field(gt=0)   # mm, beam length (x)
    h: float = Field(gt=0)   # mm, height (z, load direction)
    b: float = Field(gt=0)   # mm, width (y)


class PlateWithHoleGeom(BaseModel):
    """Rectangular plate with a circular hole, 2D plane stress (plate plan 0.1).

    Plate occupies x in [0, a], y in [0, b] in the z = 0 plane (corner origin).
    Hole center = (xc, yc) in the same corner-origin coordinates; the centered
    Kirsch/Howland baseline is xc = a/2, yc = b/2.
    """
    model_config = ConfigDict(extra="forbid")
    a: float = Field(gt=0)   # mm, plate length along x (load direction)
    b: float = Field(gt=0)   # mm, plate width along y
    t: float = Field(gt=0)   # mm, thickness (plane stress *SOLID SECTION)
    r: float = Field(gt=0)   # mm, hole radius
    xc: float = Field(gt=0)  # mm, hole-center x (corner origin; a/2 -> centered)
    yc: float = Field(gt=0)  # mm, hole-center y (corner origin; b/2 -> centered)

    def ligaments(self) -> dict:
        """Material widths between the hole and each plate edge (mm)."""
        return {
            "x_min": self.xc - self.r, "x_max": self.a - self.xc - self.r,
            "y_min": self.yc - self.r, "y_max": self.b - self.yc - self.r,
        }

    @model_validator(mode="after")
    def _hole_leaves_usable_material(self):
        """The hole must leave a ligament the pipeline can actually analyse.

        "> 0" is a mathematical condition, not an engineering one: a hole
        0.001 mm from the edge passes it, and is not a plate with a hole at
        all. It also has a concrete failure mode -- the mesh policy puts a
        fixed number of elements across the thinnest ligament, so a vanishing
        ligament asks for a vanishing element size, which does not fail
        loudly, it just never finishes. The floor here is the same one the
        mesh layer enforces (mesh.MAX_REFINE_RATIO), so what validation
        accepts and what the mesher can build are the same set.
        """
        lig = self.ligaments()
        bad = {k: v for k, v in lig.items() if v <= 0.0}
        if bad:
            raise ValueError(
                f"hole (r={self.r}, center ({self.xc}, {self.yc})) does not fit inside "
                f"the {self.a} x {self.b} mm plate: it crosses the edge on "
                f"{sorted(bad)} (ligament {min(bad.values()):.4g} mm). "
                f"A hole of radius {self.r} needs a plate at least "
                f"{2 * self.r:.4g} mm across in each direction."
            )
        worst = min(lig.values())
        floor = min_usable_ligament(self.a, self.b)
        if worst < floor:
            raise ValueError(
                f"hole (r={self.r}, center ({self.xc}, {self.yc})) leaves only "
                f"{worst:.4g} mm of material at its thinnest point, below the "
                f"{floor:.4g} mm this pipeline can mesh and analyse. Move the hole "
                f"away from the edge or reduce its radius."
            )
        return self


class GeometryParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    cantilever_beam: Optional[CantileverBeamGeom] = None
    plate_with_hole: Optional[PlateWithHoleGeom] = None

    @model_validator(mode="after")
    def _subset_present_and_supported(self):
        if self.type not in SUPPORTED_GEOMETRIES:
            raise ValueError(
                f"geometry.type={self.type!r} not supported in the MVP "
                f"(supported: {list(SUPPORTED_GEOMETRIES)})"
            )
        if getattr(self, self.type, None) is None:
            raise ValueError(f"geometry.type={self.type!r} requires a '{self.type}' parameter subset")
        return self


class MaterialParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "STEEL"
    E: float = Field(gt=0)            # MPa
    nu: float = Field(gt=0, lt=0.5)   # Poisson ratio, physical range (0, 0.5)
    rho: float = Field(gt=0)          # tonne/mm^3


class LoadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    face: str                         # boundary entity label: a face (3D) or an edge (2D)
    dir: list[float]
    magnitude: float = Field(gt=0)    # N; sign of the load is carried by `dir`

    @field_validator("dir")
    @classmethod
    def _dir_is_3d_nonzero(cls, v: list) -> list:
        if len(v) != 3:
            raise ValueError("dir must have exactly 3 components")
        if all(abs(float(c)) < 1e-12 for c in v):
            raise ValueError("dir must not be the zero vector")
        return v


class FixedSpec(BaseModel):
    """DOF-level constraint on a boundary entity (plate plan 0.3).

    `at="all"` constrains every node of the entity; `at="mid"` constrains only
    the single node nearest the entity midpoint (a rigid-body anchor). The plain
    string form in `BCParams.fixed` remains the full clamp of all translations.
    """
    model_config = ConfigDict(extra="forbid")
    face: str
    dofs: list[int] = Field(min_length=1)
    at: Literal["all", "mid"] = "all"

    @field_validator("dofs")
    @classmethod
    def _dofs_valid(cls, v: list) -> list:
        bad = [d for d in v if d not in (1, 2, 3)]
        if bad:
            raise ValueError(f"dofs must be translations 1..3; got {bad}")
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate dofs in {v}")
        return v


def fixed_entry_face(entry) -> str:
    """Boundary label of a `bc.fixed` entry (str or FixedSpec/dict)."""
    if isinstance(entry, str):
        return entry
    return entry["face"] if isinstance(entry, dict) else entry.face


def fixed_entry_dofs(entry) -> tuple:
    """Constrained DOFs of a `bc.fixed` entry (str form == full clamp 1-3)."""
    if isinstance(entry, str):
        return (1, 2, 3)
    dofs = entry["dofs"] if isinstance(entry, dict) else entry.dofs
    return tuple(int(d) for d in dofs)


def fixed_entry_at(entry) -> str:
    """Node-selection mode of a `bc.fixed` entry: "all" or "mid" (str form == "all")."""
    if isinstance(entry, str):
        return "all"
    at = entry.get("at", "all") if isinstance(entry, dict) else entry.at
    return str(at)


class BCParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # str  -> full clamp of all translational DOFs on that entity (MVP form)
    # dict -> DOF-level constraint (FixedSpec), used by the plate roller + anchor
    fixed: list[Union[str, FixedSpec]] = Field(min_length=1)
    loads: list[LoadSpec] = Field(min_length=1)


class MeshParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elem_size: float = Field(gt=0)        # mm, target edge length
    elem_order: Literal[1, 2] = 2
    # hex/tet are 3D strategies (cantilever), quad/tri are 2D (plate); the
    # geometry cross-check lives in the root model. mesh.py enforces which
    # strategies are actually implemented.
    elem_shape: Literal["hex", "tet", "quad", "tri"] = "hex"


class AcceptanceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tol_disp: float = Field(gt=0, le=1)   # fractional displacement tolerance (0.05 == 5%)
    # Stress-concentration-factor tolerance, used ONLY by geometries whose stress
    # field is smooth and reference-backed (plate_with_hole vs the Howland fit,
    # plate plan D3/M7). The cantilever keeps stress out of auto-judgment.
    # Default set from the M7 measurement across the sampling range (worst
    # observed deviation 0.59%); see the provenance comment in params.py.
    tol_scf: float = Field(default=0.03, gt=0, le=1)


class Params(BaseModel):
    """Root schema mirroring params.py PARAMS (plan section 0.4)."""
    model_config = ConfigDict(extra="forbid")
    analysis: AnalysisParams
    geometry: GeometryParams
    material: MaterialParams
    bc: BCParams
    mesh: MeshParams
    acceptance: AcceptanceParams

    @model_validator(mode="after")
    def _bc_and_mesh_match_geometry(self):
        gtype = self.geometry.type
        labels = BOUNDARY_LABELS[gtype]

        # Boundary labels must belong to the active geometry's label set.
        referenced = ([fixed_entry_face(e) for e in self.bc.fixed]
                      + [ld.face for ld in self.bc.loads])
        bad = sorted({f for f in referenced if f not in labels})
        if bad:
            raise ValueError(
                f"boundary label(s) {bad} not valid for geometry.type={gtype!r} "
                f"(valid: {list(labels)})"
            )

        # Meshing strategy must match the geometry's dimensionality.
        shapes = GEOMETRY_ELEM_SHAPES[gtype]
        if self.mesh.elem_shape not in shapes:
            raise ValueError(
                f"mesh.elem_shape={self.mesh.elem_shape!r} not admissible for "
                f"geometry.type={gtype!r} (admissible: {list(shapes)})"
            )

        if gtype == "plate_with_hole":
            # 2D model in the x-y plane: loads may not have a z component.
            for ld in self.bc.loads:
                if abs(float(ld.dir[2])) > 1e-12:
                    raise ValueError(
                        f"plate_with_hole is 2D in the x-y plane; load dir {ld.dir} "
                        f"must have a zero z component"
                    )
            # Necessary in-plane well-posedness: DOFs 1 and 2 must each be
            # constrained somewhere (else a rigid-body mode; e.g. a roller with
            # no y anchor). The full check incl. rotation needs node coordinates
            # and lives in boundary_conditions.py (plate plan M3).
            dof_union = set()
            for e in self.bc.fixed:
                dof_union.update(fixed_entry_dofs(e))
            if not {1, 2} <= dof_union:
                raise ValueError(
                    f"plate constraints leave an in-plane rigid-body mode: constrained "
                    f"DOF union {sorted(dof_union)} must include 1 and 2 (e.g. add an "
                    f"anchor {{'face': 'edge_x_min', 'dofs': [2], 'at': 'mid'}})"
                )
        return self


def _warnings(p: Params) -> list:
    """Soft sensibility checks that should NOT block validation (plan section 2)."""
    warns = []
    if p.geometry.type == "cantilever_beam" and p.geometry.cantilever_beam is not None:
        g = p.geometry.cantilever_beam
        slender = g.L / max(g.h, g.b)
        if slender < 5:
            warns.append(f"L/max(h,b) = {slender:.1f} < 5: beam theory is unreliable here; "
                         f"the analytical displacement reference may not apply")
        elif slender < 10:
            warns.append(f"L/max(h,b) = {slender:.1f}: slenderness is borderline "
                         f"(shear deformation becomes non-negligible)")
    if p.geometry.type == "plate_with_hole" and p.geometry.plate_with_hole is not None:
        g = p.geometry.plate_with_hole
        elem_size = p.mesh.elem_size
        lam = 2.0 * g.r / g.b                      # net-section ratio d/W (load along x)
        min_lig = min(g.ligaments().values())
        if lam > 0.6:
            warns.append(f"2r/b = {lam:.2f} > 0.6: outside the Howland-fit validity range; "
                         f"the SCF auto-check will report N/A (recorded for human review)")
        if min_lig < 0.1 * min(g.a, g.b):
            warns.append(f"min ligament = {min_lig:.2f} mm < 10% of min(a, b): thin ligament, "
                         f"expect steep gradients there")
        if min_lig < 2.0 * elem_size:
            warns.append(f"min ligament = {min_lig:.2f} mm < 2 x elem_size ({elem_size} mm): "
                         f"the mesh cannot resolve the ligament; refine elem_size")
        if elem_size > g.r / 3.0:
            warns.append(f"elem_size = {elem_size} mm > r/3 ({g.r / 3.0:.2f} mm): hole "
                         f"under-resolved globally (mesh layer refines locally per policy; "
                         f"verify the hole element count in the mesh report)")
        if g.t > 0.2 * min(g.a, g.b):
            warns.append(f"t = {g.t} mm > 20% of min(a, b): plane-stress thin-plate "
                         f"assumption is questionable for this thickness")
        end_dist = min(g.xc, g.a - g.xc) - g.r
        if end_dist < 0.7 * g.b:
            warns.append(f"hole is close to the loaded/constrained ends "
                         f"(min(xc, a-xc) - r = {end_dist:.1f} mm < 0.7*b): "
                         f"end effects reach the hole; the Howland reference may not apply")
    if p.material.nu >= 0.49:
        warns.append(f"nu = {p.material.nu} is near 0.5 (near-incompressible): risk of element locking")
    E = p.material.E
    if not (E_BAND_MPA[0] <= E <= E_BAND_MPA[1]):
        warns.append(f"E = {E:g} MPa is outside the usual SI-mm band "
                     f"[{E_BAND_MPA[0]:g}, {E_BAND_MPA[1]:g}] -- check units "
                     f"(MPa = N/mm^2; steel ~ 210000; did you give GPa or Pa?)")
    rho = p.material.rho
    if not (RHO_BAND_TONNE_MM3[0] <= rho <= RHO_BAND_TONNE_MM3[1]):
        warns.append(f"rho = {rho:g} tonne/mm^3 is outside the usual SI-mm band "
                     f"[{RHO_BAND_TONNE_MM3[0]:g}, {RHO_BAND_TONNE_MM3[1]:g}] -- check units "
                     f"(steel ~ 7.85e-9; did you give g/cm^3 ~ 7.85?)")
    return warns


def validate_params(d: dict) -> tuple:
    """Validate a PARAMS dict. Returns (Params, warnings). Raises ValidationError."""
    p = Params.model_validate(d)
    return p, _warnings(p)


def _run_reject_battery(base: dict, cases: dict) -> bool:
    """Each mutation of `base` MUST be rejected by validation."""
    import copy
    ok = True
    for name, mutate in cases.items():
        d = copy.deepcopy(base)
        mutate(d)
        try:
            validate_params(d)
            print(f"  [FAIL] NOT rejected: {name}")
            ok = False
        except ValidationError:
            print(f"  [PASS] rejected: {name}")
    return ok


def _run_warn_battery(base: dict, cases: dict) -> bool:
    """Each mutation of `base` must validate but produce at least one WARNING."""
    import copy
    ok = True
    for name, mutate in cases.items():
        d = copy.deepcopy(base)
        mutate(d)
        try:
            _, ws = validate_params(d)
        except ValidationError:
            print(f"  [FAIL] {name}: hard-rejected (expected a warning)")
            ok = False
            continue
        if ws:
            print(f"  [PASS] {name}: {ws[0]}")
        else:
            print(f"  [FAIL] {name}: produced no warning")
            ok = False
    return ok


def _selftest() -> bool:
    """Self-acceptance run (plan layer 2 + plate plan M1): every prepared case in
    params.CASES must validate, and per-geometry batteries of malformed inputs
    must each be rejected / suspect values must warn."""
    from params import CASES

    print("=" * 64)
    print("schema.py acceptance (plan layer 2 + plate plan M1)")
    print("=" * 64)

    print("1) validate every prepared case in params.CASES:")
    for cname, cparams in CASES.items():
        try:
            _, warns = validate_params(cparams)
        except ValidationError as exc:
            print(f"  [FAIL] case {cname!r} did not validate:")
            print(exc)
            return False
        print(f"  [PASS] case {cname!r} validates against the schema")
        for w in warns:
            print(f"  [WARN] {w}")

    beam = CASES["cantilever_beam"]
    plate = CASES["plate_with_hole"]

    print("2) malformed cantilever inputs (each MUST be rejected):")
    beam_reject = _run_reject_battery(beam, {
        "E <= 0": lambda d: d["material"].update(E=-1.0),
        "nu >= 0.5": lambda d: d["material"].update(nu=0.5),
        "nu <= 0": lambda d: d["material"].update(nu=0.0),
        "L <= 0": lambda d: d["geometry"]["cantilever_beam"].update(L=0.0),
        "unsupported geometry.type": lambda d: d["geometry"].update(type="spaceship"),
        "missing geometry subset": lambda d: d["geometry"].pop("cantilever_beam"),
        "bad fixed face label": lambda d: d["bc"].update(fixed=["face_q_min"]),
        "zero load direction": lambda d: d["bc"]["loads"][0].update(dir=[0, 0, 0]),
        "unknown slot (typo)": lambda d: d["mesh"].update(elem_sizee=1.0),
        "tol_disp > 1": lambda d: d["acceptance"].update(tol_disp=2.0),
        "bad elem_order": lambda d: d["mesh"].update(elem_order=3),
        "2D edge label on the 3D beam": lambda d: d["bc"].update(fixed=["edge_x_min"]),
        "2D elem_shape on the 3D beam": lambda d: d["mesh"].update(elem_shape="quad"),
    })

    print("3) malformed plate inputs (each MUST be rejected):")
    plate_reject = _run_reject_battery(plate, {
        "hole crosses the x_max edge (xc too big)":
            lambda d: d["geometry"]["plate_with_hole"].update(xc=190.0),
        "hole crosses the y_max edge (yc too big)":
            lambda d: d["geometry"]["plate_with_hole"].update(yc=90.0),
        "hole crosses the x_min edge (xc too small)":
            lambda d: d["geometry"]["plate_with_hole"].update(xc=15.0),
        "hole radius r <= 0": lambda d: d["geometry"]["plate_with_hole"].update(r=0.0),
        "thickness t <= 0": lambda d: d["geometry"]["plate_with_hole"].update(t=-1.0),
        "unknown edge label":
            lambda d: d["bc"]["fixed"].__setitem__(0, {"face": "edge_q_min", "dofs": [1]}),
        "3D face label on the 2D plate": lambda d: d["bc"]["loads"][0].update(face="face_x_max"),
        "invalid dof 4":
            lambda d: d["bc"]["fixed"].__setitem__(0, {"face": "edge_x_min", "dofs": [4]}),
        "empty dofs list":
            lambda d: d["bc"]["fixed"].__setitem__(0, {"face": "edge_x_min", "dofs": []}),
        "roller without a y anchor (rigid-body mode)":
            lambda d: d["bc"].update(fixed=[{"face": "edge_x_min", "dofs": [1]}]),
        "load dir out of plane (z component)":
            lambda d: d["bc"]["loads"][0].update(dir=[0, 0, -1]),
        "3D elem_shape on the 2D plate": lambda d: d["mesh"].update(elem_shape="hex"),
        "typo slot in the plate subset":
            lambda d: d["geometry"]["plate_with_hole"].update(radius=5.0),
        "bad 'at' value":
            lambda d: d["bc"]["fixed"].__setitem__(
                0, {"face": "edge_x_min", "dofs": [1], "at": "corner"}),
    })

    print("4) suspect-but-valid values must WARN (not reject) -- a human then decides:")
    print("  -- cantilever --")
    beam_warn = _run_warn_battery(beam, {
        "rho as g/cm^3 (7.85)": lambda d: d["material"].update(rho=7.85),
        "E as GPa (210)": lambda d: d["material"].update(E=210.0),
        "stubby beam (L=20, L/h=2)": lambda d: d["geometry"]["cantilever_beam"].update(L=20.0),
    })
    print("  -- plate --")
    plate_warn = _run_warn_battery(plate, {
        "big hole 2r/b = 0.7 (outside Howland fit)":
            lambda d: d["geometry"]["plate_with_hole"].update(r=35.0),
        "thin ligament (yc=75 with r=20)":
            lambda d: d["geometry"]["plate_with_hole"].update(yc=75.0),
        "thick plate (t=30, plane stress questionable)":
            lambda d: d["geometry"]["plate_with_hole"].update(t=30.0),
        "coarse mesh vs hole (elem_size=10, r=10)":
            lambda d: (d["geometry"]["plate_with_hole"].update(r=10.0),
                       d["mesh"].update(elem_size=10.0)),
    })

    print("-" * 64)
    ok = beam_reject and plate_reject and beam_warn and plate_warn
    print(f"ACCEPTANCE: {'PASS' if ok else 'FAIL'}")
    print("=" * 64)
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
