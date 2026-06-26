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
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Neutral geometry face labels (naming contract, plan section 0.2). BC references
# must point at one of these; checked here so typos fail before meshing.
VALID_FACES = (
    "face_x_min", "face_x_max",
    "face_y_min", "face_y_max",
    "face_z_min", "face_z_max",
)
SUPPORTED_GEOMETRIES = ("cantilever_beam",)  # MVP scope; extend by adding a subset model

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


class GeometryParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    cantilever_beam: Optional[CantileverBeamGeom] = None
    # Future: plate_with_hole: Optional[PlateWithHoleGeom] = None, etc.

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
    face: str
    dir: list[float]
    magnitude: float = Field(gt=0)    # N; sign of the load is carried by `dir`

    @field_validator("face")
    @classmethod
    def _face_known(cls, v: str) -> str:
        if v not in VALID_FACES:
            raise ValueError(f"load face {v!r} not a valid label {list(VALID_FACES)}")
        return v

    @field_validator("dir")
    @classmethod
    def _dir_is_3d_nonzero(cls, v: list) -> list:
        if len(v) != 3:
            raise ValueError("dir must have exactly 3 components")
        if all(abs(float(c)) < 1e-12 for c in v):
            raise ValueError("dir must not be the zero vector")
        return v


class BCParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fixed: list[str] = Field(min_length=1)
    loads: list[LoadSpec] = Field(min_length=1)

    @field_validator("fixed")
    @classmethod
    def _fixed_faces_known(cls, v: list) -> list:
        bad = [f for f in v if f not in VALID_FACES]
        if bad:
            raise ValueError(f"fixed faces {bad} not valid labels {list(VALID_FACES)}")
        return v


class MeshParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elem_size: float = Field(gt=0)        # mm, target edge length
    elem_order: Literal[1, 2] = 2
    elem_shape: Literal["hex", "tet"] = "hex"   # "tet" reserved (mesh.py raises for now)


class AcceptanceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tol_disp: float = Field(gt=0, le=1)   # fractional displacement tolerance (0.05 == 5%)


class Params(BaseModel):
    """Root schema mirroring params.py PARAMS (plan section 0.4)."""
    model_config = ConfigDict(extra="forbid")
    analysis: AnalysisParams
    geometry: GeometryParams
    material: MaterialParams
    bc: BCParams
    mesh: MeshParams
    acceptance: AcceptanceParams


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


def _selftest() -> bool:
    """Self-acceptance run (plan section 2): the real PARAMS must validate, and a
    battery of malformed inputs must each be rejected."""
    import copy

    from params import PARAMS

    print("=" * 64)
    print("schema.py acceptance (plan layer 2)")
    print("=" * 64)

    print("1) validate the current params.py PARAMS:")
    try:
        _, warns = validate_params(PARAMS)
    except ValidationError as exc:
        print("  [FAIL] PARAMS did not validate:")
        print(exc)
        return False
    print("  [PASS] PARAMS validates against the schema")
    for w in warns:
        print(f"  [WARN] {w}")
    if not warns:
        print("  (no warnings)")

    print("2) malformed inputs (each MUST be rejected):")
    cases = {
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
    }
    all_rejected = True
    for name, mutate in cases.items():
        d = copy.deepcopy(PARAMS)
        mutate(d)
        try:
            validate_params(d)
            print(f"  [FAIL] NOT rejected: {name}")
            all_rejected = False
        except ValidationError:
            print(f"  [PASS] rejected: {name}")

    print("3) suspect-but-valid values must WARN (not reject) -- a human then decides:")
    warn_cases = {
        "rho as g/cm^3 (7.85)": lambda d: d["material"].update(rho=7.85),
        "E as GPa (210)": lambda d: d["material"].update(E=210.0),
        "stubby beam (L=20, L/h=2)": lambda d: d["geometry"]["cantilever_beam"].update(L=20.0),
    }
    all_warned = True
    for name, mutate in warn_cases.items():
        d = copy.deepcopy(PARAMS)
        mutate(d)
        try:
            _, ws = validate_params(d)
        except ValidationError:
            print(f"  [FAIL] {name}: hard-rejected (expected a warning)")
            all_warned = False
            continue
        if ws:
            print(f"  [PASS] {name}: {ws[0]}")
        else:
            print(f"  [FAIL] {name}: produced no warning")
            all_warned = False

    print("-" * 64)
    ok = all_rejected and all_warned
    print(f"ACCEPTANCE: {'PASS' if ok else 'FAIL'}")
    print("=" * 64)
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
