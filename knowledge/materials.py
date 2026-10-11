"""SI-mm materials library (plan layer 2 support).

Curated structural-material properties so the LLM never fills E / nu / rho
numbers: it only NAMES the material (semantic), and deterministic code looks up
the validated SI-mm values here. Same iron-law pattern as the reference-solution
knowledge base -- the LLM selects, code provides the numbers. This removes the
"dropped e-9" unit risk from the LLM entirely.

A named material that is NOT in this library is a hard stop: the pipeline asks
the human to add it here (in SI-mm) rather than fabricating properties.

Unit system: SI-mm. E in MPa (N/mm^2); nu dimensionless; rho in tonne/mm^3.
"""

from __future__ import annotations

# name -> SI-mm properties. Representative engineering values; refine as needed.
MATERIALS = {
    "STEEL":           {"E": 210000.0, "nu": 0.30, "rho": 7.85e-9},
    "STAINLESS_STEEL": {"E": 193000.0, "nu": 0.30, "rho": 8.00e-9},
    "ALUMINUM":        {"E": 69000.0,  "nu": 0.33, "rho": 2.70e-9},
    "COPPER":          {"E": 117000.0, "nu": 0.34, "rho": 8.96e-9},
    "BRASS":           {"E": 100000.0, "nu": 0.34, "rho": 8.50e-9},
    "TITANIUM":        {"E": 114000.0, "nu": 0.32, "rho": 4.43e-9},
    "CAST_IRON":       {"E": 120000.0, "nu": 0.26, "rho": 7.20e-9},
    "MAGNESIUM":       {"E": 45000.0,  "nu": 0.35, "rho": 1.74e-9},
}

# Yield strength fy [MPa], used only to check that a linear-elastic result has
# stayed elastic (verification.py). Each value says which grade it stands for
# and where it comes from; a material without an entry is reported as "not
# checked", never given a guessed value.
YIELD = {
    "STEEL": (355.0, "STEEL taken as structural steel S355: fy = 355 MPa for t <= 40 mm "
                     "(EN 1993-1-1:2005, Table 3.1)"),
}

# Common aliases / spellings -> canonical key (the LLM usually normalizes names,
# this is just a safety net).
_ALIASES = {
    "MILD_STEEL": "STEEL", "CARBON_STEEL": "STEEL", "A36": "STEEL",
    "STAINLESS": "STAINLESS_STEEL", "SS": "STAINLESS_STEEL",
    "AL": "ALUMINUM", "ALU": "ALUMINUM", "ALUM": "ALUMINUM", "ALUMINIUM": "ALUMINUM",
    "CU": "COPPER", "TI": "TITANIUM", "MG": "MAGNESIUM",
}


def _normalize(name: str) -> str:
    return name.strip().upper().replace(" ", "_").replace("-", "_")


def resolve_material(name: str):
    """Look a material name up in the SI-mm library.

    Returns (canonical_name, props_dict) on a hit, or (None, None) if the name is
    not known -- the caller then asks the human to define it rather than guessing.
    """
    if not name:
        return None, None
    key = _normalize(name)
    key = _ALIASES.get(key, key)
    props = MATERIALS.get(key)
    return (key, dict(props)) if props is not None else (None, None)


def material_names() -> list:
    """Sorted list of known material names (for prompts / error messages)."""
    return sorted(MATERIALS.keys())


def yield_strength(name: str):
    """(fy in MPa, its grade and source) for a library material, or None."""
    canonical, _props = resolve_material(name)
    return YIELD.get(canonical) if canonical else None
