#!/usr/bin/env python3
"""Parameterization layer 2 -- LLM part: the set_params tool (plan section 2).

Turns a natural-language problem description into a validated PARAMS dict by
having the LLM (OpenRouter) CALL a `set_params` tool whose JSON-schema IS the
Pydantic schema from schema.py. Flow:

    NL description -> LLM fills set_params slots (material = NAME only)
      -> deterministic material lookup (knowledge/materials.py, SI-mm)
           * known   -> inject E / nu / rho
           * unknown -> HARD STOP, ask the human to define it (no LLM retry)
      -> schema.py validates everything
           * valid   -> return / write params.py
           * invalid -> feed the error back, let the LLM correct, retry

Iron law: the LLM only does semantic slot-filling (geometry, BC, and the
material NAME); deterministic code supplies every number (material properties
from the library) and decides validity (schema.py). The key is read from the
environment only (llm_config). This is the layer-2 driver; task understanding
(layer 1) will later feed it a cleaned definition.
"""

from __future__ import annotations

import argparse
import json
import pprint
import sys
from pathlib import Path

from pydantic import ValidationError

import llm_config
from knowledge.materials import material_names, resolve_material
from schema import Params, validate_params

HERE = Path(__file__).resolve().parent
PARAMS_PY = HERE / "params.py"

SYSTEM_PROMPT = """You are the parameterization assistant for a linear-elastic
cantilever-beam FEA pipeline. Convert the user's description into structured
parameters by calling the set_params tool exactly once. Rules:

- Units are SI-mm: length mm, force N. Convert any other units the user gives.
- Axis convention: beam axis is x (length L); cross-section is width b (along y)
  and height h (along z, the usual load direction). The only valid face labels
  are face_x_min, face_x_max, face_y_min, face_y_max, face_z_min, face_z_max.
- Map the fixed/clamped end to bc.fixed and the loaded end to bc.loads. By
  convention fix face_x_min and load the opposite free end face_x_max unless the
  description clearly says otherwise. A downward load has dir [0, 0, -1] and a
  positive magnitude (the sign lives in dir, magnitude is > 0).
- For material, give ONLY material.name (e.g. STEEL, ALUMINUM, COPPER, TITANIUM).
  Do NOT provide E, nu, or rho -- the system looks those up from a curated SI-mm
  materials library. If the description does not name a material, use STEEL.
- For unspecified values use sensible defaults: elem_size about 1/5 of the
  smallest cross-section dimension, elem_order 2, elem_shape "hex", analysis
  static and linear (nonlinear false), acceptance.tol_disp 0.05.
- Fill every required slot. Do NOT perform engineering analysis or predict
  results; only translate the description into parameter values.
"""

DEFAULT_DESCRIPTION = (
    "A steel cantilever beam 100 mm long with a 10 mm by 10 mm square cross-section, "
    "fully fixed at one end, carrying a 1000 N downward load at the free end. "
    "Use a 2 mm mesh."
)


class UnknownMaterialError(Exception):
    """Raised when the LLM names a material that is not in the SI-mm library."""

    def __init__(self, name):
        self.name = name
        super().__init__(f"material {name!r} is not in the SI-mm materials library")


def set_params_tool_def() -> dict:
    """OpenAI/OpenRouter tool whose parameters ARE the Pydantic schema, except the
    material slot is reduced to NAME only (numbers come from the library)."""
    schema = Params.model_json_schema()
    if "MaterialParams" in schema.get("$defs", {}):
        schema["$defs"]["MaterialParams"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {
                    "type": "string",
                    "description": ("Canonical material name to look up in the SI-mm "
                                    "library, e.g. STEEL, ALUMINUM, COPPER, TITANIUM. "
                                    "Do NOT provide E, nu, or rho."),
                }
            },
            "required": ["name"],
        }
    return {
        "type": "function",
        "function": {
            "name": "set_params",
            "description": "Record the validated FEA problem parameters (SI-mm units).",
            "parameters": schema,
        },
    }


def _assistant_toolcall_msg(call) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call.id,
            "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }],
    }


def _feed_back(messages: list, call, error: str) -> None:
    """Append the model's tool call + a tool result carrying the error, so the
    next attempt can correct it."""
    messages.append(_assistant_toolcall_msg(call))
    messages.append({
        "role": "tool", "tool_call_id": call.id,
        "content": f"Rejected by the deterministic schema. Fix and call set_params again:\n{error}",
    })


def run_set_params(description: str, max_retries: int = 3, verbose: bool = True) -> dict:
    """Drive the LLM to fill set_params, resolve the material, and validate.
    Returns {params, warnings, attempts}. Raises UnknownMaterialError (halt) or
    RuntimeError (gave up)."""
    client = llm_config.make_client()
    tool = set_params_tool_def()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": description},
    ]

    last_error = "no attempt made"
    for attempt in range(1, max_retries + 1):
        resp = client.chat.completions.create(
            model=llm_config.OPENROUTER_MODEL,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "set_params"}},
        )
        msg = resp.choices[0].message
        if not getattr(msg, "tool_calls", None):
            raise RuntimeError(f"model did not call set_params (attempt {attempt}): {msg.content!r}")
        call = msg.tool_calls[0]
        raw = call.function.arguments
        if verbose:
            print(f"--- attempt {attempt}: set_params arguments ---")
            print(raw)

        # 1) JSON parse (retryable).
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as exc:
            last_error = f"arguments are not valid JSON: {exc}"
            if verbose:
                print(f"--- attempt {attempt} rejected: {last_error} ---")
            _feed_back(messages, call, last_error)
            continue

        # 2) Material name -> SI-mm library (unknown = HARD STOP, not a retry).
        mat_name = (args.get("material") or {}).get("name")
        canonical, props = resolve_material(mat_name)
        if props is None:
            raise UnknownMaterialError(mat_name)
        args["material"] = {"name": canonical, **props}
        if verbose:
            print(f"--- material '{mat_name}' -> {canonical}: {props} (SI-mm library) ---")

        # 3) Deterministic schema validation (retryable).
        try:
            model, warnings = validate_params(args)
            return {"params": model.model_dump(), "warnings": warnings, "attempts": attempt}
        except ValidationError as exc:
            last_error = f"schema validation failed:\n{exc}"
            if verbose:
                print(f"--- attempt {attempt} rejected: {last_error.splitlines()[0]} ---")
            _feed_back(messages, call, last_error)

    raise RuntimeError(f"set_params failed after {max_retries} attempts; last error:\n{last_error}")


def write_params_py(params: dict, path=PARAMS_PY) -> None:
    """Write params.py from a validated dict (the single source of truth)."""
    header = ('"""Auto-generated by parameterize.py (set_params, layer 2) from a natural-\n'
              'language description and validated by schema.py before writing. Material\n'
              'properties come from the SI-mm materials library. Single source of truth.\n'
              'Unit system: SI-mm."""\n\n')
    body = "PARAMS = " + pprint.pformat(params, sort_dicts=False, width=88) + "\n"
    Path(path).write_text(header + body)


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-driven parameterization (plan layer 2).")
    ap.add_argument("description", nargs="?", default=DEFAULT_DESCRIPTION,
                    help="natural-language problem description")
    ap.add_argument("--write", action="store_true",
                    help="overwrite params.py with the result (default: preview only)")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    print("=" * 64)
    print("parameterize.py -- set_params (plan layer 2, LLM)")
    print("=" * 64)
    print(f"model       : {llm_config.OPENROUTER_MODEL}")
    print(f"description : {args.description}")
    print("-" * 64)
    try:
        result = run_set_params(args.description, max_retries=args.retries)
    except UnknownMaterialError as exc:
        print("-" * 64)
        print(f"HALT: material {exc.name!r} is not in the SI-mm materials library.")
        print(f"Known materials: {', '.join(material_names())}")
        print("Add it (SI-mm: E in MPa, nu, rho in tonne/mm^3) to knowledge/materials.py, then re-run.")
        print("=" * 64)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}")
        return 1

    print("-" * 64)
    print(f"validated on attempt {result['attempts']}")
    print("=== validated PARAMS ===")
    print(pprint.pformat(result["params"], sort_dicts=False, width=88))
    for w in result["warnings"]:
        print(f"[WARN] {w}")
    if not result["warnings"]:
        print("(no warnings)")
    print("-" * 64)
    if args.write:
        write_params_py(result["params"])
        print(f"ACCEPTANCE: PASS -> wrote {PARAMS_PY}")
    else:
        print("ACCEPTANCE: PASS (preview only; pass --write to overwrite params.py)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
