#!/usr/bin/env python3
"""Diagnostics layer 8 (plan section 8): LLM root-cause advisor on failure.

Triggered when any pipeline stage fails. Gathers the failure context (the failed
stage's log + the relevant report + PARAMS metadata) and asks the LLM for a
Human Review Report: Failure Summary / Root Cause / Confidence / Suggested Fix /
Recommended Next Action. It is PURE ADVICE -- it never executes or modifies
anything (auto_fix_possible is an informational flag only). The deterministic
layers already decided that something failed; the LLM only explains why and
suggests a fix, then the human decides (Approve / Reject / Modify).

--dry-run gathers and prints the context WITHOUT calling the LLM (no API needed).
Unit system: SI-mm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

import llm_config
from paths import RunPaths, add_workdir_argument, from_args

STAGES = ("mesh", "bc", "inp", "solve", "physics", "postprocess", "verification")

# Reports to fold into the context, by stage (the stage log is always included).
# Values are RunPaths attributes, so the file is read from the failed run's own
# directory -- never from whichever run happened to write out/ last.
STAGE_REPORTS = {
    "solve": ["ccx_log"],
    "physics": ["physics_report"],
    "verification": ["verification_report"],
}

FAILURE_TAXONOMY = """Failure classes (use as guidance for failure_class):
  A_mesh         : meshing problems (negative Jacobian, distorted/inverted elements).
  B_solver       : ccx solve failed (singular stiffness from missing/insufficient
                   constraints, no convergence, divergence).
  C_physics      : physics validation FAIL (tip displacement direction wrong ->
                   load sign; force/moment imbalance; non-physical result).
  D_verification : displacement error exceeds tolerance (modeling/mesh issue, e.g.
                   L/h too small so Euler-Bernoulli beam theory does not apply).
"""

SYSTEM_PROMPT = (
    "You are an FEA diagnostics engineer for a linear-elastic cantilever-beam "
    "pipeline. A stage has FAILED; the failure was already determined by "
    "deterministic checks. Your job is ONLY to explain the most likely root cause "
    "and advise -- you never execute or modify anything. Call report_diagnosis "
    "with a concise, specific, actionable Human Review Report. Lower the "
    "confidence if the evidence is ambiguous. The context may include RELEVANT "
    "KNOWLEDGE retrieved by semantic search -- use it to ground your root cause "
    "and fix when it applies.\n\n" + FAILURE_TAXONOMY
)


class Diagnosis(BaseModel):
    failure_summary: str
    failure_class: Literal["A_mesh", "B_solver", "C_physics", "D_verification", "unknown"]
    root_cause: str
    confidence: Literal["high", "medium", "low"]
    suggested_fix: str
    recommended_next_action: str
    auto_fix_possible: bool  # informational flag only; nothing is auto-executed


def _tail(text: str, n: int = 60) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _params_summary(params=None) -> str:
    try:
        if params is None:
            from params import PARAMS
            params = PARAMS
        P = params
        gtype = P["geometry"]["type"]
        gsub = P["geometry"].get(gtype, {})
        mat = P["material"]
        return (f"geometry: {gtype} {gsub}\n"
                f"material: {mat.get('name')} (E={mat.get('E')} MPa, nu={mat.get('nu')}, rho={mat.get('rho')})\n"
                f"mesh: {P['mesh']}\n"
                f"bc.fixed: {P['bc']['fixed']}\n"
                f"bc.loads: {P['bc']['loads']}\n"
                f"analysis: {P['analysis']}")
    except Exception as exc:  # noqa: BLE001
        return f"(could not read params.py: {exc})"


def _retrieve_knowledge(query: str, k: int = 3) -> str:
    """RAG: retrieve relevant knowledge notes (failure patterns / guidance) to
    ground the diagnosis. Graceful -- diagnostics still works without retrieval."""
    try:
        from knowledge.vector_store import retrieve
        notes = retrieve(query[:1500], k=k, kind="note")
    except Exception as exc:  # noqa: BLE001
        return f"(knowledge retrieval unavailable: {exc.__class__.__name__})"
    return "\n".join(
        f"  - [{n.get('metadata', {}).get('type', 'note')}] "
        f"{n.get('metadata', {}).get('title', '')}: {n['text']}"
        for n in notes
    )


def gather_context(stage: str, log_path=None, params=None, paths: RunPaths = None) -> str:
    """Assemble the failure context, augmented with semantically-retrieved knowledge."""
    paths = paths or RunPaths.default()
    parts = [f"FAILED STAGE: {stage}", "", "PARAMS (metadata):", _params_summary(params), ""]

    log_text = ""
    if log_path and Path(log_path).exists():
        log_text = Path(log_path).read_text()
    elif stage == "solve" and paths.ccx_log.exists():
        log_text = paths.ccx_log.read_text()
    if log_text.strip():
        parts += ["STAGE LOG (tail):", _tail(log_text), ""]

    for attr in STAGE_REPORTS.get(stage, []):
        p = getattr(paths, attr)
        if p.exists() and (not log_path or p != Path(log_path)):
            parts += [f"--- {p.name} ---", p.read_text().strip(), ""]

    base = "\n".join(parts)
    knowledge = _retrieve_knowledge(base)
    if knowledge:
        base += "\n\nRELEVANT KNOWLEDGE (retrieved by semantic search over the KB):\n" + knowledge
    return base


def diagnose(context: str) -> Diagnosis:
    """One LLM call -> structured Diagnosis."""
    client = llm_config.make_client()
    tool = {
        "type": "function",
        "function": {
            "name": "report_diagnosis",
            "description": "Structured root-cause diagnosis and advice (no execution).",
            "parameters": Diagnosis.model_json_schema(),
        },
    }
    resp = client.chat.completions.create(
        model=llm_config.OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "report_diagnosis"}},
    )
    msg = resp.choices[0].message
    if not getattr(msg, "tool_calls", None):
        raise RuntimeError(f"model did not report a diagnosis: {msg.content!r}")
    return Diagnosis.model_validate(json.loads(msg.tool_calls[0].function.arguments))


def format_report(stage: str, d: Diagnosis) -> str:
    return "\n".join([
        "=" * 64,
        "HUMAN REVIEW REPORT (diagnostics, plan layer 8)",
        "=" * 64,
        f"failed stage  : {stage}",
        f"failure class : {d.failure_class}",
        "",
        f"1. Failure Summary:\n   {d.failure_summary}",
        f"2. Root Cause:\n   {d.root_cause}",
        f"3. Confidence: {d.confidence}",
        f"4. Suggested Fix:\n   {d.suggested_fix}",
        f"5. Recommended Next Action:\n   {d.recommended_next_action}",
        "",
        f"(auto_fix_possible = {d.auto_fix_possible} -- informational only; nothing runs automatically)",
        "-" * 64,
        "Decision required: Approve (apply fix & rerun) / Reject (stop) / Modify (change params & rerun).",
        "=" * 64,
        "",
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnostics agent (plan layer 8).")
    ap.add_argument("--stage", required=True, choices=STAGES, help="which stage failed")
    ap.add_argument("--log", default=None, help="path to the failed stage's log")
    ap.add_argument("--dry-run", action="store_true", help="print the gathered context only (no LLM call)")
    add_workdir_argument(ap)
    args = ap.parse_args()
    paths = from_args(args)

    context = gather_context(args.stage, args.log, paths=paths)
    if args.dry_run:
        print("=== gathered failure context (dry-run, no LLM) ===")
        print(context)
        return 0

    try:
        d = diagnose(context)
    except Exception as exc:  # noqa: BLE001
        print(f"diagnostics could not produce a report: {exc.__class__.__name__}: {exc}")
        return 1

    report = format_report(args.stage, d)
    paths.diagnostic_report.write_text(report)
    print(report, end="")
    print(f"(written to {paths.diagnostic_report})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
