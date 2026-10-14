#!/usr/bin/env python3
"""Record the layer-1 clarification dialogue for each demo into demo/<name>/chat.json,
so DEMO_MODE can replay the "vague request -> agent clarifies -> ready" exchange before
the dashboard (no key needed at replay time -- this records it once, with the real LLM).

Calls the same task_understanding.understand() as app.py::_branch, so the replayed
bubbles match live behaviour. Needs the fea env + the OpenRouter key in .env.

Usage:  python record_chat.py
"""
from __future__ import annotations

import json
from pathlib import Path

import task_understanding as tu

HERE = Path(__file__).resolve().parent
DEMO = HERE / "demo"

# A scripted (vague -> details) pair per demo. The vague turn names steel + cantilever
# but omits geometry and load, so must_have_gaps fires and the agent asks; the details
# turn completes it. The resolved problem matches each demo's recorded run.
SCRIPTS = {
    "cantilever_pass": (
        "I want to analyze a steel cantilever beam.",
        "It's 100 mm long, 10x10 mm square cross-section, fixed at one end, "
        "with a 1000 N downward load at the free end."),
    "stubby_fail": (
        "Can you check a short steel cantilever for me?",
        "20 mm long, 10x10 mm square section, fixed at one end, "
        "1000 N pushing straight down at the free tip."),
    # The plate turn is written to exercise the READING, not just the solving.
    # In one ordinary sentence it asks layer 1 to canonicalise a material
    # spelling, halve a DIAMETER into the radius the schema wants, convert kN
    # to N, and recognise "right in the middle" as a position it should leave
    # unset so the centred default applies. It never mentions thickness or how
    # the plate is held -- both come back as disclosed assumptions rather than
    # as questions, because neither is something this geometry lets a user choose.
    "plate_pass": (
        "I'd like to check an aluminium plate that has a hole in it.",
        "It's 240 by 120 mm with a 36 mm diameter hole right in the middle, "
        "and it's pulled with 4.5 kN."),
}


def record(name: str, vague: str, details: str) -> None:
    """The vague turn, the agent's reply, and -- unless that already sufficed --
    the details turn and the agent's reply to it: what the dashboard would show."""
    (DEMO / name).mkdir(parents=True, exist_ok=True)   # do not depend on record_demo running first
    chat: list = [("user", vague)]
    step = tu.understand(chat)
    chat.append(("agent", step.message, step.meta()))
    if step.kind != "ready":
        chat.append(("user", details))
        step = tu.understand(chat)
        chat.append(("agent", step.message, step.meta()))
    (DEMO / name / "chat.json").write_text(
        json.dumps([{"role": e[0], "text": e[1]} for e in chat], indent=2, ensure_ascii=False))
    print(f"recorded {name:<16} {step.kind}  turns={len(chat)}")
    for e in chat:
        print(f"    {e[0]:<5}: {e[1][:90]}{'…' if len(e[1]) > 90 else ''}")


def main() -> int:
    for name, (vague, details) in SCRIPTS.items():
        record(name, vague, details)
    print(f"\nchat threads in {DEMO}/<name>/chat.json — DEMO_MODE replays them before the dashboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
