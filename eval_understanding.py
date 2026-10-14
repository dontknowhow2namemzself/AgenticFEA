#!/usr/bin/env python3
"""Layer-1 evaluation: scripted conversations with known right answers, run
against a real model -- and repeated, because the model samples.

The language layer's behaviour is measured, not assumed. Before changing
OPENROUTER_MODEL, run this on the candidates and take the cheapest model that
passes every case on every repeat. Each case drives task_understanding.
understand() exactly as the dashboard does: a user turn, the agent's reply, the
next scripted user turn, and so on, with a check after every turn.

Usage:  python eval_understanding.py [--model openai/...] [--repeat 3] [--case word]
Cost:   one LLM call per user turn (24 turns per repeat).
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def ask_gaps(*gaps):
    """A question about missing information (and no contradiction) covering `gaps`."""
    def check(s):
        ok = s.kind == "ask" and not s.conflicts and set(gaps) <= set(s.gaps)
        return ok, f"{s.kind} gaps={s.gaps} conflicts={list(s.conflicts)}"
    return check


def ask_conflict(quantity, *values):
    """A question that names the contradiction on `quantity` with exactly these values."""
    def check(s):
        got = {v for v, _quote in s.conflicts.get(quantity, [])}
        ok = s.kind == "ask" and got == set(values)
        return ok, f"{s.kind} conflicts={ {q: [v for v, _ in o] for q, o in s.conflicts.items()} }"
    return check


def ask_geometry_type():
    def check(s):
        return s.kind == "ask" and s.gaps == ["geometry_type"], f"{s.kind} gaps={s.gaps}"
    return check


def kind(expected):
    def check(s):
        return s.kind == expected, f"{s.kind}: {s.reason[:70]}"
    return check


def ready(**expected):
    """Ready to run, with these resolved values (floats compared to 1e-6)."""
    def check(s):
        if s.kind != "ready":
            return False, f"{s.kind}: {s.reason[:70]}"
        r = s.resolved
        bad = {k: r.get(k) for k, v in expected.items()
               if not (abs(float(r.get(k) or 0) - v) < 1e-6 if isinstance(v, (int, float)) else r.get(k) == v)}
        return not bad, "ready" + (f" but {bad}" if bad else "")
    return check


BEAM = "A steel cantilever beam 100 mm long, 10x10 mm square section, fixed at one end, with a 1000 N downward load at the free end."

# (name, [(user turn, check after it), ...])
CASES = [
    ("length given twice (the reported conversation)", [
        ("cantilever_beam 1000mm long", ask_gaps("geometry")),
        ("200 x 20 x10, Steel, left side fixed, load on the other side 1500N", ask_conflict("L", 1000, 200)),
        ("200 mm", ready(L=200, h=20, b=10, material_name="STEEL", load_magnitude=1500)),
    ]),
    ("vague first message, then the details", [
        ("a beam calculation", ask_geometry_type()),
        ("a steel cantilever, 100 mm long, 10 x 10 mm section, fixed at one end, 1000 N down at the free end",
         ready(L=100, h=10, b=10, material_name="STEEL", load_magnitude=1000)),
    ]),
    ("very vague request", [
        ("can you help me with an FEA?", ask_geometry_type()),
    ]),
    ("capability question", [
        ("What can you do?", ask_geometry_type()),
    ]),
    ("capability question, other words", [
        ("What kind of problems can you calculate?", ask_geometry_type()),
    ]),
    ("complete cantilever in one message", [
        (BEAM, ready(L=100, h=10, b=10, material_name="STEEL", load_magnitude=1000)),
    ]),
    ("units: m, cm, kN", [
        ("a steel cantilever 0.2 m long, 20 mm high and 1 cm wide, clamped at one end, 1.5 kN down at the tip",
         ready(L=200, h=20, b=10, load_magnitude=1500)),
    ]),
    ("plate: a diameter and kN (the plate demo)", [
        ("I'd like to check an aluminium plate that has a hole in it.", ask_gaps("geometry", "load")),
        ("It's 240 by 120 mm with a 36 mm diameter hole right in the middle, and it's pulled with 4.5 kN.",
         ready(a=240, b=120, r=18, material_name="ALUMINUM", load_magnitude=4500)),
    ]),
    ("two materials in one message", [
        (BEAM + " The beam is made of aluminium.", ask_conflict("material", "STEEL", "ALUMINUM")),
    ]),
    ("load given twice", [
        ("A steel cantilever 100 mm long, 10x10 mm, fixed at one end",
         ask_gaps("load")),
        ("1000 N down at the free end. Sorry, I mean 2 kN.", ask_conflict("load", 1000, 2000)),
        ("2 kN", ready(load_magnitude=2000)),
    ]),
    ("the same value twice is not a contradiction", [
        ("A 100 mm long steel cantilever, 10x10 mm section, fixed at one end. "
         "Load: 1000 N down at the free end of the 100 mm beam.", ready(L=100, load_magnitude=1000)),
    ]),
    ("the assistant's example is not the user's value", [
        ("a steel cantilever fixed at one end with 1000 N at the free end", ask_gaps("geometry")),
        ("it is 50 mm long, 5 x 5 mm", ready(L=50, h=5, b=5)),
    ]),
    ("out of scope", [
        ("A simply supported steel beam, 2 m span, with a uniformly distributed load of 5 kN/m",
         kind("out_of_scope")),
    ]),
    ("off topic", [
        ("what's the weather in Munich tomorrow?", kind("off_topic")),
    ]),
    ("impossible plate, then corrected", [
        ("a steel plate 100 by 100 mm with a 50 mm radius hole in the middle, pulled with 1 kN",
         kind("impossible")),
        ("make the radius 20 mm", ready(a=100, b=100, r=20)),
    ]),
    ("material not in the library", [
        ("an Inconel cantilever 100 mm long, 10x10 mm, fixed at one end, 1000 N down at the free end",
         ask_gaps("material")),
    ]),
]


def run_case(name, turns, tu) -> tuple:
    conversation, notes = [], []
    for k, (text, check) in enumerate(turns, 1):
        conversation.append(("user", text))
        step = tu.understand(conversation)
        ok, detail = check(step)
        notes.append(f"turn {k}: {'ok ' if ok else 'BAD'} {detail}")
        if not ok:
            return False, notes
        conversation.append(("agent", step.message, {"asks": step.asks}))
    return True, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="OpenRouter model id (default: OPENROUTER_MODEL / llm_config)")
    ap.add_argument("--repeat", type=int, default=1, help="runs per case; a case passes only if every run does")
    ap.add_argument("--case", help="run only cases whose name contains this")
    args = ap.parse_args()
    if args.model:
        os.environ["OPENROUTER_MODEL"] = args.model
    import llm_config
    import task_understanding as tu
    llm_config.OPENROUTER_MODEL = args.model or llm_config.OPENROUTER_MODEL

    cases = [c for c in CASES if not args.case or args.case in c[0]]
    print(f"model {llm_config.OPENROUTER_MODEL} · {len(cases)} cases × {args.repeat}")
    passed, t0 = 0, time.time()
    for name, turns in cases:
        runs = [run_case(name, turns, tu) for _ in range(args.repeat)]
        good = sum(ok for ok, _ in runs)
        passed += good == args.repeat
        print(f"{'PASS' if good == args.repeat else 'FAIL'} {good}/{args.repeat}  {name}")
        for ok, notes in runs:
            if not ok:
                print("        " + "\n        ".join(notes))
                break
    print(f"\n{passed}/{len(cases)} cases passed on every run · {time.time() - t0:.0f} s")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
