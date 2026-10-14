"""Which side of a cantilever's section the load acts along. That side is the
height h, the pipeline loads along it (-z), and the reference and the yield
check assume it. When the user does not say, it is asked; a sideways load turns
the beam; a load along the beam or at an angle is out of scope.

Layer 1's model call is given as prepared reports, so no language model is
called. The last tests solve, to show the backstop on a load that is not along h.
"""

import copy

import pytest

import pipeline_run
import task_understanding as tu
from params import CASES
from paths import RunPaths
from task_understanding import LLMReport, Mention, TaskAssessment

OPENING = "a steel cantilever, 500 mm long, 20x10 as cross section, fixed at one end, 300 N down at the free end"


def mention(quantity, turn, quote, value=None, material=None):
    return Mention(quantity=quantity, turn=turn, quote=quote, value=value, material=material)


def beam_report(*mentions, **fields):
    """A cantilever with its supports, load and material stated; the sizes as given."""
    base = [mention("load", 1, "300 N", 300.0), mention("material", 1, "steel", material="STEEL")]
    return LLMReport(is_fea_task=True, geometry_type="cantilever_beam", fixed_specified=True,
                     fixed_end="one end", load_specified=True, load_end="the free end",
                     mentions=[*base, *mentions], **fields)


def opening_mentions():
    return [mention("L", 1, "500 mm long", 500.0), mention("h", 1, "20x10", 20.0),
            mention("b", 1, "20x10", 10.0)]


def first_step():
    conversation = [("user", OPENING)]
    return conversation, tu.understand(conversation, report=beam_report(*opening_mentions()))


# ---- a section given as numbers alone ------------------------------------------

def test_a_section_without_names_is_asked_about():
    _conversation, step = first_step()
    assert step.kind == "ask"
    assert step.section == (20.0, 10.0) and set(step.asks) == {"h", "b"}
    assert step.figure == "cantilever"
    assert "Which side does the load act along: 20 mm or 10 mm?" in step.message


@pytest.mark.parametrize("reply, named, h, b", [
    ("the 10 mm side", mention("h", 2, "10 mm", 10.0), 10.0, 20.0),
    ("the 20 mm side", mention("h", 2, "20 mm", 20.0), 20.0, 10.0),
    ("20 is the width", mention("b", 2, "20 is the width", 20.0), 10.0, 20.0),
])
def test_the_reply_settles_both_sides(reply, named, h, b):
    conversation, step = first_step()
    conversation += [("agent", step.message, step.meta()), ("user", reply)]
    # The model still reads b off "20x10" as it did before; the reply decides.
    step = tu.understand(conversation, report=beam_report(*opening_mentions(), named))
    assert step.kind == "ready"
    assert (step.resolved["h"], step.resolved["b"]) == (h, b)


@pytest.mark.parametrize("reply, h, b", [("the 10 mm side", 10.0, 20.0), ("20", 20.0, 10.0)])
def test_a_reply_the_model_missed_is_read_by_code(reply, h, b):
    # The model reported nothing for the reply (seen once in eight runs on luna).
    # A reply holding just one of the two numbers asked about names that side.
    conversation, step = first_step()
    conversation += [("agent", step.message, step.meta()), ("user", reply)]
    step = tu.understand(conversation, report=beam_report(*opening_mentions()))
    assert step.kind == "ready" and (step.resolved["h"], step.resolved["b"]) == (h, b)


def test_an_unclear_reply_is_asked_again_in_other_words():
    conversation, step = first_step()
    conversation += [("agent", step.message, step.meta()), ("user", "20 is the width")]
    # About the width, and the model missed it: left to the user, not guessed.
    step = tu.understand(conversation, report=beam_report(*opening_mentions()))
    assert step.kind == "ask" and step.section == (20.0, 10.0)
    assert step.message.startswith("I could not tell")


def test_named_sides_are_not_asked_about():
    conversation = [("user", "a steel cantilever, 500 mm long, 20 mm wide and 10 mm high, 300 N down at the free end")]
    step = tu.understand(conversation, report=beam_report(
        mention("L", 1, "500 mm long", 500.0), mention("b", 1, "20 mm wide", 20.0),
        mention("h", 1, "10 mm high", 10.0)))
    assert step.kind == "ready" and (step.resolved["h"], step.resolved["b"]) == (10.0, 20.0)


def test_a_reply_to_our_ordered_question_is_not_asked_about():
    question = tu.consolidated_question(
        ["geometry"], TaskAssessment(is_fea_task=True, geometry_type="cantilever_beam"))
    assert tu.SECTION_ORDER in question
    conversation = [("user", "a steel cantilever fixed at one end, 300 N down at the free end"),
                    ("agent", question, {"asks": []}), ("user", "500 x 20 x 10")]
    step = tu.understand(conversation, report=beam_report(
        mention("L", 2, "500 x 20 x 10", 500.0), mention("h", 2, "500 x 20 x 10", 20.0),
        mention("b", 2, "500 x 20 x 10", 10.0)))
    assert step.kind == "ready" and (step.resolved["h"], step.resolved["b"]) == (20.0, 10.0)


def test_a_square_section_is_not_asked_about():
    conversation = [("user", "a steel cantilever, 100 mm long, 10x10, 300 N down at the free end")]
    step = tu.understand(conversation, report=beam_report(
        mention("L", 1, "100 mm long", 100.0), mention("h", 1, "10x10", 10.0),
        mention("b", 1, "10x10", 10.0)))
    assert step.kind == "ready"


# ---- which way the load acts ----------------------------------------------------

def test_a_sideways_load_turns_the_beam_and_says_so():
    conversation = [("user", "a steel cantilever 200 mm long, 20 mm high, 10 mm wide, 300 N sideways at the free end")]
    step = tu.understand(conversation, report=beam_report(
        mention("L", 1, "200 mm long", 200.0), mention("h", 1, "20 mm high", 20.0),
        mention("b", 1, "10 mm wide", 10.0), load_axis="width", load_direction="sideways"))
    assert step.kind == "ready"
    assert (step.resolved["h"], step.resolved["b"]) == (10.0, 20.0)
    assert step.resolved["load_direction"] == "down (-z)"
    assert "turned 90 degrees" in step.assumptions[0]


def test_sideways_after_the_section_question_keeps_the_named_side():
    conversation, step = first_step()
    conversation += [("agent", step.message, step.meta()), ("user", "the 10 mm side, it is pushed sideways")]
    step = tu.understand(conversation, report=beam_report(
        *opening_mentions(), mention("h", 2, "10 mm", 10.0), load_axis="width"))
    assert step.kind == "ready"
    assert (step.resolved["h"], step.resolved["b"]) == (10.0, 20.0)    # not turned a second time
    assert step.resolved["load_direction"] == "down (-z)"


@pytest.mark.parametrize("axis", ["axial", "oblique"])
def test_a_load_along_the_beam_or_at_an_angle_is_out_of_scope(axis):
    step = tu.understand([("user", OPENING)], report=beam_report(*opening_mentions(), load_axis=axis))
    assert step.kind == "out_of_scope" and step.figure == "loadcases"


# ---- the backstop: never a reference or a yield check for a load off h -------------

def beam(h, b, direction):
    params = copy.deepcopy(CASES["cantilever_beam"])
    params["geometry"]["cantilever_beam"].update(L=200.0, h=h, b=b)
    params["bc"]["loads"][0]["dir"] = direction
    return params


def run(params, tmp_path):
    rec = pipeline_run.run_pipeline("test", params=params, paths=RunPaths(tmp_path), diagnose=False)
    verify = next(s for s in rec["stages"] if s["id"] == "verify")
    checks = next(a["items"] for a in verify["artifacts"] if a.get("kind") == "checks")
    return rec, verify["data"], {c["name"]: c["status"] for c in checks}


def test_a_sideways_load_gets_no_reference_and_no_yield_check(tmp_path_factory):
    # 20 high, 10 wide, pushed along y: it bends about the weak axis, which the
    # reference (I = b h^3 / 12) and M c / I do not describe.
    sideways, data, checks = run(beam(20.0, 10.0, [0, 1, 0]), tmp_path_factory.mktemp("sideways"))
    assert sideways["verdict"] == "N/A" and data["kb_ref"] is None
    assert checks["stress below yield"] == "N/A"           # NOT CHECKED, shown as N/A
    # The same beam turned 90 degrees, as layer 1 sets it up: the reference applies,
    # agrees, and the FE deflection is the sideways one.
    turned, data_t, _ = run(beam(10.0, 20.0, [0, 0, -1]), tmp_path_factory.mktemp("turned"))
    assert turned["verdict"] == "PASS"
    assert data_t["fem"] == pytest.approx(data["delta_fem"], rel=1e-3)
