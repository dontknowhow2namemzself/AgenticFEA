"""When a reply carries the load-case sketch. Code decides it from layer 1's
step, never the model; the model's part (reading the request) is given here as
prepared reports, so no language model is called."""

import re
from pathlib import Path

import pytest

import task_understanding as tu
from task_understanding import LLMReport

ROOT = Path(__file__).resolve().parents[1]


def step(text, **report):
    return tu.understand([("user", text)], report=LLMReport(**report))


def test_a_vague_request_gets_the_load_cases():
    s = step("Can you help me with an FEA?", is_fea_task=True)
    assert (s.kind, s.gaps, s.figure) == ("ask", ["geometry_type"], "loadcases")


def test_out_of_scope_shows_what_is_supported():
    s = step("A simply supported beam with a distributed load", is_fea_task=True,
             in_scope=False, scope_note="distributed loads are not supported")
    assert (s.kind, s.figure) == ("out_of_scope", "loadcases")


def test_no_sketch_off_topic_or_once_the_geometry_is_known():
    assert step("What's the weather tomorrow?", is_fea_task=False,
                off_topic_reason="weather").figure == ""
    s = step("a steel cantilever", is_fea_task=True, geometry_type="cantilever_beam")
    assert s.kind == "ask" and s.figure == ""          # asks for its dimensions, in words


@pytest.mark.parametrize("name", ["loadcase_cantilever.svg", "loadcase_plate.svg"])
def test_the_sketches_are_plain_drawings(name):
    svg = (ROOT / "docs" / name).read_text()
    assert svg.lstrip().startswith("<?xml") and "<svg" in svg
    # Nothing that could run or reach out, even though it is shown as an <img>.
    assert not re.search(r"<script|<foreignObject|\son\w+\s*=|javascript:|https?://(?!www\.w3\.org)",
                         svg, re.IGNORECASE)
