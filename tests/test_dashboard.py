"""The dashboard, driven headless with Streamlit's AppTest: every stage of every
recorded example, and the list of earlier live runs. A page that raises shows
the visitor a traceback, so "no exception" is the bar each test holds it to.
No language model is called and nothing is solved."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import paths

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ["cantilever_pass", "plate_pass", "stubby_fail"]


def app(monkeypatch, demo_mode: bool) -> AppTest:
    if demo_mode:
        monkeypatch.setenv("DEMO_MODE", "1")
    else:
        monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("FEA_USER_HEADER", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def click(at: AppTest, key: str) -> AppTest:
    at.button(key=key).click().run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("name", DEMOS)
def test_replay_every_stage(monkeypatch, name):
    at = app(monkeypatch, demo_mode=True)
    at.selectbox[0].select(name).run()
    for _ in range(10):                          # step through the recorded conversation
        step = [b for b in at.button if "next" in b.label]
        if not step:
            break
        step[0].click().run()
        assert not at.exception, at.exception
    stage_keys = [b.key for b in at.button if (b.key or "").startswith("stagebtn")]
    assert len(stage_keys) == 7
    for key in stage_keys:
        click(at, key)


@pytest.fixture
def runs_root(tmp_path, monkeypatch):
    """An empty out/runs/ for the live-mode tests (the dashboard reads it lazily)."""
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(paths, "RUNS_ROOT", root)
    return root


def add_run(root: Path, name: str, demo: str = "cantilever_pass"):
    """A stored live run: a recorded example's record, filed as a live run would be."""
    rec = json.loads((ROOT / "demo" / demo / "run.json").read_text())
    rec["user"] = "local"
    (root / name).mkdir()
    (root / name / "live_run.json").write_text(json.dumps(rec))


def test_live_mode_without_runs(monkeypatch, runs_root):
    at = app(monkeypatch, demo_mode=False)
    assert not [s for s in at.selectbox if s.key == "past_run"]   # nothing to list yet


def test_reopen_an_earlier_run(monkeypatch, runs_root):
    add_run(runs_root, "20261011-120000-aaaaaa")
    at = app(monkeypatch, demo_mode=False)
    menu = at.selectbox(key="past_run")
    assert menu.value is None
    assert menu.placeholder == "earlier runs (1) …"

    # A new run appears while the page is open: the list grows, nothing breaks.
    add_run(runs_root, "20261011-130000-bbbbbb", demo="plate_pass")
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="past_run").placeholder == "earlier runs (2) …"

    at.selectbox(key="past_run").select("20261011-120000-aaaaaa").run()
    assert not at.exception, at.exception
    assert at.session_state.phase == "done"
    click(at, "stagebtn6")                       # its Verify stage, with the report button


def test_a_vague_request_shows_the_load_cases(monkeypatch, runs_root):
    # Layer 1's model call is replaced by its answer to a vague request.
    import task_understanding as tu
    monkeypatch.setattr(tu, "assess", lambda conversation: tu.LLMReport(is_fea_task=True))
    at = app(monkeypatch, demo_mode=False)
    at.text_input[0].input("Can you help me with an FEA?")
    at.button[0].click().run()
    assert not at.exception, at.exception
    page = "".join(m.value for m in at.markdown)
    assert "Which problem is it?" in page
    assert 'class="sketch"' in page
    assert page.count("data:image/svg+xml;base64,") == 2      # the beam and the plate


def test_the_section_question_shows_the_beam_alone(monkeypatch, runs_root):
    import task_understanding as tu
    from task_understanding import LLMReport, Mention
    said = "a steel cantilever, 500 mm long, 20x10 as cross section, fixed at one end, 300 N down at the free end"
    report = LLMReport(is_fea_task=True, geometry_type="cantilever_beam", fixed_specified=True,
                       load_specified=True, mentions=[
                           Mention(quantity="L", turn=1, quote="500 mm long", value=500.0),
                           Mention(quantity="h", turn=1, quote="20x10", value=20.0),
                           Mention(quantity="b", turn=1, quote="20x10", value=10.0),
                           Mention(quantity="load", turn=1, quote="300 N", value=300.0),
                           Mention(quantity="material", turn=1, quote="steel", material="STEEL")])
    monkeypatch.setattr(tu, "assess", lambda conversation: report)
    at = app(monkeypatch, demo_mode=False)
    at.text_input[0].input(said)
    at.button[0].click().run()
    assert not at.exception, at.exception
    page = "".join(m.value for m in at.markdown)
    assert "Which side does the load act along: 20 mm or 10 mm?" in page
    assert 'class="sketch one"' in page and page.count("data:image/svg+xml;base64,") == 1
