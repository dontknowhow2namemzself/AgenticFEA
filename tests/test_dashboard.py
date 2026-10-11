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
