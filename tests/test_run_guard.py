"""The limits on a shared dashboard: who the user is, the daily quota, the run
slots, the clean-up of old runs, and recognising an exhausted LLM budget."""

import os
import time

import pytest

import run_guard


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Every test starts with no limits set and its own usage file."""
    for name in ("FEA_MAX_CONCURRENT_RUNS", "FEA_DAILY_RUNS_PER_USER",
                 "FEA_USER_HEADER", "FEA_KEEP_RUNS_HOURS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(run_guard, "USAGE_FILE", tmp_path / "usage.json")
    monkeypatch.setattr(run_guard, "_slots", None)


def test_user_header_is_ignored_unless_configured(monkeypatch):
    # Without a trusted proxy, a browser could send any name: everyone is one user.
    assert run_guard.user_of({"X-FEA-User": "someone"}) == run_guard.ANONYMOUS
    monkeypatch.setenv("FEA_USER_HEADER", "X-FEA-User")
    assert run_guard.user_of({"X-FEA-User": "someone"}) == "someone"
    assert run_guard.user_of({}) == run_guard.ANONYMOUS


def test_daily_quota_is_per_user(monkeypatch):
    assert run_guard.runs_left("a") is None                  # no quota set
    monkeypatch.setenv("FEA_DAILY_RUNS_PER_USER", "2")
    assert run_guard.runs_left("a") == 2
    run_guard.count_run("a")
    run_guard.count_run("a")
    assert run_guard.runs_left("a") == 0
    assert run_guard.runs_left("b") == 2


def test_run_slots(monkeypatch):
    monkeypatch.setenv("FEA_MAX_CONCURRENT_RUNS", "1")
    with run_guard.run_slot() as first:
        assert first
        with run_guard.run_slot(timeout=0.1) as second:
            assert not second                                # the one slot is taken
    with run_guard.run_slot(timeout=0.1) as again:
        assert again                                         # released after the run


def test_prune_removes_only_old_run_directories(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    old_run, new_run, other = (runs / "20260101-120000-abcdef", runs / "20260102-120000-123456",
                               runs / "keep-me")
    for d in (old_run, new_run, other):
        d.mkdir(parents=True)
    three_days_ago = time.time() - 72 * 3600
    for d in (old_run, other):
        os.utime(d, (three_days_ago, three_days_ago))

    assert run_guard.prune_old_runs(runs) == 0               # no retention set: keep all
    monkeypatch.setenv("FEA_KEEP_RUNS_HOURS", "48")
    assert run_guard.prune_old_runs(runs) == 1
    assert not old_run.exists()
    assert new_run.exists() and other.exists()               # too new / not a run directory


def test_budget_exhausted():
    class Refused(Exception):
        status_code = 402

    assert run_guard.budget_exhausted(Refused())
    assert run_guard.budget_exhausted("openai.APIStatusError: Error code: 402 - ...")
    assert not run_guard.budget_exhausted(ValueError("other"))
    assert not run_guard.budget_exhausted("Error code: 500")
