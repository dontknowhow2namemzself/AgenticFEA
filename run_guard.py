"""Limits on live runs when several people share one dashboard.

A live run spends LLM calls and a solver slot, so on a shared dashboard these
keep one visitor -- or a burst of them -- from using everything up. Each is an
environment setting, and all are off by default, so a run on your own machine
is unaffected:

  FEA_MAX_CONCURRENT_RUNS  live runs at once; a further one waits for a free slot
  FEA_DAILY_RUNS_PER_USER  live runs per user per day (UTC)
  FEA_USER_HEADER          request header that carries the signed-in user's name
  FEA_KEEP_RUNS_HOURS      run directories older than this are deleted

FEA_USER_HEADER may only be set when an authenticating proxy in front of the
dashboard sets that header on EVERY request, overwriting whatever the browser
sent; otherwise anyone could claim any name. Without it, everyone is one user.

The dashboard is a single process serving every session from threads, so a
process-wide semaphore and lock are enough; nothing here works across
processes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from paths import DEFAULT_ROOT, RUNS_ROOT

USAGE_FILE = DEFAULT_ROOT / "usage.json"
ANONYMOUS = "local"
_RUN_DIR = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")    # RunPaths.new_run() names only

_lock = threading.Lock()
_slots = None


def _int_env(name: str):
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


def user_of(headers) -> str:
    """The signed-in user's name from the trusted header, or one shared name."""
    name = os.environ.get("FEA_USER_HEADER", "").strip()
    return ((headers or {}).get(name) or ANONYMOUS) if name else ANONYMOUS


# ---- concurrency ---------------------------------------------------------
@contextmanager
def run_slot(timeout: float = 120.0):
    """Hold one of FEA_MAX_CONCURRENT_RUNS slots for the duration of a run.

    Yields True once a slot is held (or there is no limit), False if none
    freed up within `timeout` seconds. A run takes well under a minute, so a
    short wait is better than turning the visitor away at once.
    """
    global _slots
    limit = _int_env("FEA_MAX_CONCURRENT_RUNS")
    if not limit:
        yield True
        return
    with _lock:
        if _slots is None:
            _slots = threading.BoundedSemaphore(limit)
    got = _slots.acquire(timeout=timeout)
    try:
        yield got
    finally:
        if got:
            _slots.release()


# ---- daily quota -----------------------------------------------------------
def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _read_usage() -> dict:
    try:
        return json.loads(USAGE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def runs_left(user: str):
    """Live runs this user may still start today, or None when there is no quota."""
    quota = _int_env("FEA_DAILY_RUNS_PER_USER")
    if quota is None:
        return None
    with _lock:
        used = _read_usage().get(_today(), {}).get(user, 0)
    return max(quota - used, 0)


def count_run(user: str) -> None:
    """Record one started run. Only today's counts are kept."""
    with _lock:
        today = _today()
        day = _read_usage().get(today, {})
        day[user] = day.get(user, 0) + 1
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        USAGE_FILE.write_text(json.dumps({today: day}, indent=1))


# ---- retention ---------------------------------------------------------------
def prune_old_runs(runs_root: Path = RUNS_ROOT) -> int:
    """Delete run directories older than FEA_KEEP_RUNS_HOURS; returns how many.

    Only directories named the way RunPaths.new_run() names them are touched,
    so nothing else that happens to sit under out/runs/ can be removed.
    """
    hours = _int_env("FEA_KEEP_RUNS_HOURS")
    if hours is None or not Path(runs_root).is_dir():
        return 0
    cutoff = time.time() - hours * 3600
    removed = 0
    for d in Path(runs_root).iterdir():
        if d.is_dir() and _RUN_DIR.match(d.name) and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


# ---- the LLM budget -------------------------------------------------------
def budget_exhausted(error) -> bool:
    """True when the LLM provider refused for lack of credit (HTTP 402).

    Takes the exception, or the text of a failed run's log.
    """
    if isinstance(error, str):
        return "Error code: 402" in error
    return getattr(error, "status_code", None) == 402


BUDGET_MESSAGE = ("The live mode's LLM budget is used up for now. "
                  "The recorded examples still work: switch to replay above.")
