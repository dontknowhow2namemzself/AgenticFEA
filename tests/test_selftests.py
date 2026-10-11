"""The modules' own self-tests, run as a user would run them. None needs an
API key: each tests the deterministic part of its layer."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SELFTESTS = {
    "schema": ["schema.py"],                                   # parameter validation, both geometries
    "task_understanding": ["task_understanding.py", "--selftest"],   # gaps, defaults, contradictions
    "vector_store": ["knowledge/vector_store.py", "--selftest"],     # retrieval plumbing, offline
}


@pytest.mark.parametrize("name", sorted(SELFTESTS))
def test_selftest(name):
    r = subprocess.run([sys.executable, *SELFTESTS[name]], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout + r.stderr)[-4000:]
