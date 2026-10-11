"""The recorded examples in demo/: what the dashboard replays with no key and
no solver, and what the public repo publishes."""

import json
import re
from pathlib import Path

import pytest

import report
from schema import validate_params

ROOT = Path(__file__).resolve().parents[1]
DEMOS = sorted(p.parent.name for p in (ROOT / "demo").glob("*/run.json"))


def load(name):
    return json.loads((ROOT / "demo" / name / "run.json").read_text())


def test_there_are_demos():
    assert DEMOS == ["cantilever_pass", "plate_pass", "stubby_fail"]


@pytest.mark.parametrize("name", DEMOS)
def test_no_machine_paths(name):
    # demo/*/run.json is committed: no absolute path of the recording machine
    # (user name, folder layout) may be in it.
    text = (ROOT / "demo" / name / "run.json").read_text()
    assert not re.search(r"/Users/|/home/|/private/|/var/folders/|/tmp/|[A-Z]:\\\\", text)


@pytest.mark.parametrize("name", DEMOS)
def test_recorded_parameters_validate(name):
    validate_params(report.recorded_params(load(name)))


@pytest.mark.parametrize("name", DEMOS)
def test_every_file_it_shows_exists(name):
    rec = load(name)
    files = [a["path"] for s in rec["stages"] for a in s.get("artifacts", []) if "path" in a]
    files += [p for s in rec["stages"] for a in s.get("artifacts", [])
              for p in a.get("views", {}).values()]
    assert files
    assert all((ROOT / f).is_file() for f in files), [f for f in files if not (ROOT / f).is_file()]


@pytest.mark.parametrize("name", DEMOS)
def test_report(name):
    assert report.build(load(name)).startswith(b"%PDF")
