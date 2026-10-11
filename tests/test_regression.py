"""Both prepared cases through the deterministic chain, against the numbers the
pipeline was verified with.

No language model is called: prepared parameters skip layers 1-2, a failed run
is not diagnosed, and without an API key the reference is picked by the
deterministic scan that retrieval falls back to.
"""

import pytest

import pipeline_run
import report
from params import CASES
from paths import RunPaths

# case -> (relative error [%] against the reference, the validity checks it must flag).
# The default cantilever's 1000 N is beyond S355 yield (600 MPa nominal), and
# the pipeline is meant to say so.
EXPECTED = {"cantilever_beam": (0.0561, ["stress below yield"]),
            "plate_with_hole": (0.293, [])}


@pytest.fixture(scope="module", params=sorted(EXPECTED))
def run(request, tmp_path_factory):
    """One run per case, shared by the tests below; the run directory stays
    alive for the whole module, so the report can still read the renders."""
    case = request.param
    rec = pipeline_run.run_pipeline(f"test: {case}", run_id="test", params=CASES[case],
                                    paths=RunPaths(tmp_path_factory.mktemp(case)), diagnose=False)
    return case, rec


def stages(rec):
    return {s["id"]: s for s in rec["stages"]}


def test_verdict_and_error(run):
    case, rec = run
    want_error, _ = EXPECTED[case]
    assert rec["status"] == "complete"
    assert rec["verdict"] == "PASS"
    assert stages(rec)["verify"]["data"]["error_pct"] == pytest.approx(want_error, abs=5e-4)


def test_validity_warnings(run):
    case, rec = run
    _, want_warnings = EXPECTED[case]
    assert sorted(w.split(":")[0] for w in rec.get("warnings", [])) == sorted(want_warnings)


def test_every_stage_ran(run):
    _, rec = run
    st = stages(rec)
    assert list(st) == ["understand", "parameterize", "preprocess", "solve", "physics",
                        "postprocess", "verify"]
    assert all(s["status"] in ("pass", "warning") for s in rec["stages"])
    assert st["physics"]["data"]["passed"] == st["physics"]["data"]["total"]
    # renders, 3D view data and the lossless check against the .frd
    assert all(a["status"] == "pass" for a in st["postprocess"]["actions"])
    assert all(isinstance(s["seconds"], float) for s in rec["stages"])


def test_report_is_a_reproducible_pdf(run):
    _, rec = run
    pdf = report.build(rec)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 100_000          # text and the figures
    assert report.build(rec) == pdf    # the same record gives the same file
