"""calibrate, measure and gaps with a fake worker: the bands and the baseline from the same runs, a failed gate
skipping the bench, the layers ranked by recoverable time."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lab.loop.calibrate import NOISE_RUNS, calibrate
from lab.loop.gaps import gaps
from lab.loop.measure import measure
from lab.loop.types import Candidate, Measurement
from lab.records import ledger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_fixtures import make_repo  # noqa: E402
from loop_fakes import FakeWorker, make_peaks, make_target  # noqa: E402


def _calibrated(tmp_path, wk, **kw):
    t = make_target(tmp_path, make_repo(tmp_path), **kw)
    return t, calibrate(t, ("saturated",), wk, tmp_path / "work", tmp_path / "ledger", make_peaks(tmp_path))


def test_calibrate_takes_the_bands_and_the_baseline_from_the_same_runs(tmp_path):
    wk = FakeWorker()
    t, cal = _calibrated(tmp_path, wk)
    jobs = [j for j, _, _ in wk.calls]
    assert jobs == ["measure"] * NOISE_RUNS + ["equiv", "profile"]
    assert [a["seed"] for j, _, a in wk.calls if j == "measure"] == list(range(NOISE_RUNS))
    noise = [r["result"] for r in ledger.records(tmp_path / "ledger", kind="noise")]
    assert [n["values"] for n in noise] == [[100.0, 101.0, 102.0]] and cal.bands == {"saturated": noise[0]["band_pct"]}
    assert cal.baseline.metrics["saturated"]["value"] == 100.0 and cal.baseline.correct
    assert cal.baseline.profile["layers"]["kernels"]["share"] == pytest.approx(0.4)
    assert cal.peaks.source == "measured" and cal.reference == t.reference_dir and len(cal.commit) == 40
    assert (tmp_path / "ledger" / cal.baseline.profile["bundle"] / "meta.json").exists()


def test_calibrate_builds_a_missing_reference_first(tmp_path):
    wk = FakeWorker()
    t, _ = _calibrated(tmp_path, wk, reference=False)
    assert wk.calls[0][0] == "reference" and (t.reference_dir / "meta.json").exists()


def test_calibrate_refuses_without_peaks_or_a_band(tmp_path):
    t = make_target(tmp_path, make_repo(tmp_path))
    with pytest.raises(RuntimeError, match="no peaks"):
        calibrate(t, ("saturated",), FakeWorker(), tmp_path / "w1", tmp_path / "ledger", tmp_path / "none")

    class Invalid(FakeWorker):
        def measure(self, inputs, args, out):
            r = super().measure(inputs, args, out)
            r["results"][0]["valid"] = False
            return r
    with pytest.raises(RuntimeError, match="no noise band"):
        calibrate(t, ("saturated",), Invalid(), tmp_path / "w2", tmp_path / "ledger", make_peaks(tmp_path))


def test_measure_benches_a_passing_candidate_from_where_the_baseline_ended(tmp_path):
    wk = FakeWorker({"fast": 150.0})
    t, cal = _calibrated(tmp_path, wk)
    wk.calls.clear()
    (tmp_path / "fast").mkdir()
    m = measure(Candidate(tmp_path / "fast"), cal, t, wk, tmp_path / "turn", tmp_path / "ledger")
    assert [j for j, _, _ in wk.calls] == ["equiv", "measure", "profile"]
    args = wk.calls[1][2]
    assert args["seed"] == 0 and args["split"] == "seen" and args["bands"] == cal.bands and "hints" in args
    assert m.correct and m.metrics["saturated"]["value"] == 150.0 and m.profile["layers"]


def test_measure_stops_at_a_failed_gate(tmp_path):
    wk = FakeWorker(failing=("wrong",))
    t, cal = _calibrated(tmp_path, wk)
    wk.calls.clear()
    (tmp_path / "wrong").mkdir()
    m = measure(Candidate(tmp_path / "wrong"), cal, t, wk, tmp_path / "turn", tmp_path / "ledger")
    assert [j for j, _, _ in wk.calls] == ["equiv"]
    assert not m.correct and m.equiv["reasons"] == ["kl too high"] and m.metrics == {}


def test_a_failed_profile_leaves_the_measurement_standing(tmp_path):
    class NoProfile(FakeWorker):
        def profile(self, inputs, args, out):
            raise RuntimeError("nsys missing")
    wk = NoProfile()
    _, cal = _calibrated(tmp_path, wk)
    assert cal.baseline.metrics and "nsys missing" in cal.baseline.profile["error"] and gaps(cal.baseline) == ()


def test_gaps_rank_the_layers_by_recoverable_time():
    layers = {"kernels": {"share": 0.5, "pct_of_ceiling": 90.0}, "model_execution": {"share": 0.3, "pct_of_ceiling": 50.0},
              "runtime": {"share": 0.1, "pct_of_ceiling": None}, "serving": {"share": None}}
    got = gaps(Measurement(True, {}, profile={"layers": layers}))
    assert [(g.layer, round(g.pct_of_time), g.pct_of_roofline, round(g.recoverable_pct)) for g in got] == [
        ("model_execution", 30, 50.0, 15), ("runtime", 10, None, 10), ("kernels", 50, 90.0, 5)]
