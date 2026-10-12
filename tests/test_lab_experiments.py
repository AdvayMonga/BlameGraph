"""Experiment records: a measured call needs an open hypothesis and a prediction, the record keeps both with the
computed gap, and the next measured call waits for a reflection. Uses the fake instruments from the proftools tests."""
from __future__ import annotations

import json

import pytest

from lab import experiments, ledger
pytest_plugins = ["tests.test_lab_proftools"]   # its `tb` fixture: a Toolbox with fake instruments

EXPECT = {"metric": "single_stream", "value": 0.02, "unit": "absolute", "why": "fewer bytes per step"}


def spec(tb, name):
    return next(s for s in tb.specs() if s.name == name)


def call(tb, name, args):
    return spec(tb, name).fn(args)


def test_measured_tools_require_hypothesis_and_expect_in_their_schema(tb):
    specs = {s.name: s for s in tb.specs()}
    for name in experiments.MEASURED:
        assert {"hypothesis_id", "expect"} <= set(specs[name].schema["required"]), name
        assert "Requires `hypothesis_id`" in specs[name].description
    assert "hypothesis_id" not in specs["test"].schema["properties"] and "hypothesis" in specs and "reflect" in specs


@pytest.mark.parametrize("args,why", [
    ({}, "hypothesis_id"), ({"hypothesis_id": "nope", "expect": EXPECT}, "hypothesis_id"),
    ({"expect": EXPECT}, "hypothesis_id"),
])
def test_refused_without_an_open_hypothesis(tb, args, why):
    with pytest.raises(ValueError, match=why):
        call(tb, "trace", args)
    assert not list(ledger.records(tb.s.ledger_root, tool="trace"))


@pytest.mark.parametrize("bad", [None, {"metric": "x"}, {**EXPECT, "value": "fast"}, {**EXPECT, "value": True},
                                 {**EXPECT, "unit": "ms"}, {**EXPECT, "why": " "}, {**EXPECT, "value": float("nan")}])
def test_refused_without_a_numeric_prediction(tb, bad):
    h = call(tb, "hypothesis", {"hypothesis": "h", "rationale": "r"}).split()[1]
    with pytest.raises(ValueError, match="expect"):
        call(tb, "trace", {"hypothesis_id": h, "expect": bad})


def test_hypothesis_needs_text_and_keeps_optional_fields(tb):
    with pytest.raises(ValueError):
        call(tb, "hypothesis", {"hypothesis": "", "rationale": "r"})
    h = call(tb, "hypothesis", {"hypothesis": "decode attention is memory-bound", "rationale": "41% of bandwidth",
                                "abandon_if": "no gain beyond the noise band"}).split()[1]
    rec = next(ledger.records(tb.s.ledger_root, kind="hypothesis"))
    assert rec["id"] == h and rec["abandon_if"] == "no gain beyond the noise band" and rec["run"] == tb.s.run_id


def test_a_measurement_carries_the_prediction_and_blocks_until_reflected(tb):
    h = call(tb, "hypothesis", {"hypothesis": "h", "rationale": "r"}).split()[1]
    out = call(tb, "trace", {"hypothesis_id": h, "expect": {**EXPECT, "metric": "kernel_ms"}})
    rec = list(ledger.records(tb.s.ledger_root, tool="trace"))[-1]
    x = rec["experiment"]
    assert x["hypothesis"] == h and x["expect"]["metric"] == "kernel_ms" and x["actual"] is None
    assert "not a numeric metric" in x["actual_missing"] and f"Experiment record {rec['id']}" in out
    with pytest.raises(ValueError, match="no reflection"):
        call(tb, "trace", {"hypothesis_id": h, "expect": EXPECT})
    with pytest.raises(ValueError):
        call(tb, "reflect", {"record": "other", "reflection": "x"})
    call(tb, "reflect", {"record": rec["id"], "reflection": "trace has no scalar metric", "alternative": "h2"})
    with pytest.raises(ValueError, match="already"):
        call(tb, "reflect", {"record": rec["id"], "reflection": "again"})
    call(tb, "trace", {"hypothesis_id": h, "expect": EXPECT})          # unblocked
    s = experiments.summary(tb.s.ledger_root, tb.s.run_id)
    assert len(s) == 1 and len(s[0]["measurements"]) == 2
    assert s[0]["measurements"][0]["reflection"] == "trace has no scalar metric" and s[0]["measurements"][0]["alternative"] == "h2"


def test_unmeasured_tools_need_nothing(tb):
    assert call(tb, "note", {"text": "x"}) == "noted"
    assert json.loads(call(tb, "budget", {}))["cap_usd"] == 100.0


def test_outcome_gap_absolute_and_change_and_submit():
    a = {"hypothesis_id": "h", "expect": {"metric": "saturated", "value": 10.0, "unit": "absolute", "why": "w"}}
    o = experiments.outcome("bench", a, {"metrics": {"saturated": {"value": 12.0}}}, band_pct=4.0)
    assert o["actual"] == 12.0 and o["gap"] == 2.0 and o["gap_pct_of_expected"] == pytest.approx(20) and o["noise_band_pct"] == 4.0
    c = {"hypothesis_id": "h", "expect": {"metric": "saturated", "value": 10.0, "unit": "change_pct", "why": "w"}}
    o = experiments.outcome("bench", c, {"metrics": {"saturated": {"value": 12.0}}}, baseline={"saturated": {"value": 10.0}})
    assert o["actual"] == pytest.approx(20) and o["gap"] == pytest.approx(10)
    o = experiments.outcome("bench", c, {"metrics": {"saturated": {"value": 12.0}}})
    assert o["actual"] is None and "baseline" in o["actual_missing"] and "gap" not in o
    o = experiments.outcome("submit", c, metrics={"saturated": {"delta_pct": 7.5, "new": 10.75}})
    assert o["actual"] == 7.5 and o["gap"] == pytest.approx(-2.5)
    o = experiments.outcome("submit", c, metrics={"bursty": {}})
    assert "not a regime" in o["actual_missing"]


def test_a_heldout_record_may_carry_its_experiment(tmp_path):
    rec = ledger.append({"kind": "submit", "run": "r", "config": {"split": "heldout"}, "metrics": {"x": {"delta_pct": 1.0}},
                         "experiment": {"hypothesis": "h", "actual": 1.0}}, tmp_path)
    assert rec["experiment"]["actual"] == 1.0
