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
    h = call(tb, "hypothesis", {"hypothesis": "h", "rationale": "r", "layer": "kernels"}).split()[1]
    with pytest.raises(ValueError, match="expect"):
        call(tb, "trace", {"hypothesis_id": h, "expect": bad})


def test_hypothesis_needs_text_and_keeps_optional_fields(tb):
    with pytest.raises(ValueError):
        call(tb, "hypothesis", {"hypothesis": "", "rationale": "r", "layer": "kernels"})
    h = call(tb, "hypothesis", {"hypothesis": "decode attention is memory-bound", "rationale": "41% of bandwidth", "layer": "kernels",
                                "abandon_if": "no gain beyond the noise band"}).split()[1]
    rec = next(ledger.records(tb.s.ledger_root, kind="hypothesis"))
    assert rec["id"] == h and rec["abandon_if"] == "no gain beyond the noise band" and rec["run"] == tb.s.run_id


def test_a_measurement_carries_the_prediction_and_blocks_until_reflected(tb):
    h = call(tb, "hypothesis", {"hypothesis": "h", "rationale": "r", "layer": "kernels"}).split()[1]
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


def test_a_heldout_experiment_carries_scalars_only(tmp_path):
    base = {"kind": "submit", "run": "r", "config": {"split": "heldout"}, "metrics": {"x": {"delta_pct": 1.0}}}
    with pytest.raises(ledger.LedgerError):
        ledger.append({**base, "experiment": {"hypothesis": "h", "rows": [1, 2]}}, tmp_path)
    with pytest.raises(ledger.LedgerError):
        ledger.append({**base, "experiment": {"hypothesis": "h", "actual": [0.1, 0.2]}}, tmp_path)


def test_a_refused_call_needs_no_reflection(tb, monkeypatch):
    import shutil as sh
    from lab import proftools
    real = sh.which
    monkeypatch.setattr(proftools.shutil, "which", lambda n, *a, **k: None if n == "nsys" else real(n, *a, **k))
    h = call(tb, "hypothesis", {"hypothesis": "h", "rationale": "r", "layer": "kernels"}).split()[1]
    assert "refused" in call(tb, "trace", {"hypothesis_id": h, "expect": EXPECT})
    assert experiments.pending(tb.s.ledger_root, tb.s.run_id) is None


MECH = {"metric": "returncode", "value": 0, "unit": "absolute", "why": "the instrument exits cleanly"}


def test_mechanism_predictions_are_measured_at_their_paths(tb):
    h = call(tb, "hypothesis", {"hypothesis": "h", "rationale": "r", "layer": "kernels"}).split()[1]
    out = call(tb, "trace", {"hypothesis_id": h, "expect": {**EXPECT, "mechanism": [
        MECH, {**MECH, "metric": "calibrated.layers.runtime.gpu_idle_share"}]}})
    x = list(ledger.records(tb.s.ledger_root, tool="trace"))[-1]["experiment"]
    assert "mechanism" not in x["expect"]
    hit, miss = x["mechanism"]
    assert hit["actual"] == 0 and hit["gap"] == 0 and hit["why"] == MECH["why"]
    assert miss["actual"] is None and "not a number" in miss["actual_missing"]
    assert "returncode expected 0 (absolute), measured 0, gap +0" in out
    s = experiments.summary(tb.s.ledger_root, tb.s.run_id)
    assert s[0]["measurements"][0]["mechanism"] == x["mechanism"]


@pytest.mark.parametrize("bad", [[], {"metric": "m"}, [{**MECH, "value": "low"}], [{**MECH, "unit": "ms"}],
                                 [{**MECH, "why": ""}]])
def test_refused_with_a_malformed_mechanism(tb, bad):
    h = call(tb, "hypothesis", {"hypothesis": "h", "rationale": "r", "layer": "kernels"}).split()[1]
    with pytest.raises(ValueError, match="expect.mechanism"):
        call(tb, "trace", {"hypothesis_id": h, "expect": {**EXPECT, "mechanism": bad}})


def test_submit_takes_no_mechanism(tb):
    h = call(tb, "hypothesis", {"hypothesis": "h", "rationale": "r", "layer": "kernels"}).split()[1]
    with pytest.raises(ValueError, match="not taken by submit"):
        experiments.check(tb.s.ledger_root, tb.s.run_id, "submit",
                          {"hypothesis_id": h, "expect": {**EXPECT, "mechanism": [MECH]}})


def test_mechanism_change_pct_is_against_the_baseline_record():
    args = {"hypothesis_id": "h", "expect": {**EXPECT, "mechanism": [
        {**MECH, "metric": "calibrated.layers.runtime.gpu_idle_share", "unit": "change_pct", "value": -50},
        {**MECH, "metric": "calibrated.layers.serving.share", "unit": "change_pct", "value": 0}]}}
    result = {"calibrated": {"layers": {"runtime": {"gpu_idle_share": 0.1}, "serving": {"share": 0.2}}}}
    base = {"calibrated": {"layers": {"runtime": {"gpu_idle_share": 0.25}}}}
    drop, nobase = experiments.outcome("bench", args, result, base_result=base)["mechanism"]
    assert drop["actual"] == pytest.approx(-60) and drop["gap"] == pytest.approx(-10)
    assert nobase["actual"] is None and "no baseline value" in nobase["actual_missing"]


def hyp(tb, text="h", layer="kernels"):
    return call(tb, "hypothesis", {"hypothesis": text, "rationale": "r", "layer": layer}).split()[1]


def test_a_hypothesis_names_its_layer(tb):
    for bad in ({"hypothesis": "h", "rationale": "r"}, {"hypothesis": "h", "rationale": "r", "layer": "gpu"}):
        with pytest.raises(ValueError, match="layer"):
            call(tb, "hypothesis", bad)
    hyp(tb, layer="runtime")
    assert next(ledger.records(tb.s.ledger_root, kind="hypothesis"))["layer"] == "runtime"
    assert experiments.summary(tb.s.ledger_root, tb.s.run_id)[0]["layer"] == "runtime"


def test_a_batch_is_measured_in_any_order_and_reflected_on_before_the_next(tb):
    h1, h2 = hyp(tb, "a"), hyp(tb, "b", "runtime")
    call(tb, "trace", {"hypothesis_id": h1, "expect": EXPECT})
    call(tb, "trace", {"hypothesis_id": h2, "expect": EXPECT})                 # another open hypothesis: allowed
    r1, r2 = [r["id"] for r in ledger.records(tb.s.ledger_root, tool="trace")]
    with pytest.raises(ValueError, match=f"record {r1}"):
        call(tb, "trace", {"hypothesis_id": h1, "expect": EXPECT})             # its own last one is unreflected
    with pytest.raises(ValueError, match="before a new hypothesis"):
        hyp(tb, "c")
    call(tb, "reflect", {"record": r1, "reflection": "x"})
    call(tb, "trace", {"hypothesis_id": h1, "expect": EXPECT})
    r3 = list(ledger.records(tb.s.ledger_root, tool="trace"))[-1]["id"]
    with pytest.raises(ValueError, match=f"{r2}, {r3}"):
        hyp(tb, "c")
    call(tb, "reflect", {"record": r2, "reflection": "y"})
    call(tb, "reflect", {"record": r3, "reflection": "z"})
    hyp(tb, "c")


def test_a_bundle_names_its_hypotheses_and_why_they_are_independent(tb):
    h1, h2, h3 = hyp(tb, "a"), hyp(tb, "b", "runtime"), hyp(tb, "c")
    for bundle, why in (([h2], None), ([], "x"), ([h1], "x"), ([h2, h2], "x"), (["nope"], "x")):
        with pytest.raises(ValueError, match="bundle|independence"):
            call(tb, "trace", {"hypothesis_id": h1, "bundle": bundle, "independence": why, "expect": EXPECT})
    with pytest.raises(ValueError, match="must be one of this call's hypotheses"):
        call(tb, "trace", {"hypothesis_id": h1, "bundle": [h2], "independence": "x",
                           "expect": {**EXPECT, "mechanism": [{**MECH, "hypothesis": h3}]}})
    call(tb, "trace", {"hypothesis_id": h1, "bundle": [h2], "independence": "different layers",
                       "expect": {**EXPECT, "mechanism": [{**MECH, "hypothesis": h2}]}})
    rec = list(ledger.records(tb.s.ledger_root, tool="trace"))[-1]
    x = rec["experiment"]
    assert x["hypothesis"] == h1 and x["bundle"] == [h2] and x["independence"] == "different layers"
    assert x["mechanism"][0]["hypothesis"] == h2
    with pytest.raises(ValueError, match=f"record {rec['id']}"):
        call(tb, "trace", {"hypothesis_id": h2, "expect": EXPECT})             # h2 was measured in the bundle
    call(tb, "trace", {"hypothesis_id": h3, "expect": EXPECT})
    s = {h["id"]: h for h in experiments.summary(tb.s.ledger_root, tb.s.run_id)}
    assert [m["record"] for m in s[h1]["measurements"]] == [m["record"] for m in s[h2]["measurements"]] == [rec["id"]]
    assert s[h2]["measurements"][0]["bundle"] == [h2] and len(s[h3]["measurements"]) == 1
