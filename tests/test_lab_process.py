"""Process evals on lab ledgers (diagnostics/lab_process.py): prediction directions, the four cases, bundles never
measured apart, reflections checked against recorded numbers."""
from __future__ import annotations

from diagnostics.lab_process import process, record_facts


def rec(i, h, actual, expected, mech=(), bundle=None, band=1.0, unit="change_pct"):
    x = {"hypothesis": h, "expect": {"metric": "saturated", "value": expected, "unit": unit, "why": "w"},
         "actual": actual, "gap": actual - expected,
         "gap_pct_of_expected": (actual - expected) / abs(expected) * 100 if expected else None,
         "noise_band_pct": band}
    if mech:
        x["mechanism"] = [{"metric": "calibrated.layers.runtime.gpu_idle_share", "value": v, "unit": "change_pct",
                           "why": "w", "actual": a, "gap": None if a is None else a - v} for v, a in mech]
    if bundle:
        x["bundle"], x["independence"] = bundle, "different layers"
    return {"id": f"ev-20261022-{i:012x}", "kind": "bench", "tool": "bench", "run": "r", "snapshot": "abc",
            "result": {"metrics": {"saturated": {"value": 7.0, "valid": True}}}, "experiment": x}


def test_the_four_cases():
    cases = [record_facts(rec(1, "h1", 8, 5, [(-40, -50)]))["case"],
             record_facts(rec(2, "h2", 8, 5, [(-40, 10)]))["case"],
             record_facts(rec(3, "h3", 0.5, 5, [(-40, -45)]))["case"],
             record_facts(rec(4, "h4", -3, 5, [(-40, 5)]))["case"]]
    assert cases == ["worked as intended", "outcome held, mechanism did not", "mechanism held, outcome did not",
                     "neither held"]


def test_absolute_and_unmeasured_predictions_have_no_direction():
    f = record_facts(rec(1, "h1", 8, 5, [(-40, None)], unit="absolute"))
    assert "outcome_direction_held" not in f and "case" not in f
    assert f["mechanism_measured"] == 0 and f["mechanism_predicted"] == 1


def test_run_totals_bundles_and_reflections():
    recs = [rec(1, "h1", 8, 5, [(-40, -50)]), rec(2, "h2", 8, 5, bundle=["h3"]), rec(3, "h2", 2, 1),
            {"id": "ev-20261022-00000000000a", "kind": "reflection", "run": "r", "record": rec(1, "h1", 8, 5)["id"],
             "reflection": "saturated rose 31.5% as the idle share fell"}]
    p = process(recs)
    assert p["measurements"] == 3 and p["outcome_direction"] == {"held": 3, "of": 3}
    assert p["mechanism_direction"] == {"held": 1, "of": 1} and p["cases"] == {"worked as intended": 1}
    assert p["unreflected"] == [recs[1]["id"], recs[2]["id"]]
    assert p["bundles"] == 1 and p["unseparated_bundles"] == [{"record": recs[1]["id"], "tool": "bench",
                                                         "never_measured_alone": ["h3"]}]
    assert any("31.5" in c["fact"] for c in p["reflections_vs_records"])


def test_a_prediction_of_no_change_holds_inside_the_band_and_no_band_is_not_judged():
    assert record_facts(rec(1, "h1", 0.3, 0))["outcome_direction_held"] is True
    assert record_facts(rec(2, "h2", 3.0, 0))["outcome_direction_held"] is False
    f = record_facts(rec(3, "h3", 0.01, 5, band=None))
    assert "outcome_direction_held" not in f and f["outcome_direction_missing"]


def test_a_reflection_quoting_the_measured_change_is_not_flagged():
    r = rec(1, "h1", 12.3, 10)
    p = process([r, {"id": "ev-20261022-00000000000b", "kind": "reflection", "run": "r", "record": r["id"],
                     "reflection": "saturated rose 12.3%, 2.3 points above the 10% predicted"}])
    assert p["reflections_vs_records"] == []


def test_a_failed_solo_measurement_does_not_separate_a_bundle_and_odd_records_do_not_crash():
    solo = rec(2, "h3", 1, 1)
    solo["experiment"].update(actual=None, actual_missing="no value")
    p = process([rec(1, "h2", 8, 5, bundle=["h3"]), solo, {"id": "x", "kind": "bench", "experiment": {}},
                 {"id": "y", "kind": "bench", "experiment": {"hypothesis": "h9", "expect": None, "actual": None}}])
    assert p["unseparated_bundles"][0]["never_measured_alone"] == ["h2", "h3"]
