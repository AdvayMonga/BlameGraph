"""A hand-made run dir becomes bench / equiv records the index reads like the lab's own."""

import json

from lab import import_runs, ledger


def test_import_dir_writes_bench_and_equiv_records(tmp_path):
    d = tmp_path / "s9"
    d.mkdir()
    results = [{"regime": "saturated", "objective": "goodput_req_per_s", "value": 4.0, "better": "higher",
                "summary": {"n": 10}, "valid": True, "invalid_reasons": [], "probes": []}]
    (d / "regimes.json").write_text(json.dumps({"model": "m", "target": "t", "tier": "short", "split": "seen",
                                                "seed": 0, "corpus_version": "c" * 8, "at": "2026-10-18T00:00:00",
                                                "results": results}))
    (d / "equiv.json").write_text(json.dumps({"result": {"verdict": "pass", "reasons": [], "gates": {"accuracy": {
        "passed": True}}, "metrics": {"accuracy": {"score_ratio": 1.0}}, "thresholds": {}}}))
    root = tmp_path / "ledger"
    root.mkdir()
    recs = import_runs.import_dir(d, "manual-20261018", "abc1234", {"PREFILL_MODE": "mixed"}, None, "H200", root=root)
    kinds = sorted(r["kind"] for r in recs)
    assert kinds == ["bench", "equiv"]
    bench = next(r for r in recs if r["kind"] == "bench")
    assert bench["run"] == "manual-20261018" and bench["snapshot"] == "abc1234" and bench["args"]["label"] == "s9"
    assert bench["result"]["metrics"] == {"saturated": {"objective": "goodput_req_per_s", "value": 4.0,
                                                        "better": "higher", "valid": True, "invalid_reasons": []}}
    assert bench["config"]["env"] == {"PREFILL_MODE": "mixed"} and bench["cost"] == {"usd": None, "seconds": None}
    equiv = next(r for r in recs if r["kind"] == "equiv")
    assert equiv["result"]["verdict"] == "pass" and equiv["result"]["passed"] is True
    assert len(list(ledger.records(root))) == 2
