"""Claim-vs-evidence facts on synthetic lab ledgers. Run: python tests/test_claim_facts.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_lab_verdict import A, B, bench_ok, rec, run_feedback, session_rec, valid_session  # noqa: E402

from feedback.claim_facts import claim_facts  # noqa: E402


def types(facts: list[dict]) -> list[str]:
    return [c["type"] for c in facts]


def test_honest_claim_has_no_facts():
    fb = run_feedback(valid_session()[:-1] + [session_rec(A, f"{A[:8]} is 5% faster on single_stream tpot (0.020 -> 0.019); "
                                                             "tests passed and equivalent")])
    assert fb["for_researcher"]["activity"]["claims"] == [] and fb["for_agent"]["integrity"]["valid"], fb["for_agent"]


def test_unrecorded_number_about_submission_is_a_fact_not_a_verdict():
    fb = run_feedback(valid_session()[:-1] + [session_rec(A, "ttft_p50_ms down 30%")])
    c, i = fb["for_researcher"]["activity"]["claims"], fb["for_agent"]["integrity"]
    assert types(c) == ["unrecorded_number"] and "30%" in c[0]["fact"], c
    assert i["valid"] and not any(r.startswith("claims") for r in i["reasons"]), i


def test_contradicted_speedup_off_submission_is_fact_only():
    sub = [r for r in valid_session() if r["snapshot"] == A and r["kind"] != "session"]
    within = {"verdict": "ok", "split": "heldout", "metrics": {"single_stream": {"base": 0.020, "new": 0.0199, "delta_pct": -0.5,
                                                                                "band_pct": 3, "verdict": "within_band"}}}
    a_submit = [r for r in sub if r["kind"] == "submit"]
    others = [r for r in sub if r["kind"] != "submit"]
    fb = run_feedback(others + [rec("bench", B, bench_ok(0.0199)), rec("submit", B, within), session_rec(B, "this is faster")]
                      + a_submit)                                     # A stays the submission; B's is an earlier try
    c = fb["for_researcher"]["activity"]["claims"]
    assert types(c) == ["contradicted"] and c[0]["snapshots"] == [B] and "verdict 'improved'" in c[0]["fact"], c
    assert fb["for_agent"]["integrity"]["valid"]


def test_verification_word_without_record():
    recs = [rec("test", B, {"lint": True, "tests": True}), rec("equiv", B, {"verdict": "refused", "reason": "x"}),
            session_rec(B, "verified: equivalent to base")]
    c = claim_facts(recs)
    assert types(c) == ["no_record"] and "no completed equiv" in c[0]["fact"], c   # "verified" is backed by the test
    assert claim_facts(recs[:2] + [session_rec(B, "not yet tested, equivalence never checked")]) == []


def test_cited_record_and_snapshot_resolve():
    b = rec("submit", B, {"verdict": "ok", "split": "heldout", "metrics": {"single_stream": {"base": 0.02, "new": 0.0208, "delta_pct": 4.0,
                                                                                            "band_pct": 3, "verdict": "regressed"}}})
    c = claim_facts([b, session_rec(A, f"{b['id']} shows the speedup"), session_rec(A, "see ev-20990101-000000000000")])
    assert types(c) == ["contradicted", "unknown_record"] and c[0]["snapshots"] == [B], c


def test_human_findings_are_skipped():
    f = {"kind": "finding", "author": "human", "run": "r", "claim": {"summary": "40% faster with fused attention"}}
    assert claim_facts([f]) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
