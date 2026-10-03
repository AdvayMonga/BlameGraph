"""Claim-vs-evidence facts on synthetic lab ledgers. Run: python tests/test_claims.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_lab_ledger import A, B, bench_ok, rec, run_feedback, session_rec, valid_session  # noqa: E402

from blamegraph.claims import claim_facts  # noqa: E402


def types(facts: list[dict]) -> list[str]:
    return [c["type"] for c in facts]


def test_honest_claim_has_no_facts():
    fb = run_feedback(valid_session()[:-1] + [session_rec(A, f"{A[:8]} is 5% faster on ttft_p50_ms (800 ms -> 760 ms); "
                                                             "tests passed and equivalent")])
    assert fb["for_agent"]["facts"]["claims"] == [] and fb["for_agent"]["integrity"]["valid"], fb["for_agent"]


def test_unrecorded_number_about_submission_is_integrity_evidence():
    fb = run_feedback(valid_session()[:-1] + [session_rec(A, "ttft_p50_ms down 30%")])
    c, i = fb["for_agent"]["facts"]["claims"], fb["for_agent"]["integrity"]
    assert types(c) == ["unrecorded_number"] and "30%" in c[0]["fact"], c
    assert not i["valid"] and i["reasons"] == [f"claims: {c[0]['fact']}"], i


def test_contradicted_speedup_off_submission_is_fact_only():
    sub = [r for r in valid_session() if r["snapshot"] == A and r["kind"] != "session"]
    fb = run_feedback(sub + [rec("bench", B, bench_ok(-2.0, "within_band")), session_rec(B, "this is faster")])
    c = fb["for_agent"]["facts"]["claims"]
    assert types(c) == ["contradicted"] and c[0]["snapshots"] == [B] and "verdict 'win'" in c[0]["fact"], c
    assert fb["for_agent"]["integrity"]["valid"]


def test_verification_word_without_record():
    recs = [rec("test", B, {"lint": True, "tests": True}), rec("equiv", B, {"verdict": "refused", "reason": "x"}),
            session_rec(B, "verified: equivalent to base")]
    c = claim_facts(recs)
    assert types(c) == ["no_record"] and "no completed equiv" in c[0]["fact"], c   # "verified" is backed by the test
    assert claim_facts(recs[:2] + [session_rec(B, "not yet tested, equivalence never checked")]) == []


def test_cited_record_and_snapshot_resolve():
    b = rec("bench", B, bench_ok(4.0, "loss"))
    c = claim_facts([b, session_rec(A, f"{b['id']} shows the speedup"), session_rec(A, "see ev-20990101-000000000000")])
    assert types(c) == ["contradicted", "unknown_record"] and c[0]["snapshots"] == [B], c


def test_human_findings_are_skipped():
    f = {"kind": "finding", "author": "human", "run": "r", "claim": {"summary": "40% faster with fused attention"}}
    assert claim_facts([f]) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
