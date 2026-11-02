"""The regimes' load searches started from the base's result and resolved to the noise band, on synthetic probes (no
server). Run: python tests/test_regimes_suite.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from regimes import suite  # noqa: E402
from regimes.runner import find_goodput  # noqa: E402


def test_a_hinted_search_starts_at_the_base_and_stops_at_the_band():
    probe = lambda x: {"valid": True, "meets_limits": x <= 10.0}
    cold = find_goodput(probe, **suite.Ctx("u", "m").search("bursty", 3.3))
    warm = find_goodput(probe, **suite.Ctx("u", "m", hints={"bursty": {"load": 10.0}},
                                           bands={"bursty": 2.0}).search("bursty", 3.3))
    assert warm["probes"][0]["load"] == 10.0 and len(warm["probes"]) < len(cold["probes"])
    assert 10.0 * 0.98 <= warm["goodput"] <= 10.0
    # a regression: the base's load fails, the search steps down by the band
    worse = find_goodput(lambda x: {"valid": True, "meets_limits": x <= 9.0},
                         **suite.Ctx("u", "m", hints={"bursty": {"load": 10.0}}, bands={"bursty": 2.0}).search("bursty", 3.3))
    assert 9.0 * 0.98 <= worse["goodput"] <= 9.0
    # a large change either way is still found inside the probe cap: the step grows on each move
    for cap in (4.0, 25.0):
        far = find_goodput(lambda x: {"valid": True, "meets_limits": x <= cap},
                           **suite.Ctx("u", "m", hints={"bursty": {"load": 10.0}}, bands={"bursty": 2.0}).search("bursty", 3.3))
        assert far["goodput"] is not None and cap * 0.9 <= far["goodput"] <= cap, (cap, far["probes"])


def test_the_search_reports_its_bracket_and_never_stops_on_its_budget_first():
    probe = lambda x: {"valid": True, "meets_limits": x <= 3000.0}
    far = find_goodput(probe, 1.0)
    assert far["stopped"] == "resolved" and far["goodput"] <= 3000.0 < far["fails_at"]
    assert (far["fails_at"] - far["goodput"]) / far["fails_at"] <= 0.1
    fine = find_goodput(probe, **suite.Ctx("u", "m", tol=0.02).search("bursty", 1.0))
    assert (fine["fails_at"] - fine["goodput"]) / fine["fails_at"] <= 0.02
    client = find_goodput(lambda x: {"valid": x < 8, "meets_limits": True}, 1.0)
    assert client["stopped"] == "invalid" and client["fails_at"] is None and client["goodput"] == 4.0


def _saturated(capacity: int, hints: dict | None = None) -> tuple[dict, list[int]]:
    """saturated() against a server that meets the limits up to `capacity` concurrent requests."""
    calls: list[int] = []
    ctx = suite.Ctx("u", "m", hints=hints or {})
    ctx.pool = lambda cls=None: [{"id": "a", "prompt": "x"}]
    ctx.run_closed = lambda reqs, c, duration: (calls.append(c) or c, 1.0)
    real = suite._warmup, suite.summarize
    suite._warmup = lambda ctx, pool: None
    suite.summarize = lambda c, limits, elapsed: {"valid": True, "meets_limits": c <= capacity, "invalid_reasons": [],
                                                  "good_req_per_s": float(min(c, capacity))}
    try:
        return suite.saturated(ctx), calls
    finally:
        suite._warmup, suite.summarize = real


def test_saturated_starts_at_the_bases_concurrency():
    cold, cold_calls = _saturated(64)
    warm, warm_calls = _saturated(64, {"saturated": {"concurrency": 64}})
    assert cold["value"] == warm["value"] == 64.0 and cold["concurrency"] == warm["concurrency"] == 64
    assert warm_calls == [64, 128, 256, 96, 80, 72, 68, 64] and len(cold_calls) > len(warm_calls)   # then bisected
    down, calls = _saturated(16, {"saturated": {"concurrency": 64}})                   # a regression: halve until it holds
    assert down["value"] == 16.0 and calls == [64, 32, 16, 24, 20, 18, 17, 16]
    assert cold["stopped"] == down["stopped"] == "resolved"
    capped, calls = _saturated(10 ** 6)
    assert capped["stopped"] == "max_concurrency" and max(calls) == 4096


def test_saturated_bisects_between_the_doubling_levels():
    res, calls = _saturated(200)                       # 128 passes, 256 fails: the result is within 10% of 200, not 128
    assert 200 * 0.9 <= res["value"] <= 200 and res["concurrency"] == res["value"]
    assert calls[-1] == res["concurrency"]             # the confirmation reran the best bisected level


def _fast():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_engine import fast_arrivals
    return fast_arrivals()


def test_realistic_maps_sittings_to_conversations_and_salts_each():
    from regimes import workload as W
    convs = [{"id": f"c{k}", "session_id": f"s{k}", "max_tokens": 16, "expected_output_tokens": 8,
              "expected_turn_output_tokens": [3, 4, 5],
              "messages": [{"role": "user", "content": f"a{k}"}, {"role": "assistant", "content": f"b{k}"},
                           {"role": "user", "content": f"c{k}"}, {"role": "assistant", "content": f"d{k}"},
                           {"role": "user", "content": f"e{k}"}]} for k in range(3)]
    single = [{"id": f"p{k}", "prompt": f"q{k}", "max_tokens": 16} for k in range(3)]
    reqs = W.realistic(convs + single, 20.0, 5.0, "seed", _fast(), system="sys")
    assert reqs and reqs == sorted(reqs, key=lambda q: q.at_s) and all(0 <= q.at_s < 5.0 for q in reqs)
    assert reqs == W.realistic(convs + single, 20.0, 5.0, "seed", _fast(), system="sys")
    by: dict[str, list] = {}
    for q in reqs:
        by.setdefault(q.session_id, []).append(q)
    salts = {q.messages[1]["content"].split("]")[0] for q in reqs}
    assert len(salts) == len(by)                                   # one salt per sitting, none shared
    assert all(q.messages[0] == {"role": "system", "content": "sys"} for q in reqs)
    for sid, turns in by.items():
        assert len({q.id.split("#")[0].split(".t")[0] for q in turns}) == 1   # one conversation per sitting
        for q in turns:                                            # turn j carries j prior exchanges of it
            assert len(q.messages) == 2 + 2 * q.turn_index, (sid, q.turn_index, len(q.messages))
            if sid.startswith("s"):                                # a conversation: its own text, turn by turn
                k = sid.split("#")[0][1:]
                assert q.messages[1]["content"].endswith(f"a{k}") and q.messages[-1]["content"].endswith("ace"[q.turn_index] + k)
            if q.turn_index:
                assert q.max_tokens == [3, 4, 5][q.turn_index] and q.ignore_eos
    assert any(q.turn_index > 0 for q in reqs)                     # resumed and continuing sittings are there
    assert all(len(turns) == len({q.turn_index for q in turns}) for turns in by.values())
    # a pool of single-turn records gets only the turns it can supply
    only = W.realistic(single, 20.0, 5.0, "seed", _fast())
    assert only and all(q.turn_index == 0 for q in only)


def test_once_salts_every_request_and_splits_over_parts():
    from regimes import workload as W
    order = [{"id": f"r{k}", "prompt": "p", "max_tokens": 4} for k in range(7)]
    reqs = list(W.Once(order))
    assert [q.id for q in reqs] == [f"r{k}#{k}" for k in range(7)]
    assert [q.messages[0]["content"] for q in reqs] == [f"[{k + 1}] p" for k in range(7)]
    parts = [list(W.Once(order).part(k, 3)) for k in range(3)]
    assert sorted(q.id for p in parts for q in p) == sorted(q.id for q in reqs)
    cyc = W.Cycle(order)
    first = [next(iter(cyc)) for _ in range(1)]
    assert first[0].messages[0]["content"] == "[1] p"              # salted from the opening request on


def test_saturated_reports_its_curve():
    res, _ = _saturated(64)
    assert [pt["concurrency"] for pt in res["curve"]] == sorted(lv["concurrency"] for lv in res["levels"])
    assert all(set(pt) == {"concurrency", "output_tok_per_s", "tok_per_s_per_user", "ttft_p50_s", "meets_limits"}
               for pt in res["curve"])
    assert [pt["meets_limits"] for pt in res["curve"]] == [pt["concurrency"] <= 64 for pt in res["curve"]]


def test_offline_sends_the_whole_pool_at_once():
    seen = {}
    ctx = suite.Ctx("u", "m")
    ctx.pool = lambda cls=None: [{"id": f"r{k}", "prompt": "p", "max_tokens": 4} for k in range(9)]
    ctx.run_closed = lambda reqs, c, duration, procs=None: seen.update(n=len(list(reqs)), c=c, d=duration, p=procs) or ([], 3.0)
    real = suite._warmup, suite.summarize
    suite._warmup = lambda ctx, pool: None
    summary = {"valid": True, "invalid_reasons": [], "output_tok_per_s": 100.0, "meets_limits": True, "error": 0}
    suite.summarize = lambda rows, limits, elapsed: dict(summary)
    try:
        res = suite.offline(ctx)
        summary["error"] = 2                                       # rejected records: no offline number
        rejected = suite.offline(ctx)
    finally:
        suite._warmup, suite.summarize = real
    assert seen == {"n": 9, "c": 9, "d": None, "p": 4} and res["value"] == 100.0 and res["objective"] == "output_tok_per_s"
    assert rejected["value"] is None and not rejected["valid"] and rejected["invalid_reasons"] == ["2 requests rejected"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
