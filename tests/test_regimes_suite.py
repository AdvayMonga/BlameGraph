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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
