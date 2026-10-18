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
    assert warm_calls == [64, 128, 256, 64] and len(cold_calls) > len(warm_calls)      # probes, then the confirmation
    down, calls = _saturated(16, {"saturated": {"concurrency": 64}})                   # a regression: halve until it holds
    assert down["value"] == 16.0 and calls == [64, 32, 16, 16]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
