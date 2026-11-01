"""regimes.arrivals: the timing model fitted from a conversation log. A fake log with known sessions is read into
sittings by split, fitted, drawn from at steady state, and compared. Run: python tests/test_regimes_arrivals.py"""
from __future__ import annotations

import csv
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regimes import arrivals  # noqa: E402


def _log(path: Path, seed: int = 0) -> None:
    """Sessions starting Poisson at 0.05/s over two weeks: 1-3 turns 100 s apart, every fifth one resumed after an
    hour (a break), API rows in between that must be ignored."""
    rng, t, rows = random.Random(seed), 0.0, []
    for k in range(60000):
        t += rng.expovariate(0.05)
        turns = 1 + k % 3
        for j in range(turns):
            rows.append((t + 100 * j, f"s{k}", 3, "ChatGPT", 100, 50, 150, "Conversation log"))
        if k % 5 == 0:
            rows.append((t + 100 * turns + 3600, f"s{k}", 3, "ChatGPT", 100, 50, 150, "Conversation log"))
        rows.append((t + 1, "", 1, "GPT-4", 100, 50, 150, "API log"))
    rows.sort()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Timestamp", "Session ID", "Elapsed time", "Model", "Request tokens", "Response tokens",
                    "Total tokens", "Log Type"])
        w.writerows(rows)


def test_read_fit_schedule_compare():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.csv"
        _log(path)
        sittings = arrivals.read_sittings(path)
        n = len(sittings["seen"]) + len(sittings["heldout"])
        assert 60000 + 12000 - 5 < n <= 60000 + 12000, n               # one resumed sitting per fifth session
        assert 0.4 < len(sittings["seen"]) / n < 0.6                       # alternating weeks
        resumed = [s for s in sittings["seen"] if s.first_turn > 0]
        assert resumed and all(s.turns == 1 for s in resumed)
        assert {s.first_turn for s in resumed} == {1, 2, 3}
        fitted = arrivals.fit(sittings["seen"])
        assert fitted["sittings"] == len(sittings["seen"])
        assert sum(fitted["shapes"].values()) == fitted["sittings"]
        assert all(abs(g - 100.0) < 1e-9 for q in fitted["gap_s"].values() for g in q)
        assert fitted["horizon_s"] == 200.0

        model = arrivals.Model(fitted)
        sched = model.schedule(2.0, 60.0, "t")
        assert sched == model.schedule(2.0, 60.0, "t") and sched != model.schedule(2.0, 60.0, "u")
        assert all(0 <= x.at_s < 60.0 for x in sched) and sched == sorted(sched, key=lambda x: x.at_s)
        # steady state: turns of sittings begun before t=0 are in the first seconds, with their turn index
        early = [x for x in sched if x.at_s < 5.0]
        assert any(x.turn_index > 0 and not x.first for x in early), early[:5]
        assert all(x.first == (x.turn_index == 0) or x.turn_index > 0 for x in sched)
        # 2 sittings/s x mean turns per sitting, within sampling noise
        per_sitting = sum(s.turns for s in sittings["seen"]) / len(sittings["seen"])
        assert abs(len(sched) / (2.0 * 60.0 * per_sitting) - 1) < 0.25, len(sched)
        first = Counter(x.turn_index for x in sched if x.first)
        assert set(first) <= {0, 1, 2, 3}

        report = arrivals.compare(model, sittings["heldout"])
        assert report["windows"] > 0 and report["heldout_sittings"] == len(sittings["heldout"])
        med = report["per_window_median"]
        assert 0.7 < med["starts_cv"]["real"] < 1.3 and 0.7 < med["starts_cv"]["model"] < 1.3
        assert med["starts_ks_p"]["real"] > 0.01
        assert report["ks_real_vs_model"]["turns"] < 0.05 and report["ks_real_vs_model"]["gap_0"] < 0.01
        assert set(report["heldout_by_hour"]) <= {str(h) for h in range(24)}


def test_ks_two_sample_handles_ties():
    assert arrivals.ks_two_sample([1, 1, 2, 2], [1, 1, 2, 2]) == 0.0
    assert arrivals.ks_two_sample([1, 1, 1, 1], [2, 2, 2, 2]) == 1.0
    assert abs(arrivals.ks_two_sample([1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8]) - 0.5) < 1e-9


def test_quantile_draw_covers_the_range():
    model = arrivals.Model({"shapes": {"0:2": 1}, "gap_s": {"0": arrivals.quantiles([float(x) for x in range(1000)])},
                            "horizon_s": 0.0})
    rng = random.Random(1)
    draws = [model.draw_gap(0, rng) for _ in range(5000)]
    assert min(draws) >= 0.0 and max(draws) <= 999.0 and abs(sum(draws) / len(draws) - 499.5) < 15


if __name__ == "__main__":
    test_read_fit_schedule_compare()
    test_ks_two_sample_handles_ties()
    test_quantile_draw_covers_the_range()
    print("ok")
