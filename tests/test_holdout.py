"""Thresholdout held-out budget and sealed split. Run: python tests/test_holdout.py"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.holdout import HoldoutGuard, Refused, SealedSplit  # noqa: E402


def _guard(d, budget=10, threshold=0.05, sigma=0.001, seed=0, name="state.json"):
    return HoldoutGuard(Path(d) / name, budget=budget, threshold=threshold, sigma=sigma, seed=seed)


def test_small_gap_reports_seen_and_spends_nothing():
    with tempfile.TemporaryDirectory() as d:
        g = _guard(d)
        for _ in range(50):
            out = g.query(0.80, 0.81)
            assert out == {"value": 0.80, "used_heldout": False, "budget_left": 10}, out


def test_large_gap_spends_budget_and_reports_noisy_heldout():
    with tempfile.TemporaryDirectory() as d:
        g = _guard(d)
        out = g.query(0.90, 0.60)
        assert out["used_heldout"] and out["budget_left"] == 9 and abs(out["value"] - 0.60) < 0.05, out
        assert out["value"] != 0.60  # noise added


def test_exhaustion_refuses():
    with tempfile.TemporaryDirectory() as d:
        g = _guard(d, budget=2)
        g.query(0.9, 0.6); g.query(0.9, 0.6)
        for seen, heldout in [(0.9, 0.6), (0.8, 0.8)]:  # refuses even queries that would not spend
            try:
                g.query(seen, heldout)
                raise AssertionError("expected Refused")
            except Refused as e:
                assert "exhausted" in str(e)


def test_state_survives_reload():
    with tempfile.TemporaryDirectory() as d:
        g = _guard(d, budget=3)
        g.query(0.8, 0.8); g.query(0.9, 0.6)
        g2 = _guard(d, budget=3)
        s = g2.state()
        assert s["budget_left"] == 2 and s["n_queries"] == 2 and len(s["log"]) == 2, s
        assert all({"t", "i", "value", "used_heldout", "budget_left"} == set(e) for e in s["log"])  # inputs omitted
        assert g2.query(0.9, 0.6)["budget_left"] == 1
        assert not list(Path(d).glob("*.tmp"))
        try:
            _guard(d, budget=100)  # a reload cannot reset the budget
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_deterministic_given_seed():
    queries = [(0.8, 0.8), (0.9, 0.6), (0.7, 0.72), (0.95, 0.5), (0.6, 0.62)]
    with tempfile.TemporaryDirectory() as d:
        a, b, c = _guard(d, name="a.json"), _guard(d, name="b.json"), _guard(d, seed=1, name="c.json")
        ra, rb, rc = ([g.query(*q) for q in queries] for g in (a, b, c))
        assert ra == rb
        assert [r["value"] for r in ra] != [r["value"] for r in rc]


def _hill_climb(seed, guard=None, n=500, d=100, steps=1500):
    """Flip one sign of a linear model at a time, keep it if the reported score rises; labels are pure noise.
    Returns the final model's true held-out accuracy minus chance (= how far the held-out set was overfit)."""
    r = random.Random(seed)

    def split():
        return [[r.gauss(0, 1) for _ in range(d)] for _ in range(n)], [r.choice((-1, 1)) for _ in range(n)]

    def acc(scores, y):
        return sum((s > 0) == (t > 0) for s, t in zip(scores, y)) / n

    (xs, ys), (xh, yh) = split(), split()
    w = [1] * d
    ss, sh = [sum(row) for row in xs], [sum(row) for row in xh]
    best = -1.0
    for _ in range(steps):
        j = r.randrange(d)
        ns = [s - 2 * w[j] * row[j] for s, row in zip(ss, xs)]
        nh = [s - 2 * w[j] * row[j] for s, row in zip(sh, xh)]
        try:
            reported = guard.query(acc(ns, ys), acc(nh, yh))["value"] if guard else acc(nh, yh)
        except Refused:
            break
        if reported > best:
            best, ss, sh, w[j] = reported, ns, nh, -w[j]
    return acc(sh, yh) - 0.5


def test_adaptive_attacker_gains_far_less_with_guard():
    direct, guarded = [], []
    with tempfile.TemporaryDirectory() as d:
        for seed in range(8):
            direct.append(_hill_climb(seed))
            g = _guard(d, budget=20, threshold=0.08, sigma=0.005, seed=seed, name=f"{seed}.json")
            guarded.append(_hill_climb(seed, g))
    mean_direct, mean_guarded = sum(direct) / 8, sum(guarded) / 8
    assert mean_direct > 0.06, direct  # querying held-out directly overfits it
    assert mean_guarded < mean_direct / 2, (direct, guarded)


def test_sealed_split_detects_changes_and_unseals_once():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a.jsonl").write_text("a\n"); (d / "b.jsonl").write_text("b\n")
        split = SealedSplit.seal(d / "sealed.json", [d / "a.jsonl", d / "b.jsonl"], corpus_version="v1")
        assert split.verify() == []
        (d / "b.jsonl").write_text("tampered\n")
        assert SealedSplit(d / "sealed.json").verify() == ["b.jsonl"]
        try:
            split.unseal("final eval", "referee")
            raise AssertionError("expected Refused for changed files")
        except Refused as e:
            assert "changed" in str(e)
        (d / "b.jsonl").write_text("b\n")
        assert SealedSplit(d / "sealed.json").unseal("final eval", "referee") == [d / "a.jsonl", d / "b.jsonl"]
        try:
            SealedSplit(d / "sealed.json").unseal("again", "agent")
            raise AssertionError("expected Refused for second unseal")
        except Refused as e:
            assert "already unsealed" in str(e)
        log = [json.loads(ln) for ln in split.log_path.read_text().splitlines()]
        assert [(e["corpus_version"], e["who"]) for e in log] == [("v1", "referee")]
        (d / "a.jsonl").unlink()
        SealedSplit.seal(d / "sealed.json", [d / "b.jsonl"], corpus_version="v2")
        assert SealedSplit(d / "sealed.json").unseal("v2 final", "referee") == [d / "b.jsonl"]  # new version unseals


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
