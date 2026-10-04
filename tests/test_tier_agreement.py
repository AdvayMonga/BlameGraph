"""Short-tier validity on synthetic changes. Run: python tests/test_tiers.py"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validity.tier_agreement import tier_agreement  # noqa: E402


def _changes(short, full, **bands):
    return [{"name": f"c{i}", "short": s, "full": f, **bands} for i, (s, f) in enumerate(zip(short, full))]


def test_perfect_agreement_usable():
    full = [float(i) for i in range(10)]
    r = tier_agreement(_changes([2 * x + 1 for x in full], full))
    assert r["spearman_rho"] == 1.0 and r["kendall_tau_b"] == 1.0 and r["verdict"]["usable"]
    assert r["rho_ci95"][0] == 1.0 and not r["disagreements"]


def test_reversed_unusable():
    full = [float(i) for i in range(10)]
    r = tier_agreement(_changes([-x for x in full], full))
    assert r["spearman_rho"] == -1.0 and not r["verdict"]["usable"] and len(r["disagreements"]) == 45


def test_noisy_but_faithful_usable():
    rng = random.Random(1); full = [float(i) for i in range(20)]
    r = tier_agreement(_changes([x + rng.gauss(0, 1.0) for x in full], full))
    assert r["spearman_rho"] > 0.9 and r["rho_ci95"][0] >= 0.5 and r["verdict"]["usable"], r


def test_pure_noise_unusable():
    rng = random.Random(2); full = [float(i) for i in range(20)]
    r = tier_agreement(_changes([rng.gauss(0, 1) for _ in full], full))
    assert not r["verdict"]["usable"] and r["verdict"]["reasons"], r


def test_ties_hand_computed():
    # ranks short [1, 2.5, 2.5, 4] vs full [1, 2, 3, 4]: rho = 4.5 / sqrt(4.5 * 5); tau-b = 5 / sqrt(5 * 6)
    r = tier_agreement(_changes([1, 2, 2, 3], [1, 2, 3, 4]), min_n=4)
    assert abs(r["spearman_rho"] - 4.5 / (4.5 * 5) ** 0.5) < 1e-12
    assert abs(r["kendall_tau_b"] - 5 / 30 ** 0.5) < 1e-12
    assert [(d["a"], d["b"]) for d in r["disagreements"]] == [("c1", "c2")]   # short tie fails to rank them


def test_rho_hand_computed():
    # d = [0, 1, -1, 1, -1], sum d^2 = 4: rho = 1 - 6*4 / (5*24) = 0.8; 2 discordant of 10 pairs: tau = 0.6
    r = tier_agreement(_changes([1, 3, 2, 5, 4], [1, 2, 3, 4, 5]))
    assert abs(r["spearman_rho"] - 0.8) < 1e-12 and abs(r["kendall_tau_b"] - 0.6) < 1e-12


def test_disagreements_inside_band_not_reported():
    full = [0.0, 0.05, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    short = [0.05, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 5.0]      # swaps c0/c1 (inside band) and c6/c7 (outside)
    r = tier_agreement(_changes(short, full, full_band=0.1))
    assert [(d["a"], d["b"]) for d in r["disagreements"]] == [("c6", "c7")]


def test_small_n_refused():
    r = tier_agreement(_changes([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
    assert r["spearman_rho"] == 1.0 and not r["verdict"]["usable"]
    assert r["verdict"]["reasons"] == ["n=5 < 8 changes"]


def test_constant_tier_refused():
    r = tier_agreement(_changes([1.0] * 8, list(range(8))))
    assert not r["verdict"]["usable"] and "undefined" in r["verdict"]["reasons"][0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
