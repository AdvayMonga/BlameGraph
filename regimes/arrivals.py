"""Request timing fitted from real conversation traffic (BurstGPT_3's conversation log), for the open-loop regimes.

The model: sittings start as a Poisson process at the load's rate; each sitting is a run of turns of one conversation
(it may resume one: `first_turn` > 0 means that many turns of history), with the gap to the next turn drawn from the
real gaps by transition index. A session's gap over BREAK_S ends the sitting (the user comes back later as a resumed
sitting). Load rises by raising the sitting rate: the data says busy hours have more sittings, not different ones
(the fit report records turns and gaps by hour of day). `schedule` starts with users already mid-conversation, so a
short run is at steady state from its first second.

  python -m regimes.arrivals fit [--cache DIR] [--out regimes/arrivals.json]     fit on the seen weeks
  python -m regimes.arrivals compare [--model F] [--cache DIR]                    the model against the held-out weeks

Both are measurements: the fitted file carries its source's hash and the comparison reports statistics, no verdict.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = Path(__file__).with_name("arrivals.json")
DEFAULT_CACHE = Path.home() / ".cache" / "inference-server" / "traces"
BURSTGPT_CSV = "burstgpt/BurstGPT_3.csv"
WEEK_S = 7 * 86400          # even trace weeks are `seen`, odd weeks `heldout` (data/corpus/build_corpus.py's rule)
BREAK_S = 1800.0            # a gap this long ends a sitting
INDEX_BUCKETS = (0, 1, 2, 3, 4, 5, 10, 20, 40)   # gaps are kept by transition index, pooled within these buckets
HORIZON_DRAWS = 100000      # sittings drawn from the fit to find how long its own sittings run
QUANTILES = 201
WINDOW_S = 1200.0           # the comparison's windows
MIN_STARTS = 30             # a window with fewer sitting starts is not compared


@dataclass
class Sitting:
    start_s: float
    first_turn: int
    gaps: list[float]       # gap to each following turn

    @property
    def turns(self) -> int:
        return len(self.gaps) + 1


@dataclass
class Turn:
    sitting: int
    turn_index: int
    at_s: float
    first: bool             # the sitting's first turn (a sitting under way at t=0 has none in the run)


# ------------------------------------------------------------------ reading

def read_sittings(path: Path) -> dict[str, list[Sitting]]:
    """The conversation log's sessions cut into sittings at BREAK_S, by split of the sitting's first request."""
    sessions: dict[str, list[float]] = defaultdict(list)
    t0 = None
    with open(path) as f:
        rows = csv.reader(f)
        head = next(rows)
        col = {name: k for k, name in enumerate(head)}
        for row in rows:
            t = float(row[col["Timestamp"]])
            t0 = t if t0 is None else t0
            if row[col["Log Type"]] == "Conversation log":
                sessions[row[col["Session ID"]]].append(t)
    out: dict[str, list[Sitting]] = {"seen": [], "heldout": []}
    for ts in sessions.values():
        ts.sort()
        first, gaps = 0, []
        for k, (a, b) in enumerate(zip(ts, ts[1:])):
            if b - a > BREAK_S:
                out[_split(ts[first], t0)].append(Sitting(ts[first], first, gaps))
                first, gaps = k + 1, []
            else:
                gaps.append(b - a)
        out[_split(ts[first], t0)].append(Sitting(ts[first], first, gaps))
    for v in out.values():
        v.sort(key=lambda s: s.start_s)
    return out


def _split(t: float, t0: float) -> str:
    return "seen" if int((t - t0) // WEEK_S) % 2 == 0 else "heldout"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ fitting

def quantiles(xs: list[float], n: int = QUANTILES) -> list[float]:
    xs = sorted(xs)
    return [xs[round(k * (len(xs) - 1) / (n - 1))] for k in range(n)]


def _index(k: int) -> str:
    return str(max(b for b in INDEX_BUCKETS if b <= k))


def fit(sittings: list[Sitting]) -> dict:
    """Counts of (first_turn, turns) and gap quantiles by transition index; `horizon_s` is the p99.99 span of the
    fitted model's own sittings (at least the real p99.9), so a schedule's pre-roll covers what the model draws."""
    gaps: dict[str, list[float]] = defaultdict(list)
    for s in sittings:
        for k, g in enumerate(s.gaps):
            gaps[_index(s.first_turn + k)].append(g)
    shapes = Counter(f"{s.first_turn}:{s.turns}" for s in sittings)
    spans = sorted(sum(s.gaps) for s in sittings)
    out = {"sittings": len(sittings), "shapes": dict(sorted(shapes.items(), key=lambda kv: -kv[1])),
           "gap_s": {k: quantiles(v) for k, v in sorted(gaps.items())},
           "span_s_max": spans[-1], "horizon_s": spans[round(0.999 * (len(spans) - 1))]}
    model, rng = Model(out), random.Random(0)
    drawn = sorted(sum(model.draw_sitting(0.0, rng).gaps) for _ in range(HORIZON_DRAWS))  # reach 0: never None
    out["horizon_s"] = max(out["horizon_s"], drawn[round(0.9999 * (len(drawn) - 1))])
    return out


def by_hour(sittings: list[Sitting]) -> dict:
    """Turns per sitting and median gap by hour of day: whether busy hours change what a user does."""
    starts, turns, gaps = Counter(), defaultdict(list), defaultdict(list)
    for s in sittings:
        h = int(s.start_s // 3600) % 24
        starts[h] += 1
        turns[h].append(s.turns)
        gaps[h] += s.gaps
    return {str(h): {"starts": starts[h], "turns_mean": round(statistics.mean(turns[h]), 2),
                     "gap_s_median": statistics.median(gaps[h]) if gaps[h] else None} for h in sorted(starts)}


# ------------------------------------------------------------------ the model

class Model:
    def __init__(self, fitted: dict):
        self.fitted = fitted
        self.shapes = [tuple(int(x) for x in k.split(":")) for k in fitted["shapes"]]
        self.weights = list(fitted["shapes"].values())
        self.gap_q = dict(fitted["gap_s"])
        self.horizon_s = float(fitted["horizon_s"])     # sittings longer than this (0.01%) under way at t=0 are missed
        self.max_span = {sh: sum(self.gap_q[_index(sh[0] + k)][-1] for k in range(sh[1] - 1)) for sh in self.shapes}
        self.turns_mean = sum(t * c for (_, t), c in zip(self.shapes, self.weights)) / sum(self.weights)

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL) -> "Model":
        return cls(json.loads(Path(path).read_text())["fit"])

    def draw_gap(self, index: int, rng: random.Random) -> float:
        q = self.gap_q[_index(index)]
        u = rng.random() * (len(q) - 1)
        i, frac = int(u), u - int(u)
        return q[i] + (q[min(i + 1, len(q) - 1)] - q[i]) * frac

    def draw_sitting(self, start_s: float, rng: random.Random, reach_s: float = 0.0) -> Sitting | None:
        """None when the drawn shape cannot last `reach_s` (the pre-roll skips what cannot reach t=0)."""
        shape = rng.choices(self.shapes, self.weights)[0]
        if self.max_span[shape] < reach_s:
            return None
        first, turns = shape
        return Sitting(start_s, first, [self.draw_gap(first + k, rng) for k in range(turns - 1)])

    def schedule(self, rate: float, duration_s: float, seed) -> list[Turn]:
        """Turns in [0, duration_s) of sittings starting as Poisson(rate) from -horizon on, so users already
        mid-conversation at t=0 are there (steady state); sorted by time."""
        rng, t, out, k = random.Random(seed), -self.horizon_s, [], 0
        while True:
            t += rng.expovariate(rate)
            if t >= duration_s:
                return sorted(out, key=lambda x: x.at_s)
            s, at = self.draw_sitting(t, rng, -t), t
            if s is None:
                k += 1
                continue
            for j in range(s.turns):
                if j:
                    at += s.gaps[j - 1]
                if at >= duration_s:
                    break
                if at >= 0:
                    out.append(Turn(k, s.first_turn + j, at, j == 0))
            k += 1


# ------------------------------------------------------------------ comparison

def _ks_exponential(xs: list[float]) -> float:
    """KS p-value of inter-arrivals against exponential at their own mean (asymptotic)."""
    xs, m = sorted(xs), statistics.mean(xs)
    n = len(xs)
    d = max(max((i + 1) / n - (1 - math.exp(-x / m)), (1 - math.exp(-x / m)) - i / n) for i, x in enumerate(xs))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    return min(1.0, 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 100)))


def ks_two_sample(a: list[float], b: list[float]) -> float:
    """The KS statistic (max CDF distance) between two samples."""
    a, b = sorted(a), sorted(b)
    i = j = 0
    d = 0.0
    while i < len(a) and j < len(b):
        v = min(a[i], b[j])
        while i < len(a) and a[i] == v:
            i += 1
        while j < len(b) and b[j] == v:
            j += 1
        d = max(d, abs(i / len(a) - j / len(b)))
    return d


def _windows(sittings: list[Sitting], t0: float) -> dict[int, list[float]]:
    """Sitting start times by window, for the windows with at least MIN_STARTS."""
    wins: dict[int, list[float]] = defaultdict(list)
    for s in sittings:
        wins[int((s.start_s - t0) // WINDOW_S)].append(s.start_s)
    return {k: v for k, v in wins.items() if len(v) >= MIN_STARTS}


def _dispersion(times: list[float], lo: float, bin_s: float) -> float:
    counts = Counter(int((t - lo) // bin_s) for t in times)
    xs = [counts.get(i, 0) for i in range(int(WINDOW_S // bin_s))]
    m = statistics.mean(xs)
    return statistics.pvariance(xs) / m if m else 0.0


def _window_stats(starts: list[float], requests: list[float], lo: float) -> dict:
    ia = [b - a for a, b in zip(starts, starts[1:])]
    return {"starts_cv": statistics.pstdev(ia) / statistics.mean(ia), "starts_ks_p": _ks_exponential(ia),
            **{f"requests_dispersion_{b}s": _dispersion(requests, lo, b) for b in (1, 10, 60)}}


def compare(model: Model, heldout: list[Sitting], seed: int = 0) -> dict:
    """The model's traffic beside the held-out weeks': per busy window (>= MIN_STARTS sitting starts in WINDOW_S),
    the sitting starts' inter-arrival CV and KS p against exponential and the request counts' variance/mean at 1,
    10 and 60 s, each as the median over windows, real and generated at the window's own rate; and the KS statistic
    between real and model for turns per sitting, first turn and the gaps by index."""
    t0 = heldout[0].start_s
    times = sorted(s.start_s + sum(s.gaps[:j]) for s in heldout for j in range(s.turns))
    real, gen = defaultdict(list), defaultdict(list)
    for k, starts in _windows(heldout, t0).items():
        lo = t0 + k * WINDOW_S
        requests = times[bisect.bisect_left(times, lo):bisect.bisect_left(times, lo + WINDOW_S)]
        for key, v in _window_stats(sorted(starts), requests, lo).items():
            real[key].append(v)
        turns = model.schedule(len(starts) / WINDOW_S, WINDOW_S, f"{seed}:{k}")
        for key, v in _window_stats([t.at_s for t in turns if t.first], [t.at_s for t in turns], 0.0).items():
            gen[key].append(v)
    fitted = fit(heldout)
    rng = random.Random(seed)
    sample = [model.draw_sitting(0.0, rng) for _ in range(len(heldout))]
    ks = {"turns": ks_two_sample([s.turns for s in heldout], [s.turns for s in sample]),
          "first_turn": ks_two_sample([s.first_turn for s in heldout], [s.first_turn for s in sample])}
    for idx, q in fitted["gap_s"].items():
        if idx in model.gap_q:
            ks[f"gap_{idx}"] = ks_two_sample(q, model.gap_q[idx])
    return {"windows": len(real.get("starts_cv", [])), "heldout_sittings": len(heldout),
            "per_window_median": {key: {"real": statistics.median(real[key]), "model": statistics.median(gen[key])}
                                  for key in real},
            "ks_real_vs_model": ks, "heldout_by_hour": by_hour(heldout)}


# ------------------------------------------------------------------ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    f.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    c = sub.add_parser("compare")
    c.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    c.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    c.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    path = a.cache / BURSTGPT_CSV
    sittings = read_sittings(path)
    if a.cmd == "fit":
        seen = sittings["seen"]
        out = {"source": {"file": BURSTGPT_CSV, "sha256": _sha256(path), "split": "seen (even trace weeks)",
                          "break_s": BREAK_S, "sittings": {k: len(v) for k, v in sittings.items()}},
               "fit": fit(seen), "seen_by_hour": by_hour(seen)}
        a.out.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {a.out}: {len(seen)} sittings, {len(out['fit']['shapes'])} shapes, "
              f"horizon {out['fit']['horizon_s']:.0f} s, longest sitting {out['fit']['span_s_max']:.0f} s")
    else:
        print(json.dumps(compare(Model.load(a.model), sittings["heldout"], a.seed), indent=1))


if __name__ == "__main__":
    main()
