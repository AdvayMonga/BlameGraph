"""Synthetic traffic, generated fresh for each run from the public corpus and then gone: as much as a regime needs,
with the seen split's shapes and none of its prompts. Only the seen split is read, so the real held-out split is
never touched; nothing is written, and the seed (in the run's config) regenerates the same traffic.

Per class, whole sessions are drawn in shuffled passes over every real session (each shape used once per pass, so the
tails survive) with their structure (turn count, think time between turns, each request's message roles and
per-message lengths, max_tokens, and the `regime` that judges a mixed-class request). The gaps between session
starts are the real sequence, repeated as often as needed, each scaled by a seeded jitter, so a burst stays a burst
where it was (what `spike` and `cold_start` are defined by). Each message's text is real text of the same role from
the seen split, cut at sentence ends and recombined to the original length: it reads and tokenizes like the real
traffic but repeats no prompt.

  regimes: `python -m regimes run REGIME --synthetic` (Ctx.synthetic): every pool is generated, not read
  python -m regimes.synth compare [--corpus DIR] [--seed N] [--requests N]   the generated shapes beside the real ones
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from lab.corpus import CORPUS_DIR, TraceRequest, load_manifest, load_trace

JITTER = 0.1                # each gap scaled by exp(N(0, JITTER)): the same pattern, never the same instants
MIN_REQUESTS = 2000         # a generated pool holds at least this many, so a full-tier run rarely cycles it
_CUT = re.compile(r"(?<=[.!?\n])\s+")


def _sessions(reqs: list[TraceRequest]) -> list[list[TraceRequest]]:
    by: dict[str, list[TraceRequest]] = defaultdict(list)
    for r in reqs:
        by[r.session_id].append(r)
    out = [sorted(v, key=lambda r: r.turn_index) for v in by.values()]
    return sorted(out, key=lambda s: s[0].arrival_s)


def _messages(r: TraceRequest) -> list[dict]:
    return r.messages or [{"role": "user", "content": r.prompt}]


def pieces(reqs: list[TraceRequest]) -> dict[str, list[str]]:
    """Real text by role, cut at sentence ends: what generated messages are built from."""
    out: dict[str, list[str]] = defaultdict(list)
    for r in reqs:
        for m in _messages(r):
            out[m["role"]] += [p for p in _CUT.split(m["content"]) if p.strip()]
    return dict(out)


def _text(n: int, pool: list[str], rng: random.Random) -> str:
    """`n` characters of pieces from `pool`."""
    parts, size = [], 0
    while size < n:
        p = rng.choice(pool)
        parts.append(p)
        size += len(p) + 1
    return " ".join(parts)[:max(n, 1)]


def _gaps(sessions: list[list[TraceRequest]]) -> list[float]:
    starts = [s[0].arrival_s for s in sessions]
    return [b - a for a, b in zip(starts, starts[1:])] or [1.0]


def _timing(gaps: list[float], n: int, rng: random.Random) -> list[float]:
    """`n` gaps: the real sequence repeated, each jittered."""
    return [gaps[k % len(gaps)] * math.exp(rng.gauss(0.0, JITTER)) for k in range(n)]


def _draw(sessions: list, n: int, rng: random.Random) -> list:
    """`n` sessions in shuffled passes over all of them."""
    out: list = []
    while len(out) < n:
        out += rng.sample(sessions, len(sessions))
    return out[:n]


def synthesize(reqs: list[TraceRequest], n_sessions: int, seed, tag: str) -> list[TraceRequest]:
    """`n_sessions` sessions shaped like `reqs`'. `build_prompt_tokens` is the real request's count scaled by the
    character ratio (generated text tokenizes like the text it was cut from)."""
    rng = random.Random(seed)
    real = _sessions(reqs)
    pool = pieces(reqs)
    gaps = _timing(_gaps(real), max(n_sessions - 1, 0), rng)
    out, t = [], 0.0
    for k, src in enumerate(_draw(real, n_sessions, rng)):
        if k:
            t += gaps[k - 1]
        for r in src:
            msgs = [{"role": m["role"], "content": _text(len(m["content"]), pool.get(m["role"]) or pool["user"], rng)}
                    for m in _messages(r)]
            tokens = None
            if r.build_prompt_tokens:
                real_chars = sum(len(m["content"]) for m in _messages(r)) or 1
                tokens = round(r.build_prompt_tokens * sum(len(m["content"]) for m in msgs) / real_chars)
            out.append(TraceRequest(arrival_s=round(t + r.arrival_s - src[0].arrival_s, 6), session_id=f"{tag}-{k}",
                                    turn_index=r.turn_index, prompt="\n\n".join(m["content"] for m in msgs),
                                    max_tokens=r.max_tokens, sampling=dict(r.sampling),
                                    messages=msgs if r.messages else None, build_prompt_tokens=tokens, regime=r.regime))
    return sorted(out, key=lambda r: (r.arrival_s, r.session_id, r.turn_index))


def trace(cls: str, seed: int, split: str = "seen", corpus_dir: Path = CORPUS_DIR,
          min_requests: int = MIN_REQUESTS) -> list[TraceRequest]:
    """A fresh trace for one class, fitted to its seen split: whole passes over its sessions until it holds at least
    `min_requests`. `split` only names the seed stream (a held-out run draws different traffic from the same shapes)."""
    _, reqs = load_trace(cls, "seen", corpus_dir)
    passes = max(1, math.ceil(min_requests / max(len(reqs), 1)))
    return synthesize(reqs, passes * len(_sessions(reqs)), f"{cls}:{split}:{seed}", f"syn-{cls}-{split}-{seed}")


def records(corpus_dir: Path, cls: str | None, seed: int, split: str = "seen",
            min_requests: int = MIN_REQUESTS) -> list[dict]:
    """`trace` as the record dicts `regimes.workload.corpus` returns (one class, or every class for None)."""
    classes = [cls] if cls else list(load_manifest(Path(corpus_dir)).classes)
    out = []
    for c in classes:
        for k, r in enumerate(trace(c, seed, split, Path(corpus_dir), min_requests)):
            d = r.to_dict()
            d["messages"] = r.messages
            d["build_prompt_tokens"] = r.build_prompt_tokens
            d["id"] = f"syn-{c}-{split}-{seed}-{k}"
            out.append(d)
    return out


# -- comparison -----------------------------------------------------------------------------
def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def ks(a: list[float], b: list[float]) -> float | None:
    """Two-sample Kolmogorov-Smirnov distance: the largest gap between the two empirical CDFs."""
    if not a or not b:
        return None
    a, b = sorted(a), sorted(b)
    i = j = 0
    d = 0.0
    while i < len(a) and j < len(b):
        x = min(a[i], b[j])
        while i < len(a) and a[i] <= x:
            i += 1
        while j < len(b) and b[j] <= x:
            j += 1
        d = max(d, abs(i / len(a) - j / len(b)))
    return d


def shape(reqs: list[TraceRequest]) -> dict:
    """The facts a regime's load depends on, for one trace."""
    ses = _sessions(reqs)
    gaps = _gaps(ses)
    mean = sum(gaps) / len(gaps)
    sd = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
    per10: dict[int, int] = defaultdict(int)
    for r in reqs:
        per10[int(r.arrival_s // 10)] += 1
    counts = [per10.get(k, 0) for k in range(int(max(r.arrival_s for r in reqs) // 10) + 1)]
    cm = sum(counts) / len(counts)
    csd = math.sqrt(sum((c - cm) ** 2 for c in counts) / len(counts))
    p = [r.build_prompt_tokens for r in reqs if r.build_prompt_tokens is not None]
    m = [r.max_tokens for r in reqs]
    return {"requests": len(reqs), "sessions": len(ses), "turns_mean": len(reqs) / max(len(ses), 1),
            "messages_mean": sum(len(_messages(r)) for r in reqs) / max(len(reqs), 1),
            "prompt_tokens": {"p50": _q(p, .5), "p90": _q(p, .9), "p99": _q(p, .99)},
            "max_tokens": {"p50": _q(m, .5), "p90": _q(m, .9)},
            "gap_s": {"mean": mean, "cv": sd / mean if mean else None},
            "per_10s": {"mean": cm, "cv": csd / cm if cm else None},     # burstiness whatever the clock's resolution
            "_p": p, "_gaps": gaps}


def compare(corpus_dir: Path = CORPUS_DIR, seed: int = 0, min_requests: int = 0) -> dict:
    """Per class: the real seen shape beside a generated trace's, with KS distances on prompt tokens and gaps
    (`min_requests` 0 = one pass, the real trace's own size)."""
    out = {}
    for name in load_manifest(Path(corpus_dir)).classes:
        a = shape(load_trace(name, "seen", Path(corpus_dir))[1])
        b = shape(trace(name, seed, "seen", Path(corpus_dir), min_requests))
        out[name] = {"real": {k: v for k, v in a.items() if not k.startswith("_")},
                     "synthetic": {k: v for k, v in b.items() if not k.startswith("_")},
                     "ks_prompt_tokens": ks(a["_p"], b["_p"]), "ks_gap_s": ks(a["_gaps"], b["_gaps"])}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m regimes.synth", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--requests", type=int, default=0, help="at least this many per class (0: one pass)")
    a = ap.parse_args(argv)
    print(json.dumps(compare(a.corpus, a.seed, a.requests), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
