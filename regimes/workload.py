"""Workloads: the corpus (`corpus/`, real BurstGPT timing + WildChat text; built by corpus/build_corpus.py) and seeded
schedules built from it. Trace format (corpus/README.md): one request per line with `arrival_s`, `session_id`,
`turn_index`, `prompt`, `max_tokens`, `build_prompt_tokens`, and `messages` on multi-turn requests.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .client import Request

WORDS = ("river stone lantern harbor meadow copper signal winter garden orbit ledger canyon thread pilot "
         "marble violet engine summit falcon quiet number amber circuit island paper").split()


def corpus(root: str | Path, cls: str, split: str = "seen") -> list[dict]:
    """Records of one (class, split) through the verified loader (every trace hash is checked against the
    manifest; a changed trace refuses), each with a stable `id` (`<class>-<split>-<line>`)."""
    from lab.corpus import load_trace
    _, reqs = load_trace(cls, split, Path(root))
    out = []
    for k, r in enumerate(reqs):
        d = r.to_dict()
        d["messages"] = r.messages
        d["build_prompt_tokens"] = r.build_prompt_tokens
        d["id"] = f"{cls}-{split}-{k}"
        out.append(d)
    return out


def workload_file(path: str | Path) -> list[dict]:
    """Records of a `python -m workloads` file (frontier benchmark data): same record shape, no classes or splits."""
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def corpus_classes(root: str | Path) -> list[str]:
    from lab.corpus import load_manifest
    return list(load_manifest(Path(root)).classes)


def corpus_version(root: str | Path | None) -> str | None:
    if root is None:
        return None
    from lab.corpus import load_manifest
    return load_manifest(Path(root)).corpus_version


def messages(rec: dict) -> list[dict]:
    return rec.get("messages") or [{"role": "user", "content": rec["prompt"]}]


def to_request(rec: dict, at_s: float = 0.0, max_tokens: int | None = None, suffix: str = "", system: str | None = None,
               rid: str | None = None) -> Request:
    msgs = [dict(m) for m in messages(rec)]
    if suffix:
        msgs[-1]["content"] = msgs[-1]["content"] + suffix
    if system:
        msgs = [{"role": "system", "content": system}] + msgs
    return Request(rid or rec["id"], msgs, max_tokens or rec.get("max_tokens") or 2048, at_s,
                   rec.get("session_id"), rec.get("turn_index"))


def synthetic(n: int, words: int = 60, seed: int = 0, prefix: str = "syn") -> list[dict]:
    """Trace-shaped records of random words, for tests or when no corpus is given."""
    rng = random.Random(seed)
    return [{"id": f"{prefix}-{k}", "prompt": " ".join(rng.choice(WORDS) for _ in range(words)), "max_tokens": 256,
             "arrival_s": k * 0.05, "session_id": f"{prefix}-s{k}", "turn_index": 0, "build_prompt_tokens": words}
            for k in range(n)]


def shared_system_prompt(words: int = 1200, seed: int = 0) -> str:
    rng = random.Random(seed)
    return "You are a careful assistant. Reference notes follow.\n" + " ".join(rng.choice(WORDS) for _ in range(words))


def poisson(pool: list[dict], rate: float, duration_s: float, seed: int, **req_kw) -> list[Request]:
    """Open-loop arrivals at `rate` req/s for `duration_s`, requests drawn with replacement from `pool`."""
    rng, t, out = random.Random(seed), 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t >= duration_s:
            return out
        rec = rng.choice(pool)
        out.append(to_request(rec, t, rid=f"{rec['id']}#{len(out)}", **req_kw))


def replay(trace: list[dict], speed: float = 1.0, window: tuple[float, float] | None = None) -> list[Request]:
    """The trace at its own arrival times, compressed by `speed` (2.0 = twice the rate, same burst shape),
    optionally cropped to `window` (seconds of original trace time)."""
    recs = [r for r in trace if window is None or window[0] <= r["arrival_s"] < window[1]]
    t0 = min((r["arrival_s"] for r in recs), default=0.0)
    return [to_request(r, (r["arrival_s"] - t0) / speed) for r in recs]


def peak_window(trace: list[dict], width_s: float = 180.0) -> tuple[float, float]:
    """The `width_s` window centered on the busiest minute (short-tier crop of a burst trace)."""
    ts = sorted(r["arrival_s"] for r in trace)
    best, at, j = -1, ts[0] if ts else 0.0, 0
    for i, t in enumerate(ts):
        while ts[j] < t - 60:
            j += 1
        if i - j + 1 > best:
            best, at = i - j + 1, t - 30
    return at - width_s / 2, at + width_s / 2


def conversations(pool: list[dict], max_turns: int = 6) -> list[list[dict]]:
    """Each record with a stored history [user, assistant, ..., user] becomes a conversation of its first
    `max_turns` user turns; turn j sends the history up to the j-th user message."""
    convs = []
    for r in pool:
        msgs = r.get("messages") or []
        users = [i for i, m in enumerate(msgs) if m.get("role") == "user"][:max_turns]
        if len(users) >= 2:
            convs.append([{**r, "id": f"{r['id']}.t{j}", "messages": msgs[:i + 1], "turn_index": j}
                          for j, i in enumerate(users)])
    return convs


def sessions(pool: list[dict], rate: float, duration_s: float, seed: int, think_s: float = 3.0,
             system: str | None = None, max_turns: int = 6) -> list[Request]:
    """Multi-turn conversations arriving at `rate` sessions/s; turn k of a session follows turn k-1 by `think_s`.
    Each turn resends the conversation so far, and every request carries the same `system` prompt, so prefixes
    are shared within and across sessions."""
    convs = conversations(pool, max_turns)
    if not convs:
        raise ValueError("no multi-turn conversations in the pool")
    rng, t, out = random.Random(seed), 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t >= duration_s:
            return out
        conv, sid = rng.choice(convs), len(out)
        for k, rec in enumerate(conv):
            req = to_request(rec, t + k * think_s, system=system, rid=f"{rec['id']}#{sid}")
            req.session_id = f"{rec.get('session_id')}#{sid}"
            out.append(req)
