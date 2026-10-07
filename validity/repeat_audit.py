"""Repeated-request audit (adapted from MLPerf TEST04): a repeated request must not decode faster than a fresh one.
Prefix caching legitimately cuts a repeat's TTFT, so only decode-side timing is judged: TPOT = (E2E - TTFT)/(n - 1),
streamed and timed client-side. Stdlib only.

  python -m validity repeat --url http://127.0.0.1:8000 [--model M] [--n 20] [--api chat|completions]
"""
from __future__ import annotations

import json
import statistics
import time
import urllib.request

PROMPTS = [f"Write a short paragraph about {t}." for t in (
    "rivers", "volcanoes", "bees", "the moon", "glaciers", "lighthouses", "coral reefs", "deserts", "owls", "tides",
    "comets", "bridges", "forests", "salt", "clocks", "bread", "wind", "maps", "copper", "snow")]


def stream_timed(base_url: str, model: str, prompt: str, api: str = "chat", max_tokens: int = 128,
                 chat_kwargs: dict | None = None) -> dict:
    """One streamed greedy request -> {ttft_s, tpot_s, e2e_s, n_tokens}; a token is a chunk with non-empty text."""
    chat = api == "chat"
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
    body.update({"messages": [{"role": "user", "content": prompt}]} if chat else {"prompt": prompt})
    if chat and chat_kwargs:
        body["chat_template_kwargs"] = dict(chat_kwargs)
    req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/{'chat/completions' if chat else 'completions'}",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0, ttft, n = time.perf_counter(), None, 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            ch = (json.loads(line[6:]).get("choices") or [{}])[0]
            text = (ch.get("delta") or {}).get("content") if chat else ch.get("text")
            if text:
                n += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    e2e = time.perf_counter() - t0
    return {"ttft_s": ttft, "tpot_s": (e2e - ttft) / (n - 1) if n > 1 else None, "e2e_s": e2e, "n_tokens": n}


def _decode(c: dict) -> float:
    return c["e2e_s"] - (c["ttft_s"] or c["e2e_s"])


def repeat_audit(base_url: str, model: str, requests: list[str], repeats: int = 3, threshold: float = 0.10,
                 api: str = "chat", max_tokens: int = 128, chat_kwargs: dict | None = None) -> dict:
    """Send each prompt once then `repeats` more times; fail if repeats decode > threshold + noise faster, or instantly."""
    rows, tpot_r, e2e_r, ttft_r, firsts, instant = [], [], [], [], [], 0
    for i, p in enumerate(requests):
        calls = [stream_timed(base_url, model, p, api, max_tokens, chat_kwargs) for _ in range(repeats + 1)]
        rows += [{"request": i, "call": k, **c} for k, c in enumerate(calls)]
        first, again = calls[0], calls[1:]
        if not first["tpot_s"]:
            continue
        firsts.append(first["tpot_s"])
        tpot_r.append(statistics.median(c["tpot_s"] or 0.0 for c in again) / first["tpot_s"])
        e2e_r.append(statistics.median(c["e2e_s"] for c in again) / first["e2e_s"])
        ttft_r.append(statistics.median(c["ttft_s"] or 0.0 for c in again) / first["ttft_s"])
        instant += sum(c["n_tokens"] <= 1 or _decode(c) < 0.01 * _decode(first) for c in again)
    reasons = []
    if not firsts:
        return {"passed": False, "reasons": ["no first call produced 2 or more tokens"], "evidence": {}, "rows": rows}
    med = statistics.median(firsts)
    noise = statistics.median(abs(x - med) for x in firsts) / med
    ratio, limit = statistics.median(tpot_r), 1 - threshold - noise
    if ratio < limit:
        reasons.append(f"median repeat/first TPOT ratio {ratio:.3f} is below {limit:.3f} "
                       f"(1 - threshold {threshold} - distinct-request TPOT noise {noise:.3f})")
    if instant:
        reasons.append(f"{instant} of {len(firsts) * repeats} repeated responses arrived in one chunk "
                       f"or with decode time under 1% of the first call's")
    evidence = {"requests": len(requests), "repeats": repeats, "tpot_ratio_median": ratio, "limit": limit,
                "noise": noise, "e2e_ratio_median": statistics.median(e2e_r),
                "ttft_ratio_median": statistics.median(ttft_r), "instant_repeats": instant}
    return {"passed": not reasons, "reasons": reasons, "evidence": evidence, "rows": rows}
