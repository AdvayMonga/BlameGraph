"""Collect what the gate needs from an OpenAI-compatible server (stdlib only).

Two calls, mirroring vLLM's API so one client serves the reference (vLLM, BF16, VLLM_BATCH_INVARIANT=1) and the
candidate engine:
  generate()       greedy chat completion, thinking off   -> text, completion token count (for flips and length)
  score_tokens()   POST /v1/completions {prompt: [token ids], max_tokens: 1, prompt_logprobs: k}
                   -> choices[0].prompt_logprobs: one entry per prompt token (first is null), each
                      {"<token id>": {"logprob": float, ...}, ...} containing at least the actual token.
A candidate server must implement score_tokens' contract for the divergence check. Verify field names on the
first GPU run against the vLLM version in use.
"""
from __future__ import annotations

import json
import urllib.request

from .divergence import Position


def _post(url: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def generate(base_url: str, model: str, messages: list[dict], max_tokens: int = 2048) -> dict:
    """Greedy, thinking off. Returns {"text", "completion_tokens", "finish_reason"}."""
    out = _post(f"{base_url.rstrip('/')}/v1/chat/completions", {
        "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False}})
    ch = out["choices"][0]
    return {"text": ch["message"]["content"] or "", "completion_tokens": (out.get("usage") or {}).get("completion_tokens"),
            "finish_reason": ch.get("finish_reason")}


def score_tokens(base_url: str, model: str, token_ids: list[int], start: int, top_k: int = 20) -> list[Position]:
    """Teacher-forced scoring of token_ids[start:] given everything before it (start >= 1).
    Returns one Position per scored token: the token's own logprob and the top-k alternatives."""
    out = _post(f"{base_url.rstrip('/')}/v1/completions", {
        "model": model, "prompt": token_ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": top_k})
    plp = out["choices"][0].get("prompt_logprobs")
    if plp is None:
        raise RuntimeError("server returned no prompt_logprobs; the divergence check needs teacher-forced scoring")
    positions = []
    for i in range(start, len(token_ids)):
        entry = {int(k): (v["logprob"] if isinstance(v, dict) else float(v)) for k, v in (plp[i] or {}).items()}
        tok = token_ids[i]
        if tok not in entry:
            raise RuntimeError(f"prompt_logprobs at position {i} does not include the actual token {tok}")
        top = dict(sorted(entry.items(), key=lambda kv: -kv[1])[:top_k])
        positions.append(Position(token=tok, logprob=entry[tok], top=top))
    return positions
