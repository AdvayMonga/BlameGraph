"""Collect what the gate needs from an OpenAI-compatible server: greedy generation, streaming, teacher-forced scoring.
Scoring goes through the engine's `Api`: vllm (/v1/completions), sglang (/generate) or none (raises NoLogprobs)."""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field

from .divergence import Position


@dataclass(frozen=True)
class Api:
    name: str = "vllm"
    chat_kwargs: dict = field(default_factory=dict)


class NoLogprobs(RuntimeError):
    """The engine's API exposes no teacher-forced logprobs."""


DEFAULT = Api()


def _post(url: str, body: dict, timeout: float = 600, trace_id: str | None = None) -> dict:
    headers = {"Content-Type": "application/json", **({"X-Trace-Id": trace_id} if trace_id else {})}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _chat_body(api: Api, model: str, messages: list[dict], max_tokens: int) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0}
    if api.chat_kwargs:
        body["chat_template_kwargs"] = dict(api.chat_kwargs)
    return body


def generate(base_url: str, model: str, messages: list[dict], max_tokens: int = 2048, api: Api = DEFAULT,
             trace_id: str | None = None) -> dict:
    """Greedy chat completion: {"text", "completion_tokens", "finish_reason"}; `trace_id` is sent as X-Trace-Id."""
    out = _post(f"{base_url.rstrip('/')}/v1/chat/completions", _chat_body(api, model, messages, max_tokens),
                trace_id=trace_id)
    ch = out["choices"][0]
    return {"text": ch["message"]["content"] or "", "completion_tokens": (out.get("usage") or {}).get("completion_tokens"),
            "finish_reason": ch.get("finish_reason")}


def generate_stream(base_url: str, model: str, messages: list[dict], max_tokens: int = 2048,
                    api: Api = DEFAULT) -> dict:
    """The same request, streamed. Returns {"first_token_text", "text"}."""
    body = {**_chat_body(api, model, messages, max_tokens), "stream": True}
    req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    first, parts = None, []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            delta = (json.loads(line[6:]).get("choices") or [{}])[0].get("delta") or {}
            if "content" not in delta or ("role" in delta and not delta["content"]):
                continue                          # role chunk, not a token
            if first is None:
                first = delta["content"] or ""
            parts.append(delta["content"] or "")
    return {"first_token_text": first, "text": "".join(parts)}


def score_tokens(base_url: str, model: str, token_ids: list[int], start: int, top_k: int = 20,
                 api: Api = DEFAULT) -> list[Position]:
    """Teacher-forced scoring of token_ids[start:] (start >= 1): one Position per token, with its top-k alternatives."""
    if api.name == "none":
        raise NoLogprobs("the engine's API exposes no teacher-forced logprobs")
    if api.name == "sglang":
        return _score_sglang(base_url, token_ids, start, top_k)
    if api.name != "vllm":
        raise ValueError(f"unknown logprobs api {api.name!r}; one of vllm, sglang, none")
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


def _score_sglang(base_url: str, token_ids: list[int], start: int, top_k: int) -> list[Position]:
    """Score through SGLang's /generate; its logprob rows are [logprob, token_id, text]."""
    out = _post(f"{base_url.rstrip('/')}/generate", {
        "input_ids": token_ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True, "logprob_start_len": start, "top_logprobs_num": top_k})
    meta = out.get("meta_info") or {}
    own, tops = meta.get("input_token_logprobs"), meta.get("input_top_logprobs")
    if own is None:
        raise RuntimeError("server returned no input_token_logprobs; the divergence check needs teacher-forced scoring")
    positions = []
    for j, i in enumerate(range(start, len(token_ids))):
        lp, tok = own[j][0], own[j][1]
        if tok != token_ids[i]:
            raise RuntimeError(f"input_token_logprobs at position {i} is for token {tok}, expected {token_ids[i]}")
        top = {int(t[1]): float(t[0]) for t in (tops[j] if tops and j < len(tops) and tops[j] else [])}
        top[tok] = float(lp)
        positions.append(Position(token=tok, logprob=float(lp), top=dict(sorted(top.items(), key=lambda kv: -kv[1])[:top_k])))
    return positions
