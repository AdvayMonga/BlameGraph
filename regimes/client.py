"""One streamed chat request, timed client-side. Stdlib only, so the client is never the engine's dependency.

Times are seconds on the run's perf_counter clock. TTFT runs from the request's *scheduled* send time to its first
real token (a role-only chunk with empty content is not a token), so a lagging client counts against itself and is
reported separately as `lag_s`.
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field


@dataclass
class Request:
    id: str
    messages: list[dict]
    max_tokens: int
    at_s: float = 0.0                 # scheduled send, seconds after the run starts (open loop)
    session_id: str | None = None
    turn_index: int | None = None


@dataclass
class Row:
    id: str
    status: str                       # ok | error (explicit 4xx/503 rejection) | silent_drop | truncated | crash (5xx, no response)
    scheduled_s: float
    lag_s: float                      # actual send - scheduled send
    ttft_s: float | None = None
    e2e_s: float | None = None
    n_tokens: int = 0
    tpot_s: float | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    detail: str = ""
    text: str = field(default="", repr=False)

    def to_dict(self, keep_text: bool = False) -> dict:
        d = asdict(self)
        if not keep_text:
            d.pop("text")
        return d


def send(url: str, model: str, req: Request, scheduled: float, clock, timeout: float = 600,
         count_tokens=None, trace_prefix: str = "bg", chat_kwargs: dict | None = None) -> Row:
    """POST req as a streamed chat completion; `clock()` is the run clock and `scheduled` the intended send time."""
    body = {"model": model, "messages": req.messages, "max_tokens": req.max_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}
    if chat_kwargs:
        body["chat_template_kwargs"] = dict(chat_kwargs)
    headers = {"Content-Type": "application/json", "X-Trace-Id": f"{trace_prefix}-{req.id}"}
    if req.session_id is not None:
        headers["X-Session-Id"] = str(req.session_id)
    if req.turn_index is not None:
        headers["X-Turn-Index"] = str(req.turn_index)
    http_req = urllib.request.Request(f"{url.rstrip('/')}/v1/chat/completions", data=json.dumps(body).encode(),
                                      headers=headers)
    start = clock()
    row = Row(req.id, "crash", scheduled, start - scheduled)
    parts, first, usage, finish, done = [], None, None, None, False
    try:
        with urllib.request.urlopen(http_req, timeout=timeout) as r:
            row.http_status = r.status
            for raw in r:
                line = raw.strip()
                if not line.startswith(b"data: "):
                    continue
                if line == b"data: [DONE]":
                    done = True                # keep reading to EOF: closing early breaks the server's last write
                    continue
                chunk = json.loads(line[6:])
                usage = chunk.get("usage") or usage
                for ch in chunk.get("choices") or []:
                    delta = ch.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        if first is None:
                            first = clock()
                        parts.append(text)
                    finish = ch.get("finish_reason") or finish
    except urllib.error.HTTPError as e:
        explicit = e.code < 500 or e.code == 503          # a deliberate rejection; other 5xx is the server failing
        row.status, row.http_status = ("error" if explicit else "crash"), e.code
        row.detail = e.read()[:200].decode("utf-8", "replace")
        row.e2e_s = clock() - scheduled
        return row
    except (urllib.error.URLError, OSError, http.client.HTTPException, json.JSONDecodeError, ValueError) as e:
        row.detail = f"{type(e).__name__}: {e}"[:200]   # includes a stream that dies mid-way (IncompleteRead)
        row.e2e_s = clock() - scheduled
        return row
    end = clock()
    row.text, row.finish_reason, row.e2e_s = "".join(parts), finish, end - scheduled
    if usage and usage.get("completion_tokens") is not None:
        row.n_tokens = int(usage["completion_tokens"])
    else:
        row.n_tokens = len(parts)
    if first is not None:
        row.ttft_s = first - scheduled
    recount(row, count_tokens)
    if not row.text:
        row.status, row.detail = "silent_drop", "200 with no generated text"
    elif not done and finish is None:
        row.status, row.detail = "truncated", "stream ended without a finish reason or [DONE]"
    else:
        row.status = "ok"
    return row


def recount(row: Row, count_tokens=None) -> Row:
    """Re-count output tokens with the reference tokenizer (when given) and derive TPOT = (E2E - TTFT) / (n - 1)."""
    if count_tokens is not None:
        row.n_tokens = len(count_tokens(row.text)) if row.text else 0
    row.tpot_s = (row.e2e_s - row.ttft_s) / (row.n_tokens - 1) if row.ttft_s is not None and row.n_tokens > 1 else None
    return row                                    # a one-token answer has no TPOT and never meets the limits


def wall_clock():
    """A run clock: seconds since its creation."""
    t0 = time.perf_counter()
    return lambda: time.perf_counter() - t0
