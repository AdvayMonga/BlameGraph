"""One streamed chat request, timed client-side. Stdlib only, so the client is never the engine's dependency.

Times are seconds on the run's perf_counter clock. TTFT runs from the request's *scheduled* send time to its first
real token (a role-only chunk with empty content is not a token), so a lagging client counts against itself and is
reported separately as `lag_s`.
"""
from __future__ import annotations

import asyncio
import http.client
import json
import time
import urllib.error
import urllib.parse
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
    ignore_eos: bool = False          # fixed answer length: the server runs to max_tokens past its stop token


@dataclass
class Row:
    id: str
    status: str                       # ok | error (explicit 4xx/503 rejection) | silent_drop | truncated | crash (5xx, no response)
    scheduled_s: float
    lag_s: float                      # actual send - scheduled send
    ttft_s: float | None = None
    e2e_s: float | None = None
    n_tokens: int = 0
    prompt_tokens: int | None = None   # from the server's usage
    cached_tokens: int | None = None   # prompt tokens the server served from its prefix cache, when it reports them
    tpot_s: float | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    client_cpu: float | None = None    # closed loop: CPU time / wall time of the process that sent it, over its call
    detail: str = ""
    text: str = field(default="", repr=False)

    def to_dict(self, keep_text: bool = False) -> dict:
        d = asdict(self)
        if not keep_text:
            d.pop("text")
        return d


def _encode(model: str, req: Request, chat_kwargs: dict | None, trace_prefix: str) -> tuple[bytes, dict]:
    """The request's JSON body and headers."""
    body = {"model": model, "messages": req.messages, "max_tokens": req.max_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}
    if chat_kwargs:
        body["chat_template_kwargs"] = dict(chat_kwargs)
    if req.ignore_eos:            # special tokens kept in the text, so a re-count from it stays exact
        body.update(ignore_eos=True, skip_special_tokens=False)
    headers = {"Content-Type": "application/json", "X-Trace-Id": f"{trace_prefix}-{req.id}"}
    if req.session_id is not None:
        headers["X-Session-Id"] = str(req.session_id)
    if req.turn_index is not None:
        headers["X-Turn-Index"] = str(req.turn_index)
    return json.dumps(body).encode(), headers


class _Stream:
    """An SSE response read line by line: the generated text, first-token time, usage and finish reason."""

    def __init__(self):
        self.parts, self.first, self.usage, self.finish, self.done = [], None, None, None, False

    def feed(self, raw: bytes, clock) -> None:
        line = raw.strip()
        if not line.startswith(b"data: "):
            return
        if line == b"data: [DONE]":
            self.done = True                   # keep reading to EOF: closing early breaks the server's last write
            return
        chunk = json.loads(line[6:])
        self.usage = chunk.get("usage") or self.usage
        for ch in chunk.get("choices") or []:
            text = (ch.get("delta") or {}).get("content")
            if text:
                if self.first is None:
                    self.first = clock()
                self.parts.append(text)
            self.finish = ch.get("finish_reason") or self.finish

    def row(self, row: Row, req: Request, scheduled: float, end: float, count_tokens) -> Row:
        """The finished row: text, tokens, TTFT and TPOT, and the stream's status."""
        row.text, row.finish_reason, row.e2e_s = "".join(self.parts), self.finish, end - scheduled
        usage = self.usage
        if usage and usage.get("prompt_tokens") is not None:
            row.prompt_tokens = int(usage["prompt_tokens"])
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            row.cached_tokens = int(cached) if cached is not None else None
        if usage and usage.get("completion_tokens") is not None:
            row.n_tokens = int(usage["completion_tokens"])
        else:
            row.n_tokens = len(self.parts)
        if self.first is not None:
            row.ttft_s = self.first - scheduled
        recount(row, count_tokens)
        if not row.text:
            row.status, row.detail = "silent_drop", "200 with no generated text"
        elif not self.done and self.finish is None:
            row.status, row.detail = "truncated", "stream ended without a finish reason or [DONE]"
        elif req.ignore_eos and self.finish != "length":
            row.status, row.detail = "truncated", f"fixed-length answer ended with finish {self.finish!r} before max_tokens"
        else:
            row.status = "ok"
        return row


def _rejected(row: Row, status: int, detail: bytes, clock, scheduled: float) -> Row:
    explicit = status < 500 or status == 503          # a deliberate rejection; other 5xx is the server failing
    row.status, row.http_status = ("error" if explicit else "crash"), status
    row.detail, row.e2e_s = detail[:200].decode("utf-8", "replace"), clock() - scheduled
    return row


def send(url: str, model: str, req: Request, scheduled: float, clock, timeout: float = 600,
         count_tokens=None, trace_prefix: str = "bg", chat_kwargs: dict | None = None) -> Row:
    """POST req as a streamed chat completion; `clock()` is the run clock and `scheduled` the intended send time."""
    data, headers = _encode(model, req, chat_kwargs, trace_prefix)
    http_req = urllib.request.Request(f"{url.rstrip('/')}/v1/chat/completions", data=data, headers=headers)
    start = clock()
    row, stream = Row(req.id, "crash", scheduled, start - scheduled), _Stream()
    try:
        with urllib.request.urlopen(http_req, timeout=timeout) as r:
            row.http_status = r.status
            for raw in r:
                stream.feed(raw, clock)
    except urllib.error.HTTPError as e:
        return _rejected(row, e.code, e.read(), clock, scheduled)
    except (urllib.error.URLError, OSError, http.client.HTTPException, json.JSONDecodeError, ValueError) as e:
        row.detail = f"{type(e).__name__}: {e}"[:200]   # includes a stream that dies mid-way (IncompleteRead)
        row.e2e_s = clock() - scheduled
        return row
    return stream.row(row, req, scheduled, clock(), count_tokens)


async def asend(url: str, model: str, req: Request, clock, timeout: float = 600, count_tokens=None,
                trace_prefix: str = "bg", chat_kwargs: dict | None = None, lead_s: float = 0.1) -> Row:
    """`send` on an event loop, sent at `req.at_s` on `clock`: the body is encoded and the connection opened `lead_s`
    before, so at the send time only the request's bytes are written. One loop drives a process's whole schedule; a
    thread per request made a burst's sends queue behind each other (and the interpreter lock)."""
    scheduled = req.at_s
    await asyncio.sleep(max(0.0, scheduled - lead_s - clock()))
    data, headers = _encode(model, req, chat_kwargs, trace_prefix)
    u = urllib.parse.urlsplit(url)
    head = "".join(f"{k}: {v}\r\n" for k, v in {**headers, "Host": u.netloc, "Content-Length": len(data),
                                                 "Connection": "close"}.items())
    msg = f"POST {u.path.rstrip('/')}/v1/chat/completions HTTP/1.1\r\n{head}\r\n".encode() + data
    writer, refused = None, None
    try:
        reader, writer = await asyncio.open_connection(u.hostname, u.port)
    except OSError as e:
        refused = e
    await asyncio.sleep(max(0.0, scheduled - clock()))
    row, stream = Row(req.id, "crash", scheduled, clock() - scheduled), _Stream()
    try:
        if refused is not None:
            raise refused
        writer.write(msg)
        async with asyncio.timeout(timeout):
            await writer.drain()
            status = int((await reader.readline()).split()[1])
            fields = {}
            while (line := await reader.readline()) not in (b"\r\n", b"\n", b""):
                k, _, v = line.decode("latin-1").partition(":")
                fields[k.strip().lower()] = v.strip()
            row.http_status = status
            if status >= 400:
                return _rejected(row, status, await reader.read(200), clock, scheduled)
            async for raw in _lines(reader, fields.get("transfer-encoding", "").lower() == "chunked"):
                stream.feed(raw, clock)
    except (OSError, TimeoutError, asyncio.IncompleteReadError, json.JSONDecodeError, ValueError, IndexError) as e:
        row.detail = f"{type(e).__name__}: {e}"[:200]   # includes a stream that dies mid-way
        row.e2e_s = clock() - scheduled
        return row
    finally:
        if writer is not None:
            writer.close()
    return stream.row(row, req, scheduled, clock(), count_tokens)


async def _lines(reader, chunked: bool):
    """The response body's lines, de-chunked; a chunked body that ends without its last chunk raises."""
    if not chunked:
        while line := await reader.readline():
            yield line
        return
    buf = b""
    while True:
        size = int((await reader.readline()).split(b";")[0], 16)
        if size == 0:
            break
        buf += await reader.readexactly(size)
        await reader.readexactly(2)
        *done, buf = buf.split(b"\n")
        for line in done:
            yield line + b"\n"
    if buf:
        yield buf


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
