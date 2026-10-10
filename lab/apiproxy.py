"""The model API key stays out of every jail: the agent's CLI talks to this proxy, which adds the lab's credential,
forwards to the Anthropic API, prices each response's usage, and refuses once the session's dollar cap is spent.

Only /v1/messages, /v1/messages/count_tokens and /v1/models[/id] are forwarded (not batches, files or anything else); a request naming a model
without a price here, or asking for fast mode, is refused (fail closed: anything unpriced could spend unmetered).
Prices are first-party per-million-token rates (claude-api reference, cached 2026-09-25)."""

from __future__ import annotations

import http.client
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

UPSTREAM = "https://api.anthropic.com"
PLACEHOLDER = "lab-proxy"                    # what the jailed CLI holds instead of a key
OAUTH_BETA = "oauth-2025-04-20"
# $/MTok: input, output, cache read, cache write 5 min, cache write 1 h
PRICES = {
    "claude-fable-5-1": (10.0, 50.0, 0.25, 12.5, 20.0),
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0, 8.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5, 4.0),
    "claude-haiku-4-5": (1.0, 5.0, 0.10, 1.25, 2.0),
}
ALLOWED = ("/v1/messages", "/v1/messages/count_tokens", "/v1/models")
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization",
       "proxy-authenticate", "content-length", "host", "accept-encoding"}


def price(model: str) -> tuple | None:
    """The model's rates; a `[1m]`-style suffix the CLI adds does not change them."""
    return PRICES.get(model.split("[")[0])


def cost_usd(model: str, usage: dict) -> float:
    p_in, p_out, p_read, p_5m, p_1h = price(model)
    created = usage.get("cache_creation") or {}
    w5, w1 = created.get("ephemeral_5m_input_tokens"), created.get("ephemeral_1h_input_tokens")
    if w5 is None and w1 is None:                    # no breakdown: every write at the 5-minute rate
        w5, w1 = usage.get("cache_creation_input_tokens") or 0, 0
    return ((usage.get("input_tokens") or 0) * p_in + (usage.get("output_tokens") or 0) * p_out
            + (usage.get("cache_read_input_tokens") or 0) * p_read + (w5 or 0) * p_5m + (w1 or 0) * p_1h) / 1e6


def credential() -> dict[str, str]:
    """The headers that authenticate upstream, from the lab's own environment."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {"x-api-key": os.environ["ANTHROPIC_API_KEY"]}
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return {"authorization": f"Bearer {os.environ['CLAUDE_CODE_OAUTH_TOKEN']}"}
    raise RuntimeError("no model credential for the proxy: set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN")


class _Usage:
    """Usage read off a response as it streams by: JSON, or SSE message_start + message_delta."""

    def __init__(self, sse: bool):
        self.sse, self.buf, self.model, self.usage = sse, b"", None, {}

    def feed(self, chunk: bytes) -> None:
        self.buf += chunk
        if not self.sse:
            return
        *lines, self.buf = self.buf.split(b"\n")
        for line in lines:
            if line.startswith(b"data:"):
                self._event(line[5:].strip())

    def _event(self, data: bytes) -> None:
        try:
            e = json.loads(data)
        except ValueError:
            return
        if e.get("type") == "message_start":
            self.model = e["message"].get("model")
            self.usage.update(e["message"].get("usage") or {})
        elif e.get("type") == "message_delta":
            self.usage.update(e.get("usage") or {})

    def done(self) -> None:
        if self.sse:
            self.feed(b"\n")
            return
        try:
            body = json.loads(self.buf)
        except ValueError:
            return
        if isinstance(body, dict) and isinstance(body.get("usage"), dict):
            self.model, self.usage = body.get("model"), body["usage"]


class Proxy:
    """`with Proxy(cap_usd, host) as p: ... p.url`; `p.spent_usd` and `p.usage` are what the session cost."""

    def __init__(self, cap_usd: float, host: str = "127.0.0.1", upstream: str = UPSTREAM):
        self.cap_usd, self.spent_usd, self.requests = cap_usd, 0.0, 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}
        self.auth = credential()
        self.upstream = urlsplit(upstream)
        self.lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                proxy._handle(self, None)

            def do_POST(self):
                n = self.headers.get("content-length")
                if n is None:
                    return proxy._refuse(self, 411, "a request body needs a Content-Length")
                proxy._handle(self, self.rfile.read(int(n)))

        self.server = ThreadingHTTPServer((host, 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://{host}:{self.server.server_address[1]}"

    def __enter__(self) -> "Proxy":
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _refuse(self, h, status: int, why: str) -> None:
        body = json.dumps({"type": "error", "error": {"type": "permission_error", "message": f"lab proxy: {why}"}}).encode()
        h.send_response(status)
        h.send_header("content-type", "application/json")
        h.send_header("content-length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)

    def _handle(self, h, body: bytes | None) -> None:
        path = urlsplit(h.path).path
        if path not in ALLOWED and not (path.startswith("/v1/models/") and path.count("/") == 3):
            return self._refuse(h, 403, f"{path} is not forwarded")
        model = None
        if body:
            try:
                req = json.loads(body)
            except ValueError:
                return self._refuse(h, 400, "the request body is not JSON")
            model = req.get("model")
            if path == "/v1/messages" and (not isinstance(model, str) or price(model) is None):
                return self._refuse(h, 403, f"no price for model {model!r}; priced: {sorted(PRICES)}")
            if req.get("speed") == "fast":
                return self._refuse(h, 403, "fast mode is not priced here")
        with self.lock:
            if path == "/v1/messages" and self.spent_usd >= self.cap_usd:
                return self._refuse(h, 403, f"this session's ${self.cap_usd:.2f} is spent")
        headers = {k: v for k, v in h.headers.items() if k.lower() not in HOP | {"x-api-key", "authorization"}}
        headers.update(self.auth, host=self.upstream.netloc, **{"accept-encoding": "identity"})
        if "authorization" in self.auth:
            betas = [b for b in headers.pop("anthropic-beta", "").split(",") if b.strip()]
            headers["anthropic-beta"] = ",".join(dict.fromkeys([*betas, OAUTH_BETA]))
        conn = (http.client.HTTPSConnection if self.upstream.scheme == "https" else http.client.HTTPConnection)(
            self.upstream.netloc, timeout=900)
        try:
            conn.request(h.command, h.path, body=body, headers=headers)
            resp = conn.getresponse()
            usage = _Usage("text/event-stream" in resp.getheader("content-type", ""))
            h.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    h.send_header(k, v)
            h.send_header("transfer-encoding", "chunked")
            h.end_headers()
            while chunk := resp.read1(65536):
                usage.feed(chunk)
                h.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                h.wfile.flush()
            h.wfile.write(b"0\r\n\r\n")
            usage.done()
        finally:
            conn.close()
        if path == "/v1/messages" and usage.usage:
            spent = cost_usd(usage.model if usage.model and price(usage.model) else model, usage.usage)
            with self.lock:
                self.spent_usd += spent
                self.requests += 1
                for k in self.usage:
                    self.usage[k] += usage.usage.get(k) or 0
