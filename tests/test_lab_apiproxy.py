"""lab.apiproxy against a fake Anthropic API: credential injection, usage pricing (JSON and SSE), the cap, refusals."""

from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from lab import apiproxy

USAGE = {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 5000,
         "cache_creation_input_tokens": 300,
         "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200}}
SSE = (b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-opus-5-5","usage":'
       + json.dumps({**USAGE, "output_tokens": 1}).encode() + b'}}\n\n'
       b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
       b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":200}}\n\n'
       b'event: message_stop\ndata: {"type":"message_stop"}\n\n')


@pytest.fixture
def upstream():
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
            out, kind = ((SSE, "text/event-stream") if body.get("stream") else
                         (json.dumps({"model": body["model"], "usage": USAGE}).encode(), "application/json"))
            self.send_response(200)
            self.send_header("content-type", kind)
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    srv.shutdown()


def post(proxy, path, body, headers=None):
    c = http.client.HTTPConnection(urlsplit(proxy.url).netloc, timeout=10)
    c.request("POST", path, body=json.dumps(body), headers={"x-api-key": apiproxy.PLACEHOLDER,
                                                             "content-type": "application/json", **(headers or {})})
    r = c.getresponse()
    return r.status, r.read()


def expected(model: str) -> float:
    p_in, p_out, p_read, p_5m, p_1h = apiproxy.PRICES[model]
    return (1000 * p_in + 200 * p_out + 5000 * p_read + 100 * p_5m + 200 * p_1h) / 1e6


def test_the_lab_key_replaces_the_jails_placeholder(upstream, monkeypatch):
    url, seen = upstream
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")
    with apiproxy.Proxy(5.0, upstream=url) as p:
        status, _ = post(p, "/v1/messages?beta=true", {"model": "claude-opus-5-5", "max_tokens": 10})
    assert status == 200 and seen[0]["headers"]["x-api-key"] == "sk-real"
    assert seen[0]["path"] == "/v1/messages?beta=true" and "authorization" not in seen[0]["headers"]


def test_an_oauth_token_goes_as_bearer_with_its_beta(upstream, monkeypatch):
    url, seen = upstream
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oat-real")
    with apiproxy.Proxy(5.0, upstream=url) as p:
        post(p, "/v1/messages", {"model": "claude-opus-5-5"}, {"anthropic-beta": "fast-x"})
    h = seen[0]["headers"]
    assert h["authorization"] == "Bearer oat-real" and "x-api-key" not in h
    assert set(h["anthropic-beta"].split(",")) == {"fast-x", apiproxy.OAUTH_BETA}


def test_json_and_streamed_usage_are_priced_and_the_stream_passes_through(upstream, monkeypatch):
    url, _ = upstream
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")
    with apiproxy.Proxy(5.0, upstream=url) as p:
        post(p, "/v1/messages", {"model": "claude-haiku-4-5"})
        status, body = post(p, "/v1/messages", {"model": "claude-opus-5-5", "stream": True})
    assert status == 200 and body == SSE
    assert p.spent_usd == pytest.approx(expected("claude-haiku-4-5") + expected("claude-opus-5-5"))
    assert p.requests == 2 and p.usage["output_tokens"] == 400 and p.usage["cache_read_input_tokens"] == 10000


def test_a_spent_cap_refuses_the_next_request(upstream, monkeypatch):
    url, seen = upstream
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")
    with apiproxy.Proxy(expected("claude-opus-5-5") / 2, upstream=url) as p:
        assert post(p, "/v1/messages", {"model": "claude-opus-5-5"})[0] == 200
        status, body = post(p, "/v1/messages", {"model": "claude-opus-5-5"})
    assert status == 403 and b"is spent" in body and len(seen) == 1


@pytest.mark.parametrize("path,body,why", [
    ("/v1/messages", {"model": "some-new-model"}, b"no price"),
    ("/v1/messages", {"model": "claude-opus-5-5", "speed": "fast"}, b"fast mode"),
    ("/v1/messages/batches", {"requests": []}, b"not forwarded"),
    ("/v1/files", {}, b"not forwarded"),
])
def test_anything_unpriced_is_refused_and_never_sent(upstream, monkeypatch, path, body, why):
    url, seen = upstream
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")
    with apiproxy.Proxy(5.0, upstream=url) as p:
        status, out = post(p, path, body)
    assert status in (403, 400) and why in out and seen == []


def test_count_tokens_is_forwarded_and_costs_nothing(upstream, monkeypatch):
    url, seen = upstream
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")
    with apiproxy.Proxy(5.0, upstream=url) as p:
        assert post(p, "/v1/messages/count_tokens", {"model": "claude-opus-5-5"})[0] == 200
    assert len(seen) == 1 and p.spent_usd == 0


def test_no_credential_means_no_proxy(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        apiproxy.Proxy(5.0)


def test_a_client_that_drops_mid_stream_is_charged_its_input_and_all_of_max_tokens(monkeypatch):
    import time
    seen = []

    class Slow(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            start = b'data: {"type":"message_start","message":{"model":"claude-opus-5-5","usage":{"input_tokens":1000,"output_tokens":1}}}\n\n'
            self.wfile.write(start)
            self.wfile.flush()
            try:
                for _ in range(50):                     # the rest never comes before the client hangs up
                    time.sleep(0.05)
                    self.wfile.write(b'data: {"type":"ping"}\n\n')
                    self.wfile.flush()
            except OSError:
                pass
            seen.append(True)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")
    with apiproxy.Proxy(50.0, upstream=f"http://127.0.0.1:{srv.server_address[1]}") as p:
        c = http.client.HTTPConnection(urlsplit(p.url).netloc, timeout=10)
        c.request("POST", "/v1/messages", body=json.dumps({"model": "claude-opus-5-5", "max_tokens": 4000, "stream": True}),
                  headers={"content-type": "application/json"})
        r = c.getresponse()
        r.read1(200)
        c.close()                                       # hang up
        deadline = time.time() + 10
        while not p.requests and time.time() < deadline:
            time.sleep(0.1)
    srv.shutdown()
    p_in, p_out = apiproxy.PRICES["claude-opus-5-5"][:2]
    assert p.spent_usd == pytest.approx((1000 * p_in + 4000 * p_out) / 1e6) and p.reserved_usd == pytest.approx(0)


def test_a_request_whose_worst_case_would_pass_the_cap_is_refused_before_it_goes(upstream, monkeypatch):
    url, seen = upstream
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")
    with apiproxy.Proxy(0.10, upstream=url) as p:                 # 10k output tokens of opus is $0.20
        status, body = post(p, "/v1/messages", {"model": "claude-opus-5-5", "max_tokens": 10000})
    assert status == 403 and b"would be" in body and seen == []
