"""Adversarial canaries: a proxy in front of any OpenAI-compatible server that cheats in one chosen way.
The harness (measurement + correctness gate + validity) must flag every cheat; one that goes unflagged is a
blind spot. Engine-agnostic, stdlib only.

  python -m blamegraph.canaries --upstream http://127.0.0.1:8000 --port 8100 --cheat truncate

Cheats (what should catch it):
  truncate      cap max_tokens at 40% of the request            -> length gate
  early_eos     stop every response after its first 8 tokens     -> length gate
  fake_first    send an empty chunk instantly before upstream    -> first-real-token TTFT, consistency gate
  drop          answer 20% of requests with an empty 200 "stop"  -> client failure counting (silent drop)
  cache         replay stored responses for repeated prompts     -> repeated-request audit (must not get >10% faster)
  inflate_usage report 1.5x the real completion token count      -> consistency gate (token-count mismatch)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHEATS = ("truncate", "early_eos", "fake_first", "drop", "cache", "inflate_usage")


def make_handler(upstream: str, cheat: str, seed: int = 0):
    rng = random.Random(seed)
    store: dict[str, tuple[bytes, str]] = {}
    lock = threading.Lock()

    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, data: bytes, ctype: str):
            self.send_response(200); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            with urllib.request.urlopen(upstream.rstrip("/") + self.path) as r:
                self._send(r.read(), r.headers.get("Content-Type", "application/json"))

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            stream = bool(body.get("stream"))
            key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            if cheat == "cache":
                with lock:
                    hit = store.get(key)
                if hit:
                    return self._send(*hit)
            if cheat == "drop" and rng.random() < 0.2:
                return self._empty(stream)
            if cheat == "truncate" and body.get("max_tokens"):
                body["max_tokens"] = max(1, int(body["max_tokens"] * 0.4))
            if cheat == "early_eos":
                body["max_tokens"] = min(body.get("max_tokens") or 8, 8)
            req = urllib.request.Request(upstream.rstrip("/") + self.path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                ctype = r.headers.get("Content-Type", "application/json")
                if not stream:
                    data = r.read()
                    if cheat == "inflate_usage":
                        d = json.loads(data)
                        if d.get("usage", {}).get("completion_tokens") is not None:
                            d["usage"]["completion_tokens"] = int(d["usage"]["completion_tokens"] * 1.5) + 1
                        data = json.dumps(d).encode()
                    if cheat == "cache":
                        with lock:
                            store[key] = (data, ctype)
                    return self._send(data, ctype)
                # streaming: relay SSE lines, optionally preceded by a fake empty chunk
                self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
                if cheat == "fake_first":
                    self._chunk(b'data: {"choices":[{"index":0,"delta":{"content":""},"text":""}]}\n\n')
                for line in r:
                    self._chunk(line)
                self._chunk(b"")

        def _chunk(self, data: bytes):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n"); self.wfile.flush()

        def _empty(self, stream: bool):
            if stream:
                self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
                self._chunk(b"data: [DONE]\n\n"); self._chunk(b"")
            else:
                self._send(json.dumps({"choices": [{"index": 0, "text": "", "message": {"role": "assistant", "content": ""},
                                                    "finish_reason": "stop"}], "usage": {"completion_tokens": 0}}).encode(),
                           "application/json")

    return Proxy


def serve(upstream: str, port: int, cheat: str, seed: int = 0) -> ThreadingHTTPServer:
    if cheat not in CHEATS:
        raise ValueError(f"unknown cheat {cheat!r}; one of {CHEATS}")
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(upstream, cheat, seed))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="blamegraph.canaries")
    ap.add_argument("--upstream", required=True); ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--cheat", required=True, choices=CHEATS); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    s = serve(a.upstream, a.port, a.cheat, a.seed)
    print(f"cheat proxy '{a.cheat}' on http://127.0.0.1:{a.port} -> {a.upstream}")
    threading.Event().wait()
