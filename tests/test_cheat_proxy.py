"""Each cheat proxy behaves as specified and is caught by the matching check. Run: python tests/test_canaries.py"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from canaries.cheat_proxy import serve  # noqa: E402
from correctness import consistency, length_ratio  # noqa: E402

WORDS = [f"w{i}" for i in range(64)]


class Upstream(BaseHTTPRequestHandler):
    """Fake engine: emits min(max_tokens, 64) words, 2 ms per token, streaming or not."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = min(body.get("max_tokens") or 64, 64)
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            for w in WORDS[:n]:
                time.sleep(0.002)
                line = f'data: {json.dumps({"choices": [{"index": 0, "text": w + " "}]})}\n\n'.encode()
                self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n"); self.wfile.flush()
            end = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(end):x}\r\n".encode() + end + b"\r\n0\r\n\r\n")
        else:
            time.sleep(0.002 * n)
            data = json.dumps({"choices": [{"index": 0, "text": " ".join(WORDS[:n]), "finish_reason": "length"}],
                               "usage": {"completion_tokens": n}}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


def _post(url, body):
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def _stream_chunks(url, body):
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps({**body, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    texts = []
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.strip()
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                texts.append(json.loads(line[6:])["choices"][0].get("text", ""))
    return texts


def _with(cheat):
    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream); threading.Thread(target=up.serve_forever, daemon=True).start()
    px = serve(f"http://127.0.0.1:{up.server_address[1]}", 0, cheat)
    return up, px, f"http://127.0.0.1:{px.server_address[1]}"


def test_cheats_are_caught():
    tok = lambda s: s.split()
    body = {"prompt": "p", "max_tokens": 50}
    # truncate / early_eos -> length gate fails
    for cheat in ("truncate", "early_eos"):
        up, px, url = _with(cheat)
        n = json.loads(_post(url, body))["usage"]["completion_tokens"]
        assert length_ratio([50] * 5, [n] * 5)["ratio"] < 0.9, (cheat, n)
        up.shutdown(); px.shutdown()
    # fake_first -> first streamed chunk is empty, consistency gate fails
    up, px, url = _with("fake_first")
    chunks = _stream_chunks(url, body)
    assert chunks[0] == "" and not consistency([{"text": "".join(chunks), "first_token_text": chunks[0]}], tok)["consistent"]
    up.shutdown(); px.shutdown()
    # inflate_usage -> reported count disagrees with the text
    up, px, url = _with("inflate_usage")
    d = json.loads(_post(url, body))
    assert not consistency([{"text": d["choices"][0]["text"], "reported_tokens": d["usage"]["completion_tokens"]}], tok)["consistent"]
    up.shutdown(); px.shutdown()
    # drop -> some 200 responses are silently empty
    up, px, url = _with("drop")
    empties = sum(json.loads(_post(url, {**body, "prompt": f"p{i}"}))["choices"][0]["text"] == "" for i in range(40))
    assert empties > 0
    up.shutdown(); px.shutdown()
    # cache -> repeating a request becomes much faster (repeated-request audit: must not get >10% faster)
    up, px, url = _with("cache")
    t0 = time.perf_counter(); _post(url, body); first = time.perf_counter() - t0
    t0 = time.perf_counter(); _post(url, body); again = time.perf_counter() - t0
    assert again < 0.9 * first, (first, again)
    up.shutdown(); px.shutdown()


if __name__ == "__main__":
    test_cheats_are_caught(); print("ok test_cheats_are_caught")
