"""Repeated-request audit passes prefix caching and fails replayed answers. Run: python tests/test_repeat_audit.py"""
from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validity.repeat_audit import repeat_audit  # noqa: E402
from canaries.cheat_proxy import serve  # noqa: E402

PROMPTS = ["a", "b", "c"]


def _upstream():
    """Fake streaming engine: TTFT 60 ms (6 ms for a prompt seen before, like prefix caching), then 8 ms per token."""
    seen = set()

    class Upstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _chunk(self, data: bytes):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n"); self.wfile.flush()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            chat = self.path.endswith("/chat/completions")
            key = json.dumps(body.get("messages") or body.get("prompt"))
            time.sleep(0.006 if key in seen else 0.06); seen.add(key)
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            if chat:
                self._chunk(b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n')
            for i in range(16):
                if i:
                    time.sleep(0.008)
                ch = {"index": 0, "delta": {"content": f"w{i} "}} if chat else {"index": 0, "text": f"w{i} "}
                self._chunk(f"data: {json.dumps({'choices': [ch]})}\n\n".encode())
            self._chunk(b"data: [DONE]\n\n"); self._chunk(b"")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_prefix_caching_passes():
    for api in ("chat", "completions"):
        up, url = _upstream()
        res = repeat_audit(url, "m", PROMPTS, api=api)
        assert res["passed"], (api, res["reasons"], res["evidence"])
        assert res["evidence"]["ttft_ratio_median"] < 0.5, res["evidence"]   # repeats' TTFT did drop
        assert len(res["rows"]) == 12 and all(r["n_tokens"] == 16 for r in res["rows"])
        up.shutdown()


def test_replayed_answers_fail():
    for api in ("chat", "completions"):
        up, upurl = _upstream()
        px = serve(upurl, 0, "cache")
        res = repeat_audit(f"http://127.0.0.1:{px.server_address[1]}", "m", PROMPTS, api=api)
        assert not res["passed"] and len(res["reasons"]) == 2, (api, res)
        assert res["evidence"]["instant_repeats"] == 9 and res["evidence"]["tpot_ratio_median"] < 0.1, res["evidence"]
        up.shutdown(); px.shutdown()


if __name__ == "__main__":
    test_prefix_caching_passes(); print("ok test_prefix_caching_passes")
    test_replayed_answers_fail(); print("ok test_replayed_answers_fail")
