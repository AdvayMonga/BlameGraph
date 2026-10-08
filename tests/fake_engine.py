"""A fake OpenAI-compatible streaming engine with a fixed number of decode slots, for the regime tests.

Requests wait for a free slot (so queueing raises TTFT), then stream one token per `token_s`. With `queue_limit`, a
request arriving to a full queue gets an explicit 503. With $TELEMETRY_DIR set, each streamed request appends
{trace_id, tokens_out} to `requests.jsonl` there, like an engine's own per-request rows. Run standalone:
  python tests/fake_engine.py --port N [--delay S]
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_server(slots: int = 4, token_s: float = 0.002, out_tokens: int = 20, queue_limit: int | None = None,
                port: int = 0, wrong_every: int = 0) -> ThreadingHTTPServer:
    """`wrong_every` = N: every Nth multiple-choice answer is wrong (a known-bad candidate for the gate)."""
    counter = [0]
    sem, lock, waiting = threading.Semaphore(slots), threading.Lock(), [0]

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _chunk(self, obj):
            line = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
            self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n"); self.wfile.flush()

        def do_GET(self):
            data = b'{"ok": true}'
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def _json(self, obj, status=200):
            data = json.dumps(obj).encode()
            self.send_response(status); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/v1/completions"):        # teacher-forced scoring, vLLM-shaped: deterministic
                ids = body["prompt"]
                plp = [None] + [{str(t): {"logprob": -0.1 - (t % 7) / 100}, str(t + 1): {"logprob": -2.0}} for t in ids[1:]]
                return self._json({"choices": [{"text": "", "prompt_logprobs": plp}]})
            if not body.get("stream"):
                last = body["messages"][-1]["content"]
                n = min(out_tokens, body.get("max_tokens") or out_tokens)
                counter[0] += 1
                wrong = wrong_every and counter[0] % wrong_every == 0
                text = ("42" if "17 + 25" in last else ("The answer is (B)." if wrong else "The answer is (C).")
                        if "pick one" in last else " ".join(f"t{i}" for i in range(n)))
                time.sleep(token_s * n)
                return self._json({"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                                   "usage": {"completion_tokens": n}})
            with lock:
                if queue_limit is not None and waiting[0] >= queue_limit:
                    data = b'{"error": "overloaded"}'
                    self.send_response(503); self.send_header("Content-Length", str(len(data))); self.end_headers()
                    self.wfile.write(data); return
                waiting[0] += 1
            sem.acquire()
            with lock:
                waiting[0] -= 1
            try:
                last = body["messages"][-1]["content"]
                text = ["42"] if "17 + 25" in last else [f"t{i} " for i in range(min(out_tokens, body.get("max_tokens") or out_tokens))]
                self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
                self._chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})
                for t in text:
                    time.sleep(token_s)
                    self._chunk({"choices": [{"index": 0, "delta": {"content": t}}]})
                self._chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                self._chunk({"choices": [], "usage": {"completion_tokens": len(text)}})
                if os.environ.get("TELEMETRY_DIR"):
                    with lock, open(os.path.join(os.environ["TELEMETRY_DIR"], "requests.jsonl"), "a") as f:
                        f.write(json.dumps({"trace_id": self.headers.get("X-Trace-Id"), "tokens_out": len(text)}) + "\n")
                self._chunk(b"[DONE]")
                self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
            finally:
                sem.release()

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def spawn(slots: int = 4, token_s: float = 0.002, out_tokens: int = 20, queue_limit: int | None = None):
    """The fake engine in its own process (so it can't steal the client's interpreter time); returns (proc, url)."""
    import socket
    import subprocess
    import sys
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    cmd = [sys.executable, __file__, "--port", str(port), "--slots", str(slots), "--token-s", str(token_s),
           "--out-tokens", str(out_tokens)] + (["--queue-limit", str(queue_limit)] if queue_limit is not None else [])
    proc = subprocess.Popen(cmd)
    for _ in range(200):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)
    return proc, f"http://127.0.0.1:{port}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--slots", type=int, default=4); ap.add_argument("--token-s", type=float, default=0.002)
    ap.add_argument("--out-tokens", type=int, default=20); ap.add_argument("--queue-limit", type=int)
    ap.add_argument("--wrong-every", type=int, default=0)
    a = ap.parse_args()
    time.sleep(a.delay)                      # stands in for loading weights and compiling
    make_server(a.slots, a.token_s, a.out_tokens, a.queue_limit, a.port, a.wrong_every)
    threading.Event().wait()
