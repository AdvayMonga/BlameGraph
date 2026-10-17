"""The practice engine for `python -m lab.rehearsal`: an OpenAI-compatible server with a fixed number of decode
slots and a fixed time per token. Requests wait for a free slot, then stream one word per TOKEN_S; each word is one
token in the target model's tokenizer. Multiple-choice questions ("pick one") are answered "(C)". vLLM-shaped
teacher-forced scoring at /v1/completions. Stdlib only: this file is copied into the practice engine's repo.

  python server.py --port N [--wrong-every K]
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SLOTS = 2               # requests decoded at once; the rest wait
TOKEN_S = 0.02          # seconds per generated token
WORDS = [" the", " of", " and", " to", " in", " a", " is", " that", " for", " it"]
ANSWER, ANSWER_TOKENS = "The answer is (C).", 6
WRONG, WRONG_TOKENS = "The answer is (B).", 6
COLD, COLD_TOKENS = "42", 2


def answer(messages: list[dict], max_tokens: int, wrong: bool = False) -> tuple[list[str], int]:
    """The reply as streamed chunks and its token count."""
    last = messages[-1]["content"]
    if "17 + 25" in last:
        return [COLD], COLD_TOKENS
    if "pick one" in last:
        return ([WRONG], WRONG_TOKENS) if wrong else ([ANSWER], ANSWER_TOKENS)
    n = max(1, max_tokens)
    return [WORDS[i % len(WORDS)] for i in range(n)], n


def make_server(port: int = 0, wrong_every: int = 0) -> ThreadingHTTPServer:
    """`wrong_every` = K: every Kth multiple-choice answer is wrong (validation's known-bad launch)."""
    slots, lock, counter = threading.Semaphore(SLOTS), threading.Lock(), [0]

    def reply(messages: list[dict], max_tokens: int) -> tuple[list[str], int]:
        with lock:
            counter[0] += 1
            wrong = bool(wrong_every) and "pick one" in messages[-1]["content"] and counter[0] % wrong_every == 0
        return answer(messages, max_tokens, wrong)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, obj, status=200):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _chunk(self, obj):
            line = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
            self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
            self.wfile.flush()

        def do_GET(self):
            self._json({"ok": True})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/v1/completions"):           # teacher-forced scoring: deterministic logprobs
                ids = body["prompt"]
                plp = [None] + [{str(t): {"logprob": -0.1 - (t % 7) / 100}, str(t + 1): {"logprob": -2.0}}
                                for t in ids[1:]]
                return self._json({"choices": [{"text": "", "prompt_logprobs": plp}]})
            prompt_tokens = sum(len(m["content"].split()) for m in body["messages"])
            chunks, n = reply(body["messages"], body.get("max_tokens") or 16)
            with slots:
                if not body.get("stream"):
                    time.sleep(TOKEN_S * n)
                    return self._json({"choices": [{"message": {"role": "assistant", "content": "".join(chunks)},
                                                    "finish_reason": "stop"}],
                                       "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n}})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self._chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})
                for c in chunks:
                    time.sleep(TOKEN_S * (n if len(chunks) == 1 else 1))
                    self._chunk({"choices": [{"index": 0, "delta": {"content": c}}]})
                self._chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                self._chunk({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n}})
                self._chunk(b"[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--wrong-every", type=int, default=0)
    a = ap.parse_args()
    make_server(a.port, a.wrong_every)
    threading.Event().wait()
