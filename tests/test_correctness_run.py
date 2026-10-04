"""End-to-end gate flow against fake servers: reference -> candidates -> calibrate -> verdicts.
A 'good' candidate (small symmetric churn, tiny logprob noise) must pass; a 'bad' one (systematic answer loss,
flattened distributions) must fail. Run: python tests/test_equiv_run.py"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from correctness import run  # noqa: E402
from correctness.gate import Thresholds  # noqa: E402


def h(*xs) -> int:
    return int(hashlib.md5("|".join(map(str, xs)).encode()).hexdigest(), 16)


class Enc:
    def encode(self, text):
        return [h(w) % 1000 + 1 for w in text.split()]

    def chat_ids(self, messages):
        return self.encode(" ".join(m["content"] for m in messages))


def items(n=300):
    out = []
    for k in range(n):
        if k % 2:
            out.append({"id": f"mc-{k}", "task": "mmlu_pro", "gold": "ABCD"[k % 4], "max_tokens": 64,
                        "messages": [{"role": "user", "content": f"question {k} pick one"}]})
        else:
            out.append({"id": f"m-{k}", "task": "math", "gold": str(k), "max_tokens": 64,
                        "messages": [{"role": "user", "content": f"problem {k} compute"}]})
    return out


def answer(item_text: str, mode: str) -> str:
    k = int(item_text.split()[1]); is_mc = item_text.startswith("question")
    right = h("ref", k) % 10 < 6
    if mode == "good" and h("churn", k) % 25 == 0:
        right = not right                                  # symmetric churn: both directions
    if mode == "bad" and right and h("loss", k) % 3 == 0:
        right = False                                      # systematic loss
    if is_mc:
        gold = "ABCD"[k % 4]; letter = gold if right else "ABCD"[(k + 1) % 4]
        return f"reasoning words here so the answer is ({letter})"
    return f"working it out step by step gives \\boxed{{{k if right else k + 1}}}"


def logprobs(ids, mode, top_k):
    plp = [None]
    for i in range(1, len(ids)):
        t = ids[i]
        base = -0.2 - 0.05 * (h(ids[i - 1], i) % 5)
        noise = {"ref": 0.0, "good": 0.01, "bad": 0.6}[mode] * ((h(mode, i, t) % 200) / 100 - 1)
        lp = base + noise
        alts = {t + j: lp - 1.5 * j * (0.4 if mode == "bad" else 1.0) for j in range(1, top_k)}
        plp.append({str(t): {"logprob": lp}, **{str(a): {"logprob": v} for a, v in alts.items()}})
    return plp


def server(mode):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/completions":
                return self._json({"choices": [{"text": "", "prompt_logprobs": logprobs(body["prompt"], mode, body["prompt_logprobs"])}]})
            text = answer(body["messages"][-1]["content"], mode)
            if not body.get("stream"):
                return self._json({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                   "usage": {"completion_tokens": len(text.split())}})
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            words = text.split(" ")
            chunks = [{"role": "assistant"}] + [{"content": (w if i == 0 else " " + w)} for i, w in enumerate(words)]
            for d in chunks:
                line = f'data: {json.dumps({"choices": [{"delta": d}]})}\n\n'.encode()
                self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
            end = b"data: [DONE]\n\n"; self.wfile.write(f"{len(end):x}\r\n".encode() + end + b"\r\n0\r\n\r\n")

        def _json(self, d):
            data = json.dumps(d).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_address[1]}"


def test_gate_flow():
    enc, its = Enc(), items()
    servers = {m: server(m) for m in ("ref", "good", "bad")}
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            meta = run.reference(servers["ref"][1], "m", d / "ref", its, enc, top_k=5, div_n=60, concurrency=8)
            assert meta["n_sequences"] == 60
            res = {}
            for name, mode in (("good1", "good"), ("bad1", "bad"), ("good2", "good")):
                res[name] = run.candidate(d / "ref", servers[mode][1], "m", d / f"{name}.json", enc, Thresholds(), concurrency=8)
            th = run.calibrate_files([str(d / "good1.json")], [str(d / "bad1.json")], str(d / "th.json"))
            good = run.candidate(d / "ref", servers["good"][1], "m", d / "good3.json", enc, th, concurrency=8)
            bad = run.candidate(d / "ref", servers["bad"][1], "m", d / "bad2.json", enc, th, concurrency=8)
            assert good["passed"], good["reasons"]
            assert not bad["passed"] and any("divergence" in r for r in bad["reasons"]) and any("flips:" in r for r in bad["reasons"]), bad["reasons"]
            rec = json.loads((d / "bad2.json").read_text())["ledger_record"]
            assert rec["kind"] == "equiv" and rec["gates"]["consistency"]["passed"] is True
    finally:
        for s, _ in servers.values():
            s.shutdown()


if __name__ == "__main__":
    test_gate_flow(); print("ok test_gate_flow")
