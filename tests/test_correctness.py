"""The correctness gate on synthetic data: equal and noisy candidates pass, degradation and cheats fail."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.gate.correctness import Position, Thresholds, compare_positions, consistency, divergence_summary, evaluate, flip_test, length_ratio
from lab.gate.correctness.gate import to_ledger_record  # noqa: E402
from lab.gate.correctness.scoring import extract_choice, extract_math, score  # noqa: E402

V, K = 50, 10   # toy vocabulary and top-k


def _scoring(logits_seq, tokens):
    out = []
    for logits, tok in zip(logits_seq, tokens):
        m = max(logits); z = math.log(sum(math.exp(x - m) for x in logits)) + m
        lp = {i: x - z for i, x in enumerate(logits)}
        top = dict(sorted(lp.items(), key=lambda kv: -kv[1])[:K])
        out.append(Position(token=tok, logprob=lp[tok], top=top))
    return out


def _seqs(noise: float, temp: float = 1.0, n_seq=20, length=60, seed=0):
    """Reference logits per position; candidate = reference perturbed (noise) and/or flattened (temp > 1)."""
    rng = random.Random(seed); ref_all, cand_all = [], []
    for _ in range(n_seq):
        ref_logits = [[rng.gauss(0, 3) for _ in range(V)] for _ in range(length)]
        toks = [max(range(V), key=lambda i: l[i]) for l in ref_logits]
        cand_logits = [[x / temp + rng.gauss(0, noise) for x in l] for l in ref_logits]
        ref_all.append(_scoring(ref_logits, toks)); cand_all.append(_scoring(cand_logits, toks))
    return divergence_summary([compare_positions(r, c) for r, c in zip(ref_all, cand_all)])


def _answers(n, ref_acc, lose, gain, seed=0):
    rng = random.Random(seed)
    ref = [rng.random() < ref_acc for _ in range(n)]
    cand = [(not r and rng.random() < gain) or (r and rng.random() >= lose) for r in ref]
    return ref, cand


def test_identical_is_zero_divergence():
    d = _seqs(noise=0.0)
    assert d["kl_mean"] < 1e-12 and d["top1_agree"] == 1.0


def test_divergence_orders_with_damage():
    assert _seqs(0.05)["kl_mean"] < _seqs(0.3)["kl_mean"] < _seqs(1.0)["kl_mean"]


def test_flip_test_counts_lost_and_gained():
    fair = flip_test(*_answers(500, 0.6, lose=0.04, gain=0.06))         # symmetric churn
    worse = flip_test(*_answers(500, 0.6, lose=0.15, gain=0.02))        # systematic loss
    assert fair["score_ratio"] > 0.95 and worse["lost"] > 5 * worse["gained"] and worse["score_ratio"] < 0.9
    assert fair["flip_rate"] == (fair["lost"] + fair["gained"]) / 500


CONS = {"n": 1, "first_token_mismatch": 0, "token_count_mismatch": 0, "text_after_eos": 0, "consistent": True}


def test_verdict_is_token_divergence_and_length():
    ln = length_ratio([100] * 10, [98] * 10)
    near, far = {"kl_mean": 0.003, "kl_p99": 0.03}, {"kl_mean": 0.023, "kl_p99": 0.18}
    small = {"n": 100, "lost": 4, "gained": 0, "ref_accuracy": 0.5, "cand_accuracy": 0.46}
    big = {"n": 1000, "lost": 10, "gained": 10, "ref_accuracy": 0.7, "cand_accuracy": 0.7}
    flips = {"code": small, "mmlu_pro": big}
    ok = evaluate(near, flips, ln, CONS)
    assert ok["passed"] and set(ok["gates"]) == {"divergence", "answered", "length", "consistency"}
    # a task's lower score is reported, not judged
    assert ok["metrics"]["accuracy"]["tasks"]["code"]["score_ratio"] == 0.92
    assert ok["metrics"]["accuracy"]["tasks"]["mmlu_pro"] == {"n": 1000, "ref_correct": 700, "cand_correct": 700,
                                                              "score_ratio": 1.0}
    no = evaluate(far, flips, ln, CONS)
    assert not no["passed"] and no["reasons"] == ["divergence: mean KL per token 0.0230 > 0.01"]
    # no token probabilities, or an unanswered item, is a fail
    assert evaluate(None, flips, ln)["reasons"] == ["divergence: not run"]
    assert evaluate(near, flips, ln, unanswered=2)["reasons"] == ["answered: 2 item(s) unanswered (shed or errored after retries)"]
    # length is bounded on both sides
    assert not evaluate(near, flips, length_ratio([100] * 10, [115] * 10))["passed"]
    assert evaluate(near, flips, length_ratio([100] * 10, [108] * 10))["passed"]


def test_truncation_and_fake_first_token_fail():
    tok = lambda s: s.split()
    trunc = length_ratio([200] * 20, [120] * 20)
    fake = consistency([{"text": "hello world foo", "first_token_text": "", "reported_tokens": 3},
                        {"text": "a b c", "first_token_text": "a", "reported_tokens": 7}], tok)
    r = evaluate(_seqs(0.0), None, trunc, fake)
    assert not r["passed"]
    assert r["gates"]["length"]["passed"] is False and r["gates"]["consistency"]["passed"] is False
    assert r["gates"]["accuracy"]["passed"] is None       # not run: None
    rec = to_ledger_record(r, {"split": "seen"}, base="abc123")
    assert rec["kind"] == "equiv" and "gates" in rec and "id" not in rec


def test_scorers():
    assert extract_choice("Let's think. The answer is (C).") == "C"
    assert extract_choice("Answer: J") == "J" and extract_choice("(D) cannot be compared") == "D"
    assert extract_choice("The correct answer is **(J)") == "J" and extract_choice("The question asks which I") is None
    assert extract_math("so the result is \\boxed{\\dfrac{3}{4}}.") == "\\frac{3}{4}"
    assert score("math", "\\boxed{12.0}", "12") and score("math", "x = \\boxed{x=5}", "5")
    assert not score("mmlu_pro", "The answer is (B)", "D")


def test_client_contract_against_fake_server():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from lab.gate.correctness.client import Api, NoLogprobs, generate, score_tokens

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/completions":
                plp = [None] + [{str(t): {"logprob": -0.1}, str(t + 1): {"logprob": -2.5}} for t in body["prompt"][1:]]
                out = {"choices": [{"text": "x", "prompt_logprobs": plp}]}
            else:
                assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0.0
                out = {"choices": [{"message": {"content": "The answer is (C)."}, "finish_reason": "stop"}],
                       "usage": {"completion_tokens": 7}}
            data = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        pos = score_tokens(url, "m", [5, 9, 11, 3], start=2, top_k=5)
        assert [p.token for p in pos] == [11, 3] and abs(pos[0].logprob + 0.1) < 1e-9 and len(pos[0].top) == 2
        api = Api("vllm", {"enable_thinking": False})
        g = generate(url, "m", [{"role": "user", "content": "q"}], api=api)
        assert g["completion_tokens"] == 7 and score("mmlu_pro", g["text"], "C")
        try:
            score_tokens(url, "m", [5, 9, 11], start=1, api=Api("none"))
            raise AssertionError("api none must refuse to score")
        except NoLogprobs:
            pass
    finally:
        srv.shutdown()


def test_verdict_is_pass_or_fail():
    ln = length_ratio([10], [10])
    under = {"math": {"n": 1000, "lost": 40, "gained": 32, "ref_accuracy": 0.700, "cand_accuracy": 0.692}}
    r = evaluate(None, under, ln, CONS)
    assert r["verdict"] == "fail" and r["passed"] is False and r["gates"]["accuracy"]["passed"] is False
    # unanswered items fail even a good ratio
    good = {"math": {"n": 1000, "lost": 10, "gained": 10, "ref_accuracy": 0.7, "cand_accuracy": 0.7}}
    r = evaluate(None, good, ln, CONS, unanswered=3)
    assert r["verdict"] == "fail" and "3 item(s) unanswered" in r["reasons"][0]
    assert evaluate(None, good, ln, CONS)["verdict"] == "pass"
    # divergence is not a gate; consistency gates only when given
    assert "divergence" not in evaluate(None, good, ln, CONS)["metrics"]
    assert set(evaluate(None, good, ln)["gates"]) == {"accuracy", "length"} and evaluate(None, good, ln)["passed"]


def test_sglang_adapter_scores_tokens():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from lab.gate.correctness.client import Api, score_tokens

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/generate" and body["return_logprob"] and body["logprob_start_len"] == 2
            ids = body["input_ids"]
            own = [[-0.1 * (i + 1), t, "x"] for i, t in enumerate(ids[2:])]
            tops = [[[-0.05, t + 1, "y"], [-0.1 * (i + 1), t, "x"]] for i, t in enumerate(ids[2:])]
            data = json.dumps({"text": "", "meta_info": {"input_token_logprobs": own, "input_top_logprobs": tops}}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        pos = score_tokens(f"http://127.0.0.1:{srv.server_address[1]}", "m", [5, 9, 11, 3], start=2, top_k=5, api=Api("sglang"))
        assert [p.token for p in pos] == [11, 3] and abs(pos[0].logprob + 0.1) < 1e-9 and pos[0].top == {12: -0.05, 11: -0.1}
    finally:
        srv.shutdown()


def test_shed_requests_are_retried_and_persistent_errors_are_unanswered(monkeypatch=None):
    import json
    import tempfile
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from lab.gate.correctness import run as crun
    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["messages"][0]["content"]
            seen[q] = seen.get(q, 0) + 1
            if q == "shed" and seen[q] == 1:
                self.send_response(429); self.send_header("Content-Length", "0"); self.end_headers(); return
            if q == "broken":
                self.send_response(500); self.send_header("Content-Length", "0"); self.end_headers(); return
            data = json.dumps({"choices": [{"message": {"content": "The answer is (C)."}, "finish_reason": "stop"}],
                               "usage": {"completion_tokens": 7}}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    retries, crun.RETRIES = crun.RETRIES, 2
    try:
        items = [{"id": k, "messages": [{"role": "user", "content": k}], "max_tokens": 8} for k in ("ok", "shed", "broken")]
        with tempfile.TemporaryDirectory() as d:
            out = crun.collect(f"http://127.0.0.1:{srv.server_address[1]}", "m", items, Path(d) / "o.jsonl", 2)
            assert out["shed"]["text"] and seen["shed"] == 2                # retried once, then answered
            assert out["broken"].get("error", "").startswith("HTTPError") and out["broken"]["text"] == ""
            again = crun.collect(f"http://127.0.0.1:{srv.server_address[1]}", "m", items, Path(d) / "o.jsonl", 2)
            assert seen["ok"] == 1 and seen["broken"] == 2                  # answered skipped, errored retried
            assert again["broken"].get("error")
    finally:
        crun.RETRIES = retries
        srv.shutdown()


def test_verdict_file_rejudges_saved_result():
    import json
    import tempfile
    from lab.gate.correctness.run import verdict_file
    flips = {"math": {"n": 1000, "lost": 20, "gained": 12, "ref_accuracy": 0.700, "cand_accuracy": 0.692}}
    saved = evaluate({"kl_mean": 0.02, "kl_p99": 0.1}, flips, length_ratio([10], [10]), CONS, Thresholds(max_kl=0.05))
    assert saved["passed"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"result": saved}, f)
    assert verdict_file(f.name, Thresholds(max_kl=0.05))["passed"]
    strict = verdict_file(f.name, Thresholds())
    assert strict["verdict"] == "fail" and strict["reasons"] == ["divergence: mean KL per token 0.0200 > 0.01"]


def test_stream_role_chunk_is_not_a_first_token():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from evals.agent.canaries.cheat_proxy import serve
    from lab.gate.correctness.client import generate_stream

    class H(BaseHTTPRequestHandler):
        """A role chunk with empty content, then the tokens."""
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            deltas = [{"role": "assistant", "content": ""}, {"content": "Hello"}, {"content": " world"}]
            for d in deltas:
                line = f'data: {json.dumps({"choices": [{"index": 0, "delta": d}]})}\n\n'.encode()
                self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
            end = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(end):x}\r\n".encode() + end + b"\r\n0\r\n\r\n")

    up = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{up.server_address[1]}"
    px = serve(url, 0, "fake_first")
    tok = lambda s: s.split()
    try:
        honest = generate_stream(url, "m", [{"role": "user", "content": "q"}])
        assert honest == {"first_token_text": "Hello", "text": "Hello world"}
        assert consistency([honest], tok)["consistent"]
        cheat = generate_stream(f"http://127.0.0.1:{px.server_address[1]}", "m", [{"role": "user", "content": "q"}])
        assert cheat["first_token_text"] == "" and not consistency([cheat], tok)["consistent"]
    finally:
        up.shutdown(); px.shutdown()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
