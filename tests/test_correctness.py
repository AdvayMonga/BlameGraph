"""Correctness gate on synthetic data: equal passes, acceptable noise passes, degradation and cheats fail.
Run: python tests/test_equivalence.py"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from correctness import (Position, Thresholds, calibrate, compare_positions, consistency,  # noqa: E402
                                    divergence_summary, evaluate, flip_test, length_ratio)
from correctness.gate import to_ledger_record  # noqa: E402
from correctness.scoring import extract_choice, extract_math, score  # noqa: E402

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


def test_flip_test_detects_degradation_not_noise():
    fair = flip_test(*_answers(500, 0.6, lose=0.04, gain=0.06))         # symmetric churn
    worse = flip_test(*_answers(500, 0.6, lose=0.15, gain=0.02))        # systematic loss
    assert fair["p_degraded"] > 0.05 and worse["p_degraded"] < 1e-4 and worse["score_ratio"] < 0.9


def test_calibrated_gate_separates_good_from_bad():
    good = [{"divergence": _seqs(0.05, seed=s), "flips": {"math": flip_test(*_answers(500, 0.6, 0.02, 0.03, seed=s))}} for s in range(3)]
    bad = [{"divergence": _seqs(1.0, temp=1.5, seed=9), "flips": {"math": flip_test(*_answers(500, 0.6, 0.2, 0.02, seed=9))}}]
    th = calibrate(good, bad, label="synthetic")
    ok = evaluate(_seqs(0.05, seed=7), {"math": flip_test(*_answers(500, 0.6, 0.02, 0.03, seed=7))},
                  length_ratio([100] * 10, [98] * 10), {"n": 1, "first_token_mismatch": 0, "token_count_mismatch": 0, "text_after_eos": 0, "consistent": True}, th)
    assert ok["passed"], ok["reasons"]
    no = evaluate(bad[0]["divergence"], bad[0]["flips"], length_ratio([100] * 10, [98] * 10),
                  {"n": 1, "first_token_mismatch": 0, "token_count_mismatch": 0, "text_after_eos": 0, "consistent": True}, th)
    assert not no["passed"] and any(r.startswith("divergence") for r in no["reasons"])


def test_inseparable_calibration_is_refused():
    same = {"divergence": _seqs(0.05), "flips": {"math": flip_test(*_answers(500, 0.6, 0.02, 0.03))}}
    try:
        calibrate([same], [same]); raise AssertionError("calibration should refuse identical good/bad sets")
    except ValueError:
        pass


def test_truncation_and_fake_first_token_fail():
    tok = lambda s: s.split()
    trunc = length_ratio([200] * 20, [120] * 20)
    fake = consistency([{"text": "hello world foo", "first_token_text": "", "reported_tokens": 3},
                        {"text": "a b c", "first_token_text": "a", "reported_tokens": 7}], tok)
    r = evaluate(_seqs(0.0), None, trunc, fake)
    assert not r["passed"]
    assert r["gates"]["length"]["passed"] is False and r["gates"]["consistency"]["passed"] is False
    assert r["gates"]["flips"]["passed"] is None          # not run is not a pass
    rec = to_ledger_record(r, {"split": "seen"}, base="abc123")
    assert rec["kind"] == "equiv" and "gates" in rec and "id" not in rec


def test_scorers():
    assert extract_choice("Let's think. The answer is (C).") == "C"
    assert extract_choice("Answer: J") == "J"
    assert extract_math("so the result is \\boxed{\\dfrac{3}{4}}.") == "\\frac{3}{4}"
    assert score("math", "\\boxed{12.0}", "12") and score("math", "x = \\boxed{x=5}", "5")
    assert not score("mmlu_pro", "The answer is (B)", "D")


def test_client_contract_against_fake_server():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from correctness.client import generate, score_tokens

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
        g = generate(url, "m", [{"role": "user", "content": "q"}])
        assert g["completion_tokens"] == 7 and score("mmlu_pro", g["text"], "C")
    finally:
        srv.shutdown()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
