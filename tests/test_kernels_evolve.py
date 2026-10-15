"""The layer-1 loop end to end on CPU, with no model calls: tasks, shape capture and split, the judge, and a real
OpenEvolve run whose "model" is a scripted stand-in that returns one known-good diff and one cheating diff.
Run: python tests/test_kernels_evolve.py (skips without openevolve)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from kernels import shapes  # noqa: E402
from kernels.evolve.core import judge  # noqa: E402
from kernels.tasks import load  # noqa: E402
from lab import capture  # noqa: E402

SEED = ROOT / "kernels" / "evolve" / "seeds" / "toy_rmsnorm.py"
FAST = '''<<<<<<< SEARCH
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        r = x[i].float()
        out[i] = (r / torch.sqrt((r * r).mean() + 1e-6) * w.float()).to(x.dtype)
    return out
=======
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)
>>>>>>> REPLACE'''
CHEAT = '''<<<<<<< SEARCH
    out = torch.empty_like(x)
=======
    return x.clone()
    out = torch.empty_like(x)
>>>>>>> REPLACE'''


def test_judge_scores_a_correct_kernel_and_zeroes_a_wrong_one():
    t = load("toy_rmsnorm")
    ok = judge(t, SEED, list(t.seen_default), [], None, "cpu", None, 2, iters=3)
    assert ok["metrics"]["correct"] == 1.0 and 0 < ok["metrics"]["combined_score"]
    assert json.loads(ok["artifacts"]["timings"])[0]["baseline_ms"] > 0
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.py"
        bad.write_text("def kernel(x, w):\n    return x.clone()\n")
        r = judge(t, bad, list(t.seen_default), [], None, "cpu", None, 1)
        assert r["metrics"]["combined_score"] == 0.0 and "correctness_seen" in r["artifacts"]["gates"]
        missing = Path(d) / "none.py"
        missing.write_text("x = 1\n")
        assert "no callable `kernel`" in judge(t, missing, list(t.seen_default), [], None, "cpu", None, 1)["artifacts"]["error"]


def test_paged_decode_reference_matches_dense_attention():
    t = load("paged_decode")
    shape = (3, 8, 2, 16, 4, 11)
    q, k, v, table, lens = t.make_inputs(shape, torch.float32, "cpu", torch.Generator().manual_seed(0))
    out = t.reference(q, k, v, table, lens)
    i = 1
    n = int(lens[i])
    kk = k[table[i].long()].permute(1, 0, 2, 3).reshape(2, -1, 16)[:, :n].repeat_interleave(4, 0)
    vv = v[table[i].long()].permute(1, 0, 2, 3).reshape(2, -1, 16)[:, :n].repeat_interleave(4, 0)
    dense = torch.nn.functional.scaled_dot_product_attention(q[i][:, None], kk, vv).squeeze(1)
    assert torch.allclose(out[i], dense, atol=1e-5) and out.shape == (3, 8, 16) and int(lens[0]) == 11
    flops, nbytes = t.cost(shape, torch.bfloat16)
    assert flops > 0 and nbytes > 0


def test_capture_records_calls_and_the_split_holds_out_the_largest():
    t = load("paged_decode")
    import types
    mod = types.ModuleType("fake_engine_kernels")

    def paged_decode_attention(q, k_pool, v_pool, block_tables, seq_lens, scale=1.0):
        return q
    mod.paged_decode_attention = paged_decode_attention
    sys.modules["fake_engine_kernels"] = mod
    user = types.ModuleType("fake_engine_model")
    user.paged_decode_attention = paged_decode_attention           # a `from x import f` copy
    sys.modules["fake_engine_model"] = user
    with tempfile.TemporaryDirectory() as d:
        with capture.capture(["fake_engine_kernels:paged_decode_attention"], Path(d)):
            for n in (1, 1, 2, 4, 8, 16, 32, 64, 128, 256):
                q = torch.zeros(n, 32, 128)
                pool = torch.zeros(4, 4, 16, 128)
                user.paged_decode_attention(q, pool, pool, torch.zeros(n, 2, dtype=torch.int32),
                                            torch.full((n,), 17 * n, dtype=torch.int32), scale=0.1)
        assert user.paged_decode_attention is paged_decode_attention          # restored
        calls = capture.load(Path(d))
        assert len(calls) == 10 and calls[0]["kwargs"]["scale"] == {"value": 0.1} and calls[0]["sample"]
        task = type(t)(**{**t.__dict__, "entry": "fake_engine_kernels:paged_decode_attention"})
        counts = shapes.from_captures(task, [Path(d)])
        assert counts[(1, 32, 4, 128, 16, 17)] == 2
        s = shapes.split(task, counts)
        assert s["source"] == "captures" and (256, 32, 4, 128, 16, 17 * 256) in s["heldout"]
        assert max(x[0] for x in s["seen"]) < min(x[0] for x in s["heldout"])
        back = shapes.read(shapes.write(Path(d) / "s.json", s))
        assert back["seen"] == s["seen"]


def _scripted_class():
    """A stand-in model, at module level so OpenEvolve's worker processes can unpickle it: odd calls (counted in a
    file, since every worker has its own instance) return the cheat, even calls the real speedup."""
    from openevolve.llm.base import LLMInterface

    class Scripted(LLMInterface):
        def __init__(self, cfg=None):
            self.model = "scripted"          # the ensemble logs vars(model)["model"]

        async def generate(self, prompt, **kw):
            return await self.generate_with_context("", [{"role": "user", "content": prompt}])

        async def generate_with_context(self, system_message, messages, **kw):
            p = Path(os.environ["SCRIPTED_COUNTER"])
            n = int(p.read_text()) + 1 if p.exists() else 1
            p.write_text(str(n))
            return CHEAT if n % 2 else FAST
    return Scripted


try:
    Scripted = _scripted_class()
    Scripted.__qualname__ = "Scripted"
except ImportError:
    Scripted = None


def test_openevolve_finds_the_known_good_edit_and_rejects_the_cheat():
    try:
        from openevolve import run_evolution
        from openevolve.config import Config, LLMModelConfig
    except ImportError:
        print("skip: openevolve not installed")
        return

    with tempfile.TemporaryDirectory() as d:
        os.environ["SCRIPTED_COUNTER"] = str(Path(d) / "calls")
        t = load("toy_rmsnorm")
        cfg_file = Path(d) / "eval.json"
        cfg_file.write_text(json.dumps({"task": t.name, "seen": [list(s) for s in t.seen_default],
                                        "device": "cpu", "iters": 3}))
        os.environ["KERNEL_EVAL_CONFIG"] = str(cfg_file)
        c = Config()
        c.llm.models = [LLMModelConfig(name="scripted", init_client=Scripted, weight=1.0)]
        c.llm.update_model_params({"init_client": Scripted})
        c.evaluator.cascade_evaluation, c.evaluator.cascade_thresholds = True, [0.5]
        c.evaluator.parallel_evaluations = 1
        c.database.num_islands, c.database.in_memory = 1, True
        res = run_evolution(str(SEED), str(ROOT / "kernels" / "evolve" / "evaluator.py"), c, iterations=4,
                            output_dir=str(Path(d) / "oe"), cleanup=False)
        assert "rsqrt" in res.best_code and "return x.clone()" not in res.best_code
        seed = judge(t, SEED, list(t.seen_default), [], None, "cpu", None, 2, iters=3)["metrics"]["combined_score"]
        assert res.metrics.get("correct") == 1.0 and res.best_score > 3 * seed     # the naive seed is ~0.1x
        print("best", res.best_score)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
