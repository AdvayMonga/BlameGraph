"""kernels.search on CPU with no model calls: the SEARCH/REPLACE parser, the tree's selection and caps, a whole search
on the toy task with a scripted model and the real (jailed) evaluator, and the Claude client's request against a fake
SDK client. Run: python tests/test_kernels_search.py"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LAB_TARGET", str(ROOT / "targets" / "inference-server.toml"))
os.environ.setdefault("LAB_ENGINE_PYTHON", sys.executable)
os.environ.setdefault("LAB_NO_JAIL", "1")

from kernels.search import diff  # noqa: E402
from kernels.search.tree import Node, Tree, fingerprint  # noqa: E402

SEED = ROOT / "kernels" / "evolve" / "seeds" / "toy_rmsnorm.py"
LOOP = '''    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        r = x[i].float()
        out[i] = (r / torch.sqrt((r * r).mean() + 1e-6) * w.float()).to(x.dtype)
    return out'''
FAST = f'''<<<<<<< SEARCH
{LOOP}
=======
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)
>>>>>>> REPLACE'''
WRONG = '''<<<<<<< SEARCH
    out = torch.empty_like(x)
=======
    return x.clone()
    out = torch.empty_like(x)
>>>>>>> REPLACE'''


def test_diff_applies_exact_blocks_and_refuses_the_rest():
    code = "a = 1\nb = 2\nb = 2\n"
    new, why = diff.apply(code, "<<<<<<< SEARCH\na = 1\n=======\na = 3\n>>>>>>> REPLACE")
    assert new == "a = 3\nb = 2\nb = 2\n" and why == ""
    assert diff.apply(code, "no blocks")[0] is None
    assert "2 times" in diff.apply(code, "<<<<<<< SEARCH\nb = 2\n=======\nb = 4\n>>>>>>> REPLACE")[1]
    assert "0 times" in diff.apply(code, "<<<<<<< SEARCH\nc = 1\n=======\nc = 4\n>>>>>>> REPLACE")[1]


def test_tree_selects_open_nodes_and_closes_failing_ones():
    t = Tree(seed=0, epsilon=0.0, max_children=5, max_failures=2)
    ok = {"combined_score": 1.0, "correct": 1.0}
    root = t.add(Node("r", None, "x = 1", ok, {}, 0))
    t.add(Node("a", "r", "x = 2", {"combined_score": 2.0, "correct": 1.0}, {}, 1))
    assert t.select().id == "a" and t.best().id == "a"
    for i in range(2):
        t.add(Node(f"f{i}", "a", "", {"combined_score": 0.0, "correct": 0.0}, {}, 2 + i))
    assert t.select().id == "r"                          # `a` closed after 2 failed children
    assert [n.id for n in t.lineage(t.nodes["f0"])] == ["r", "a", "f0"] and root.children == ["a"]
    assert fingerprint("x = 1  # note\n") == fingerprint("x=1\n")


def test_a_search_on_the_toy_task_finds_the_speedup_and_keeps_every_node():
    from kernels.evolve import evaluator
    from kernels.evolve.run import record, retime, system_message
    from kernels.search.loop import FORMAT, cascade, search
    from kernels.tasks import load
    from lab import ledger
    t = load("toy_rmsnorm")
    naive = iter([WRONG, "no diff here", FAST])                         # answers when shown the naive seed
    comment = "<<<<<<< SEARCH\n    xf = x.float()\n=======\n    # same\n    xf = x.float()\n>>>>>>> REPLACE"
    prompts = []

    def scripted(system, user):                     # a comment-only edit, or FAST again, is a duplicate
        prompts.append((system, user))
        return (next(naive, FAST) if "for i in range" in user else comment), 0.25

    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "eval.json"
        cfg.write_text(json.dumps({"task": t.name, "seen": [list(s) for s in t.seen_default], "device": "cpu",
                                   "iters": 3}))
        os.environ["KERNEL_EVAL_CONFIG"] = str(cfg)
        out = Path(d) / "run"
        tree, cost = search(SEED.read_text(), system_message(t) + "\n\n" + FORMAT, scripted,
                            cascade(evaluator.judge_stage), out, evals=10, usd=100.0)
        best = tree.best()
        root = next(n for n in tree.nodes.values() if n.parent is None)    # the naive seed is ~0.1x the baseline
        assert "rsqrt" in best.code and best.metrics["correct"] == 1.0 and best.score > 3 * root.score
        assert cost["evals"] == 3 and cost["usd"] == 0.25 * cost["model_calls"]    # stopped: no node left open
        assert tree.select() is None
        errors = [n.artifacts.get("error", "") for n in tree.nodes.values()]
        assert any("no SEARCH/REPLACE" in e for e in errors) and any("identical to a program" in e for e in errors)
        system, first = prompts[0]
        assert "How a program is judged" in system and "SEARCH/REPLACE" in system
        assert "# Its measurements" in first and "timings (per seen shape" in first and "consider" not in first.lower()
        assert "Programs already derived from it\n- " in prompts[1][1]           # the failed child is shown as a fact
        rows = [json.loads(x) for x in (out / "trace.jsonl").read_text().splitlines()]
        assert len(rows) == len(tree.nodes) and rows[0]["parent_id"] is None
        code, metrics, retimed = retime(out / "trace.jsonl", 2)
        assert "rsqrt" in code and all(len(x["scores"]) == 3 for x in retimed)
        best_file = out / "best.py"
        best_file.write_text(code)
        rec = record(t, {"seen": list(t.seen_default)}, [{"name": "scripted", "weight": 1.0}], 4, "cpu",
                     out / "trace.jsonl", best_file, metrics, root=Path(d) / "ledger", retimed=retimed,
                     tool="kernels.search", cost=cost)
        stored = list(ledger.records(Path(d) / "ledger", kind="kernels"))
        assert stored[0]["tool"] == "kernels.search" and stored[0]["cost"] == cost
        assert len(rec["result"]["candidates"]) == len(tree.nodes)


def test_the_claude_client_streams_with_effort_and_fallbacks_and_prices_the_call():
    from types import SimpleNamespace
    from kernels.search.llm import FALLBACK_BETA, Claude
    seen = {}

    class Stream:
        def __init__(self, **kw):
            seen.update(kw)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 1_000_000, "output_tokens": 100_000})
            return SimpleNamespace(model="claude-opus-5-5", usage=usage, stop_reason="end_turn",
                                   content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="hi")])

    c = Claude.__new__(Claude)
    c.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: Stream(**kw))))
    c.model, c.effort, c.max_tokens = "claude-opus-5-5", "high", 64000
    text, usd = c("sys", "user")
    assert text == "hi" and abs(usd - (4.0 + 2.0)) < 1e-9                 # $4/MTok in + $20/MTok out
    assert seen["output_config"] == {"effort": "high"} and seen["fallbacks"] == "default"
    assert seen["betas"] == [FALLBACK_BETA] and "thinking" not in seen
    seen.clear()
    c.model = "claude-haiku-4-5"
    c("sys", "user")
    assert "output_config" not in seen and "fallbacks" not in seen


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
