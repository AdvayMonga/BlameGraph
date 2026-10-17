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
    from kernels.search.loop import cascade, search
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
        tree, cost = search(SEED.read_text(), system_message(t), scripted,
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


def test_with_a_judge_each_step_is_plan_then_code_and_a_failed_child_gets_one_repair():
    from kernels.search.loop import search
    calls = []

    def review(system, user):
        mode = "correction" if "critical_issue" in system else "optimization"
        calls.append(("judge", mode, user))
        return json.dumps({"plan": f"{mode} plan"}), 1.0

    def propose(system, user):
        calls.append(("coder", system, user))
        return ("<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE" if "x = 1" in user else "nothing"), 0.5

    judged = []

    def judge(code):                             # x = 2 fails; nothing else is ever judged
        judged.append(code)
        ok = code == "x = 1\n"
        m = {"combined_score": 1.0 if ok else 0.0, "correct": 1.0 if ok else 0.0, "stage": 2}
        return {"metrics": m, "artifacts": {} if ok else {"error": "stage 1: wrong output"}}

    with tempfile.TemporaryDirectory() as d:
        tree, cost = search("x = 1\n", "SYSTEM", propose, judge, Path(d), evals=3, usd=4.0, review=review)
    modes = [c[1] for c in calls if c[0] == "judge"]
    assert modes[:2] == ["optimization", "correction"]                       # the failed child is repaired next
    coder = [c for c in calls if c[0] == "coder"]
    assert "optimization plan" in coder[0][2] and "SEARCH/REPLACE" in coder[0][1] and "SEARCH/REPLACE" not in \
        [c for c in calls if c[0] == "judge"][0][2]
    assert "stage 1: wrong output" in calls[2][2]                            # the judge sees why the child failed
    repaired = [n for n in tree.nodes.values() if n.mode == "correction"]
    assert repaired and repaired[0].parent == next(n.id for n in tree.nodes.values() if n.code == "x = 2\n")
    assert cost["usd"] <= 4.0 + 1.5 and cost["model_calls"] == 2 * (len(tree.nodes) - 1)


def test_the_search_stops_near_the_roofline_and_states_the_gap():
    from kernels.search.loop import prompt, search
    from kernels.search.tree import Node, Tree
    rows = [{"shape": [1], "weight": 0.75, "pct_of_roofline": 96.0}, {"shape": [2], "weight": 0.25,
                                                                      "pct_of_roofline": 80.0}]
    n = Node("a", None, "x = 1\n", {"combined_score": 1.0, "correct": 1.0}, {"timings": json.dumps(rows)}, 0)
    assert abs(n.pct_of_roofline - 92.0) < 1e-9
    t = Tree()
    t.add(n)
    assert "92.0% of its roofline bound" in prompt(t, n)
    judge = lambda code: {"metrics": n.metrics, "artifacts": n.artifacts}
    never = lambda system, user: (_ for _ in ()).throw(AssertionError("no model call near the roofline"))
    with tempfile.TemporaryDirectory() as d:
        _, cost = search("x = 1\n", "S", never, judge, Path(d), evals=5, usd=1.0, stop_pct=90.0)
    assert cost["stopped_near_roofline"] and cost["model_calls"] == 0


def test_ncu_counters_parse_and_a_host_without_ncu_says_so():
    from kernels.evolve import jailed, ncu
    from kernels.search.loop import cascade, facts
    from kernels.search.tree import Node
    head = ",".join(f'"{h}"' for h in ["ID", "Kernel Name", *ncu.METRICS])
    units = ",".join(['""', '""'] + ['"%"'] * len(ncu.METRICS))
    row = lambda i, name, dur: ",".join([f'"{i}"', f'"{name}"'] + ['"1,234.5"'] * (len(ncu.METRICS) - 1) + [f'"{dur}"'])
    out = "==PROF== Connected\n" + "\n".join([head, units, row(0, "small", 10), row(1, "big", 900)]) + "\n==PROF== done"
    p = ncu.parse(out)
    assert [k["name"] for k in p["kernels"]] == ["big", "small"] and p["kernels"][0]["dram__bytes_read.sum"] == 1234.5
    assert ncu.parse("no csv here")["kernels"] == []
    if __import__("shutil").which("ncu") is None:
        r = jailed.judge_jailed({"task": "toy_rmsnorm", "seen": [[8, 256]], "device": "cpu"}, "x = 1\n", 3)
        assert "ncu_missing" in r["artifacts"]
    stages = []

    def fake(path, stage):
        stages.append(stage)
        return {"metrics": {"combined_score": 1.2, "correct": 1.0}, "artifacts": {"ncu": json.dumps(p)} if stage == 3
                else {"timings": "[]"}}
    r = cascade(fake, counters=True)("x = 1\n")
    assert stages == [1, 2, 3] and "ncu" in r["artifacts"] and "timings" in r["artifacts"]
    assert "ncu counters" in facts(Node("n", None, "", r["metrics"], r["artifacts"], 0))


CHUNKED = """import torch

CONFIG = {"chunk": 1, "eps": 1e-6}
TUNABLES = {"chunk": [1, 16, 64]}


def kernel(x, w):
    out = torch.empty_like(x)
    c = CONFIG["chunk"]
    for i in range(0, x.shape[0], c):
        r = x[i:i + c].float()
        out[i:i + c] = (r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + CONFIG["eps"]) * w.float()).to(x.dtype)
    return out
"""


def test_the_grid_search_picks_launch_parameters_and_rewrites_them_as_constants():
    from kernels.evolve import evaluator
    from kernels.search import tune
    from kernels.search.__main__ import _tune
    from kernels.search.loop import cascade
    from kernels.tasks import load
    assert tune.params(CHUNKED) == ({"chunk": 1, "eps": 1e-6}, {"chunk": [1, 16, 64]})
    assert tune.params("CONFIG = {}\n") is None and tune.params("CONFIG = {'a': 1}\nTUNABLES = {'b': [2]}\n") is None
    moved = tune.with_config(CHUNKED, {"chunk": 64, "eps": 1e-6})
    assert "CONFIG = {'chunk': 64, 'eps': 1e-06}\n" in moved and moved.count("CONFIG =") == 1 and "TUNABLES" in moved
    g = tune.grid({"a": 1, "b": 1}, {"a": [1, 2], "b": [1, 2, 3]}, points=3)
    assert g[0] == {"a": 1, "b": 1} and len(g) == 3 and len({json.dumps(x, sort_keys=True) for x in g}) == 3
    t = load("toy_rmsnorm")
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "eval.json"
        cfg.write_text(json.dumps({"task": t.name, "seen": [[64, 256], [64, 512]], "device": "cpu", "iters": 5}))
        os.environ["KERNEL_EVAL_CONFIG"] = str(cfg)
        judge = cascade(evaluator.judge_stage)
        code, result, rows = tune.sweep(CHUNKED, judge, points=3)
        assert len(rows) == 3 and "'chunk': 1," not in code and result["metrics"]["correct"] == 1.0
        best = max(rows, key=lambda r: r["score"])
        assert best["config"]["chunk"] > 1 and best["score"] > rows[0]["score"]          # chunk 1 is the slow one
        out, metrics, rec = _tune(CHUNKED, {"combined_score": rows[0]["score"]}, judge, 3)
        assert rec["promoted"] and rec["median"] > rec["base_median"] and "'chunk': 1," not in out
        assert _tune("def kernel(x, w): ...\n", {}, judge, 3)[2] == {}                  # no grid declared


def test_the_ab_harness_runs_both_arms_and_gates_each_winner_on_held_out_shapes():
    from kernels.search import ab
    from kernels.tasks import load
    t = load("toy_rmsnorm")
    fast = SEED.read_text().replace(LOOP, "    xf = x.float()\n    return (xf * torch.rsqrt(xf.pow(2).mean(-1, "
                                         "keepdim=True) + 1e-6) * w.float()).to(x.dtype)")
    def arm(code):
        def run(a, out, seed):
            (out / "best_program.py").write_text(code)
            return {"ledger": None}
        return run
    saved = dict(ab.ARMS)
    ab.ARMS.clear()
    ab.ARMS.update({"openevolve": arm(SEED.read_text()), "kernels.search": arm(fast)})
    try:
        with tempfile.TemporaryDirectory() as d:
            shapes = Path(d) / "shapes.json"
            shapes.write_text(json.dumps({"seen": [list(x) for x in t.seen_default],
                                          "heldout": [list(x) for x in t.heldout_default]}))
            assert ab.main(["--task", t.name, "--shapes", str(shapes), "--out", str(Path(d) / "ab"), "--device", "cpu",
                            "--evals", "2"]) == 0
            rows = json.loads((Path(d) / "ab" / "ab.json").read_text())
    finally:
        ab.ARMS.clear()
        ab.ARMS.update(saved)
    by = {r["arm"]: r for r in rows}
    assert set(by) == {"openevolve", "kernels.search"} and all(r["gates_passed"] for r in rows)
    assert by["kernels.search"]["seen_score"] > 3 * by["openevolve"]["seen_score"]    # rsqrt vs the naive row loop
    assert by["kernels.search"]["heldout_speedup"] > 0


def test_the_openevolve_arm_gets_the_same_single_model_and_the_facts_only_prompts():
    from types import SimpleNamespace
    from kernels.evolve import run
    from kernels.search import ab
    seen = {}
    orig = run.main
    run.main = lambda argv: seen.update(argv=argv) or print('{"ledger": "x"}')
    try:
        with tempfile.TemporaryDirectory() as d:
            a = SimpleNamespace(model="claude-opus-5-5", task="moe_gemm", shapes="s.json", evals=30, device="cuda",
                                initial=None)
            assert ab._openevolve(a, Path(d), 1) == {"ledger": "x"}
            import yaml
            cfg = yaml.safe_load((Path(d) / "config.yaml").read_text())
    finally:
        run.main = orig
    assert cfg["llm"]["models"] == [{"name": "claude-opus-5-5", "weight": 1.0}] and cfg["random_seed"] == 1
    assert Path(cfg["prompt"]["template_dir"]).is_dir() and seen["argv"][seen["argv"].index("--iterations") + 1] == "29"


def test_openevolve_calls_are_priced_into_a_usage_log():
    from kernels.evolve.run import usage
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "usage.jsonl"
        log.write_text(json.dumps({"model": "claude-opus-5-5", "usd": 0.5}) + "\n" +
                       json.dumps({"model": "claude-opus-5-5", "usd": 0.25}) + "\n")
        assert usage(log) == {"usd": 0.75, "model_calls": 2} and usage(Path(d) / "none") == {"usd": 0, "model_calls": 0}


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
