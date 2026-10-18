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
import re  # noqa: E402

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


def test_mcts_descends_by_uct_widens_progressively_and_skips_closed_nodes():
    import math
    t = Tree(seed=0, max_failures=2)
    ok = lambda sc: {"combined_score": sc, "correct": 1.0, "stage": 2}        # noqa: E731
    root = t.add(Node("r", None, "x = 1", ok(1.0), {}, 0))
    assert t.select().id == "r" and root.visits == 1 and abs(root.value - 0.5) < 1e-9     # reward = score/root/2
    t.add(Node("a", "r", "x = 2", ok(1.8), {}, 1))
    t.add(Node("b", "r", "x = 3", ok(0.6), {}, 1))
    assert root.visits == 3 and len(root.children) == 2 > math.ceil(3 ** 0.5) - 1
    assert t.select().id == "a"                         # root may not widen (2 >= ceil(3^0.5)); UCT prefers `a`
    for i in range(2):
        t.add(Node(f"f{i}", "a", "", {"combined_score": 0.0, "correct": 0.0, "stage": 1}, {}, 2 + i))
    assert not t.expandable(t.nodes["a"]) and t.select().id in ("r", "b")      # `a` closed after 2 failed children
    assert [n.id for n in t.lineage(t.nodes["f0"])] == ["r", "a", "f0"]
    assert t.nodes["f0"].visits == 1 and t.nodes["a"].visits == 3
    t.add(Node("u", "b", "", {"combined_score": 0.0, "correct": 0.0, "stage": 0}, {"error": "no diff"}, 9))
    assert t.nodes["u"].visits == 0                     # never judged: no reward backed up
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


def test_an_expansion_ranks_many_hypotheses_measures_a_beam_and_repairs_failures_in_a_chain():
    from kernels.search.loop import choose, hypotheses, roi, search
    calls = []

    def review(system, user):
        if "critical_issue" in system:
            calls.append(("judge", "correction", user))
            return json.dumps({"critical_issue": "x"}), 1.0
        calls.append(("judge", "optimization", user))
        hs = [{"modification_plan": f"set x = {v}", "expected_speedup": sp, "implementation_risk": 1,
               "performance_risk": 1} for v, sp in ((2, 1.1), (3, 1.9), (9, 1.5), (4, 1.0))]
        return "Here: " + json.dumps(hs), 1.0

    def propose(system, user):
        calls.append(("coder", system, user))
        m = re.search(r"set x = (\d+)", user)
        if m:
            return f"<<<<<<< SEARCH\nx = 1\n=======\nx = {m.group(1)}\n>>>>>>> REPLACE", 0.5
        cur = re.search(r"```python\nx = (\d+)", user).group(1)                        # a repair: edit what is shown
        return f"<<<<<<< SEARCH\nx = {cur}\n=======\nx = {10 + len(calls)}\n>>>>>>> REPLACE", 0.5

    def judge(code):                             # x = 3 is the win; x = 9 (and every repair of it) fails
        x = int(code.split("=")[1])
        ok = x in (1, 2, 3, 4)
        m = {"combined_score": {1: 1.0, 2: 1.1, 3: 1.6, 4: 1.0}.get(x, 0.0), "correct": 1.0 if ok else 0.0, "stage": 2}
        return {"metrics": m, "artifacts": {} if ok else {"error": "stage 1: wrong output"}}

    assert len(hypotheses("pre " + json.dumps([{"a": 1}, {"b": 2}, 3]) + " post", 16)) == 2
    assert hypotheses("not json", 4) == [{"plan": "not json"}] and hypotheses("", 4) == []
    assert roi({"expected_speedup": 2.0}, 100.0) > roi({"expected_speedup": 2.0, "implementation_risk": 3}, 100.0)
    assert roi({"expected_speedup": 2.0, "implementation_risk": 1, "performance_risk": 1}, 10.0) > 2.0  # gap 10 > 5
    import random as _r
    picks = choose([0.1, 0.9, 0.5, 0.3], 3, 1, _r.Random(0))
    assert [i for i, rnd in picks if not rnd] == [1, 2] and sum(rnd for _, rnd in picks) == 1
    with tempfile.TemporaryDirectory() as d:
        tree, cost = search("x = 1\n", "SYSTEM", propose, judge, Path(d), evals=8, usd=100.0, review=review,
                            n_hypotheses=16, beam=3, explore=0, max_repairs=3)
        log = [json.loads(x) for x in (Path(d) / "hypotheses.jsonl").read_text().splitlines()]
    first = log[0]
    assert len(first["hypotheses"]) == 4 and [m["index"] for m in first["measured"]] == [1, 2, 0]   # ROI order
    coder = [c for c in calls if c[0] == "coder"]
    assert "SEARCH/REPLACE" in coder[0][1] and "set x = 3" in coder[0][2]
    assert "SEARCH/REPLACE" not in [c for c in calls if c[0] == "judge"][0][2]
    chain = [n for n in tree.nodes.values() if n.mode == "correction"]
    assert [n.repairs for n in sorted(chain, key=lambda n: n.repairs)] == [1, 2, 3]       # repaired up to 3 deep
    assert "stage 1: wrong output" in next(c for c in calls if c[1] == "correction")[2]
    assert "set x = 3" in tree.best().judge and tree.best().score == 1.6


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


def test_diagnosis_facts_bound_type_two_shape_counters_ranking_and_patience():
    from types import SimpleNamespace
    import torch
    from kernels.evolve.core import _roofline
    from kernels.search.loop import ncu_facts, search
    from kernels.search.tree import Node
    from kernels.share import rank
    from kernels.tasks import load
    t = load("toy_rmsnorm")                     # 4 FLOPs and ~8 bytes per element: memory-bound on any GPU
    hw = SimpleNamespace(bf16_flops=1e15, bw_bytes_s=4e12)
    x = t.make_inputs((1024, 4096), torch.float32, "cpu", torch.Generator().manual_seed(0))
    r = _roofline(t, (1024, 4096), torch.float32, 0.05, hw, [x])
    assert r["bound"] == "memory" and abs(r["pct_of_roofline"] - r["pct_of_peak_bandwidth"]) < 1e-9
    assert r["pct_of_peak_flops"] < r["pct_of_peak_bandwidth"] and _roofline(t, (8, 8), torch.float32, 1.0, None,
                                                                             [x]) == {"pct_of_roofline": None}
    two = {"shapes": [{"shape": [1, 256], "kernels": [{"name": "k_small"}]},
                      {"shape": [512, 256], "kernels": [{"name": "k_big"}]}]}
    f = ncu_facts(Node("n", None, "", {}, {"ncu": json.dumps(two)}, 0))
    assert "shape [1, 256]" in f and "k_big" in f
    with tempfile.TemporaryDirectory() as d:
        bundles = []
        for i, dur in enumerate((200_000.0, 100_000.0)):
            b = Path(d) / f"b{i}"
            b.mkdir()
            (b / "meta.json").write_text(json.dumps({"window": {"wall_s": 1.0}}))
            (b / "kernels.json").write_text(json.dumps({"kernels": [
                {"name": "_moe_gemm_kernel", "count": 1, "total_us": dur},
                {"name": "_gqa_decode_splitk_kernel", "count": 1, "total_us": 100_000.0}]}))
            bundles.append(b)
        gate = {"artifacts": {"timings": json.dumps([{"weight": 1.0, "pct_of_roofline": 90.0}])}}
        out = rank(bundles, [load("moe_gemm"), load("paged_decode")], {"paged_decode": gate})
    moe, pd = (next(r for r in out["tasks"] if r["task"] == n) for n in ("moe_gemm", "paged_decode"))
    assert moe["share_by_profile"] == [0.2, 0.1] and abs(moe["share"] - 0.15) < 1e-9 and moe["gain_at_roofline"] is None
    assert abs(pd["gain_at_roofline"] - 0.1 * 0.1) < 1e-9 and pd["pct_of_roofline"] == 90.0
    flat = {"metrics": {"combined_score": 1.0, "correct": 1.0}, "artifacts": {}}
    n = iter(range(100))
    with tempfile.TemporaryDirectory() as d:
        _, cost = search("x = 0\n", "S", lambda s, u: (f"<<<<<<< SEARCH\nx = 0\n=======\nx = {next(n) + 1}\n>>>>>>> "
                                                       "REPLACE", 0.0), lambda code: flat, Path(d), evals=50, usd=1.0,
                         patience=3)
    assert cost["stopped_no_progress"] and cost["evals"] <= 5                # no score ever beats the root's


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


def test_regions_pick_a_config_per_size_and_the_dispatcher_sets_it_per_call():
    import torch
    from kernels.evolve import evaluator
    from kernels.search import tune
    from kernels.search.loop import cascade
    from kernels.tasks import load
    small, big = {"chunk": 1, "eps": 1e-6}, {"chunk": 64, "eps": 1e-6}
    rows = [{"config": small, "score": 1.0, "timings": [{"shape": [4, 256], "speedup": 2.0, "weight": 0.5},
                                                         {"shape": [512, 256], "speedup": 0.5, "weight": 0.5}]},
            {"config": big, "score": 1.0, "timings": [{"shape": [4, 256], "speedup": 0.8, "weight": 0.5},
                                                       {"shape": [512, 256], "speedup": 3.0, "weight": 0.5}]}]
    table = tune.regions(rows)
    assert table == [(4, small), (None, big)]
    assert tune.regions([rows[0], {**rows[1], "timings": rows[0]["timings"]}]) is None       # one point wins everywhere
    t = load("toy_rmsnorm")
    for prog in (CHUNKED, "# EVOLVE-BLOCK-START\n" + CHUNKED.replace("def kernel(", "def entry(") +
                 "# EVOLVE-BLOCK-END\n\nkernel = entry\n"):
        code = tune.dispatch(prog, t.size_expr, table)
        ns = {}
        exec(compile(code, "<dispatched>", "exec"), ns)
        for rows_, want in ((4, 1), (512, 64)):
            x, w = t.make_inputs((rows_, 256), torch.float32, "cpu", torch.Generator().manual_seed(0))
            assert torch.allclose(ns["kernel"](x, w), t.reference(x, w), atol=1e-5) and ns["CONFIG"]["chunk"] == want
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "eval.json"
        cfg.write_text(json.dumps({"task": t.name, "seen": [[4, 256], [512, 256]], "device": "cpu", "iters": 3}))
        os.environ["KERNEL_EVAL_CONFIG"] = str(cfg)
        r = cascade(evaluator.judge_stage)(tune.dispatch(CHUNKED, t.size_expr, table))
    assert r["metrics"]["correct"] == 1.0, r["artifacts"]                  # passes the screen and every gate
    from kernels.evolve.ship import module_source
    marked = "# EVOLVE-BLOCK-START\n" + CHUNKED.replace("def kernel(", "def entry(") + "# EVOLVE-BLOCK-END\n\nkernel = entry\n"
    shipped = module_source(tune.dispatch(marked, t.size_expr, table), "entry")
    assert "_dispatch_impl = entry" in shipped and shipped.rstrip().endswith("entry = _dispatched")    # ships wrapped
    assert all(load(n).size_expr for n in ("paged_decode", "moe_gemm", "rmsnorm", "add_rmsnorm", "rmsnorm_rope",
                                           "silu_mul", "topk_softmax"))


def test_screeners_score_hypotheses_and_the_log_keeps_labels():
    from kernels.search import screen
    from kernels.search.loop import search
    hs = [{"modification_plan": "a", "expected_speedup": 1.5}, {"modification_plan": "b", "expected_speedup": 1.1}]
    assert screen.parse("ok: [0.2, 1.7]", 2) == [0.2, 1.0] and screen.parse("[0.1]", 2) is None
    seen = {}
    fake = lambda system, user: seen.update(system=system, user=user) or ("[0.1, 0.9]", 0.002)      # noqa: E731
    scores, usd = screen.haiku(fake)("TASK FACTS", "PROGRAM AND MEASUREMENTS", hs)
    assert scores == [0.1, 0.9] and usd == 0.002 and "PROGRAM AND MEASUREMENTS" in seen["user"]
    assert '1. {"modification_plan": "b"' in seen["user"] and "probability" in seen["system"]
    scores, _ = screen.haiku(lambda s, u: ("no idea", 0.0))("S", "F", hs)
    assert scores[0] > scores[1]                                   # unparsable: MANTIS ROI order instead
    assert screen.make("roi") is None and len(screen.make("random")("S", "F", hs)[0]) == 2

    def review(system, user):
        return json.dumps([{"modification_plan": f"set x = {v}", "expected_speedup": 1.0} for v in (2, 3, 4)]), 0.0

    def propose(system, user):
        v = re.search(r"set x = (\d+)", user).group(1)
        return f"<<<<<<< SEARCH\nx = 1\n=======\nx = {v}\n>>>>>>> REPLACE", 0.0

    def judge(code):
        x = int(code.split("=")[1])
        return {"metrics": {"combined_score": {1: 1.0, 2: 0.9, 3: 1.4, 4: 1.2}[x], "correct": 1.0, "stage": 2},
                "artifacts": {}}

    pick_third = lambda system, facts, hs: ([0.0, 0.1, 0.9][: len(hs)], 0.01)                          # noqa: E731
    with tempfile.TemporaryDirectory() as d:
        tree, cost = search("x = 1\n", "S", propose, judge, Path(d), evals=2, usd=1.0, review=review, beam=1,
                            explore=0, screen=pick_third)
        row = json.loads((Path(d) / "hypotheses.jsonl").read_text().splitlines()[0])
    assert row["measured"] == [{"index": 2, "random": False, "child_id": row["measured"][0]["child_id"],
                                "child_score": 1.2, "improved": True}] and row["parent_score"] == 1.0
    assert row["scores"] == [0.0, 0.1, 0.9] and cost["usd"] == 0.01


def test_the_judge_can_query_the_ledger_and_search_the_web_across_turns():
    from types import SimpleNamespace
    from kernels.search import llm
    turns, sent = iter(["tool_use", "pause_turn", "end_turn"]), []

    class Stream:
        def __init__(self, **kw):
            sent.append(kw)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            stop = next(turns)
            usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 1000, "output_tokens": 100,
                                                       "server_tool_use": {"web_search_requests": 2}})
            content = {"tool_use": [SimpleNamespace(type="tool_use", id="t1", name="query",
                                                    input={"sql": "SELECT 1"})],
                       "pause_turn": [SimpleNamespace(type="text", text="searching")],
                       "end_turn": [SimpleNamespace(type="text", text="[plan]")]}[stop]
            return SimpleNamespace(model="claude-opus-5-5", usage=usage, stop_reason=stop, content=content)

    c = llm.Claude.__new__(llm.Claude)
    c.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: Stream(**kw))))
    c.model, c.effort, c.max_tokens, c.tools, c.max_turns, c.ledger_root = "claude-opus-5-5", "high", 1000, True, 8, None
    orig = llm.run_query
    llm.run_query = lambda sql, root=None: json.dumps({"columns": ["x"], "rows": [[1]], "sql": sql})
    try:
        text, usd = c("S", "U")
    finally:
        llm.run_query = orig
    assert text == "[plan]" and len(sent) == 3 and {t["name"] for t in sent[0]["tools"]} == {"web_search", "query"}
    results = next(m["content"] for m in sent[-1]["messages"] if m["role"] == "user" and isinstance(m["content"], list))
    assert results[0]["type"] == "tool_result" and "SELECT 1" in results[0]["content"]
    assert abs(usd - 3 * (1000 * 4 + 100 * 20) / 1e6 - 3 * 2 * llm.WEB_SEARCH_USD) < 1e-9


def test_inspirations_show_other_branches_and_a_merge_competes_for_the_beam():
    from kernels.search.loop import merge_hypothesis, prompt
    t = Tree()
    ok = lambda sc: {"combined_score": sc, "correct": 1.0, "stage": 2}        # noqa: E731
    t.add(Node("r", None, "a = 1\nb = 1\n", ok(1.0), {}, 0))
    t.add(Node("x", "r", "a = 2\nb = 1\n", ok(1.5), {}, 1))
    t.add(Node("y", "r", "a = 1\nb = 2\n", ok(1.3), {}, 1))
    m = merge_hypothesis(t, t.nodes["y"])
    assert m["merge_from"] == "x" and m["expected_speedup"] == 1.5
    judge_view = prompt(t, t.nodes["y"], coder=False)
    assert "# Other programs in the tree" in judge_view and "+a = 2" in judge_view and "Program r" not in \
        judge_view.split("# Other programs")[1]
    assert "+a = 2" in prompt(t, t.nodes["y"], json.dumps(m)) and "+a = 2" not in prompt(t, t.nodes["y"], '{"p": 1}')
    assert merge_hypothesis(t, t.nodes["x"])["merge_from"] == "y"


def test_a_flagged_candidate_is_passed_over_at_promotion():
    from kernels.evolve.run import retime
    from kernels.search.detect import detector
    from kernels.tasks import load
    spent = []
    flag = detector(lambda s, u: (('{"gaming": true, "evidence": "if x.shape[0] == 64"}' if "SPECIAL" in u
                                   else '{"gaming": false, "evidence": ""}'), 0.003), spent)
    assert flag("x = 1  # SPECIAL\n", {}) == "flagged as gaming: if x.shape[0] == 64"
    assert flag("x = 1\n", {}) is None and spent == [0.003, 0.003]
    t = load("toy_rmsnorm")
    fast = SEED.read_text().replace(LOOP, "    xf = x.float()\n    return (xf * torch.rsqrt(xf.pow(2).mean(-1, "
                                         "keepdim=True) + 1e-6) * w.float()).to(x.dtype)")
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "eval.json"
        cfg.write_text(json.dumps({"task": t.name, "seen": [[8, 256]], "device": "cpu", "iters": 3}))
        os.environ["KERNEL_EVAL_CONFIG"] = str(cfg)
        trace = Path(d) / "trace.jsonl"
        rows = [{"child_id": "gamer", "child_code": fast + "# SPECIAL\n", "child_metrics": {"combined_score": 9.0}},
                {"child_id": "honest", "child_code": fast, "child_metrics": {"combined_score": 2.0}}]
        trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        code, metrics, timed = retime(trace, 1, exclude=flag)
        assert "SPECIAL" not in code and metrics["correct"] == 1.0
        assert timed[0] == {"id": "gamer", "selected_score": 9.0, "excluded": "flagged as gaming: if x.shape[0] == 64"}
        assert timed[1]["id"] == "honest" and len(timed[1]["scores"]) == 3
        assert retime(trace, 1, exclude=lambda c, r: "no")[0] is None                     # all flagged: none promoted


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
    c.model, c.effort, c.max_tokens, c.tools, c.max_turns, c.ledger_root = "claude-opus-5-5", "high", 64000, False, 8, \
        None
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
