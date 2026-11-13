"""generate.kernels.search on CPU with no model calls: the SEARCH/REPLACE parser, a whole search
on the toy task with a scripted model and the real (jailed) evaluator, and the Claude client's request against a fake
SDK client. Run: python tests/test_kernels_search.py"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LAB_TARGET", str(ROOT / "targets" / "inference-server.toml"))
os.environ.setdefault("LAB_ENGINE_PYTHON", sys.executable)
os.environ.setdefault("LAB_NO_JAIL", "1")

from generate.kernels.search import diff  # noqa: E402

SEED = ROOT / "generate" / "kernels" / "evolve" / "seeds" / "toy_rmsnorm.py"
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
    spaced = "def f():\n    x = 1   \n    return x\n"                    # trailing spaces the model did not copy
    assert diff.apply(spaced, "<<<<<<< SEARCH\n    x = 1\n    return x\n=======\n    return 2\n>>>>>>> REPLACE") == \
        ("def f():\n    return 2\n", "")


def test_the_search_arm_gets_the_one_model_and_openevolves_promotion():
    from types import SimpleNamespace
    from generate.kernels.search import ab
    from generate.kernels.search import __main__ as cli
    seen, orig = [], cli.main
    cli.main = lambda argv, **kw: seen.append((argv, kw)) or print('{"ledger": null}')
    try:
        ab.ARMS["kernels.search"](SimpleNamespace(task="t", shapes="s", evals=3, model="m", device="cpu", initial=None),
                                  Path("/tmp/x"), 0)
    finally:
        cli.main = orig
    assert seen[0][1] == {"models": ("m",), "retime_k": 3, "detector": None} and set(ab.ARMS) == {"openevolve",
                                                                                                 "kernels.search"}
    try:
        ab.main(["--task", "t", "--shapes", "s", "--out", "/tmp/x", "--arms", "nope"])
        raise AssertionError("an unknown arm ran")
    except SystemExit:
        pass


def test_share_with_gates_ranks_only_the_gated_tasks():
    from generate.kernels.tasks import share
    from generate.kernels.evolve import run
    seen, orig = {}, share.rank
    share.rank = lambda bundles, tasks, gates: seen.update(tasks=[t.name for t in tasks]) or {"tasks": []}
    try:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "g.json").write_text("{}")
            run.main(["share", "--profile", d, "--gate", f"paged_decode={d}/g.json"])
            assert seen["tasks"] == ["paged_decode"]               # not every task naming the same kernel
            run.main(["share", "--profile", d])
            assert len(seen["tasks"]) > 1
    finally:
        share.rank = orig


def test_a_search_on_the_toy_task_finds_the_speedup_and_keeps_every_program():
    from generate.kernels.evolve.run import record, retime, system_message
    from generate.kernels.search.diversity import Novelty
    from generate.kernels.search.evaluate import Evaluator
    from generate.kernels.search.loop import search
    from generate.kernels.search.program import Config
    from generate.kernels.tasks import load
    from data.knowledge import ledger
    t = load("toy_rmsnorm")
    naive = iter([WRONG, "no diff here", FAST])                         # answers when shown the naive seed
    comment = "<<<<<<< SEARCH\n    xf = x.float()\n=======\n    # same\n    xf = x.float()\n>>>>>>> REPLACE"
    prompts = []

    def scripted(system, user):                     # a comment-only edit is a duplicate
        prompts.append((system, user))
        return (next(naive, FAST) if "for i in range" in user.split("# The starting program")[0] else comment), 0.25

    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "eval.json"
        cfg.write_text(json.dumps({"task": t.name, "seen": [list(s) for s in t.seen_default], "device": "cpu",
                                   "iters": 3}))
        os.environ["KERNEL_EVAL_CONFIG"] = str(cfg)
        out = Path(d) / "run"
        pop, cost = search(SEED.read_text(), {}, system_message(t), [scripted], Evaluator(), Novelty(), out,
                           Config(generations=2, ideas=3, parents=1, workers=1))
        root, best = pop[0], max(pop, key=lambda p: p.score)             # the naive seed is ~0.1x the baseline
        assert "rsqrt" in best.code and best.metrics["correct"] == 1.0 and best.score > 3 * root.score
        assert cost == {"usd": 1.5, "model_calls": 6, "evals": 2, "generations": 2} and len(pop) == 2
        fates = [x["fate"] for line in (out / "hypotheses.jsonl").read_text().splitlines()
                 for x in json.loads(line)["ideas"]]
        assert fates[0] == "failed correctness" and "no SEARCH/REPLACE" in fates[1] and fates[2] == "full exam"
        assert all("identical to a program" in f for f in fates[3:])
        system, first = prompts[0]
        assert "How a program is judged" in system and "SEARCH/REPLACE" in system
        assert "# Its measurements" in first and "timings (per seen shape" in first and "consider" not in first.lower()
        assert "# The starting program" in prompts[3][1] and f"-> {best.id}" in prompts[3][1]
        rows = [json.loads(x) for x in (out / "trace.jsonl").read_text().splitlines()]
        assert len(rows) == len(pop) and rows[0]["parent_id"] is None
        code, metrics, retimed = retime(out / "trace.jsonl", 2)
        assert "rsqrt" in code and all(len(x["scores"]) == 3 for x in retimed)
        best_file = out / "best.py"
        best_file.write_text(code)
        rec = record(t, {"seen": list(t.seen_default)}, [{"name": "scripted", "weight": 1.0}], 4, "cpu",
                     out / "trace.jsonl", best_file, metrics, root=Path(d) / "ledger", retimed=retimed,
                     tool="kernels.search", cost=cost)
        stored = list(ledger.records(Path(d) / "ledger", kind="kernels"))
        assert stored[0]["tool"] == "kernels.search" and stored[0]["cost"] == cost
        assert len(rec["result"]["candidates"]) == len(pop)


def test_diagnosis_facts_bound_type_two_shape_counters_and_ranking():
    from types import SimpleNamespace
    import torch
    from evals.kernels.core import _roofline
    from generate.kernels.search import context
    from generate.kernels.search.program import Program
    from generate.kernels.tasks.share import rank
    from generate.kernels.tasks import load
    t = load("toy_rmsnorm")                     # 4 FLOPs and ~8 bytes per element: memory-bound on any GPU
    hw = SimpleNamespace(bf16_flops=1e15, bw_bytes_s=4e12)
    x = t.make_inputs((1024, 4096), torch.float32, "cpu", torch.Generator().manual_seed(0))
    r = _roofline(t, (1024, 4096), torch.float32, 0.05, hw, [x])
    assert r["bound"] == "memory" and abs(r["pct_of_roofline"] - r["pct_of_peak_bandwidth"]) < 1e-9
    assert r["pct_of_peak_flops"] < r["pct_of_peak_bandwidth"] and _roofline(t, (8, 8), torch.float32, 1.0, None,
                                                                             [x]) == {"pct_of_roofline": None}
    two = {"shapes": [{"shape": [1, 256], "kernels": [{"name": "k_small"}]},
                      {"shape": [512, 256], "kernels": [{"name": "k_big"}]}]}
    f = context.counters(Program("n", None, "", {}, {"ncu": json.dumps(two)}, 0))
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


def test_ncu_counters_parse_and_a_host_without_ncu_says_so():
    from evals.kernels import jailed, ncu
    from generate.kernels.search import context
    from generate.kernels.search.evaluate import Evaluator
    from generate.kernels.search.program import Program
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
    program = Program("n", None, "x = 1\n", {}, {"timings": "[]"}, 0)
    Evaluator(lambda code, stage: {"metrics": {}, "artifacts": {"ncu": json.dumps(p)}}, counters=True).profile(program)
    assert "ncu counters" in context.facts(program) and "big" in context.facts(program)


def test_the_ab_harness_runs_both_arms_and_gates_each_winner_on_held_out_shapes():
    from generate.kernels.search import ab
    from generate.kernels.tasks import load
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
                            "--evals", "2", "--arms", "openevolve,kernels.search"]) == 0
            rows = json.loads((Path(d) / "ab" / "ab.json").read_text())
            assert json.loads((Path(d) / "ab" / "kernels.search-0" / "gate.json").read_text())["passed"]
            ab.ARMS.update({"openevolve": None, "kernels.search": None})      # finished arms are reused, not rerun
            env = {"KERNEL_EVAL_SSH": json.dumps(["sh", "-c"]),                  # the gate over a stand-in for ssh
                   "KERNEL_EVAL_REMOTE": f"cd {Path(__file__).parent.parent} && PATH={Path(sys.executable).parent}:$PATH"}
            with mock.patch.dict(os.environ, env):
                assert ab.main(["--task", t.name, "--shapes", str(shapes), "--out", str(Path(d) / "ab"),
                                "--device", "cpu", "--evals", "2", "--arms", "openevolve,kernels.search"]) == 0
            again = json.loads((Path(d) / "ab" / "ab.json").read_text())
    finally:
        ab.ARMS.clear()
        ab.ARMS.update(saved)
    assert [r["seen_score"] > 0 and r["gates_passed"] for r in again] == [True, True]
    by = {r["arm"]: r for r in rows}
    assert set(by) == {"openevolve", "kernels.search"} and all(r["gates_passed"] for r in rows)
    assert by["kernels.search"]["seen_score"] > 3 * by["openevolve"]["seen_score"]    # rsqrt vs the naive row loop
    assert by["kernels.search"]["heldout_speedup"] > 0


def test_the_openevolve_arm_gets_the_same_single_model_and_the_facts_only_prompts():
    from types import SimpleNamespace
    from generate.kernels.evolve import run
    from generate.kernels.search import ab
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
    from generate.kernels.evolve.run import usage
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "usage.jsonl"
        log.write_text(json.dumps({"model": "claude-opus-5-5", "usd": 0.5}) + "\n" +
                       json.dumps({"model": "claude-opus-5-5", "usd": 0.25}) + "\n")
        assert usage(log) == {"usd": 0.75, "model_calls": 2} and usage(Path(d) / "none") == {"usd": 0, "model_calls": 0}


def test_the_judge_can_query_the_ledger_and_search_the_web_across_turns():
    from types import SimpleNamespace
    from generate.kernels.search import llm
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
    results = next(m["content"] for m in sent[-1]["messages"] if m["role"] == "user"
                   and m["content"][0].get("type") == "tool_result")
    assert results[0]["type"] == "tool_result" and "SELECT 1" in results[0]["content"]
    assert abs(usd - 3 * (1000 * 4 + 100 * 20) / 1e6 - 3 * 2 * llm.WEB_SEARCH_USD) < 1e-9


def test_the_stable_parts_of_a_prompt_are_sent_as_cached_blocks_and_the_marker_never_reaches_the_model():
    from types import SimpleNamespace
    from generate.kernels.search import llm
    sent = []

    def stream(client, **kw):
        sent.append(kw)
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 1,
                                                    "cache_read_input_tokens": 1_000_000})
        return SimpleNamespace(model="claude-opus-5-5", usage=usage, stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text="ok")])
    c = llm.Claude.__new__(llm.Claude)
    c.client, c.model, c.effort, c.max_tokens, c.tools, c.max_turns, c.ledger_root = None, "claude-opus-5-5", "high", \
        100, False, 8, None
    orig, llm._stream = llm._stream, stream
    try:
        _, usd = c("SETTING" + llm.CACHE_SPLIT + "ROLE", "PROGRAM" + llm.CACHE_SPLIT + "PLAN")
        c("SETTING", "PROGRAM ONLY")
    finally:
        llm._stream = orig
    system, user = sent[0]["system"], sent[0]["messages"][0]["content"]
    assert [b["text"] for b in system] == ["SETTING", "ROLE"] and all(b["cache_control"] for b in system)
    assert [b["text"] for b in user] == ["PROGRAM", "PLAN"] and "cache_control" in user[0] and "cache_control" not in user[1]
    assert llm.CACHE_SPLIT not in json.dumps(sent)
    assert "cache_control" not in sent[1]["messages"][0]["content"][0]        # one part: nothing to share
    assert 0.2 < usd < 0.3                                    # 1M cache-read tokens at $0.20, not the $4 input rate


def test_a_turn_whose_stream_dies_is_sent_again_and_a_real_error_is_not():
    from types import SimpleNamespace
    from generate.kernels.search import llm

    class ReadTimeout(Exception):
        pass
    calls = []

    def stream(client, **kw):
        calls.append(kw)
        if len(calls) < 3:
            raise ReadTimeout("The read operation timed out")
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 1})
        return SimpleNamespace(model="claude-opus-5-5", usage=usage, stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text="ok")])
    c = llm.Claude.__new__(llm.Claude)
    c.client, c.model, c.effort, c.max_tokens, c.tools, c.max_turns, c.ledger_root = None, "claude-opus-5-5", "high", \
        100, False, 8, None
    orig = llm._stream
    llm._stream = stream
    import time
    real_sleep, time.sleep = time.sleep, lambda s: None
    try:
        assert c("S", "U")[0] == "ok" and len(calls) == 3
        calls.clear()
        llm._stream = lambda client, **kw: (_ for _ in ()).throw(ValueError("bad request"))
        try:
            c("S", "U")
            raise AssertionError("a non-transient error was swallowed")
        except ValueError:
            pass
    finally:
        llm._stream, time.sleep = orig, real_sleep


def test_a_flagged_candidate_is_passed_over_at_promotion():
    from generate.kernels.evolve.run import retime
    from generate.kernels.search.detect import detector
    from generate.kernels.tasks import load
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
    from generate.kernels.search.llm import FALLBACK_BETA, Claude
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
    from types import SimpleNamespace as NS
    dated = NS(model="claude-haiku-4-5-20251001", usage=NS(model_dump=lambda: {"input_tokens": 1_000_000}))
    assert abs(c._usd(dated) - 1.0) < 1e-9                          # priced as claude-haiku-4-5 ($1/MTok in)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)


def test_the_setting_states_seen_shapes_with_calls_weight_and_traffic_never_the_heldout_ones():
    from generate.kernels.evolve.run import setting
    from generate.kernels.tasks.shapes import source
    from generate.kernels.tasks import load
    task = load("toy_rmsnorm")
    sets = {"seen": [(64, 1024), (8, 1024)], "heldout": [(4096, 1024)],
            "calls": {"[64, 1024]": 30, "[8, 1024]": 10, "[4096, 1024]": 99},
            "sources": {"[64, 1024]": {"mixed_1": 20, "long_context": 10}}}
    text = setting(task, sets, "cpu", {"float32": [0.02, 0.02], "bfloat16": [1, 1]})
    assert "[64, 1024]: 30 calls, 75.0% of the score (captured under: long_context 10, mixed_1 20)" in text
    assert "[8, 1024]: 10 calls, 25.0% of the score" in text and "4096" not in text
    assert 'per dtype: {"float32": [0.02, 0.02]}' in text and "Device: cpu" in text      # its dtypes only
    assert source(Path("/x/profile-mixed_1/20261019-1/captures")) == "mixed_1"


def test_selection_carries_each_buckets_sources():
    from generate.kernels.tasks.shapes import select
    from generate.kernels.tasks import load
    task = load("paged_decode")
    seen = [(b, 32, 4, 128, 16, c) for b in (1, 8, 64, 128, 256) for c in (512, 4096)]
    sets = {"seen": seen, "heldout": [], "calls": {json.dumps(list(s)): 1 for s in seen},
            "sources": {json.dumps(list(s)): {"mixed_1": 1} for s in seen}}
    out = select(task, sets, k=3)
    assert sum(sum(v.values()) for v in out["sources"].values()) == sum(out["calls"].values())


def test_a_query_tool_that_cannot_run_answers_the_model_instead_of_crashing_the_search():
    from generate.kernels.search import llm
    with mock.patch.dict(sys.modules, {"data.knowledge.index": None}):
        out = llm.run_query("SELECT 1")
    assert out.startswith("query failed: ") and "Error" in out


def test_the_cli_starts_the_search_from_every_competitor_port_of_its_task_with_the_model_mix(monkeypatch, tmp_path):
    from generate.kernels.search import __main__ as cli
    program = tmp_path / "start.py"
    program.write_text("# EVOLVE-BLOCK-START\nimport torch\n\n\ndef silu_mul(x):\n    return x\n"
                       "# EVOLVE-BLOCK-END\n\nkernel = silu_mul\n")
    monkeypatch.setattr(cli, "_shapes", lambda task, a: {})
    monkeypatch.setattr(cli, "prepare", lambda *a, **k: ({}, program))
    monkeypatch.setattr(cli, "setting", lambda *a, **k: "SETTING")
    monkeypatch.setattr(cli, "Claude", lambda model, *a, **k: model)
    monkeypatch.setattr(cli, "_embedder", lambda root: None)
    seen = {}

    def search(start, seeds, setting, models, *a):
        seen.update(seeds=seeds, setting=setting, models=models)
        raise SystemExit("judged")
    monkeypatch.setattr(cli, "search", search)
    try:
        cli.main(["--task", "silu_mul", "--shapes", "s.json", "--out", str(tmp_path / "out"), "--device", "cpu"])
    except SystemExit as e:
        assert str(e) == "judged"
    assert list(seen["seeds"]) == ["vllm.silu_and_mul"] and "_silu_mul_engine = silu_mul" in seen["seeds"][
        "vllm.silu_and_mul"]
    assert seen["models"] == ["claude-sonnet-5-5"] * 3 + ["claude-opus-5-5"] and "SETTING" in seen["setting"]
