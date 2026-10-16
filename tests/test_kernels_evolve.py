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
os.environ.setdefault("LAB_TARGET", str(ROOT / "targets" / "inference-server.toml"))
os.environ.setdefault("LAB_ENGINE_PYTHON", sys.executable)
os.environ.setdefault("LAB_NO_JAIL", "1")          # no srt or container jail on a dev box; the jail's argv is tested

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


def test_the_screen_refuses_exploit_routes_and_passes_real_kernels():
    from kernels.evolve.core import screen
    for src in ("import gc\n", "torch.cuda.Stream()\n", "x.elapsed_time(y)\n", "import threading\n",
                "import time\n", "sys.modules['kernels'].timing = 1\n", "from kernels import timing\n",
                "os.system('id')\n", "open('/etc/passwd')\n", "import ctypes\n", "inspect.stack()\n"):
        assert screen(src), src
    ev = ROOT / "tests" / "fixtures" / "paged_decode_evolved.py"     # the best program of the 2026-10-14 run
    assert screen(SEED.read_text()) == [] and screen(ev.read_text()) == []
    assert screen("x = os.environ.get('A', '0')\n") == []
    assert screen("import torch\nclass A:\n    def f(self, x):\n        self.a = 1\n        x.b = 2\n") == []
    assert screen("import torch\nx = torch.empty(3)\nx.tag = 1\n")             # bound to a call: refused
    assert screen("import torch\nm = torch.cuda\nm.synchronize = None\n") and screen("def k(x):\n    type(x).f = 1\n")
    assert screen("def k(x):\n    T = type(x)\n    T.zero_ = None\n") and screen("from torch.cuda import Stream\n")
    t = load("toy_rmsnorm")
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.py"
        bad.write_text("import gc\ndef kernel(x, w):\n    return x\n")
        r = judge(t, bad, list(t.seen_default), [], None, "cpu", None, 1)
        assert r["metrics"]["combined_score"] == 0.0 and "refused before running" in r["artifacts"]["error"]


def test_value_distributions_catch_a_kernel_tuned_to_one_scale():
    t = load("toy_rmsnorm")
    with tempfile.TemporaryDirectory() as d:
        clamp = Path(d) / "clamp.py"                 # right for N(0,1) inputs, wrong once values grow
        clamp.write_text("import torch\ndef kernel(x, w):\n"
                         "    if torch.isfinite(x).all() and x.abs().max() > 8:\n        return x.clone()\n"
                         "    xf = x.float()\n"
                         "    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)\n")
        r = judge(t, clamp, list(t.seen_default), [], None, "cpu", None, 2, iters=3)
        g = json.loads(r["artifacts"]["gates"])
        assert r["metrics"]["combined_score"] == 0.0 and g["values_x3"]["passed"] is False


def load_program_text(src: str):
    from kernels.evolve.core import load_program
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "k.py"
        p.write_text(src)
        return load_program(p)


def test_noise_rule_and_physics_alarm():
    from kernels.evolve import core
    assert not core._shown({"median": 1.05, "p25": 0.97, "p75": 1.1})
    assert core._shown({"median": 2.0, "p25": 1.8, "p75": 2.2}) and core._shown({"median": 0.5, "p25": 0.4, "p75": 0.6})
    from kernels import time_pair
    t0 = load("toy_rmsnorm")
    slow = load_program_text("import torch\ndef kernel(x, w):\n    for _ in range(20):\n        y = x.float() @ x.float().T\n"
                             "    return x\n")
    p = time_pair(slow, t0.reference, t0.make_inputs, (64, 256), torch.float32, iters=9, warmup=1)
    assert p["ratio"]["median"] < 1 and p["a"]["median_ms"] > p["b"]["median_ms"]       # a is slower: ratio below 1
    t = load("toy_rmsnorm")
    same = judge(t, SEED, list(t.seen_default), [], None, "cpu", None, 2, iters=3)
    assert same["metrics"]["raw_speedup"] > 0 and "beyond_noise" in json.loads(same["artifacts"]["timings"])[0]
    held = judge(t, SEED, list(t.seen_default), list(t.heldout_default), None, "cpu", None, 2, iters=3)
    assert held["metrics"]["heldout_speedup"] > 0 and len(json.loads(held["artifacts"]["heldout_timings"])) == 2
    err = json.loads(same["artifacts"]["error_vs_reference"])["candidate"]
    assert err["max_abs"]["float32"] is not None and err["cases"] > 0
    from validity.peaks import Peaks
    fast = Peaks("toy", 1.0, 1.0, 1e9, "spec")      # a 1 FLOP/s, 1 B/s machine: any real time beats its bound
    orig = core._peaks
    core._peaks = lambda device: fast
    try:
        r = judge(t, SEED, list(t.seen_default), [], None, "cpu", None, 2, iters=3)
    finally:
        core._peaks = orig
    g = json.loads(r["artifacts"]["gates"])
    assert r["metrics"]["combined_score"] == 0.0 and g["physics"]["passed"] is False


def test_paged_decode_checks_with_no_edge_shapes_do_not_fall_back_to_the_defaults():
    from kernels import check_kernel
    t = load("paged_decode")                    # its default edge shapes (Hq = 1 < Hkv) are not valid GQA inputs
    r = check_kernel(t.reference, t.reference, t.make_inputs, [(2, 8, 2, 16, 4, 11)], [], dtypes=(torch.float32,),
                     edge_shapes=[], repeats=1, layouts=())
    assert r["passed"]


def test_real_sequence_lengths_replace_synthetic_ones_where_captured():
    t = load("paged_decode")
    shape, other = (3, 8, 2, 16, 4, 11), (2, 8, 2, 16, 4, 9)
    real = t.with_ints({json.dumps(list(shape)): [[5, 9, 11]]})
    g = torch.Generator().manual_seed(0)
    assert real.make_inputs(shape, torch.float32, "cpu", g)[4].tolist() == [5, 9, 11]
    assert int(real.make_inputs(other, torch.float32, "cpu", g)[4][0]) == 9               # not captured: synthetic
    q, k, v, table, lens = real.make_inputs(shape, torch.float32, "cpu", torch.Generator().manual_seed(1))
    out = t.reference(q, k, v, table, lens)
    assert out.shape == (3, 8, 16) and torch.isfinite(out).all()
    assert load("toy_rmsnorm").with_ints({"[8, 256]": [[1]]}) is load("toy_rmsnorm")      # no int args: unchanged


def test_graph_timing_is_cuda_only_and_a_kernel_that_cannot_be_timed_scores_zero():
    import dataclasses
    from kernels import time_pair
    from kernels.evolve import core
    from kernels.evolve.run import system_message
    t = load("toy_rmsnorm")
    p = time_pair(t.reference, t.reference, t.make_inputs, (8, 256), torch.float32, iters=3, warmup=1, graphs=True)
    assert p["clock"] == "host"                                  # no CUDA here: timed eagerly
    g = dataclasses.replace(t, graphs=True)
    assert judge(g, SEED, list(t.seen_default), [], None, "cpu", None, 2, iters=3)["metrics"]["correct"] == 1.0
    orig = core.time_pair

    def no_capture(*a, **k):
        raise RuntimeError("operation not permitted when stream is capturing")
    core.time_pair = no_capture
    try:
        r = judge(g, SEED, list(t.seen_default), [], None, "cpu", None, 2, iters=3)
    finally:
        core.time_pair = orig
    gate = json.loads(r["artifacts"]["gates"])["timing"]
    assert r["metrics"]["combined_score"] == 0.0 and not gate["passed"] and "must capture" in gate["reason"]
    assert load("paged_decode").graphs and "CUDA graph" in system_message(load("paged_decode"))


def test_share_of_the_window_per_task_and_the_kernels_no_task_covers():
    from kernels.evolve.run import main
    from kernels.share import shares
    with tempfile.TemporaryDirectory() as d:
        b = Path(d)
        (b / "meta.json").write_text(json.dumps({"window": {"wall_s": 1.0}}))
        (b / "kernels.json").write_text(json.dumps({"kernels": [
            {"name": "fused_moe_kernel", "count": 10, "total_us": 500_000.0},
            {"name": "_paged_decode_splitk_kernel", "count": 10, "total_us": 150_000.0},
            {"name": "_splitk_combine_kernel", "count": 10, "total_us": 50_000.0},
            {"name": "rms_norm_kernel", "count": 10, "total_us": 20_000.0}]}))
        r = shares(b, [load("paged_decode"), load("toy_rmsnorm")])
        pd = r["tasks"]["paged_decode"]
        assert abs(pd["share_of_window"] - 0.2) < 1e-9 and abs(pd["speedup_if_free"] - 1.25) < 1e-9
        assert len(pd["kernels"]) == 2 and r["tasks"]["toy_rmsnorm"]["share_of_window"] == 0
        assert [k["name"] for k in r["unmatched"]] == ["fused_moe_kernel", "rms_norm_kernel"]
        assert main(["share", "--profile", str(b)]) == 0


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
    real = t.cost(shape, torch.bfloat16, (q, k, v, table, lens))          # the bound for these exact lengths
    assert real[1] < nbytes and real[0] == 4 * 8 * int(lens.sum()) * 16


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
        ints = shapes.ints_from_captures(task, [Path(d)])
        assert ints[json.dumps([1, 32, 4, 128, 16, 17])] == [[17]] and len(ints) == 9       # first call per shape
        kept = shapes.select(task, {**s, "ints": ints}, k=3)["ints"]
        assert set(kept) <= {json.dumps(list(x)) for x in [*shapes.select(task, s, k=3)["seen"], *s["heldout"]]}
        sel = shapes.select(task, s, k=3)
        assert len(sel["seen"]) == 3 and sel["selected_from"] == len(s["seen"]) and sel["heldout"] == s["heldout"]
        assert sum(sel["calls"].values()) <= sum(counts[x] for x in s["seen"])
        assert shapes.select(task, s, k=99) is s
        back = shapes.read(shapes.write(Path(d) / "s.json", s))
        assert back["seen"] == s["seen"]


ADVICE = ("consider", "focus", "suggest", "to improve", "inspire", "memory access pattern", "thoughtful", "valuable",
          "simplif", "should", "try ")


def test_prompts_carry_facts_never_advice():
    try:
        from openevolve.config import load_config
        from openevolve.prompt.sampler import PromptSampler
    except ImportError:
        print("skip: openevolve not installed")
        return
    from kernels.evolve.run import HERE, system_message
    cfg = load_config(str(HERE / "config.yaml"))
    cfg.prompt.system_message = system_message(load("paged_decode"))
    s = PromptSampler(cfg.prompt)
    prev = [{"id": "a", "code": "x", "metrics": {"combined_score": 2.0}, "changes": "c"}]
    for diff in (True, False):
        p = s.build_prompt(current_program="def kernel(): ...", parent_program="def kernel(): ...",
                           program_metrics={"combined_score": 1.0}, previous_programs=prev, top_programs=prev,
                           inspirations=prev, diff_based_evolution=diff,
                           program_artifacts={"gates": "{}"}, feature_dimensions=cfg.database.feature_dimensions)
        text = (p["system"] + p["user"]).lower()
        assert not [w for w in ADVICE if w in text], [w for w in ADVICE if w in text]
        assert "score declined: 2.0000" in text and "paged_decode_attention" in text


def test_remote_evaluation_sends_the_source_on_stdin_and_never_a_key():
    try:
        from kernels.evolve import evaluator
    except ImportError:
        print("skip: openevolve not installed")
        return
    t = load("toy_rmsnorm")
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "eval.json"
        cfg.write_text(json.dumps({"task": t.name, "seen": [list(s) for s in t.seen_default], "device": "cpu", "iters": 3}))
        fake_ssh = Path(d) / "ssh"                  # stands in for ssh: runs the remote shell string here
        fake_ssh.write_text("#!/bin/sh\nshift\nenv | grep -c _API_KEY > " + str(Path(d) / "keys") + "\nexec sh -c \"$@\"\n")
        fake_ssh.chmod(0o755)
        old = dict(os.environ)
        os.environ.update(KERNEL_EVAL_SSH=json.dumps([str(fake_ssh), "host"]), KERNEL_EVAL_REMOTE=f"cd {ROOT} &&",
                          KERNEL_EVAL_REMOTE_CONFIG=str(cfg), KERNEL_EVAL_CONFIG=str(cfg), ANTHROPIC_API_KEY="sk-x")
        try:
            r = evaluator.evaluate_stage1(str(SEED))
        finally:
            os.environ.clear()
            os.environ.update(old)
        assert r.metrics["correct"] == 1.0 and (Path(d) / "keys").read_text().strip() == "0"


RIGHT = "    xf = x.float()\n    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)\n"
# Published exploit routes (Sakana, CUDA-L1, METR, KernelBot) as toy kernels: each must score 0, for the reason given.
CANARIES = {
    "timer via __import__": ("import torch\n__import__('time').perf_counter = lambda: 0.0\n"
                             "def kernel(x, w):\n" + RIGHT, "refused"),
    "timer statistics patched through an alias": ("import torch\nimport statistics as s\n"
                                                  "s.quantiles = lambda t, n: [0.0, 1e-9, 1e-9]\n"
                                                  "def kernel(x, w):\n" + RIGHT, "refused"),
    "checker's tensor op patched through an alias": ("import torch as T\nT.Tensor.abs = lambda self: self * 0\n"
                                                     "def kernel(x, w):\n    return x.clone()\n", "refused"),
    "sync made a no-op": ("import torch\nimport torch as T\nT.cuda.synchronize = lambda *a, **k: None\n"
                          "def kernel(x, w):\n" + RIGHT, "refused"),
    # past the screen (the module reached through a list): caught at run time
    "timer statistics patched at import": ("import torch\nimport statistics\nmods = [statistics]\n"
                                           "mods[0].quantiles = lambda t, n: [0.0, 1e-9, 1e-9]\n"
                                           "def kernel(x, w):\n" + RIGHT, "patched"),
    "checker's tensor op patched during the call": ("import torch\nmods = [torch]\ndef kernel(x, w):\n"
                                                    "    mods[0].Tensor.abs = lambda self: self * 0\n"
                                                    "    return x.clone()\n", "patched"),
    "the reference imported": ("from kernels.tasks.toy_rmsnorm import reference as kernel\n", "refused"),
    "background thread": ("import threading, torch\ndef kernel(x, w):\n" + RIGHT, "refused"),
    "output never written": ("import torch\ndef kernel(x, w):\n    return torch.empty_like(x)\n", "correctness"),
    "answers cached by pointer": ("import torch\n_c = {}\ndef kernel(x, w):\n    k = (x.data_ptr(), tuple(x.shape))\n"
                                  "    if k not in _c:\n        _c[k] = _f(x, w)\n    return _c[k]\n"
                                  "def _f(x, w):\n" + RIGHT, "memoization"),
    "a module reached through getattr": ("import torch\nc = getattr(torch, 'cuda')\ndef kernel(x, w):\n" + RIGHT,
                                         "refused"),
    "right on seen shapes only": ("import torch\ndef kernel(x, w):\n    if x.shape[0] > 64:\n"
                                  "        return torch.zeros_like(x)\n" + RIGHT, "heldout"),
}


def test_exploit_canaries_all_score_zero():
    from kernels.evolve.jailed import judge_jailed
    t = load("toy_rmsnorm")
    cfg = {"task": t.name, "seen": [list(s) for s in t.seen_default], "heldout": [list(s) for s in t.heldout_default],
           "device": "cpu", "iters": 3}
    assert judge_jailed(cfg, "import torch\ndef kernel(x, w):\n" + RIGHT, 2)["metrics"]["correct"] == 1.0
    for name, (src, why) in CANARIES.items():
        r = judge_jailed(cfg, src, 2)
        said = (r["artifacts"].get("error") or "") + (r["artifacts"].get("gates") or "")
        assert r["metrics"]["combined_score"] == 0.0, name
        want = {"refused": "refused before running", "patched": "patched what the evaluator uses"}.get(why, why)
        assert want in said, (name, said[:800])


def test_a_kernel_right_while_checked_and_wrong_while_timed_scores_zero():
    from kernels.evolve.jailed import judge_jailed
    t = load("toy_rmsnorm")                      # wrong from the 5th call on one input object: only timing gets there
    src = ("import torch\n_seen = []\ndef kernel(x, w):\n    n = sum(1 for t in _seen if t is x)\n"
           "    _seen.append(x)\n    if n >= 4:\n        return torch.zeros_like(x)\n" + RIGHT)
    r = judge_jailed({"task": t.name, "seen": [list(s) for s in t.seen_default], "device": "cpu", "iters": 20}, src, 2)
    gates = json.loads(r["artifacts"]["gates"])
    assert r["metrics"]["combined_score"] == 0.0 and gates["correctness_seen"]["passed"]
    assert "differs from the reference" in gates["timing"]["reason"]


def test_the_jail_sees_a_fresh_tree_and_a_copied_harness_never_this_repo():
    from kernels.evolve import jailed
    from lab.safety import container, grader, jail
    with tempfile.TemporaryDirectory() as d:
        h = jailed.stage_harness(Path(d) / "h")
        names = {p.relative_to(h).as_posix() for p in h.rglob("*") if p.is_file()}
        assert "kernels/evolve/core.py" in names and "lab/target.py" in names and "validity/peaks.py" in names
        assert not any(n.startswith(("lab/ledger", "lab/runs", "corpus", "data")) or n.endswith(".env") for n in names)
    seen, saved = {}, (jail.backend, container.available, container._docker, grader.subprocess.run)

    def fake_run(argv, **kw):                       # stands in for docker: `run` answers, cleanup is a no-op
        if argv[1] != "run":
            return grader.subprocess.CompletedProcess(argv, 0, "", "")
        seen["argv"], seen["config"] = argv, json.loads(Path(argv[argv.index("kernels.evolve.core") + 1]).read_text())
        return grader.subprocess.CompletedProcess(argv, 0, '\n{"metrics": {"combined_score": 1.0}, "artifacts": {}}\n', "")
    jail.backend, container.available, container._docker = (lambda: "container"), (lambda: True), (lambda: "docker")
    grader.subprocess.run = fake_run
    try:
        t = load("toy_rmsnorm")
        r = jailed.judge_jailed({"task": t.name, "seen": [[8, 256]], "device": "cpu"}, "def kernel(x, w): ...\n", 1)
    finally:
        jail.backend, container.available, container._docker, grader.subprocess.run = saved
    argv, mounts = seen["argv"], [a for a in seen["argv"] if a.startswith("type=bind")]
    assert r["metrics"]["combined_score"] == 1.0 and argv[:2] == ["docker", "run"]
    assert argv[argv.index("--network") + 1] == "none" and "heldout" not in seen["config"]
    assert any("harness" in m and m.endswith(",readonly") for m in mounts)
    assert not any(f"src={ROOT}," in m for m in mounts)          # this repo is never mounted


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
        c.evolution_trace.enabled, c.evolution_trace.include_code = True, True
        res = run_evolution(str(SEED), str(ROOT / "kernels" / "evolve" / "evaluator.py"), c, iterations=4,
                            output_dir=str(Path(d) / "oe"), cleanup=False)
        assert "rsqrt" in res.best_code and "return x.clone()" not in res.best_code
        seed = judge(t, SEED, list(t.seen_default), [], None, "cpu", None, 2, iters=3)["metrics"]["combined_score"]
        assert res.metrics.get("correct") == 1.0 and res.best_score > 3 * seed     # the naive seed is ~0.1x
        from kernels.evolve.run import record
        from lab import ledger
        best = Path(d) / "best.py"
        best.write_text(res.best_code)
        rec = record(t, {"seen": list(t.seen_default)}, c, 4, "cpu", Path(d) / "oe" / "evolution_trace.jsonl", best,
                     res.metrics, root=Path(d) / "ledger")
        stored = list(ledger.records(Path(d) / "ledger", kind="kernels"))
        assert stored and stored[0]["id"] == rec["id"] and len(rec["result"]["candidates"]) >= 1
        assert all({"id", "parent", "metrics"} <= x.keys() for x in rec["result"]["candidates"])
        assert (Path(d) / "ledger" / rec["result"]["trace"]).exists() and rec["config"]["split"] == "seen"
        from kernels.evolve.run import retime
        code, metrics, retimed = retime(Path(d) / "oe" / "evolution_trace.jsonl", 2)
        assert "rsqrt" in code and metrics["correct"] == 1.0 and all(len(x["scores"]) == 3 for x in retimed)
        print("best", res.best_score)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
