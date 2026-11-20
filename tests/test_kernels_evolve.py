"""The layer-1 loop end to end on CPU with a scripted model: tasks, shapes, the judge, an OpenEvolve run."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LAB_TARGET", str(ROOT / "evals" / "targets" / "inference-server.toml"))
os.environ.setdefault("LAB_ENGINE_PYTHON", sys.executable)
os.environ.setdefault("LAB_NO_JAIL", "1")          # no srt or container jail

import torch  # noqa: E402

from lab.kernels.tasks import shapes  # noqa: E402
from lab.kernels.eval.core import judge  # noqa: E402
from lab.kernels.tasks import load  # noqa: E402
from lab.profile import capture  # noqa: E402

SEED = ROOT / "lab" / "kernels" / "evolve" / "seeds" / "toy_rmsnorm.py"
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


CUDA_PROGRAM = r'''import torch
from torch.utils.cpp_extension import load_inline

CUDA = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void scale_k(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 2.0f * x[i];
}
torch::Tensor scale(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    scale_k<<<(n + 255) / 256, 256, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""
_ext = load_inline("scale_ext", cpp_sources="torch::Tensor scale(torch::Tensor x);", cuda_sources=CUDA,
                   functions=["scale"], extra_cuda_cflags=["-O3"])


def kernel(x):
    return _ext.scale(x)
'''


def test_a_programs_cpp_builds_once_for_all_its_stages_and_only_for_that_source():
    from lab.kernels.eval import jailed
    seen = []

    def fake(tree, argv, *, timeout_s, env_extra, read, memory=None):
        ext = Path(env_extra["TORCH_EXTENSIONS_DIR"])
        seen.append((str(tree), sorted(p.name for p in tree.rglob("*") if p.parent == tree or ext in p.parents)))
        (ext / "scale_ext").mkdir(parents=True, exist_ok=True)
        (ext / "scale_ext" / "scale_ext.so").write_bytes(b"built")
        (tree / "memo.pt").write_bytes(b"left behind")                      # removed before the next judgment
        (ext / "scale_ext" / "answers.bin").write_bytes(b"kept for later")   # removed right after this one
        if not (tree / "candidate.py.link").is_symlink():
            (tree / "candidate.py.link").symlink_to("/etc/hosts")          # unlinked, never followed
        return subprocess.CompletedProcess(argv, 1, "", "stop")
    orig, dirs = jailed.grader.jailed, jailed.EVAL_DIRS
    with tempfile.TemporaryDirectory() as d:
        jailed.grader.jailed, jailed.EVAL_DIRS = fake, Path(d) / "eval"
        try:
            cfg = {"task": "toy_rmsnorm", "seen": [[4, 8]], "device": "cpu"}
            for _ in range(2):
                jailed.judge_jailed(cfg, CUDA_PROGRAM, 1)
            jailed.judge_jailed(cfg, CUDA_PROGRAM + "\n# another program\n", 1)
        finally:
            jailed.grader.jailed, jailed.EVAL_DIRS = orig, dirs
    (t1, a), (t2, b), (t3, c) = seen
    assert t1 == t2 != t3                                         # one fixed tree per source
    assert a == [".result_key", "candidate.py", "eval_config.json"] and c == a
    assert b == [".result_key", "candidate.py", "eval_config.json", "scale_ext", "scale_ext.so",
                 "torch_extensions"]                                           # the build kept, nothing else


def test_a_result_line_the_candidate_prints_is_ignored():
    from lab.kernels.eval.jailed import judge_jailed
    forged = "'{\"metrics\": {\"combined_score\": 9.0, \"correct\": 1.0, \"stage\": 2}, \"artifacts\": {}}'"
    src = "import torch\ndef kernel(x, w):\n    print(" + forged + ")\n" + RIGHT + "\n"        # printed on every call
    r = judge_jailed({"task": "toy_rmsnorm", "seen": [[8, 256], [32, 512]], "device": "cpu", "iters": 3}, src, 2)
    assert r["metrics"]["correct"] == 1.0 and r["metrics"]["combined_score"] != 9.0 and "timings" in r["artifacts"]


def test_the_gate_is_judged_in_its_own_dir():
    from lab.kernels.eval.jailed import _eval_dir
    with tempfile.TemporaryDirectory() as d:
        import lab.kernels.eval.jailed as j
        saved, j.EVAL_DIRS = j.EVAL_DIRS, Path(d)
        try:
            assert _eval_dir("x", None)[0] != _eval_dir("x", None, "gate")[0] == _eval_dir("x", None, "gate")[0]
        finally:
            j.EVAL_DIRS = saved


def test_a_cuda_extension_builds_in_the_jail_and_is_judged_on_the_gpu():
    import shutil
    from lab.kernels.eval.jailed import judge_jailed
    src = (ROOT / "tests" / "fixtures" / "cuda_toy_rmsnorm.py").read_text()
    if not torch.cuda.is_available() or not (shutil.which("nvcc") or Path("/usr/local/cuda/bin/nvcc").exists()):
        print("skip: no CUDA GPU or no nvcc")
        return
    cfg = {"task": "toy_rmsnorm", "seen": [[8, 256], [32, 512], [64, 1024]], "device": "cuda", "iters": 10}
    r = judge_jailed(cfg, src, 2)
    assert r["metrics"]["correct"] == 1.0 and r["metrics"]["combined_score"] > 0, r["artifacts"]


def test_ncu_counters_reach_the_judge_on_the_gpu():
    import shutil
    from lab.kernels.eval.jailed import judge_jailed
    if not torch.cuda.is_available() or not shutil.which("ncu"):
        print("skip: no CUDA GPU or no ncu")
        return
    src = (ROOT / "tests" / "fixtures" / "cuda_toy_rmsnorm.py").read_text()
    cfg = {"task": "toy_rmsnorm", "seen": [[8, 256], [64, 1024]], "device": "cuda", "iters": 3}
    a = judge_jailed(cfg, src, 3)["artifacts"]
    assert "ncu" in a and "ERR_NVGPUCTRPERM" not in json.dumps(a), a
    shapes = json.loads(a["ncu"])["shapes"]
    assert len(shapes) == 2 and all(s.get("kernels") for s in shapes), shapes      # counters on both shapes
def test_the_evaluators_modules_are_out_of_reach():
    from lab.kernels.eval import core
    t = load("toy_rmsnorm")
    before = core._watched(t)
    real = sys.modules["lab.kernels.eval.check"]
    sys.modules["lab.kernels.eval.check"] = type(sys)("lab.kernels.eval.check")                    # a module swapped under its name
    try:
        assert "sys.modules[lab.kernels.eval.check]" in core._patched(before, t)
    finally:
        sys.modules["lab.kernels.eval.check"] = real
    assert core._patched(before, t) == [] and core._compare is real._compare      # compare bound at import


def test_value_distributions_catch_a_kernel_tuned_to_one_scale():
    t = load("toy_rmsnorm")
    with tempfile.TemporaryDirectory() as d:
        clamp = Path(d) / "clamp.py"                 # right for N(0,1) inputs, wrong once values grow
        clamp.write_text("import torch\ndef kernel(x, w):\n"
                         "    if torch.isfinite(x).all() and x.abs().max() > 8:\n        return x.clone()\n"
                         "    xf = x.float()\n"
                         "    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)\n")
        seen = list(t.seen_default)
        for held, full in ((seen[:1], False), ([], True)):     # the gate: held-out shapes, or none but `full`
            r = judge(t, clamp, seen, held, None, "cpu", None, 2, iters=3, full=full)
            g = json.loads(r["artifacts"]["gates"])
            assert r["metrics"]["combined_score"] == 0.0 and g["values_x3"]["passed"] is False
        g = json.loads(judge(t, clamp, seen, [], None, "cpu", None, 2, iters=3)["artifacts"]["gates"])
        assert "values_x3" not in g and "edge_cases" in g          # selection: correctness and timing, no value scales


def test_a_task_without_nonfinite_inputs_gets_none_in_any_check_of_the_gate():
    import dataclasses
    from lab.kernels.eval import core
    t = dataclasses.replace(load("toy_rmsnorm"), nonfinite=False)
    seen, orig = [], core.check_kernel
    core.check_kernel = lambda *a, **kw: (seen.append(kw.get("nonfinite", True)), orig(*a, **kw))[1]
    try:
        r = judge(t, SEED, list(t.seen_default), [], None, "cpu", None, 2, iters=3, full=True)
    finally:
        core.check_kernel = orig
    assert r["metrics"]["correct"] == 1.0 and len(seen) > 1 and not any(seen)


def load_program_text(src: str):
    from lab.kernels.eval.core import load_program
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "k.py"
        p.write_text(src)
        return load_program(p)


def test_noise_rule_and_physics_alarm():
    from lab.kernels.eval import core
    assert not core._shown({"median": 1.05, "p25": 0.97, "p75": 1.1})
    assert core._shown({"median": 2.0, "p25": 1.8, "p75": 2.2}) and core._shown({"median": 0.5, "p25": 0.4, "p75": 0.6})
    from lab.kernels.eval import time_pair
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
    from lab.infra.context.peaks import Peaks
    fast = Peaks("toy", 1.0, 1.0, 1e9, "spec")      # a 1 FLOP/s, 1 B/s machine
    orig = core._peaks
    core._peaks = lambda device: fast
    try:
        r = judge(t, SEED, list(t.seen_default), [], None, "cpu", None, 2, iters=3)
    finally:
        core._peaks = orig
    g = json.loads(r["artifacts"]["gates"])
    assert r["metrics"]["combined_score"] == 0.0 and g["physics"]["passed"] is False


def test_split_holds_out_a_region_even_when_calls_pile_up_at_the_largest_value():
    import dataclasses
    from collections import Counter
    from lab.kernels.tasks import randn
    t = load("toy_rmsnorm")
    counts = Counter({(m, 64, 96, 8, m, 1): (1000 if m == 512 else 3) for m in range(8, 513, 8)})
    s = shapes.split(t, counts)
    assert s["cut_by"] == "distinct" and s["heldout"] and (512, 64, 96, 8, 512, 1) in s["seen"]   # the top stays seen
    h = [x[0] for x in s["heldout"]]
    assert not any(min(h) < x[0] < max(h) for x in s["seen"]) and max(h) < 512                   # a band below the top
    two = Counter({(n, 32, 4, 128, 16, L): (900 if (n, L) == (33, 8032) else 10)
                   for n in range(1, 257, 8) for L in (650, 8032)})
    two.update({(n, 32, 4, 128, 16, 8032): 300 for n in range(200, 257, 8)})
    s2 = shapes.split(dataclasses.replace(t, heldout_axes=(0, 5)), two)
    held = sum(two[x] for x in s2["heldout"])
    assert s2["heldout"] and held <= sum(two.values()) / 2 and (33, 32, 4, 128, 16, 8032) in s2["seen"]
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert torch.equal(randn((4, 5), g1, "cpu", torch.float32), randn((4, 5), g2, "cpu", torch.float32))


def test_graph_timing_is_cuda_only_and_a_kernel_that_cannot_be_timed_scores_zero():
    import dataclasses
    from lab.kernels.eval import time_pair
    from lab.kernels.eval import core
    from lab.kernels.evolve.run import system_message
    t = load("toy_rmsnorm")
    p = time_pair(t.reference, t.reference, t.make_inputs, (8, 256), torch.float32, iters=3, warmup=1, graphs=True)
    assert p["clock"] == "host"                                  # no CUDA: host clock
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
    assert "CUDA graph" in system_message(g)


def test_ship_takes_a_region_with_a_passing_held_out_gate_only():
    from lab.kernels.evolve.run import initial_program
    from lab.kernels.eval.ship import gate_passed, module_source, ship
    program = ("# EVOLVE-BLOCK-START\ndef entry(x):\n    return x\n\ndef faster(x):\n    return x * 1\n"
               "# EVOLVE-BLOCK-END\n\nkernel = faster\n")
    assert module_source(program, "entry").endswith("entry = faster\n")
    assert "EVOLVE" not in module_source(initial_program(load("toy_rmsnorm")), "kernel")
    assert module_source("def kernel(x):\n    return x\n", "entry").endswith("entry = kernel\n")    # a rewrite
    ok = {"metrics": {"correct": 1.0},
          "artifacts": {"gates": json.dumps({"correctness_heldout": {"passed": True, "cases": 6},
                                             "physics": {"passed": True}})}}
    flagged = json.loads(ok["artifacts"]["gates"])
    assert gate_passed(ok)
    assert not gate_passed({**ok, "artifacts": {"gates": json.dumps({**flagged, "physics": {"passed": False}})}})
    assert not gate_passed({**ok, "artifacts": {"gates": json.dumps(
        {**flagged, "correctness_heldout": {"passed": True, "cases": 0}})}})      # held-out set empty
    for gate, why in (({"metrics": {"correct": 1.0}, "artifacts": {"gates": "{}"}}, "held-out"), (ok, "only a region")):
        try:
            ship(load("toy_rmsnorm"), program, gate, Path("nowhere"))
            raise AssertionError("shipped")
        except ValueError as e:
            assert why in str(e)


def test_capture_records_calls_and_the_split_holds_out_a_band():
    import dataclasses
    import types
    mod = types.ModuleType("fake_engine_kernels")

    def shape_of(call):
        a = call["args"]
        if "values" not in a[4]:
            return None
        return (*a[0]["shape"][:2], a[1]["shape"][1], a[0]["shape"][2], a[1]["shape"][2], max(a[4]["values"]))

    def decode_attention(q, k_pool, v_pool, block_tables, seq_lens, scale=1.0):
        return q
    mod.decode_attention = decode_attention
    sys.modules["fake_engine_kernels"] = mod
    user = types.ModuleType("fake_engine_model")
    user.decode_attention = decode_attention           # a `from x import f` copy
    sys.modules["fake_engine_model"] = user
    with tempfile.TemporaryDirectory() as d:
        with capture.capture(["fake_engine_kernels:decode_attention"], Path(d)):
            for n in (1, 1, 2, 4, 8, 16, 32, 64, 128, 256):
                q = torch.zeros(n, 32, 128)
                pool = torch.zeros(4, 4, 16, 128)
                user.decode_attention(q, pool, pool, torch.zeros(n, 2, dtype=torch.int32),
                                            torch.full((n,), 17 * n, dtype=torch.int32), scale=0.1)
        assert user.decode_attention is decode_attention          # restored
        calls = capture.load(Path(d))
        assert len(calls) == 10 and calls[0]["kwargs"]["scale"] == {"value": 0.1} and calls[0]["sample"]
        task = dataclasses.replace(load("toy_rmsnorm"), entry="fake_engine_kernels:decode_attention",
                                   shape_of=shape_of, heldout_axes=(0, 5))
        counts = shapes.from_captures(task, [Path(d)])
        assert counts[(1, 32, 4, 128, 16, 17)] == 2
        s = shapes.split(task, counts)
        assert s["source"] == "captures" and (256, 32, 4, 128, 16, 17 * 256) in s["seen"]       # the largest stays seen
        h = [x[0] for x in s["heldout"]]
        assert h and not any(min(h) < x[0] < max(h) for x in s["seen"])                           # a band
        ints = shapes.ints_from_captures(task, [Path(d)])
        assert ints[json.dumps([1, 32, 4, 128, 16, 17])] == [[17]] and len(ints) == 9       # first call per shape
        from unittest import mock
        with mock.patch.object(capture, "SMALL_INT", 8), tempfile.TemporaryDirectory() as d2:
            with capture.capture(["fake_engine_kernels:decode_attention"], Path(d2)):
                for n in (4, 16):                            # 16 lengths: not logged by value
                    user.decode_attention(torch.zeros(n, 32, 128), pool, pool, torch.zeros(n, 2, dtype=torch.int32),
                                                torch.full((n,), 17, dtype=torch.int32), scale=0.1)
            part = shapes.ints_from_captures(task, [Path(d2)])
            assert list(part) == [json.dumps([4, 32, 4, 128, 16, 17])]                     # none rather than a part
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
    from lab.kernels.evolve.run import HERE, system_message
    cfg = load_config(str(HERE / "config.yaml"))
    import dataclasses
    cfg.prompt.system_message = system_message(dataclasses.replace(load("toy_rmsnorm"), entry="fake_pkg.mod:entry"))
    s = PromptSampler(cfg.prompt)
    prev = [{"id": "a", "code": "x", "metrics": {"combined_score": 2.0}, "changes": "c"}]
    for diff in (True, False):
        p = s.build_prompt(current_program="def kernel(): ...", parent_program="def kernel(): ...",
                           program_metrics={"combined_score": 1.0}, previous_programs=prev, top_programs=prev,
                           inspirations=prev, diff_based_evolution=diff,
                           program_artifacts={"gates": "{}"}, feature_dimensions=cfg.database.feature_dimensions)
        text = (p["system"] + p["user"]).lower()
        assert not [w for w in ADVICE if w in text], [w for w in ADVICE if w in text]
        assert "score declined: 2.0000" in text and "fake_pkg.mod:entry" in text


def test_remote_evaluation_sends_the_source_on_stdin_and_never_a_key():
    from lab.kernels.eval import evaluator
    t = load("toy_rmsnorm")
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "eval.json"
        cfg.write_text(json.dumps({"task": t.name, "seen": [list(s) for s in t.seen_default], "device": "cpu", "iters": 3}))
        fake_ssh = Path(d) / "ssh"                  # stands in for ssh: runs the remote shell string here
        fake_ssh.write_text("#!/bin/sh\nshift\nenv | grep -c _API_KEY > " + str(Path(d) / "keys") + "\nexec sh -c \"$@\"\n")
        fake_ssh.chmod(0o755)
        old = dict(os.environ)
        os.environ.update(KERNEL_EVAL_SSH=json.dumps([str(fake_ssh), "host"]), KERNEL_EVAL_REMOTE=f"cd {ROOT} &&",
                          KERNEL_EVAL_CONFIG=str(cfg), ANTHROPIC_API_KEY="sk-x")       # config travels on stdin
        try:
            r = evaluator.judge_stage(str(SEED), 1)
        finally:
            os.environ.clear()
            os.environ.update(old)
        assert r["metrics"]["correct"] == 1.0 and (Path(d) / "keys").read_text().strip() == "0"


RIGHT = "    xf = x.float()\n    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)\n"
# Exploit routes as toy kernels: each must score 0, for the reason given.
CANARIES = {
    "timer via __import__": ("import torch\n__import__('time').perf_counter = lambda: 0.0\n"
                             "def kernel(x, w):\n" + RIGHT, "patched"),
    "timer statistics patched through an alias": ("import torch\nimport statistics as s\n"
                                                  "s.quantiles = lambda t, n: [0.0, 1e-9, 1e-9]\n"
                                                  "def kernel(x, w):\n" + RIGHT, "patched"),
    "checker's tensor op patched through an alias": ("import torch as T\nT.Tensor.abs = lambda self: self * 0\n"
                                                     "def kernel(x, w):\n    return x.clone()\n", "patched"),
    "sync made a no-op": ("import torch\nimport torch as T\nT.cuda.synchronize = lambda *a, **k: None\n"
                          "def kernel(x, w):\n" + RIGHT, "patched"),
    "timer statistics patched at import": ("import torch\nimport statistics\nmods = [statistics]\n"
                                           "mods[0].quantiles = lambda t, n: [0.0, 1e-9, 1e-9]\n"
                                           "def kernel(x, w):\n" + RIGHT, "patched"),
    "checker's tensor op patched during the call": ("import torch\nmods = [torch]\ndef kernel(x, w):\n"
                                                    "    mods[0].Tensor.abs = lambda self: self * 0\n"
                                                    "    return x.clone()\n", "patched"),
    "output never written": ("import torch\ndef kernel(x, w):\n    return torch.empty_like(x)\n", "correctness"),
    "answers cached by pointer": ("import torch\n_c = {}\ndef kernel(x, w):\n    k = (x.data_ptr(), tuple(x.shape))\n"
                                  "    if k not in _c:\n        _c[k] = _f(x, w)\n    return _c[k]\n"
                                  "def _f(x, w):\n" + RIGHT, "memoization"),
    "right on seen shapes only": ("import torch\ndef kernel(x, w):\n    if x.shape[0] > 64:\n"
                                  "        return torch.zeros_like(x)\n" + RIGHT, "heldout"),
}


def test_exploit_canaries_all_score_zero():
    from lab.kernels.eval.jailed import judge_jailed
    t = load("toy_rmsnorm")
    cfg = {"task": t.name, "seen": [list(s) for s in t.seen_default], "heldout": [list(s) for s in t.heldout_default],
           "device": "cpu", "iters": 3}
    assert judge_jailed(cfg, "import torch\ndef kernel(x, w):\n" + RIGHT, 2)["metrics"]["correct"] == 1.0
    for name, (src, why) in CANARIES.items():
        r = judge_jailed(cfg, src, 2)
        said = (r["artifacts"].get("error") or "") + (r["artifacts"].get("gates") or "")
        assert r["metrics"]["combined_score"] == 0.0, name
        want = {"patched": "patched what the evaluator uses"}.get(why, why)
        assert want in said, (name, said[:800])


def test_a_kernel_right_while_checked_and_wrong_while_timed_scores_zero():
    from lab.kernels.eval.jailed import judge_jailed
    t = load("toy_rmsnorm")                      # wrong from the 5th call on one input object
    src = ("import torch\n_seen = []\ndef kernel(x, w):\n    n = sum(1 for t in _seen if t is x)\n"
           "    _seen.append(x)\n    if n >= 4:\n        return torch.zeros_like(x)\n" + RIGHT)
    r = judge_jailed({"task": t.name, "seen": [list(s) for s in t.seen_default], "device": "cpu", "iters": 20}, src, 2)
    gates = json.loads(r["artifacts"]["gates"])
    assert r["metrics"]["combined_score"] == 0.0 and gates["correctness_seen"]["passed"]
    assert "differs from the reference" in gates["timing"]["reason"]


def test_outputs_are_verified_again_outside_the_candidates_process():
    from lab.kernels.eval.check import _gen
    from lab.kernels.eval.jailed import judge_jailed
    from lab.kernels.eval.verify import verify
    t = load("toy_rmsnorm")
    cfg = {"task": t.name, "seen": [list(s) for s in t.seen_default], "heldout": [list(s) for s in t.heldout_default],
           "device": "cpu", "iters": 3}
    r = judge_jailed(cfg, "import torch\ndef kernel(x, w):\n" + RIGHT, 2)
    g = json.loads(r["artifacts"]["gates"])["verified_outside"]
    assert r["metrics"]["correct"] == 1.0 and g["passed"] and "recomputed outside" in g["reason"]
    from lab.kernels.eval.core import _scaled
    from lab.kernels.eval.verify import expected
    recs = []
    for kind, shape, dt, seed, scale in sorted(expected(cfg, t, False), key=str):
        make = t.make_inputs if scale is None else _scaled(t.make_inputs, scale)
        x, w = make(tuple(json.loads(shape)), getattr(torch, dt), "cpu", _gen(seed))
        recs.append({"kind": kind, "shape": json.loads(shape), "dtype": dt, "seed": seed, "scale": scale,
                     "outputs": [t.reference(x, w)]})
    with tempfile.TemporaryDirectory() as d:
        torch.save({"records": recs}, Path(d) / "ok.pt")
        assert verify(cfg, Path(d) / "ok.pt")[0]
        torch.save({"records": recs[:1]}, Path(d) / "some.pt")          # cases dropped from the record
        ok, why = verify(cfg, Path(d) / "some.pt")
        assert not ok and "expected outputs were not recorded" in why
        assert not verify(cfg, Path(d) / "some.pt", timed=False)[0] and not verify(cfg, Path(d) / "ok.pt", timed=True)[0]
        x = recs[0]["outputs"][0]
        torch.save({"records": [{**recs[0], "outputs": [x + 1]}] + recs[1:]}, Path(d) / "bad.pt")   # wrong values
        ok, why = verify(cfg, Path(d) / "bad.pt")
        assert not ok and "differs from the reference" in why
        assert not verify(cfg, Path(d) / "missing.pt")[0]
        (Path(d) / "link.pt").symlink_to(Path(d) / "ok.pt")             # never followed on this side
        assert verify(cfg, Path(d) / "link.pt") == (False, "the outputs file is not a regular file under the size cap")


def test_the_jail_sees_a_fresh_tree_and_a_copied_harness_never_this_repo():
    from lab.kernels.eval import jailed
    from lab.infra.sandbox import container, grader, jail
    with tempfile.TemporaryDirectory() as d:
        h = jailed.stage_harness(Path(d) / "h")
        names = {p.relative_to(h).as_posix() for p in h.rglob("*") if p.is_file()}
        assert "lab/kernels/eval/core.py" in names and "evals/targets/target.py" in names and "lab/infra/context/peaks.py" in names
        assert not any(n.startswith(("local", "evals/regimes", "lab/records", "evals/public")) or n.endswith(".env") for n in names)
    seen, saved = {}, (jail.backend, container.available, container._docker, grader.subprocess.run)

    def fake_run(argv, **kw):                       # stands in for docker: `run` answers, cleanup is a no-op
        if argv[1] != "run":
            return grader.subprocess.CompletedProcess(argv, 0, "", "")
        seen["argv"], seen["config"] = argv, json.loads(Path(argv[argv.index("lab.kernels.eval.core") + 1]).read_text())
        key = Path(seen["config"]["result_key"]).read_text()
        line = json.dumps({"metrics": {"combined_score": 1.0}, "artifacts": {}, "result_key": key})
        return grader.subprocess.CompletedProcess(argv, 0, '\n{"metrics": {"combined_score": 9.0}}\n' + line + "\n", "")
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
    assert argv[argv.index("--memory") + 1] == argv[argv.index("--memory-swap") + 1] == "32g"     # host RAM capped
    assert any("harness" in m and m.endswith(",readonly") for m in mounts)
    assert not any(f"src={ROOT}," in m for m in mounts)          # this repo is never mounted


def _scripted_class():
    """A stand-in model: odd calls (counted in a file) return the cheat, even calls the real speedup."""
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
        res = run_evolution(str(SEED), str(ROOT / "lab" / "kernels" / "eval" / "evaluator.py"), c, iterations=4,
                            output_dir=str(Path(d) / "oe"), cleanup=False)
        assert "rsqrt" in res.best_code and "return x.clone()" not in res.best_code
        seed = judge(t, SEED, list(t.seen_default), [], None, "cpu", None, 2, iters=3)["metrics"]["combined_score"]
        assert res.metrics.get("correct") == 1.0 and res.best_score > 3 * seed
        from lab.kernels.evolve.run import record
        from lab.records import ledger
        best = Path(d) / "best.py"
        best.write_text(res.best_code)
        rec = record(t, {"seen": list(t.seen_default)}, [{"name": "scripted", "weight": 1.0}], 4, "cpu",
                     Path(d) / "oe" / "evolution_trace.jsonl", best, res.metrics, root=Path(d) / "ledger")
        stored = list(ledger.records(Path(d) / "ledger", kind="kernels"))
        assert stored and stored[0]["id"] == rec["id"] and len(rec["result"]["candidates"]) >= 1
        assert all({"id", "parent", "metrics"} <= x.keys() for x in rec["result"]["candidates"])
        assert (Path(d) / "ledger" / rec["result"]["trace"]).exists() and rec["config"]["split"] == "seen"
        from lab.kernels.evolve.run import retime
        code, metrics, retimed = retime(Path(d) / "oe" / "evolution_trace.jsonl", 2)
        assert "rsqrt" in code and metrics["correct"] == 1.0 and all(len(x["scores"]) == 3 for x in retimed)
        print("best", res.best_score)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)


def test_captures_with_no_call_of_the_task_are_refused_not_replaced_by_defaults(tmp_path):
    from types import SimpleNamespace
    from lab.kernels.evolve.run import _shapes
    (tmp_path / "calls.jsonl").write_text("")                   # no call seen
    try:
        _shapes(load("toy_rmsnorm"), SimpleNamespace(shapes=None, captures=[str(tmp_path)], equal_captures=False))
        raise AssertionError("empty captures fell back to the default shapes")
    except SystemExit as e:
        assert "no call" in str(e)
