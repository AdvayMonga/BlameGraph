"""The layer-1 loop end to end on CPU, with no model calls: tasks, shape capture and split, the judge, and a real
OpenEvolve run whose "model" is a scripted stand-in that returns one known-good diff and one cheating diff.
Run: python tests/test_kernels_evolve.py (skips without openevolve)."""
from __future__ import annotations

import json
import os
import subprocess
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
                "torch.save(out, 'memo.pt')\n", "Path('m').write_bytes(b)\n", "import pickle\n", "np.load('m.npy')\n",
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


def test_the_screen_passes_a_cuda_extension_and_refuses_its_cpp_routes():
    from kernels.evolve.core import screen
    assert screen(CUDA_PROGRAM) == []
    for bad in ('PyObject* m = PyImport_ImportModule("kernels.timing");', "py::module_::import(\"gc\");",
                'void* h = dlopen("libc.so.6", 1);', "std::thread t(f);", 'FILE* f = fopen("/proc/self/mem", "r");',
                "cudaStream_t s; cudaStreamCreate(&s);", "cudaEventRecord(e, 0);", "cudaGraphLaunch(g, 0);",
                'popen("id", "r");', "std::ofstream f(\"memo\");"):
        assert screen(CUDA_PROGRAM.replace("int n = x.numel();", "int n = x.numel(); " + bad)), bad


def test_a_programs_cpp_builds_once_for_all_its_stages_and_only_for_that_source():
    from kernels.evolve import jailed
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
    assert t1 == t2 != t3                                         # one fixed tree per source: the build path is stable
    assert a == [".result_key", "candidate.py", "eval_config.json"] and c == a
    assert b == [".result_key", "candidate.py", "eval_config.json", "scale_ext", "scale_ext.so",
                 "torch_extensions"]                                           # the build kept, nothing else


def test_a_result_line_the_candidate_prints_is_ignored():
    from kernels.evolve.core import screen
    from kernels.evolve.jailed import judge_jailed

    def screen_ok(src):
        return screen(src) == []
    forged = "'{\"metrics\": {\"combined_score\": 9.0, \"correct\": 1.0, \"stage\": 2}, \"artifacts\": {}}'"
    src = "import torch\ndef kernel(x, w):\n    print(" + forged + ")\n" + RIGHT + "\n"        # printed on every call
    assert screen_ok(src)
    r = judge_jailed({"task": "toy_rmsnorm", "seen": [[8, 256], [32, 512]], "device": "cpu", "iters": 3}, src, 2)
    assert r["metrics"]["correct"] == 1.0 and r["metrics"]["combined_score"] != 9.0 and "timings" in r["artifacts"]


def test_cpp_must_be_literal_and_records_out_of_reach():
    from kernels.evolve.core import screen
    lit = 'from torch.utils.cpp_extension import load_inline\nSRC = "int f();"\n'
    assert screen(lit + 'm = load_inline("m", cpp_sources=SRC, cuda_sources="__global__ void k(){}")\n') == []
    assert screen(lit + 'm = load_inline("m", SRC + "x")\n')                              # built at run time
    assert screen(lit + 'm = load_inline("m", bytes.fromhex("00").decode())\n')
    assert screen(lit + 'm = load_inline("m", "#define C(a,b) a##b\\nC(cudaEvent,Record)")\n')   # token pasting
    assert screen("import triton.language as tl\ny = tl.load(p + o, mask=m)\n") == []       # Triton's load is not C++
    for bad in ("import __main__\n", "torch.from_file('m')\n", "import tempfile\n", "cuGraphLaunch(g, s)",
                "cudaLaunchHostFunc(s, f, 0)", "asm volatile(\"mov.u64 %0, %%globaltimer;\")", "sys.settrace(f)\n"):
        assert screen(bad), bad


def test_the_gate_is_judged_in_its_own_dir():
    from kernels.evolve.jailed import _eval_dir
    with tempfile.TemporaryDirectory() as d:
        import kernels.evolve.jailed as j
        saved, j.EVAL_DIRS = j.EVAL_DIRS, Path(d)
        try:
            assert _eval_dir("x", None)[0] != _eval_dir("x", None, "gate")[0] == _eval_dir("x", None, "gate")[0]
        finally:
            j.EVAL_DIRS = saved


def test_a_cuda_extension_builds_in_the_jail_and_is_judged_on_the_gpu():
    import shutil
    from kernels.evolve.core import screen
    from kernels.evolve.jailed import judge_jailed
    src = (ROOT / "tests" / "fixtures" / "cuda_toy_rmsnorm.py").read_text()
    assert screen(src) == []
    if not torch.cuda.is_available() or not (shutil.which("nvcc") or Path("/usr/local/cuda/bin/nvcc").exists()):
        print("skip: no CUDA GPU or no nvcc")
        return
    cfg = {"task": "toy_rmsnorm", "seen": [[8, 256], [32, 512], [64, 1024]], "device": "cuda", "iters": 10}
    r = judge_jailed(cfg, src, 2)
    assert r["metrics"]["correct"] == 1.0 and r["metrics"]["combined_score"] > 0, r["artifacts"]


def test_ncu_counters_reach_the_judge_on_the_gpu():
    import shutil
    from kernels.evolve.jailed import judge_jailed
    if not torch.cuda.is_available() or not shutil.which("ncu"):
        print("skip: no CUDA GPU or no ncu")
        return
    src = (ROOT / "tests" / "fixtures" / "cuda_toy_rmsnorm.py").read_text()
    cfg = {"task": "toy_rmsnorm", "seen": [[8, 256], [64, 1024]], "device": "cuda", "iters": 3}
    a = judge_jailed(cfg, src, 3)["artifacts"]
    assert "ncu" in a and "ERR_NVGPUCTRPERM" not in json.dumps(a), a
    shapes = json.loads(a["ncu"])["shapes"]
    assert len(shapes) == 2 and all(s.get("kernels") for s in shapes), shapes      # counters on both shapes
def test_the_fused_tasks_start_launches_what_the_engine_module_launches():
    """The start edits only change how the kernels module is reached: with a stand-in kernels module, the original
    fused.py and the edited start make the same launches with the same arguments, and the start passes the screen."""
    import types
    from kernels.evolve.core import screen
    from kernels.evolve.run import _start
    from lab import engine
    try:
        src = (engine.repo() / "src" / "inference_server" / "models" / "fused.py").read_text()
    except Exception:                            # noqa: BLE001  no engine checkout here (CI)
        print("skip: no engine source")
        return
    log = []

    class Launch:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *a, **k: log.append((self.name, grid, [tuple(x.shape) if hasattr(x, "shape") else x
                                                                 for x in a], sorted(k.items())))
    stub = types.ModuleType("inference_server.models.fused_kernels")
    stub.triton = types.SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length(), cdiv=lambda a, b: -(-a // b))
    for name in ("_rmsnorm_kernel", "_add_rmsnorm_kernel", "_rmsnorm_rope_kernel", "_silu_mul_kernel",
                 "_kv_store_kernel", "_topk_softmax_kernel", "_add_cast_kernel"):
        setattr(stub, name, Launch(name))
    saved = {k: sys.modules.get(k) for k in ("inference_server", "inference_server.models",
                                             "inference_server.models.fused_kernels")}
    pkg, models = types.ModuleType("inference_server"), types.ModuleType("inference_server.models")
    pkg.models, models.fused_kernels = models, stub
    sys.modules.update({"inference_server": pkg, "inference_server.models": models,
                        "inference_server.models.fused_kernels": stub})
    try:
        runs = []
        for task in ("rmsnorm", "silu_mul", "topk_softmax"):
            start = _start(load(task), src, "x").split("# EVOLVE-BLOCK-END")[0].replace("# EVOLVE-BLOCK-START\n", "")
            assert screen(start) == [], (task, screen(start))
        import importlib.util
        find, enabled = importlib.util.find_spec, []
        importlib.util.find_spec = lambda name, *a: object() if name == "triton" else find(name, *a)   # triton present
        try:
            for code in (src, start):
                ns, log[:] = {"__name__": "fused_under_test"}, []
                exec(compile(code, "fused.py", "exec"), ns)             # noqa: S102  the module under test
                bf = dict(dtype=torch.bfloat16)
                x, w = torch.randn(4, 64, **bf), torch.ones(64, **bf)
                ns["rmsnorm"](x, w, 1e-6), ns["add_rmsnorm"](x, x.clone(), w, 1e-6)
                q, cs = torch.randn(1, 2, 4, 16, **bf), torch.randn(1, 2, 16, **bf)
                ns["rmsnorm_rope"](q, torch.ones(16, **bf), cs, cs.clone(), 1e-6)
                ns["silu_mul"](torch.randn(4, 128, **bf))
                ns["topk_softmax"](torch.randn(4, 128, **bf), 8, True, fp32=True)
                runs.append(list(log))
                enabled.append(ns["ENABLED"])
        finally:
            importlib.util.find_spec = find
    finally:
        for k, v in saved.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)
    assert len(runs[0]) == 5 and runs[0] == runs[1] and enabled == [True, True]   # same launches, args, switch
    from kernels.evolve.ship import unstart
    assert unstart(load("rmsnorm"), start).rstrip() == src.rstrip()      # shipping undoes the edits
def test_the_fp8_moe_tasks_judge_the_engines_calls():
    """References against an independent per-pair computation; an engine-shaped kernel passes every gate and the value
    scalings (fp8 weights kept, their scales scaled); a 5% scale error fails; each task reads only its own calls."""
    from kernels.check import _gen, check_kernel
    from kernels.evolve.core import _scaled
    from kernels.tasks.moe_gemm import _block_m
    t, u = load("moe_gemm_fp8"), load("moe_gemm_scatter_fp8")

    def pairs(sid, eid, M, E):
        bm, out = _block_m(sid, eid), {}
        for blk, e in enumerate(eid.tolist()):
            out.update({r: e for r in sid[blk * bm:(blk + 1) * bm].tolist() if r < M and e < E})
        return out
    a, tok, w, sc, sid, eid = t.make_inputs(t.seen_default[1], torch.bfloat16, "cpu", _gen(0))
    pe, wd = pairs(sid, eid, tok.numel(), w.shape[0]), w.float() * sc[..., None]
    want = torch.stack([a[tok[r]].float() @ wd[pe[r]].T for r in range(tok.numel())])
    assert w.dtype == torch.float8_e4m3fn and (t.reference(a, tok, w, sc, sid, eid).float() - want).abs().max() < 2e-2
    a, w, sc, sid, eid, tok, pw, out0 = u.make_inputs(u.seen_default[1], torch.bfloat16, "cpu", _gen(1))
    pe, wd, want = pairs(sid, eid, a.shape[0], w.shape[0]), w.float() * sc[..., None], torch.zeros_like(out0)
    for r in range(a.shape[0]):
        want[tok[r]] += pw[r] * (a[r].float() @ wd[pe[r]].T)
    assert (u.reference(a, w, sc, sid, eid, tok, pw, out0) - want).abs().max() < 1e-3

    def gemm(a, tok, w, sorted_ids, expert_ids, M, block_m=16, split_k=None, w_scale=None):
        return t.reference(a, tok if tok is not None else torch.empty(0, dtype=torch.int32), w, w_scale, sorted_ids,
                           expert_ids)

    def scatter(a, w, sorted_ids, expert_ids, M, tok, pair_w, out, block_m=16, w_scale=None):
        out += u.reference(a, w, w_scale, sorted_ids, expert_ids, tok, pair_w, out)
    for task, fn in ((t, gemm), (u, scatter)):
        r = check_kernel(task.reference, task.bind(fn), task.make_inputs, list(task.seen_default),
                         list(task.heldout_default), dtypes=task.dtypes, edge_shapes=task.edge_shapes(task.seen_default[0]),
                         layouts=())
        assert r["passed"], (task.name, r["reasons"])
        assert check_kernel(task.reference, task.bind(fn), _scaled(task.make_inputs, -1.0), list(task.seen_default[:2]),
                            [], dtypes=task.dtypes, edge_shapes=[], layouts=())["passed"]
    off = lambda a, tok, w, s, e, M, block_m=16, split_k=None, w_scale=None: gemm(a, tok, w, s, e, M, w_scale=w_scale * 1.05)  # noqa: E731
    assert not check_kernel(t.reference, t.bind(off), t.make_inputs, list(t.seen_default), [], dtypes=t.dtypes,
                            edge_shapes=[], layouts=())["passed"]
    arg = lambda shape, dtype: {"shape": shape, "dtype": dtype}                                       # noqa: E731
    ints = {**arg([32], "int32"), "values": []}
    fp8 = {"args": [arg([4, 2048], "bfloat16"), ints, arg([128, 1536, 2048], "float8_e4m3fn"), ints, ints, {"value": 32}],
           "kwargs": {"w_scale": arg([128, 1536], "float32")}}
    bf16 = {"args": fp8["args"][:2] + [arg([128, 1536, 2048], "bfloat16")] + fp8["args"][3:], "kwargs": {}}
    assert t.shape_of(fp8) == load("moe_gemm").shape_of(bf16) == (32, 2048, 1536, 128, 4, 1)
    assert t.shape_of(bf16) is None and load("moe_gemm").shape_of(fp8) is None
def test_the_evaluators_modules_are_out_of_reach():
    from kernels.evolve import core
    for bad in ("import sys as s\n", "from sys import modules as M\n", "import os, sys\n", "import os\nm = os.sys\n",
                "import atexit\n", "import weakref\n", "import torch\nwith torch.profiler.profile(): pass\n",
                "import pynvml\n", 'C = "cuptiSubscribe(&s, cb, 0);"\n', "import torch\nD = torch.ops\nD['x'] = 1\n",
                "import os\nos.environ['TORCH_ALLOW_TF32'] = '1'\n"):
        assert core.screen(bad), bad
    assert core.screen("def k(x):\n    out = x.clone()\n    out[0] = 1\n    return out\n") == []   # its own tensors
    t = load("toy_rmsnorm")
    before = core._watched(t)
    real = sys.modules["kernels.check"]
    sys.modules["kernels.check"] = type(sys)("kernels.check")                    # a module swapped under its name
    try:
        assert "sys.modules[kernels.check]" in core._patched(before, t)
    finally:
        sys.modules["kernels.check"] = real
    assert core._patched(before, t) == [] and core._compare is real._compare      # compare bound at import


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


def test_paged_decode_tables_are_as_wide_as_the_engines_graphs():
    from kernels.check import _gen
    from kernels.tasks import from_config
    t, shape = load("paged_decode"), (3, 8, 2, 16, 16, 40)
    wide = from_config({"task": "paged_decode", "options": {"context_window": 32768}})
    narrow_in, wide_in = t.make_inputs(shape, torch.float32, "cpu", _gen(0)), wide.make_inputs(shape, torch.float32,
                                                                                                "cpu", _gen(0))
    assert narrow_in[3].shape == (3, 3) and wide_in[3].shape == (3, 2048) and not wide_in[3][:, 3:].any()
    assert torch.equal(t.reference(*narrow_in), t.reference(*wide_in))        # same answer, the engine's layout
    real = wide.with_ints({json.dumps(list(shape)): [[40, 7, 33]]})
    assert real.make_inputs(shape, torch.float32, "cpu", _gen(0))[3].shape == (3, 2048)


def test_paged_decode_checks_with_no_edge_shapes_do_not_fall_back_to_the_defaults():
    from kernels import check_kernel
    t = load("paged_decode")                    # its default edge shapes (Hq = 1 < Hkv) are not valid GQA inputs
    r = check_kernel(t.reference, t.reference, t.make_inputs, [(2, 8, 2, 16, 4, 11)], [], dtypes=(torch.float32,),
                     edge_shapes=[], repeats=1, layouts=())
    assert r["passed"]


def test_split_holds_out_a_region_even_when_calls_pile_up_at_the_largest_value():
    from collections import Counter
    from kernels.tasks import randn
    t = load("moe_gemm")
    counts = Counter({(m, 64, 96, 8, m, 1): (1000 if m == 512 else 3) for m in range(8, 513, 8)})
    s = shapes.split(t, counts)
    assert s["cut_by"] == "distinct" and s["heldout"] and (512, 64, 96, 8, 512, 1) in s["seen"]   # the top stays seen
    assert max(x[0] for x in s["seen"] if x[0] != 512) < min(x[0] for x in s["heldout"])          # a whole region
    two = Counter({(n, 32, 4, 128, 16, L): (900 if (n, L) == (33, 8032) else 10)                  # 2026-10-19 regate
                   for n in range(1, 257, 8) for L in (650, 8032)})
    two.update({(n, 32, 4, 128, 16, 8032): 300 for n in range(200, 257, 8)})
    s2 = shapes.split(load("paged_decode"), two)
    held = sum(two[x] for x in s2["heldout"])
    assert s2["heldout"] and held <= sum(two.values()) / 2 and (33, 32, 4, 128, 16, 8032) in s2["seen"]
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert torch.equal(randn((4, 5), g1, "cpu", torch.float32), randn((4, 5), g2, "cpu", torch.float32))


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
            {"name": "_gqa_decode_splitk_kernel", "count": 10, "total_us": 30_000.0},          # #106's names
            {"name": "_splitk_combine_vec_kernel", "count": 10, "total_us": 20_000.0},
            {"name": "rms_norm_kernel", "count": 10, "total_us": 20_000.0}]}))
        r = shares(b, [load("paged_decode"), load("toy_rmsnorm")])
        pd = r["tasks"]["paged_decode"]
        assert abs(pd["share_of_window"] - 0.2) < 1e-9 and abs(pd["speedup_if_free"] - 1.25) < 1e-9
        assert len(pd["kernels"]) == 3 and r["tasks"]["toy_rmsnorm"]["share_of_window"] == 0
        assert [k["name"] for k in r["unmatched"]] == ["fused_moe_kernel", "rms_norm_kernel"]
        assert main(["share", "--profile", str(b)]) == 0


def test_ship_commits_the_module_on_a_new_engine_branch_without_touching_its_tree():
    import dataclasses
    import subprocess
    from kernels.evolve.run import initial_program
    from kernels.evolve.ship import module_source, ship
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "engine"
        (repo / "src" / "fake_pkg").mkdir(parents=True)
        (repo / "src" / "fake_pkg" / "mod.py").write_text("def entry(x):\n    return x\n")
        git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout
        git("init", "-q")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "base", "--allow-empty")
        git("add", ".")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "engine")
        (repo / "src" / "fake_pkg" / "mod.py").write_text("def entry(x):\n    return x  # the user's edit\n")
        task = dataclasses.replace(load("toy_rmsnorm"), entry="fake_pkg.mod:entry")
        program = ("# EVOLVE-BLOCK-START\ndef entry(x):\n    return x\n\ndef faster(x):\n    return x * 1\n"
                   "# EVOLVE-BLOCK-END\n\nkernel = faster\n")
        assert module_source(program, "entry").endswith("entry = faster\n")
        assert "EVOLVE" not in module_source(initial_program(load("toy_rmsnorm")), "kernel")
        ok = {"metrics": {"correct": 1.0},
              "artifacts": {"gates": json.dumps({"correctness_heldout": {"passed": True, "cases": 6},
                                                 "physics": {"passed": True}})}}
        assert module_source("def kernel(x):\n    return x\n", "entry").endswith("entry = kernel\n")    # a rewrite
        old = dict(os.environ)
        os.environ.update(GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t", GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t")
        try:
            out = ship(task, program, ok, repo, branch="kernels/test")
        finally:
            os.environ.clear()
            os.environ.update(old)
        assert out["path"] == "src/fake_pkg/mod.py" and "entry = faster" in git("show", "kernels/test:src/fake_pkg/mod.py")
        assert git("diff", "--name-only", out["base"], "kernels/test").split() == ["src/fake_pkg/mod.py"]
        assert "the user's edit" in (repo / "src" / "fake_pkg" / "mod.py").read_text()       # working tree untouched
        assert git("rev-parse", "--abbrev-ref", "HEAD").strip() != "kernels/test"
        from kernels.evolve.ship import gate_passed
        flagged = json.loads(ok["artifacts"]["gates"])
        assert not gate_passed({**ok, "artifacts": {"gates": json.dumps({**flagged, "physics": {"passed": False}})}})
        assert not gate_passed({**ok, "artifacts": {"gates": json.dumps(
            {**flagged, "correctness_heldout": {"passed": True, "cases": 0}})}})      # held-out set empty
        try:
            ship(task, program, {"metrics": {"correct": 1.0}, "artifacts": {"gates": "{}"}}, repo, branch="kernels/x")
            raise AssertionError("shipped without a held-out gate")
        except ValueError:
            pass


def test_moe_gemm_task_reference_routing_capture_and_judge():
    import dataclasses
    from kernels.tasks import moe_gemm as mg
    g = torch.Generator().manual_seed(0)
    M, K, N, E, T = 37, 32, 24, 8, 9
    sorted_e = torch.sort(torch.randint(0, E, (M,), generator=g)).values
    tok, sorted_ids, expert_ids = mg._routing(M, E, T, 1, g, sorted_e)
    assert sorted_ids.numel() == M + E * 15 and int((sorted_ids < M).sum()) == M
    a, w = torch.randn(T, K, generator=g), torch.randn(E, N, K, generator=g)
    out = mg.reference(a, tok, w, sorted_ids, expert_ids)
    loop = torch.stack([a[tok[i]] @ w[sorted_e[i]].T for i in range(M)])
    assert torch.allclose(out, loop, atol=1e-4)
    rows = torch.randn(M, K, generator=g)                                   # down projection: rows given directly
    empty = torch.empty(0, dtype=torch.int32)
    assert torch.allclose(mg.reference(rows, empty, w, sorted_ids, expert_ids),
                          torch.stack([rows[i] @ w[sorted_e[i]].T for i in range(M)]), atol=1e-4)
    seen = {}
    mg._call(lambda *x, **k: seen.update(args=x, kw=k) or torch.zeros(M, N), rows, empty, w, sorted_ids, expert_ids)
    padded = torch.full((sorted_ids.numel() // 16 * 16,), M, dtype=torch.int32)        # align_fused's exact padding
    assert padded.numel() // expert_ids.numel() == 16
    f32 = mg._call(lambda *x, **k: torch.zeros(M, N, dtype=torch.float32), rows.bfloat16(), empty, w, sorted_ids,
                   expert_ids)
    assert f32.dtype == torch.bfloat16                                                  # split-K's fp32, compared as bf16
    assert seen["args"][1] is None and seen["args"][5] == M and seen["kw"] == {"block_m": 16}
    call = {"fn": "x", "args": [{"shape": [T, K]}, {"shape": [M], "values": tok.tolist()}, {"shape": [E, N, K]},
                                {"shape": [sorted_ids.numel()], "values": sorted_ids.tolist()},
                                {"shape": [expert_ids.numel()], "values": expert_ids.tolist()}, {"value": M}],
            "kwargs": {}}
    t = load("moe_gemm")
    shape = t.shape_of(call)
    assert shape == (M, K, N, E, T, 1)
    assert t.shape_of({**call, "args": [{"shape": [M, K]}, {"value": None}, *call["args"][2:]]}) == (M, K, N, E, M, 0)
    ints = [x["values"] for x in call["args"] if "values" in x]
    real = t.with_ints({json.dumps(list(shape)): ints}).make_inputs(shape, torch.float32, "cpu", g)
    assert real[1].tolist() == tok.tolist() and real[3].tolist() == sorted_ids.tolist()
    assert t.cost(shape, torch.bfloat16, real)[1] <= t.cost(shape, torch.bfloat16)[1] and t.graphs
    toy = dataclasses.replace(t, entry=None)                    # the engine's Triton kernel needs CUDA: reference baseline
    with tempfile.TemporaryDirectory() as d:
        prog = Path(d) / "k.py"
        prog.write_text("import torch\n\n\ndef kernel(a, tok, w, sorted_ids, expert_ids, M, block_m=16):\n"
            "    bm = block_m\n    E, N, _ = w.shape\n"
            "    expert = torch.full((sorted_ids.numel(),), E, dtype=torch.long)\n"
            "    expert[: expert_ids.numel() * bm] = expert_ids.long().repeat_interleave(bm)\n"
            "    real = (sorted_ids < M) & (expert < E)\n    ids, ex = sorted_ids[real].long(), expert[real]\n"
            "    rows = a[tok[ids]] if tok is not None else a[ids]\n"
            "    out = torch.empty(M, N, dtype=torch.float32)\n"
            "    out[ids] = torch.einsum('mk,mnk->mn', rows.float(), w[ex].float())\n    return out.to(a.dtype)\n")
        r = judge(toy, prog, list(t.seen_default), list(t.heldout_default), None, "cpu", None, 2, iters=3)
    assert r["metrics"]["correct"] == 1.0, r["artifacts"]


def test_the_starting_program_can_come_from_an_older_engine_commit():
    import dataclasses
    import subprocess
    from kernels.evolve.run import initial_program_at
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "engine"
        (repo / "src" / "fake_pkg").mkdir(parents=True)
        f = repo / "src" / "fake_pkg" / "mod.py"
        git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                                        capture_output=True, text=True, check=True).stdout
        git("init", "-q")
        f.write_text("def entry(x):\n    return x  # old\n")
        git("add", ".")
        git("commit", "-qm", "old")
        f.write_text("def entry(x):\n    return x  # new\n")
        git("commit", "-qam", "new")
        task = dataclasses.replace(load("toy_rmsnorm"), entry="fake_pkg.mod:entry")
        prog = initial_program_at(task, repo, "HEAD~1")
    assert "# old" in prog and "# new" not in prog and prog.rstrip().endswith("kernel = entry")


def test_the_fused_kernel_tasks_check_their_own_references_and_read_captured_shapes():
    from kernels import check_kernel
    from kernels.tasks import rmsnorm_rope
    for name in ("rmsnorm", "add_rmsnorm", "rmsnorm_rope", "silu_mul", "topk_softmax"):
        t = load(name)
        r = check_kernel(t.reference, lambda *x, t=t: t.reference(*x), t.make_inputs, list(t.seen_default)[:2],
                         list(t.heldout_default), dtypes=t.dtypes, seeds=(0,), repeats=1, layouts=(),
                         edge_shapes=t.edge_shapes(t.seen_default[0]))
        assert r["passed"], (name, r["reasons"])
        inputs = t.make_inputs(t.seen_default[0], torch.bfloat16, "cpu", torch.Generator().manual_seed(0))
        call = {"fn": t.entry, "args": [{"shape": list(x.shape)} for x in inputs], "kwargs": {}}
        assert t.shape_of(call) == tuple(t.seen_default[0]), name
        assert all(v > 0 for v in t.cost(t.seen_default[0], torch.bfloat16)) and t.graphs and t.kernel_names
    x, w, cos, sin = rmsnorm_rope.make_inputs((2, 3, 4, 128), torch.float32, "cpu", torch.Generator().manual_seed(1))
    x32 = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w                    # the HF formula, fp32
    half = 64
    hf = x32 * cos[:, :, None] + torch.cat((-x32[..., half:], x32[..., :half]), -1) * sin[:, :, None]
    assert torch.allclose(rmsnorm_rope.reference(x, w, cos, sin), hf, atol=1e-4)
    t = load("topk_softmax")
    w, ids = t.reference(torch.randn(5, 128, generator=torch.Generator().manual_seed(2)))
    assert ids.dtype == torch.int64 and ids.shape == (5, 8) and torch.allclose(w.sum(-1), torch.ones(5))
    assert t.scales == (3.0, -1.0)


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
    from kernels.evolve import evaluator
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


def test_outputs_are_verified_again_outside_the_candidates_process():
    from kernels.check import _gen
    from kernels.evolve.jailed import judge_jailed
    from kernels.evolve.verify import verify
    t = load("toy_rmsnorm")
    cfg = {"task": t.name, "seen": [list(s) for s in t.seen_default], "heldout": [list(s) for s in t.heldout_default],
           "device": "cpu", "iters": 3}
    r = judge_jailed(cfg, "import torch\ndef kernel(x, w):\n" + RIGHT, 2)
    g = json.loads(r["artifacts"]["gates"])["verified_outside"]
    assert r["metrics"]["correct"] == 1.0 and g["passed"] and "recomputed outside" in g["reason"]
    from kernels.evolve.core import _scaled
    from kernels.evolve.verify import expected
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
        rec = record(t, {"seen": list(t.seen_default)}, [{"name": "scripted", "weight": 1.0}], 4, "cpu",
                     Path(d) / "oe" / "evolution_trace.jsonl", best, res.metrics, root=Path(d) / "ledger")
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
