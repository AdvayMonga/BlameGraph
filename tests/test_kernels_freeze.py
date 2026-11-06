"""A winner's autotune frozen into a table before the gate and ship (evals/kernels/freeze.py), on CPU: the rewrite,
the nearest-key lookup, the freeze stage with a stand-in for Triton's autotuner, and ship's refusal of a live one.
Run: python tests/test_kernels_freeze.py"""
from __future__ import annotations

import ast
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LAB_TARGET", str(ROOT / "targets" / "inference-server.toml"))

import torch  # noqa: E402

from evals.kernels import freeze  # noqa: E402
from evals.kernels.core import judge, screen  # noqa: E402
from generate.kernels.tasks import load  # noqa: E402

TWO = '''import triton
import triton.language as tl

CONFIGS = [triton.Config({"BLOCK": 64}, num_warps=4)]


@triton.autotune(configs=CONFIGS, key=["M"])
@triton.jit
def a_kernel(x, M, BLOCK: tl.constexpr):
    pass


@triton.jit
def _b(x, M, N, BLOCK: tl.constexpr):
    pass


b_kernel = triton.autotune(
    configs=CONFIGS,
    key=["M", "N"],
)(_b)


def kernel(x):
    a_kernel[(1,)](x, x.shape[0])
    b_kernel[(1,)](x, x.shape[0], x.shape[1])
    return x
'''
TABLES = {"a_kernel": {"keys": ["M"], "table": [[[16, "torch.bfloat16"], {"BLOCK": 32, "num_warps": 4}],
                                                [[256, "torch.bfloat16"], {"BLOCK": 128, "num_warps": 8}]]},
          "b_kernel": {"keys": ["M", "N"], "table": [[[16, 768, "torch.bfloat16"], {"BLOCK": 64, "num_stages": 3}]]}}


def test_rewrite_freezes_decorator_and_call_forms_and_leaves_the_rest():
    out = freeze.rewrite(TWO, TABLES)
    ast.parse(out)
    assert not freeze.sites(out) and out.count("class _FrozenTune") == 1 and freeze.MARK in out
    assert "@triton.jit\ndef a_kernel" in out and "a_kernel = _FrozenTune(a_kernel, ['M'], [" in out
    assert "b_kernel = _FrozenTune(_b, ['M', 'N'], [" in out and "{'BLOCK': 128, 'num_warps': 8}" in out
    assert out.index("class _FrozenTune") < out.index("def a_kernel")       # bound before its first use
    assert not screen(out)                                                  # the evaluator's screen accepts it
    one = freeze.rewrite(TWO, {"a_kernel": TABLES["a_kernel"]})             # only what ran is frozen
    assert [s["name"] for s in freeze.sites(one)] == ["b_kernel"]
    assert freeze.rewrite(one, {"b_kernel": TABLES["b_kernel"]}).count("class _FrozenTune") == 1   # helper once
    plain = "import torch\n\ndef kernel(x):\n    return x\n"
    assert freeze.rewrite(plain, TABLES) == plain
    called = []
    assert freeze.freeze(plain, lambda s: called.append(s)) == (plain, {}) and not called   # never judged


def test_lookup_takes_the_tuned_key_else_the_smallest_above_else_the_closest():
    ns: dict = {}
    exec(freeze.HELPER, ns)                                                 # the helper as the frozen module has it
    pick = ns["_frozen_pick"]
    table = [[[m, 768, "torch.bfloat16"], {"BLOCK": m}] for m in (16, 64, 256)]
    assert pick(table, [64, 768, "torch.bfloat16"]) == {"BLOCK": 64}
    assert pick(table, [10, 768, "torch.bfloat16"]) == {"BLOCK": 16}
    assert pick(table, [65, 768, "torch.bfloat16"]) == {"BLOCK": 256}
    assert pick(table, [5000, 768, "torch.bfloat16"]) == {"BLOCK": 256}     # beyond the table: its largest
    assert pick(table, [1, 768, "torch.bfloat16"]) == {"BLOCK": 16}
    mixed = table + [[[64, 768, "torch.float16"], {"BLOCK": -1}]]
    assert pick(mixed, [64, 768, "torch.float16"]) == {"BLOCK": -1}         # dtypes match exactly first
    assert pick(mixed, [20, 768, "torch.float16"]) == {"BLOCK": -1}
    flags = [[[True, 16], {"S": 1}], [[False, 16], {"S": 2}]]
    assert pick(flags, [False, 300]) == {"S": 2}                            # a bool is matched, never ranked
    key = ns["_frozen_key"]
    x = torch.zeros(2, 3)
    assert key(["x", "M", "B"], ["M"], (x, 7), {"B": 2, "grid": (1,)}) == [7, "torch.float32"]


FAKE = '''import torch

LAUNCHES = []


class Config:
    def __init__(self, kwargs, num_warps=4):
        self.kwargs, self.num_warps, self.num_stages, self.pre_hook = kwargs, num_warps, 2, None


class Jit:
    def __init__(self, f):
        self.f, self.arg_names = f, list(f.__code__.co_varnames[:3])

    def __getitem__(self, grid):
        return lambda *a, **k: self.f(*a, **k)


class Tuner:
    """Triton's autotuner as far as the freeze stage sees it: configs, keys, fn, run, best_config."""
    def __init__(self, fn, configs, key):
        self.fn, self.configs, self.keys, self.arg_names = fn, configs, key, fn.arg_names

    def run(self, *args, grid=None, warmup=False, **kwargs):
        self.best_config = self.configs[0] if args[2] <= 16 else self.configs[1]
        c = self.best_config
        return self.fn[grid](*args, **kwargs, **c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)

    def __getitem__(self, grid):
        return lambda *a, **k: self.run(*a, grid=grid, warmup=False, **k)


def autotune(configs, key):
    return lambda fn: Tuner(fn, configs, key)


@autotune(configs=[Config({"BLOCK": 1}), Config({"BLOCK": 2}, num_warps=8)], key=["M"])
@Jit
def _rms(x, w, M, BLOCK, num_warps, num_stages):
    LAUNCHES.append((M, BLOCK, num_warps))
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)


def kernel(x, w):
    return _rms[(1,)](x, w, x.shape[0])
'''


def test_the_freeze_stage_records_each_choice_and_the_frozen_program_judges_and_launches_it():
    t = load("toy_rmsnorm")
    seen = list(t.seen_default)                                             # rows 8, 32, 64
    with tempfile.TemporaryDirectory() as d:
        prog = Path(d) / "p.py"
        prog.write_text(FAKE)
        r = judge(t, prog, seen, [], None, "cpu", None, freeze.FREEZE_STAGE)
        tables = json.loads(r["artifacts"]["frozen"])
        assert tables["_rms"]["keys"] == ["M"]
        assert sorted((k[0], c["BLOCK"], c["num_warps"]) for k, c in tables["_rms"]["table"]) == \
            [(8, 1, 4), (32, 2, 8), (64, 2, 8)]
        new, got = freeze.freeze(FAKE, lambda s: r)
        assert got == tables and not freeze.sites(new) and not screen(new)
        frozen = Path(d) / "frozen.py"
        frozen.write_text(new)
        ok = judge(t, frozen, seen, list(t.heldout_default), None, "cpu", None, 2, iters=3)   # the gate's checks
        assert ok["metrics"]["correct"] == 1.0 and ok["metrics"]["combined_score"] > 0, ok["artifacts"]
        from evals.kernels.core import load_module
        mod = load_module(frozen)
        for rows, block in ((4, 1), (8, 1), (10, 2), (64, 2), (500, 2)):    # tuned, between, beyond
            mod.kernel(torch.randn(rows, 16), torch.randn(16))
            assert mod.LAUNCHES[-1][:2] == (rows, block)
        bad = Path(d) / "bad.py"
        bad.write_text("x = 1\n")
        assert "error" in judge(t, bad, seen, [], None, "cpu", None, freeze.FREEZE_STAGE)["artifacts"]
    try:
        freeze.freeze(FAKE, lambda s: {"metrics": {}, "artifacts": {"error": "load failed"}})
        raise AssertionError("a failed freeze stage passed")
    except RuntimeError as e:
        assert "load failed" in str(e)
    loose = "import triton\n\ndef make(k):\n    return triton.autotune(configs=[], key=['M'])(k)\n"
    try:
        freeze.freeze(loose, lambda s: {})
        raise AssertionError("an autotune bound to no module-level name was frozen")
    except ValueError as e:
        assert "line 4" in str(e)


def test_pipeline_freezes_where_the_evaluator_runs_and_writes_the_frozen_program():
    from generate.kernels.tasks import pipeline
    t = load("toy_rmsnorm")
    calls = []

    def judged(task, caps, program, out, env, held, stage):
        calls.append(stage)
        return {"metrics": {"stage": stage}, "artifacts": {"frozen": json.dumps(TABLES)}}
    saved, pipeline._judged = pipeline._judged, judged
    try:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "best_program.py"
            p.write_text(TWO)
            got = pipeline.freeze(t, [], p, Path(d), {})
            assert got == Path(d) / "frozen_program.py" and not freeze.sites(got.read_text())
            assert calls == [freeze.FREEZE_STAGE]
            p.write_text("def kernel(x, w):\n    return x\n")
            assert pipeline.freeze(t, [], p, Path(d), {}) == p and calls == [freeze.FREEZE_STAGE]
    finally:
        pipeline._judged = saved


def test_ship_refuses_a_live_autotune_the_engine_did_not_have():
    from evals.kernels.ship import ship
    base = ("import triton\n\n@triton.jit\ndef _k(x, M, N):\n    pass\n\n"
            "tuned = triton.autotune(configs=C, key=['N'])(_k)\n\ndef entry(x):\n    return x\n")
    assert freeze.introduced(base.replace("key=['N']", "key=[ 'N' ]"), base) == []      # as written, spacing aside
    assert freeze.introduced(base.replace("key=['N']", "key=['M', 'N']"), base) == \
        ["triton.autotune(configs=C, key=['M', 'N'])"]
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "engine"
        (repo / "src" / "fake_pkg").mkdir(parents=True)
        (repo / "src" / "fake_pkg" / "mod.py").write_text(base)
        git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,  # noqa: E731
                                        check=True).stdout
        git("init", "-q")
        git("add", ".")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "engine")
        task = dataclasses.replace(load("toy_rmsnorm"), entry="fake_pkg.mod:entry")
        ok = {"metrics": {"correct": 1.0},
              "artifacts": {"gates": json.dumps({"correctness_heldout": {"passed": True, "cases": 6}})}}
        keyed_m = base.replace("key=['N']", "key=['M', 'N']")
        live = f"# EVOLVE-BLOCK-START\n{keyed_m}# EVOLVE-BLOCK-END\n\nkernel = entry\n"
        try:
            ship(task, live, ok, repo, branch="kernels/live")
            raise AssertionError("shipped a kernel that tunes while serving")
        except ValueError as e:
            assert "autotunes at run time" in str(e)
        old = dict(os.environ)
        os.environ.update(GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t", GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t")
        try:
            same = f"# EVOLVE-BLOCK-START\n{base}# EVOLVE-BLOCK-END\n\nkernel = entry\n"   # the engine's own, untouched
            assert ship(task, same, ok, repo, branch="kernels/same")["path"] == "src/fake_pkg/mod.py"
            frozen = freeze.rewrite(live, {"tuned": {"keys": ["M", "N"], "table": [[[16, 8], {"B": 1}]]}})
            assert ship(task, frozen, ok, repo, branch="kernels/frozen")["path"] == "src/fake_pkg/mod.py"
        finally:
            os.environ.clear()
            os.environ.update(old)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
