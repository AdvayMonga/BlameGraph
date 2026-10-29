"""KernelBench problems as kernel tasks, on CPU with a small made-up problem (no download): inputs and weights, the
held-out size constant, the starting program, the judge on a correct rewrite and a wrong one, fast_p.
Run: python tests/test_kernelbench.py"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LAB_NO_JAIL", "1")

import torch  # noqa: E402

from kernels import kernelbench as kb  # noqa: E402
from kernels.evolve.core import judge, screen  # noqa: E402
from kernels.evolve.run import _target_tolerances, initial_program, system_message  # noqa: E402
from kernels.tasks import from_config, load  # noqa: E402

PROBLEM = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """A linear layer, a bias and a clamp at a scalar the inputs carry."""
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x, limit):
        return torch.clamp(self.linear(x) + self.bias, max=limit)

batch_size = 16
in_features = 32
out_features = 8
# a comment that says signal

def get_inputs():
    return [torch.rand(batch_size, in_features), 0.5]

def get_init_inputs():
    return [in_features, out_features]
'''
FAST = '''
class ModelNew(Model):
    def forward(self, x, limit):
        return torch.addmm(self.linear.bias + self.bias, x, self.linear.weight.t()).clamp(max=limit)
'''
WRONG = '''
class ModelNew(Model):
    def forward(self, x, limit):
        return self.linear(x)
'''


def task():
    return load("kb:9:1", source=PROBLEM)


def program(d: str, cls: str | None) -> Path:
    src = initial_program(task())
    if cls:
        head, tail = src.split("# EVOLVE-BLOCK-END")
        src = head.replace('class ModelNew(Model):\n    """What KernelBench scores: built like Model (same arguments, the '
                           'same parameters in the same order)."""\n', cls) + "# EVOLVE-BLOCK-END" + tail
    p = Path(d) / "program.py"
    p.write_text(src)
    return p


def test_inputs_are_the_problems_own_seeded_and_the_size_constant_scales_held_out():
    t = task()
    assert t.seen_default == ((1,),) and t.heldout_default == ((kb.HELDOUT_FACTOR,),)
    a = t.make_inputs((1,), torch.float32, "cpu", torch.Generator().manual_seed(3))
    b = t.make_inputs((1,), torch.float32, "cpu", torch.Generator().manual_seed(3))
    c = t.make_inputs((1,), torch.float32, "cpu", torch.Generator().manual_seed(4))
    assert len(a) == 1 and a[0].shape == (16, 32) and torch.equal(a[0], b[0]) and not torch.equal(a[0], c[0])
    assert t.make_inputs((2,), torch.float32, "cpu", torch.Generator())[0].shape == (32, 32)
    assert t.reference(*a).shape == (16, 8) and float(t.reference(*a).max()) <= 0.5      # the scalar is passed back in


def test_a_constant_the_constructor_reads_is_never_the_held_out_axis():
    src = PROBLEM.replace("return [torch.rand(batch_size, in_features), 0.5]",
                          "return [torch.rand(in_features, in_features), 0.5]")
    assert kb._size_constant(src, _ns(src)) is None
    assert load("kb:9:2", source=src).heldout_default == ()


def _ns(src):
    ns = {}
    exec(src, ns)
    return ns


def test_the_starting_program_is_the_baseline_and_passes_the_screen():
    src = initial_program(task())
    assert "class ModelNew(Model)" in src and "def kernel(" in src and "signal" not in src and not screen(src)
    with tempfile.TemporaryDirectory() as d:
        r = judge(task(), program(d, None), [(1,)], [(2,)], None, "cpu", _target_tolerances(task()), 2, iters=3)
        assert r["metrics"]["correct"] == 1.0, r


def test_the_judge_passes_a_correct_rewrite_and_zeroes_a_wrong_one():
    tol = _target_tolerances(task())
    assert tol == {"float32": [kb.TOLERANCE, kb.TOLERANCE]}
    with tempfile.TemporaryDirectory() as d:
        ok = judge(task(), program(d, FAST), [(1,)], [(2,)], None, "cpu", tol, 2, iters=3)
        assert ok["metrics"]["correct"] == 1.0 and ok["metrics"]["combined_score"] > 0, ok
        bad = judge(task(), program(d, WRONG), [(1,)], [], None, "cpu", tol, 2, iters=3)
        assert bad["metrics"]["combined_score"] == 0.0 and "correctness_seen" in bad["artifacts"]["gates"]


def test_a_jailed_judgment_reads_the_problem_from_its_config_and_verifies_outside():
    from kernels.evolve.jailed import judge_jailed
    with tempfile.TemporaryDirectory() as d:
        src = program(d, FAST).read_text()
    cfg = {"task": "kb:9:1", "source": PROBLEM, "seen": [[1]], "device": "cpu", "iters": 3,
           "tolerances": _target_tolerances(task())}
    r = judge_jailed(cfg, src, 2)
    assert r["metrics"]["correct"] == 1.0 and r["metrics"]["combined_score"] > 0, r["artifacts"]
    assert '"verified_outside"' in r["artifacts"]["gates"]


def test_the_eval_config_carries_the_problem_into_the_jail():
    t = from_config({"task": "kb:9:1", "source": PROBLEM})
    assert t.name == "kb:9:1" and t.nonfinite is False and t.layouts == () and t.scales == ()
    msg = system_message(t)
    assert "NaN" not in msg and "scaled by" not in msg and "eager PyTorch `Model`" in msg and "load_inline" in msg


def test_fast_p_counts_correct_and_faster_over_all_problems():
    rows = [{"correct": True, "correct_v01": True, "speedup_eager": 2.5, "speedup_compile": 0.9,
             "static": {"valid": False}},
            {"correct": True, "correct_v01": True, "speedup_eager": 1.2, "speedup_compile": 0.5,
             "static": {"valid": True}},
            {"correct": False, "correct_v01": True, "speedup_eager": None, "speedup_compile": None},
            {"error": "exited 1"}]
    s = kb.summary(rows)
    assert (s["fast_0_eager_pinned"], s["fast_1_eager_pinned"], s["fast_2_eager_pinned"]) == (0.5, 0.5, 0.25)
    assert s["fast_1_compile_pinned"] == 0.0 and s["n"] == 4 and s["fast_0_eager_v01"] == 0.5
    assert (s["fast_1_eager_pinned_static"], s["fast_2_eager_pinned_static"], s["static_valid"]) == (0.25, 0.0, 1)


def test_the_benchmarks_own_evaluation_imports_without_its_llm_clients():
    try:
        src = kb.root() / "src"
    except SystemExit:
        print("skip: KernelBench not fetched")
        return
    kb._import_kernelbench(src)
    from kernelbench import eval as kbe
    from kernelbench import kernel_static_checker
    assert kbe.get_tolerance_for_precision("fp32") == kb.TOLERANCE      # the pinned commit's, as the module states
    valid, errors, _ = kernel_static_checker.validate_kernel_static(initial_program(task()), precision="fp32")
    assert not valid and errors                                         # the start is PyTorch, not a kernel


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
