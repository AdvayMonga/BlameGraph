"""FlashInfer-Bench definitions as kernel tasks, on CPU with a small made-up dataset (no download): workloads as
shapes, recorded tensors read from safetensors, each workload's scalars, the refusals, the starting program,
the judge on a correct rewrite and a wrong one, the solution and TraceSet written for the benchmark's own evaluator,
the leaderboard's score. Run: python tests/test_flashinfer_bench.py"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LAB_NO_JAIL", "1")
os.environ.setdefault("LAB_TARGET", str(ROOT / "targets" / "inference-server.toml"))

import torch  # noqa: E402
from safetensors.torch import save  # noqa: E402

from generate.kernels.tasks import flashinfer_bench as fib  # noqa: E402
from evals.kernels.core import judge, screen  # noqa: E402
from generate.kernels.evolve.run import _target_tolerances, initial_program, system_message  # noqa: E402
from generate.kernels.tasks import from_config, load  # noqa: E402

NAME = "gather_scale_h8"
DEFINITION = {
    "name": NAME, "op_type": "gather", "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": 8}},
    "inputs": {"x": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"},
               "idx": {"shape": ["batch_size"], "dtype": "int32"},
               "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"},
               "scale": {"shape": None, "dtype": "float32"}},
    "outputs": {"out": {"shape": ["batch_size", "hidden_size"], "dtype": "bfloat16"}},
    "reference": "import torch\n\n@torch.no_grad()\ndef run(x, idx, weight, scale):\n"
                 "    y = x.to(torch.float32)[idx.long()] * weight.to(torch.float32) * scale\n"
                 "    return y.to(x.dtype)",
}
BATCHES = [1, 2, 3, 4, 6, 8, 8, 12, 16, 32]           # 8 twice: two workloads with equal axes
FAST = '''
@torch.no_grad()
def run(x, idx, weight, scale):
    return (x.float().index_select(0, idx.long()) * (weight.float() * scale)).to(x.dtype)
'''
WRONG = '''
@torch.no_grad()
def run(x, idx, weight, scale):
    return x * weight
'''


def dataset(d: Path, definition: dict = DEFINITION, scales=None) -> Path:
    """A TraceSet-shaped dataset dir with a manifest, as `fib.fetch` leaves one."""
    files = {}

    def put(rel: str, data: bytes) -> None:
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(data)
        files[rel] = hashlib.sha256(data).hexdigest()

    name, op = definition["name"], definition["op_type"]
    put(f"definitions/{op}/{name}.json", json.dumps(definition).encode())
    lines = []
    for i, b in enumerate(BATCHES):
        blob = f"blob/workloads/{op}/{name}/{name}_{i}.safetensors"
        put(blob, save({"idx": torch.arange(b, dtype=torch.int32).flip(0)}))
        inputs = {"x": {"type": "random"}, "idx": {"type": "safetensors", "path": f"./{blob}", "tensor_key": "idx"},
                  "scale": {"type": "scalar", "value": (scales or [0.5])[i % len(scales or [0.5])]}}
        lines.append(json.dumps({"definition": name, "solution": None, "evaluation": None,
                                 "workload": {"uuid": f"w{i}", "axes": {"batch_size": b}, "inputs": inputs}}))
    put(f"workloads/{op}/{name}.jsonl", ("\n".join(lines) + "\n").encode())
    base = {"name": "flashinfer_wrapper_0", "definition": name, "author": "baseline",
            "spec": {"language": "python", "target_hardware": ["cuda"], "entry_point": "main.py::run",
                     "destination_passing_style": False}, "sources": [{"path": "main.py", "content": "def run(): pass"}]}
    put(f"solutions/baseline/{op}/{name}/flashinfer_wrapper_0.json", json.dumps(base).encode())
    (d / "manifest.json").write_text(json.dumps({"dataset": fib.DATASET, "revision": fib.REVISION, "files": files}))
    return d


def program(d: str, body: str | None, t) -> Path:
    src = initial_program(t)
    if body:
        head, tail = src.split("# EVOLVE-BLOCK-END")
        src = head.split("# EVOLVE-BLOCK-START")[0] + "# EVOLVE-BLOCK-START\nimport torch\n" + body + \
            "# EVOLVE-BLOCK-END" + tail
    p = Path(d) / "program.py"
    p.write_text(src)
    return p


def test_workloads_are_shapes_with_recorded_tensors_and_their_scalars():
    with tempfile.TemporaryDirectory() as d:
        t = load(f"fib:{NAME}", samples=str(dataset(Path(d))))
        assert len(t.seen_default) == len(BATCHES) and (8, 0) in t.seen_default and (8, 1) in t.seen_default
        a = t.make_inputs((3, 0), torch.bfloat16, "cpu", torch.Generator().manual_seed(1))
        b = t.make_inputs((3, 0), torch.bfloat16, "cpu", torch.Generator().manual_seed(1))
        c = t.make_inputs((3, 0), torch.bfloat16, "cpu", torch.Generator().manual_seed(2))
        assert [x.shape for x in a] == [(3, 8), (3,), (8,), ()] and a[3].device.type == "cpu"
        assert torch.equal(a[1], torch.tensor([2, 1, 0], dtype=torch.int32))  # recorded, not drawn
        assert torch.equal(a[0], b[0]) and not torch.equal(a[0], c[0])
        out = t.reference(*a)
        assert torch.allclose(out.float(), (a[0].float()[[2, 1, 0]] * a[2].float() * 0.5).bfloat16().float())
        assert t.tolerances == {"bfloat16": [fib.ATOL / 2, fib.RTOL / 2]} and t.nonfinite is False


def test_the_held_out_band_is_never_the_largest_workload():
    with tempfile.TemporaryDirectory() as d:
        sets = fib.shape_sets(load(f"fib:{NAME}", samples=str(dataset(Path(d)))))
        assert sets["heldout"] and (32, 0) in sets["seen"] and not set(sets["seen"]) & set(sets["heldout"])


def test_each_workload_keeps_its_own_scalars_and_a_program_gets_python_numbers():
    seen = []
    with tempfile.TemporaryDirectory() as d:
        t = load(f"fib:{NAME}", samples=str(dataset(Path(d), scales=[0.5, 0.25])))
        for shape in ((1, 0), (2, 0)):
            t.bind(lambda *xs: seen.append(xs[3]))(*t.make_inputs(shape, torch.bfloat16, "cpu", torch.Generator()))
    assert seen == [0.5, 0.25] and all(type(x) is float for x in seen)


def test_definitions_another_evaluator_judges_are_refused():
    with tempfile.TemporaryDirectory() as d:
        for defn, why in (({**DEFINITION, "op_type": "sampling"}, "distribution"),
                          ({**DEFINITION, "name": "moe_fp8_block_scale_x"}, "matched-element")):
            p = Path(d) / defn["name"] / defn["op_type"]
            dataset(p, defn)
            try:
                load(f"fib:{defn['name']}", samples=str(p))
                raise AssertionError(f"{defn['name']} was not refused")
            except ValueError as e:
                assert why in str(e), e


def test_the_judge_passes_a_correct_rewrite_and_zeroes_a_wrong_one():
    with tempfile.TemporaryDirectory() as d:
        t = load(f"fib:{NAME}", samples=str(dataset(Path(d) / "ds")))
        assert not screen(initial_program(t)) and "def kernel(" in initial_program(t)
        tol, seen = _target_tolerances(t), [(8, 0), (3, 0)]
        for body, ok in ((None, True), (FAST, True), (WRONG, False)):
            r = judge(t, program(d, body, t), seen, [(16, 0)], None, "cpu", tol, 2, iters=3)
            assert (r["metrics"]["correct"] == 1.0) == ok, (body, r["artifacts"].get("gates"))
        assert r["metrics"]["combined_score"] == 0.0 and "correctness_seen" in r["artifacts"]["gates"]


def test_a_jailed_judgment_mounts_the_dataset_and_verifies_outside():
    from evals.kernels.jailed import judge_jailed
    with tempfile.TemporaryDirectory() as d:
        ds = dataset(Path(d) / "ds")
        t = load(f"fib:{NAME}", samples=str(ds))
        src = program(d, FAST, t).read_text()
        cfg = {"task": f"fib:{NAME}", "samples": str(ds), "seen": [[8, 0]], "device": "cpu", "iters": 3,
               "tolerances": _target_tolerances(t)}
        r = judge_jailed(cfg, src, 2)
        assert r["metrics"]["correct"] == 1.0 and r["metrics"]["combined_score"] > 0, r["artifacts"]
        assert '"verified_outside"' in r["artifacts"]["gates"]
        t2 = from_config({"task": f"fib:{NAME}", "samples": str(ds)})
        assert t2.name == f"fib:{NAME}" and "NaN" not in system_message(t2) and "scaled by" not in system_message(t2)


def test_the_solution_and_trace_set_are_what_the_benchmarks_evaluator_reads():
    with tempfile.TemporaryDirectory() as d:
        ds = dataset(Path(d) / "ds")
        src = initial_program(load(f"fib:{NAME}", samples=str(ds)))
        s = fib.solution(f"fib:{NAME}", src)
        assert s["definition"] == NAME and s["spec"]["entry_point"] == "main.py::kernel"
        assert s["spec"]["destination_passing_style"] is False and s["sources"][0]["content"] == src
        old = fib.root
        fib.root = lambda: ds
        try:
            names = fib._trace_set(f"fib:{NAME}", src, Path(d) / "ts")
        finally:
            fib.root = old
        assert names == [s["name"], "flashinfer_wrapper_0"]
        ts = Path(d) / "ts"
        assert (ts / f"solutions/{fib.AUTHOR}/gather/{NAME}/{s['name']}.json").exists()
        assert (ts / f"solutions/baseline/gather/{NAME}/flashinfer_wrapper_0.json").exists()
        w = json.loads((ts / f"workloads/gather/{NAME}.jsonl").read_text().splitlines()[0])["workload"]
        assert Path(w["inputs"]["idx"]["path"]).is_absolute() and Path(w["inputs"]["idx"]["path"]).exists()


def test_the_leaderboards_score_is_the_mean_baseline_over_ours_and_zero_on_any_failure():
    row = lambda w, sol, st, ms, ref=4.0: {"workload": w, "solution": sol, "status": st, "latency_ms": ms,  # noqa: E731
                                           "reference_latency_ms": ref}
    rows = [row("a", "me", "PASSED", 1.0), row("b", "me", "PASSED", 2.0),
            row("a", "fi", "PASSED", 2.0), row("b", "fi", "PASSED", 1.0)]
    v = fib.leaderboard(rows, "me", "fi")
    assert v["correct"] and v["vs_baseline"] == (2.0 + 0.5) / 2 and v["vs_reference"] == (4.0 + 2.0) / 2
    bad = fib.leaderboard(rows[:1] + [row("b", "me", "INCORRECT_NUMERICAL", 0.0)] + rows[2:], "me", "fi")
    assert not bad["correct"] and bad["vs_baseline"] == 0.0 and bad["failed"] == 1
    s = fib.summary([{**v}, {**bad}])
    assert s["score"] == round(1.25 / 2, 4) and s["fast_1"] == 0.5 and s["fast_2"] == 0.0 and s["correct"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
