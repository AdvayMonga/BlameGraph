"""KernelBench (Stanford's ScalingIntelligence/KernelBench, MIT): PyTorch programs whose `Model.forward` a candidate
rewrites as faster GPU code, in a `ModelNew` with the same constructor. Level 1: 100 single operators, level 2: 100
fusions, level 3: 50 full architectures. The benchmark kernel-generation work reports on (Kevin, CudaForge, KernelPro,
evolutionary searches); metrics: correct, speedup over eager PyTorch, fast_p = share of problems correct and more than
p times faster. Researcher-only: nothing in `lab/` reads it, and it is no agent tool.

`python -m kernels.kernelbench fetch` downloads the repo at COMMIT (the problems and the benchmark's own evaluation
code) into data/kernelbench/<commit>/, with a manifest of every file's sha256; nothing is re-hosted. Each problem is a
kernel task `kb:LEVEL:ID` (`kernels.tasks.load`): inputs from its own get_inputs (made on the device), weights from
`Model(*get_init_inputs())` under WEIGHT_SEED, fp32, reference and baseline its eager forward under no_grad,
KernelBench's tolerance at COMMIT. The starting program is the problem plus `class ModelNew(Model)`, the class
KernelBench scores, and the `kernel` entry the evaluator calls. The search's checks follow the benchmark's contract:
contiguous inputs, no value scalings, no NaN/Inf or edge shapes, and no roofline (an algorithmic rewrite, e.g. a
diagonal matmul done as a row scaling, beats any bound computed from the reference's work). The anti-gaming gates stay
(inputs unchanged, no aliasing, no kept answers, outputs verified outside the jail). Our own addition is a held-out
shape: the first module-level size constant get_inputs reads and Model's construction does not, doubled.

`score` runs the benchmark's own `eval_kernel_against_ref` on a program's ModelNew, jailed (5 correctness trials,
100 timed, cuda events with L2 flushed, speedup over its own eager timing), again at v0/v0.1's fp32 tolerance (1e-2;
most published numbers), its static checker, and torch.compile's time. `run` puts problems through
`kernels.search.ab`'s arms (same shapes, start, evaluator, model, budget) and then `score`, and writes fast_p per arm.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import sys
import tarfile
import time
import types
import urllib.request
from pathlib import Path

import torch

from kernels.tasks import Task

REPO = "ScalingIntelligence/KernelBench"
COMMIT = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"       # main, 2026-03-05 (0.2.0.dev0)
SOURCE = f"https://codeload.github.com/{REPO}/tar.gz/{COMMIT}"
DATA = Path(__file__).resolve().parents[1] / "data" / "kernelbench"
WEIGHT_SEED = 0
TOLERANCE = 1e-4                  # fp32 atol = rtol at COMMIT (eval.get_tolerance_for_precision)
V01_TOLERANCE = 1e-2              # fp32 atol = rtol in v0 and v0.1
CORRECT_TRIALS, PERF_TRIALS, EVAL_SEED = 5, 100, 42
HELDOUT_FACTOR = 2
HELDOUT_MAX_BYTES = 16 * 2**30    # official inputs larger than this get no held-out shape
SCORE_TIMEOUT_S = 1800.0


def root() -> Path:
    d = DATA / COMMIT[:12]
    if not (d / "manifest.json").exists():
        raise SystemExit(f"KernelBench is not fetched: python -m kernels.kernelbench fetch (into {d})")
    return d


def fetch() -> Path:
    """The repo at COMMIT: every problem file and the benchmark's own evaluation package, with sha256s."""
    dest = DATA / COMMIT[:12]
    if (dest / "manifest.json").exists():
        return dest
    with urllib.request.urlopen(SOURCE, timeout=120) as r:
        raw = r.read()
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for m in tar.getmembers():
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            parts = Path(rel).parts
            keep = (len(parts) == 3 and parts[0] == "KernelBench" and parts[1].startswith("level")
                    or len(parts) == 3 and parts[:2] == ("src", "kernelbench")) and rel.endswith(".py") \
                or rel == "LICENSE"
            if not m.isfile() or not keep or ".." in parts:
                continue
            data = tar.extractfile(m).read()
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            files[rel] = hashlib.sha256(data).hexdigest()
    levels = {}
    for rel in files:
        if rel.startswith("KernelBench/level"):
            lv = rel.split("/")[1].removeprefix("level")
            levels[lv] = levels.get(lv, 0) + 1
    (dest / "manifest.json").write_text(json.dumps({
        "repo": REPO, "commit": COMMIT, "source": SOURCE, "sha256": hashlib.sha256(raw).hexdigest(),
        "problems": dict(sorted(levels.items())), "files": files, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        indent=1))
    return dest


def _manifest() -> dict:
    return json.loads((root() / "manifest.json").read_text())


def _parse(name: str) -> tuple[int, int]:
    _, level, pid = name.split(":")
    return int(level), int(pid)


def problems(level: int | None = None) -> list[str]:
    """Every fetched problem as `kb:LEVEL:ID`, in level and id order."""
    out = []
    for rel in _manifest()["files"]:
        if rel.startswith("KernelBench/level"):
            lv, fname = rel.split("/")[1:]
            out.append((int(lv.removeprefix("level")), int(fname.split("_")[0])))
    return [f"kb:{lv}:{pid}" for lv, pid in sorted(out) if level is None or lv == level]


def problem_source(name: str) -> str:
    """The problem file's text, checked against the manifest."""
    level, pid = _parse(name)
    files = _manifest()["files"]
    rel = next((r for r in files if r.startswith(f"KernelBench/level{level}/{pid}_")), None)
    if rel is None:
        raise KeyError(f"{name}: no such KernelBench problem at {COMMIT[:12]}")
    data = (root() / rel).read_bytes()
    if hashlib.sha256(data).hexdigest() != files[rel]:
        raise ValueError(f"{rel}: sha256 differs from the manifest")
    return data.decode()


def _size_constant(src: str, ns: dict) -> str | None:
    """The first module-level int constant get_inputs reads that neither get_init_inputs nor Model reads: the axis a
    held-out shape scales (None when there is none)."""
    names = [t.id for n in ast.parse(src).body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
             and type(n.value.value) is int for t in n.targets if isinstance(t, ast.Name)]
    used = set(ns["get_inputs"].__code__.co_names)
    other = set(ns["get_init_inputs"].__code__.co_names)
    for f in vars(ns["Model"]).values():
        other |= set(getattr(getattr(f, "__code__", None), "co_names", ()))
    return next((n for n in names if n in used and n not in other), None)


@contextlib.contextmanager
def _sized(ns: dict, axis: str | None, scale: int):
    """get_inputs with the size constant scaled (its globals are `ns`)."""
    old = ns.get(axis)
    if axis:
        ns[axis] = old * scale
    try:
        yield
    finally:
        if axis:
            ns[axis] = old


def task(name: str, source: str | None = None) -> Task:
    """The problem as a kernel task. A shape is (scale,): (1,) the official sizes, (HELDOUT_FACTOR,) the held-out."""
    src = problem_source(name) if source is None else source
    ns = {"__name__": f"kb_{name.replace(':', '_')}"}
    exec(compile(src, f"<{name}>", "exec"), ns)          # a pinned, hashed problem file of the benchmark
    axis = _size_constant(src, ns)
    try:
        with torch.device("meta"):
            official = list(ns["get_inputs"]())
    except Exception:                                     # noqa: BLE001  an op without a meta kernel: once on the CPU
        official = list(ns["get_inputs"]())
    scalars = {i: x for i, x in enumerate(official) if not isinstance(x, torch.Tensor)}
    nbytes = sum(x.numel() * x.element_size() for x in official if isinstance(x, torch.Tensor))
    nets: dict = {}

    def full(tensors) -> list:
        it = iter(tensors)
        return [scalars[i] if i in scalars else next(it) for i in range(len(official))]

    def baseline(*args):
        dev = next(a.device for a in args if isinstance(a, torch.Tensor))
        if dev not in nets:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(WEIGHT_SEED)
                nets[dev] = ns["Model"](*ns["get_init_inputs"]()).to(dev).requires_grad_(False)
        with torch.no_grad():
            return nets[dev](*args)

    def reference(*tensors):
        return baseline(*full(tensors))

    def make_inputs(shape, dtype, device, g):
        dev = torch.device(device)
        seed = int(torch.randint(0, 2**62, (1,), generator=g))
        cuda = [dev.index or 0] if dev.type == "cuda" else []
        with _sized(ns, axis, shape[0]), torch.random.fork_rng(devices=cuda), dev:
            torch.manual_seed(seed)
            xs = ns["get_inputs"]()
        return tuple(x.to(dev, dtype if x.is_floating_point() else x.dtype).contiguous()
                     for x in xs if isinstance(x, torch.Tensor))

    held = axis is not None and nbytes * HELDOUT_FACTOR <= HELDOUT_MAX_BYTES
    return Task(name, None, reference, make_inputs, shape_of=lambda call: None, cost=lambda s, d, inputs=None: (0, 0),
                call=lambda fn, *ts: fn(*full(ts)), heldout_axes=(0,) if held else (), edge_shapes=lambda s: [],
                dtypes=(torch.float32,), layouts=(), scales=(), baseline_fn=baseline,
                start_program=lambda: start_program(src), seen_default=((1,),),
                heldout_default=((HELDOUT_FACTOR,),) if held else (),
                tolerances={"float32": [TOLERANCE, TOLERANCE]}, nonfinite=False)


def start_program(src: str) -> str:
    """The problem with `ModelNew(Model)` (what KernelBench scores) and the evaluator's `kernel` entry. The problem's
    comment lines are left out: the source screen matches words anywhere (two problems' comments say "signal")."""
    code = "\n".join(x for x in src.rstrip().splitlines() if not x.lstrip().startswith("#"))
    return f'''# EVOLVE-BLOCK-START
{code}


class ModelNew(Model):
    """What KernelBench scores: built like Model (same arguments, the same parameters in the same order)."""
# EVOLVE-BLOCK-END


_nets = {{}}


def kernel(*args):
    """The evaluator's entry: ModelNew built once per device under the weight seed, called on the inputs."""
    import torch
    dev = next(a.device for a in args if isinstance(a, torch.Tensor))
    if dev not in _nets:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed({WEIGHT_SEED})
            _nets[dev] = ModelNew(*get_init_inputs()).to(dev).requires_grad_(False)
    with torch.no_grad():
        return _nets[dev](*args)
'''


def _import_kernelbench(src: Path) -> None:
    """The benchmark's package without its __init__, whose utils import LLM clients (openai, litellm) the evaluation
    never calls; `get_gpu_vendor`, the one name the evaluation takes from utils, answers nvidia."""
    pkg = types.ModuleType("kernelbench")
    pkg.__path__ = [str(src / "kernelbench")]
    utils = types.ModuleType("kernelbench.utils")
    utils.get_gpu_vendor = lambda device=None: "nvidia"
    sys.modules.update({"kernelbench": pkg, "kernelbench.utils": utils})
    try:
        import requests  # noqa: F401  imported by eval.py, unused on the evaluation path
    except ImportError:
        sys.modules["requests"] = types.ModuleType("requests")


def _official(cfg_path: str) -> int:
    """Inside the jail: the benchmark's own evaluation of one program, as one JSON line."""
    cfg = json.loads(Path(cfg_path).read_text())
    _import_kernelbench(Path(cfg["kb_src"]))
    from kernelbench import eval as kbe
    from kernelbench import kernel_static_checker, timing
    ref, prog, dev = cfg["reference"], Path(cfg["program"]).read_text(), torch.device("cuda:0")
    backend = "triton" if "triton" in prog else "cuda"
    kw = dict(seed_num=EVAL_SEED, num_correct_trials=CORRECT_TRIALS, build_dir=cfg["build"], device=dev,
              backend=backend, precision=torch.float32)
    r = kbe.eval_kernel_against_ref(ref, prog, num_perf_trials=PERF_TRIALS, measure_performance=True,
                                    timing_method="cuda_event", **kw)
    out = {"tolerance": TOLERANCE, "compiled": r.compiled, "correct": r.correctness, "runtime_us": r.runtime,
           "eager_us": r.ref_runtime, "metadata": {k: str(v) for k, v in r.metadata.items()}}
    kbe.get_tolerance_for_precision = lambda precision: V01_TOLERANCE
    out["correct_v01"] = kbe.eval_kernel_against_ref(ref, prog, measure_performance=False, **kw).correctness
    valid, errors, warnings = kernel_static_checker.validate_kernel_static(prog, backend=backend, precision="fp32")
    out["static"] = {"valid": valid, "errors": errors, "warnings": warnings}
    try:
        out["compile_us"] = timing.measure_ref_program_time(cfg["name"], ref, num_trials=PERF_TRIALS,
                                                            use_torch_compile=True, device=dev)["mean"]
    except Exception as e:                       # noqa: BLE001  inductor fails on some problems: a fact
        out["compile_error"] = f"{type(e).__name__}: {e}"[:500]
    print(json.dumps(out, default=str))
    return 0


def score(name: str, program: Path, work: Path) -> dict:
    """The benchmark's own verdict on `program`'s ModelNew, run jailed on this host's GPU, with the speedups."""
    from kernels.evolve import jailed
    from lab import engine
    from lab.safety import grader
    work = Path(work).resolve()
    tree = work / "tree"
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "program.py").write_text(Path(program).read_text())
    cfg = {"name": name, "reference": problem_source(name), "program": str(tree / "program.py"),
           "build": str(tree / "build"), "kb_src": str(root() / "src")}
    (tree / "score_config.json").write_text(json.dumps(cfg))
    harness = jailed.stage_harness(work / "harness")
    home = jailed.cuda_home()
    env = {"PYTHONPATH": str(harness), "TORCH_EXTENSIONS_DIR": str(tree / "build"),
           **({"CUDA_HOME": str(home)} if home else {})}
    p = grader.jailed(tree, [engine.python(), "-m", "kernels.kernelbench", "_official", str(tree / "score_config.json")],
                      timeout_s=SCORE_TIMEOUT_S, env_extra=env, read=[harness, root() / "src"] + ([home] if home else []),
                      memory=jailed.MEMORY)
    lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
    if p.returncode != 0 or not lines:
        return {"problem": name, "error": f"exited {p.returncode}: {(p.stderr or p.stdout)[-2000:]}"}
    r = json.loads(lines[-1])
    ok = r["correct"] and r.get("runtime_us", -1) > 0
    return {"problem": name, **r, "speedup_eager": r["eager_us"] / r["runtime_us"] if ok and r["eager_us"] > 0 else None,
            "speedup_compile": r["compile_us"] / r["runtime_us"] if ok and r.get("compile_us") else None}


def fast_p(rows: list[dict], p: float, key: str = "speedup_eager", correct: str = "correct",
           static: bool = False) -> float:
    """KernelBench's fast_p: the share of problems whose program is correct and more than p times faster; `static`:
    and passes the benchmark's static checker (a custom kernel, none of its known hacks)."""
    ok = [bool(r.get(correct)) and (r.get(key) or 0) > p and (not static or bool((r.get("static") or {}).get("valid")))
          for r in rows]
    return sum(ok) / len(rows) if rows else 0.0


def summary(rows: list[dict]) -> dict:
    """fast_0, fast_1, fast_2 over eager (the published convention) and over torch.compile, at both tolerances, and
    over eager at the pinned tolerance counting only programs the static checker passes."""
    out = {"n": len(rows)}
    for tol, correct in (("pinned", "correct"), ("v01", "correct_v01")):
        for base in ("eager", "compile"):
            for p in (0.0, 1.0, 2.0):
                out[f"fast_{p:g}_{base}_{tol}"] = round(fast_p(rows, p, f"speedup_{base}", correct), 4)
    for p in (0.0, 1.0, 2.0):
        out[f"fast_{p:g}_eager_pinned_static"] = round(fast_p(rows, p, static=True), 4)
    out["static_valid"] = sum(bool((r.get("static") or {}).get("valid")) for r in rows)
    return out


def run(names: list[str], arms: str, out: Path, extra: list[str]) -> dict:
    """Each problem through kernels.search.ab's arms (the same shapes, start, evaluator and budget for every arm),
    then each arm's winner through `score`; writes <out>/kernelbench.json."""
    from collections import Counter

    from kernels import shapes as shapes_mod
    from kernels.search import ab
    from kernels.tasks import load
    out = Path(out)
    rows = []
    for name in names:
        d = out / name.replace(":", "-")
        d.mkdir(parents=True, exist_ok=True)
        shapes = shapes_mod.write(d / "shapes.json", shapes_mod.split(load(name), Counter()))
        ab.main(["--task", name, "--shapes", str(shapes), "--out", str(d), "--arms", arms, *extra])
        for r in json.loads((d / "ab.json").read_text()):
            arm_dir = d / f"{r['arm']}-{r['seed']}"
            s = score(name, arm_dir / "best_program.py", arm_dir / "official")
            (arm_dir / "official.json").write_text(json.dumps(s, indent=1))
            rows.append({**s, "arm": r["arm"], "seed": r["seed"], "ours": r})
        (out / "kernelbench.json").write_text(json.dumps({"commit": COMMIT, "rows": rows}, indent=1, default=str))
    by_arm = {arm: summary([r for r in rows if r["arm"] == arm]) for arm in dict.fromkeys(r["arm"] for r in rows)}
    (out / "kernelbench.json").write_text(json.dumps({"commit": COMMIT, "rows": rows, "summary": by_arm}, indent=1,
                                                     default=str))
    return by_arm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kernels.kernelbench", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    ls = sub.add_parser("list")
    ls.add_argument("--level", type=int)
    sc = sub.add_parser("score", help="the benchmark's own evaluation of a program's ModelNew (GPU)")
    sc.add_argument("--problem", required=True)
    sc.add_argument("--program", required=True)
    sc.add_argument("--out", required=True)
    rn = sub.add_parser("run", help="problems through kernels.search.ab's arms, then score (GPU, model calls)")
    rn.add_argument("--level", type=int)
    rn.add_argument("--problems", help="comma-separated kb:LEVEL:ID (default: every problem of --level)")
    rn.add_argument("--arms", default="openevolve,kernels.search")
    rn.add_argument("--out", required=True)
    of = sub.add_parser("_official")
    of.add_argument("config")
    a, extra = ap.parse_known_args(argv)
    if extra and a.cmd != "run":
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    if a.cmd == "fetch":
        print(fetch())
    elif a.cmd == "list":
        print("\n".join(problems(a.level)))
    elif a.cmd == "score":
        print(json.dumps(score(a.problem, Path(a.program), Path(a.out)), indent=1))
    elif a.cmd == "run":
        names = a.problems.split(",") if a.problems else problems(a.level)
        print(json.dumps(run(names, a.arms, Path(a.out), extra), indent=1))     # extra: ab's own flags
    elif a.cmd == "_official":
        return _official(a.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
