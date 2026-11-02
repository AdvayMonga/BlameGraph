"""FlashInfer-Bench (flashinfer-ai, MLSys 2026, Apache-2.0): LLM-serving kernels (GEMM, RMSNorm, paged and ragged
GQA/MLA attention, GDN, MoE, sampling) as definitions, each a PyTorch reference `run` with fixed model axes, and
workloads recorded from SGLang serving real prompts (Llama-3.1-8B, Qwen3-30B-A3B, DeepSeek-V3, ...): the variable
axes, and the tensors that must be real (page tables, sequence offsets) as safetensors. A solution is source files
plus an entry point; its leaderboard (bench.flashinfer.ai) scores a definition as the mean over its workloads of the
FlashInfer baseline's latency over the solution's, 0 if any workload fails. Researcher-only: nothing in `lab/` reads it,
and it is no agent tool.

`python -m kernels.flashinfer_bench fetch [--definitions A,B]` downloads the dataset at REVISION (every definition,
workload list and baseline solution; the named definitions' safetensors too) into data/flashinfer_bench/<revision>/,
laid out as the benchmark's own TraceSet, with a manifest of every file's sha256; nothing is re-hosted. Each
definition the default evaluator judges is a kernel task `fib:NAME` (`kernels.tasks.load`): a shape is a workload,
(its variable axes..., k) with k telling apart workloads with equal axes; recorded tensors from its safetensors, the rest
drawn like the benchmark draws them (normal floats, fp8 clamped to +-2, ints in its ranges); scalars are each
workload's own, carried as 0-d CPU tensors and handed to a program as Python numbers, as the benchmark hands them.
Definitions the benchmark judges with another evaluator (sampling, low-bit MoE, DSA) or that have no workloads are
refused. The reference and baseline are its `run`, the starting program too.
The search's checker holds a program to half the benchmark's atol and rtol: |d| <= atol/2 + rtol/2 * |ref| implies
the benchmark's rule (an element fails only when |d| > atol and |d| / |ref| > rtol). Contiguous inputs, no value
scalings, NaN/Inf or edge shapes, no roofline (definitions state no work); the anti-gaming gates stay. Our addition is
the held-out split: workloads in a band of the first variable axis (`kernels.shapes.split`), never seen in selection.

`score` runs the benchmark's own evaluator (`flashinfer_bench`, installed at EVALUATOR_COMMIT) jailed, on a TraceSet of
the definition, all its workloads, the program as a solution and the FlashInfer baseline; `solution` writes the
program as the benchmark's Solution JSON (what the leaderboard takes). `run` puts definitions through
`kernels.search.ab`'s arms (same shapes, start, evaluator, model, budget) and then `score`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import torch

from kernels.tasks import Task

DATASET = "flashinfer-ai/flashinfer-trace"
REVISION = "da915083d4c7c5e61aa3005e3d17ae488e0fc71c"         # main, 2026-06-09
EVALUATOR = "flashinfer-ai/flashinfer-bench"
EVALUATOR_COMMIT = "15618a67e8d7b7f49ca61b5ba3b9f26a876d0bee"  # main, 2026-09-25
TREE = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
RAW = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
DATA = Path(__file__).resolve().parents[1] / "data" / "flashinfer_bench"
ATOL = RTOL = 1e-2                # the default evaluator's at EVALUATOR_COMMIT (bench/config.py ResolvedEvalConfig)
INT_RANGES = {torch.int8: (-128, 128), torch.int16: (-1024, 1024), torch.int32: (-1024, 1024),
              torch.int64: (-1024, 1024)}
AUTHOR = "blamegraph"
SCORE_TIMEOUT_S = 3600.0


def root() -> Path:
    d = DATA / REVISION[:12]
    if not (d / "manifest.json").exists():
        raise SystemExit(f"FlashInfer-Bench is not fetched: python -m kernels.flashinfer_bench fetch (into {d})")
    return d


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def _tree(path: str) -> list[str]:
    """Every file under `path` in the dataset at REVISION."""
    out = []
    for x in json.loads(_get(f"{TREE}/{path}")):
        out += _tree(x["path"]) if x["type"] == "directory" else [x["path"]]
    return out


def fetch(names: list[str] | None = None) -> Path:
    """The dataset at REVISION: every definition, workload list and baseline solution, and the safetensors the named
    definitions' workloads read, with sha256s. Adds to an earlier fetch."""
    dest = DATA / REVISION[:12]
    man = json.loads((dest / "manifest.json").read_text()) if (dest / "manifest.json").exists() else \
        {"dataset": DATASET, "revision": REVISION, "files": {}}
    files = man["files"]

    def keep(rel: str) -> None:
        if rel in files or ".." in Path(rel).parts:
            return
        data = _get(f"{RAW}/{rel}")
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_bytes(data)
        files[rel] = hashlib.sha256(data).hexdigest()

    def save() -> None:
        man.update(definitions=sum(r.startswith("definitions/") for r in files),
                   fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (dest / "manifest.json").write_text(json.dumps(man, indent=1))

    if not any(r.startswith("definitions/") for r in files):
        for rel in _tree("definitions") + _tree("workloads") + _tree("solutions/baseline"):
            keep(rel)
        save()
    for name in names or []:
        for w in _workloads(name, dest):
            for spec in w["inputs"].values():
                if spec["type"] == "safetensors":
                    keep(str(Path(spec["path"])))
    save()
    return dest


def _files(d: Path) -> dict:
    return json.loads((d / "manifest.json").read_text())["files"]


def _read(d: Path, rel: str) -> bytes:
    """A dataset file, checked against the manifest."""
    data = (d / rel).read_bytes()
    if hashlib.sha256(data).hexdigest() != _files(d).get(rel):
        raise ValueError(f"{rel}: sha256 differs from the manifest (or is not in it)")
    return data


def _path(d: Path, kind: str, name: str) -> str:
    ext = ".jsonl" if kind == "workloads" else ".json"
    rel = next((r for r in _files(d) if r.startswith(f"{kind}/") and Path(r).name == name + ext), None)
    if rel is None:
        raise KeyError(f"fib:{name}: no such {kind[:-1]} at {REVISION[:12]}")
    return rel


def definition(name: str, d: Path | None = None) -> dict:
    d = d or root()
    return json.loads(_read(d, _path(d, "definitions", name)))


def _workloads(name: str, d: Path) -> list[dict]:
    """The definition's workloads ([] when the dataset has none for it)."""
    try:
        rel = _path(d, "workloads", name)
    except KeyError:
        return []
    return [json.loads(x)["workload"] for x in _read(d, rel).decode().splitlines() if x.strip()]


def definitions(d: Path | None = None) -> list[str]:
    """Every fetched definition as `fib:NAME`, by op type and name."""
    rels = sorted(r for r in _files(d or root()) if r.startswith("definitions/"))
    return [f"fib:{Path(r).stem}" for r in rels]


def refused(defn: dict, wls: list[dict]) -> str | None:
    """Why the default evaluator's contract does not cover this definition (None when it does)."""
    if not wls:
        return "no workloads at this revision"
    if defn["op_type"] == "sampling":
        return "sampling is judged by a distribution test (the benchmark's SamplingEvaluator)"
    if "moe_fp8_block_scale" in defn["name"]:
        return "low-bit MoE is judged by a matched-element ratio (the benchmark's LowBitEvaluator)"
    if defn["name"].startswith("dsa_"):
        return "DSA is judged by its own evaluators"
    rand = sorted({k for w in wls for k, sp in defn["inputs"].items() if sp["shape"] is None
                   and w["inputs"].get(k, {"type": "random"})["type"] != "scalar"})
    return f"inputs {rand} are random scalars" if rand else None


def _dtype(s: str) -> torch.dtype:
    return getattr(torch, s.replace("float4_e2m1", "float4_e2m1fn_x2"))


def _rand(shape, dt, device, g) -> torch.Tensor:
    """A random input as the benchmark draws it (bench/utils.py _rand_tensor), from a generator seeded by `g`."""
    d = torch.Generator(device=device).manual_seed(int(torch.randint(0, 2**62, (1,), generator=g)))
    if dt in (torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, generator=d, device=device, dtype=dt)
    if dt.is_floating_point:
        return torch.randn(shape, generator=d, device=device).clamp_(-2.0, 2.0).to(dt)
    if dt is torch.bool:
        return torch.randint(0, 2, shape, generator=d, device=device, dtype=torch.bool)
    lo, hi = INT_RANGES[dt]
    return torch.randint(lo, hi, shape, generator=d, device=device, dtype=dt)


def task(name: str, samples: str | None = None) -> Task:
    """The definition as a kernel task; `samples` is the fetched dataset dir (default: this host's)."""
    d = Path(samples) if samples else root()
    bare = name.removeprefix("fib:")
    defn, wls = definition(bare, d), _workloads(bare, d)
    why = refused(defn, wls)
    if why:
        raise ValueError(f"{name}: {why}")
    ns = {"__name__": f"fib_{bare}"}
    exec(compile(defn["reference"], f"<{name}>", "exec"), ns)    # a pinned, hashed definition of the benchmark
    ref = ns["run"]
    var = [a for a, s in defn["axes"].items() if s["type"] == "var"]
    const = {a: s["value"] for a, s in defn["axes"].items() if s["type"] == "const"}
    order = list(defn["inputs"])
    scalars = {k for k, sp in defn["inputs"].items() if sp["shape"] is None}
    by_shape, seen = {}, Counter()
    for w in wls:
        key = tuple(w["axes"][a] for a in var)
        by_shape[key + (seen[key],)] = w
        seen[key] += 1
    blobs: dict = {}
    out_dtypes = [_dtype(s["dtype"]) for s in defn["outputs"].values()]

    def full(ts) -> list:
        return [x.item() if k in scalars else x for k, x in zip(order, ts)]

    def reference(*ts):
        return ref(*full(ts))

    def make_inputs(shape, dtype, device, g):
        w = by_shape[tuple(shape)]
        axes = {**const, **w["axes"]}
        out = []
        for k in order:
            spec, dt = w["inputs"].get(k, {"type": "random"}), _dtype(defn["inputs"][k]["dtype"])
            if spec["type"] == "scalar":                 # on the CPU: .item() costs no device sync when timed
                v = spec["value"]
                out.append(torch.tensor(v, dtype=torch.bool if isinstance(v, bool) else
                                        torch.int64 if isinstance(v, int) else torch.float64))
            elif spec["type"] == "safetensors":
                rel = str(Path(spec["path"]))
                if rel not in blobs:
                    from safetensors.torch import load
                    blobs[rel] = load(_read(d, rel))
                out.append(blobs[rel][spec["tensor_key"]].to(device).contiguous())
            else:
                out.append(_rand([axes[a] for a in defn["inputs"][k]["shape"]], dt, device, g))
        return tuple(out)

    shapes = list(by_shape)
    tol = {str(dt).removeprefix("torch."): [ATOL / 2, RTOL / 2] for dt in dict.fromkeys(out_dtypes)}
    return Task(name, None, reference, make_inputs, shape_of=lambda call: None, cost=lambda s, dt, inputs=None: (0, 0),
                call=lambda fn, *ts: fn(*full(ts)), heldout_axes=tuple(range(len(var))), edge_shapes=lambda s: [],
                dtypes=(out_dtypes[0],), layouts=(), scales=(), baseline_fn=ref,
                start_program=lambda: start_program(defn), seen_default=tuple(shapes),
                tolerances=tol, nonfinite=False)


def shape_sets(t: Task) -> dict:
    """Seen and held-out workloads: every workload counted once, held out by `kernels.shapes.split`'s band."""
    from kernels import shapes as shapes_mod
    return shapes_mod.split(t, Counter({s: 1 for s in t.seen_default}))


def start_program(defn: dict) -> str:
    """The definition's reference `run` and the evaluator's `kernel` entry (the solution's entry point)."""
    return f'''# EVOLVE-BLOCK-START
{defn["reference"].rstrip()}
# EVOLVE-BLOCK-END


def kernel(*args):
    """The evaluator's entry and the solution's: the definition's inputs in order, its outputs returned."""
    return run(*args)
'''


def solution(name: str, program: str, author: str = AUTHOR) -> dict:
    """The program as the benchmark's Solution (data/solution.py): one source file, entry `main.py::kernel`, outputs
    returned (not destination-passing)."""
    bare = name.removeprefix("fib:")
    digest = hashlib.sha256(program.encode()).hexdigest()[:8]
    return {"name": f"{author}_{bare}_{digest}", "definition": bare, "author": author,
            "spec": {"language": "triton" if "triton" in program else "python", "target_hardware": ["cuda"],
                     "entry_point": "main.py::kernel", "dependencies": [], "destination_passing_style": False},
            "sources": [{"path": "main.py", "content": program}],
            "description": f"kernels.search program {digest}"}


def _trace_set(name: str, program: str, dest: Path) -> list[str]:
    """A TraceSet holding only `name`: its definition, its workloads with safetensors paths made absolute (into the
    fetched dataset), the program's solution and the FlashInfer baselines. Returns the solutions' names."""
    d, bare = root(), name.removeprefix("fib:")
    op = definition(bare)["op_type"]
    for kind, rel in (("definitions", _path(d, "definitions", bare)), ("workloads", _path(d, "workloads", bare))):
        text = _read(d, rel).decode()
        if kind == "workloads":
            lines = []
            for x in text.splitlines():
                if x.strip():
                    t = json.loads(x)
                    for spec in t["workload"]["inputs"].values():
                        if spec["type"] == "safetensors":
                            spec["path"] = str(d / Path(spec["path"]))
                    lines.append(json.dumps(t))
            text = "\n".join(lines) + "\n"
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(text)
    sols = [solution(name, program)]
    sols += [json.loads(_read(d, r)) for r in _files(d) if r.startswith(f"solutions/baseline/{op}/{bare}/")]
    for s in sols:
        p = dest / "solutions" / s["author"] / op / bare / f"{s['name']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(s))
    return [s["name"] for s in sols]


def _official(cfg_path: str) -> int:
    """Inside the jail: the benchmark's own evaluator on one TraceSet, every workload's verdict as one JSON line."""
    import importlib.metadata
    cfg = json.loads(Path(cfg_path).read_text())
    url = json.loads(importlib.metadata.distribution("flashinfer-bench").read_text("direct_url.json") or "{}")
    commit = (url.get("vcs_info") or {}).get("commit_id")
    if commit != EVALUATOR_COMMIT:
        print(json.dumps({"error": f"flashinfer-bench is at {commit}, not {EVALUATOR_COMMIT}: pip install "
                                   f"git+https://github.com/{EVALUATOR}@{EVALUATOR_COMMIT}"}))
        return 0
    from flashinfer_bench.bench import Benchmark, BenchmarkConfig
    from flashinfer_bench.data import TraceSet
    bench = Benchmark(TraceSet.from_path(cfg["trace_set"]),
                      BenchmarkConfig(use_isolated_runner=True, definitions=[cfg["definition"]]))
    try:
        traces = [t for ts in bench.run_all(dump_traces=False).traces.values() for t in ts]
    finally:
        bench.close()
    rows = []
    for t in traces:
        ev, perf = t.evaluation, t.evaluation.performance
        rows.append({"workload": t.workload.uuid, "solution": t.solution, "status": ev.status.value,
                     "latency_ms": perf.latency_ms if perf else None,
                     "reference_latency_ms": perf.reference_latency_ms if perf else None,
                     "max_abs": ev.correctness.max_absolute_error if ev.correctness else None,
                     "max_rel": ev.correctness.max_relative_error if ev.correctness else None})
    print(json.dumps({"evaluator_commit": commit, "rows": rows}, default=str))
    return 0


def leaderboard(rows: list[dict], ours: str, baseline: str | None) -> dict:
    """The verdict from per-workload rows: correct on every workload; the leaderboard's score (mean of the baseline's
    latency over ours, 0 if any workload fails: web/apps/web/lib/analytics.ts computeSolutionScore); the mean speedup
    over the reference (the paper's)."""
    mine = {r["workload"]: r for r in rows if r["solution"] == ours}
    base = {r["workload"]: r for r in rows if r["solution"] == baseline}
    correct = bool(mine) and all(r["status"] == "PASSED" for r in mine.values())
    ratios = [base[w]["latency_ms"] / r["latency_ms"] for w, r in mine.items()
              if correct and base.get(w, {}).get("status") == "PASSED" and r["latency_ms"]]
    vs_ref = [r["reference_latency_ms"] / r["latency_ms"] for r in mine.values() if correct and r["latency_ms"]]
    return {"correct": correct, "workloads": len(mine), "failed": sum(r["status"] != "PASSED" for r in mine.values()),
            "vs_baseline": (sum(ratios) / len(ratios) if len(ratios) == len(mine) else None) if correct else 0.0,
            "vs_reference": sum(vs_ref) / len(vs_ref) if vs_ref else None}


def score(name: str, program: Path, work: Path) -> dict:
    """The benchmark's own verdict on `program` over every workload of `name`, run jailed on this host's GPU, beside
    the FlashInfer baseline's."""
    from kernels.evolve import jailed
    from lab import engine
    from lab.safety import grader
    work = Path(work).resolve()
    tree = work / "tree"
    tree.mkdir(parents=True, exist_ok=True)
    names = _trace_set(name, Path(program).read_text(), tree / "trace_set")
    cfg = {"definition": name.removeprefix("fib:"), "trace_set": str(tree / "trace_set")}
    (tree / "score_config.json").write_text(json.dumps(cfg))
    harness = jailed.stage_harness(work / "harness")
    home = jailed.cuda_home()
    env = {"PYTHONPATH": str(harness), "TORCH_EXTENSIONS_DIR": str(tree / "build"), "HOME": str(tree),
           **({"CUDA_HOME": str(home)} if home else {})}
    p = grader.jailed(tree, [engine.python(), "-m", "kernels.flashinfer_bench", "_official",
                             str(tree / "score_config.json")],
                      timeout_s=SCORE_TIMEOUT_S, env_extra=env, read=[harness, root()] + ([home] if home else []),
                      memory=jailed.MEMORY)
    lines = [x for x in p.stdout.splitlines() if x.startswith("{")]
    if p.returncode != 0 or not lines:
        return {"definition": name, "error": f"exited {p.returncode}: {(p.stderr or p.stdout)[-2000:]}"}
    r = json.loads(lines[-1])
    if "error" in r:
        return {"definition": name, **r}
    baseline = next((n for n in names[1:]), None)
    return {"definition": name, "solution": names[0], "baseline": baseline, "evaluator_commit": r["evaluator_commit"],
            **leaderboard(r["rows"], names[0], baseline), "rows": r["rows"]}


def summary(rows: list[dict]) -> dict:
    """Over definitions: the leaderboard's author score (the plain mean of per-definition scores), the share correct,
    and fast_p over the FlashInfer baseline."""
    n = len(rows)
    vs = [r.get("vs_baseline") or 0.0 for r in rows]
    out = {"n": n, "correct": sum(bool(r.get("correct")) for r in rows), "score": round(sum(vs) / n, 4) if n else 0.0}
    for p in (0.0, 1.0, 2.0):
        out[f"fast_{p:g}"] = round(sum(bool(r.get("correct")) and v > p for r, v in zip(rows, vs)) / n, 4) if n else 0.0
    return out


def run(names: list[str], arms: str, out: Path, extra: list[str]) -> dict:
    """Each definition through kernels.search.ab's arms (the same shapes, start, evaluator and budget for every arm),
    then each arm's winner through `score`; writes <out>/flashinfer_bench.json."""
    from kernels import shapes as shapes_mod
    from kernels.search import ab
    from kernels.tasks import load
    out = Path(out)
    fetch([n.removeprefix("fib:") for n in names])
    rows = []
    for name in names:
        d = out / name.replace(":", "-")
        d.mkdir(parents=True, exist_ok=True)
        shapes = shapes_mod.write(d / "shapes.json", shape_sets(load(name)))
        ab.main(["--task", name, "--shapes", str(shapes), "--out", str(d), "--arms", arms, *extra])
        for r in json.loads((d / "ab.json").read_text()):
            arm_dir = d / f"{r['arm']}-{r['seed']}"
            s = score(name, arm_dir / "best_program.py", arm_dir / "official")
            (arm_dir / "official.json").write_text(json.dumps(s, indent=1))
            (arm_dir / "solution.json").write_text(json.dumps(solution(name, (arm_dir / "best_program.py").read_text()),
                                                              indent=1))
            rows.append({**s, "arm": r["arm"], "seed": r["seed"], "ours": r})
        (out / "flashinfer_bench.json").write_text(json.dumps({"revision": REVISION, "rows": rows}, indent=1))
    by_arm = {arm: summary([r for r in rows if r["arm"] == arm]) for arm in dict.fromkeys(r["arm"] for r in rows)}
    (out / "flashinfer_bench.json").write_text(json.dumps({"revision": REVISION, "rows": rows, "summary": by_arm},
                                                          indent=1, default=str))
    return by_arm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kernels.flashinfer_bench", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    fe = sub.add_parser("fetch")
    fe.add_argument("--definitions", help="comma-separated names whose safetensors to fetch too")
    sub.add_parser("list", help="every definition, and why the ones not taken are refused")
    so = sub.add_parser("solution", help="a program as the benchmark's Solution JSON")
    so.add_argument("--definition", required=True)
    so.add_argument("--program", required=True)
    so.add_argument("--out", required=True)
    sc = sub.add_parser("score", help="the benchmark's own evaluation of a program (GPU)")
    sc.add_argument("--definition", required=True)
    sc.add_argument("--program", required=True)
    sc.add_argument("--out", required=True)
    rn = sub.add_parser("run", help="definitions through kernels.search.ab's arms, then score (GPU, model calls)")
    rn.add_argument("--definitions", required=True, help="comma-separated fib:NAME")
    rn.add_argument("--arms", default="openevolve,kernels.search")
    rn.add_argument("--out", required=True)
    of = sub.add_parser("_official")
    of.add_argument("config")
    a, extra = ap.parse_known_args(argv)
    if extra and a.cmd != "run":
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    if a.cmd == "fetch":
        print(fetch(a.definitions.split(",") if a.definitions else None))
    elif a.cmd == "list":
        for n in definitions():
            bare = n.removeprefix("fib:")
            why = refused(definition(bare), _workloads(bare, root()))
            print(n + (f"  (refused: {why})" if why else ""))
    elif a.cmd == "solution":
        Path(a.out).write_text(json.dumps(solution(a.definition, Path(a.program).read_text()), indent=1))
    elif a.cmd == "score":
        print(json.dumps(score(a.definition, Path(a.program), Path(a.out)), indent=1))
    elif a.cmd == "run":
        print(json.dumps(run(a.definitions.split(","), a.arms, Path(a.out), extra), indent=1))   # extra: ab's own
    elif a.cmd == "_official":
        return _official(a.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
