"""python -m kernels.evolve run --task T [--shapes SHAPES.json | --captures DIR ...] --out DIR [--iterations N]
                               [--config CONFIG.yaml] [--device cuda] [--initial PROGRAM.py]
python -m kernels.evolve gate --task T --shapes SHAPES.json --program BEST.py [--device cuda]
   (T may be region:FIRST:LAST with --samples DIR, a CAPTURE_REGION dir, and --shared-weights FILE, a holder file)
python -m kernels.evolve share --profile BUNDLE [--profile ...] [--gate TASK=GATE.json ...]   (tasks ranked by headroom)
python -m kernels.evolve initial --task T --engine REPO --ref REF --out P.py   (start from the kernel at an engine commit)
python -m kernels.evolve ship --task T --program BEST.py --gate GATE.json --engine REPO [--base REF] [--budget USD]

`run` replicates OpenEvolve (AlphaEvolve's open re-implementation) on one kernel task: the starting program is the
engine's current kernel (its module's source, with `kernel` bound to the entry point), selection sees the seen shapes
only, and every candidate is judged in its own process. `gate` judges a program on the held-out shapes, which
selection never saw. Model calls cost money: the config names the models and the run's size."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import re
import sys
from pathlib import Path

import torch

from kernels import shapes as shapes_mod
from kernels.evolve.jailed import judge_jailed
from kernels.tasks import TASKS, load, options_from_target

HERE = Path(__file__).resolve().parent


def initial_program(task) -> str:
    """The engine's module source with `kernel` bound to the entry point, all of it open to evolution."""
    if task.entry is None:
        return (HERE / "seeds" / f"{task.name}.py").read_text()
    if task.start_program is not None:
        return task.start_program()
    if task.name.startswith("region:"):              # its entry is the stations tuple, not a function
        raise ValueError(f"{task.name}: a region's starting program needs its holder (`shared_weights`: the layer "
                         "config and the weights' constants are written into it)")
    mod, _, attr = task.entry.partition(":")
    return _start(task, inspect.getsource(importlib.import_module(mod)), attr)


def _start(task, src: str, attr: str) -> str:
    """The engine module as the starting program: the task's start edits applied (each must apply exactly once, so a
    changed engine module is noticed), `kernel` bound to the entry point."""
    for old, new in task.start_edits:
        if src.count(old) != 1:
            raise ValueError(f"{task.name}: a start edit no longer matches the engine module: {old[:60]!r}")
        src = src.replace(old, new)
    return f"# EVOLVE-BLOCK-START\n{src}\n# EVOLVE-BLOCK-END\n\nkernel = {attr}\n"


def initial_program_at(task, repo: Path, ref: str) -> str:
    """The same starting program, from the engine module as it was at `ref` in the engine repo (e.g. the kernel before
    a change, for an A/B with known headroom). The baseline stays the engine's current kernel."""
    from kernels.evolve.ship import _git, _module_path
    mod, _, attr = task.entry.partition(":")
    src = _git(repo, "show", f"{ref}:{_module_path(repo, _git(repo, 'rev-parse', ref), mod)}")
    return _start(task, src, attr)


def system_message(task) -> str:
    """What the program is and how it is judged, from the evaluator's own constants: facts, never advice."""
    from kernels.evolve import core
    base = f"the engine's current `{task.entry}`" if task.entry else "the task's reference"
    refused = "; ".join(dict.fromkeys(core.FORBIDDEN.values()))
    graphs = (", both captured in CUDA graphs and timed as graph replays, as the engine runs them (a kernel that "
              "cannot be captured scores 0)") if task.graphs else ""
    from kernels.evolve.jailed import TIMEOUT_S
    cuda = ("\n- A program may build CUDA C++ when it is imported, with `torch.utils.cpp_extension.load_inline` (the "
            "evaluation host's CUDA toolkit at $CUDA_HOME, ninja on PATH, the GPU's own architecture unless the program "
            f"passes flags). The build is not timed; it counts against each stage's {TIMEOUT_S:.0f} s limit and is kept "
            "per program source across its stages and re-timings. Its C++ is screened like the Python.") \
        if task.entry else ""
    return f"""The program defines `kernel`, a replacement for {base}. It must compute what the task's reference \
computes, on the same inputs, within dtype tolerance ({', '.join(str(d).removeprefix('torch.') for d in task.dtypes)}).

How a program is judged:
- Stage 1: correctness on the two most-called seen shapes, one seed. Stage 2: correctness on every seen shape (three \
seeds, edge shapes, NaN/Inf inputs, inputs scaled by {', '.join(f'{k:g}' for k in core.SCALES)}), inputs left \
unchanged, no output aliasing, no answers kept across calls; then peak memory at most {core.MEMORY_FACTOR}x the \
baseline's + {core.MEMORY_SLACK // 2**20} MiB; then timing against {base}{graphs}.
- Score: the call-weighted geometric mean of the speedup over the seen shapes, the program and the baseline timed \
in pairs on the same inputs; a shape where the middle half of the paired ratios includes 1 counts as 1.0. A failed gate scores 0; so does a time at which the kernel would run at \
over {core.PHYSICS_MARGIN_PCT}% of its roofline bound. Held-out shapes are judged separately and are not part of the score.
- Refused before running: source that uses {refused}, or writes into anything it imports. Scores 0: anything the \
evaluator uses found patched.
- Lower precision inside the kernel is allowed while the output stays within tolerance; each result reports its \
error against the reference beside the baseline's.{cuda}

The goal is the highest score."""


def setting(task, sets: dict, device: str, tol: dict) -> str:
    """The facts a program is judged under, the same for every generator: the seen shapes with their calls, their
    share of the score and the traffic they were captured under; the GPU, its measured peaks and the toolchain; the
    engine's launch settings; the tolerances. Never the held-out shapes."""
    from kernels.check import TOLERANCES
    calls, src = sets.get("calls") or {}, sets.get("sources") or {}
    seen = [json.dumps(list(x)) for x in sets["seen"]]
    total = sum(calls.get(k, 1) for k in seen) or 1
    out = ["# The setting (facts)", "Seen shapes, each with the calls it stands for in the captured traffic (its own "
           "and, when shapes were bucketed for selection, those of its bucket) and its weight in the score:"]
    for k in seen:
        by = ", ".join(f"{c} {n}" for c, n in sorted((src.get(k) or {}).items()))
        out.append(f"- {k}: {calls.get(k, 1)} calls, {100 * calls.get(k, 1) / total:.1f}% of the score"
                   + (f" (captured under: {by})" if by else ""))
    out.append(_hardware(device))
    try:
        from lab import target
        out.append(f"The engine's launch settings: {json.dumps(target.load().engine.env, sort_keys=True)}")
    except Exception:                           # noqa: BLE001  no target: nothing to state
        pass
    opts = options_from_target(task)
    if opts:
        out.append(f"Options the task's inputs are built with: {json.dumps(opts)}")
    if task.graphs:
        out.append("The engine replays this entry point inside CUDA graphs.")
    eff = {str(d).removeprefix("torch."): list(TOLERANCES.get(d, ())) for d in task.dtypes}
    eff.update({k: v for k, v in tol.items() if k in eff})
    out.append(f"Tolerances (atol, rtol) per dtype: {json.dumps(eff)}")
    return "\n".join(out)


def _hardware(device: str) -> str:
    """The device, its limits and measured peaks, and the toolchain, as one line of facts."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return f"Device: {device}; torch {torch.__version__}."
    p = torch.cuda.get_device_properties(device)
    facts = {"gpu": p.name, "compute_capability": f"{p.major}.{p.minor}", "sms": p.multi_processor_count,
             "memory_gib": round(p.total_memory / 2**30, 1)}
    for k in ("L2_cache_size", "shared_memory_per_block_optin", "shared_memory_per_multiprocessor",
              "regs_per_multiprocessor", "max_threads_per_multi_processor", "warp_size"):
        v = getattr(p, k, None)
        if v is not None:
            facts[k] = v
    from validity import peaks
    pk = peaks.load(p.name)
    if pk:
        facts.update(peak_hbm_gb_s=round(pk.bw_bytes_s / 1e9), peak_bf16_tflops=round(pk.bf16_flops / 1e12),
                     peaks_source=pk.source)
    facts.update(torch=torch.__version__, cuda=torch.version.cuda)
    try:
        import triton
        facts["triton"] = triton.__version__
    except ImportError:
        pass
    return f"The GPU and toolchain: {json.dumps(facts)}"


def _arch(device: str) -> str | None:
    """The GPU's architecture as TORCH_CUDA_ARCH_LIST spells it (9.0a on Hopper), for builds made without a device."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return f"{major}.{minor}" + ("a" if major in (9, 10) else "")


def tolerances(target_tol: dict | None) -> dict:
    """The target spec's [kernels] tolerances ({dtype: [atol, rtol]}), as the checker's override."""
    return {k: list(v) for k, v in (target_tol or {}).items()}


def task_name(s: str) -> str:
    """--task: a named task or `region:FIRST:LAST`."""
    if s in TASKS or re.fullmatch(r"region:\w+:\w+", s):
        return s
    raise argparse.ArgumentTypeError(f"{s!r} is none of {', '.join(TASKS)}, nor region:FIRST:LAST")


def task_args(p) -> None:
    """--task, and a region's --samples and --shared-weights."""
    p.add_argument("--task", required=True, type=task_name)
    p.add_argument("--samples", help="a region's CAPTURE_REGION dir (also its --captures when none are given)")
    p.add_argument("--shared-weights", help="a region's holder file (kernels.shared_weights)")


def region_args(a) -> dict:
    """A region task's `samples` and `shared_weights`, absolute: what `load` and the eval config take."""
    if not a.task.startswith("region:"):
        return {}
    return {k: str(Path(v).resolve()) for k, v in (("samples", getattr(a, "samples", None)),
                                                   ("shared_weights", getattr(a, "shared_weights", None))) if v}


def _shapes(task, a) -> dict:
    if a.shapes:
        return shapes_mod.read(a.shapes)
    dirs = [Path(d) for d in (a.captures or [region_args(a).get("samples")]) if d]   # a region's: its samples
    counts = shapes_mod.from_captures(task, dirs, equal=getattr(a, "equal_captures", False))
    if dirs and not counts:                      # never the task's default shapes in place of real ones
        raise SystemExit(f"the captures hold no call of {task.entry} (calls replayed from a CUDA graph never reach "
                         "Python: capture with the engine's graphs off, the target's [kernels] capture_env)")
    sets = shapes_mod.split(task, counts)
    return {**sets, "ints": shapes_mod.ints_from_captures(task, dirs),
            "sources": shapes_mod.sources_from_captures(task, dirs)} if dirs else sets


def _ints(sets: dict, which: tuple[str, ...]) -> dict:
    """The captured ints of the shapes in `which` (selection gets the seen ones only)."""
    keys = {json.dumps(list(s)) for k in which for s in sets[k]}
    return {k: v for k, v in (sets.get("ints") or {}).items() if k in keys}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernels.evolve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "gate", "split"):
        p = sub.add_parser(name)
        task_args(p)
        p.add_argument("--shapes")
        p.add_argument("--captures", action="append")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    shp = sub.add_parser("share")
    shp.add_argument("--profile", action="append", required=True, help="a `profile` bundle dir (repeat per class)")
    shp.add_argument("--gate", action="append", default=[], help="TASK=GATE.json: the engine kernel's gate result")
    ini = sub.add_parser("initial")
    ini.add_argument("--task", required=True, choices=TASKS)
    ini.add_argument("--engine", required=True, help="the engine repo")
    ini.add_argument("--ref", required=True, help="the engine commit whose kernel to start from")
    ini.add_argument("--out", required=True)
    sh = sub.add_parser("ship")
    task_args(sh)
    sh.add_argument("--program", required=True)
    sh.add_argument("--gate", required=True, help="`gate`'s JSON output for the program")
    sh.add_argument("--engine", required=True, help="the engine repo")
    sh.add_argument("--base", default="HEAD")
    sh.add_argument("--budget", type=float, help="also start the researcher run (`lab.experiment start`)")
    r = sub.choices["run"]
    r.add_argument("--out", required=True)
    r.add_argument("--iterations", type=int, default=20)
    r.add_argument("--config", default=str(HERE / "config.yaml"))
    r.add_argument("--initial")
    r.add_argument("--timing-shapes", type=int, default=8, help="seen shapes used for selection (by calls)")
    r.add_argument("--retime", type=int, default=3, help="top candidates re-timed 3x each before one is promoted")
    sub.choices["gate"].add_argument("--program", required=True)
    sub.choices["split"].add_argument("--write", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "ship":
        from kernels.evolve.ship import ship
        out = ship(load(a.task, **region_args(a)), Path(a.program).read_text(), json.loads(Path(a.gate).read_text()), Path(a.engine),
                   a.base)
        if a.budget is not None:
            from lab import experiment
            out["run"] = experiment.start(Path(a.engine), out["base"], out["branch"], a.budget)
        print(json.dumps(out, indent=1))
        return 0
    if a.cmd == "initial":
        Path(a.out).write_text(initial_program_at(load(a.task), Path(a.engine), a.ref))
        print(a.out)
        return 0
    if a.cmd == "share":
        from kernels.share import rank
        gates = {k: json.loads(Path(v).read_text()) for k, v in (g.split("=", 1) for g in a.gate)}
        names = list(gates) or list(TASKS)       # with gates, only the gated tasks: two tasks can name one kernel
        print(json.dumps(rank([Path(p) for p in a.profile], [load(n) for n in names], gates), indent=1))
        return 0
    task = load(a.task, **region_args(a))
    sets = _shapes(task, a)
    tol = tolerances(_target_tolerances())
    if a.cmd == "split":
        print(shapes_mod.write(a.write, sets))
        return 0
    if a.cmd == "gate":
        out = judge_jailed({"task": task.name, "seen": [list(s) for s in sets["seen"]],
                            "heldout": [list(s) for s in sets["heldout"]], "weights": sets.get("calls"),
                            "device": a.device, "tolerances": tol, "ints": _ints(sets, ("seen", "heldout")),
                            "options": options_from_target(task), **region_args(a)},
                           Path(a.program).read_text(), 2, timeout_s=3600)
        print(json.dumps(out, indent=1))
        from kernels.evolve.ship import gate_passed
        return 0 if gate_passed(out) else 1
    out = Path(a.out)
    sets, program = prepare(task, sets, out, a.device, tol, a.timing_shapes, a.initial, region_args(a))
    if "OPENAI_API_KEY" not in os.environ and "ANTHROPIC_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]     # the config's endpoint is Anthropic's
    from kernels.evolve.evaluator import seconds
    from openevolve import run_evolution
    from openevolve.config import load_config
    from kernels.evolve.llm import init_client
    config = load_config(a.config)
    config.prompt.system_message = system_message(task) + "\n\n" + setting(task, sets, a.device, tol)
    if "anthropic.com" in (config.llm.api_base or ""):       # Claude refuses a temperature; OpenEvolve always sends one
        config.llm.update_model_params({"init_client": init_client}, overwrite=True)
    os.environ["KERNELS_USAGE_LOG"] = str(out / "usage.jsonl")
    res = run_evolution(str(program), str(HERE / "evaluator.py"), config, iterations=a.iterations,
                        output_dir=str(out / "openevolve"), cleanup=False)
    trace = out / "openevolve" / "evolution_trace.jsonl"
    code, metrics, retimed = retime(trace, a.retime) if a.retime else (None, None, [])
    (out / "best_program.py").write_text(code or res.best_code)
    (out / "best_metrics.json").write_text(json.dumps(metrics or res.metrics, indent=1, default=str))
    rec = record(task, sets, [{"name": m.name, "weight": m.weight} for m in config.llm.models], a.iterations,
                 a.device, trace, out / "best_program.py", metrics or res.metrics, retimed=retimed,
                 cost={**usage(out / "usage.jsonl"), "eval_s": seconds(out / "eval_seconds.log"),
                       "evals": sum(1 for x in trace.read_text().splitlines() if x.strip()) if trace.exists() else 0})
    print(json.dumps({"best_score": res.best_score, "metrics": res.metrics, "out": str(out), "ledger": rec["id"]},
                     default=str))
    return 0


def usage(log: Path) -> dict:
    """A run's model cost from the usage log `kernels.evolve.llm.ClaudeCompat` appends to: {usd, model_calls}."""
    rows = [json.loads(x) for x in Path(log).read_text().splitlines() if x.strip()] if Path(log).exists() else []
    return {"usd": round(sum(r["usd"] for r in rows), 4), "model_calls": len(rows)}


def prepare(task, sets: dict, out: Path, device: str, tol: dict, timing_shapes: int, initial: str | None,
            region: dict | None = None):
    """The run dir: selection's config (seen shapes only, their weights and captured ints; the held-out shapes never
    enter it; a region's `samples` and `shared_weights`) as $KERNEL_EVAL_CONFIG, the shape sets, and the starting
    program. Returns (sets, program path)."""
    out.mkdir(parents=True, exist_ok=True)
    sets = shapes_mod.select(task, sets, timing_shapes)
    cfg_path = out / "eval_config.json"
    seen = {json.dumps(list(x)) for x in sets["seen"]}
    cfg_path.write_text(json.dumps({"task": task.name, "seen": [list(s) for s in sets["seen"]],
                                    "weights": {k: v for k, v in (sets.get("calls") or {}).items() if k in seen},
                                    "device": device, "tolerances": tol, "ints": _ints(sets, ("seen",)),
                                    "arch": _arch(device),
                                    "options": options_from_target(task), **(region or {})}))
    (out / "shapes.json").write_text(json.dumps({k: [list(s) for s in v] if k in ("seen", "heldout") else v
                                                 for k, v in sets.items()}, indent=1))
    program = Path(initial) if initial else out / "initial_program.py"
    if not initial:
        program.write_text(initial_program(task))
    os.environ["KERNEL_EVAL_CONFIG"] = str(cfg_path)
    os.environ["KERNELS_EVAL_LOG"] = str(out / "eval_seconds.log")        # every judgment's wall seconds
    return sets, program


def retime(trace: Path, k: int, repeats: int = 3, exclude=None) -> tuple[str | None, dict | None, list]:
    """The top `k` candidates by score re-judged `repeats` times through the evaluator (stage 2, local or remote);
    the one with the highest median score is promoted. One timing of each is what selection saw, and one timing can
    win by noise. `exclude(code, row) -> reason | None` passes over a candidate (e.g. flagged as gaming the
    evaluator), the next-best taking its place. Returns (code, metrics of its median run, every re-timing, the
    excluded ones included with their reason)."""
    import statistics
    import tempfile
    from kernels.evolve import evaluator
    rows = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()] if trace.exists() else []
    best: dict = {}
    for r in rows:
        if r.get("child_code") and (r.get("child_metrics") or {}).get("combined_score", 0) > 0:
            best.setdefault(r["child_id"], r)
    ranked, top, out = sorted(best.values(), key=lambda r: -r["child_metrics"]["combined_score"]), [], []
    for r in ranked:
        if len(top) >= k:
            break
        why = exclude(r["child_code"], r) if exclude else None
        if why:
            out.append({"id": r["child_id"], "selected_score": r["child_metrics"]["combined_score"], "excluded": why})
        else:
            top.append(r)
    for r in top:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(r["child_code"])
        runs = [evaluator.judge_stage(f.name, 2)["metrics"] for _ in range(repeats)]
        Path(f.name).unlink()
        scores = [m.get("combined_score", 0.0) for m in runs]
        if min(scores) <= 0:                     # a gate failed in one re-judgment: an intermittent kernel is out
            out.append({"id": r["child_id"], "selected_score": r["child_metrics"]["combined_score"], "scores": scores,
                        "excluded": "a re-judgment scored 0 (a gate failed)"})
            continue
        med = statistics.median(scores)
        out.append({"id": r["child_id"], "selected_score": r["child_metrics"]["combined_score"], "scores": scores,
                    "median": med, "code": r["child_code"],
                    "metrics": min(runs, key=lambda m: abs(m.get("combined_score", 0.0) - med))})
    timed = [x for x in out if "median" in x]
    if not timed:
        return None, None, out
    win = max(timed, key=lambda x: x["median"])
    return win["code"], win["metrics"], [{k: v for k, v in x.items() if k not in ("code", "metrics")} for x in out]


def record(task, sets: dict, models: list[dict], iterations: int, device: str, trace: Path, best: Path,
           best_metrics: dict, root: Path | None = None, retimed: list | None = None, tool: str = "kernels.evolve.run",
           cost: dict | None = None, extra: dict | None = None) -> dict:
    """One `kernels` ledger record per run: every evaluated candidate's seen-shape metrics and lineage, the trace and
    the best program as blobs. Held-out shapes stay out (`gate` judges them; its output stays in the run dir)."""
    import time
    from lab import ledger
    root = root or ledger.ROOT
    rows = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()] if trace.exists() else []
    cands = [{"id": r["child_id"], "parent": r["parent_id"], "iteration": r["iteration"],
              "island": r.get("island_id"), "metrics": r.get("child_metrics")} for r in rows]
    return ledger.append({
        "kind": "kernels", "run": f"kernels-{time.strftime('%Y%m%d-%H%M%S')}", "tool": tool,
        "config": {"split": "seen", "task": task.name, "entry": task.entry, "iterations": iterations,
                   "device": device, "seen_shapes": len(sets["seen"]), "selected_from": sets.get("selected_from"),
                   "models": models},
        "result": {"best": best_metrics, "candidates": cands, "retimed": retimed or [], **(extra or {}),
                   "trace": ledger.put_blob(trace, root, ".jsonl") if trace.exists() else None,
                   "best_program": ledger.put_blob(best, root, ".py")},
        **({"cost": cost} if cost else {})}, root)


def _target_tolerances() -> dict | None:
    try:
        from lab import target
        return (target.load().raw.get("kernels") or {}).get("tolerances")
    except Exception:                           # noqa: BLE001  no target: the checker's defaults
        return None


if __name__ == "__main__":
    sys.exit(main())
