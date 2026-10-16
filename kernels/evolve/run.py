"""python -m kernels.evolve run --task T [--shapes SHAPES.json | --captures DIR ...] --out DIR [--iterations N]
                               [--config CONFIG.yaml] [--device cuda] [--initial PROGRAM.py]
python -m kernels.evolve gate --task T --shapes SHAPES.json --program BEST.py [--device cuda]

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
import sys
from pathlib import Path

import torch

from kernels import shapes as shapes_mod
from kernels.evolve.jailed import judge_jailed
from kernels.tasks import TASKS, load

HERE = Path(__file__).resolve().parent


def initial_program(task) -> str:
    """The engine's module source with `kernel` bound to the entry point, all of it open to evolution."""
    if task.entry is None:
        return (HERE / "seeds" / f"{task.name}.py").read_text()
    mod, _, attr = task.entry.partition(":")
    src = inspect.getsource(importlib.import_module(mod))
    return f"# EVOLVE-BLOCK-START\n{src}\n# EVOLVE-BLOCK-END\n\nkernel = {attr}\n"


def system_message(task) -> str:
    """What the program is and how it is judged, from the evaluator's own constants: facts, never advice."""
    from kernels.evolve import core
    base = f"the engine's current `{task.entry}`" if task.entry else "the task's reference"
    refused = "; ".join(dict.fromkeys(core.FORBIDDEN.values()))
    graphs = (", both captured in CUDA graphs and timed as graph replays, as the engine runs them (a kernel that "
              "cannot be captured scores 0)") if task.graphs else ""
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
error against the reference beside the baseline's.

The goal is the highest score."""


def tolerances(target_tol: dict | None) -> dict:
    """The target spec's [kernels] tolerances ({dtype: [atol, rtol]}), as the checker's override."""
    return {k: list(v) for k, v in (target_tol or {}).items()}


def _shapes(task, a) -> dict:
    if a.shapes:
        return shapes_mod.read(a.shapes)
    dirs = [Path(d) for d in a.captures or []]
    sets = shapes_mod.split(task, shapes_mod.from_captures(task, dirs))
    return {**sets, "ints": shapes_mod.ints_from_captures(task, dirs)} if dirs else sets


def _ints(sets: dict, which: tuple[str, ...]) -> dict:
    """The captured ints of the shapes in `which` (selection gets the seen ones only)."""
    keys = {json.dumps(list(s)) for k in which for s in sets[k]}
    return {k: v for k, v in (sets.get("ints") or {}).items() if k in keys}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernels.evolve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "gate", "split"):
        p = sub.add_parser(name)
        p.add_argument("--task", required=True, choices=TASKS)
        p.add_argument("--shapes")
        p.add_argument("--captures", action="append")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    task = load(a.task)
    sets = _shapes(task, a)
    tol = tolerances(_target_tolerances())
    if a.cmd == "split":
        print(shapes_mod.write(a.write, sets))
        return 0
    if a.cmd == "gate":
        out = judge_jailed({"task": task.name, "seen": [list(s) for s in sets["seen"]],
                            "heldout": [list(s) for s in sets["heldout"]], "weights": sets.get("calls"),
                            "device": a.device, "tolerances": tol, "ints": _ints(sets, ("seen", "heldout"))},
                           Path(a.program).read_text(), 2, timeout_s=3600)
        print(json.dumps(out, indent=1))
        return 0 if out["metrics"]["correct"] else 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = shapes_mod.select(task, sets, a.timing_shapes)
    cfg_path = out / "eval_config.json"          # selection never sees the held-out shapes
    cfg_path.write_text(json.dumps({"task": task.name, "seen": [list(s) for s in sets["seen"]],
                                    "weights": sets.get("calls"), "device": a.device, "tolerances": tol,
                                    "ints": _ints(sets, ("seen",))}))
    (out / "shapes.json").write_text(json.dumps({k: [list(s) for s in v] if k in ("seen", "heldout") else v
                                                 for k, v in sets.items()}, indent=1))
    program = Path(a.initial) if a.initial else out / "initial_program.py"
    if not a.initial:
        program.write_text(initial_program(task))
    os.environ["KERNEL_EVAL_CONFIG"] = str(cfg_path)
    if "OPENAI_API_KEY" not in os.environ and "ANTHROPIC_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]     # the config's endpoint is Anthropic's
    from openevolve import run_evolution
    from openevolve.config import load_config
    from kernels.evolve.llm import init_client
    config = load_config(a.config)
    config.prompt.system_message = system_message(task)
    if "anthropic.com" in (config.llm.api_base or ""):       # Claude refuses a temperature; OpenEvolve always sends one
        config.llm.update_model_params({"init_client": init_client}, overwrite=True)
    res = run_evolution(str(program), str(HERE / "evaluator.py"), config, iterations=a.iterations,
                        output_dir=str(out / "openevolve"), cleanup=False)
    trace = out / "openevolve" / "evolution_trace.jsonl"
    code, metrics, retimed = retime(trace, a.retime) if a.retime else (None, None, [])
    (out / "best_program.py").write_text(code or res.best_code)
    (out / "best_metrics.json").write_text(json.dumps(metrics or res.metrics, indent=1, default=str))
    rec = record(task, sets, config, a.iterations, a.device, trace, out / "best_program.py", metrics or res.metrics,
                 retimed=retimed)
    print(json.dumps({"best_score": res.best_score, "metrics": res.metrics, "out": str(out), "ledger": rec["id"]},
                     default=str))
    return 0


def retime(trace: Path, k: int, repeats: int = 3) -> tuple[str | None, dict | None, list]:
    """The top `k` candidates by score re-judged `repeats` times through the evaluator (stage 2, local or remote);
    the one with the highest median score is promoted. One timing of each is what selection saw, and one timing can
    win by noise. Returns (code, metrics of its median run, every re-timing)."""
    import statistics
    import tempfile
    from kernels.evolve import evaluator
    rows = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()] if trace.exists() else []
    best: dict = {}
    for r in rows:
        if r.get("child_code") and (r.get("child_metrics") or {}).get("combined_score", 0) > 0:
            best.setdefault(r["child_id"], r)
    top = sorted(best.values(), key=lambda r: -r["child_metrics"]["combined_score"])[:k]
    out = []
    for r in top:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(r["child_code"])
        runs = [evaluator.evaluate_stage2(f.name).metrics for _ in range(repeats)]
        Path(f.name).unlink()
        scores = [m.get("combined_score", 0.0) for m in runs]
        med = statistics.median(scores)
        out.append({"id": r["child_id"], "selected_score": r["child_metrics"]["combined_score"], "scores": scores,
                    "median": med, "code": r["child_code"],
                    "metrics": min(runs, key=lambda m: abs(m.get("combined_score", 0.0) - med))})
    if not out:
        return None, None, []
    win = max(out, key=lambda x: x["median"])
    return win["code"], win["metrics"], [{k: v for k, v in x.items() if k not in ("code", "metrics")} for x in out]


def record(task, sets: dict, config, iterations: int, device: str, trace: Path, best: Path, best_metrics: dict,
           root: Path | None = None, retimed: list | None = None) -> dict:
    """One `kernels` ledger record per run: every evaluated candidate's seen-shape metrics and lineage, the trace and
    the best program as blobs. Held-out shapes stay out (`gate` judges them; its output stays in the run dir)."""
    import time
    from lab import ledger
    root = root or ledger.ROOT
    rows = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()] if trace.exists() else []
    cands = [{"id": r["child_id"], "parent": r["parent_id"], "iteration": r["iteration"],
              "island": r.get("island_id"), "metrics": r.get("child_metrics")} for r in rows]
    return ledger.append({
        "kind": "kernels", "run": f"kernels-{time.strftime('%Y%m%d-%H%M%S')}", "tool": "kernels.evolve.run",
        "config": {"split": "seen", "task": task.name, "entry": task.entry, "iterations": iterations,
                   "device": device, "seen_shapes": len(sets["seen"]), "selected_from": sets.get("selected_from"),
                   "models": [{"name": m.name, "weight": m.weight} for m in config.llm.models]},
        "result": {"best": best_metrics, "candidates": cands, "retimed": retimed or [],
                   "trace": ledger.put_blob(trace, root, ".jsonl") if trace.exists() else None,
                   "best_program": ledger.put_blob(best, root, ".py")}}, root)


def _target_tolerances() -> dict | None:
    try:
        from lab import target
        return (target.load().raw.get("kernels") or {}).get("tolerances")
    except Exception:                           # noqa: BLE001  no target: the checker's defaults
        return None


if __name__ == "__main__":
    sys.exit(main())
