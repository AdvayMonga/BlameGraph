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
    return f"""The program defines `kernel`, a replacement for {base}. It must compute what the task's reference \
computes, on the same inputs, within dtype tolerance ({', '.join(str(d).removeprefix('torch.') for d in task.dtypes)}).

How a program is judged:
- Stage 1: correctness on the two most-called seen shapes, one seed. Stage 2: correctness on every seen shape (three \
seeds, edge shapes, NaN/Inf inputs, inputs scaled by {', '.join(f'{k:g}' for k in core.SCALES)}), inputs left \
unchanged, no output aliasing, no answers kept across calls; then peak memory at most {core.MEMORY_FACTOR}x the \
baseline's + {core.MEMORY_SLACK // 2**20} MiB; then timing against {base}.
- Score: the call-weighted geometric mean of the speedup over the seen shapes; a shape whose difference is within \
the timer's spread counts as 1.0. A failed gate scores 0; so does a time at which the kernel would run at \
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
    return shapes_mod.split(task, shapes_mod.from_captures(task, [Path(d) for d in a.captures or []]))


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
                            "device": a.device, "tolerances": tol}, Path(a.program).read_text(), 2, timeout_s=3600)
        print(json.dumps(out, indent=1))
        return 0 if out["metrics"]["correct"] else 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = shapes_mod.select(task, sets, a.timing_shapes)
    cfg_path = out / "eval_config.json"          # selection never sees the held-out shapes
    cfg_path.write_text(json.dumps({"task": task.name, "seen": [list(s) for s in sets["seen"]],
                                    "weights": sets.get("calls"), "device": a.device, "tolerances": tol}))
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
    (out / "best_program.py").write_text(res.best_code)
    (out / "best_metrics.json").write_text(json.dumps(res.metrics, indent=1, default=str))
    print(json.dumps({"best_score": res.best_score, "metrics": res.metrics, "out": str(out)}, default=str))
    return 0


def _target_tolerances() -> dict | None:
    try:
        from lab import target
        return (target.load().raw.get("kernels") or {}).get("tolerances")
    except Exception:                           # noqa: BLE001  no target: the checker's defaults
        return None


if __name__ == "__main__":
    sys.exit(main())
