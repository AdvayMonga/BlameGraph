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
from kernels.evolve.core import judge
from kernels.tasks import TASKS, load

HERE = Path(__file__).resolve().parent


def initial_program(task) -> str:
    """The engine's module source with `kernel` bound to the entry point, all of it open to evolution."""
    if task.entry is None:
        return (HERE / "seeds" / f"{task.name}.py").read_text()
    mod, _, attr = task.entry.partition(":")
    src = inspect.getsource(importlib.import_module(mod))
    return f"# EVOLVE-BLOCK-START\n{src}\n# EVOLVE-BLOCK-END\n\nkernel = {attr}\n"


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
        out = judge(task, Path(a.program), sets["seen"], sets["heldout"], sets.get("calls"), a.device,
                    {getattr(torch, k): tuple(v) for k, v in tol.items()} or None, stage=2)
        print(json.dumps(out, indent=1))
        return 0 if out["metrics"]["correct"] else 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
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
