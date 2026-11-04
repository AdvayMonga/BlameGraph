"""lab <command> [args]: every command of the environment under one name, and every tool the agent has, called by the
same name with the same arguments (`lab TOOL --run RUN ...`) on a researcher run from `lab start`."""
from __future__ import annotations

import argparse
import ast
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# command -> the module it runs; a dict is a group: `lab kernels search ...`
COMMANDS = {
    "session": "lab.session", "rehearsal": "lab.rehearsal", "dashboard": "evals.researcher.dashboard",
    "validate": "evals.agent.validate", "feedback": "evals.researcher.feedback",
    "store": "data.knowledge", "records": "data.knowledge.ledger", "import-runs": "data.knowledge.import_runs",
    "workloads": "data.workloads",
    "corpus": {"build": "data.corpus.build_corpus", "fetch": "data.corpus.fetch_traces", "lengths": "data.corpus.lengths"},
    "regimes": "measure.loadgen", "clientcheck": "measure.loadgen.clientcheck", "synth": "measure.loadgen.synth",
    "arrivals": "measure.loadgen.arrivals", "sim": "measure.sim", "crossover": "measure.profile.crossover",
    "vm": "measure.cloud.vm", "gpucheck": "measure.cloud.gpucheck", "jail": "measure.sandbox.container",
    "peaks": "measure.context.peaks", "layers": "measure.context.layers",
    "trace-contract": "measure.runner.trace_contract",
    "correctness": "evals.agent.correctness", "repeat-audit": "evals.agent.repeat_audit",
    "cheat-proxy": "evals.researcher.canaries", "canary": "evals.researcher.canaries.canary",
    "compare": {"run": "evals.benches.compare.run", "report": "evals.benches.compare.report",
                "profile": "evals.benches.compare.profile", "gap": "evals.benches.compare.gap"},
    "kernels": {"search": "generate.kernels.search", "ab": "generate.kernels.search.ab",
                "evolve": "generate.kernels.evolve", "library": "generate.kernels.library",
                "archive": "generate.kernels.archive", "ptx": "generate.kernels.library.ptx",
                "competitors": "generate.kernels.library.competitors",
                "kernelbench": "generate.kernels.tasks.kernelbench",
                "flashinfer-bench": "generate.kernels.tasks.flashinfer_bench"},
}
RUN_HELP = "a researcher run (lab start)"


def source(module: str) -> Path:
    p = ROOT.joinpath(*module.split("."))
    return p.with_suffix(".py") if p.with_suffix(".py").exists() else p / "__main__.py"


def summary(module: str) -> str:
    """The first line of the module's docstring (a package's own when its __main__ has none)."""
    for f in (source(module), source(module).with_name("__init__.py")):
        doc = f.exists() and ast.get_docstring(ast.parse(f.read_text()))
        if doc:
            return doc.split("\n")[0][:96]
    return ""


def usage() -> str:
    rows = [f"  {c + ' ' + s:<20} {summary(m)}" for c, v in COMMANDS.items()
            for s, m in (v.items() if isinstance(v, dict) else [("", v)])]
    return "\n".join(["lab <command> [args]   (lab <command> -h: its arguments)", "", *rows, "",
                      "  start                a researcher run: --base REF --change REF --budget USD [--repo PATH]",
                      "  baseline             the base commit measured once for a run: --run RUN",
                      "  TOOL                 any tool the agent has, same name and arguments: --run RUN ..."])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    cmd, *rest = argv
    module = COMMANDS.get(cmd)
    if isinstance(module, dict):
        if not rest or rest[0] not in module:
            print(f"usage: lab {cmd} {{{','.join(module)}}} ...", file=sys.stderr)
            return 2
        sub, *rest = rest
        cmd, module = f"{cmd} {sub}", module[sub]
    if module:
        sys.argv = [f"lab {cmd}", *rest]
        runpy.run_module(module, run_name="__main__")
        return 0
    try:
        print(experiment_command(cmd, rest))
    except (ValueError, PermissionError) as e:      # a refusal is the lab's answer, not a crash
        print(f"refused: {e}", file=sys.stderr)
        return 1
    return 0


def experiment_command(cmd: str, argv: list[str]) -> str:
    from data.knowledge import ledger
    from lab import experiment
    from targets import engine
    if cmd == "start":
        ap = argparse.ArgumentParser(prog="lab start", description="a run whose workspace is BASE plus the files "
                                     "CHANGE touched inside the write surface; prints the run id")
        ap.add_argument("--repo", type=Path, default=None, help="the engine repo (default: LAB_ENGINE_REPO)")
        ap.add_argument("--base", required=True)
        ap.add_argument("--change", required=True)
        ap.add_argument("--budget", type=float, required=True, help="dollars of GPU time for this run")
        a = ap.parse_args(argv)
        return experiment.start(a.repo or engine.repo(), a.base, a.change, a.budget, ledger.ROOT, experiment.RUNS)
    pre = argparse.ArgumentParser(prog=f"lab {cmd}", add_help=False)
    pre.add_argument("--run")
    run = pre.parse_known_args(argv)[0].run
    if run is None:
        sys.exit(f"lab: {cmd!r} is not a command; a lab tool needs --run RUN (lab start). lab -h lists the commands.")
    tb = experiment.toolbox(run, ledger.ROOT, experiment.RUNS)
    if cmd == "baseline":
        ap = argparse.ArgumentParser(prog="lab baseline", description="the run's base commit, measured once")
        ap.add_argument("--run", required=True, help=RUN_HELP)
        ap.parse_args(argv)
        return json.dumps(tb.baseline(tb.s.workspace.base)["result"])
    spec = next((t for t in tb.specs() if t.name == cmd.replace("-", "_")), None)
    if spec is None:
        raise ValueError(f"no lab command or tool {cmd!r}; tools: {sorted(t.name for t in tb.specs())}")
    a = vars(parser(spec).parse_args(argv))
    return spec.fn({k: v for k, v in a.items() if v is not None and k != "run"})


def parser(spec) -> argparse.ArgumentParser:
    """The tool's JSON schema as flags: `--hypothesis-id H`, objects and non-text arrays as JSON."""
    ap = argparse.ArgumentParser(prog=f"lab {spec.name.replace('_', '-')}", description=spec.description)
    ap.add_argument("--run", required=True, help=RUN_HELP)
    for k, p in spec.schema.get("properties", {}).items():
        t, items = p.get("type"), p.get("items", {}).get("type")
        kw = ({"type": int} if t == "integer" else {"type": float} if t == "number"
              else {"action": argparse.BooleanOptionalAction} if t == "boolean"
              else {"nargs": "+"} if t == "array" and items == "string"
              else {"type": json.loads, "metavar": "JSON"} if t in ("object", "array") else {})
        ap.add_argument("--" + k.replace("_", "-"), dest=k, required=k in spec.schema.get("required", ()),
                        choices=p.get("enum"), help=(p.get("description") or "").replace("%", "%%") or None, **kw)
    return ap
