"""Command line used by the harness wrappers and by the loop. Stdlib only.

  python -m blamegraph.tool.cli session-start --task-dir DIR --scenario A
  python -m blamegraph.tool.cli measure --task-dir DIR --port 8000 -- <evaluate.py args...>   (runs the real grader, records)
  python -m blamegraph.tool.cli submit  --task-dir DIR [--pristine-grader-hash HASH]
  python -m blamegraph.tool.cli validate --task-dir DIR
  python -m blamegraph.tool.cli context --task-dir DIR --scenario A [--landscape PATH] [--text]
  python -m blamegraph.tool.cli landscape-add --task-dir DIR --scenario A --landscape PATH --session-id ID
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .context import pack, pack_text
from .landscape import Landscape
from .ledger import Ledger
from .probe import live_server
from .validate import validate

STANDARD_FLAGS = {"--quick", "--json-output-file", "--server-url", "--model", "--host", "--port"}
LEDGER_NAME = ".blamegraph/ledger.jsonl"


def paths(task_dir: str):
    td = Path(task_dir)
    return td, td / LEDGER_NAME, td / "start_server.sh", [td / "evaluate.py"]


def is_standard(args: list[str]) -> bool:
    flags = {a.split("=")[0] for a in args if a.startswith("--")}
    env_overrides = [k for k in os.environ if k.startswith("INFERENCE_BENCH_") and k != "INFERENCE_BENCH_BASE_MODEL"]
    return not (flags - STANDARD_FLAGS) and not env_overrides


def cmd_session_start(a):
    td, lp, cfg, graders = paths(a.task_dir)
    led = Ledger.open(lp)
    led.start_session(task={"scenario": a.scenario, "task_dir": str(td)}, grader_paths=graders)
    led.record_config(cfg)
    print(json.dumps({"ledger": str(lp), "events": len(led.events)}))


def cmd_measure(a, rest: list[str]):
    """Run the real grader (`python3 <grader> rest...`), then record the result it wrote."""
    td, lp, cfg, graders = paths(a.task_dir)
    led = Ledger.open(lp)
    if not led.events:
        led.start_session(task={"task_dir": str(td)}, grader_paths=graders)
    out_file = None
    for i, tok in enumerate(rest):
        if tok == "--json-output-file" and i + 1 < len(rest):
            out_file = Path(rest[i + 1])
        elif tok.startswith("--json-output-file="):
            out_file = Path(tok.split("=", 1)[1])
    if out_file is not None and not out_file.is_absolute():
        out_file = td / out_file
    port = a.port
    for i, tok in enumerate(rest):
        if tok == "--server-url" and i + 1 < len(rest) and ":" in rest[i + 1]:
            try:
                port = int(rest[i + 1].rsplit(":", 1)[1].split("/")[0])
            except ValueError:
                pass
    server = live_server(port, cfg).to_dict() if port else None
    proc = subprocess.run([sys.executable, a.grader, *rest], cwd=td)   # graders resolve files relative to the task dir
    metrics, err = None, None
    if out_file and out_file.exists():
        try:
            metrics = json.loads(out_file.read_text())
        except json.JSONDecodeError as e:
            err = f"unreadable metrics file: {e}"
    else:
        err = f"grader exit {proc.returncode}, no metrics file"
    ev = led.record_measure(cfg, mode="quick" if "--quick" in rest else "full", flags=[t for t in rest if t.startswith("--")],
                            metrics=metrics, standard=is_standard(rest), server=server, error=err,
                            request_set=next((rest[i + 1] for i, t in enumerate(rest) if t == "--requests-file" and i + 1 < len(rest)), None))
    print(json.dumps({"recorded": ev["id"], "stale_server": ev["stale"], "cache_state": ev["cache_state"], "standard": ev["standard"], "error": err}), file=sys.stderr)
    sys.exit(proc.returncode)


def cmd_submit(a):
    td, lp, cfg, graders = paths(a.task_dir)
    led = Ledger.open(lp)
    led.record_submission(cfg, graders)
    pristine = {str(graders[0]): a.pristine_grader_hash} if a.pristine_grader_hash else None
    v = validate(led, cfg, graders, pristine)
    out = td / ".blamegraph" / "validation.json"
    out.write_text(json.dumps(v.to_dict(), indent=1))
    print(json.dumps(v.to_dict()))
    sys.exit(0 if v.valid else 3)


def cmd_validate(a):
    td, lp, cfg, graders = paths(a.task_dir)
    v = validate(Ledger.open(lp), cfg, graders, {str(graders[0]): a.pristine_grader_hash} if a.pristine_grader_hash else None)
    print(json.dumps(v.to_dict(), indent=1))


def cmd_context(a):
    td, lp, cfg, graders = paths(a.task_dir)
    land = Landscape(Path(a.landscape)) if a.landscape else None
    ctx = pack(Ledger.open(lp), cfg, a.scenario, graders, land, baseline_metric=a.baseline_metric)
    print(pack_text(ctx) if a.text else json.dumps(ctx, indent=1))


def cmd_landscape_add(a):
    td, lp, cfg, graders = paths(a.task_dir)
    land = Landscape(Path(a.landscape))
    n = land.add_session(Ledger.open(lp), a.scenario, a.session_id)
    land.save()
    print(json.dumps({"added": n, "points": len(land.points)}))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        k = argv.index("--"); argv, rest = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(prog="blamegraph.tool")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("session-start"); p.add_argument("--task-dir", required=True); p.add_argument("--scenario", default=None)
    p = sub.add_parser("measure"); p.add_argument("--task-dir", required=True); p.add_argument("--grader", required=True, help="path to the real evaluate.py"); p.add_argument("--port", type=int, default=None)
    p = sub.add_parser("submit"); p.add_argument("--task-dir", required=True); p.add_argument("--pristine-grader-hash", default=None)
    p = sub.add_parser("validate"); p.add_argument("--task-dir", required=True); p.add_argument("--pristine-grader-hash", default=None)
    p = sub.add_parser("context"); p.add_argument("--task-dir", required=True); p.add_argument("--scenario", required=True); p.add_argument("--landscape", default=None); p.add_argument("--baseline-metric", type=float, default=None); p.add_argument("--text", action="store_true")
    p = sub.add_parser("landscape-add"); p.add_argument("--task-dir", required=True); p.add_argument("--scenario", required=True); p.add_argument("--landscape", required=True); p.add_argument("--session-id", required=True)
    a = ap.parse_args(argv)
    {"session-start": cmd_session_start, "submit": cmd_submit, "validate": cmd_validate, "context": cmd_context, "landscape-add": cmd_landscape_add}.get(a.cmd, lambda a: cmd_measure(a, rest))(a)


if __name__ == "__main__":
    main()
