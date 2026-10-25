"""python -m kernels.search.ab --task T --shapes SHAPES.json --out DIR [--initial PROGRAM.py] [--evals N] [--usd X]
                             [--model M] [--seeds K] [--device cuda] [--arms A,B,...]

OpenEvolve against kernels.search on one task, everything else equal: the same shapes, starting program, evaluator,
model (OpenEvolve given the same single model, so the search and prompting are what differ), number of programs judged
and post-search pipeline: the top 3 re-timed 3x, highest median promoted, no grid tuning or gaming detector for any
arm (`SAME_PROMOTION`; on 2026-10-18 only the kernels.search arms got them, and the win came from the tuning). Each arm and seed runs in its own dir; each winner then goes through `gate` (all seen shapes
plus the held-out ones, jailed). The summary, per arm: re-timed seen score, held-out speedup, whether every gate
passed, model $ and calls, programs judged; written to <out>/ab.json. An arm dir holding a result.json is reused, so
an interrupted A/B resumes. With $KERNEL_EVAL_SSH the gate runs on that host, like the evaluator. Model calls and GPU
time cost money."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import torch


def _last_json(text: str) -> dict:
    lines = [x for x in text.splitlines() if x.startswith("{")]
    return json.loads(lines[-1]) if lines else {}


def _openevolve(a, out: Path, seed: int) -> dict:
    import yaml
    from kernels.evolve import run
    cfg = yaml.safe_load((run.HERE / "config.yaml").read_text())
    cfg["llm"]["models"] = [{"name": a.model, "weight": 1.0}]
    cfg["random_seed"] = seed
    cfg["prompt"]["template_dir"] = str(run.HERE / "prompts")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run.main(["run", "--task", a.task, "--shapes", a.shapes, "--out", str(out), "--iterations", str(a.evals - 1),
                  "--config", str(out / "config.yaml"), "--device", a.device,
                  *(["--initial", a.initial] if a.initial else [])])
    return _last_json(buf.getvalue())


SAME_PROMOTION = ("--retime", "3", "--tune-points", "0", "--detector-model", "none")   # what OpenEvolve's run does


def _search(a, out: Path, seed: int, extra: tuple = ()) -> dict:
    from kernels.search.__main__ import main
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["--task", a.task, "--shapes", a.shapes, "--out", str(out), "--evals", str(a.evals), "--usd", str(a.usd),
              "--model", a.model, "--judge-model", a.model, "--wide-model", a.model, "--cheap-model", a.model,
              "--device", a.device, "--seed", str(seed),
              *(["--gpu-s", str(a.gpu_s)] if getattr(a, "gpu_s", None) else []),
              *(["--initial", a.initial] if a.initial else []), *SAME_PROMOTION, *extra])
    return _last_json(buf.getvalue())


def _variant(*extra: str):
    return lambda a, out, seed: _search(a, out, seed, extra)


# Each arm differs from `kernels.search` (the funnel, judge tools on) in one thing, so a difference in the summary is
# that thing's: the judge's web search and ledger SQL, KernelPro's literal UCT constant, the beam in place of the
# funnel (the 2026-10-24 A/B's `kernels.search` arm), or the whole generator.
ARMS = {"openevolve": _openevolve, "kernels.search": _search, "search-no-tools": _variant("--no-judge-tools"),
        "search-c1.41": _variant("--uct-c", "1.4142"), "search-beam": _variant("--no-funnel")}


def _gate(a, program: Path) -> dict:
    from kernels.evolve import run
    from kernels.evolve.ship import gate_passed
    remote = os.environ.get("KERNEL_EVAL_SSH")
    if remote:                    # the GPU is where the evaluator runs: the gate runs there, on copies sent by stdin
        tar = io.BytesIO()
        with tarfile.open(fileobj=tar, mode="w") as t:
            t.add(a.shapes, "shapes.json")
            t.add(program, "best_program.py")
        cmd = ('d=$(mktemp -d) && tar -xf - -C "$d" && ' + os.environ.get("KERNEL_EVAL_REMOTE", "") +
               f' python -m kernels.evolve gate --task {a.task} --shapes "$d/shapes.json"'
               f' --program "$d/best_program.py" --device {a.device}')
        text = subprocess.run(json.loads(remote) + [cmd], input=tar.getvalue(), capture_output=True).stdout.decode()
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run.main(["gate", "--task", a.task, "--shapes", a.shapes, "--program", str(program), "--device", a.device])
        text = buf.getvalue()
    g = json.loads(text[text.index("{"):]) if "{" in text else {}       # gate prints one indented JSON object
    return {**g, "passed": gate_passed(g) if g else False}


def _cost(arm_dir: Path) -> dict:
    from lab import ledger
    rid = json.loads((arm_dir / "result.json").read_text()).get("ledger")
    rec = next((r for r in ledger.records(ledger.ROOT) if r.get("id") == rid), None) if rid else None
    return (rec or {}).get("cost") or {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernels.search.ab")
    ap.add_argument("--task", required=True)
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--initial")
    ap.add_argument("--evals", type=int, default=30)
    ap.add_argument("--usd", type=float, default=10.0, help="kernels.search's cap; OpenEvolve has none (measured)")
    ap.add_argument("--gpu-s", type=float, help="kernels.search arms stop at this many judging seconds (the funnel's "
                    "rounds judge many more programs per evaluation counted, so compare at equal GPU time)")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arms", default="openevolve,kernels.search", help=f"comma-separated, of: {', '.join(ARMS)}")
    a = ap.parse_args(argv)
    arms = [x for x in a.arms.split(",") if x]
    unknown = [x for x in arms if x not in ARMS]
    if unknown:
        ap.error(f"unknown arms {unknown}; one of {list(ARMS)}")
    out, rows = Path(a.out), []
    for seed in range(a.seeds):
        for arm in arms:
            fn = ARMS[arm]
            d = out / f"{arm}-{seed}"
            d.mkdir(parents=True, exist_ok=True)
            if (d / "result.json").exists():         # a finished arm from an earlier invocation is kept, not rerun
                res = json.loads((d / "result.json").read_text())
            else:
                res = fn(a, d, seed)
                (d / "result.json").write_text(json.dumps(res, default=str))
            g = _gate(a, d / "best_program.py")
            (d / "gate.json").write_text(json.dumps(g, indent=1))       # what `kernels.evolve ship --gate` reads
            c = _cost(d)
            m = (g.get("metrics") or {})
            rows.append({"arm": arm, "seed": seed, "seen_score": m.get("combined_score"),
                         "heldout_speedup": m.get("heldout_speedup"), "gates_passed": g["passed"],
                         "usd": c.get("usd"), "model_calls": c.get("model_calls"), "evals": c.get("evals"),
                         "ledger": res.get("ledger")})
            (out / "ab.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
