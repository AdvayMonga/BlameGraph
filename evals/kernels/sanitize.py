"""compute-sanitizer on a would-be winner, part of the held-out gate (stage 6 of evals.kernels.jailed): memcheck
(out-of-bounds and misaligned accesses), racecheck (shared-memory races), initcheck (uninitialized global reads) and
synccheck (barrier misuse), each in its own jailed process around one call per shape. Output checks miss a read past
the end that returns a harmless value, a race won by luck and memory that happens to be zero; the sanitizer sees them.

`python -m evals.kernels.sanitize CONFIG PROGRAM` is the probe the sanitizer wraps: one call of the candidate on each
of the config's shapes, seed 0, no timer. The verdict is the sanitizer's own output and exit code, never the probe's."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
EXIT = 86                                        # the sanitizer's exit code when it reported an error
_SUMMARY = re.compile(r"^=========\s+ERROR SUMMARY:\s+(\d+) errors?", re.M)
_RACE = re.compile(r"^=========\s+RACECHECK SUMMARY:.*?\((\d+) errors?", re.M)
_KERNEL = re.compile(r"\bat (?:0x[0-9a-fA-F]+ in )?(?:[^\s:]+:\d+:)?([A-Za-z_][\w:~<>]*)\s*(?:\(|\+0x)")
_BANNER = ("COMPUTE-SANITIZER", "ERROR SUMMARY", "RACECHECK SUMMARY", "Target application returned",
           "Warning")                              # racecheck's hazard warnings are not errors


def argv(exe: str, tool: str) -> list[str]:
    """The sanitizer's command line for one tool, before the probe's own."""
    return [exe, "--tool", tool, "--error-exitcode", str(EXIT), "--print-limit", "20",
            "--report-api-errors", "no",         # host API errors are Triton's loader's, the baseline's too
            *(["--leak-check", "no"] if tool == "memcheck" else [])]


def shapes(seen: list, heldout: list) -> list:
    """The smallest and largest seen shape and the largest held-out shape, by element count: a batch of 1 and the
    largest batch rarely touch the same edges."""
    size = lambda s: math.prod(int(x) for x in s)          # noqa: E731
    seen = sorted(seen, key=size)
    out = [seen[0], seen[-1]] + ([max(heldout, key=size)] if heldout else [])
    return [s for i, s in enumerate(out) if s not in out[:i]]


def parse(tool: str, stdout: str, returncode: int) -> dict:
    """One tool's run -> {tool, passed, errors, kinds, kernels, first_lines}. Passed only on exit 0 with a summary of
    0 errors; the candidate shares stdout, so every summary line counts and a printed fake can only add errors."""
    lines = [x for x in stdout.splitlines() if x.startswith("=========")]
    counts = [int(m) for m in _SUMMARY.findall(stdout)] + [int(m) for m in _RACE.findall(stdout)]
    errors = max(counts, default=0)
    heads = [i for i, x in enumerate(lines) if x[9:10] == " " and x[10:11] not in (" ", "")
             and not x[10:].startswith(_BANNER)]
    kinds = list(dict.fromkeys(re.sub(r"0x[0-9a-fA-F]+", "0x…", lines[i][10:].strip()) for i in heads))
    kernels = list(dict.fromkeys(m for x in lines for m in _KERNEL.findall(x)))
    passed = returncode == 0 and bool(counts) and errors == 0 and not heads
    out = {"tool": tool, "passed": passed, "errors": errors, "kinds": kinds[:5], "kernels": kernels[:5]}
    if not passed:
        first = lines[heads[0]:heads[0] + 12] if heads else stdout.splitlines()[-12:]
        out["first_lines"] = first
        if not heads and not counts:
            out["note"] = f"exited {returncode} with no sanitizer report"
    return out


def verdict(runs: list[dict]) -> dict:
    """The gate entry from each tool's parsed run."""
    bad = [r for r in runs if not r["passed"]]
    if not bad:
        return {"passed": True, "reason": "clean under " + ", ".join(r["tool"] for r in runs)}
    b = bad[0]
    what = "; ".join(b["kinds"][:2]) or b.get("note", "failed")
    where = f" in {', '.join(b['kernels'][:2])}" if b["kernels"] else ""
    return {"passed": False, "reason": f"{b['tool']}: {b['errors']} error(s): {what}{where}", "evidence": bad}


def probe(cfg: dict, program: Path) -> int:
    import torch
    from evals.kernels.core import load_program, screen
    from generate.kernels.tasks import from_config
    refused = screen(program.read_text())
    if refused:
        print("refused: " + "; ".join(refused), file=sys.stderr)
        return 2
    task = from_config(cfg)
    fn = task.bind(load_program(program))
    device = cfg.get("device", "cuda")
    for s in cfg["seen"]:
        fn(*task.make_inputs(tuple(s), task.dtypes[0], device, torch.Generator().manual_seed(0)))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(probe(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])))
