"""One-time setup of a target on this machine: the corpus, the correctness reference, the noise bands.

  lab setup --task TASK.toml [--target T]

A step is skipped where its output exists: the corpus, the reference, a noise band for each of the task's regimes.
GPU work runs on the worker (LAB_WORKER).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from evals.targets import target
from lab.task import Task


def steps(t: target.Target, regimes: tuple[str, ...]) -> list[list[str]]:
    """The `lab` commands still to run for `t` and these regimes, in order."""
    from lab.evaltools import noise_band_pct
    todo = []
    if not t.corpus_dir.is_dir() or not any(t.corpus_dir.glob("*/heldout.jsonl")):
        todo.append(["store", "pull", "--corpus"])
    if t.reference_dir is not None and not target.reference_complete(t.reference_dir):
        todo.append(["correctness", "reference", "--out", str(t.reference_dir)])
    unmeasured = [n for n in regimes if noise_band_pct(n, t) is None]
    if unmeasured:
        todo.append(["validate", "--regimes", ",".join(unmeasured)])
    return todo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lab setup", description=__doc__.split("\n")[0])
    target.add_argument(ap)
    ap.add_argument("--task", required=True, help="task spec TOML; its regimes are the ones measured")
    a = ap.parse_args(argv)
    if a.target:
        os.environ["LAB_TARGET"] = a.target
    for step in steps(target.load(), Task.parse(a.task).regimes):
        print("lab", *step, flush=True)
        code = subprocess.run([sys.executable, "-m", "lab", *step]).returncode
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
