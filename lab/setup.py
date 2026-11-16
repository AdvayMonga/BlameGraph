"""One-time setup of a target on this machine: the corpus, the correctness reference, the validation.

  lab setup [--target T]

Each step is skipped where its output exists.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from evals.targets import target


def steps(t: target.Target) -> list[list[str]]:
    """The `lab` commands still to run for `t`, in order."""
    todo = []
    if not t.corpus_dir.is_dir() or not any(t.corpus_dir.glob("*/heldout.jsonl")):
        todo.append(["store", "pull", "--corpus"])
    if not target.reference_complete(t.reference_dir):
        todo.append(["correctness", "reference", "--out", str(t.reference_dir)])
    return [*todo, ["validate"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lab setup", description=__doc__.split("\n")[0])
    target.add_argument(ap)
    a = ap.parse_args(argv)
    if a.target:
        os.environ["LAB_TARGET"] = a.target
    for step in steps(target.load()):
        print("lab", *step, flush=True)
        code = subprocess.run([sys.executable, "-m", "lab", *step]).returncode
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
