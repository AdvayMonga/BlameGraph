#!/usr/bin/env python3
"""Pristine grader entry point (what evaluate.py used to be). Called by the BlameGraph measure wrapper."""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/opt")
from inference_eval.runner import build_parser, run_evaluation  # noqa: E402


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(Path(os.environ.get("BLAMEGRAPH_TASK_DIR", Path.cwd())), args)


if __name__ == "__main__":
    main()
