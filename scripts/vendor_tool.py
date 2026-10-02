"""Vendor the stdlib-only tool package into the InferenceBench fork as src/eval/inference/blamegraph_tool/,
plus grader_entry.py (the pristine grader the wrapper calls). Usage: vendor_tool.py [../InferenceBench]"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "blamegraph" / "tool"
FILES = ["__init__.py", "ledger.py", "probe.py", "validate.py", "landscape.py", "context.py", "cli.py"]
GRADER_ENTRY = '''#!/usr/bin/env python3
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
'''


def main(dest_repo: str = "../InferenceBench"):
    dest = (ROOT / dest_repo).resolve() / "src" / "eval" / "inference" / "blamegraph_tool"
    dest.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy(SRC / f, dest / f)
    (dest.parent / "grader_entry.py").write_text(GRADER_ENTRY)
    print("vendored", len(FILES), "files to", dest, "+ grader_entry.py")


if __name__ == "__main__":
    main(*sys.argv[1:])
