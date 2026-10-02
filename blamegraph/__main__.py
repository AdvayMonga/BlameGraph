"""python -m blamegraph feedback PATH [--scenario A] [--agent-text]"""
from __future__ import annotations

import argparse
import json

from .feedback import feedback, render_for_agent


def main():
    ap = argparse.ArgumentParser(prog="blamegraph")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("feedback", help="feedback for one finished session (run dir or session .jsonl)")
    p.add_argument("path")
    p.add_argument("--scenario", default=None)
    p.add_argument("--agent-text", action="store_true", help="print only the agent-facing half as plain text")
    a = ap.parse_args()
    fb = feedback(a.path, a.scenario)
    print(render_for_agent(fb) if a.agent_text else json.dumps(fb, indent=1, default=str))


if __name__ == "__main__":
    main()
