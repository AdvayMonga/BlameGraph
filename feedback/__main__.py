"""python -m feedback PATH [--scenario A] [--run RUN_ID] [--agent-text]"""
from __future__ import annotations

import argparse
import json

from feedback.report import feedback, render_for_agent


def main():
    ap = argparse.ArgumentParser(prog="feedback", description="feedback for one finished session (run dir, session .jsonl, or lab ledger)")
    ap.add_argument("path")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--run", default=None, help="lab ledger: the run to report (default: the latest)")
    ap.add_argument("--agent-text", action="store_true", help="print only the agent-facing half as plain text")
    a = ap.parse_args()
    fb = feedback(a.path, a.scenario, a.run)
    print(render_for_agent(fb) if a.agent_text else json.dumps(fb, indent=1, default=str))


if __name__ == "__main__":
    main()
