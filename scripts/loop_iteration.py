"""One loop iteration's bookkeeping. Usage:
  loop_iteration.py ingest  --eval-dir DIR [--landscape PATH] [--session-id ID]      -> session result JSON
  loop_iteration.py compare --candidate a.json,b.json,... --incumbent x.json,y.json,...  (ingest outputs, seed-ordered)
  loop_iteration.py context --task-dir DIR --scenario A --landscape PATH [--text]       (what the next session may read)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.tool.context import pack, pack_text  # noqa: E402
from blamegraph.tool.landscape import Landscape  # noqa: E402
from blamegraph.tool.ledger import Ledger  # noqa: E402
from blamegraph.tool.loop import ingest, paired_compare  # noqa: E402

DEFAULT_LAND = Path(__file__).resolve().parent.parent / "data" / "landscape.jsonl"


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("--eval-dir", required=True); p.add_argument("--landscape", default=str(DEFAULT_LAND)); p.add_argument("--session-id"); p.add_argument("--scenario")
    p = sub.add_parser("compare"); p.add_argument("--candidate", required=True); p.add_argument("--incumbent", required=True)
    p = sub.add_parser("context"); p.add_argument("--task-dir", required=True); p.add_argument("--scenario", required=True); p.add_argument("--landscape", default=str(DEFAULT_LAND)); p.add_argument("--text", action="store_true")
    a = ap.parse_args()
    if a.cmd == "ingest":
        land = Landscape(Path(a.landscape)); r = ingest(Path(a.eval_dir), land, a.session_id, a.scenario); land.save()
        print(json.dumps(r.to_dict(), indent=1))
    elif a.cmd == "compare":
        load = lambda s: [json.loads(Path(f).read_text()).get("official_metric") for f in s.split(",")]
        print(json.dumps(paired_compare(load(a.candidate), load(a.incumbent)), indent=1))
    else:
        td = Path(a.task_dir); ctx = pack(Ledger.open(td / ".blamegraph/ledger.jsonl"), td / "start_server.sh", a.scenario, [td / "evaluate.py"], Landscape(Path(a.landscape)))
        print(pack_text(ctx) if a.text else json.dumps(ctx, indent=1))


if __name__ == "__main__":
    main()
