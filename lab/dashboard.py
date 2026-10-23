"""A live view of lab runs for the researcher: `python -m lab.dashboard [--port 8080]`, then open the printed URL.

Reads only the ledger and each run's `live.jsonl` (lab/live.py), so it touches neither the engine, the GPU nor the
agent's workspace, and nothing it shows reaches the agent. Served on 127.0.0.1 only."""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from lab import engine, experiments, ledger

PAGE = Path(__file__).parent / "dashboard.html"
RUNS_DIR = engine.ENV_ROOT / "lab" / "runs"
MEASURED_KINDS = ("baseline", "bench", "equiv", "submit", "profile", "test")
FEED = 60                                   # agent events shown, newest first


def events(run_dir: Path) -> list[dict]:
    """A run's progress lines; a torn last line (a write in flight) is skipped."""
    p = run_dir / "live.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _epoch(at: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(at).timestamp()


def runs(recs: list[dict], runs_dir: Path) -> list[dict]:
    """Every run with ledger records or a progress file, most recently active first."""
    last: dict[str, float] = {}
    for r in recs:
        if r.get("run"):
            last[r["run"]] = max(last.get(r["run"], 0.0), _epoch(r["at"]))
    for p in runs_dir.glob("*/live.jsonl"):
        last[p.parent.name] = max(last.get(p.parent.name, 0.0), p.stat().st_mtime)
    return [{"id": k, "last": v} for k, v in sorted(last.items(), key=lambda kv: -kv[1])]


def _headline(r: dict) -> dict:
    """The few numbers a measured record is read by."""
    res = r.get("result") or {}
    if r["kind"] in ("bench", "baseline"):
        return {"metrics": {k: {"value": m.get("value"), "objective": m.get("objective"), "valid": m.get("valid")}
                            for k, m in (res.get("metrics") or {}).items()}}
    if r["kind"] == "submit":
        return {"metrics": r.get("metrics")}
    if r["kind"] == "equiv":
        acc = ((res.get("gates") or {}).get("accuracy") or {}).get("evidence") or {}
        return {"tier": (r.get("config") or {}).get("tier"), "score_ratio": acc.get("score_ratio"), "n": acc.get("n")}
    if r["kind"] == "test":
        return {"lint": res.get("lint"), "tests": res.get("tests")}
    return {}


def _now(evs: list[dict]) -> dict:
    """What is happening at the end of the progress file: the agent's last step, the open worker job, the load."""
    agent = next((e for e in reversed(evs) if e["event"] == "agent"), None)
    doing = None
    if agent:
        doing = ({"what": f"running {agent['name']}", "detail": agent.get("detail"), "since": agent["ts"]}
                 if agent["kind"] == "tool" else {"what": "model is thinking", "since": agent["ts"]})
    open_jobs: dict[str, float] = {}
    for e in evs:
        if e["event"] == "job":
            if e["state"] == "start":
                open_jobs[e["job"]] = e["ts"]
            else:
                open_jobs.pop(e["job"], None)
    regime = next((i for i in range(len(evs) - 1, -1, -1) if evs[i]["event"] == "regime"), None)
    load = None
    if regime is not None:
        load = {"regime": evs[regime]["regime"], "since": evs[regime]["ts"],
                "probes": [e for e in evs[regime + 1:] if e["event"] == "probe"]}
    return {"agent": doing, "jobs": [{"job": j, "since": t} for j, t in open_jobs.items()], "load": load}


def state(run: str | None, ledger_root: Path = ledger.ROOT, runs_dir: Path = RUNS_DIR) -> dict:
    recs = list(ledger.records(ledger_root))
    all_runs = runs(recs, runs_dir)
    run = run or (all_runs[0]["id"] if all_runs else None)
    if run is None:
        return {"runs": [], "run": None, "now": time.time()}
    mine = [r for r in recs if r.get("run") == run]
    evs = events(runs_dir / run)
    start = max((i for i, e in enumerate(evs) if e["event"] == "run"), default=None)
    evs = evs[start:] if start is not None else evs                   # the latest start of this run
    sessions = [e for e in evs if e["event"] == "session"]
    cur = sessions[-1] if sessions else None
    recorded = {r["session"] for r in mine if r["kind"] == "session"}
    since_cur = evs[evs.index(cur):] if cur else []
    live_model = (max((e.get("spent_usd") or 0.0 for e in since_cur if e["event"] == "agent"), default=0.0)
                  if cur and cur["session"] not in recorded else 0.0)
    model = sum((r.get("cost") or {}).get("usd", 0.0) for r in mine if r["kind"] == "session") + live_model
    gpu = sum((r.get("cost") or {}).get("usd", 0.0) for r in mine if r["kind"] != "session")
    begun = next((e for e in reversed(evs) if e["event"] == "run"), None)
    ended = next((e for e in reversed(evs) if e["event"] == "run_end"), None)
    return {
        "now": time.time(), "runs": all_runs[:40], "run": run,
        "status": ("ended: " + str(ended["stopped"])) if ended else ("running" if begun else "no progress file"),
        "last_event": evs[-1]["ts"] if evs else None,
        "budget": {"cap_usd": begun["cap_usd"] if begun else None, "model_usd": model, "gpu_usd": gpu},
        "session": cur and {"id": cur["session"], "n": cur["n"], "cap_usd": cur["cap_usd"], "since": cur["ts"],
                            "tool_calls": sum(e["event"] == "agent" and e["kind"] == "tool" for e in since_cur),
                            "model_usd": live_model, "ended": cur["session"] in recorded},
        "now_doing": _now(evs),
        "feed": [e for e in evs if e["event"] == "agent" and e["kind"] != "result"][-FEED:][::-1],
        "hypotheses": experiments.summary(ledger_root, run),
        "measurements": [{"id": r["id"], "at": r["at"], "kind": r["kind"], "tool": r.get("tool"),
                          "verdict": (r.get("result") or {}).get("verdict"),
                          "reason": (r.get("result") or {}).get("reason"), "seconds": (r.get("cost") or {}).get("seconds"),
                          "usd": (r.get("cost") or {}).get("usd"), **_headline(r)}
                         for r in mine if r["kind"] in MEASURED_KINDS][::-1],
    }


def serve(port: int, ledger_root: Path, runs_dir: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlsplit(self.path)
            if u.path == "/":
                body, kind = PAGE.read_bytes(), "text/html; charset=utf-8"
            elif u.path == "/state":
                run = (parse_qs(u.query).get("run") or [None])[0]
                body, kind = json.dumps(state(run, ledger_root, runs_dir), default=str).encode(), "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"dashboard: http://127.0.0.1:{port}")
    httpd.serve_forever()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.dashboard")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ledger", type=Path, default=ledger.ROOT)
    ap.add_argument("--runs", type=Path, default=RUNS_DIR)
    a = ap.parse_args(argv)
    serve(a.port, a.ledger, a.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
