"""What a run keeps: `record` after the calibration and after every turn, `finish` once at its end.

  runs/loop/<run>/<n>.json   the run's status after its n-th record, in the store
  runs/loop/<run>/report.md  the report `finish` writes
"""
from __future__ import annotations

import json
import re
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from lab.evaltools import BENCH_TIER
from lab.loop.types import Calibration, Measurement
from lab.records import ledger, sync
from lab.records.remote import Remote

PREFIX = "runs/loop"
STATUS = re.compile(r"/(\d+)\.json$")


def _put(remote: Remote, key: str, text: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "f").write_text(text)
        remote.put(key, Path(d) / "f")


def status(remote: Remote, run: str) -> dict | None:
    """The run's last saved status; None when it has none."""
    n = max((int(m.group(1)) for k in remote.keys(f"{PREFIX}/{run}/") if (m := STATUS.search(k))), default=None)
    if n is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        remote.get(f"{PREFIX}/{run}/{n:05d}.json", Path(d) / "s.json")
        return json.loads((Path(d) / "s.json").read_text())


def last_run(remote: Remote) -> str | None:
    """The newest run in the store (run ids sort by their start time)."""
    return max({k.split("/")[2] for k in remote.keys(f"{PREFIX}/")}, default=None)


def measurement(m: Measurement) -> dict:
    return {k: v for k, v in asdict(m).items() if k != "seconds"}


def baseline(cal: Calibration, run: str, usd: float, root: Path) -> dict:
    """The calibration's baseline as the run's `baseline` record."""
    m = cal.baseline
    return ledger.append({"kind": "baseline", "run": run, "config": {"split": "seen", "tier": BENCH_TIER,
                                                                      "commit": cal.commit},
                          "result": {"verdict": "ok", **measurement(m), "bands_pct": cal.bands,
                                     "peaks": asdict(cal.peaks)},
                          "cost": {"usd": usd, "seconds": m.seconds}}, root)


def turn(run: str, n: int, result: dict, usd: float, seconds: float, root: Path) -> dict:
    """Turn `n` as a `turn` record: its gap, candidate, measurement, score, win and pull request, or its error."""
    return ledger.append({"kind": "turn", "run": run, "config": {"split": "seen", "tier": BENCH_TIER, "turn": n},
                          "patch": result.get("patch"), "result": {k: v for k, v in result.items() if k != "patch"},
                          "cost": {"usd": usd, "seconds": seconds}}, root)


def record(remote: Remote, run: str, state: dict, root: Path = ledger.ROOT) -> None:
    """The ledger's new lines and the run's next status into the store."""
    sync.push_ledger(root, remote)
    n = len([k for k in remote.keys(f"{PREFIX}/{run}/") if STATUS.search(k)])
    _put(remote, f"{PREFIX}/{run}/{n:05d}.json", json.dumps({**state, "run": run, "updated": time.time()}))


def report(run: str, records: list[dict]) -> str:
    """The run in text: the baseline, then each turn's gap, outcome and pull request."""
    lines = [f"# {run}", ""]
    for r in records:
        res = r["result"]
        if r["kind"] == "baseline":
            lines += [f"baseline {r['config']['commit'][:12]}: " + ", ".join(
                f"{n} {h['value']:g} (band {res['bands_pct'][n]:.1f}%)" for n, h in res["metrics"].items()), ""]
            continue
        head = f"turn {r['config']['turn']} [{(res.get('gap') or {}).get('layer', '-')}]"
        if res.get("error"):
            lines.append(f"{head}: failed: {res['error']}")
        elif not res.get("candidate"):
            lines.append(f"{head}: no candidate")
        elif not res["correct"]:
            lines.append(f"{head}: {res['candidate']}: incorrect: {'; '.join(res['equiv'].get('reasons') or [])}")
        else:
            gains = ", ".join(f"{n} {g:+.1f}%" for n, g in res["score"]["gains_pct"].items() if g is not None)
            lines.append(f"{head}: {res['candidate']}: {gains}: "
                         + (f"win, {res['pr']['url']}" if res["win"] else "no win"))
    return "\n".join(lines) + "\n"


def finish(remote: Remote, run: str, state: dict, records: list[dict], stop: Callable[[], None],
           root: Path = ledger.ROOT) -> None:
    """The report and the whole ledger into the store with the final status, then `stop` (this VM stops itself);
    `stop` runs even when an upload fails."""
    try:
        _put(remote, f"{PREFIX}/{run}/report.md", report(run, records))
        sync.push(root, remote)
        record(remote, run, state, root)
    except Exception:                            # noqa: BLE001
        traceback.print_exc()
    finally:
        stop()
