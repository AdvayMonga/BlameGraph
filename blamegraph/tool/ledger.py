"""Append-only ledger written at the source by the harness wrappers.

One JSON object per line in `ledger.jsonl`. Event kinds:
  config      {hash, path, size, diff_lines, content}       start_server.sh changed (content hashed after normalization)
  launch      {config_hash, pid}                             a server was started from that config
  measure     {id, config_hash, live_hash, mode, standard, flags, cache_state, metrics, failure_rate, quality, request_set}
  submission  {config_hash, n_measurements, last_measure_t}  recorded by the harness at session end
  grader      {hash}                                          hash of evaluate.py / runner at session start and end
Every event carries t (unix seconds), minute (since session start), tool_version.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

TOOL_VERSION = "0.1.0"


def normalize(script: str) -> str:
    out = []
    for ln in script.splitlines():
        ln = re.sub(r"(?<!\\)#.*$", "", ln).strip()
        if ln:
            out.append(re.sub(r"\s+", " ", ln))
    return "\n".join(out)


def config_hash(script: str) -> str:
    return hashlib.sha1(normalize(script).encode()).hexdigest()[:12]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


@dataclass
class Ledger:
    path: Path
    session_start: float | None = None
    events: list[dict] = field(default_factory=list)

    @classmethod
    def open(cls, path: str | os.PathLike) -> "Ledger":
        p = Path(path)
        led = cls(path=p)
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    led.events.append(json.loads(line))
            starts = [e["t"] for e in led.events if e.get("kind") == "session_start"]
            led.session_start = starts[0] if starts else (led.events[0]["t"] if led.events else None)
        return led

    # ---- writing ----
    def _append(self, kind: str, **data) -> dict:
        now = time.time()
        if self.session_start is None:
            self.session_start = now
        ev = {"kind": kind, "t": round(now, 3), "minute": round((now - self.session_start) / 60, 2), "tool_version": TOOL_VERSION, **data}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(ev) + "\n")
        self.events.append(ev)
        return ev

    def start_session(self, task: dict, grader_paths: list[Path]) -> dict:
        return self._append("session_start", task=task, grader={str(p): file_hash(p) for p in grader_paths})

    def record_config(self, path: Path) -> dict:
        content = path.read_text() if path.exists() else ""
        h = config_hash(content)
        prev = self.current_config_hash()
        if prev == h:
            return {"kind": "config", "hash": h, "unchanged": True}
        return self._append("config", hash=h, path=str(path), size=len(content), content=content)

    def record_launch(self, config_path: Path, pid: int | None = None) -> dict:
        self.record_config(config_path)
        return self._append("launch", config_hash=self.current_config_hash(), pid=pid)

    def record_measure(self, config_path: Path, mode: str, flags: list[str], metrics: dict | None, standard: bool,
                       request_set: str | None = None, error: str | None = None, server: dict | None = None) -> dict:
        """`server` is probe.live_server(...).to_dict(): used to detect stale servers and warm caches without launch hooks."""
        self.record_config(config_path)
        cur = self.current_config_hash()
        started = (server or {}).get("started_at")
        # a new server process since the last measurement counts as a launch of the current config
        last_started = next((e.get("server_started_at") for e in reversed(self.events) if e["kind"] == "measure"), None)
        if started and started != last_started:
            self._append("launch", config_hash=cur if not (server or {}).get("config_changed_after_start") else None,
                         pid=(server or {}).get("pid"), server_started_at=started, inferred=True)
        observed = bool((server or {}).get("pid"))
        if observed:
            live = self.live_config_hash()
            stale = bool(server.get("config_changed_after_start"))
            if stale and live == cur:
                live = None   # the file changed after the server started: we do not know what the server is running
        else:
            # no serving process found on the port: attribute the measurement to the current file, unverified
            live, stale = cur, False
        n_prior = sum(1 for e in self.events if e["kind"] == "measure" and e.get("server_started_at") == started and started)
        burst = ((metrics or {}).get("profiles") or {}).get("burst") or {}
        qc = (metrics or {}).get("quality_check") or {}
        return self._append(
            "measure", id=f"m{sum(1 for e in self.events if e['kind'] == 'measure') + 1}", config_hash=cur, live_hash=live,
            stale=stale, mode=mode, standard=standard, flags=flags, server_observed=observed, server_started_at=started, server_pid=(server or {}).get("pid"),
            cache_state="warm" if n_prior else "cold",
            metrics=metrics, failure_rate=burst.get("failure_rate"), quality_pass=qc.get("pass"), request_set=request_set, error=error,
        )

    def record_submission(self, config_path: Path, grader_paths: list[Path]) -> dict:
        self.record_config(config_path)
        h = self.current_config_hash()
        ms = [e for e in self.events if e["kind"] == "measure" and e.get("live_hash") == h and not e.get("error")]
        return self._append("submission", config_hash=h, n_measurements=len(ms), n_standard_full=sum(1 for e in ms if e["standard"] and e["mode"] == "full"),
                            last_measure_t=ms[-1]["t"] if ms else None, grader={str(p): file_hash(p) for p in grader_paths})

    # ---- reading ----
    def current_config_hash(self) -> str | None:
        for e in reversed(self.events):
            if e["kind"] == "config":
                return e["hash"]
        return None

    def live_config_hash(self) -> str | None:
        for e in reversed(self.events):
            if e["kind"] == "launch":
                return e["config_hash"]
        return None

    def launch_index(self) -> int:
        return sum(1 for e in self.events if e["kind"] == "launch")

    def configs(self) -> list[dict]:
        return [e for e in self.events if e["kind"] == "config"]

    def measurements(self) -> list[dict]:
        return [e for e in self.events if e["kind"] == "measure"]

    def submission(self) -> dict | None:
        subs = [e for e in self.events if e["kind"] == "submission"]
        return subs[-1] if subs else None

    def grader_hashes(self) -> tuple[dict | None, dict | None]:
        start = next((e["grader"] for e in self.events if e["kind"] == "session_start"), None)
        sub = self.submission()
        return start, (sub or {}).get("grader")

    def iter(self, kind: str) -> Iterator[dict]:
        return (e for e in self.events if e["kind"] == kind)
