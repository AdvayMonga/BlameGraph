"""Observe the live server without hooks: which process serves the port, when it started, what config file it was
launched after. Stdlib only so it can run inside the harness container."""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class LiveServer:
    port: int
    pid: int | None
    started_at: float | None      # unix seconds
    command: str | None
    config_mtime: float | None
    config_changed_after_start: bool | None   # the "stale server" condition

    def to_dict(self) -> dict:
        return asdict(self)


def _pid_on_port(port: int) -> int | None:
    # try lsof, then ss, then /proc/net/tcp
    for cmd in (["lsof", "-t", "-iTCP:%d" % port, "-sTCP:LISTEN"], ["ss", "-ltnp", "sport = :%d" % port]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
        except Exception:
            continue
        m = re.search(r"pid=(\d+)", out) or re.match(r"\s*(\d+)", out)
        if m:
            return int(m.group(1))
    try:
        hex_port = "%04X" % port
        inode = None
        for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
            parts = line.split()
            if parts[1].endswith(":" + hex_port) and parts[3] == "0A":
                inode = parts[9]
                break
        if inode:
            for pid in filter(str.isdigit, os.listdir("/proc")):
                try:
                    for fd in os.listdir(f"/proc/{pid}/fd"):
                        if os.readlink(f"/proc/{pid}/fd/{fd}") == f"socket:[{inode}]":
                            return int(pid)
                except Exception:
                    continue
    except Exception:
        pass
    return None


def _proc_start(pid: int) -> tuple[float | None, str | None]:
    try:
        if Path(f"/proc/{pid}/stat").exists():
            stat = Path(f"/proc/{pid}/stat").read_text()
            starttime_ticks = int(stat.rsplit(")", 1)[1].split()[19])
            hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
            boot = next(float(l.split()[1]) for l in Path("/proc/stat").read_text().splitlines() if l.startswith("btime"))
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            return boot + starttime_ticks / hz, cmd
        out = subprocess.run(["ps", "-o", "lstart=,command=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
        if out:
            lstart, cmd = out[:24].strip(), out[24:].strip()
            return time.mktime(time.strptime(lstart, "%a %b %d %H:%M:%S %Y")), cmd
    except Exception:
        pass
    return None, None


def live_server(port: int, config_path: Path) -> LiveServer:
    pid = _pid_on_port(port)
    started, cmd = _proc_start(pid) if pid else (None, None)
    mtime = config_path.stat().st_mtime if config_path.exists() else None
    changed = (mtime is not None and started is not None and mtime > started + 1.0) if (mtime and started) else None
    return LiveServer(port=port, pid=pid, started_at=started, command=cmd, config_mtime=mtime, config_changed_after_start=changed)
