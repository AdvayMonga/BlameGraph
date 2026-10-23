"""Progress lines for the researcher's dashboard (`lab/dashboard.py`): one JSON line per event appended to `path`, the
running run's `<run_dir>/live.jsonl`. Nothing in the run reads them; with no run going, `emit` does nothing."""

from __future__ import annotations

import json
import time
from pathlib import Path

path: Path | None = None            # set by lab.session.run for its run


def emit(event: str, **fields) -> None:
    if path is None:
        return
    try:
        with open(path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **fields}, default=str) + "\n")
    except OSError:                 # a dashboard is never worth failing a run
        pass
