"""Shape sets from captured calls: the distinct shapes a kernel saw, how often, and a split that holds out whole
regions of shape space (the largest values on the task's held-out axes) rather than random samples, so a kernel
tuned on the seen shapes is tested where it was never tuned."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from kernels.tasks import Task
from lab import capture


def from_captures(task: Task, dirs: list[Path]) -> Counter:
    """{shape: calls} over every capture dir, for calls to the task's entry point."""
    c: Counter = Counter()
    for d in dirs:
        for call in capture.load(d):
            if task.entry and call["fn"] != task.entry:
                continue
            s = task.shape_of(call)
            if s is not None:
                c[tuple(s)] += 1
    return c


def split(task: Task, counts: Counter, quantile: float = 0.8) -> dict:
    """Held out: a shape whose value on any held-out axis is above that axis's `quantile` (by calls). Seen: the rest.
    Each list is ordered by call count, most frequent first."""
    if not counts:
        return {"seen": list(task.seen_default), "heldout": list(task.heldout_default), "source": "defaults"}
    cut = {}
    for ax in task.heldout_axes:
        vals = sorted(v for s, n in counts.items() for v in [s[ax]] * n)
        cut[ax] = vals[min(len(vals) - 1, int(quantile * len(vals)))]
    held = {s for s in counts if any(s[ax] > cut[ax] for ax in task.heldout_axes)}
    order = sorted(counts, key=lambda s: -counts[s])
    seen = [s for s in order if s not in held]
    return {"seen": seen, "heldout": [s for s in order if s in held], "source": "captures",
            "cut": {str(k): v for k, v in cut.items()}, "calls": {json.dumps(list(s)): n for s, n in counts.items()}}


def write(path: Path, sets: dict) -> Path:
    Path(path).write_text(json.dumps({k: [list(s) for s in v] if k in ("seen", "heldout") else v
                                      for k, v in sets.items()}, indent=1))
    return Path(path)


def read(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    return {**d, "seen": [tuple(s) for s in d["seen"]], "heldout": [tuple(s) for s in d["heldout"]]}
