"""Shape sets from captured calls: the distinct shapes a kernel saw, how often, and a split that holds out whole
regions of shape space (the largest values on the task's held-out axes) rather than random samples, so a kernel
tuned on the seen shapes is tested where it was never tuned."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from kernels.tasks import Task
from lab import capture


def _ours(task: Task, call: dict) -> bool:
    """A captured call to the task's entry point (or its `capture_fn`: a region's calls)."""
    fn = task.capture_fn or task.entry
    return not fn or call["fn"] == fn


def from_captures(task: Task, dirs: list[Path], equal: bool = False) -> Counter:
    """{shape: calls} over every capture dir, for calls to the task's entry point. `equal`: each dir's calls are
    scaled to the same total, so each capture's traffic (a scored regime) weighs the same (decided 2026-10-23)."""
    per = []
    for d in dirs:
        c: Counter = Counter()
        for call in capture.load(d):
            if not _ours(task, call):
                continue
            s = task.shape_of(call)
            if s is not None:
                c[tuple(s)] += 1
        per.append(c)
    top = max((sum(c.values()) for c in per), default=0)
    out: Counter = Counter()
    for c in per:
        total = sum(c.values())
        for s, n in c.items():
            out[s] += max(1, round(n * top / total)) if equal and total else n
    return out


def source(d: Path) -> str:
    """The traffic a capture dir was recorded under: `profile-<class>/<bundle>/captures` -> <class>."""
    for q in (Path(d), *Path(d).parents):
        if q.name.startswith("profile-"):
            return q.name.removeprefix("profile-")
    return Path(d).name


def sources_from_captures(task: Task, dirs: list[Path]) -> dict:
    """{json shape: {source: calls}}: how many of each shape's calls each capture's traffic produced."""
    out: dict = {}
    for d in dirs:
        for call in capture.load(d):
            if not _ours(task, call):
                continue
            s = task.shape_of(call)
            if s is not None:
                by = out.setdefault(json.dumps(list(s)), {})
                by[source(d)] = by.get(source(d), 0) + 1
    return out


def ints_from_captures(task: Task, dirs: list[Path]) -> dict:
    """{json shape: the 1-D int args by value} from the first captured call of each shape (e.g. its real sequence
    lengths), for `Task.with_ints`. Calls logged by shape only (inside a graph recording) carry none."""
    out: dict = {}
    for d in dirs:
        for call in capture.load(d):
            if not _ours(task, call):
                continue
            s = task.shape_of(call)
            vals = [a["values"] for a in call.get("args") or () if "values" in a]   # region calls have none
            if s is not None and vals and json.dumps(list(s)) not in out:
                out[json.dumps(list(s))] = vals
    return out


def split(task: Task, counts: Counter, quantile: float = 0.8) -> dict:
    """Held out: a shape whose value on any held-out axis is above that axis's `quantile` (by calls; by distinct values
    when that holds out nothing, i.e. most calls sit at the largest value). By distinct values the most-called shape
    always stays seen, and axes are added smallest region first while at most half of all calls are held out (on
    2026-10-19 the union of two such regions held out 83% of calls, the most-called shapes among them).
    Seen: the rest. Each list is ordered by call count, most frequent first."""
    if not counts:
        return {"seen": list(task.seen_default), "heldout": list(task.heldout_default), "source": "defaults"}
    cut = {}
    for ax in task.heldout_axes:
        vals = sorted(v for s, n in counts.items() for v in [s[ax]] * n)
        cut[ax] = vals[min(len(vals) - 1, int(quantile * len(vals)))]
    held = {s for s in counts if any(s[ax] > cut[ax] for ax in task.heldout_axes)}
    by = "calls"
    if not held:                                 # calls pile up at the largest value: cut over distinct values instead
        by, top, total, cut = "distinct", max(counts, key=lambda s: counts[s]), sum(counts.values()), {}
        tails = {}
        for ax in task.heldout_axes:
            vals = sorted({s[ax] for s in counts})
            c = vals[min(len(vals) - 1, int(quantile * len(vals)))]
            tails[ax] = (c, {s for s in counts if s[ax] > c} - {top})
        for ax, (c, region) in sorted(tails.items(), key=lambda kv: sum(counts[s] for s in kv[1][1])):
            if region and sum(counts[s] for s in held | region) <= total / 2:
                held, cut[ax] = held | region, c
    order = sorted(counts, key=lambda s: -counts[s])
    seen = [s for s in order if s not in held]
    return {"seen": seen, "heldout": [s for s in order if s in held], "source": "captures",
            "cut": {str(k): v for k, v in cut.items()}, "cut_by": by, "calls": {json.dumps(list(s)): n for s, n in counts.items()}}


def select(task: Task, sets: dict, k: int = 8, bins: int = 4) -> dict:
    """At most `k` seen shapes for selection, chosen by calls: shapes are bucketed by the other axes plus a quantile bin
    of each held-out axis, the most-called shape stands for its bucket, and it carries the bucket's calls as its weight.
    The held-out set is untouched; `gate` still judges every held-out shape."""
    seen, calls = sets["seen"], sets.get("calls") or {}
    if len(seen) <= k:
        return sets
    n = {s: calls.get(json.dumps(list(s)), 1) for s in seen}
    edges = {}
    for ax in task.heldout_axes:
        vals = sorted(s[ax] for s in seen)
        edges[ax] = [vals[min(len(vals) - 1, int(q * len(vals) / bins))] for q in range(1, bins)]

    def key(s):
        rest = tuple(v for i, v in enumerate(s) if i not in edges)
        return rest + tuple(sum(s[ax] > e for e in edges[ax]) for ax in sorted(edges))
    buckets: dict = {}
    for s in seen:
        buckets.setdefault(key(s), []).append(s)
    reps = sorted(((max(b, key=lambda s: n[s]), sum(n[s] for s in b), b) for b in buckets.values()),
                  key=lambda x: -x[1])[:k]
    out = {**sets, "seen": [s for s, _, _ in reps], "calls": {json.dumps(list(s)): w for s, w, _ in reps},
           "selected_from": len(seen)}
    if "sources" in sets:                        # a representative carries its bucket's sources, like its calls
        out["sources"] = {}
        for s, _, b in reps:
            agg = out["sources"].setdefault(json.dumps(list(s)), {})
            for m in b:
                for c, v in (sets["sources"].get(json.dumps(list(m))) or {}).items():
                    agg[c] = agg.get(c, 0) + v
    if "ints" in sets:                           # the held-out shapes' ints stay: `gate` uses them
        keep = {json.dumps(list(s)) for s, _, _ in reps} | {json.dumps(list(s)) for s in sets["heldout"]}
        out["ints"] = {k: v for k, v in sets["ints"].items() if k in keep}
    return out


def write(path: Path, sets: dict) -> Path:
    Path(path).write_text(json.dumps({k: [list(s) for s in v] if k in ("seen", "heldout") else v
                                      for k, v in sets.items()}, indent=1))
    return Path(path)


def read(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    return {**d, "seen": [tuple(s) for s in d["seen"]], "heldout": [tuple(s) for s in d["heldout"]]}
