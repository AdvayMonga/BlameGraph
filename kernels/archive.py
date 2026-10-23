"""The idea archive (decided 2026-10-22): every distinct kernel idea with what became of it, built from the `kernels`
ledger records' hypothesis logs and traces. Measurements only (the no-beliefs rule, 2026-10-09): an idea's text is the
judge's, its outcomes are the evaluator's.

One entry per distinct idea: an idea whose embedding is within `DISTINCT` cosine of an earlier one on the same task is
folded into it, keeping every outcome (the user: every distinct idea, not every idea). It is read three ways: by
similarity to a new idea (the funnel's past-failure check; the judge's archive slice), as the best outcome among
similar ideas, and as a retry list (ideas that never beat their parent, tried only on parents that are no longer the
task's best: the context that blocked them has changed).

A search on the GPU host reads an exported file (`export`), since the ledger's history lives on the controller.

  python -m kernels.archive export --out FILE [--ledger ROOT]
  python -m kernels.archive query --archive FILE [--task T] [-k N] TEXT
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lab import ledger
from lab.search import Cache, Embedder, _cosine, bm25

DISTINCT = 0.92              # cosine above which two ideas on one task are one idea (bge-base: paraphrases ~0.9+)
FIELDS = ("bottleneck", "optimisation_method", "modification_plan")


@dataclass
class Outcome:
    run: str                 # the `kernels` record
    parent_id: str
    parent_score: float
    child_id: str | None     # None: never measured (not picked for the beam, or the edit never reached the GPU)
    child_score: float | None = None
    improved: bool | None = None
    stage: int | None = None
    error: str | None = None


@dataclass
class Entry:
    id: str
    task: str
    text: str
    idea: dict
    vector: list[float] | None = None
    outcomes: list[Outcome] = field(default_factory=list)

    def best(self) -> Outcome | None:
        scored = [o for o in self.outcomes if o.child_score is not None]
        return max(scored, key=lambda o: o.child_score) if scored else None

    def ever_improved(self) -> bool:
        return any(o.improved for o in self.outcomes)


def text(idea: dict) -> str:
    return "\n".join(str(idea[k]) for k in FIELDS if idea.get(k)) or json.dumps(idea, sort_keys=True)


def ideas(root: Path) -> list[tuple[str, dict, Outcome]]:
    """(task, idea, outcome) for every named change of every `kernels.search` run on the ledger, oldest first."""
    out = []
    for r in ledger.records(root, kind="kernels"):
        res, task = r.get("result") or {}, (r.get("config") or {}).get("task")
        if not res.get("hypotheses") or not task:
            continue
        children = _children(root / res["trace"]) if res.get("trace") else {}
        for line in (root / res["hypotheses"]).read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            got = {m["index"]: m for m in row.get("measured") or []}
            for i, h in enumerate(row.get("hypotheses") or []):
                m, idea = got.get(i), {k: v for k, v in h.items() if k in FIELDS} or h
                c = children.get(m["child_id"], {}) if m else {}
                out.append((task, idea, Outcome(r["id"], row.get("parent_id"), row.get("parent_score"),
                                                m["child_id"] if m else None, m.get("child_score") if m else None,
                                                m.get("improved") if m else None, c.get("stage"), c.get("error"))))
    return out


def _children(trace: Path) -> dict:
    """child id -> {stage, error} from a run's trace."""
    out = {}
    if not trace.exists():
        return out
    for line in trace.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        m, a = r.get("child_metrics") or {}, r.get("artifacts") or {}
        m = json.loads(m) if isinstance(m, str) else m
        a = json.loads(a) if isinstance(a, str) else a
        out[r.get("child_id")] = {"stage": m.get("stage"), "error": (a.get("error") or "")[:300] or None}
    return out


def build(root: Path, embedder: Embedder | None = None) -> list[Entry]:
    """Distinct ideas with all their outcomes. Without an embedder, only identical texts are folded."""
    rows = ideas(root)
    vecs = Cache(root, embedder.name).vectors([text(i) for _, i, _ in rows], embedder) if embedder else [None] * len(rows)
    entries: list[Entry] = []
    for (task, idea, outcome), v in zip(rows, vecs):
        t = text(idea)
        same = next((e for e in entries if e.task == task and (e.text == t or (
            v is not None and e.vector is not None and _cosine(v, e.vector) >= DISTINCT))), None)
        if same is None:
            same = Entry(f"idea-{len(entries) + 1}", task, t, idea, v)
            entries.append(same)
        same.outcomes.append(outcome)
    return entries


def export(entries: list[Entry], path: Path) -> Path:
    Path(path).write_text("".join(json.dumps(asdict(e)) + "\n" for e in entries))
    return Path(path)


def load(path: Path) -> list[Entry]:
    out = []
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        out.append(Entry(**{**d, "outcomes": [Outcome(**o) for o in d["outcomes"]]}))
    return out


def matches(entries: list[Entry], query: str, task: str | None = None, k: int = 5,
            embedder: Embedder | None = None) -> list[tuple[float, Entry]]:
    """The `k` entries most like `query` (cosine of embeddings, else BM25), on `task` when given."""
    pool = [e for e in entries if task is None or e.task == task]
    if not pool or not query.strip():
        return []
    if embedder is not None and all(e.vector is not None for e in pool):
        q = embedder.query(query)
        scored = [(_cosine(q, e.vector), e) for e in pool]
    else:
        scored = list(zip(bm25(query, [e.text for e in pool]), pool))
    return sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])[:k]


def retry(entries: list[Entry], best_parent: dict[str, str]) -> list[Entry]:
    """Ideas that never beat their parent and were tried only on parents other than each task's current best."""
    return [e for e in entries if e.task in best_parent and e.outcomes and not e.ever_improved()
            and all(o.parent_id != best_parent[e.task] for o in e.outcomes)]


def facts(entry: Entry, score: float | None = None) -> dict:
    """An entry as the judge sees it: the idea and each outcome, measured numbers only."""
    return {"idea": entry.idea, **({"similarity": round(score, 3)} if score is not None else {}),
            "outcomes": [{k: v for k, v in asdict(o).items() if v is not None} for o in entry.outcomes]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernels.archive")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--out", required=True)
    ex.add_argument("--ledger", default=str(ledger.ROOT))
    q = sub.add_parser("query")
    q.add_argument("--archive", required=True)
    q.add_argument("--task")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("text")
    a = ap.parse_args(argv)
    embedder = _embedder(Path(getattr(a, "ledger", ledger.ROOT)))
    if a.cmd == "export":
        entries = build(Path(a.ledger), embedder)
        export(entries, Path(a.out))
        print(json.dumps({"entries": len(entries), "outcomes": sum(len(e.outcomes) for e in entries),
                          "embedder": embedder.name if embedder else None, "out": a.out}))
        return 0
    for s, e in matches(load(Path(a.archive)), a.text, a.task, a.k, embedder):
        print(json.dumps(facts(e, s)))
    return 0


def _embedder(root: Path) -> Embedder | None:
    try:
        from lab.search import FastEmbed
        return FastEmbed(root / "index" / "models")
    except Exception:                            # noqa: BLE001  without fastembed: identical texts only, BM25 lookup
        return None


if __name__ == "__main__":
    sys.exit(main())
