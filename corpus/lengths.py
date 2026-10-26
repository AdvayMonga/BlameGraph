"""Each corpus record's `expected_output_tokens`: the reference's own answer length (greedy, the record's messages and
max_tokens, the target's chat kwargs), and for a multi-turn record `expected_turn_output_tokens`, the same for each
user-turn prefix of its messages (the shared-prefix regime's turns). Measured once on a served reference and written
into every trace of both splits; regimes then send each request at that length with ignore_eos
(`regimes.workload.to_request`), so a base and a candidate generate the same tokens' worth of work. Every trace
changes, so the manifest is rebuilt: a new corpus version, whose `prompts_version` keeps the held-out budget's key.
Held-out traces must be present (`python -m store pull --corpus`) and pushed again afterwards. Each answer is
appended to a log as it arrives (default `data/corpus-lengths-<version>.jsonl`), so a rerun resumes.

    python -m corpus.lengths --url URL [--target T] [--concurrency 64]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lab.corpus import (
    CORPUS_DIR,
    REPO_ROOT,
    SPLITS,
    CorpusError,
    Manifest,
    TraceRequest,
    build_manifest,
    load_manifest,
    read_trace,
    write_trace,
)

NOTE = " expected_output_tokens: the reference's answer length per record (corpus/lengths.py)."
# per trace, per record: (answer length, per-turn lengths for a multi-turn record else None)
Lengths = dict[str, list[tuple[int | None, list[int] | None]]]


def _files(m: Manifest) -> list[str]:
    return [c.trace_file(s) for c in m.classes.values() for s in SPLITS]


def _turns(r: TraceRequest) -> list[list[dict]]:
    """A multi-turn record's user-turn prefixes before its last (the last is the record itself)."""
    msgs = r.messages or []
    users = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
    return [msgs[:i + 1] for i in users[:-1]] if len(users) >= 2 else []


def _traces(corpus_dir: Path, m: Manifest) -> dict[str, list[TraceRequest]]:
    for rel in _files(m):
        if not (corpus_dir / rel).exists():
            raise CorpusError(f"trace {rel} is missing; held-out traces are not in git: `python -m store pull --corpus`")
    return {rel: read_trace(corpus_dir / rel) for rel in _files(m)}


def read(corpus_dir: Path = CORPUS_DIR) -> Lengths:
    """Each trace's lengths in line order (None where unmeasured)."""
    return {rel: [(r.expected_output_tokens, r.expected_turn_output_tokens) for r in reqs]
            for rel, reqs in _traces(corpus_dir, load_manifest(corpus_dir)).items()}


def apply(corpus_dir: Path, lengths: Lengths) -> Manifest:
    """Write the lengths into every trace (each through a temp file) and rebuild the manifest."""
    m = load_manifest(corpus_dir)
    traces = _traces(corpus_dir, m)
    for rel, reqs in traces.items():
        if len(lengths.get(rel, [])) != len(reqs):
            raise ValueError(f"{rel}: {len(lengths.get(rel, []))} lengths for {len(reqs)} records")
    tmp = {}
    for rel, reqs in traces.items():
        for r, (n, turns) in zip(reqs, lengths[rel]):
            r.expected_output_tokens, r.expected_turn_output_tokens = n, turns
        write_trace(tmp.setdefault(rel, corpus_dir / f"{rel}.tmp"), reqs)
    for rel, p in tmp.items():
        os.replace(p, corpus_dir / rel)
    notes = m.notes if m.notes.endswith(NOTE) else m.notes + NOTE
    new = build_manifest(m.classes, corpus_dir, notes=notes, dilation=m.dilation)
    new.prompts_version = m.budget_key
    mp = corpus_dir / "manifest.json"
    mp.with_suffix(".json.tmp").write_text(json.dumps(new.to_dict(), indent=2) + "\n")
    os.replace(mp.with_suffix(".json.tmp"), mp)
    return new


def measure(url: str, model: str, chat_kwargs: dict, corpus_dir: Path = CORPUS_DIR, concurrency: int = 64,
            attempts: int = 3, log: Path | None = None) -> Lengths:
    """The served reference's completion tokens for every record and turn prefix, resuming from `log`. Raises after
    every request was tried when any failed `attempts` times (with backoff); the answers so far stay in the log."""
    from regimes.client import Request, send
    m = load_manifest(corpus_dir)
    traces = _traces(corpus_dir, m)
    log = log or REPO_ROOT / "data" / f"corpus-lengths-{m.corpus_version[:12]}.jsonl"
    done: dict[tuple[str, int, int | None], int] = {}
    if log.exists():
        for line in log.read_text().splitlines():
            d = json.loads(line)
            done[(d["rel"], d["line"], d["turn"])] = d["n"]
    jobs = [(rel, k, j, msgs, r.max_tokens) for rel, reqs in traces.items() for k, r in enumerate(reqs)
            for j, msgs in [(None, r.chat_messages()), *enumerate(_turns(r))]]

    def one(job) -> int:
        rel, k, j, msgs, max_tokens = job
        for a in range(attempts):
            if a:
                time.sleep(2 ** a)
            row = send(url, model, Request(f"len-{rel}-{k}-{j}", msgs, max_tokens), time.monotonic(), time.monotonic,
                       chat_kwargs=chat_kwargs)
            if row.status == "ok":
                return row.n_tokens
        raise RuntimeError(f"{rel} line {k} turn {j}: {row.status} {row.detail}")

    failed = []
    log.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(concurrency) as ex, open(log, "a") as f:
        futs = {ex.submit(one, job): job[:3] for job in jobs if job[:3] not in done}
        for fut in as_completed(futs):
            try:
                done[futs[fut]] = n = fut.result()
            except RuntimeError as e:
                failed.append(str(e))
                continue
            rel, k, j = futs[fut]
            f.write(json.dumps({"rel": rel, "line": k, "turn": j, "n": n}) + "\n")
            f.flush()
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(jobs)} requests failed; the rest are kept in {log} and a rerun "
                           f"resumes: " + "; ".join(failed[:5]))
    return {rel: [(done[(rel, k, None)], [done[(rel, k, j)] for j in range(len(_turns(r)))] + [done[(rel, k, None)]]
                   if _turns(r) else None) for k, r in enumerate(reqs)] for rel, reqs in traces.items()}


if __name__ == "__main__":
    from lab import target
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="the target's reference, served")
    ap.add_argument("--target")
    ap.add_argument("--corpus", default=str(CORPUS_DIR))
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--log", type=Path)
    a = ap.parse_args()
    t = target.load(a.target)
    m = apply(Path(a.corpus), measure(a.url, t.model, t.chat_kwargs, Path(a.corpus), a.concurrency, log=a.log))
    print(f"corpus_version {m.corpus_version} (held-out budget key {m.budget_key})")
