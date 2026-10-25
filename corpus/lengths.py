"""Each corpus record's `expected_output_tokens`: the reference's own answer length (greedy, the record's messages and
max_tokens, the target's chat kwargs), measured once on a served reference and written into every trace of both
splits. Regimes then send each record at that length with ignore_eos (`regimes.workload.to_request`), so a base and
a candidate generate the same tokens' worth of work. Every trace changes, so the manifest is rebuilt: a new corpus
version. Held-out traces must be present (`python -m store pull --corpus`) and pushed again afterwards.

    python -m corpus.lengths --url URL [--target T] [--concurrency 64]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lab.corpus import CORPUS_DIR, SPLITS, Manifest, build_manifest, load_manifest, read_trace, write_trace

NOTE = " expected_output_tokens: the reference's answer length per record (corpus/lengths.py)."


def _files(m: Manifest) -> list[str]:
    return [c.trace_file(s) for c in m.classes.values() for s in SPLITS]


def read(corpus_dir: Path = CORPUS_DIR) -> dict[str, list[int | None]]:
    """Each trace's lengths in line order (None where unmeasured)."""
    return {rel: [r.expected_output_tokens for r in read_trace(corpus_dir / rel)]
            for rel in _files(load_manifest(corpus_dir))}


def apply(corpus_dir: Path, lengths: dict[str, list[int]]) -> Manifest:
    """Write the lengths into every trace and rebuild the manifest."""
    m = load_manifest(corpus_dir)
    traces = {rel: read_trace(corpus_dir / rel) for rel in _files(m)}
    for rel, reqs in traces.items():
        if len(lengths.get(rel, [])) != len(reqs):
            raise ValueError(f"{rel}: {len(lengths.get(rel, []))} lengths for {len(reqs)} records")
    for rel, reqs in traces.items():
        for r, n in zip(reqs, lengths[rel]):
            r.expected_output_tokens = n
        write_trace(corpus_dir / rel, reqs)
    notes = m.notes if m.notes.endswith(NOTE) else m.notes + NOTE
    new = build_manifest(m.classes, corpus_dir, notes=notes, dilation=m.dilation)
    (corpus_dir / "manifest.json").write_text(json.dumps(new.to_dict(), indent=2) + "\n")
    return new


def measure(url: str, model: str, chat_kwargs: dict, corpus_dir: Path = CORPUS_DIR, concurrency: int = 64,
            attempts: int = 3) -> dict[str, list[int]]:
    """The served reference's completion tokens for every record of every trace, in line order."""
    from regimes.client import Request, send
    rels = _files(load_manifest(corpus_dir))
    jobs = [(rel, k, r) for rel in rels for k, r in enumerate(read_trace(corpus_dir / rel))]

    def one(job) -> int:
        rel, k, r = job
        for _ in range(attempts):
            row = send(url, model, Request(f"len-{rel}-{k}", r.chat_messages(), r.max_tokens), time.monotonic(),
                       time.monotonic, chat_kwargs=chat_kwargs)
            if row.status == "ok":
                return row.n_tokens
        raise RuntimeError(f"{rel} line {k}: {row.status} {row.detail}")

    with ThreadPoolExecutor(concurrency) as ex:
        ns = list(ex.map(one, jobs))
    out: dict[str, list[int]] = {rel: [] for rel in rels}
    for (rel, _, _), n in zip(jobs, ns):
        out[rel].append(n)
    return out


if __name__ == "__main__":
    from lab import target
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="the target's reference, served")
    ap.add_argument("--target")
    ap.add_argument("--corpus", default=str(CORPUS_DIR))
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    t = target.load(a.target)
    m = apply(Path(a.corpus), measure(a.url, t.model, t.chat_kwargs, Path(a.corpus), a.concurrency))
    print(f"corpus_version {m.corpus_version}")
