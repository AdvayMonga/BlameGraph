"""Search over every run's experiments: each `hypothesis` record with its rationale, and each measurement made under
it with its `expect`, measured value, gap and `reflection`. One document per hypothesis, its fields verbatim.

Ranking is reciprocal-rank fusion of two orders: BM25 over the words, and cosine similarity of embeddings from a
local model (fastembed, ONNX on CPU; no key, nothing leaves the machine). Embeddings are cached under
<ledger>/index/embeddings/ by model and content hash, so a document is embedded once, and again only when a new
measurement or reflection changes it. Researcher runs are left out unless the caller is one (lab/experiments.py)."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Protocol

from lab import experiments, ledger

MODEL = "BAAI/bge-base-en-v1.5"
RRF_K = 60                      # the usual constant (Cormack et al. 2009): rank r contributes 1 / (60 + r)
BM25_K1, BM25_B = 1.2, 0.75
MAX_K = 50
_WORD = re.compile(r"[a-z0-9_.]+")
# English function words: BM25 ignores them, else "why are tokens slow" ranks a document by its "are".
STOPWORDS = frozenset("""a an and are as at be been being but by can could did do does doing for from had has have
having he her here hers him his how i if in into is it its itself just me more most my no nor not of off on once only
or other our ours out over own same she should so some such than that the their theirs them then there these they
this those through to too under until up very was we were what when where which while who whom why will with would
you your yours""".split())


class Embedder(Protocol):
    name: str

    def passages(self, texts: list[str]) -> list[list[float]]: ...

    def query(self, text: str) -> list[float]: ...


class FastEmbed:
    """`MODEL` through fastembed; the weights download once into <ledger>/index/models."""

    def __init__(self, cache: Path, model: str = MODEL):
        from fastembed import TextEmbedding
        import fastembed
        self.name = f"{model}@fastembed-{fastembed.__version__}"
        self._m = TextEmbedding(model, cache_dir=str(cache))

    def passages(self, texts):
        return [v.tolist() for v in self._m.passage_embed(texts)]

    def query(self, text):
        return next(iter(self._m.query_embed([text]))).tolist()


# -- documents ------------------------------------------------------------------------------
def documents(root: Path, researcher: bool = False) -> list[dict]:
    """Every hypothesis on the ledger with its measurements and reflections, oldest first."""
    recs = [r for r in ledger.records(root) if researcher or not experiments.researcher(r.get("run"))]
    refl = {r["record"]: r for r in recs if r["kind"] == "reflection"}
    docs = {r["id"]: {"hypothesis_id": r["id"], "run": r.get("run"), "session": r.get("session"), "at": r.get("at"),
                      **{k: r[k] for k in ("hypothesis", "rationale", "abandon_if", "from_record") if k in r},
                      "measurements": []}
            for r in recs if r["kind"] == "hypothesis"}
    for r in recs:
        x = r.get("experiment")
        if not isinstance(x, dict) or x.get("hypothesis") not in docs:
            continue
        res = r.get("result") if isinstance(r.get("result"), dict) else {}
        m = {"record": r["id"], "tool": r.get("tool"), "at": r.get("at"), "snapshot": r.get("snapshot"),
             "config": r.get("config"), "gpu": (res.get("gpu") or {}).get("name") if isinstance(res.get("gpu"), dict)
             else res.get("gpu"), **{k: x[k] for k in ("expect", "actual", "actual_missing", "gap",
                                                        "gap_pct_of_expected", "noise_band_pct", "mechanism") if k in x}}
        if r["id"] in refl:
            m["reflection"] = refl[r["id"]]["reflection"]
            if refl[r["id"]].get("alternative"):
                m["alternative"] = refl[r["id"]]["alternative"]
        docs[x["hypothesis"]]["measurements"].append(m)
    return list(docs.values())


def text(doc: dict) -> str:
    """What is indexed: the person- or agent-written text and the prediction, with the measured numbers."""
    parts = [doc.get("hypothesis", ""), doc.get("rationale", ""), doc.get("abandon_if", "")]
    for m in doc["measurements"]:
        e = m.get("expect") or {}
        parts += [f"{m.get('tool')} {e.get('metric')} expected {e.get('value')} {e.get('unit')} measured "
                  f"{m.get('actual')}", e.get("why", ""), m.get("reflection", ""), m.get("alternative", "")]
        parts += [f"{p.get('metric')} expected {p.get('value')} {p.get('unit')} measured {p.get('actual')}\n"
                  f"{p.get('why', '')}" for p in m.get("mechanism") or []]
    return "\n".join(p for p in parts if p)


# -- ranking --------------------------------------------------------------------------------
def words(s: str) -> list[str]:
    return [w for w in (t.strip(".") for t in _WORD.findall(s.lower())) if w and w not in STOPWORDS]


def bm25(query: str, texts: list[str]) -> list[float]:
    toks = [words(t) for t in texts]
    n, avg = len(toks), (sum(map(len, toks)) / len(toks) if toks else 0.0)
    df = Counter(w for t in toks for w in set(t))
    out = []
    for t in toks:
        tf, s = Counter(t), 0.0
        for w in set(words(query)):
            if tf[w]:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (BM25_K1 + 1) / (tf[w] + BM25_K1 * (1 - BM25_B + BM25_B * len(t) / (avg or 1)))
        out.append(s)
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def _ranks(scores: list[float], keep: Callable[[float], bool]) -> dict[int, int]:
    """Doc index -> 1-based rank by score, over the docs `keep` admits."""
    order = sorted((i for i, s in enumerate(scores) if keep(s)), key=lambda i: -scores[i])
    return {i: r + 1 for r, i in enumerate(order)}


class Cache:
    """Content hash -> vector, one JSON line each, per embedder."""

    def __init__(self, root: Path, name: str):
        self.path = Path(root) / "index" / "embeddings" / (re.sub(r"[^A-Za-z0-9.-]+", "_", name) + ".jsonl")
        self.v: dict[str, list[float]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    d = json.loads(line)
                except ValueError:                  # a torn last line: re-embedded on demand
                    continue
                self.v[d["h"]] = d["v"]

    def vectors(self, texts: list[str], embedder: Embedder) -> list[list[float]]:
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self.v]
        if todo:
            new = embedder.passages([t for _, t in todo])
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                for (k, _), v in zip(todo, new):
                    self.v[k] = v
                    f.write(json.dumps({"h": k, "v": v}) + "\n")
        return [self.v[k] for k in keys]


def search(root: Path, query: str, k: int = 10, embedder: Embedder | None = None, researcher: bool = False) -> dict:
    """The `k` best documents for `query`, each with its rank under BM25 and under the embeddings (None when that
    order did not place it: no shared word, or no embeddings)."""
    k = max(1, min(int(k), MAX_K))
    docs = documents(root, researcher)
    out = {"query": query, "documents": len(docs), "ranking": f"reciprocal-rank fusion (k={RRF_K}) of BM25 and "
           + (f"cosine similarity of {embedder.name} embeddings" if embedder else "nothing else"), "results": []}
    if not docs or not query.strip():
        return out
    texts = [text(d) for d in docs]
    lex = _ranks(bm25(query, texts), lambda s: s > 0)
    sem: dict[int, int] = {}
    if embedder is not None:
        vecs = Cache(root, embedder.name).vectors(texts, embedder)
        q = embedder.query(query)
        sem = _ranks([_cosine(q, v) for v in vecs], lambda s: True)
    fused = {i: sum(1 / (RRF_K + r[i]) for r in (lex, sem) if i in r) for i in set(lex) | set(sem)}
    for i in sorted(fused, key=lambda i: (-fused[i], i))[:k]:
        out["results"].append({"bm25_rank": lex.get(i), "embedding_rank": sem.get(i), **docs[i]})
    return out
