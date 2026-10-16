"""`search`: one document per hypothesis with its measurements and reflections, ranked by BM25 fused with
embeddings; researcher runs hidden from the agent; embeddings cached by content. A fake embedder stands in for the
model (one test runs the real one when fastembed and its weights are present)."""
from __future__ import annotations

import json
import os

import pytest

from lab import ledger, search

# Words that mean the same thing share a dimension: what an embedding model gives and BM25 cannot.
CONCEPTS = [{"latency", "tpot", "slow", "slower"}, {"attention", "decode"}, {"memory", "bandwidth", "bytes"},
            {"scheduler", "admission", "batch"}]


class FakeEmbed:
    name = "fake-concepts"

    def __init__(self):
        self.embedded = 0

    def _v(self, s):
        ws = set(search.words(s))
        return [float(len(ws & c)) for c in CONCEPTS] + [0.01]

    def passages(self, texts):
        self.embedded += len(texts)
        return [self._v(t) for t in texts]

    def query(self, text):
        return self._v(text)


def hyp(root, run, text, rationale="r"):
    return ledger.append({"kind": "hypothesis", "run": run, "session": f"{run}-s1", "hypothesis": text,
                          "rationale": rationale}, root)


def measure(root, run, h, actual, metric="single_stream"):
    return ledger.append({"kind": "bench", "run": run, "session": f"{run}-s1", "tool": "bench", "snapshot": "abc",
                          "config": {"split": "seen", "tier": "short"}, "result": {"gpu": {"name": "NVIDIA H200"}},
                          "experiment": {"hypothesis": h["id"], "actual": actual, "gap": actual - 0.02,
                                         "expect": {"metric": metric, "value": 0.02, "unit": "absolute",
                                                    "why": "fewer bytes per step"}}}, root)


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "ledger"
    h1 = hyp(r, "run-a", "the scheduler admits too few requests per batch", "queue depth 40 at 2 req/s")
    m = measure(r, "run-a", h1, 0.031)
    ledger.append({"kind": "reflection", "run": "run-a", "session": "run-a-s1", "record": m["id"],
                   "reflection": "admission was not the limit", "alternative": "prefill blocks decode"}, r)
    hyp(r, "run-b", "decode attention reads each KV tile once per query head", "41% of bandwidth")
    hyp(r, "run-b", "tpot is dominated by the sampler", "sampler 30% of step")
    hyp(r, "researcher-x", "a person's guess about tpot")
    return r


def test_documents_hold_each_hypothesis_with_its_measurements_and_reflections(root):
    docs = search.documents(root)
    assert [d["hypothesis"][:13] for d in docs] == ["the scheduler", "decode attent", "tpot is domin"]
    m = docs[0]["measurements"][0]
    assert m["actual"] == 0.031 and m["gpu"] == "NVIDIA H200" and m["config"]["tier"] == "short"
    assert m["reflection"] == "admission was not the limit" and m["alternative"] == "prefill blocks decode"
    assert m["expect"]["why"] == "fewer bytes per step" and docs[0]["rationale"] == "queue depth 40 at 2 req/s"
    assert len(search.documents(root, researcher=True)) == 4


def test_embeddings_find_what_shares_no_word_with_the_query(root):
    out = search.search(root, "latency regression", embedder=FakeEmbed())
    top = out["results"][0]
    assert top["hypothesis"].startswith("tpot is dominated") and top["bm25_rank"] is None and top["embedding_rank"] == 1
    assert "fake-concepts" in out["ranking"] and out["documents"] == 3
    assert all(not r["run"].startswith("researcher-") for r in out["results"])


def test_bm25_alone_ranks_by_shared_words_and_needs_one(root):
    out = search.search(root, "scheduler admission", embedder=None)
    assert out["results"][0]["hypothesis"].startswith("the scheduler") and out["results"][0]["bm25_rank"] == 1
    assert len(out["results"]) == 1 and "nothing else" in out["ranking"]
    assert search.search(root, "  ", embedder=None)["results"] == []


def test_researcher_runs_only_for_a_researcher(root):
    found = search.search(root, "a person's guess", embedder=None, researcher=True)["results"]
    assert found and found[0]["run"] == "researcher-x"
    assert all(r["run"] != "researcher-x" for r in search.search(root, "person guess", embedder=None)["results"])


def test_embeddings_are_cached_by_content(root):
    e = FakeEmbed()
    search.search(root, "latency", embedder=e)
    assert e.embedded == 3
    search.search(root, "memory", embedder=e)
    assert e.embedded == 3                                   # nothing changed: nothing re-embedded
    h = search.documents(root)[2]
    m = measure(root, "run-b", {"id": h["hypothesis_id"]}, 0.05)
    ledger.append({"kind": "reflection", "run": "run-b", "session": "run-b-s1", "record": m["id"], "reflection": "no"}, root)
    fresh = FakeEmbed()
    search.search(root, "latency", embedder=fresh)           # a new process: the cache file is read back
    assert fresh.embedded == 1                               # only the document that changed


def test_k_is_bounded(root):
    assert len(search.search(root, "latency attention scheduler", k=1, embedder=FakeEmbed())["results"]) == 1
    assert len(search.search(root, "latency", k=10_000, embedder=FakeEmbed())["results"]) == 3


pytest_plugins = ["tests.test_lab_proftools"]   # its `tb` fixture: a Toolbox on a tmp ledger


def test_the_tool_reports_a_missing_model_and_still_ranks_by_words(tb, monkeypatch):
    hyp(tb.s.ledger_root, tb.s.run_id, "the scheduler admits too few")

    def broken(cache):
        raise ImportError("No module named 'fastembed'")
    monkeypatch.setattr(search, "FastEmbed", broken)
    spec = next(s for s in tb.specs() if s.name == "search")
    assert spec.schema["required"] == ["query"] and "Costs no GPU time" in spec.description
    out = json.loads(spec.fn({"query": "scheduler"}))
    assert out["embeddings_unavailable"].startswith("ImportError") and out["results"][0]["bm25_rank"] == 1


def test_the_tool_uses_the_embedder_and_hides_researcher_runs(tb, monkeypatch):
    hyp(tb.s.ledger_root, tb.s.run_id, "tpot is dominated by the sampler")
    hyp(tb.s.ledger_root, "researcher-y", "tpot from a person")
    monkeypatch.setattr(search, "FastEmbed", lambda cache: FakeEmbed())
    out = json.loads(tb.search({"query": "latency", "k": 5}))
    assert [r["run"] for r in out["results"]] == [tb.s.run_id] and "embeddings_unavailable" not in out


def test_stopwords_never_rank_a_document(root):
    assert search.search(root, "why is it the", embedder=None)["results"] == []


@pytest.mark.skipif(not os.environ.get("LAB_TEST_REAL_EMBEDDINGS"), reason="downloads the model (~440 MB)")
def test_the_real_model_finds_a_paraphrase(tmp_path):
    r = tmp_path / "ledger"
    for t in ("the scheduler admits too few requests per batch", "MoE expert GEMMs are launch bound at batch 1",
              "per-token decode latency is dominated by the sampler", "prefix cache eviction scans the radix tree"):
        hyp(r, "run-a", t)
    out = search.search(r, "why are tokens generated slowly", embedder=search.FastEmbed(tmp_path / "models"))
    top = out["results"][0]
    assert top["hypothesis"].startswith("per-token decode latency") and top["embedding_rank"] == 1
    assert search.MODEL in out["ranking"]
