"""The idea archive (kernels/archive.py): ideas and outcomes from `kernels` records, near-duplicates folded, similarity
lookup, the retry list, export and load. A fake embedder: no model download."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from pathlib import Path

from kernels import archive
from lab import ledger


class FakeEmbed:
    name = "fake"
    VOCAB = ("fuse", "combine", "split", "tile", "registers", "fp8", "launch")

    def _v(self, t):
        t = t.lower()
        return [float(t.count(w)) + 0.01 for w in self.VOCAB]

    def passages(self, texts):
        return [self._v(t) for t in texts]

    def query(self, text):
        return self._v(text)


def idea(b, m="m", p="p"):
    return {"bottleneck": b, "optimisation_method": m, "modification_plan": p, "expected_speedup": 1.1}


def run(root: Path, task: str, rows: list[dict], children: dict) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    hyp, trace = root / "h.jsonl", root / "t.jsonl"
    hyp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    trace.write_text("".join(json.dumps({"child_id": c, "child_metrics": json.dumps(m), "artifacts": json.dumps(a)})
                             + "\n" for c, (m, a) in children.items()))
    return ledger.append({"kind": "kernels", "run": "kernels-x", "config": {"task": task}, "result": {
        "hypotheses": ledger.put_blob(hyp, root, ".jsonl"), "trace": ledger.put_blob(trace, root, ".jsonl")}}, root)


def test_ideas_fold_near_duplicates_keep_every_outcome_and_answer_lookups(tmp_path):
    root = tmp_path / "ledger"
    rows = [{"parent_id": "r", "parent_score": 1.0,
             "hypotheses": [idea("fuse the combine into the split kernel"), idea("tile registers fp8")],
             "measured": [{"index": 0, "child_id": "a", "child_score": 1.12, "improved": True},
                          {"index": 1, "child_id": "b", "child_score": 0.0, "improved": False}]},
            {"parent_id": "a", "parent_score": 1.12,
             "hypotheses": [idea("Fuse the combine into the split"), idea("one launch")],
             "measured": [{"index": 0, "child_id": "c", "child_score": 1.05, "improved": False}]}]
    run(root, "paged_decode", rows, {"b": ({"stage": 1, "correct": 0.0}, {"error": "max_abs 3.1"}),
                                     "a": ({"stage": 2}, {}), "c": ({"stage": 2}, {})})
    es = archive.build(root, FakeEmbed())
    assert [len(e.outcomes) for e in es] == [2, 1, 1]          # the two "fuse the combine" ideas are one
    assert es[1].outcomes[0].error == "max_abs 3.1" and es[1].outcomes[0].stage == 1
    assert es[2].outcomes[0].child_id is None                  # named, never measured
    assert es[0].best().child_score == 1.12 and es[0].ever_improved()
    top = archive.matches(es, "fuse combine split", "paged_decode", k=1, embedder=FakeEmbed())
    assert top[0][1] is es[0] and not archive.matches(es, "fuse", "moe_gemm", embedder=FakeEmbed())
    assert archive.retry(es, {"paged_decode": "a"}) == [es[1]]    # failed on r only; the best is now a
    path = archive.export(es, tmp_path / "archive.jsonl")
    back = archive.load(path)
    assert [e.text for e in back] == [e.text for e in es] and back[1].outcomes[0].error == "max_abs 3.1"
    assert "max_abs 3.1" in json.dumps(archive.facts(back[1]))


def test_without_an_embedder_only_identical_ideas_fold_and_lookup_is_by_words(tmp_path):
    root = tmp_path / "ledger"
    rows = [{"parent_id": "r", "parent_score": 1.0, "hypotheses": [idea("fuse"), idea("fuse"), idea("tile")],
             "measured": []}]
    run(root, "rmsnorm", rows, {})
    es = archive.build(root)
    assert len(es) == 2 and len(es[0].outcomes) == 2
    assert archive.matches(es, "tile the rows")[0][1].text.startswith("tile")


if __name__ == "__main__":
    import inspect
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(*[Path(tempfile.mkdtemp()) for _ in inspect.signature(fn).parameters]); print("ok", name)
