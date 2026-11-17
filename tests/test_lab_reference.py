"""The correctness reference built by worker jobs: each continues the last, until its meta.json is written."""
import json
import os

import pytest

os.environ.setdefault("LAB_TARGET", "evals/targets/inference-server.toml")

from lab.gate.correctness import hf
from lab.infra.runner import worker


class Worker:
    """Answers `per` items a call; writes meta.json once every item is answered."""

    def __init__(self, n: int, per: int):
        self.n, self.per, self.calls = n, per, 0

    def call(self, job, inputs, args, out):
        assert job == "reference" and args["tasks"] == ["mmlu_pro"]
        self.calls += 1
        prior = inputs["prior"] / "outputs.jsonl"
        rows = prior.read_text().splitlines() if prior.exists() else []
        rows += [json.dumps({"id": k}) for k in range(len(rows), min(self.n, len(rows) + self.per))]
        out.mkdir(parents=True, exist_ok=True)
        (out / "outputs.jsonl").write_text("\n".join(rows) + "\n")
        if len(rows) == self.n:
            (out / "meta.json").write_text("{}")
        return {"complete": len(rows) == self.n}


def test_build_calls_the_worker_until_the_reference_is_complete(tmp_path):
    wk = Worker(n=5, per=2)
    hf.build(tmp_path / "ref", ("mmlu_pro",), wk)
    assert wk.calls == 3 and (tmp_path / "ref" / "meta.json").exists()
    assert len((tmp_path / "ref" / "outputs.jsonl").read_text().splitlines()) == 5
    hf.build(tmp_path / "ref", ("mmlu_pro",), wk)                       # complete: no job
    assert wk.calls == 3
    with pytest.raises(RuntimeError, match="answered nothing new"):
        hf.build(tmp_path / "stuck", ("mmlu_pro",), Worker(n=5, per=0))


def test_the_reference_job_continues_the_prior_dir(tmp_path, monkeypatch):
    from lab.gate.correctness import tasks
    monkeypatch.setattr(tasks, "load", lambda names, seed=0: [{"id": "a"}, {"id": "b"}])
    seen = {}

    def reference(model, out, items, chat_kwargs, seconds):
        seen.update(model=model, items=items, seconds=seconds, prior=(out / "outputs.jsonl").read_text())
        return None

    (tmp_path / "prior").mkdir(); (tmp_path / "prior" / "outputs.jsonl").write_text("kept\n")
    reply = worker.Local(reference=reference).call("reference", {"prior": tmp_path / "prior"},
                                                   {"tasks": ["mmlu_pro"], "seconds": 5}, tmp_path / "out")
    assert reply == {"complete": False} and seen["prior"] == "kept\n" and seen["seconds"] == 5
    assert seen["items"] == [{"id": "a"}, {"id": "b"}] and seen["model"]
