"""lab setup: the one-time commands for a target, each skipped where its output exists."""
import dataclasses
import os

os.environ.setdefault("LAB_TARGET", "evals/targets/inference-server.toml")

from evals.targets import target
from lab import setup


def test_steps_skip_what_exists(tmp_path):
    t = target.load()
    fresh = setup.steps(dataclasses.replace(t, corpus_dir=tmp_path / "corpus", reference_dir=tmp_path / "ref"))
    assert fresh == [["store", "pull", "--corpus"], ["correctness", "reference", "--out", str(tmp_path / "ref")],
                     ["validate"]]
    (tmp_path / "corpus" / "bursty").mkdir(parents=True); (tmp_path / "corpus" / "bursty" / "heldout.jsonl").write_text("")
    (tmp_path / "ref").mkdir(); (tmp_path / "ref" / "meta.json").write_text("{}")
    done = setup.steps(dataclasses.replace(t, corpus_dir=tmp_path / "corpus", reference_dir=tmp_path / "ref"))
    assert done == [["validate"]]

