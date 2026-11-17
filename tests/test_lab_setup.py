"""lab setup: the one-time commands for a target, each skipped where its output exists."""
import dataclasses
import os

os.environ.setdefault("LAB_TARGET", "evals/targets/inference-server.toml")

from evals.targets import target
from lab import setup


def test_steps_skip_what_exists(tmp_path, monkeypatch):
    from lab import evaltools
    bands = {"saturated": 2.0}
    monkeypatch.setattr(evaltools, "noise_band_pct", lambda n, t=None, root=None: bands.get(n))
    t = target.load()
    want = ("single_stream", "saturated")
    fresh = setup.steps(dataclasses.replace(t, corpus_dir=tmp_path / "corpus", reference_dir=tmp_path / "ref"), want)
    assert fresh == [["store", "pull", "--corpus"], ["correctness", "reference", "--out", str(tmp_path / "ref")],
                     ["validate", "--regimes", "single_stream"]]             # saturated has a band
    (tmp_path / "corpus" / "bursty").mkdir(parents=True); (tmp_path / "corpus" / "bursty" / "heldout.jsonl").write_text("")
    (tmp_path / "ref").mkdir(); (tmp_path / "ref" / "meta.json").write_text("{}")
    bands["single_stream"] = 1.0
    assert setup.steps(dataclasses.replace(t, corpus_dir=tmp_path / "corpus", reference_dir=tmp_path / "ref"), want) == []

