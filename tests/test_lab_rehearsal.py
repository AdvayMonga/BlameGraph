"""`python -m lab.rehearsal setup`: the practice engine is a working git repo, the target loads, and the references
are complete. A word-per-token encoder stands in for the model's tokenizer."""
from __future__ import annotations

import subprocess
import sys

import pytest

from lab import rehearsal, target


class Words:
    def encode(self, text):
        return [hash(w) % 1000 + 1 for w in text.split()]

    def chat_ids(self, messages):
        return self.encode(" ".join(m["content"] for m in messages))


def test_setup_builds_a_runnable_rehearsal(tmp_path):
    out = rehearsal.setup(tmp_path / "r", n_dev=12, n_full=16, encoder=Words())
    d = tmp_path / "r"
    assert out["references"]["dev"]["n_items"] == 12 and out["references"]["full"]["n_items"] == 16
    assert target.reference_complete(d / "ref") and target.reference_complete(d / "ref-full")
    t = target.load(str(d / "target.toml"))
    assert t.engine_repo == d / "engine" and t.submit_tier == "dev" and t.reference_dir_for("full") == d / "ref-full"
    log = subprocess.run(["git", "-C", str(d / "engine"), "log", "--oneline"], capture_output=True, text=True).stdout
    assert len(log.splitlines()) == 1
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=d / "engine",
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with pytest.raises(SystemExit, match="exists"):
        rehearsal.setup(d, encoder=Words())
