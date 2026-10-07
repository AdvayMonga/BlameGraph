"""lab.validate on fakes: the reference, a known-good (same fake) and a known-bad (every 4th answer wrong) candidate,
plus noise bands from repeated runs. Run: python -m pytest tests/test_lab_validate.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lab import target, validate
from regimes import suite

FAKE = Path(__file__).resolve().parent / "fake_engine.py"


class Enc:
    def encode(self, text):
        return [hash(w) % 1000 + 1 for w in text.split()]

    def chat_ids(self, messages):
        return self.encode(" ".join(m["content"] for m in messages))


@pytest.fixture
def spec(tmp_path, monkeypatch):
    t = tmp_path / "fake.toml"
    t.write_text(f'''
[model]
name = "fake"
[engine]
repo = "{tmp_path}"
python = "{sys.executable}"
launch = "python {FAKE} --port {{port}}"
test = "true"
lint = "true"
[reference]
launch = "python {FAKE} --port {{port}} --out-tokens 6"
dir = "{tmp_path / 'ref'}"
startup_timeout_s = 30
[correctness]
tasks = ["mmlu_pro"]
[limits]
interactive = {{ ttft_s = 0.25, tpot_s = 0.05 }}
conversational = {{ ttft_s = 0.25, tpot_s = 0.05 }}
[corpus]
dir = "{tmp_path / 'no-corpus'}"
[validation]
good = {{ same = "python {FAKE} --port {{port}} --out-tokens 6" }}
bad = {{ wrong4 = "python {FAKE} --port {{port}} --out-tokens 6 --wrong-every 4" }}
''')
    monkeypatch.setenv("LAB_TARGET", str(t)); monkeypatch.setenv("LAB_ENGINE_PYTHON", sys.executable)
    monkeypatch.setattr(validate, "NOISE_DIR", tmp_path / "noise")
    target._cache.clear()
    saved = dict(suite.TIERS["short"]); suite.TIERS["short"].update(probe_s=1.0, final_s=1.5)
    yield target.load()
    suite.TIERS["short"].update(saved); target._cache.clear()


def test_validate_separates_and_measures_bands(spec, tmp_path, monkeypatch):
    from correctness import tasks
    items = [{"id": f"mc-{k}", "task": "mmlu_pro", "gold": "C", "max_tokens": 6,
              "messages": [{"role": "user", "content": f"question {k} pick one"}]} for k in range(60)]
    monkeypatch.setattr(tasks, "load", lambda names, tier="dev", **kw: items)
    rep = validate.validate(spec, noise_runs=2, regimes=["single_stream"], out=tmp_path / "out", encoder=Enc())
    assert rep["candidates"]["same"]["verdict"] == "pass"
    assert rep["candidates"]["wrong4"]["verdict"] == "fail", rep["candidates"]["wrong4"]
    assert rep["gate_separates"] is True
    noise = json.loads((tmp_path / "noise" / "single_stream.json").read_text())
    assert noise["runs"] == 2 and noise["band_pct"] is not None and rep["bands_measured"] is True
    assert (tmp_path / "ref" / "outputs.jsonl").exists() and (tmp_path / "out" / "report.json").exists()
