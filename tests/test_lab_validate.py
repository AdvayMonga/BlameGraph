"""lab.gate.validate on a fake engine: noise bands from repeated runs."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lab.records import ledger
from lab.gate import validate
from evals.targets import target
from evals.loadgen import suite

FAKE = Path(__file__).resolve().parent / "fake_engine.py"


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
[limits]
interactive = {{ ttft_s = 0.25, tpot_s = 0.05 }}
conversational = {{ ttft_s = 0.25, tpot_s = 0.05 }}
[corpus]
dir = "{tmp_path / 'no-corpus'}"
''')
    monkeypatch.setenv("LAB_TARGET", str(t)); monkeypatch.setenv("LAB_ENGINE_PYTHON", sys.executable)
    target._cache.clear()
    saved = dict(suite.TIERS["min"]); suite.TIERS["min"].update(probe_s=1.0, final_s=1.5)
    yield target.load()
    suite.TIERS["min"].update(saved); target._cache.clear()


def test_validate_measures_bands(spec, tmp_path):
    rep = validate.validate(spec, noise_runs=2, regimes=["single_stream"], out=tmp_path / "out", root=tmp_path / "ledger")
    [rec] = ledger.records(tmp_path / "ledger", kind="noise")
    noise = rec["result"]
    assert noise["runs"] == 2 and noise["band_pct"] is not None and rep["bands_measured"] is True
    assert noise["server"] == "engine" and noise["target"] == spec.name
    assert "engine_commit" in noise and noise["engine_commit"] is None     # not a git repo
    assert rep["noise"]["single_stream"] == noise and (tmp_path / "out" / "report.json").exists()


def test_a_searched_band_is_never_below_the_search_resolution():
    assert validate.band_pct([80.0, 80.0, 80.0], validate.RESOLUTION_PCT) == validate.RESOLUTION_PCT
    assert validate.band_pct([8.0, 10.0], validate.RESOLUTION_PCT) > validate.RESOLUTION_PCT
    assert validate.band_pct([0.004, 0.004]) == 0.0 and validate.band_pct([80.0]) is None
