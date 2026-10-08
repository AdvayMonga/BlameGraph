"""lab.gpu against a fake nvidia-smi: field fallbacks, unsupported fields as None, epoch column on the samples."""

from __future__ import annotations

import csv
import io
import os
import stat
import sys
import time
from datetime import datetime

import pytest

from lab import gpu

FAKE = """#!{python}
import sys, time
args = sys.argv[1:]
fields = next(a for a in args if a.startswith("--query-gpu=")).split("=", 1)[1].split(",")
if any(f in {bad!r} for f in fields):
    sys.exit(6)
vals = {{"clocks_throttle_reasons.active": "0x0000000000000004", "clocks.applications.memory": "[N/A]",
        "timestamp": "2026/10/08 01:02:03.250"}}
row = ", ".join(vals.get(f, "1") for f in fields)
if "-lms" in args:
    print(", ".join(fields), flush=True)
    while True:
        print(row, flush=True)
        time.sleep(0.01)
print(row)
"""


@pytest.fixture
def fake_smi(tmp_path, monkeypatch):
    def install(bad=("clocks_event_reasons.active",)):
        p = tmp_path / "nvidia-smi"
        p.write_text(FAKE.format(python=sys.executable, bad=set(bad)))
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
        gpu.supported.cache_clear()
    yield install
    gpu.supported.cache_clear()


def test_no_nvidia_smi_is_none(monkeypatch):
    monkeypatch.setattr(gpu.shutil, "which", lambda _: None)
    assert gpu.query() is None and gpu.reasons_field() is None
    assert gpu.Sampler("x.csv").start() is False


def test_query_falls_back_to_the_older_throttle_field(fake_smi):
    fake_smi()
    q = gpu.query()
    assert q["clocks_event_reasons.active"] == "0x0000000000000004" and q["name"] == "1"
    assert q["clocks.applications.graphics"] == "1" and q["clocks.applications.memory"] is None


def test_query_without_any_reasons_field_reports_none(fake_smi):
    fake_smi(bad=gpu.REASONS + ("clocks.applications.graphics",))
    q = gpu.query()
    assert q["clocks_event_reasons.active"] is None and q["clocks.applications.graphics"] is None
    assert q["clocks.sm"] == "1"


def test_sampler_writes_epoch_seconds(fake_smi, tmp_path):
    fake_smi()
    s = gpu.Sampler(tmp_path / "gpu.csv")
    assert s.start()
    time.sleep(0.3)
    s.stop()
    rows = list(csv.DictReader(io.StringIO((tmp_path / "gpu.csv").read_text()), skipinitialspace=True))
    assert rows and not (tmp_path / "gpu.smi.csv").exists()
    want = datetime(2026, 10, 8, 1, 2, 3, 250000).timestamp()
    assert float(rows[0]["epoch_s"]) == pytest.approx(want, abs=1e-3)
    assert rows[0]["clocks_throttle_reasons.active"] == "0x0000000000000004"


def test_unparseable_timestamp_leaves_epoch_empty():
    out = gpu.add_epoch("timestamp, x\nnot a time, 1\n")
    assert out.splitlines() == ["epoch_s,timestamp,x", ",not a time,1"]
