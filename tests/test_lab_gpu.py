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


def test_compute_apps_parses_pids_memory_and_names(monkeypatch):
    import subprocess
    monkeypatch.setattr(gpu, "available", lambda: True)
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, "123, 4000, python\n456, [N/A], /usr/bin/vllm, worker\n", ""))
    assert gpu.compute_apps() == [{"pid": 123, "used_mib": 4000, "name": "python"},
                                  {"pid": 456, "used_mib": None, "name": "/usr/bin/vllm, worker"}]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_descendants_include_children():
    import subprocess
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        assert {os.getpid(), child.pid} <= gpu.descendants(os.getpid())
    finally:
        child.kill()


def _device_dir(tmp_path, source, fields, lines):
    import json
    d = tmp_path / source; d.mkdir()
    (d / "meta.json").write_text(json.dumps({"available": True, "source": source, "fields": fields}))
    (d / "samples.csv").write_text("".join(f"{t},{line}\n" for t, line in lines))
    return d


def test_read_samples_parses_dcgm_lines_and_drops_headers(tmp_path):
    d = _device_dir(tmp_path, "dcgm", list(gpu.DCGM_FIELDS.values()), [
        (1.0, "#Entity         SMACT        SMOCC        TENSO        DRAMA        PCITX   PCIRX   POWER   SMCLK  MMCLK  DVCCTR"),
        (1.0, "ID                                                                                    W"),
        (1.1, "GPU 0           0.039        0.010        0.008        N/A          1328024 4133503 75.507  345    3201   1"),
        (1.2, "GPU 0           0.9")])                       # cut off when the sampler stopped: missing fields absent
    rows = list(gpu.read_samples(d))
    assert len(rows) == 2
    assert rows[0] == {"epoch_s": 1.1, "source": "dcgm", "sm_active": 0.039, "sm_occupancy": 0.01, "tensor_active": 0.008,
                       "dram_active": None, "pcie_tx_bytes": 1328024.0, "pcie_rx_bytes": 4133503.0, "power_w": 75.507,
                       "sm_clock_mhz": 345.0, "mem_clock_mhz": 3201.0, "throttle_reasons": 1}
    assert rows[1] == {"epoch_s": 1.2, "source": "dcgm", "sm_active": 0.9}


def test_read_samples_names_nvidia_smi_fields_like_dcgm(tmp_path):
    fields = [*gpu.SAMPLE, gpu.REASONS[0]]
    d = _device_dir(tmp_path, "nvidia-smi", fields, [
        (1.0, "timestamp, utilization.gpu [%], utilization.memory [%], clocks.sm [MHz], clocks.mem [MHz], power.draw [W], "
              "temperature.gpu, memory.used [MiB], clocks_event_reasons.active"),
        (1.1, "2026/10/27 10:00:00.100, 90, 40, 1980, 3201, 650.5, 61, 120000, 0x0000000000000004"),
        (1.2, "2026/10/27 10:00:00.200, 91, [N/A], 1980, 3201, 651.0, 61, 120000, 0x0000000000000000")])
    rows = list(gpu.read_samples(d))
    assert [r["epoch_s"] for r in rows] == [1.1, 1.2]
    assert rows[0] == {"epoch_s": 1.1, "source": "nvidia-smi", "timestamp": "2026/10/27 10:00:00.100", "utilization_gpu": 90.0,
                       "utilization_memory": 40.0, "sm_clock_mhz": 1980.0, "mem_clock_mhz": 3201.0, "power_w": 650.5,
                       "temperature_gpu": 61.0, "memory_used": 120000.0, "throttle_reasons": 4}
    assert rows[1]["utilization_memory"] is None


def test_read_samples_without_a_source_is_empty(tmp_path):
    import json
    (tmp_path / "meta.json").write_text(json.dumps({"available": False}))
    assert list(gpu.read_samples(tmp_path)) == []
