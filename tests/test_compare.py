"""The engine comparison's driver and report on fake engines (no GPU, no docker). Run: python tests/test_compare.py"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compare import report, run as R  # noqa: E402

FAKE = Path(__file__).resolve().parent / "fake_engine.py"
os.environ.setdefault("LAB_TARGET", str(FAKE.parent.parent / "targets" / "inference-server.toml"))  # run as a script too


def _spec(slots: dict[str, int]) -> dict:
    spec = R.load()
    spec["common"]["engine_cpus"] = spec["common"]["client_cpus"] = "0"
    spec["common"]["tokenizer"] = None
    spec["engines"] = {name: {"kind": "venv", "version": "fake", "api": "none", "health": "/health",
                              "cmd": f"{sys.executable} {FAKE} --port {{port}} --slots {n} --token-s 0.002 {{env}}",
                              "precision": {"bf16": "", "fp8": ""}, "checkpoint": "fake"} for name, n in slots.items()}
    return spec


def test_the_driver_launches_checks_and_measures_each_configuration():
    R.gpu_free = lambda timeout_s=300.0: True
    real = R.command
    R.command = lambda *a: real(*a).split(" env ", 1)[1]
    try:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            R.run(out, ["a", "b"], ["bf16"], ["launch", "regimes"], 1, "single_stream", 0.05, spec=_spec({"a": 8, "b": 2}))
            for e in ("a", "b"):
                launch = json.loads((out / f"{e}-bf16" / "launch.json").read_text())
                assert launch["ok"] and launch["fixed_tokens"] == 300, launch
                res = json.loads((out / f"{e}-bf16" / "regimes-0.json").read_text())["results"]
                assert res[0]["regime"] == "single_stream" and res[0]["valid"], res[0]["invalid_reasons"]
            text = report.render(out)
            assert "| single_stream |" in text and "fewer than 2 repeats" in text
    finally:
        R.command = real


def test_a_win_is_clear_only_beyond_noise_and_the_bracket():
    s = lambda v, spread=1.0, fails=None, n=3, better="higher": {"value": v, "n": n, "of": n, "spread_pct": spread,
                                                                  "fails_at": fails, "stopped": ["resolved"],
                                                                  "better": better, "invalid": []}
    assert report.verdict({"a": s(10.0, fails=10.2), "b": s(8.0, fails=8.2)}, 2.0)["clear"]
    near = report.verdict({"a": s(10.0, fails=10.2), "b": s(9.9, fails=10.1)}, 2.0)
    assert near["winner"] == "a" and not near["clear"] and "within noise" in near["why"]
    inside = report.verdict({"a": s(10.0, fails=10.2), "b": s(9.0, fails=12.0)}, 2.0)
    assert not inside["clear"] and "bracket" in inside["why"]
    lower = report.verdict({"a": s(0.0036, better="lower"), "b": s(0.0041, better="lower")}, 2.0)
    assert lower["winner"] == "a" and lower["clear"]
    assert not report.verdict({"a": s(10.0, n=1), "b": s(5.0, n=1)}, 2.0)["clear"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
