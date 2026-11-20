"""Fakes for the run's parts: a worker that answers measure, equiv, profile and reference jobs, and a target."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.targets import target

GPU = {"name": "Fake GPU", "driver_version": "1.0"}


class FakeWorker:
    """`values`: {tree name: saturated tokens/s}; `failing`: tree names the gate fails; `shares`: the profile's
    kernel and decode seconds of a 10 s window."""

    def __init__(self, values: dict | None = None, failing: tuple = (), shares: tuple = (4.0, 7.0)):
        self.values, self.failing, self.shares, self.calls = values or {}, failing, shares, []

    def call(self, job: str, inputs: dict, args: dict, out: Path) -> dict:
        self.calls.append((job, {k: Path(v).name for k, v in inputs.items()}, args))
        return getattr(self, job)(inputs, args, Path(out))

    def measure(self, inputs, args, out):
        base = self.values.get(inputs["tree"].name, 100.0)
        (out / "client_rows.jsonl").write_text("")
        return {"results": [{"regime": n, "objective": "tokens_per_s", "value": base + args["seed"], "better": "higher",
                             "valid": True, "invalid_reasons": [], "profile": None} for n in args["names"]],
                "ready_s": 1.0, "gpu": GPU}

    def equiv(self, inputs, args, out):
        ok = inputs["tree"].name not in self.failing
        return {"verdict": "pass" if ok else "fail", "passed": ok, "reasons": [] if ok else ["kl too high"],
                "gates": {}, "metrics": {}, "thresholds": {}, "killed": []}

    def profile(self, inputs, args, out):
        kernel_s, decode_s = self.shares
        files = out / "files"
        files.mkdir(parents=True)
        (files / "meta.json").write_text(json.dumps({"window": {"wall_s": 10.0, "start": 0.0, "end": 10.0}}))
        (files / "kernels.json").write_text(json.dumps({"total_us": kernel_s * 1e6, "launches": 5, "kernels": [
            {"name": "gemm", "count": 5, "total_us": kernel_s * 1e6}]}))
        (files / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in (
            {"kind": "phase", "name": "decode", "ts": 1.0, "dur_s": decode_s},
            {"kind": "phase", "name": "admit", "ts": 1.0, "dur_s": 0.5},
            {"kind": "step", "ts": 1.0, "dur_s": 8.0, "active": 2})))
        return {"returncode": 0, "output": "", "gpu": GPU, "killed": []}

    def reference(self, inputs, args, out):
        (out / "meta.json").write_text("{}")
        return {"complete": True}


def make_target(tmp_path: Path, repo: Path, reference: bool = True) -> target.Target:
    ref = tmp_path / "reference"
    if reference:
        ref.mkdir()
        (ref / "meta.json").write_text("{}")
    spec = tmp_path / "fake.toml"
    spec.write_text(f'''
[model]
name = "fake"
chat_kwargs = {{ }}
[engine]
repo = "{repo}"
python = "{sys.executable}"
api = "vllm"
launch = "true"
health = "/health"
write = ["src/inference_server/*"]
test = "true"
lint = "true"
[reference]
dir = "{ref}"
[limits]
interactive = {{ ttft_s = 0.25, tpot_s = 0.05 }}
conversational = {{ ttft_s = 0.25, tpot_s = 0.05 }}
[corpus]
dir = "{tmp_path / 'no-corpus'}"
''')
    return target.parse(spec)


def make_peaks(tmp_path: Path) -> Path:
    d = tmp_path / "peaks"
    d.mkdir()
    (d / "fake-gpu-1-0.json").write_text(json.dumps({"gpu": GPU["name"], "bf16_flops": 1e15, "bw_bytes_s": 3e12,
                                                     "mem_bytes": 8e10, "source": "measured", "driver": "1.0"}))
    return d
