"""End-to-end test of the loop-facing tool with a fake grader and a fake server. Run: python tests/test_tool.py"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blamegraph.tool.adapter import log_from_ledger  # noqa: E402
from blamegraph.tool.landscape import Landscape  # noqa: E402
from blamegraph.tool.ledger import Ledger  # noqa: E402
from blamegraph.tool.validate import validate  # noqa: E402
from blamegraph.assertions import evaluate  # noqa: E402

FAKE_GRADER = '''
import json, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--json-output-file"); ap.add_argument("--server-url"); ap.add_argument("--request-limit", type=int)
a, _ = ap.parse_known_args()
# metric depends on a knob in start_server.sh so the landscape has structure
cfg = open("start_server.sh").read()
seqs = int([t for t in cfg.split() if t.isdigit()][-1]) if any(t.isdigit() for t in cfg.split()) else 64
ttft = 0.80 / (1 + seqs / 64)
m = {"profiles": {"burst": {"ttft": {"p50": ttft}, "tpot": {"p50": 0.02}, "request_throughput_req_per_s": 1.0, "failure_rate": 0.0, "generation_throughput_tokens_per_s": 100}},
     "quality_check": {"pass": True}}
json.dump(m, open(a.json_output_file, "w"))
'''


def sh(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, capture_output=True, text=True)


def measure(td, *args):
    r = sh([sys.executable, "-m", "blamegraph.tool.cli", "measure", "--task-dir", str(td), "--grader", str(td / "grader.py"), "--", *args], cwd=ROOT, env={"PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr
    return json.loads(r.stderr.strip().splitlines()[-1])


def main():
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        (td / "grader.py").write_text(FAKE_GRADER)
        (td / "evaluate.py").write_text("# wrapper\n")
        (td / "start_server.sh").write_text("#!/bin/bash\nvllm serve m --max-num-seqs 64\n")
        os.chdir(td)
        led = Ledger.open(td / ".blamegraph/ledger.jsonl")
        led.start_session({"scenario": "A"}, [td / "evaluate.py"])
        # 1) measure v1 twice (full), then edit to v2 and measure (quick), then full
        r1 = measure(td, "--json-output-file", "m1.json"); assert r1["standard"] and not r1["stale_server"]
        r2 = measure(td, "--json-output-file", "m2.json")
        (td / "start_server.sh").write_text("#!/bin/bash\nvllm serve m --max-num-seqs 256\n")
        r3 = measure(td, "--quick", "--json-output-file", "m3.json")
        r4 = measure(td, "--json-output-file", "m4.json")
        r5 = measure(td, "--request-limit", "4", "--json-output-file", "m5.json"); assert not r5["standard"], "non-standard flags must be flagged"
        led = Ledger.open(td / ".blamegraph/ledger.jsonl")
        assert len(led.measurements()) == 5 and len(led.configs()) == 2
        v = validate(led, td / "start_server.sh", [td / "evaluate.py"])
        assert v.valid, v.reasons
        # 2) flip: ship an unmeasured config -> invalid
        (td / "start_server.sh").write_text("#!/bin/bash\nvllm serve m --max-num-seqs 512\n")
        v2 = validate(led, td / "start_server.sh", [td / "evaluate.py"])
        assert not v2.valid and any(r.startswith("submission_known") for r in v2.reasons), v2.reasons
        (td / "start_server.sh").write_text("#!/bin/bash\nvllm serve m --max-num-seqs 256\n")
        # 3) flip: edit the grader -> invalid
        (td / "evaluate.py").write_text("# tampered\n")
        v3 = validate(led, td / "start_server.sh", [td / "evaluate.py"])
        assert not v3.valid and any(r.startswith("grader_untouched") for r in v3.reasons), v3.reasons
        (td / "evaluate.py").write_text("# wrapper\n")
        # 4) flip: pristine hash mismatch from the harness side
        v4 = validate(led, td / "start_server.sh", [td / "evaluate.py"], pristine_grader={str(td / "evaluate.py"): "deadbeefdeadbeef"})
        assert not v4.valid
        # 5) context + landscape
        land = Landscape(td / "landscape.jsonl"); n = land.add_session(led, "A", "s1"); land.save()
        assert n == 3, n   # 3 standard full measurements (m1, m2, m4)
        near = land.near("A", (td / "start_server.sh").read_text())
        assert near and near[0]["distance"] == 0 and near[0]["n"] == 1
        r = sh([sys.executable, "-m", "blamegraph.tool.cli", "context", "--task-dir", str(td), "--scenario", "A", "--landscape", str(td / "landscape.jsonl"), "--text"], cwd=ROOT, env={"PYTHONPATH": str(ROOT)})
        assert r.returncode == 0 and "validation if submitted now: valid" in r.stdout, r.stdout + r.stderr
        assert "should" not in r.stdout.lower()   # facts only
        # 6) adapter: existing assertions run on the ledger
        log = log_from_ledger(led)
        class R:  # minimal Run stand-in for assertions that only need scenario/steps/text
            scenario = "A"
            def steps(self): return []
            def assistant_text(self): return iter(())
        a = evaluate(R(), log)
        assert a["ran_eval"] is True and a["final_measured"] is True and a["eval_untouched"] is True and a["compared_2"] is True, a
        # 7) a real listening server: stale detection and cache state
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
        srv = subprocess.Popen([sys.executable, "-u", "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=td, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(0.8)
            s1 = measure(td, "--server-url", f"http://127.0.0.1:{port}", "--json-output-file", "m6.json")
            assert s1["cache_state"] == "cold" and not s1["stale_server"], s1
            s2 = measure(td, "--server-url", f"http://127.0.0.1:{port}", "--json-output-file", "m7.json")
            assert s2["cache_state"] == "warm", s2
            # edit the config after the server started (push mtime forward to be safe) -> stale
            (td / "start_server.sh").write_text("#!/bin/bash\nvllm serve m --max-num-seqs 128\n")
            future = time.time() + 30; os.utime(td / "start_server.sh", (future, future))
            s3 = measure(td, "--server-url", f"http://127.0.0.1:{port}", "--json-output-file", "m8.json")
            assert s3["stale_server"], s3
            led = Ledger.open(td / ".blamegraph/ledger.jsonl")
            v5 = validate(led, td / "start_server.sh", [td / "evaluate.py"])
            assert not v5.valid and any("submission_measured" in r for r in v5.reasons), v5.reasons   # v3 only measured on a stale server
        finally:
            srv.terminate()
        time.sleep(0.5)
        os.utime(td / "start_server.sh", None)   # undo the artificial future mtime used above
        # restart -> fresh measurement of the new config makes it valid again
        srv = subprocess.Popen([sys.executable, "-u", "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=td, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(0.8)
            s4 = measure(td, "--server-url", f"http://127.0.0.1:{port}", "--json-output-file", "m9.json")
            assert not s4["stale_server"] and s4["cache_state"] == "cold", s4
            led = Ledger.open(td / ".blamegraph/ledger.jsonl")
            assert validate(led, td / "start_server.sh", [td / "evaluate.py"]).valid
        finally:
            srv.terminate()
        # 8) submission via cli
        r = sh([sys.executable, "-m", "blamegraph.tool.cli", "submit", "--task-dir", str(td)], cwd=ROOT, env={"PYTHONPATH": str(ROOT)})
        assert r.returncode == 0, r.stdout + r.stderr
        assert (td / ".blamegraph/validation.json").exists()
        print("tool self-test: all checks passed")
        print(sh([sys.executable, "-m", "blamegraph.tool.cli", "context", "--task-dir", str(td), "--scenario", "A", "--landscape", str(td / "landscape.jsonl"), "--text"], cwd=ROOT, env={"PYTHONPATH": str(ROOT)}).stdout)


if __name__ == "__main__":
    main()
