"""The megakernel crossover's host-side logic (lab/profile/crossover.py)."""
from __future__ import annotations

import json
import types
from pathlib import Path

from lab.profile import crossover


def _proc(stdout="", returncode=0, stderr=""):
    return types.SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)


def test_step_times_by_batch_and_the_faster_path(tmp_path):
    runs = {}
    for path, ms in (("megakernel", {1: 4.1, 2: 5.4, 4: 7.0}), ("kernels", {1: 5.6, 2: 5.7, 4: 6.0})):
        for b, v in ms.items():
            d = tmp_path / f"{path}-512-b{b}" / "x"
            d.mkdir(parents=True)
            lines = [{"kind": "step", "active": b, "dur_s": v / 1e3} for _ in range(3)]
            lines += [{"kind": "step", "active": 1, "dur_s": 9.0},                      # the run's tail
                      {"kind": "step", "active": b, "prefilling": 1, "dur_s": 1.0}, {"kind": "phase", "dur_s": 9}]
            (d / "events.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
            runs[(path, 512, b)] = crossover.steps(tmp_path / f"{path}-512-b{b}")
    t = crossover.table(runs)["512"]
    assert t["1"]["faster"] == "megakernel" and t["2"]["faster"] == "megakernel" and t["4"]["faster"] == "kernels"
    assert t["2"]["megakernel_ms"] == 5.4 and t["2"]["megakernel_steps"] == 3        # tail and prefill left out
    off = crossover.table(runs, {(512, 1)})["512"]["1"]
    assert off["megakernel_not_on"] and "faster" not in off


def test_crossover_reruns_on_a_changed_setting_and_marks_a_megakernel_that_never_ran(tmp_path, monkeypatch):
    import evals.targets.target
    monkeypatch.setattr(evals.targets.target, "load", lambda: types.SimpleNamespace(engine=types.SimpleNamespace(env={"W": "1"})))
    ran = []

    def run(argv, env, **k):
        d, plen = Path(argv[argv.index("--out") + 1]), int(argv[argv.index("--prompt-len") + 1])
        ran.append((env["MEGAKERNEL"], plen))
        d.mkdir(parents=True)
        b = int(argv[argv.index("--requests") + 1])
        (d / "events.jsonl").write_text(json.dumps({"kind": "step", "active": b, "dur_s": 0.004}))
        on = env["MEGAKERNEL"] == "1" and plen != 4096 and env["W"] == "1"
        (d / "serve.log").write_text(crossover.MEGA_ON + ": batch <= 16" if on else "")       # the engine's log
        return _proc("")
    monkeypatch.setattr(crossover.subprocess, "run", run)
    out = tmp_path / "new" / "dir"                                                     # made by the run
    args = ["--out", str(out), "--batches", "1", "--prompt-lens", "512,4096"]
    crossover.main(args)
    t = json.loads((out / "crossover.json").read_text())
    assert len(ran) == 4 and t["4096"]["1"]["megakernel_not_on"] and t["512"]["1"]["faster"]
    crossover.main(args)
    assert len(ran) == 4                                                               # same settings: reused
    crossover.main(args + ["--max-tokens", "64"])
    assert len(ran) == 8                                                               # changed: run again
    try:
        crossover.main(args + ["--batches", "1,32"])
        raise AssertionError("a batch past the megakernel's rows was run")
    except SystemExit:
        pass
    assert len(ran) == 8
