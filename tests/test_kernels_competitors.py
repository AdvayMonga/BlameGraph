"""The competitor timings' and the crossover's host-side logic (kernels/competitors.py, kernels/crossover.py): a wrong
competitor is reported and never timed; SASS is filtered to the kernels that matter; step times by batch from a
timeline; the faster path per batch and prompt length. The GPU parts run on the GPU host."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402


from kernels import competitors, crossover  # noqa: E402
from kernels.tasks import load  # noqa: E402


def test_a_wrong_competitor_is_reported_not_timed():
    t = load("toy_rmsnorm")
    right = lambda *xs: t.reference(*xs)          # noqa: E731
    wrong = lambda *xs: t.reference(*xs) * 2      # noqa: E731
    assert competitors.check(t, right, [(8, 256)], "cpu") is None
    assert "differs from the reference" in competitors.check(t, wrong, [(8, 256)], "cpu")
    assert "raised" in competitors.check(t, lambda *xs: 1 / 0, [(8, 256)], "cpu")


def test_sass_is_filtered_to_the_named_kernels(tmp_path, monkeypatch):
    dump = ("\n\tcode for sm_90\n\t\tFunction : _Z15rms_norm_kernelIfEvv\n\t/*0000*/ MOV R1 ;\n"
            "\n\t\tFunction : _Z9otherKernv\n\t/*0000*/ BRA 0x0 ;\n")
    monkeypatch.setattr(competitors.subprocess, "run", lambda *a, **k: type("P", (), {"stdout": dump})())
    got = competitors.sass("lib.so", r"rms_norm_kernel")
    assert "rms_norm_kernel" in got and "otherKern" not in got
    dump2 = dump + "\n\t\tFunction : _Z15rms_norm_kernelIdEvv\n\t/*0000*/ NOP ;\n"

    def run(argv, *a, **k):
        if "cu++filt" in argv[0]:
            return type("P", (), {"stdout": "void rms_norm_kernel<float>()\nvoid rms_norm_kernel<double>()\n"})()
        return type("P", (), {"stdout": dump2})()
    monkeypatch.setattr(competitors.subprocess, "run", run)
    only = competitors.sass("lib.so", r"rms_norm_kernel", {"void rms_norm_kernel<float>()"})
    assert "IfEvv" in only and "IdEvv" not in only                   # only the instantiation that ran


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


if __name__ == "__main__":
    import inspect
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_") and "monkeypatch" not in inspect.signature(fn).parameters:
            fn(*[Path(tempfile.mkdtemp()) for _ in inspect.signature(fn).parameters]); print("ok", name)
