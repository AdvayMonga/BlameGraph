"""The competitor timings' and the crossover's host-side logic (kernels/competitors.py, kernels/crossover.py): a wrong
competitor is reported and never timed, and so is one whose timed outputs differ; SASS and PTX are filtered to the
kernels that matter; adapters' per-call work; step times by batch from a timeline; the faster path per batch and prompt
length, and when a run is reused. The GPU parts run on the GPU host."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402
import types  # noqa: E402

import torch  # noqa: E402

from kernels import competitors, crossover  # noqa: E402
from kernels.tasks import load  # noqa: E402


def test_a_wrong_competitor_is_reported_not_timed():
    t = load("toy_rmsnorm")
    right = lambda *xs: t.reference(*xs)          # noqa: E731
    wrong = lambda *xs: t.reference(*xs) * 2      # noqa: E731
    assert competitors.check(t, right, [(8, 256)], "cpu") is None
    assert "differs from the reference" in competitors.check(t, wrong, [(8, 256)], "cpu")
    assert "raised" in competitors.check(t, lambda *xs: 1 / 0, [(8, 256)], "cpu")


def _proc(stdout="", returncode=0, stderr=""):
    return types.SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)


def test_sass_is_filtered_to_the_named_kernels(tmp_path, monkeypatch):
    dump = ("\n\tcode for sm_90\n\t\tFunction : _Z15rms_norm_kernelIfEvv\n\t/*0000*/ MOV R1 ;\n"
            "\n\t\tFunction : _Z9otherKernv\n\t/*0000*/ BRA 0x0 ;\n")
    monkeypatch.setattr(competitors.subprocess, "run", lambda *a, **k: _proc(dump))
    got = competitors.sass("lib.so", r"rms_norm_kernel")
    assert "rms_norm_kernel" in got and "otherKern" not in got
    dump2 = dump + "\n\t\tFunction : _Z15rms_norm_kernelIdEvv\n\t/*0000*/ NOP ;\n"

    def run(argv, *a, **k):
        if "cu++filt" in argv[0]:
            return _proc("void rms_norm_kernel<float>()\nvoid rms_norm_kernel<double>()\n")
        return _proc(dump2)
    monkeypatch.setattr(competitors.subprocess, "run", run)
    only = competitors.sass("lib.so", r"rms_norm_kernel", {"void rms_norm_kernel<float>()"})
    assert "IfEvv" in only and "IdEvv" not in only                   # only the instantiation that ran


def test_sass_keeps_the_running_gpus_code_and_raises_when_cuobjdump_fails(monkeypatch):
    dump = ("\n\tcode for sm_80\n\t\tFunction : _Z15rms_norm_kernelv\n\t/*0000*/ OLD ;\n"
            "\n\tcode for sm_90a\n\t\tFunction : _Z15rms_norm_kernelv\n\t/*0000*/ NEW ;\n")
    monkeypatch.setattr(competitors.subprocess, "run", lambda *a, **k: _proc(dump))
    got = competitors.sass("lib.so", r"rms_norm_kernel", arch="sm_90")
    assert "NEW" in got and "OLD" not in got
    assert "OLD" in competitors.sass("lib.so", r"rms_norm_kernel", arch="sm_80")
    monkeypatch.setattr(competitors.subprocess, "run", lambda *a, **k: _proc("", 1, "no device code"))
    try:
        competitors.sass("lib.so", r"rms_norm_kernel")
        raise AssertionError("a failed cuobjdump returned")
    except RuntimeError as e:
        assert "no device code" in str(e)


def test_sass_matches_template_instantiations_whose_args_hold_parentheses(monkeypatch):
    dump = ("\n\tcode for sm_90\n\t\tFunction : _ZN4vllm3moe17topkGatingSoftmaxILi0EEEvv\n\t/*0000*/ ZERO ;\n"
            "\n\t\tFunction : _ZN4vllm3moe17topkGatingSoftmaxILi1EEEvv\n\t/*0000*/ ONE ;\n")
    demangled = ("void vllm::moe::topkGatingSoftmax<(vllm::moe::ScoringFunc)0>(float const*, int)\n"
                 "void vllm::moe::topkGatingSoftmax<(vllm::moe::ScoringFunc)1>(float const*, int)\n")
    monkeypatch.setattr(competitors.subprocess, "run",
                        lambda argv, *a, **k: _proc(demangled if "cu++filt" in argv[0] else dump))
    ran = {"void vllm::moe::topkGatingSoftmax<(vllm::moe::ScoringFunc)0>(float const*, int, float)"}   # other args
    got = competitors.sass("lib.so", r"topkGating", ran)
    assert "ZERO" in got and "ONE" not in got


def test_flashinfer_sass_comes_from_its_jit_built_libraries(tmp_path, monkeypatch):
    monkeypatch.setattr(competitors, "_flashinfer_jit_dir", lambda: tmp_path)
    try:
        competitors.library_sass("flashinfer.batch_decode", set(), "sm_90")
        raise AssertionError("no library, no error")
    except FileNotFoundError as e:
        assert str(tmp_path) in str(e)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "a.so").write_text("")
    (tmp_path / "b.so").write_text("")
    dump = "\n\tcode for sm_90\n\t\tFunction : _Z17BatchDecodeKernelv\n\t/*0000*/ HMMA ;\n"
    monkeypatch.setattr(competitors.subprocess, "run",
                        lambda argv, *a, **k: _proc(dump) if argv[-1].endswith("a.so") else _proc("", 1, "bad elf"))
    assert "HMMA" in competitors.library_sass("flashinfer.batch_decode", set(), "sm_90")
    monkeypatch.setattr(competitors.subprocess, "run", lambda *a, **k: _proc("", 1, "bad elf"))
    try:
        competitors.library_sass("flashinfer.batch_decode", set(), "sm_90")
        raise AssertionError("nothing carded, no error")
    except RuntimeError as e:
        assert "bad elf" in str(e)


def test_flashinfer_plans_once_per_input_set_and_a_call_is_run_alone(monkeypatch):
    made = []

    class Wrapper:
        def __init__(self, workspace, kv_layout):
            self.workspace = workspace
            made.append(self)

        def plan(self, indptr, indices, last, *a, **k):
            self.lists = (indptr.tolist(), indices.tolist(), last.tolist())

        def run(self, q, kv):
            return q * 2
    monkeypatch.setitem(sys.modules, "flashinfer", types.SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=Wrapper))
    competitors._reset()
    q, pool = torch.zeros(2, 4, 8), torch.zeros(10, 2, 16, 8)
    sets = [(torch.tensor([[5, 6, 7], [8, 9, 0]], dtype=torch.int32), torch.tensor([33, 10])) for _ in range(2)]
    for bt, lens in sets:
        competitors._flashinfer_decode(q, pool, pool, bt, lens)
    assert len(made) == 2 and made[0].workspace is made[1].workspace                  # one shared workspace
    assert made[0].lists == ([0, 3, 4], [5, 6, 7, 8], [1, 10])
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("host work in a call"))  # noqa: E731
    monkeypatch.setattr(torch, "cumsum", boom)
    monkeypatch.setattr(torch, "cat", boom)
    for bt, lens in sets:
        assert torch.equal(competitors._flashinfer_decode(q, pool, pool, bt, lens), q * 2)
    assert len(made) == 2
    competitors.CONFIGS["x"], competitors._MOE[1] = {}, 1
    competitors._reset()
    assert not competitors._FI and not competitors._MOE and not competitors.CONFIGS


def test_timed_paths_skip_copies_and_conversions(monkeypatch):
    def fused_add_rms_norm(x, res, w, eps):
        res.add_(x)
        x.copy_(res * w)
    monkeypatch.setattr(competitors, "_vllm", lambda *m: None)
    monkeypatch.setattr(torch.ops, "_C", types.SimpleNamespace(fused_add_rms_norm=fused_add_rms_norm))
    h, a, w = torch.ones(2, 4), torch.ones(2, 4), torch.full((4,), 2.0)
    res, x = competitors._vllm_fused_add_rms_norm(h, a, w)
    assert res is not h and torch.equal(h, torch.ones(2, 4))                           # checked: inputs untouched
    res, x = competitors._vllm_fused_add_rms_norm(h, a, w, timed=True)
    assert res is not h and torch.equal(h, torch.ones(2, 4)) and float(res[0, 0]) == 2.0     # timed: on its own copy
    again, _ = competitors._vllm_fused_add_rms_norm(h, a, w, timed=True)
    assert again is res                                                                  # made once per input set

    def topk_softmax(w, ids, tok, logits, *rest):
        w.fill_(0.5)
        ids.zero_()
    monkeypatch.setattr(torch.ops, "_moe_C", types.SimpleNamespace(topk_softmax=topk_softmax))
    logits = torch.zeros(3, 16)
    assert competitors._vllm_topk_softmax(logits)[1].dtype == torch.int64
    out = competitors._vllm_topk_softmax(logits, timed=True)
    assert out[1].dtype == torch.int32
    assert competitors.UNTIMED["vllm.topk_softmax"](out, logits)[1].dtype == torch.int64


def test_timed_outputs_are_checked_and_each_shape_keeps_its_spread_and_config(monkeypatch):
    t = load("toy_rmsnorm")

    def right(x, w, timed=False):
        competitors.CONFIGS["c"] = {"rows": x.shape[0]}
        return t.reference(x, w)
    per = competitors.time_competitor(t, "c", right, t.reference, [(8, 256), (16, 256)], {"[8, 256]": 3}, 6, "cpu")
    assert [r["calls"] for r in per] == [3, 1] and [r["config"]["rows"] for r in per] == [8, 16]
    assert all(r["ratio_p25"] <= r["ours_over_competitor"] <= r["ratio_p75"] for r in per)
    lies = lambda x, w, timed=False: t.reference(x, w) * (2 if timed else 1)            # noqa: E731
    why = competitors.time_competitor(t, "c", lies, t.reference, [(8, 256)], {}, 6, "cpu")
    assert why.startswith("at shape [8, 256]: its timed outputs differ")
    flipped = lambda x, w, timed=False: t.reference(x, w).flip(0) if timed else t.reference(x, w)   # noqa: E731
    assert isinstance(competitors.time_competitor(t, "c", flipped, t.reference, [(8, 256)], {}, 6, "cpu"), str)
    monkeypatch.setitem(competitors.UNTIMED, "c", lambda o, x, w: o.flip(0))           # its own order, mapped back
    assert isinstance(competitors.time_competitor(t, "c", flipped, t.reference, [(8, 256)], {}, 6, "cpu"), list)
    in_place = lambda x, w, timed=False: x.copy_(t.reference(x, w)) if timed else t.reference(x, w)  # noqa: E731
    monkeypatch.setattr(competitors, "IN_PLACE", {"p"})
    assert "changed" in competitors.time_competitor(t, "p", in_place, t.reference, [(8, 256)], {}, 6, "cpu")  # shared
    scratch = {}                                          # in place on its own copy of each input set: timed

    def own(x, w, timed=False):
        if not timed:
            return t.reference(x, w)
        return scratch.setdefault(x.data_ptr(), torch.empty_like(x)).copy_(t.reference(x, w))
    assert isinstance(competitors.time_competitor(t, "p", own, t.reference, [(8, 256)], {}, 6, "cpu"), list)
    raises = lambda x, w, timed=False: 1 / 0 if timed else t.reference(x, w)           # noqa: E731
    assert competitors.time_competitor(t, "r", raises, t.reference, [(8, 256)], {}, 6, "cpu").startswith(
        "timing raised ZeroDivisionError")


def test_our_moe_routing_in_vllms_layout_reads_the_same_rows_and_experts():
    import torch
    from kernels.tasks.moe_gemm import _routing
    g = torch.Generator().manual_seed(0)
    T, E, k, M = 6, 8, 2, 12
    experts = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])           # real routing: k per token
    order = torch.argsort(experts.flatten(), stable=True)
    tok = (order // k).to(torch.int32)                                                    # our pair i = sorted position
    _, sorted_ids, expert_ids = _routing(M, E, T, 1, g, sorted_e=experts.flatten()[order])
    a = torch.randn(T, 4)
    p, top_k, gather, v_ids, e_ids, padded = competitors._vllm_routing(a, tok, sorted_ids, expert_ids, E, 4)
    assert top_k == k and not gather and int(padded) % 4 == 0
    our_e = experts.flatten()[order]                                                      # expert of our pair i
    vrow_e = e_ids.long().repeat_interleave(4)[: int(padded)]
    for slot, vp in enumerate(v_ids[: int(padded)].tolist()):
        if vp < M:
            i = int((p == vp).nonzero())
            assert vp // top_k == int(tok[i]) and int(vrow_e[slot]) == int(our_e[i])     # same row, same expert
    tok2, s2, e2 = _routing(M, E, T, 1, g)                                                # tokens with uneven pairs
    if len(set(torch.bincount(tok2.long(), minlength=T).tolist())) > 1:
        assert competitors._vllm_routing(a, tok2, s2, e2, E, 4)[2]                       # rows gathered instead


def test_ptx_per_implementation_and_the_weighted_ratio(tmp_path):
    (tmp_path / "h1").mkdir()
    (tmp_path / "h1" / "fused_moe_kernel.ptx").write_text("THEIRS")
    (tmp_path / "h1" / "_moe_gemm_kernel.ptx").write_text("OURS")
    assert competitors.ptx(tmp_path, r"_moe_gemm_kernel") == "OURS"
    assert competitors.ptx(tmp_path, r"fused_moe_kernel") == "THEIRS"
    r = competitors.weighted([{"calls": 3, "ours_over_competitor": 2.0}, {"calls": 1, "ours_over_competitor": 0.5}])
    assert abs(r - 2 ** 0.5) < 1e-9
    assert competitors.weighted([{"calls": 1, "ratio_p25": 4.0}], "ratio_p25") == 4.0
    before = set(tmp_path.rglob("*.ptx"))                                               # a second user of the kernel
    (tmp_path / "h2").mkdir()
    (tmp_path / "h2" / "fused_moe_kernel.ptx").write_text("W8A8")
    assert competitors.ptx(tmp_path, r"fused_moe_kernel", before) == "W8A8"


def test_a_competitor_is_fastest_only_clear_of_the_spread():
    def c(med, p25, p75):
        return {"per_shape": [{}], "ours_over_competitor": med, "ratio_p25": p25, "ratio_p75": p75}
    assert competitors.fastest({"a": c(1.2, 1.05, 1.3), "b": c(1.4, 1.1, 1.6), "x": {"not_timed": "wrong"}}) == "b"
    assert competitors.fastest({"a": c(1.02, 0.95, 1.1), "b": c(0.8, 0.7, 0.9)}) == "within_noise"
    assert competitors.fastest({"a": c(0.8, 0.7, 0.9), "x": {"not_timed": "wrong"}}) == "ours"
    assert competitors.fastest({}) == "ours"


def test_vllm_configs_from_the_shapes_expert_shape():
    seen = []
    fm = types.SimpleNamespace(get_moe_configs=lambda E, N, dtype: seen.append((E, N, dtype)),
                               get_default_config=lambda M, E, N, K, topk, dtype: {"N": N, "K": K, "topk": topk})
    shapes = [(64, 2048, 1536, 128, 8, 1), (128, 2048, 1536, 128, 16, 1)]
    assert competitors.vllm_configs(fm, shapes, None) == {8: {"N": 768, "K": 2048, "topk": 8},
                                                          16: {"N": 768, "K": 2048, "topk": 8}}
    assert seen == [(128, 768, None)]
    fm.get_moe_configs = lambda E, N, dtype: {1: "tuned"}
    assert competitors.vllm_configs(fm, shapes, None) == {1: "tuned"}
    try:
        competitors.vllm_configs(fm, shapes + [(8, 768, 2048, 128, 8, 0)], None)
        raise AssertionError("two expert shapes, one table")
    except ValueError:
        pass


def test_the_fused_moe_port_applies_every_edit_once_or_refuses():
    kernel = ("@triton.jit\ndef fused_moe_kernel(\n    a_ptr,\n    b_ptr,\n    USE_TD: tl.constexpr = False,\n):\n"
              "    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)\n"
              "    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)\n"
              "    token_mask = offs_token < num_valid_tokens\n"
              "    x = offs_token[:, None] // top_k * stride_am\n    y = offs_token[None, :] // top_k * stride_am\n")
    prog = "# EVOLVE-BLOCK-START\ndef moe_gemm(): pass\n# EVOLVE-BLOCK-END\n\nkernel = moe_gemm\n"
    out = competitors.port_fused_moe(prog, kernel, "@triton.jit\ndef write_zeros_to_output(): pass\n",
                                     {"1": {"BLOCK_SIZE_N": 64}}, "0.32.0")
    assert "tok_ptr" in out and "offs_row[:, None]" in out and "_moe_gemm_engine = moe_gemm" in out
    assert out.index("def fused_moe_kernel") < out.index("# EVOLVE-BLOCK-END") < out.index("kernel = moe_gemm")
    compile(out.replace("@triton.jit", ""), "port", "exec")
    configs = {}
    exec(out[out.index("_VLLM_CONFIGS = "):].split("\n", 1)[0], {}, configs)              # the table's keys are ints
    assert all(isinstance(k, int) for k in configs["_VLLM_CONFIGS"])
    assert "EM = expert_ids.numel() * block_m" in out and "sorted_ids.numel()" not in out    # whole blocks only
    try:
        competitors.port_fused_moe(prog, kernel.replace("    b_ptr,", "    bb_ptr,"), "", {}, "x")
        raise AssertionError("a changed kernel was ported")
    except ValueError as e:
        assert "no longer applies" in str(e)


def test_the_topk_softmax_port_builds_vllms_kernel_at_import_and_passes_the_screen(monkeypatch):
    import re

    from kernels.evolve import core, run
    engine = ("import torch\n\n\ndef rmsnorm(x, w):\n    return x\n\n\n"
              "def topk_softmax(logits, k, renorm, fp32=False):\n    return logits, logits\n")
    monkeypatch.setattr(run, "initial_program",
                        lambda task: run._start(types.SimpleNamespace(start_edits=()), engine, "topk_softmax"))
    prog = competitors.start_program(load("topk_softmax"), "vllm.topk_softmax", [])
    assert core.screen(prog) == []
    compile(prog, "start", "exec")
    assert prog.rstrip().endswith("kernel = topk_softmax") and "def rmsnorm" in prog and "load_inline(" in prog
    assert "_topk_softmax_engine = topk_softmax" in prog
    assert prog.index("load_inline(") < prog.index("# EVOLVE-BLOCK-END")
    assert not re.search(r'#\s*include\s*"', competitors.TOPK_CUDA)                        # no vLLM-local headers
    assert competitors.start_program(load("rmsnorm"), "vllm.topk_softmax", []) is None


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
    import lab.target
    monkeypatch.setattr(lab.target, "load", lambda: types.SimpleNamespace(engine=types.SimpleNamespace(env={"W": "1"})))
    ran = []

    def run(argv, env, **k):
        d, plen = Path(argv[argv.index("--out") + 1]), int(argv[argv.index("--prompt-len") + 1])
        ran.append((env["MEGAKERNEL"], plen))
        d.mkdir(parents=True)
        b = int(argv[argv.index("--requests") + 1])
        (d / "events.jsonl").write_text(json.dumps({"kind": "step", "active": b, "dur_s": 0.004}))
        on = env["MEGAKERNEL"] == "1" and plen != 4096 and env["W"] == "1"
        return _proc(crossover.MEGA_ON + ": batch <= 16" if on else "")
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


if __name__ == "__main__":
    import inspect
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_") and "monkeypatch" not in inspect.signature(fn).parameters:
            fn(*[Path(tempfile.mkdtemp()) for _ in inspect.signature(fn).parameters]); print("ok", name)


def test_a_demangled_name_matches_the_profilers_without_its_return_type():
    cxx = ("std::enable_if<(((int)0)==(0))&&vllm::_typeConvert<c10::BFloat16>::exists, void>::type "
           "vllm::fused_add_rms_norm_kernel<c10::BFloat16, 0, false>(c10::BFloat16*, long, float, int)")
    prof = "void vllm::fused_add_rms_norm_kernel<c10::BFloat16, 0, false>(c10::BFloat16*, long, float, int)"
    assert competitors._signature(cxx) == competitors._signature(prof) == "vllm::fused_add_rms_norm_kernel<c10::BFloat16,0,false>"
