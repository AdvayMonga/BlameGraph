"""The whole decode step as one region (kernels/step.py, lab.capture.step): a step captured from a fake megakernel,
the task built from it and a CPU holder (shapes, inputs, the weights passed uncloned), and, where the engine imports
(run with LAB_ENGINE_REPO set), the fp32 reference through the engine's stations on a tiny model and the starting
program built from the engine's decode_step.cu. The megakernel itself needs an H200: not run here."""
from __future__ import annotations

import importlib.util
import json
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kernels import step  # noqa: E402
from lab import capture  # noqa: E402

L, NB, NKV, BS, D, H, NQ, E, K, I, V = 2, 12, 2, 16, 16, 64, 4, 4, 2, 32, 50


def pools(g):
    return [SimpleNamespace(k=torch.randn(NB, NKV, BS, D, generator=g).bfloat16(),
                            v=torch.randn(NB, NKV, BS, D, generator=g).bfloat16()) for _ in range(L)]


def fake_megakernel(ps):
    """decode_step as the engine's is called: writes each layer's K/V for the new token, returns (logits, next)."""
    def decode_step(mdl, tokens, pos, bt):
        for p in ps:
            p.k[bt[torch.arange(len(pos)), pos // BS], :, pos % BS] = 1.0
            p.v[bt[torch.arange(len(pos)), pos // BS], :, pos % BS] = 2.0
        logits = torch.zeros(len(tokens), V, dtype=torch.bfloat16)
        return logits, logits.float().argmax(-1)
    mk = types.ModuleType("inference_server.models.megakernel")
    mk.decode_step = decode_step
    return mk


def capture_steps(tmp, monkeypatch):
    ps = pools(torch.Generator().manual_seed(0))
    mk = fake_megakernel(ps)
    monkeypatch.setitem(__import__("sys").modules, "inference_server.models.megakernel", mk)
    bt = torch.tensor([[7, 3, -1], [5, 9, 11]])
    with capture.step(ps, tmp / "caps"):
        mk.decode_step(None, torch.tensor([1, 2]), torch.tensor([20, 40]), bt)      # rows at 21 and 41 tokens
        mk.decode_step(None, torch.tensor([1, 2]), torch.tensor([21, 41]), bt)      # same shape: logged only
        mk.decode_step(None, torch.tensor([3]), torch.tensor([5]), bt[:1, :1])
    return tmp / "caps", ps


def test_a_captured_step_keeps_the_touched_blocks_of_every_layer_before_and_after(tmp_path, monkeypatch):
    caps, ps = capture_steps(tmp_path, monkeypatch)
    calls = capture.load(caps)
    assert [c["fn"] for c in calls] == [capture.STEP] * 3 and [bool(c.get("sample")) for c in calls] == [1, 0, 1]
    s = torch.load(caps / "samples" / calls[0]["sample"], weights_only=True)
    blocks = [3, 5, 7, 9, 11]                    # row 0: 2 blocks (21 tokens), row 1: 3 (41)
    assert s["reads"]["k_cache"].shape == (L, 5, NKV, BS, D)
    assert s["reads"]["bt"].tolist() == [[blocks.index(7), blocks.index(3), 0], [1, 3, 4]]
    assert (s["writes"]["v_cache"][:, blocks.index(3), :, 4] == 2).all()           # pos 20: block 1, slot 4
    assert not (s["reads"]["v_cache"][:, blocks.index(3), :, 4] == 2).all()


def held_step(tmp, cfg=None):
    from kernels.shared_weights import Holder
    g = torch.Generator().manual_seed(1)
    w = {}
    for i in range(L):
        w.update({f"{i}.w.input_norm": torch.ones(H), f"{i}.w.qkv": torch.randn((NQ + 2 * NKV) * D, H, generator=g) / 8,
                  f"{i}.w.q_norm": torch.ones(D), f"{i}.w.k_norm": torch.ones(D),
                  f"{i}.w.o_proj": torch.randn(H, NQ * D, generator=g) / 8, f"{i}.w.post_norm": torch.ones(H),
                  f"{i}.w.router": torch.randn(E, H, generator=g), f"{i}.w.gate_up": torch.randn(E, 2 * I, H, generator=g) / 8,
                  f"{i}.w.gate_up_scale": None, f"{i}.w.down": torch.randn(E, H, I, generator=g) / 8,
                  f"{i}.w.down_scale": None})
    w = {k: v.bfloat16() if v is not None else None for k, v in w.items()}
    w.update(embed=torch.randn(V, H, generator=g).bfloat16(), final_norm=torch.ones(H).bfloat16(),
             lm_head=torch.randn(V, H, generator=g).bfloat16(),
             inv_freq=1.0 / (1e6 ** (torch.arange(0, D, 2).float() / D)))
    cfg = cfg or SimpleNamespace(num_q_heads=NQ, num_kv_heads=NKV, head_dim=D, scale=D ** -0.5, eps=1e-6, num_experts=E,
                          top_k=K, norm_topk_prob=True)
    return Holder(w, tmp / "w.pkl", {"cfg": cfg, "n_layers": L})


def test_the_step_task_reads_its_captures_and_passes_the_weights_uncloned(tmp_path, monkeypatch):
    from kernels import shapes
    from kernels.check import _gen
    caps, _ = capture_steps(tmp_path, monkeypatch)
    h = held_step(tmp_path)
    t = step.task(str(caps), str(h.path))
    assert t.name == capture.STEP and set(t.seen_default) == {(2, 3), (1, 1)} and t.heldout_axes == (0, 1)
    assert shapes.from_captures(t, [caps]) == {(2, 3): 2, (1, 1): 1}
    x0, x1 = t.make_inputs((2, 3), torch.bfloat16, "cpu", _gen(0)), t.make_inputs((2, 3), torch.bfloat16, "cpu", _gen(1))
    assert [a.shape for a in x0] == [(2,), (2,), (2, 3), (L, 5, NKV, BS, D), (L, 5, NKV, BS, D)]
    assert all(torch.equal(a, b) for a, b in zip(x0[:3], x1[:3])) and not torch.equal(x0[3], x1[3])  # only K/V vary
    w = t.bind(lambda *xs: xs[-1])(*x0)
    assert len(w) > len(x0) and w["0.w.qkv"].data_ptr() == h.tensors["0.w.qkv"].data_ptr()   # the holder's memory
    flops, nbytes = t.cost((2, 3), torch.bfloat16, x0)
    assert flops == 0 and nbytes > 2 * x0[3].numel() * 2


def test_the_whole_step_does_not_ship():
    from kernels.evolve import ship
    with pytest.raises(ValueError, match="no hook for a whole-step override"):
        ship.region_files(SimpleNamespace(name=capture.STEP), "", "pkg", [], [], True)


engine = pytest.mark.skipif(importlib.util.find_spec("inference_server") is None,
                            reason="the engine does not import here (set LAB_ENGINE_REPO)")


@engine
def test_the_reference_runs_the_engines_stations_over_every_layer_in_fp32(tmp_path, monkeypatch):
    from inference_server.models.stations import LayerConfig
    caps, _ = capture_steps(tmp_path, monkeypatch)
    h = held_step(tmp_path)
    t = step.task(str(caps), str(h.path))
    from kernels.shared_weights import attach
    w = {**attach(h.path), "cfg": LayerConfig(NQ, NKV, D, D ** -0.5, 1e-6, E, K, True)}     # the engine's own config
    xs = t.make_inputs((2, 3), torch.bfloat16, "cpu", torch.Generator().manual_seed(0))
    logits, nxt, k, v = step.reference(w)(*xs)
    assert logits.shape == (2, V) and logits.dtype == torch.bfloat16 and nxt.tolist() == logits.float().argmax(-1).tolist()
    changed = (k != xs[3]).any(dim=(2, 4))                   # [layer, block, slot]
    bt = xs[2]
    want = torch.zeros_like(changed)
    for r, p in enumerate(xs[1].tolist()):
        want[:, bt[r, p // BS], p % BS] = True
    assert torch.equal(changed, want) and k.dtype == xs[3].dtype


@engine
def test_the_starting_program_is_the_engines_megakernel_built_inline_and_passes_the_screen():
    from pathlib import Path
    from kernels.evolve import core
    src = step.start_program()
    cu = (Path(importlib.util.find_spec(step.MODULE).origin).parent / "decode_step.cu").read_text()
    assert core.screen(src) == [] and "globaltimer" not in src and "load_inline(" in src
    assert cu.replace(*step.START_EDITS[0]) in src and f"ROW = {step.ROW!r}" in src


PROGRAM = '''# EVOLVE-BLOCK-START
import torch
from inference_server.models import stations as st
from inference_server.models.paged_kv_cache import step_inputs

ROW = ("w.input_norm", "w.qkv", "w.q_norm", "w.k_norm", "w.o_proj", "w.post_norm", "w.router", "w.gate_up",
       "w.gate_up_scale", "w.down", "w.down_scale")


def kernel(tokens, pos, bt, k_cache, v_cache, w):
    h = w["embed"][tokens.long()].float()[:, None, :]
    f = pos[:, None, None].float() * w["inv_freq"].float()[None, None, :]
    ctx = {**step_inputs(bt, pos, k_cache.shape[3]), "cos": torch.cat([f, f], -1).cos(),
           "sin": torch.cat([f, f], -1).sin(), "cfg": w["cfg"]}
    ks, vs = [], []
    for i in range(w["n_layers"]):
        lw = {k: w[f"{i}.{k}"] for k in ROW}
        lw = {k: x.float() if x is not None and k not in ("w.gate_up", "w.down") else x for k, x in lw.items()}
        c = {**ctx, **lw, "h": h, "k_cache": k_cache[i].float(), "v_cache": v_cache[i].float()}
        for s in st.reference():
            c.update(s.fn(c))
        h = c["h"]
        ks.append(c["k_cache"])
        vs.append(c["v_cache"])
    x = h[:, 0]
    n = w["final_norm"].float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + w["cfg"].eps))
    logits = n @ w["lm_head"].float().T
    return logits.to(torch.bfloat16), logits.argmax(-1), torch.stack(ks).to(k_cache.dtype), torch.stack(vs).to(v_cache.dtype)
# EVOLVE-BLOCK-END
'''


@engine
def test_the_step_is_judged_end_to_end_on_the_cpu(tmp_path, monkeypatch):
    from inference_server.models.stations import LayerConfig
    from kernels.evolve import core
    caps, _ = capture_steps(tmp_path, monkeypatch)
    h = held_step(tmp_path, LayerConfig(NQ, NKV, D, D ** -0.5, 1e-6, E, K, True))
    monkeypatch.setattr(step, "baseline", lambda *xs: step.reference(xs[-1])(*xs[:-1]))   # the megakernel needs CUDA
    t = step.task(str(caps), str(h.path))
    program = tmp_path / "candidate.py"
    program.write_text(PROGRAM)
    r = core.judge(t, program, [(2, 3), (1, 1)], [], None, "cpu", None, 2, iters=3)
    assert r["metrics"]["correct"] == 1.0, r["artifacts"]
    gates = json.loads(r["artifacts"]["gates"])
    assert gates["inputs_unchanged"]["passed"] and gates["memoization"]["passed"] and r["metrics"]["combined_score"] > 0
    assert h.changed() == []
