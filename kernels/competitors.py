"""Other implementations of our kernel tasks, timed on our own shapes beside the engine's kernel, and their compiled
code kept for the PTX comparison's cards (kernels/ptx.py). Run on the GPU host in a venv that has the competitors'
libraries (vLLM, FlashInfer) beside this repo and the engine's source:

  uv venv ~/comp && uv pip install --python ~/comp/bin/python vllm flashinfer-python ninja   (FlashInfer JIT-builds;
  ~/comp/bin on PATH)
  PYTHONPATH=~/BlameGraph:~/inference-server/src ~/comp/bin/python -m kernels.competitors \\
      --task rmsnorm --shapes shapes-rmsnorm.json --out DIR

Each competitor is first checked against the task's reference on every shape (a wrong one is reported, never timed),
then timed in pairs with the engine's kernel on the same inputs (kernels.time_pair). DIR gets `timings.json` and one
text file of compiled code per implementation: the engine's Triton kernels' PTX from Triton's cache, a library's SASS
from `cuobjdump -sass` of its extension, filtered to the kernels that ran."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import torch

from kernels.check import TOLERANCES, _compare

ADAPTERS: dict[str, dict[str, Callable]] = {}       # task -> {competitor: make(inputs...) -> outputs}
SASS_NAMES: dict[str, str] = {}                     # competitor -> regex of its device kernels' names


def adapter(task: str, name: str, sass: str):
    def reg(fn):
        ADAPTERS.setdefault(task, {})[name] = fn
        SASS_NAMES[name] = sass
        return fn
    return reg


EPS = 1e-6


def _vllm(*mods: str):
    """vLLM's custom ops register when their extension modules are imported."""
    import importlib
    for m in mods:
        importlib.import_module(m)


@adapter("rmsnorm", "vllm.rms_norm", r"rms_norm_kernel")
def _vllm_rms_norm(x, w):
    _vllm("vllm._C")
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x, w, EPS)
    return out


@adapter("add_rmsnorm", "vllm.fused_add_rms_norm", r"fused_add_rms_norm_kernel")
def _vllm_fused_add_rms_norm(h, a, w):
    _vllm("vllm._C")
    res, x = h.clone(), a.clone()                   # in place: res becomes h + a, x its norm
    torch.ops._C.fused_add_rms_norm(x, res, w, EPS)
    return res, x


@adapter("silu_mul", "vllm.silu_and_mul", r"act_and_mul_kernel")
def _vllm_silu_and_mul(x):
    _vllm("vllm._C")
    out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


@adapter("topk_softmax", "vllm.topk_softmax", r"topkGating|moeSoftmax|topk")
def _vllm_topk_softmax(logits):
    from kernels.tasks._fused import TOP_K
    _vllm("vllm._moe_C")
    m = logits.shape[0]
    w = torch.empty((m, TOP_K), dtype=torch.float32, device=logits.device)
    ids = torch.empty((m, TOP_K), dtype=torch.int32, device=logits.device)
    tok = torch.empty((m, TOP_K), dtype=torch.int32, device=logits.device)
    try:
        torch.ops._moe_C.topk_softmax(w, ids, tok, logits, True)            # renormalize in the kernel (newer vLLM)
    except (RuntimeError, TypeError):
        torch.ops._moe_C.topk_softmax(w, ids, tok, logits)
        w = w / w.sum(dim=-1, keepdim=True)
    return w, ids.long()


@adapter("paged_decode", "flashinfer.batch_decode", r"BatchDecode|batch_decode|SingleDecode")
def _flashinfer_decode(q, k_pool, v_pool, block_tables, seq_lens):
    """FlashInfer's paged decode on our pool (its HND layout: [pages, kv heads, page size, head dim])."""
    import flashinfer
    n, hq, d = q.shape
    hkv, bs = k_pool.shape[1], k_pool.shape[2]
    pages = (seq_lens + bs - 1) // bs
    indptr = torch.zeros(n + 1, dtype=torch.int32, device=q.device)
    indptr[1:] = torch.cumsum(pages, 0)
    indices = torch.cat([block_tables[i, :int(pages[i])] for i in range(n)]).to(torch.int32)
    last = (seq_lens - (pages - 1) * bs).to(torch.int32)
    key = (block_tables.data_ptr(), seq_lens.data_ptr(), n, hq, hkv, d, bs)    # planned once per input set: the plan
    if key not in _FI:                                                          # is host work, not the kernel
        wrap = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(128 * 2**20, dtype=torch.uint8, device=q.device), kv_layout="HND")
        wrap.plan(indptr, indices, last, hq, hkv, d, bs, q_data_type=q.dtype, kv_data_type=k_pool.dtype)
        _FI[key] = wrap
    return _FI[key].run(q, (k_pool, v_pool))


_FI: dict = {}


def check(task, fn, shapes, device) -> str | None:
    """Why `fn` is not a correct implementation of the task on these shapes, or None."""
    from kernels.check import _gen
    dt = task.dtypes[0]
    atol, rtol = TOLERANCES[dt]
    for s in shapes:
        xs = task.make_inputs(tuple(s), dt, device, _gen(0))
        try:
            d = _compare(task.reference(*[x.clone() for x in xs]), fn(*[x.clone() for x in xs]), atol, rtol)
        except Exception as e:                   # noqa: BLE001  a signature that differs in this library version
            return f"raised {type(e).__name__}: {str(e)[:300]}"
        if not d["ok"]:
            return f"differs from the reference at shape {list(s)}: {d.get('error') or d}"
    return None


def sass(module_file: str, pattern: str) -> str:
    """The SASS of the functions in a library's extension whose names match `pattern`."""
    exe = shutil.which("cuobjdump") or str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "cuobjdump")
    p = subprocess.run([exe, "-sass", module_file], capture_output=True, text=True)
    blocks = re.split(r"(?=\n\s+Function : )", p.stdout)
    return "".join(b for b in blocks if re.search(r"Function : \S*(" + pattern + r")", b))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernels.competitors")
    ap.add_argument("--task", required=True)
    ap.add_argument("--shapes", required=True, help="a shapes file (kernels.shapes.write): its seen shapes are timed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args(argv)
    cache = Path(tempfile.mkdtemp(prefix="triton-cache-"))
    os.environ["TRITON_CACHE_DIR"] = str(cache)                    # the engine's Triton kernels' PTX lands here
    from kernels import shapes as shapes_mod
    from kernels.tasks import load, options_from_target
    from kernels.timing import time_pair
    task = load(a.task)
    task = task.with_options(options_from_target(task)) if task.options else task
    sets = shapes_mod.read(a.shapes)
    seen = [tuple(s) for s in sets["seen"]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ours = task.bind(task.baseline())
    rows = {"task": a.task, "shapes": [list(s) for s in seen], "competitors": {}}
    for name, fn in (ADAPTERS.get(a.task) or {}).items():
        why = check(task, fn, seen, "cuda")
        if why:
            rows["competitors"][name] = {"not_timed": why}
            continue
        per = []
        for s in seen:
            p = time_pair(fn, ours, task.make_inputs, s, task.dtypes[0], "cuda", iters=a.iters)
            per.append({"shape": list(s), "competitor_ms": p["a"]["median_ms"], "ours_ms": p["b"]["median_ms"],
                        "ours_over_competitor": p["ratio"]["median"]})
        rows["competitors"][name] = {"per_shape": per}
        mod = {"vllm": "vllm._moe_C" if "topk" in name else "vllm._C", "flashinfer": "flashinfer"}[name.split(".")[0]]
        try:
            text = sass(sys.modules[mod].__file__ if mod in sys.modules else __import__(mod).__file__, SASS_NAMES[name])
        except Exception as e:                   # noqa: BLE001
            text = ""
            rows["competitors"][name]["sass_missing"] = f"{type(e).__name__}: {e}"
        (out / f"{name}.sass").write_text(text)
    (out / "ours.ptx").write_text("".join(p.read_text() for p in sorted(cache.rglob("*.ptx"))))
    (out / "timings.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps({"task": a.task, "competitors": {k: ("timed" if "per_shape" in v else v["not_timed"][:80])
                                                      for k, v in rows["competitors"].items()}, "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
