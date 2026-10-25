"""The fp8-KV paged decode task (kernels/tasks/paged_decode_fp8.py): fp8 pools at a unit scale, its reference against
the bf16 task's on the same values, each task reading only its own captured calls, and the capture env it asks for."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from kernels.tasks import load  # noqa: E402


def test_fp8_pools_and_the_reference_on_their_values():
    t, bf = load("paged_decode_fp8"), load("paged_decode")
    shape = (3, 8, 2, 16, 4, 9)
    q, k, v, table, lens = t.make_inputs(shape, torch.bfloat16, "cpu", torch.Generator().manual_seed(0))
    assert k.dtype == v.dtype == torch.float8_e4m3fn and q.dtype == torch.bfloat16
    got = t.reference(q, k, v, table, lens)
    want = bf.reference(q, k.to(torch.bfloat16), v.to(torch.bfloat16), table, lens)       # unit scale: same values
    assert torch.equal(got, want)
    flops, bytes_ = t.cost(shape, torch.bfloat16)
    assert bytes_ < bf.cost(shape, torch.bfloat16)[1] and flops == bf.cost(shape, torch.bfloat16)[0]


def test_each_task_reads_only_its_own_pools_calls_and_fp8_asks_for_fp8_kv():
    def call(dtype):
        return {"args": [{"shape": [3, 8, 16]}, {"shape": [10, 2, 4, 16], "dtype": dtype}, {"shape": [10, 2, 4, 16]},
                         {"shape": [3, 4]}, {"values": [9, 4, 1]}]}
    t, bf = load("paged_decode_fp8"), load("paged_decode")
    assert t.shape_of(call("float8_e4m3fn")) == (3, 8, 2, 16, 4, 9) and t.shape_of(call("bfloat16")) is None
    assert bf.shape_of(call("bfloat16")) == (3, 8, 2, 16, 4, 9) and bf.shape_of(call("float8_e4m3fn")) is None
    assert dict(t.capture_env) == {"KV_CACHE_DTYPE": "fp8"} and bf.capture_env == ()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
