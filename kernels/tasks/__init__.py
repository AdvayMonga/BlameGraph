"""Kernel tasks: what a candidate kernel must compute, on which shapes, against which reference.

A task names the engine function it can replace (`entry`, the baseline), a reference implementation, an input
builder, how to read a shape from a captured call (lab/capture.py), how to split shapes into seen and held-out
regions, and the bytes and FLOPs a call must move (the roofline bound). Tensors only cross the candidate boundary;
scalars a call needs (e.g. an attention scale) are derived from the shapes inside `call`."""
from __future__ import annotations

import dataclasses
import functools
import importlib
import json
from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass(frozen=True)
class Task:
    name: str
    entry: str | None                       # "module:function" in the engine, the baseline; None for a toy
    reference: Callable                     # reference(*inputs) -> output tensor(s)
    make_inputs: Callable                   # make_inputs(shape, dtype, device, generator) -> tuple of tensors
    shape_of: Callable                      # shape_of(captured call dict) -> shape tuple, or None to skip the call
    cost: Callable                          # cost(shape, dtype, inputs=None) -> (flops, bytes) a call must do and
                                            # move; with inputs, for those exact inputs (e.g. real sequence lengths)
    call: Callable = None                   # call(fn, *inputs) -> fn's output (adds scalars); default fn(*inputs)
    heldout_axes: tuple[int, ...] = ()      # shape axes whose largest values are held out as a region
    edge_shapes: Callable = None            # edge_shapes(shape) -> shapes; default: the checker's
    dtypes: tuple = (torch.bfloat16,)
    layouts: tuple = ("strided", "transposed")  # non-contiguous variants its callers pass (lab/capture records it)
    graphs: bool = False                    # its callers replay it inside CUDA graphs: it must capture, and is timed
                                            # as a graph replay (no host launch overhead)
    scales: tuple | None = None            # value scalings checked beyond the task's own (None: the evaluator's; ties in
                                            # bf16 make ×0.01 meaningless for an exact top-k)
    size_expr: str | None = None            # the call's size along heldout_axes[0], as Python over the entry's `args`,
                                            # from tensor shapes only (no device sync): per-region dispatch reads it
    kernel_names: tuple[str, ...] = ()      # regexes of the device kernels its entry launches (kernels/share.py)
    make_real: Callable = None              # make_real(shape, dtype, device, generator, ints) -> inputs, `ints` being a
                                            # captured call's 1-D int args by value (e.g. its real sequence lengths)
    start_edits: tuple = ()                 # (old, new) exact edits to the engine module before it is the starting
                                            # program: same behaviour, spelled so the source screen accepts it
    options: tuple[str, ...] = ()           # deployment facts make_inputs/make_real take as keywords (from the
                                            # target spec: `options_from_target`), e.g. the engine's context window
    error_budget: float | None = None       # regions: a case's atol is at least this times the baseline's own error
    baseline_fn: Callable | None = None     # the baseline when it is not an engine entry point (a region's stations)
    watch: tuple[str, ...] = ()             # further modules the evaluator watches for patches (a region's stations)
    seen_default: tuple = field(default_factory=tuple)      # shapes to use with no capture (tests, toys)
    heldout_default: tuple = field(default_factory=tuple)

    def bind(self, fn: Callable) -> Callable:
        """fn as the checker and the timer call it: tensors in, the task's scalars added."""
        return (lambda *xs: self.call(fn, *xs)) if self.call else fn

    def with_options(self, opts: dict | None) -> "Task":
        """This task with its declared options (e.g. context_window) passed to make_inputs and make_real."""
        kw = {k: v for k, v in (opts or {}).items() if k in self.options}
        if not kw:
            return self
        real = self.make_real and functools.partial(self.make_real, **kw)
        return dataclasses.replace(self, make_inputs=functools.partial(self.make_inputs, **kw), make_real=real)

    def with_ints(self, ints: dict | None) -> "Task":
        """This task with inputs built from captured int args where a shape has them ({json shape: ints}); random
        values elsewhere (edge shapes, scaled variants keep the captured ints)."""
        if not ints or not self.make_real:
            return self
        synthetic = self.make_inputs

        def make(shape, dtype, device, g):
            got = ints.get(json.dumps(list(shape)))
            return self.make_real(shape, dtype, device, g, got) if got else synthetic(shape, dtype, device, g)
        return dataclasses.replace(self, make_inputs=make)

    def baseline(self) -> Callable | None:
        if self.baseline_fn is not None:
            return self.baseline_fn
        if not self.entry:
            return None
        mod, _, attr = self.entry.partition(":")
        return getattr(importlib.import_module(mod), attr)


def randn(size, g, device, dtype, scale: float = 1.0) -> torch.Tensor:
    """Normal values made on `device` by a generator seeded from `g` (the checker's seeded CPU generator): the same
    values for the same seed on a device, and fast for the large tensors (expert weights, KV pools) that took seconds
    each to generate on the CPU."""
    d = torch.Generator(device=device).manual_seed(int(torch.randint(0, 2**62, (1,), generator=g)))
    return (torch.randn(size, generator=d, device=device) * scale).to(dtype)


def load(name: str) -> Task:
    return importlib.import_module(f"kernels.tasks.{name}").TASK


TASKS = ("toy_rmsnorm", "paged_decode", "moe_gemm", "moe_gemm_fp8", "moe_gemm_scatter_fp8", "rmsnorm", "add_rmsnorm",
         "rmsnorm_rope", "silu_mul", "topk_softmax")


def from_config(cfg: dict) -> Task:
    """The task an evaluation config names, with its deployment options and captured ints: what every judge uses."""
    return load(cfg["task"]).with_options(cfg.get("options")).with_ints(cfg.get("ints"))


def options_from_target(task: Task) -> dict:
    """The task's declared options, read from the target spec's engine env (context_window <- CONTEXT_WINDOW)."""
    if not task.options:
        return {}
    from lab import target
    env = target.load().engine.env
    return {"context_window": int(env["CONTEXT_WINDOW"])} if "context_window" in task.options and \
        "CONTEXT_WINDOW" in env else {}
