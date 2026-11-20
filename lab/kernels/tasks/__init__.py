"""Kernel tasks: what a candidate kernel must compute, on which shapes, against which reference.
Only tensors cross the candidate boundary; a task's `call` adds the scalars."""
from __future__ import annotations

import dataclasses
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
    cost: Callable                          # cost(shape, dtype, inputs=None) -> (flops, bytes) a call must do and move
    call: Callable = None                   # call(fn, *inputs) -> fn's output (adds scalars); default fn(*inputs)
    heldout_axes: tuple[int, ...] = ()      # shape axes whose largest values are held out as a region
    heldout_values: tuple = ()              # (axis, values) pairs: a shape with one of those values there is held out
    edge_shapes: Callable = None            # edge_shapes(shape) -> shapes; default: the checker's
    dtypes: tuple = (torch.bfloat16,)
    layouts: tuple = ("strided", "transposed")  # non-contiguous variants its callers pass
    graphs: bool = False                    # timed as a CUDA graph replay; it must capture
    scales: tuple | None = None            # value scalings checked beyond the task's own (None: the evaluator's)
    make_real: Callable = None              # make_real(shape, dtype, device, generator, ints) -> inputs
    error_budget: float | None = None       # regions: a case's atol is at least this times the baseline's own error
    baseline_fn: Callable | None = None     # the baseline when it is not an engine entry point
    watch: tuple[str, ...] = ()             # further modules the evaluator watches for patches
    start_program: Callable | None = None   # start_program() -> the starting program when the entry's module is not it
    capture_env: tuple = ()                 # (name, value) launch env its calls need to occur
    capture_fn: str | None = None           # the captured calls' `fn` to read shapes from; None: `entry`
    seen_default: tuple = field(default_factory=tuple)      # shapes to use with no capture (tests, toys)
    heldout_default: tuple = field(default_factory=tuple)
    tolerances: dict | None = None          # {dtype: [atol, rtol]} in place of the target's
    nonfinite: bool = True                  # NaN/Inf inputs must give the reference's NaN/Inf positions
    judged: Callable | None = None          # a program's outputs as they are compared (raises on a broken rule)

    def bind(self, fn: Callable) -> Callable:
        """fn as the checker and the timer call it: tensors in, the task's scalars added."""
        return (lambda *xs: self.call(fn, *xs)) if self.call else fn

    def with_ints(self, ints: dict | None) -> "Task":
        """This task with inputs built from captured int args where a shape has them ({json shape: ints})."""
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
    """Normal values made on `device` by a generator seeded from `g`."""
    d = torch.Generator(device=device).manual_seed(int(torch.randint(0, 2**62, (1,), generator=g)))
    return (torch.randn(size, generator=d, device=device) * scale).to(dtype)


def load(name: str, **region) -> Task:
    """The task `name`: a module here, `region:FIRST:LAST`, `kb:LEVEL:ID` or `fib:NAME`.
    `region` keywords: `samples`, `shared_weights`, `source`, `engine`; a region given neither samples nor a holder
    is `regions.light`."""
    if name.startswith("fib:"):
        from lab.kernels.tasks.flashinfer_bench import task
        return task(name, region.get("samples"))
    if name.startswith("kb:"):
        from lab.kernels.tasks.kernelbench import task
        return task(name, region.get("source"))
    if name.startswith("region:"):
        from lab.kernels.tasks import regions
        from lab.profile.capture import STEP
        if not region.keys() - {"engine"}:
            return regions.light(name, eng=regions.load_engine(region["engine"]) if region.get("engine") else None)
        if name == STEP:
            from lab.kernels.tasks import step
            return step.task(**{k: v for k, v in region.items() if k != "engine"})
        _, first, last = name.split(":")
        return regions.from_engine(first, last, **region)
    return importlib.import_module(f"lab.kernels.tasks.{name}").TASK


TASKS = ("toy_rmsnorm",)


def from_config(cfg: dict) -> Task:
    """The task an evaluation config names, with its captured ints."""
    region = {k: cfg[k] for k in ("samples", "shared_weights", "source", "engine") if cfg.get(k)} \
        if cfg["task"].startswith(("region:", "kb:", "fib:")) else {}
    return load(cfg["task"], **region).with_ints(cfg.get("ints"))
