"""Kernel tasks: what a candidate kernel must compute, on which shapes, against which reference.

A task names the engine function it can replace (`entry`, the baseline), a reference implementation, an input
builder, how to read a shape from a captured call (lab/capture.py), how to split shapes into seen and held-out
regions, and the bytes and FLOPs a call must move (the roofline bound). Tensors only cross the candidate boundary;
scalars a call needs (e.g. an attention scale) are derived from the shapes inside `call`."""
from __future__ import annotations

import importlib
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
    seen_default: tuple = field(default_factory=tuple)      # shapes to use with no capture (tests, toys)
    heldout_default: tuple = field(default_factory=tuple)

    def bind(self, fn: Callable) -> Callable:
        """fn as the checker and the timer call it: tensors in, the task's scalars added."""
        return (lambda *xs: self.call(fn, *xs)) if self.call else fn

    def baseline(self) -> Callable | None:
        if not self.entry:
            return None
        mod, _, attr = self.entry.partition(":")
        return getattr(importlib.import_module(mod), attr)


def load(name: str) -> Task:
    return importlib.import_module(f"kernels.tasks.{name}").TASK


TASKS = ("toy_rmsnorm", "paged_decode")
