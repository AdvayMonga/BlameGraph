"""Kernel equivalence: does a candidate kernel compute the reference op, without side effects, on any shape?

`check_kernel` gates: correctness on seen and held-out shapes, edge cases (size-1, non-power-of-two, non-contiguous,
NaN/Inf), memoization (inputs overwritten in place), inputs unchanged, no aliasing, determinism (reported; gated
only if required). `time_kernel` times honestly; `check_memoization` compares repeated vs fresh inputs.
"""
from .check import TOLERANCES, check_kernel
from .timing import check_memoization, time_kernel

__all__ = ["TOLERANCES", "check_kernel", "check_memoization", "time_kernel"]
