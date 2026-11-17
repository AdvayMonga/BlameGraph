"""Correctness gate: is the optimized server still the same model as the reference?
Checks: divergence, length, consistency, and per-task accuracy as a fact; `gate.evaluate()` gives the verdict."""
from .checks import consistency, length_ratio
from .divergence import Position, compare_positions, divergence_summary
from .flips import flip_test
from .gate import Thresholds, evaluate, task_accuracy

__all__ = ["Position", "compare_positions", "divergence_summary", "flip_test", "length_ratio", "consistency",
           "Thresholds", "evaluate", "task_accuracy"]
