"""Correctness gate: is the optimized server still the same model as the reference?
Checks: accuracy, divergence, length, consistency; `gate.evaluate()` combines them into one verdict."""
from .checks import consistency, length_ratio
from .divergence import Position, compare_positions, divergence_summary
from .flips import flip_test
from .gate import Thresholds, evaluate, pooled_accuracy

__all__ = ["Position", "compare_positions", "divergence_summary", "flip_test", "length_ratio", "consistency",
           "Thresholds", "evaluate", "pooled_accuracy"]
