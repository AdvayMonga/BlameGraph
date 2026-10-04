"""Correctness gate: is the optimized server still the same model (within the allowed quality budget)?

Checks, all against a reference (the unmodified model in BF16, batch-invariant mode):
  divergence   teacher-forced per-token distributions on reference text (KL, top-1 agreement, ref-token logprob delta)
  flips        paired right/wrong answers on task sets; McNemar test for degradation plus a flip-rate cap
  length       outputs are not shorter than the reference (no truncation / early EOS)
  consistency  reported tokens match the returned text (first token, token count, nothing after EOS)
`gate.evaluate()` combines them into one verdict shaped like a lab-ledger `equiv` record.
"""
from .checks import consistency, length_ratio
from .divergence import Position, compare_positions, divergence_summary
from .flips import flip_test
from .gate import Thresholds, calibrate, evaluate

__all__ = ["Position", "compare_positions", "divergence_summary", "flip_test", "length_ratio", "consistency",
           "Thresholds", "calibrate", "evaluate"]
