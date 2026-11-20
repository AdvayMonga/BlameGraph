"""What the parts of a run pass to each other."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lab.infra.context.peaks import Peaks


@dataclass(frozen=True)
class Candidate:
    tree: Path                  # the engine tree to serve
    patch: str = ""             # its diff from the tree it was made from
    note: str = ""              # the change in one line


@dataclass(frozen=True)
class Measurement:
    correct: bool                                     # the gate passed
    equiv: dict                                       # the gate's verdict, reasons, metrics and thresholds
    metrics: dict = field(default_factory=dict)       # {regime: headline}; empty when the gate failed
    calibrated: dict = field(default_factory=dict)    # {regime: the value against its ceiling}
    profile: dict = field(default_factory=dict)       # bundle (ledger blob), layers, kernels; or error
    artifacts: dict = field(default_factory=dict)     # the bench's passive data as ledger blobs
    gpu: dict | None = None
    seconds: float = 0.0


@dataclass(frozen=True)
class Calibration:
    regimes: tuple[str, ...]
    bands: dict                 # {regime: noise band %}
    reference: Path             # the reference answers the gate compares against
    peaks: Peaks
    commit: str                 # the engine commit measured
    baseline: Measurement


@dataclass(frozen=True)
class Gap:
    layer: str                          # kernels | model_execution | runtime | serving
    pct_of_time: float                  # its share of the profiled window
    pct_of_roofline: float | None       # 100 = at its ceiling; None where the layer has none
    recoverable_pct: float              # % of the window gone at its ceiling; its whole share without one
