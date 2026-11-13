"""The search's records: its settings, a judged program, and one model call's idea."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    generations: int = 10
    ideas: int = 24          # model calls per generation, spread evenly over its parents
    parents: int = 8
    workers: int = 8         # model calls at once
    seed: int = 0


@dataclass
class Program:
    """A program that took the full exam."""
    id: str
    parent: str | None
    code: str
    metrics: dict
    artifacts: dict
    generation: int
    plan: dict = field(default_factory=dict)     # the idea it was written from (empty for the start and the seeds)
    kind: str = ""                               # "competitor": a seed judged beside the start

    @property
    def score(self) -> float:
        return float(self.metrics.get("combined_score") or 0.0)

    @property
    def usable(self) -> bool:
        """Correct and scored: it can be a parent."""
        return self.metrics.get("correct") == 1.0 and self.score > 0


@dataclass
class Idea:
    """One model call: the change it named, the program it wrote and what became of it."""
    id: str
    parent: str
    generation: int
    plan: dict
    model: str | None
    usd: float
    code: str | None                             # None: the answer could not be applied, or repeats another program
    fate: str = ""
    child: Program | None = None
