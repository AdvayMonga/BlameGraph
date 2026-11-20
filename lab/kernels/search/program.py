"""The search's records: its settings, a judged program, and one idea."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    generations: int = 4
    ideas: int = 24          # changes the naming call names per generation
    merges: int = 4          # combined changes the review call may add to them
    ports: int = 4           # public implementations ported as starting programs before the first generation
    parents: int = 8
    workers: int = 8         # writer calls at once
    seed: int = 0


@dataclass
class Program:
    """A program that was built and timed."""
    id: str
    parent: str | None
    code: str
    metrics: dict
    artifacts: dict
    generation: int
    plan: dict = field(default_factory=dict)     # the idea it was written from (empty for the start)
    kind: str = ""                               # "competitor": a port, judged beside the start

    @property
    def score(self) -> float:
        return float(self.metrics.get("combined_score") or 0.0)

    @property
    def usable(self) -> bool:
        """Correct and scored: it can be a parent."""
        return self.metrics.get("correct") == 1.0 and self.score > 0


@dataclass
class Idea:
    """One change of a generation's final list: what was named, the program a writer made of it, what became of it."""
    id: str
    parent: str
    generation: int
    plan: dict                                   # the change and its prediction, as the review call wrote it
    merged: list = field(default_factory=list)   # the named changes (n1, n2, ...) it keeps or combines
    calls: dict = field(default_factory=dict)    # role -> the model call (calls.jsonl) that produced it
    model: str | None = None                     # the writer
    program: str | None = None                   # the program the writer made, by its key in texts.jsonl
    code: str | None = None                      # None: the answer could not be applied, or repeats another program
    fate: str = ""
    child: Program | None = None
    reflection: str = ""                         # the next naming call's, on its prediction against what was measured
