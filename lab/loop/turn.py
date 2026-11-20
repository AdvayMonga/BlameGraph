"""One turn: one gap and the current best in, one candidate out. A way is how a candidate gets made."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from lab.budget import Budget
from lab.loop.types import Candidate, Gap

Way = Callable[[Gap, Candidate, Path, Budget], "Candidate | None"]
KERNEL, CHANGE = "kernels", "change"


def turn(gap: Gap, best: Candidate, work: Path, budget: Budget, ways: dict[str, Way]) -> Candidate | None:
    """A candidate built on `best` in `work`: `ways[KERNEL]` for the kernels layer's gap, `ways[CHANGE]` for any
    other; None when the way made none. A way adds the model dollars it spent to `budget`."""
    return ways[KERNEL if gap.layer == KERNEL else CHANGE](gap, best, work, budget)
