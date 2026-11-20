"""The model's choice of the gap a turn works on."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Callable

from lab.budget import Budget
from lab.loop.types import Gap, Measurement

SYSTEM = "You are given an inference engine's measured gaps. Reply with the `layer` of the one gap to work on next, and nothing else."


def pick(gaps: tuple[Gap, ...], best: Measurement, budget: Budget, ask: Callable[[str, str], tuple[str, float]]) -> Gap:
    """The gap whose layer `ask(system, user) -> (text, usd)` names; the first gap when it names none."""
    text, usd = ask(SYSTEM, json.dumps({"gaps": [asdict(g) for g in gaps], "metrics": best.metrics,
                                        "layers": best.profile.get("layers")}, indent=1, default=str))
    budget.spent_usd += usd
    return next((g for g in gaps if g.layer == text.strip().strip("`\"'")), gaps[0])
