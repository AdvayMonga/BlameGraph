"""System-1 screeners: score a judge's hypotheses before any code is written, so the beam measures the likeliest.
Each returns (one score per hypothesis, $ spent). Published evidence screens written code, not hypotheses (GPU
Forecasters: propose 4x, measure the top quarter, 4 of 6 tasks no worse); screening hypotheses is ours, so every
screener is an A/B arm, and the random slot of each beam keeps its labels unbiased for a learned replacement
(a fine-tuned local classifier once there are a few hundred labels).

- `roi`: MANTIS's ROI on the judge's own estimates (no model call).
- `haiku`: Claude Haiku 4.5 reads the same facts the judge read and gives, per hypothesis, the probability that the
  program written from it beats its parent beyond the timer's noise.
- `random`: uniform scores (the control)."""
from __future__ import annotations

import json
import random
import re
from typing import Callable

from kernels.search.loop import roi

HAIKU = """You score proposed changes to a GPU kernel program before any code is written. You are shown the program,
its measurements and a numbered list of proposed changes. For each, give the probability (0 to 1) that a program
written from it will be correct and faster than this one beyond the timer's noise. Reply with one JSON array of
numbers, one per change, in order, and nothing else."""


def parse(text: str, n: int) -> list[float] | None:
    m = re.search(r"\[.*?\]", text or "", re.S)
    try:
        v = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(v, list) or len(v) != n:
        return None
    try:
        return [min(max(float(x), 0.0), 1.0) for x in v]
    except (TypeError, ValueError):
        return None


def haiku(call: Callable[[str, str], tuple[str, float]]):
    """A screener over `call` (a Claude client: (system, user) -> (text, $)); unparsable replies fall back to ROI order."""
    def screen(system: str, facts: str, hs: list[dict]) -> tuple[list[float], float]:
        listing = "\n".join(f"{i}. {json.dumps(h)}" for i, h in enumerate(hs))
        text, usd = call(system + "\n\n" + HAIKU, f"{facts}\n\n# Proposed changes\n{listing}")
        scores = parse(text, len(hs))
        if scores is None:                       # no usable scores: the published triage instead
            return [roi(h, None) for h in hs], usd
        return scores, usd
    return screen


def uniform(seed: int = 0):
    rng = random.Random(seed)
    return lambda system, facts, hs: ([rng.random() for _ in hs], 0.0)


def make(name: str, seed: int = 0):
    """The screener named on the command line; None = MANTIS ROI inside the loop."""
    if name == "roi":
        return None
    if name == "random":
        return uniform(seed)
    if name == "haiku":
        from kernels.search.llm import Claude
        return haiku(Claude("claude-haiku-4-5", max_tokens=2000))
    raise ValueError(f"unknown screener {name!r}: roi, haiku or random")
