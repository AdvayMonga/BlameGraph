"""Diversity: which programs a generation builds on, and which new programs repeat one already written."""
from __future__ import annotations

import difflib
import hashlib
import io
import random
import re
import tokenize

from generate.kernels.search.program import Program

POOL = 32                    # the best usable programs parents are drawn from
FLOOR = 0.05                 # added to each draw weight, so the weakest of the pool can still be picked
NEAR = 0.95                  # cosine of two siblings' changes at which the later one is a near-duplicate


def pick(population: list[Program], n: int, rng: random.Random) -> list[Program]:
    """The best usable program, then up to n - 1 more drawn without replacement, weighted by percentile score."""
    pool = sorted((p for p in population if p.usable), key=lambda p: (-p.score, p.generation))[:POOL]
    out, rest = pool[:1], pool[1:]
    while rest and len(out) < n:
        weights = [sum(q.score < p.score for q in population) / len(population) + FLOOR for p in rest]
        out.append(rest.pop(rng.choices(range(len(rest)), weights=weights)[0]))
    return out


def fingerprint(code: str) -> str:
    """Hash of the program without its Python comments and whitespace; strings are kept whole."""
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
    try:
        body = " ".join(t.string for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type not in skip)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        body = code
    return hashlib.sha256(re.sub(r"\s+", "", body).encode()).hexdigest()[:16]


def change(base: str, code: str) -> str:
    """The lines `code` adds to or removes from `base`."""
    return "".join(x for x in difflib.unified_diff(base.splitlines(True), code.splitlines(True), n=0)
                   if x[:1] in "+-" and not x.startswith(("+++", "---")))


class Novelty:
    """Every program written this run; `refuse` says why a new one repeats one of them."""

    def __init__(self, embedder=None):
        self.embedder = embedder                 # data.knowledge.search's; None: identical programs only
        self.seen: set[str] = set()
        self.changes: dict[str, list] = {}       # parent id -> the vectors of its children's changes

    def add(self, code: str) -> None:
        self.seen.add(fingerprint(code))

    def refuse(self, code: str, parent: Program) -> str | None:
        """Why `code` repeats a program already written, else None (and it is remembered)."""
        from data.knowledge.search import _cosine
        if fingerprint(code) in self.seen:
            return "identical to a program already written (comments and whitespace aside)"
        if self.embedder is not None:
            v = self.embedder.passages([change(parent.code, code)])[0]
            near = max((_cosine(v, w) for w in self.changes.get(parent.id, [])), default=0.0)
            if near >= NEAR:
                return f"a near-duplicate of another change to the same parent (cosine {near:.3f})"
            self.changes.setdefault(parent.id, []).append(v)
        self.add(code)
        return None
