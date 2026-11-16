"""Diversity: which programs a generation builds on, which named changes are close to an idea already had, and
which new programs repeat one already written."""
from __future__ import annotations

import hashlib
import io
import random
import re
import tokenize

from lab.kernels.search.program import Idea, Program

POOL = 32                    # the best usable programs parents are drawn from
FLOOR = 0.05                 # added to each draw weight, so the weakest of the pool can still be picked
CLOSE = 0.88                 # cosine of two ideas' texts at which they are close


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


class Novelty:
    """Every program written this run; `refuse` says why a new one repeats one of them."""

    def __init__(self):
        self.seen: set[str] = set()

    def add(self, code: str) -> None:
        self.seen.add(fingerprint(code))

    def refuse(self, code: str) -> str | None:
        """Why `code` repeats a program already written, else None (and it is remembered)."""
        if fingerprint(code) in self.seen:
            return "identical to a program already written (comments and whitespace aside)"
        self.add(code)
        return None


class Similar:
    """The named changes whose text is close to another idea's: in the same list, earlier in this search, or in an
    earlier run (`past`: the task's entries of the idea archive, embedded)."""

    def __init__(self, embedder=None, past=()):
        self.embedder, self.past = embedder, list(past)   # lab.records.search's; None: nothing is matched
        self.run: list[tuple[list, Idea]] = []            # this search's ideas, embedded

    def _vectors(self, plans: list[dict]) -> list:
        from lab.kernels.archive import text
        return self.embedder.passages([text(p) for p in plans])

    def match(self, named: list[dict]) -> list[dict]:
        """One row per close pair: {named: its index, like: an index of `named`, an Idea or an archive entry, where,
        cosine}."""
        from lab.records.search import _cosine
        if self.embedder is None or not named:
            return []
        vs, out = self._vectors(named), []
        for k, v in enumerate(vs):
            others = [("this list", j, w) for j, w in enumerate(vs[:k])] + [("this search", i, w) for w, i in self.run] \
                + [("an earlier run", e, e.vector) for e in self.past if e.vector is not None]
            out += [{"named": k, "like": x, "where": where, "cosine": round(c, 3)}
                    for where, x, w in others if (c := _cosine(v, w)) >= CLOSE]
        return out

    def add(self, ideas: list[Idea]) -> None:
        """A generation's ideas, to be matched against from now on."""
        if self.embedder is not None and ideas:
            self.run += zip(self._vectors([i.plan for i in ideas]), ideas)
