"""The search tree: every evaluated program is a node, kept whether it won, lost or failed. Parents are chosen by
Monte Carlo tree search as KernelPro does (UCT, C = sqrt(2) on its scale, progressive widening alpha = 0.5: a node takes another
expansion while its children number fewer than visits^alpha; KernelPro: MCTS 4.60x vs greedy 3.65x, p = 0.004). A
node closes after `max_failures` judged children that failed, or as many edits that never reached the GPU (unapplied,
duplicate), counted apart (ours; KernelPro marks a node terminal after 3 failed repairs), so search leaves dead ends
without forgetting them, and a node whose edits keep missing cannot spend the model budget."""
from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass, field


@dataclass
class Node:
    id: str
    parent: str | None
    code: str
    metrics: dict
    artifacts: dict
    iteration: int
    proposal: str = ""                       # the model text that produced it (empty for the root)
    judge: str = ""                          # the judge's plan it was written from (empty without a judge)
    mode: str = "optimization"               # kept in the trace; "correction" only in runs before 2026-10-27
    visits: int = 0                          # MCTS: judged programs in its subtree, itself included
    children: list[str] = field(default_factory=list)
    failures: int = 0                        # judged children that failed
    misses: int = 0                          # edits that never reached the GPU

    @property
    def score(self) -> float:
        return float(self.metrics.get("combined_score") or 0.0)

    @property
    def pct_of_roofline(self) -> float | None:
        """Call-weighted % of the roofline bound over the seen shapes it was timed on (None if untimed)."""
        import json
        rows = [r for r in json.loads(self.artifacts.get("timings") or "[]") if r.get("pct_of_roofline") is not None]
        w = sum(r.get("weight") or 0 for r in rows)
        return sum((r.get("weight") or 0) * r["pct_of_roofline"] for r in rows) / w if rows and w else None


def fingerprint(code: str) -> str:
    """Hash of the program with its Python comments and whitespace removed: two proposals that differ only there are
    one. Strings are kept whole, so a C++ `#define` or `#pragma` inside one still counts."""
    import io
    import tokenize
    try:
        body = " ".join(t.string for t in tokenize.generate_tokens(io.StringIO(code).readline)
                        if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                                          tokenize.DEDENT))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        body = code
    return hashlib.sha256(re.sub(r"\s+", "", body).encode()).hexdigest()[:16]


UCT_C = 0.6      # KernelPro's sqrt 2 over its ln(speedup) rewards (about 2.3 wide for 0.5x-5x), scaled to ours in [0, 1]


class Tree:
    def __init__(self, seed: int = 0, c: float = UCT_C, alpha: float = 0.5, max_failures: int = 3):
        self.nodes: dict[str, Node] = {}
        self.seen: set[str] = set()
        self.rng, self.c, self.alpha, self.max_failures = random.Random(seed), c, alpha, max_failures

    def add(self, node: Node) -> Node:
        """Add a node; a judged one (stage > 0) also backs its reward up its lineage."""
        self.nodes[node.id] = node
        self.seen.add(fingerprint(node.code))
        if node.parent:
            p = self.nodes[node.parent]
            p.children.append(node.id)
            if node.score <= 0 and self.judged(node):
                p.failures += 1
            elif not self.judged(node):
                p.misses += 1
        if self.judged(node):
            for n in self.lineage(node):
                n.visits += 1
        return node

    def root(self) -> Node:
        return next(n for n in self.nodes.values() if n.parent is None)

    @staticmethod
    def judged(node: Node) -> bool:
        return bool(node.metrics.get("stage", 1))

    def reward(self, node: Node) -> float:
        """The node's percentile among every judged program in the tree (ties split), so rewards keep spreading as the
        tree improves. A reward relative to the root clipped to 1 at 2x, the 2026-10-18 A/B's, saturated for every
        child once the start was 0.14x of the engine's kernel, and UCT then only counted visits."""
        others = [n for n in self.nodes.values() if n is not node and self.judged(n)]
        if not others:
            return 0.5
        below = sum(n.score < node.score for n in others) + 0.5 * sum(n.score == node.score for n in others)
        return below / len(others)

    def value(self, node: Node) -> float:
        """The best reward among the judged programs in the node's subtree (max backup), recomputed at every
        selection: only the best program found is kept, so a branch is worth its best. The mean, as KernelPro backs
        up, hid the 2026-10-19 run's best program (1.42x) under its 0.86x parent: never expanded in 13 programs."""
        rs = [self.reward(n) for n in self.subtree(node) if self.judged(n)]
        return max(rs) if rs else 0.0

    def subtree(self, node: Node) -> list[Node]:
        out, stack = [], [node]
        while stack:
            n = stack.pop()
            out.append(n)
            stack += [self.nodes[c] for c in n.children]
        return out

    def expandable(self, n: Node) -> bool:
        """Correct and scored (a correct program zeroed by the memory, timing or physics gate is not a parent)."""
        return (n.metrics.get("correct") == 1.0 and n.score > 0 and n.failures < self.max_failures
                and n.misses < self.max_failures)

    def _open_below(self, n: Node) -> bool:
        return self.expandable(n) or any(self._open_below(self.nodes[c]) for c in n.children)

    def open(self) -> list[Node]:
        return [n for n in self.nodes.values() if self.expandable(n)]

    def uct(self, parent: Node, visits: int, value: float) -> float:
        return value + self.c * math.sqrt(math.log(max(parent.visits, 1)) / max(visits, 1))

    def select(self) -> Node | None:
        """Descend from the root by UCT. A node under its widening cap takes a new child only when KernelPro's virtual
        child (the node's own reward as its prior, a fair share of the node's visits) has a higher UCT than its best
        existing child; otherwise the search descends. Widening whenever under the cap (the 2026-10-18 A/Bs) kept
        every tree at depth 2: the root and its weak children took new children while the best nodes waited."""
        node = self.root()
        if not self._open_below(node):
            return None
        while True:
            kids = [self.nodes[c] for c in node.children if self.nodes[c].visits and self._open_below(self.nodes[c])]
            if not kids:
                return node if self.expandable(node) else None
            best = max(kids, key=lambda k: self.uct(node, k.visits, self.value(k)))
            if self.expandable(node) and len(node.children) < math.ceil(max(node.visits, 1) ** self.alpha):
                virtual = self.uct(node, node.visits // (len(node.children) + 1), self.reward(node))
                if virtual > self.uct(node, best.visits, self.value(best)):
                    return node
            node = best

    def best(self) -> Node:
        return max(self.nodes.values(), key=lambda n: (n.score, -n.iteration))

    def lineage(self, node: Node) -> list[Node]:
        out = []
        while node:
            out.append(node)
            node = self.nodes.get(node.parent) if node.parent else None
        return out[::-1]
