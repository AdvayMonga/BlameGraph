"""The search tree: every evaluated program is a node, kept whether it won, lost or failed. Parents are chosen by
Monte Carlo tree search as KernelPro does (UCT, C = sqrt(2), progressive widening alpha = 0.5: a node takes another
expansion while its children number fewer than visits^alpha; KernelPro: MCTS 4.60x vs greedy 3.65x, p = 0.004). A
node closes after `max_failures` failed children (KernelPro marks one terminal after 3), so search leaves dead ends
without forgetting them."""
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
    mode: str = "optimization"               # or "correction": a repair of a failed parent
    repairs: int = 0                         # repairs in the chain that led here (a repair of a repair counts 2)
    visits: int = 0                          # MCTS: judged programs in its subtree, itself included
    value: float = 0.0                       # MCTS: sum of their rewards
    children: list[str] = field(default_factory=list)
    failures: int = 0

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
    """Hash of the program with comments and whitespace removed: two proposals that differ only there are one."""
    body = re.sub(r"#[^\n]*", "", code)
    return hashlib.sha256(re.sub(r"\s+", "", body).encode()).hexdigest()[:16]


class Tree:
    def __init__(self, seed: int = 0, c: float = math.sqrt(2), alpha: float = 0.5, max_failures: int = 3):
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
            if node.score <= 0:
                p.failures += 1
        if node.metrics.get("stage", 1):
            r = self.reward(node)
            for n in self.lineage(node):
                n.visits, n.value = n.visits + 1, n.value + r
        return node

    def root(self) -> Node:
        return next(n for n in self.nodes.values() if n.parent is None)

    def reward(self, node: Node) -> float:
        """Score relative to the root (the engine's kernel), clipped to [0, 2] and halved into [0, 1]."""
        base = self.root().score or 1.0
        return min(node.score / base, 2.0) / 2

    def expandable(self, n: Node) -> bool:
        return n.metrics.get("correct") == 1.0 and n.failures < self.max_failures

    def _open_below(self, n: Node) -> bool:
        return self.expandable(n) or any(self._open_below(self.nodes[c]) for c in n.children)

    def open(self) -> list[Node]:
        return [n for n in self.nodes.values() if self.expandable(n)]

    def select(self) -> Node | None:
        """Descend from the root by UCT; stop at a node that may still widen (or whose children are all closed)."""
        node = self.root()
        if not self._open_below(node):
            return None
        while True:
            kids = [self.nodes[c] for c in node.children if self.nodes[c].visits and self._open_below(self.nodes[c])]
            if self.expandable(node) and (len(node.children) < math.ceil(max(node.visits, 1) ** self.alpha) or not kids):
                return node
            if not kids:
                return node if self.expandable(node) else None
            node = max(kids, key=lambda k: k.value / k.visits
                       + self.c * math.sqrt(math.log(max(node.visits, 1)) / k.visits))

    def best(self) -> Node:
        return max(self.nodes.values(), key=lambda n: (n.score, -n.iteration))

    def lineage(self, node: Node) -> list[Node]:
        out = []
        while node:
            out.append(node)
            node = self.nodes.get(node.parent) if node.parent else None
        return out[::-1]
