"""The search tree: every evaluated program is a node, kept whether it won, lost or failed. A parent is chosen
ε-greedily among the nodes still open (the best by score, else a random open one); a node closes after `max_children`
children or `max_failures` failed ones (STARK's caps), so search leaves dead ends without forgetting them."""
from __future__ import annotations

import hashlib
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
    children: list[str] = field(default_factory=list)
    failures: int = 0

    @property
    def score(self) -> float:
        return float(self.metrics.get("combined_score") or 0.0)


def fingerprint(code: str) -> str:
    """Hash of the program with comments and whitespace removed: two proposals that differ only there are one."""
    body = re.sub(r"#[^\n]*", "", code)
    return hashlib.sha256(re.sub(r"\s+", "", body).encode()).hexdigest()[:16]


class Tree:
    def __init__(self, seed: int = 0, epsilon: float = 0.3, max_children: int = 5, max_failures: int = 3):
        self.nodes: dict[str, Node] = {}
        self.seen: set[str] = set()
        self.rng, self.epsilon = random.Random(seed), epsilon
        self.max_children, self.max_failures = max_children, max_failures

    def add(self, node: Node) -> Node:
        self.nodes[node.id] = node
        self.seen.add(fingerprint(node.code))
        if node.parent:
            p = self.nodes[node.parent]
            p.children.append(node.id)
            if node.score <= 0:
                p.failures += 1
        return node

    def open(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.metrics.get("correct") == 1.0
                and len(n.children) < self.max_children and n.failures < self.max_failures]

    def select(self) -> Node | None:
        cands = self.open()
        if not cands:
            return None
        if self.rng.random() < self.epsilon:
            return self.rng.choice(cands)
        return max(cands, key=lambda n: (n.score, -n.iteration))

    def best(self) -> Node:
        return max(self.nodes.values(), key=lambda n: (n.score, -n.iteration))

    def lineage(self, node: Node) -> list[Node]:
        out = []
        while node:
            out.append(node)
            node = self.nodes.get(node.parent) if node.parent else None
        return out[::-1]
