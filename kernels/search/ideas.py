"""The idea table (decided 2026-10-27, after K-Search, arXiv 2602.19128): every idea the genetic search names is kept
with its measured history, so an idea that needs several steps, or whose first step is slower, is refined again
instead of being tried once and dropped.

An idea is its text (bottleneck, method, plan, predicted counters), the tree program its attempts start from, and each
attempt's measurement (the full exam's score where it reached it, else its quick timing) with the diff of its best
attempt. Its gain is its best measurement over its parent's score. Each generation the `PERSIST` best active ideas
(by measured gain; the model's view never ranks them) get `DRAFTS` more drafts each, shown their earlier attempts; an
idea closes after `PATIENCE` generations without a better measurement. After a generation, one judge call reads the
table and may close ideas or add sub-ideas of one (K-Search's Prune and Insert; the next generation drafts them).
Measurements only; the table is in-run state, written to `<out>/ideas.json`."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

PERSIST = 4                  # ideas refined per generation
DRAFTS = 2                   # drafts of each per generation
PATIENCE = 3                 # generations without a better measurement before an idea closes
SUB = 4                      # sub-ideas the edit call may add per generation
KEYS = ("bottleneck", "optimisation_method", "modification_plan", "mechanism")

EDIT = """Below is the table of ideas this search has tried on the program, each with its measured attempts. Answer with
one JSON array and nothing else, of objects {"action": "close", "id": "<idea id>", "why": "<max 20 words>"} for ideas
the measurements show will not improve, and at most %d {"action": "add", "from": "<idea id>", "idea": {"bottleneck":
"<max 30 words>", "optimisation_method": "<max 35 words>", "modification_plan": "<max 35 words>", "mechanism": []}}
for sub-ideas that continue one idea from its best attempt. An empty array is an answer.""" % SUB


@dataclass
class Idea:
    id: str
    idea: dict
    parent_id: str
    base_score: float
    status: str = "active"                   # or "closed"
    best: float = 0.0                        # best measurement so far (full exam, else quick timing)
    best_child: str | None = None            # the tree node of its best attempt, when that one reached the full exam
    best_diff: str = ""
    since_best: int = 0                      # generations since `best` last rose
    attempts: list = field(default_factory=list)
    closed_why: str = ""

    def gain(self) -> float:
        return self.best / self.base_score if self.base_score else 0.0

    def history(self) -> str:
        lines = [f"- generation {a['generation']}: {a['fate']}" + (f"; quick timing {a['quick']:.4f}" if a.get("quick")
                 is not None else "") + (f"; full exam {a['child']:.4f}" if a.get("child") is not None else "")
                 for a in self.attempts]
        head = f"Its parent scored {self.base_score:.4f}; its best attempt {self.best:.4f}."
        tail = f"\n# Its best attempt so far, as a diff from the program shown\n```diff\n{self.best_diff}```" \
            if self.best_diff else ""
        return head + "\n" + "\n".join(lines) + tail


class Table:
    def __init__(self):
        self.ideas: dict[str, Idea] = {}
        self.pending: list[dict] = []        # sub-ideas the edit call added, drafted next generation

    def record(self, row: dict, gen: int) -> None:
        """A round's log row: each idea's fate as one attempt (one per idea per generation: its best draft)."""
        scores = {p["id"]: p["score"] for p in row.get("parents") or []}
        best: dict[str, dict] = {}
        for x in row.get("ideas") or []:
            h, iid = x.get("idea") or {}, (x.get("idea") or {}).get("idea_id")
            if not iid:
                continue
            if iid not in self.ideas:
                self.ideas[iid] = Idea(iid, {k: h[k] for k in KEYS if k in h}, h.get("parent_id"),
                                       float(scores.get(h.get("parent_id")) or 0.0))
            q = (x.get("quick") or {}).get("combined_score")
            a = {"generation": gen, "fate": x.get("fate", ""), "quick": q, "child": x.get("child_score"),
                 "child_id": x.get("child_id"), "diff": x.get("diff", "")}
            m = a["child"] if a["child"] is not None else (q or 0.0)
            if iid not in best or m > best[iid]["m"]:
                best[iid] = {**a, "m": m}
        for iid, a in best.items():
            i = self.ideas[iid]
            i.attempts.append({k: v for k, v in a.items() if k not in ("m", "diff")})
            if a["m"] > i.best:
                i.best, i.since_best = a["m"], 0
                i.best_diff = a["diff"][:4000]
                i.best_child = a["child_id"] if a["child"] is not None and a["child"] > i.base_score else i.best_child
            else:
                i.since_best += 1
            if i.since_best >= PATIENCE:
                i.status, i.closed_why = "closed", f"{PATIENCE} generations without a better measurement"

    def persistent(self, tree) -> list[dict]:
        """This generation's refinement drafts: the PERSIST best active ideas by measured gain, DRAFTS each, from
        their best examined attempt where it beat their parent, else their parent, shown their history."""
        live = [i for i in self.ideas.values() if i.status == "active" and i.attempts and i.best > 0
                and (i.best_child in tree.nodes or i.parent_id in tree.nodes)]
        live.sort(key=lambda i: (-i.gain(), i.since_best))
        out = []
        for i in live[:PERSIST]:
            base = i.best_child if i.best_child in tree.nodes else i.parent_id
            out += [{**i.idea, "parent_id": base, "idea_id": i.id, "draft": d, "history": i.history(),
                     "slice": "persistence"} for d in range(DRAFTS)]
        return out

    def take_pending(self, tree) -> list[dict]:
        got = [h for h in self.pending if h["parent_id"] in tree.nodes]
        self.pending = []
        return got

    def edit(self, text: str, tree) -> list[dict]:
        """Apply the edit call's answer (closes and sub-ideas); returns what was applied."""
        from kernels.search.funnel import _json
        import uuid
        try:
            acts = json.loads(_json(text))
        except ValueError:
            acts = []
        done = []
        for a in acts if isinstance(acts, list) else []:
            if not isinstance(a, dict):
                continue
            i = self.ideas.get(str(a.get("id") or a.get("from") or ""))
            if a.get("action") == "close" and i and i.status == "active":
                i.status, i.closed_why = "closed", f"edit: {str(a.get('why', ''))[:200]}"
                done.append({"close": i.id})
            elif a.get("action") == "add" and i and isinstance(a.get("idea"), dict) and len(self.pending) < SUB:
                base = i.best_child if i.best_child in tree.nodes else i.parent_id
                sub = {k: a["idea"][k] for k in KEYS if k in a["idea"]}
                self.pending.append({**sub, "parent_id": base, "idea_id": uuid.uuid4().hex[:8], "from_idea": i.id,
                                     "history": i.history(), "slice": "sub-idea"})
                done.append({"add": self.pending[-1]["idea_id"], "from": i.id})
        return done

    def view(self) -> str:
        """The table as the edit call reads it: each idea's text, status, gain and attempts."""
        return "\n\n".join(f"## Idea {i.id} ({i.status}; gain {i.gain():.4f})\n{json.dumps(i.idea)}\n{i.history()}"
                           for i in sorted(self.ideas.values(), key=lambda i: -i.gain()))

    def dump(self) -> dict:
        return {"ideas": [asdict(i) for i in self.ideas.values()], "pending": self.pending}
