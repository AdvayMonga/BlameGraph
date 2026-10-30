"""The genetic search (kernels/search/generations.py) with a fake model and a fake evaluator: generation 0 from the
start alone, parents drawn by fitness with the best always among them, one slice per call rotated, the model mix
cycling, children under their own parent, exactly `generations` generations whatever they score, the four idea types
and their adaptive split, persistence lines, the memory in every call, and the archive reading each idea's own parent
back."""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernels import archive  # noqa: E402
from kernels.search import generations as G  # noqa: E402
from kernels.search.funnel import Funnel, Models, generation  # noqa: E402
from kernels.search.llm import CACHE_SPLIT  # noqa: E402
from kernels.search.tree import Node, Tree  # noqa: E402

START = "a = 1\n"
MADE = [0]                   # changes made so far, so every program is distinct: `a = <parent's a>*100 + k`


class Model:
    """Each call makes one change, `a = <parent's a>*100 + k` (k counting on); records which model, slice and parent
    each call used."""
    def __init__(self, tag):
        self.tag, self.calls, self.systems = tag, [], []

    def __call__(self, system, user):
        cur = int(re.search(r"```python\na = (\d+)", user).group(1))
        sl = next((s for s, m in (("ptx", "PTX comparison"), ("winners", "winners elsewhere"), ("compiler", "compiler's"),
                                  ("crossover", "Another program")) if m in user), "measurements")
        self.calls.append((self.tag, sl, cur))
        self.systems.append(system)
        k = MADE[0] % 90
        MADE[0] += 1
        return (json.dumps({"bottleneck": "b", "optimisation_method": "m", "modification_plan": f"k {k}"})
                + f"\n<<<<<<< SEARCH\na = {cur}\n=======\na = {cur * 100 + k + 1}\n>>>>>>> REPLACE", 0.1)


def stage(code, n):
    a = int(re.search(r"a = (\d+)", code).group(1))
    if n == 1:
        return {"metrics": {"correct": 1.0, "stage": 1}, "artifacts": {}}
    score = (a % 100) / 10 + (a // 100) / 1000          # the k-th change scores k/10: later ones faster, parents' a adds
    return {"metrics": {"correct": 1.0, "combined_score": score, "stage": n}, "artifacts": {}}


def funnel(ptx="cards", **kw):
    MADE[0] = 0
    opus, sonnet = Model("opus"), Model("sonnet")
    return Funnel(Models(opus, (opus, opus, opus, sonnet)), stage, ptx=ptx, parallel=2, **kw)


def calls(f):
    return f.models.judge.calls + f.models.judges[-1].calls


def rows_of(d):
    return [json.loads(x) for x in (d / "hypotheses.jsonl").read_text().splitlines()]


def test_generation_zero_is_the_start_alone_and_later_ones_draw_parents_by_fitness(tmp_path):
    f = funnel()
    tree, cost = G.search(START, "SYS", f, lambda c: stage(c, 2), tmp_path, usd=50.0, gen_size=8, n_parents=2,
                          generations=2, types=False)
    rows = rows_of(tmp_path)
    assert [r["iteration"] for r in rows] == [0, 1, 2] and cost["generations"] == 3
    g0 = rows[0]
    assert g0["parent_id"] == tree.root().id and g0["named"] == G.GEN0 and g0["examined"] == G.GEN0   # all correct
    assert {c[1] for c in calls(f) if c[2] == 1} == {"measurements", "ptx"}         # slices rotated over the calls
    for r in rows[1:]:
        assert r["parent_id"] is None and len(r["parents"]) == 2 and r["examined"] == r["named"] == 8
        assert r["parents"][0]["score"] == max(p["score"] for p in r["parents"])      # the best is always a parent
        kids = [n for n in tree.nodes.values() if n.iteration == r["iteration"]]
        assert {k.parent for k in kids} <= {p["id"] for p in r["parents"]}             # each under its own parent
        assert all(x["idea"]["parent_id"] in {p["id"] for p in r["parents"]} for x in r["ideas"])
    assert cost["evals"] == 1 + G.GEN0 + 2 * 8 and len(tree.nodes) == cost["evals"] and f.calls == G.GEN0 + 16
    trace = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert len(trace) == len(tree.nodes) and trace[0]["parent_id"] is None


def test_each_call_reads_one_rotated_slice_and_every_fourth_call_goes_to_the_second_model():
    f, t = funnel(), Tree()
    ok = lambda sc: {"correct": 1.0, "combined_score": sc, "stage": 2}      # noqa: E731
    ps = [t.add(Node(f"p{i}", None, f"a = {i + 1}\n", ok(1.0 + i), {}, 0)) for i in range(4)]
    generation(f, t, [G.Slot(p, "refine", 1) for p in ps], "SYS", 1, lambda n, c: t.add(n), [].append)
    got = sorted(calls(f), key=lambda c: c[2])
    assert [c[1] for c in got] == ["ptx", "measurements", "ptx", "measurements"]     # rotated across the calls
    assert [c[0] for c in got].count("sonnet") == 1 and len(got) == 4               # 3 : 1


def test_parents_are_drawn_without_replacement_with_a_floor_so_the_weakest_can_be_picked():
    t, rng = Tree(), random.Random(1)
    ok = lambda sc: {"correct": 1.0, "combined_score": sc, "stage": 2}      # noqa: E731
    t.add(Node("r", None, "a = 0\n", ok(1.0), {}, 0))
    for i in range(6):
        t.add(Node(f"c{i}", "r", f"a = {i + 1}\n", ok(1.1 + i / 10), {}, 1))
    picked = [tuple(p.id for p in G.parents(t, 3, rng)) for _ in range(200)]
    assert all(p[0] == "c5" for p in picked) and all(len(set(p)) == 3 for p in picked)
    assert any("r" in p for p in picked) and any("c0" in p for p in picked)       # the weakest still get drawn
    hits = {n: sum(n in p for p in picked) for n in ("c4", "c0")}
    assert hits["c4"] > hits["c0"]                                                   # but less often
    assert G.parents(Tree(), 2, rng) == []


def test_the_search_runs_every_generation_however_little_improves(tmp_path):
    f = funnel()
    orig = f.stage

    def worse(code, n):                               # every child scores below its parent
        r = orig(code, n)
        if n == 2:
            r = {**r, "metrics": {**r["metrics"], "combined_score": 0.001}}
        return r
    f.stage = worse
    _, cost = G.search(START, "SYS", f, lambda c: {"metrics": {"correct": 1.0, "combined_score": 1.0, "stage": 2},
                                                   "artifacts": {}},
                       tmp_path, usd=1000.0, gen_size=6, n_parents=2, generations=5, types=False)
    rows = rows_of(tmp_path)
    assert all(r["improved"] == 0 for r in rows) and [r["named"] for r in rows[1:]] == [6] * 5
    assert cost["generations"] == 6 and set(cost) == {"usd", "model_calls", "evals", "generations", "types"}


def test_each_generation_splits_its_ideas_over_the_four_types(tmp_path):
    f = funnel()
    tree, cost = G.search(START, "SYS", f, lambda c: stage(c, 2), tmp_path, usd=50.0, gen_size=12, n_parents=2,
                          generations=2)
    rows = rows_of(tmp_path)
    assert {x["idea"]["type"] for x in rows[0]["ideas"]} == {"start"} and "types" not in rows[0]
    for r in rows[1:]:
        q = r["types"]
        assert set(q) == {"refine", "persist", "fresh", "crossover"} and sum(q.values()) == 12   # generation 0 opened lines
        assert all(n >= G.TYPE_FLOOR for n in q.values())
        named = {t: sum(x["idea"]["type"] == t for x in r["ideas"]) for t in q}
        assert named == q                                          # every idea tagged with the type it was made for
        kids = [n for n in tree.nodes.values() if n.iteration == r["iteration"]]
        assert {t: sum(k.kind == t for k in kids) for t in q} == q                  # every correct one measured
    assert any(c[1] == "crossover" for c in calls(f))
    assert set(cost["types"]) == set(G.TYPES) and sum(v["examined"] for v in cost["types"].values()) == 24
    xs = [x for r in rows[1:] for x in r["ideas"] if x["idea"]["type"] == "crossover"]
    assert xs and all(x["idea"]["with"] in tree.nodes for x in xs)


def test_persist_fresh_and_crossover_calls_ask_for_their_own_kind_of_change():
    m = Model("opus")
    f = Funnel(Models(m), stage, parallel=2)
    t, ok = Tree(), (lambda sc: {"correct": 1.0, "combined_score": sc, "stage": 2})
    root = t.add(Node("r", None, "a = 1\n", ok(1.0), {}, 0))
    head = t.add(Node("h", "r", "a = 2\n", ok(1.5), {}, 1, judge='{"optimisation_method": "wider tiles"}'))
    seen = []
    f.models.judge = lambda s, u: (seen.append(u), m(s, u))[1]
    plan = [G.Slot(head, "persist", 1), G.Slot(root, "fresh", 1), G.Slot(root, "crossover", 1, head)]
    generation(f, t, plan, "SYS", 2, lambda n, c: t.add(n), [].append)
    persist, fresh, cross = seen
    assert "from 1.0000 to 1.5000" in persist and "wider tiles" in persist and "same change further" in persist
    assert "overall structure" in fresh and "Another program" not in fresh
    assert "Another program in this generation, h" in cross and "```python\na = 2\n" in cross     # its full code


def test_the_split_adapts_to_each_types_rate_of_new_bests_with_a_floor():
    even = G.split(24, {t: [0, 0] for t in G.TYPES}, set(G.TYPES))
    assert even == G.TYPES                                         # no measurements yet: the base split
    hot = G.split(24, {"refine": [20, 0], "persist": [10, 8], "fresh": [10, 0], "crossover": [5, 0]}, set(G.TYPES))
    assert hot["persist"] > 6 and min(hot.values()) >= G.TYPE_FLOOR and sum(hot.values()) == 24
    some = G.split(16, {t: [0, 0] for t in G.TYPES}, {"refine", "fresh"})       # no line open, one parent
    assert set(some) == {"refine", "fresh"} and sum(some.values()) == 16
    assert sum(G.split(5, {t: [0, 0] for t in G.TYPES}, set(G.TYPES)).values()) == 5
    assert G.shares(8, {"a": 8, "b": 6, "c": 6, "d": 4}) == {"a": 3, "b": 2, "c": 2, "d": 1}


def test_a_line_moves_to_a_better_persist_child_and_closes_after_three_misses():
    t, ok = Tree(), (lambda sc: {"correct": 1.0, "combined_score": sc, "stage": 2})
    t.add(Node("r", None, "a = 1\n", ok(1.0), {}, 0))
    h = t.add(Node("h", "r", "a = 2\n", ok(1.2), {}, 1))
    lines = G.advance(t, {}, [], [h])
    assert lines == {"h": 0}                                       # beat its parent: heads a line
    worse = t.add(Node("w", "h", "a = 3\n", ok(1.1), {}, 2, kind="persist"))
    plan = [G.Slot(h, "persist", 1)]
    lines = G.advance(t, lines, plan, [worse])
    assert lines == {"h": 1}
    better = t.add(Node("b", "h", "a = 4\n", ok(1.3), {}, 3, kind="persist"))
    assert G.advance(t, lines, plan, [better]) == {"b": 0}         # the line moves on
    for _ in range(2):
        lines = G.advance(t, lines, plan, [])
    assert lines == {}                                             # three generations without a better child


def test_every_call_gets_the_memory(tmp_path):
    other = archive.Entry("i1", "other_task", "split k", {"optimisation_method": "split k"}, outcomes=[
        archive.Outcome("k", "p", 1.0, "c", 1.3, True, 2, bound="memory")])
    lost = archive.Entry("i2", "toy", "bigger tiles", {"optimisation_method": "bigger tiles"}, outcomes=[
        archive.Outcome("k", "p", 1.0, "c", 0.8, False, 2), archive.Outcome("k", "p", 1.0, None, error="precheck")])
    MADE[0] = 0
    m = Model("opus")
    f = Funnel(Models(m), stage, parallel=2, task="toy", library=[other, lost])
    G.search(START, "SYS", f, lambda c: stage(c, 2), tmp_path, usd=50.0, gen_size=8, n_parents=2, generations=1)
    assert m.systems and all("split k" in s and '"best_gain": 1.3' in s and '"bound": "memory"' in s for s in m.systems)
    assert all("bigger tiles" in s and '"worst_gain": 0.8' in s for s in m.systems)
    first = m.systems[0]
    assert first.index("split k") < first.index(CACHE_SPLIT) < first.index("bigger tiles")    # general part first
    assert "## This run so far" in m.systems[-1] and "## This run so far" not in first


def test_the_archive_reads_each_ideas_own_parent_from_a_generation(tmp_path):
    from lab import ledger
    f = funnel()
    tree, _ = G.search(START, "SYS", f, lambda c: stage(c, 2), tmp_path, usd=50.0, gen_size=6, n_parents=2,
                       generations=1)
    root = tmp_path / "ledger"
    root.mkdir()
    ledger.append({"kind": "kernels", "config": {"task": "toy"}, "result": {
        "hypotheses": ledger.put_blob(tmp_path / "hypotheses.jsonl", root, ".jsonl"),
        "trace": ledger.put_blob(tmp_path / "trace.jsonl", root, ".jsonl")}}, root)
    rows = archive.ideas(root)
    gen1 = [o for _, _, o in rows if o.parent_id and o.parent_id != tree.root().id]
    assert gen1 and all(o.parent_score == tree.nodes[o.parent_id].score for o in gen1)
    assert all(o.parent_id in tree.nodes for _, _, o in rows)


if __name__ == "__main__":
    import inspect
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(*[Path(tempfile.mkdtemp()) for _ in inspect.signature(fn).parameters]); print("ok", name)
