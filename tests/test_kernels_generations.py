"""The genetic search (kernels/search/generations.py) with fake models and a fake evaluator: generation 0 from the
start alone, parents drawn by fitness with the best always among them, one rotated slice per parent call, the judge mix
cycling, children under their own parent, the shrink rule, racing at the quick cutoff, the quick-vs-full rank
correlation, patience in generations, crossover ideas, and the archive reading each idea's own parent back."""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernels import archive  # noqa: E402
from kernels.search import generations as G  # noqa: E402
from kernels.search.funnel import Funnel, Models, generation, race, spearman  # noqa: E402
from kernels.search.tree import Node, Tree  # noqa: E402

START = "a = 1\n"
NAMED = [0]                  # ideas named so far, so every idea of a round is distinct: `a = <parent's a>*100 + k`


class Judge:
    """Names `n` ideas per call, each `set a = <parent's a>*100 + k` (k counting on within the round); records which
    model and slice each call used."""
    def __init__(self, tag):
        self.tag, self.calls = tag, []

    def __call__(self, system, user):
        n = int(re.search(r"Name (\d+)", system).group(1))
        cur = int(re.search(r"```python\na = (\d+)", user).group(1))
        sl = next((s for s, m in (("ptx", "PTX comparison"), ("winners", "winners elsewhere"), ("compiler", "compiler's"),
                                  ("crossover", "Another program")) if m in user), "measurements")
        self.calls.append((self.tag, sl, cur))
        ks = range(NAMED[0] % 90, NAMED[0] % 90 + n)
        NAMED[0] += n
        return json.dumps([{"bottleneck": "b", "optimisation_method": "m",
                            "modification_plan": f"set a = {cur * 100 + k + 1}"} for k in ks]), 0.1


def coder(system, user):
    n, cur = re.findall(r"set a = (\d+)", user), re.search(r"```python\na = (\d+)", user).group(1)
    return (f"<<<<<<< SEARCH\na = {cur}\n=======\na = {n[-1]}\n>>>>>>> REPLACE", 0.01) if n else ("", 0.01)


def stage(code, n):
    a = int(re.search(r"a = (\d+)", code).group(1))
    if n == 1:
        return {"metrics": {"correct": 1.0, "stage": 1}, "artifacts": {}}
    score = (a % 100) / 10 + (a // 100) / 1000          # the k-th idea scores k/10: later ideas faster, parents' a adds
    return {"metrics": {"correct": 1.0, "combined_score": score, "stage": n}, "artifacts": {}}


def funnel(ptx="cards", **kw):
    NAMED[0] = 0
    opus, sonnet = Judge("opus"), Judge("sonnet")
    first = lambda s, u: (u.split("\n\n")[0], 0.0)          # noqa: E731  a merge keeps the first idea
    return Funnel(Models(opus, first, coder, coder, (opus, opus, opus, sonnet)), stage, ptx=ptx, keep=2, parallel=2,
                  **kw)


def test_generation_zero_is_the_start_alone_and_later_ones_draw_parents_by_fitness(tmp_path):
    f = funnel()
    tree, cost = G.search(START, "SYS", f, lambda c: stage(c, 2), tmp_path, evals=100, usd=50.0, gen_size=8, keep=3,
                          n_parents=2, generations=2, patience=None)
    rows = [json.loads(x) for x in (tmp_path / "hypotheses.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in rows] == [0, 1, 2] and cost["generations"] == 3
    g0 = rows[0]
    assert g0["parent_id"] == tree.root().id and g0["named"] == G.GEN0 and g0["examined"] == G.GEN0_KEEP
    j = f.models.judge
    gen0_calls = [c for c in j.calls + f.models.judges[-1].calls if c[2] == 1]
    assert {c[1] for c in gen0_calls} == {"measurements", "ptx"}        # one call per slice, GEN0 split over them
    for r in rows[1:]:
        assert r["parent_id"] is None and len(r["parents"]) == 2 and r["examined"] == 3 and r["named"] == 8
        assert r["parents"][0]["score"] == max(p["score"] for p in r["parents"])      # the best is always a parent
        kids = [n for n in tree.nodes.values() if n.iteration == r["iteration"]]
        assert {k.parent for k in kids} <= {p["id"] for p in r["parents"]}             # each under its own parent
        assert all(x["idea"]["parent_id"] in {p["id"] for p in r["parents"]} for x in r["ideas"])
        assert r["quick_vs_full_spearman"] is not None and r["improved"] >= 0
    assert cost["evals"] == 1 + G.GEN0_KEEP + 2 * 3 and len(tree.nodes) == cost["evals"]
    trace = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert len(trace) == len(tree.nodes) and trace[0]["parent_id"] is None


def test_each_parent_call_reads_one_rotated_slice_and_every_fourth_call_goes_to_the_second_judge():
    f, t = funnel(), Tree()
    ok = lambda sc: {"correct": 1.0, "combined_score": sc, "stage": 2}      # noqa: E731
    ps = [t.add(Node(f"p{i}", None, f"a = {i + 1}\n", ok(1.0 + i), {}, 0)) for i in range(4)]
    generation(f, t, ps, "SYS", 1, lambda n, c: t.add(n), [].append, quota=3, keep=2)
    calls = sorted(f.models.judge.calls + f.models.judges[-1].calls, key=lambda c: c[2])
    assert [c[1] for c in calls] == ["measurements", "ptx", "measurements", "ptx"]   # rotated across the parents
    assert [c[0] for c in calls].count("sonnet") == 1 and len(calls) == 4             # 3 : 1


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


def test_a_generation_with_few_improvements_shrinks_the_next_one(tmp_path):
    f = funnel()
    orig = f.stage

    def worse(code, n):                               # every child scores below its parent
        r = orig(code, n)
        if n in (2, 4):
            r = {**r, "metrics": {**r["metrics"], "combined_score": 0.001}}
        return r
    f.stage = worse
    _, cost = G.search(START, "SYS", f, lambda c: {"metrics": {"correct": 1.0, "combined_score": 1.0, "stage": 2},
                                                   "artifacts": {}},
                       tmp_path, evals=100, usd=50.0, gen_size=12, keep=2, n_parents=2, generations=2, patience=None)
    rows = [json.loads(x) for x in (tmp_path / "hypotheses.jsonl").read_text().splitlines()]
    assert rows[0]["improved"] == 0 and [r["named"] for r in rows[1:]] == [8, 8]        # 2/3 of 12, from generation 0 on
    assert cost["evals"] == 1 + G.GEN0_KEEP + 2 + 2 and not cost["stopped_no_progress"]


def test_the_search_stops_after_patience_generations_with_no_new_best(tmp_path):
    f = funnel()
    flat = {"metrics": {"correct": 1.0, "combined_score": 5.0, "stage": 2}, "artifacts": {}}
    _, cost = G.search(START, "SYS", f, lambda c: flat, tmp_path, evals=1000, usd=50.0, gen_size=4, keep=1,
                       n_parents=2, generations=20, patience=2)
    assert cost["stopped_no_progress"] and cost["generations"] == 2 and not cost["stopped_near_roofline"]


def test_racing_times_the_programs_at_the_cutoff_again_and_the_rank_correlation_is_spearman():
    timed = []

    def st(code, n):
        timed.append(code)
        return {"metrics": {"combined_score": 1.0 + 0.5 * (code == "x")}, "artifacts": {}}
    f = Funnel(Models(None, None, None, None), st)
    quick = [(None, None, c, {"metrics": {"combined_score": s}, "artifacts": {}})
             for c, s in (("a", 2.0), ("b", 1.02), ("c", 1.0), ("x", 0.99), ("d", 0.5))]
    out = race(f, quick, keep=2)
    assert sorted(timed) == ["b", "c", "x"]                              # within 5% of the cutoff only
    assert [r[2] for r in out][:2] == ["a", "x"] and out[1][3]["metrics"]["raced"] == [0.99, 1.5]
    assert race(f, quick[:2], keep=2) == quick[:2]
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9 and abs(spearman([1, 2, 3], [3, 2, 1]) + 1) < 1e-9
    assert spearman([1, 2], [1, 2]) is None and spearman([1, 1, 1], [1, 2, 3]) is None
    assert abs(spearman([1, 2, 2, 4], [1, 2, 3, 4]) - 0.9486832980505138) < 1e-6


def test_crossover_ideas_show_another_parents_diff_and_are_measured_like_the_rest(tmp_path):
    f = funnel()
    tree, _ = G.search(START, "SYS", f, lambda c: stage(c, 2), tmp_path, evals=100, usd=50.0, gen_size=10, keep=2,
                       n_parents=2, generations=1, patience=None, crossover=0.2)
    rows = [json.loads(x) for x in (tmp_path / "hypotheses.jsonl").read_text().splitlines()]
    xs = [x for x in rows[1]["ideas"] if x["idea"].get("slice") == "crossover"]
    assert len(xs) == 2 and all(x["idea"]["with"] in {p["id"] for p in rows[1]["parents"]} for x in xs)
    assert any(c[1] == "crossover" for c in f.models.judge.calls + f.models.judges[-1].calls)
    assert not any(x["idea"].get("slice") == "crossover" for x in rows[0]["ideas"])          # none in generation 0


def test_the_archive_reads_each_ideas_own_parent_from_a_generation(tmp_path):
    from lab import ledger
    f = funnel()
    tree, _ = G.search(START, "SYS", f, lambda c: stage(c, 2), tmp_path, evals=100, usd=50.0, gen_size=6, keep=2,
                       n_parents=2, generations=1, patience=None)
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


def test_the_idea_table_keeps_measured_history_ranks_by_gain_and_closes_after_patience():
    from kernels.search import ideas as I
    t, table = Tree(), I.Table()
    ok = lambda sc: {"correct": 1.0, "combined_score": sc, "stage": 2}      # noqa: E731
    t.add(Node("p", None, "a = 1\n", ok(1.0), {}, 0))
    t.add(Node("c", "p", "a = 2\n", ok(1.2), {}, 1))
    idea = lambda i: {"bottleneck": "b", "optimisation_method": f"m{i}", "modification_plan": "x", "idea_id": f"i{i}",  # noqa: E731
                      "parent_id": "p"}
    row = {"parents": [{"id": "p", "score": 1.0}], "ideas": [
        {"idea": idea(1), "fate": "full exam", "child_id": "c", "child_score": 1.2, "quick": {"combined_score": 1.19},
         "diff": "+a = 2"},
        {"idea": idea(2), "fate": "quick speed test: not among the fastest", "quick": {"combined_score": 0.9}},
        {"idea": idea(2), "fate": "quick speed test: not among the fastest", "quick": {"combined_score": 0.95}}]}
    table.record(row, 0)
    i1, i2 = table.ideas["i1"], table.ideas["i2"]
    assert (i1.best, i1.best_child, i1.best_diff) == (1.2, "c", "+a = 2") and i2.best == 0.95   # its better draft
    assert len(i2.attempts) == 1 and abs(i1.gain() - 1.2) < 1e-9
    drafts = table.persistent(t)
    assert [d["idea_id"] for d in drafts] == ["i1"] * I.DRAFTS + ["i2"] * I.DRAFTS            # by measured gain
    assert drafts[0]["parent_id"] == "c" and drafts[-1]["parent_id"] == "p"   # from its best examined attempt
    assert "full exam 1.2000" in drafts[0]["history"] and "+a = 2" in drafts[0]["history"]
    for g in range(1, I.PATIENCE + 1):                       # no better measurement for PATIENCE generations
        table.record({"parents": [{"id": "p", "score": 1.0}], "ideas": [
            {"idea": idea(2), "fate": "quick speed test: scored 0", "quick": {"combined_score": 0.0}}]}, g)
    assert i2.status == "closed" and i1.status == "active"
    got = table.edit(json.dumps([{"action": "close", "id": "i1", "why": "w"},
                                 {"action": "add", "from": "i1", "idea": {"bottleneck": "s", "modification_plan": "y"}}]),
                     t)
    assert i1.status == "closed" and len(got) == 2 and table.pending[0]["parent_id"] == "c"
    assert table.take_pending(t)[0]["from_idea"] == "i1" and table.pending == []
    assert table.edit("not json", t) == []


def test_the_ideas_mode_refines_kept_ideas_drafts_sub_ideas_and_sends_an_immigrant_whole_program_changes(tmp_path):
    f = funnel()
    systems, users = [], []
    j, s5 = f.models.judge, f.models.judges[-1]

    def edit_aware(model):
        def call(system, user):
            systems.append(system)
            users.append(user)
            if "table of ideas" in system:                   # the table's edit: one sub-idea of the best idea
                first = re.search(r"## Idea (\w+)", user).group(1)
                return json.dumps([{"action": "add", "from": first, "idea": {
                    "bottleneck": "sub", "optimisation_method": "sub", "modification_plan": "set a = 777"}}]), 0.01
            return model(system, user)
        return call
    f.models.judge = edit_aware(j)
    f.models.judges = tuple(edit_aware(m) for m in (j, j, j, s5))
    drafts = []
    f.models.wide = lambda system, user: (drafts.append(user), coder(system, user))[1]
    tree, cost = G.search(START, "SYS", f, lambda c: stage(c, 2), tmp_path, evals=200, usd=50.0, gen_size=12, keep=3,
                          n_parents=3, generations=2, patience=None, ideas=True)
    rows = [json.loads(x) for x in (tmp_path / "hypotheses.jsonl").read_text().splitlines()]
    slices_ = [{x["idea"].get("slice") for x in r["ideas"]} for r in rows]
    assert "persistence" in slices_[1] and "sub-idea" in slices_[2]            # refined; the edit's sub-idea drafted
    assert any("Name" in s and "replace its algorithm" in s for s in systems)  # the immigrant's whole-program changes
    assert any("This idea's earlier attempts" in u for u in drafts)              # the coder sees the history
    table = json.loads((tmp_path / "ideas.json").read_text())
    assert table["ideas"] and all(i["attempts"] for i in table["ideas"])
    assert all("history" not in x["idea"] for r in rows for x in r["ideas"])   # histories stay in the table
    assert any(x.get("diff") for r in rows for x in r["ideas"])


def test_a_search_stops_starting_work_at_its_wall_clock(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(G.time, "monotonic", lambda: clock[0])
    import kernels.search.funnel as F
    monkeypatch.setattr(F.time, "monotonic", lambda: clock[0])
    f = funnel()
    orig = f.stage

    def slow(code, n):
        clock[0] += 10.0                                  # every evaluator job takes 10 s of wall-clock
        return orig(code, n)
    f.stage = slow
    _, cost = G.search(START, "SYS", f, lambda c: stage(c, 2), tmp_path, evals=10**6, usd=10**6, gen_size=8, keep=2,
                       n_parents=2, generations=50, patience=None, wall_s=300)
    rows = [json.loads(x) for x in (tmp_path / "hypotheses.jsonl").read_text().splitlines()]
    assert cost["stopped_wall_clock"] and rows[-1].get("stopped") == "wall-clock" and clock[0] < 400
