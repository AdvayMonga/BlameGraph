"""The funnel (kernels/search/funnel.py): one round from named hypotheses to the full exam, with fake models and a
fake evaluator: duplicates merged, an idea like an archived failure dropped unless the judge names a difference, edits
that do not apply dropped, correctness failures repaired once, the quick test keeping the fastest, only fully examined
programs in the tree, and every idea's fate in the log (which the archive reads back)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402
import re  # noqa: E402

from kernels import archive  # noqa: E402
from kernels.search.funnel import Funnel, Models, expand  # noqa: E402
from kernels.search.loop import search  # noqa: E402
from kernels.search.tree import Node, Tree  # noqa: E402

START = "a = 1\nb = 0\n"


def h(n):
    return {"bottleneck": f"b{n}", "optimisation_method": f"m{n}", "modification_plan": f"set a = {n}"}


class Judge:
    """Names three ideas per slice (one shared by every slice); for the past-failure check keeps nothing."""
    def __init__(self):
        self.calls = []

    def __call__(self, system, user):
        self.calls.append(user)
        if "archived records" in system:
            return json.dumps([{"index": i, "keep": False, "differs": ""} for i in range(20)]), 0.1
        k = len(self.calls)
        return json.dumps([h(5), h(10 + k), h(13 if k == 1 else 20 + k)]), 0.1


def coder(system, user):
    """Writes the plan's `set a = N` (the last one in the prompt)."""
    n = re.findall(r"set a = (\d+)", user)
    return (f"<<<<<<< SEARCH\na = 1\n=======\na = {n[-1]}\n>>>>>>> REPLACE", 0.01) if n else ("", 0.01)


def stage(code, n):
    a = int(re.search(r"a = (\d+)", code).group(1))
    if n == 1:
        return {"metrics": {"correct": float(a != 13), "stage": 1}, "artifacts": {} if a != 13 else {"error": "wrong"}}
    return {"metrics": {"correct": 1.0, "combined_score": a / 10, "stage": n}, "artifacts": {}}


def merge(system, user):
    return json.dumps({"bottleneck": "b5", "optimisation_method": "m5", "modification_plan": "set a = 5"}), 0.001


def funnel(**kw):
    return Funnel(Models(Judge(), merge, coder, coder), stage, per_call=3, keep=2, parallel=2, **kw)


def test_one_round_keeps_the_fastest_in_the_tree_and_logs_every_idea():
    f, t, rows, kept = funnel(ptx="cards"), Tree(), [], []           # two slices: measurements, ptx
    root = t.add(Node("r", None, START, {"correct": 1.0, "combined_score": 0.1, "stage": 2}, {}, 0))
    kids = expand(f, t, root, "SYS", 1, lambda n, c: kept.append(t.add(n)), rows.append)
    row = rows[0]
    assert row["named"] == 6 and row["distinct"] == 5          # h(5) named by both slices, merged
    assert [k.score for k in kids] == [2.2, 1.2] and kept == kids        # the two fastest only
    fates = {f2["idea"]["modification_plan"]: f2["fate"] for f2 in row["ideas"]}
    assert fates["set a = 13"] == "failed correctness, and its repair"   # repaired once, still wrong
    assert fates["set a = 5"].startswith("quick speed test") and fates["set a = 22"] == "full exam"
    assert len(t.nodes) == 3 and f.spent > 0


def test_an_idea_like_an_archived_failure_is_dropped_unless_a_difference_is_named():
    e = archive.Entry("idea-1", "toy", "b5\nm5\nset a = 5", h(5),
                      outcomes=[archive.Outcome("k", "p", 1.0, "c", 0.5, False)])
    f, t, rows = funnel(archive=[e], task="toy"), Tree(), []
    root = t.add(Node("r", None, START, {"correct": 1.0, "combined_score": 0.1, "stage": 2}, {}, 0))
    expand(f, t, root, "SYS", 1, lambda n, c: t.add(n), rows.append)
    fates = {x["idea"]["modification_plan"]: x["fate"] for x in rows[0]["ideas"]}
    assert fates["set a = 5"].startswith("like archived ideas") and rows[0]["after_archive"] == rows[0]["distinct"] - 1
    assert any("ideas in the archive" in u for u in f.models.judge.calls)              # the archive slice ran


def test_the_search_runs_funnel_rounds_and_the_archive_reads_them_back(tmp_path):
    from lab import ledger
    f = funnel()
    tree, cost = search(START, "SYS", coder, lambda c: stage(c, 2), tmp_path, evals=5, usd=50.0,
                        review=f.models.judge, funnel=f)
    assert tree.best().score >= 2.2 and cost["usd"] > 0
    root = tmp_path / "ledger"
    root.mkdir()
    ledger.append({"kind": "kernels", "config": {"task": "toy"}, "result": {
        "hypotheses": ledger.put_blob(tmp_path / "hypotheses.jsonl", root, ".jsonl"),
        "trace": ledger.put_blob(tmp_path / "trace.jsonl", root, ".jsonl")}}, root)
    es = archive.build(root)
    assert any(o.child_score for e in es for o in e.outcomes)
    assert any((o.error or "").startswith("failed correctness") for e in es for o in e.outcomes)


def test_a_predicted_counter_change_is_measured_against_the_parent():
    from kernels.search.funnel import mechanism
    ncu = lambda v: {"ncu": json.dumps({"shapes": [{"shape": [1], "kernels": [{"name": "k", "dram__bytes.sum": v}]}]})}  # noqa: E731
    parent = Node("p", None, START, {}, ncu(100.0), 0)
    child = Node("c", "p", START, {}, ncu(70.0), 1)
    got = mechanism({"mechanism": [{"counter": "dram__bytes.sum", "change_pct": -25}, {"counter": "x", "change_pct": 1}]},
                    parent, child)
    assert abs(got[0]["actual_change_pct"] + 30) < 1e-9 and got[1]["actual_change_pct"] is None and got[1]["missing"]


def test_the_precheck_drops_what_does_not_build_and_feeds_the_compiler_slice():
    report = json.dumps([{"kernel": "k", "registers": 32, "stack": 0, "shared": 0, "local": 0}])

    def staged(code, n):
        if n == 5:
            ok = "a = 12" not in code                                   # one idea does not build
            return {"metrics": {"stage": 5, "correct": float(ok)},
                    "artifacts": {"precheck": "imported" if ok else "the CUDA C++ build failed: k.cu(3): error",
                                  **({"compile_report": report} if ok else {})}}
        return stage(code, n)
    f = Funnel(Models(Judge(), merge, coder, coder), staged, per_call=3, keep=2, parallel=2, ptx="cards", precheck=True)
    t, rows = Tree(), []
    root = t.add(Node("r", None, START, {"correct": 1.0, "combined_score": 0.1, "stage": 2}, {}, 0))
    kids = expand(f, t, root, "SYS", 1, lambda n, c: t.add(n), rows.append)
    fates = {x["idea"]["modification_plan"]: x["fate"] for x in rows[0]["ideas"]}
    assert fates["set a = 12"].startswith("precheck: the CUDA C++ build failed")
    assert root.artifacts["compile_report"] == report and all(k.artifacts["compile_report"] == report for k in kids)
    assert any("The compiler's report for this program" in u for u in f.models.judge.calls)


def test_a_round_stops_mid_way_when_a_budget_runs_out_and_says_so():
    f, t, rows = funnel(ptx="cards"), Tree(), []
    f.usd_cap = 0.05                                      # the first judge call costs 0.1: the second is refused
    root = t.add(Node("r", None, START, {"correct": 1.0, "combined_score": 0.1, "stage": 2}, {}, 0))
    assert expand(f, t, root, "SYS", 1, lambda n, c: t.add(n), rows.append) == []
    assert rows[0]["stopped"] == "model budget" and rows[0]["named"] == 0
    used = [0.0]
    g = funnel(ptx="cards")
    g.gpu_s_cap, g.gpu_used = 2.0, lambda: used[0]

    def staged(code, n):
        used[0] += 1.0
        return stage(code, n)
    g.stage = staged
    rows2, t2 = [], Tree()
    expand(g, t2, t2.add(Node("r", None, START, {"correct": 1.0, "combined_score": 0.1, "stage": 2}, {}, 0)),
           "SYS", 1, lambda n, c: None, rows2.append)
    assert rows2[0]["stopped"] == "GPU budget" and used[0] == 2.0


def test_ncu_is_not_asked_for_again_once_the_host_refuses_and_programs_are_never_written_twice():
    from kernels.search.loop import REFUSED
    calls = []

    def staged(code, n):
        calls.append(n)
        return {"metrics": {"stage": 3}, "artifacts": {"ncu_error": REFUSED}} if n == 3 else stage(code, n)
    f = Funnel(Models(Judge(), merge, coder, coder), staged, per_call=3, keep=2, parallel=1, ptx="cards", ncu=True)
    t = Tree()
    root = t.add(Node("r", None, START, {"correct": 1.0, "combined_score": 0.1, "stage": 2}, {}, 0))
    kids = expand(f, t, root, "SYS", 1, lambda n, c: t.add(n), [].append)
    assert calls.count(3) == 1 and not f.ncu and "ncu_missing" in root.artifacts      # the parent's, asked once
    assert all("ncu" not in k.artifacts and "ncu_missing" not in k.artifacts for k in kids)   # children: as parents
    rows = []
    expand(f, t, root, "SYS", 2, lambda n, c: t.add(n), rows.append)       # the same ideas again, on the same parent
    again = {x["idea"]["modification_plan"]: x["fate"] for x in rows[0]["ideas"]}
    assert again["set a = 5"].startswith("not applied: identical")      # written in round 1, never written again


def test_a_malformed_verdict_never_crashes_and_unmeasured_archive_ideas_never_block():
    from kernels.search.funnel import past_failures
    measured = archive.Entry("i1", "toy", "b5\nm5\nset a = 5", h(5),
                             outcomes=[archive.Outcome("k", "p", 1.0, "c", 0.5, False)])
    unmeasured = archive.Entry("i2", "toy", "b6\nm6\nset a = 6", h(6), outcomes=[archive.Outcome("k", "p", 1.0, None)])

    class Bad:
        def __call__(self, system, user):
            return json.dumps([{"keep": True, "differs": "x"}, {"index": "0", "keep": True, "differs": "parent changed"}]), 0
    f = Funnel(Models(Bad(), merge, coder, coder), stage, archive=[measured, unmeasured], task="toy")
    kept, dropped = past_failures(f, "S", "B", [h(5), h(6)])
    assert [k["modification_plan"] for k in kept] == ["set a = 5", "set a = 6"] and not dropped
    assert kept[0]["differs_from_archive"] == "parent changed"


def test_counters_are_taken_when_a_program_first_becomes_a_parent_and_the_strong_coder_writes_only_the_kept():
    from kernels.search.funnel import Funnel, Models, expand
    from kernels.search.tree import Node, Tree
    stages, strong = [], []

    def staged(code, n):
        stages.append(n)
        if n == 3:
            return {"metrics": {}, "artifacts": {"ncu": json.dumps({"shapes": [{"kernels": [{"name": "k", "x": 1}]}]})}}
        return stage(code, n)

    def strong_coder(system, user):
        strong.append(user)
        return coder(system, user)
    f = Funnel(Models(Judge(), merge, coder, strong_coder), staged, per_call=3, keep=2, parallel=2, ptx="cards",
               ncu=True, workers=4)
    t = Tree()
    root = t.add(Node("r", None, START, {"correct": 1.0, "combined_score": 0.1, "stage": 2}, {}, 0))
    rows = []
    kids = expand(f, t, root, "SYS", 1, lambda n, c: t.add(n), rows.append)
    assert stages.count(3) == 1 and "ncu" in root.artifacts and stages.index(3) == 0     # before any child is made
    assert all("ncu" not in k.artifacts for k in kids)
    rewrites = [u for u in strong if "passed correctness" in u]
    assert len(rewrites) == len(kids) == 2                    # the kept two only, not every correct draft
    assert rows[0]["rewritten"] == 0 and all(x.get("written_by") == "draft" for x in rows[0]["ideas"]
                                               if x["fate"] == "full exam")    # a rewrite to the same code: refused
    expand(f, t, kids[0], "SYS", 2, lambda n, c: t.add(n), [].append)
    assert stages.count(3) == 2 and "ncu" in kids[0].artifacts                  # profiled once chosen as a parent


def test_model_calls_run_in_parallel_the_first_alone_and_a_program_written_twice_in_a_batch_is_refused():
    import threading
    import time
    from kernels.search.funnel import Funnel, Models, pmap, written
    f = Funnel(Models(None, None, None, None), stage, workers=4)
    live, peak, order = [0], [0], []
    lock = threading.Lock()

    def slow(x):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
            order.append(("start", x, live[0]))
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return x * 2
    assert pmap(f, slow, range(9)) == [x * 2 for x in range(9)]
    assert order[0] == ("start", 0, 1) and order[1][0] == "start" and peak[0] == 4     # the first alone, then 4 at once
    seen = set()
    out = written(f, [("a = 1\n", ""), ("a=1\n", ""), (None, "no diff"), ("a = 2\n", "")], seen)
    assert [c for c, _ in out] == ["a = 1\n", None, None, "a = 2\n"] and "identical" in out[1][1] and len(seen) == 2


if __name__ == "__main__":
    import inspect
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(*[Path(tempfile.mkdtemp()) for _ in inspect.signature(fn).parameters]); print("ok", name)
