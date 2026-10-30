"""One round (kernels/search/funnel.py) with a fake model and a fake evaluator: one call per change, its hypothesis
and its SEARCH/REPLACE blocks in one answer, edits that do not apply or repeat a program dropped, every correct program
measured and in the tree, and every idea's fate in the log (which the archive reads back)."""
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


class Model:
    """The k-th call sets `a` to the k-th of `values` and says so in its hypothesis; its code has braces in it."""
    def __init__(self, values=(5, 13, 22, 7)):
        self.values, self.calls, self.systems = values, [], []

    def __call__(self, system, user):
        self.calls.append(user)
        self.systems.append(system)
        n = self.values[(len(self.calls) - 1) % len(self.values)]
        return (json.dumps(h(n)) + f"\n<<<<<<< SEARCH\na = 1\n=======\na = {n}  # {{x: [1]}}\n>>>>>>> REPLACE",
                0.1)


def stage(code, n):
    a = int(re.search(r"a = (\d+)", code).group(1))
    if n == 1:
        return {"metrics": {"correct": float(a != 13), "stage": 1}, "artifacts": {} if a != 13 else {"error": "wrong"}}
    return {"metrics": {"correct": 1.0, "combined_score": a / 10, "stage": n}, "artifacts": {}}


def funnel(model=None, **kw):
    return Funnel(Models(model or Model()), stage, per_call=4, parallel=2, **kw)


def root_of(t):
    return t.add(Node("r", None, START, {"correct": 1.0, "combined_score": 0.1, "stage": 2}, {}, 0))


def test_one_call_per_change_and_every_correct_program_joins_the_tree():
    f, t, rows, kept = funnel(ptx="cards"), Tree(), [], []           # two slices: measurements, ptx
    kids = expand(f, t, root_of(t), "SYS", 1, lambda n, c: kept.append(t.add(n)), rows.append)
    row = rows[0]
    assert row["named"] == 4 and f.calls == 4 and row["correct"] == 3
    assert sorted(k.score for k in kids) == [0.5, 0.7, 2.2] and kept == kids and len(t.nodes) == 4
    fates = {x["idea"]["modification_plan"]: x["fate"] for x in row["ideas"]}
    assert fates["set a = 13"] == "failed correctness" and fates["set a = 22"] == "full exam"
    assert all(json.loads(k.judge)["bottleneck"] for k in kids)     # the hypothesis, braces in code
    assert sum("feature cards" in u for u in f.models.judge.calls) == 2                 # slices rotated over the calls


def test_a_call_sees_the_starting_program_other_programs_as_full_code_the_memory_and_the_format():
    f, t = funnel(Model((5,))), Tree()
    root = root_of(t)
    t.add(Node("x", "r", "a = 2\nb = 1\n", {"correct": 1.0, "combined_score": 1.5, "stage": 2}, {}, 1))
    y = t.add(Node("y", "r", "a = 1\nb = 2\n", {"correct": 1.0, "combined_score": 1.3, "stage": 2}, {}, 1))
    f.memory = ("# GENERAL", "# TASK")
    expand(f, t, y, "SYS", 1, lambda n, c: t.add(n), [].append)
    user, system = f.models.judge.calls[0], f.models.judge.systems[0]
    assert f"# The starting program {root.id}" in user and "```python\na = 1\nb = 0\n" in user
    assert "## Program x (score 1.5000)\n```python\na = 2\nb = 1\n" in user                # full code, not a diff
    assert "# GENERAL" in system and "# TASK" in system and "SEARCH/REPLACE" in system


def test_the_search_runs_rounds_and_the_archive_reads_them_back(tmp_path):
    from lab import ledger
    tree, cost = search(START, "SYS", Model(), lambda c: stage(c, 2), tmp_path, evals=5, usd=50.0, funnel=funnel())
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
            ok = "a = 22" not in code                                   # one change does not build
            return {"metrics": {"stage": 5, "correct": float(ok)},
                    "artifacts": {"precheck": "imported" if ok else "the CUDA C++ build failed: k.cu(3): error",
                                  **({"compile_report": report} if ok else {})}}
        return stage(code, n)
    f = Funnel(Models(Model()), staged, per_call=4, parallel=2, precheck=True)
    t, rows = Tree(), []
    root = root_of(t)
    kids = expand(f, t, root, "SYS", 1, lambda n, c: t.add(n), rows.append)
    fates = {x["idea"]["modification_plan"]: x["fate"] for x in rows[0]["ideas"]}
    assert fates["set a = 22"].startswith("precheck: the CUDA C++ build failed")
    assert root.artifacts["compile_report"] == report and all(k.artifacts["compile_report"] == report for k in kids)
    assert any("The compiler's report for this program" in u for u in f.models.judge.calls)


def test_a_round_stops_mid_way_when_the_model_budget_runs_out_and_says_so():
    f, t, rows = funnel(workers=1), Tree(), []
    f.usd_cap = 0.15                                      # each call costs 0.1: the third is refused
    assert expand(f, t, root_of(t), "SYS", 1, lambda n, c: t.add(n), rows.append) == []
    assert rows[0]["stopped"] == "model budget" and f.calls == 2


def test_ncu_is_not_asked_for_again_once_the_host_refuses_and_programs_are_never_written_twice():
    from kernels.search.loop import REFUSED
    calls = []

    def staged(code, n):
        calls.append(n)
        return {"metrics": {"stage": 3}, "artifacts": {"ncu_error": REFUSED}} if n == 3 else stage(code, n)
    f = Funnel(Models(Model()), staged, per_call=4, parallel=1, ncu=True)
    t = Tree()
    root = root_of(t)
    kids = expand(f, t, root, "SYS", 1, lambda n, c: t.add(n), [].append)
    assert calls.count(3) == 1 and not f.ncu and "ncu_missing" in root.artifacts      # the parent's, asked once
    assert all("ncu" not in k.artifacts and "ncu_missing" not in k.artifacts for k in kids)   # children: as parents
    rows = []
    expand(f, t, root, "SYS", 2, lambda n, c: t.add(n), rows.append)       # the same changes again, on the same parent
    again = {x["idea"]["modification_plan"]: x["fate"] for x in rows[0]["ideas"]}
    assert again["set a = 5"].startswith("not applied: identical")      # written in round 1, never written again


def test_counters_are_taken_when_a_program_first_becomes_a_parent():
    stages = []

    def staged(code, n):
        stages.append(n)
        if n == 3:
            return {"metrics": {}, "artifacts": {"ncu": json.dumps({"shapes": [{"kernels": [{"name": "k", "x": 1}]}]})}}
        return stage(code, n)
    f = Funnel(Models(Model()), staged, per_call=4, parallel=2, ncu=True, workers=4)
    t = Tree()
    root = root_of(t)
    kids = expand(f, t, root, "SYS", 1, lambda n, c: t.add(n), [].append)
    assert stages.count(3) == 1 and "ncu" in root.artifacts and stages.index(3) == 0     # before any child is made
    assert all("ncu" not in k.artifacts for k in kids)
    expand(f, t, kids[0], "SYS", 2, lambda n, c: t.add(n), [].append)
    assert stages.count(3) == 2 and "ncu" in kids[0].artifacts                  # profiled once chosen as a parent


def test_model_calls_run_in_parallel_the_first_alone_and_a_program_written_twice_in_a_batch_is_refused():
    import threading
    import time
    from kernels.search.funnel import pmap, written
    f = Funnel(Models(None), stage, workers=4)
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


def test_every_program_is_tuned_on_every_seen_shape_before_its_full_exam():
    from kernels.search import tune
    prog = "CONFIG = {'t': 1}\nTUNABLES = {'t': [1, 2, 4]}\n\n\ndef kernel(x):\n    return x\n"
    calls = []

    def tstage(code, n):
        calls.append((n, tune.params(code)[0]["t"], "TUNED-DISPATCH-START" in code))
        if n == 6:                               # point t=2 wins small calls, t=4 large ones
            rows = [{"config": {"t": t}, "score": s, "timings": [{"shape": [4], "speedup": a, "weight": 0.5},
                                                                 {"shape": [512], "speedup": b, "weight": 0.5}]}
                    for t, s, a, b in ((1, 1.0, 1.0, 1.0), (2, 1.3, 2.0, 0.9), (4, 1.2, 0.8, 1.8))]
            return {"metrics": {"stage": n}, "artifacts": {"tune": json.dumps(rows)}}
        return {"metrics": {"correct": 1.0, "combined_score": 1.0, "stage": n}, "artifacts": {}}
    f = Funnel(Models(None), tstage, tune=True, size_expr="args[0].shape[0]")
    code, r = f.measure(prog)                    # every shape, a dispatcher per size region
    assert calls[0][0] == 6 and calls[1] == (2, 2, True) and "TUNED-DISPATCH-START" in code
    code2, _ = f.measure(code)                   # tuned again: the old dispatcher removed first
    assert calls[2] == (6, 2, False) and code2.count("TUNED-DISPATCH-START") == 1
    assert Funnel(Models(None), tstage).measure(prog)[0] == prog          # tuning off
    from kernels.search.loop import facts
    shown = facts(Node("n", None, code, r["metrics"], r["artifacts"], 0))
    assert "grid points timed" in shown and '"config": {"t": 4}, "score": 1.2' in shown       # a fact for the model


if __name__ == "__main__":
    import inspect
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(*[Path(tempfile.mkdtemp()) for _ in inspect.signature(fn).parameters]); print("ok", name)
