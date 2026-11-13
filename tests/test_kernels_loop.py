"""The kernel search's parts with fakes of the others (no GPU, no model): the loop, context, diversity and the
evaluator's phases. Run: python tests/test_kernels_loop.py"""
from __future__ import annotations

import json
import random
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate.kernels import archive  # noqa: E402
from generate.kernels.search import context, diversity  # noqa: E402
from generate.kernels.search.evaluate import EXAM, REFUSED, Evaluator  # noqa: E402
from generate.kernels.search.llm import CACHE_SPLIT  # noqa: E402
from generate.kernels.search.loop import search  # noqa: E402
from generate.kernels.search.program import Config, Idea, Program  # noqa: E402

START = "a = 1\n"


class Model:
    """Each call makes one change, `a = <parent's a>*100 + k` (k counting on over every Model), and keeps its prompts."""
    made, lock = [0], threading.Lock()

    def __init__(self, tag="m"):
        self.model, self.calls = tag, []

    def __call__(self, system, user):
        cur = int(re.search(r"```python\na = (\d+)", user).group(1))
        with Model.lock:
            k = Model.made[0] % 90
            Model.made[0] += 1
            self.calls.append((system, user))
        return (json.dumps({"bottleneck": "b", "optimisation_method": f"m{k}", "modification_plan": f"k {k}"})
                + f"\n<<<<<<< SEARCH\na = {cur}\n=======\na = {cur * 100 + k + 1}\n>>>>>>> REPLACE", 0.1)


def stage(code, n):
    """Correct unless the program says `wrong`; the k-th change scores k/10, its parents' changes a little more."""
    a = int(re.search(r"a = (\d+)", code).group(1))
    if "wrong" in code and n != 5:
        return {"metrics": {"correct": 0.0, "stage": n}, "artifacts": {"error": "mismatch"}}
    if n != 2:
        return {"metrics": {"correct": 1.0, "stage": n}, "artifacts": {}}
    return {"metrics": {"correct": 1.0, "combined_score": (a % 100) / 10 + (a // 100) / 1000, "stage": 2}, "artifacts": {}}


def run(out, models=None, seeds=None, start=START, **cfg):
    Model.made[0] = 0
    models = models or [Model()]
    pop, cost = search(start, seeds or {}, "SETTING", models, Evaluator(stage), diversity.Novelty(), out,
                       Config(**{"generations": 2, "ideas": 6, "parents": 2, "workers": 2, **cfg}))
    return pop, cost, [json.loads(x) for x in (out / "hypotheses.jsonl").read_text().splitlines()], models


def program(i, score, parent=None, code=None, **artifacts):
    return Program(f"p{i}", parent, code or f"a = {i}\n", {"correct": 1.0, "combined_score": score}, artifacts, 0)


def test_every_generation_makes_its_ideas_and_each_judged_program_joins_under_its_parent(tmp_path):
    pop, cost, rows, _ = run(tmp_path)
    assert [r["named"] for r in rows] == [6, 6] and cost == {"usd": 1.2, "model_calls": 12, "evals": 13, "generations": 2}
    assert len(rows[0]["parents"]) == 1 and len(rows[1]["parents"]) == 2          # the start alone, then two parents
    by = {p.id: p for p in pop}
    assert pop[0].parent is None and all(by[p.parent].generation < p.generation for p in pop[1:])
    assert rows[1]["parents"][0]["id"] == max(pop[:7], key=lambda p: p.score).id   # the best is always a parent
    trace = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [t["child_id"] for t in trace] == [p.id for p in pop] and trace[1]["child_code"] == pop[1].code
    x = rows[0]["ideas"][0]
    assert x["fate"] == EXAM and x["improved"] and x["idea"]["parent_id"] == pop[0].id and x["diff"].count("\n") >= 3


def test_the_archive_reads_a_run_back(tmp_path):
    from data.knowledge import ledger
    run(tmp_path)
    root = tmp_path / "ledger"
    root.mkdir()
    ledger.append({"kind": "kernels", "config": {"task": "toy"}, "result": {
        "hypotheses": ledger.put_blob(tmp_path / "hypotheses.jsonl", root, ".jsonl"),
        "trace": ledger.put_blob(tmp_path / "trace.jsonl", root, ".jsonl")}}, root)
    assert sum(1 for e in archive.build(root) for o in e.outcomes if o.child_score) == 12


def test_it_runs_every_generation_though_nothing_improves_and_stops_when_no_program_can_be_a_parent(tmp_path):
    flat = lambda code, n: {"metrics": {"correct": 1.0, "combined_score": 1.0, "stage": n}, "artifacts": {}}  # noqa: E731
    Model.made[0] = 0
    _, cost = search(START, {}, "S", [Model()], Evaluator(flat), diversity.Novelty(), tmp_path / "flat",
                     Config(generations=3, ideas=2, parents=1, workers=2))
    assert cost["generations"] == 3 and cost["model_calls"] == 6
    pop, cost, rows, _ = run(tmp_path / "dead", start="a = 1  # wrong\n")
    assert len(pop) == 1 and rows == [] and cost["generations"] == 0


def test_a_generations_calls_cycle_through_the_models_and_seeds_are_judged_beside_the_start(tmp_path):
    a, b = Model("sonnet"), Model("opus")
    pop, _, rows, _ = run(tmp_path, models=[a, a, b], seeds={"vllm": "a = 7\n"}, generations=1)
    assert [x["idea"]["model"] for x in rows[0]["ideas"]] == ["sonnet", "sonnet", "opus"] * 2
    assert pop[1].kind == "competitor" and pop[1].parent == pop[0].id and pop[1].plan == {"competitor": "vllm"}
    assert {p["id"] for p in rows[0]["parents"]} == {pop[0].id, pop[1].id}


def test_a_wave_of_calls_sees_what_the_waves_before_it_proposed_for_the_same_program(tmp_path):
    _, _, _, models = run(tmp_path, generations=1, ideas=4, workers=2)
    users = [u for _, u in models[0].calls]
    assert not any("already proposed" in u for u in users[:2])
    assert all("already proposed" in u and "m0 / k 0" in u and "m1 / k 1" in u for u in users[2:])


def test_a_program_that_fails_a_check_is_never_timed_and_its_idea_keeps_why(tmp_path):
    class Wrong(Model):
        def __call__(self, system, user):
            text, usd = super().__call__(system, user)
            return (text.replace("\n>>>>>>> REPLACE", "  # wrong\n>>>>>>> REPLACE") if len(self.calls) == 1 else text), usd
    pop, cost, rows, _ = run(tmp_path, models=[Wrong()], generations=1, workers=1)
    fates = [x["fate"] for x in rows[0]["ideas"]]
    assert fates.count("failed correctness") == 1 and fates.count(EXAM) == 5 and len(pop) == 6 and cost["evals"] == 6
    assert "child_id" not in next(x for x in rows[0]["ideas"] if x["fate"] != EXAM)


def test_checks_run_before_any_timing_and_the_compilers_report_stays_with_the_program():
    order = []

    def st(code, n):
        order.append(n)
        if n == 5:
            return {"metrics": {"correct": 0.0 if "nobuild" in code else 1.0}, "artifacts": {"compile_report": "R", "precheck": "ld"}}
        return stage(code, n)
    ev = Evaluator(st)
    ok, wrong, nobuild = ev.evaluate(["a = 3\n", "a = 4  # wrong\n", "a = 5  # nobuild\n"])
    assert max(i for i, n in enumerate(order) if n != 2) < order.index(2) and order.count(2) == 1 and ev.evals == 1
    assert ok.examined and ok.artifacts["compile_report"] == "R" and ok.metrics["combined_score"] == 0.3
    assert wrong.fate == "failed correctness" and nobuild.fate == "precheck: ld" and not nobuild.examined


def test_counters_are_taken_once_per_program_and_never_again_after_the_host_refuses():
    asked = []

    def st(code, n):
        asked.append(code)
        return {"metrics": {}, "artifacts": {"ncu": "{}"} if code == "ok" else {"ncu_error": REFUSED}}
    ev, a, b, c = Evaluator(st, counters=True), program(1, 1.0, code="ok"), program(2, 1.0, code="no"), program(3, 1.0)
    for p in (a, a, b, b, c):
        ev.profile(p)
    assert asked == ["ok", "no"] and a.artifacts == {"ncu": "{}"} and "unavailable" in b.artifacts["ncu_missing"]
    assert c.artifacts == {}


def test_parents_are_the_best_then_a_weighted_draw_without_replacement_of_usable_programs():
    pop = [program(i, 1.0 + i / 10) for i in range(6)] + [program(9, 0.0)]
    pop[2].metrics["correct"] = 0.0
    drawn = [diversity.pick(pop, 3, random.Random(s)) for s in range(200)]
    assert all(d[0] is pop[5] and len({p.id for p in d}) == 3 for d in drawn)
    seen = {p.id for d in drawn for p in d}
    assert "p0" in seen and not seen & {"p2", "p9"}                               # the weakest can be drawn; unusable never
    assert sum(pop[4] in d for d in drawn) > sum(pop[0] in d for d in drawn)       # weighted by score
    assert diversity.pick([pop[6]], 3, random.Random(0)) == [] and len(diversity.pick(pop[:2], 8, random.Random(0))) == 2


def test_a_repeat_is_refused_identical_anywhere_and_near_duplicate_only_among_siblings():
    class Embedder:
        def passages(self, texts):
            return [[1.0, 0.0] if "x" in t else [0.0, 1.0] for t in texts]
    n, p, q = diversity.Novelty(Embedder()), program(1, 1.0), program(2, 1.0)
    n.add(p.code)
    assert "identical" in n.refuse("a = 1  # a comment\n", q)
    assert n.refuse("a = 1\nx = 1\n", p) is None and "near-duplicate" in n.refuse("a = 1\nx = 2\n", p)
    assert n.refuse("a = 2\nx = 3\n", q) is None and n.refuse("a = 1\ny = 1\n", p) is None
    assert diversity.Novelty().refuse("a = 1\nx = 2\n", p) is None                # no embedder: identical programs only
    c = 'C = """\n#define B 64\n"""\n'                                           # a string is kept whole
    assert diversity.fingerprint(c) != diversity.fingerprint(c.replace("64", "128"))


def test_a_call_reads_the_setting_the_history_the_program_its_facts_and_the_other_programs():
    timings = json.dumps([{"shape": [4], "speedup": 1.2, "weight": 1, "pct_of_roofline": 40.0, "secret": 1}])
    root, mid = program(0, 1.0), program(1, 1.5, "p0", timings=timings, compile_report="REPORT", ncu_missing="no ncu")
    leaf, other = program(2, 1.2, "p1"), program(3, 2.0, "p0")
    pop = [root, mid, leaf, other]
    ideas = [Idea("i1", "p0", 1, {"optimisation_method": "fuse"}, "m", 0.1, "c", EXAM, mid),
             Idea("i2", "p0", 1, {"optimisation_method": "unroll"}, "m", 0.1, None, "not applied: no block")]
    system, user = context.build("SETTING", mid, pop, ideas, [{"optimisation_method": "tile"}])
    stable, rest = system.split(CACHE_SPLIT)
    assert stable == "SETTING" and '"idea": "fuse", "gain": 1.5' in rest and '"furthest": "not applied: no block"' in rest
    assert rest.endswith(context.FORMAT) and context.ROLE in rest
    shown, asked = user.split(CACHE_SPLIT)
    assert "# Program p1 (score 1.5000; best in the search 2.0000)" in shown and "p0 (1.0000) -> p1 (1.5000)" in shown
    assert "- p2: score 1.2000" in shown and "# The starting program p0" in shown and "## Program p3 (score 2.0000)" in shown
    assert "40.0% of its roofline bound" in shown and "REPORT" in shown and "no ncu" in shown and "secret" not in shown
    assert "- tile" in asked and asked.endswith(context.REQUEST)
    assert "# The starting program" not in context.build("S", root, pop, [], [])[1]
    assert context.build("S", root, [root], [], [])[0] == "S" + CACHE_SPLIT + "\n\n" + context.ROLE + "\n\n" + context.FORMAT


def test_an_answers_idea_is_read_best_effort():
    assert context.read('{"bottleneck": "b"}\n<<<<<<< SEARCH\n{"x": 1}')["bottleneck"] == "b"
    assert context.read("just words\n<<<<<<< SEARCH") == {"plan": "just words\n"} and context.read("") == {}


if __name__ == "__main__":
    import inspect
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(*[Path(tempfile.mkdtemp()) for _ in inspect.signature(fn).parameters]); print("ok", name)
