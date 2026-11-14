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


class Namer:
    """Names the changes asked for over the programs shown (m0, m1, ...), reflects on every idea it is shown, and
    reviews by keeping each named change (or as `review(named)` says; None: an answer that is no JSON). Asked for
    ports it names two of the program shown, and writes each as a Writer does."""

    def __init__(self, tag="opus", review=None):
        self.model, self.calls, self.review, self.pen = tag, [], review, Writer(tag)

    def __call__(self, system, user):
        ask = user.split(CACHE_SPLIT)[-1]
        if "# The change to make" in ask:
            return self.pen(system, user)
        self.calls.append((system, user))
        if "public implementations" in ask:
            start = re.search(r"^# Program (\w+) \(score", user, re.M).group(1)
            return json.dumps({"ideas": [{"parent": start, "source": f"lib.k{k}", "modification_plan": f"port {k}"}
                                         for k in range(2)]}), 0.5
        reflections = [{"id": i, "reflection": f"r {i}"} for i in re.findall(r'\{"id": "(\w+)"', ask)]
        if "final list" in ask:
            named = [json.loads(x.split(": ", 1)[1]) for x in ask.splitlines() if re.match(r"n\d+: ", x)]
            ideas = self.review(named) if self.review else [{**p, "from": [f"n{k}"]} for k, p in enumerate(named, 1)]
            return ("not json" if ideas is None else json.dumps({"ideas": ideas})), 0.5
        if "Name " not in ask:
            return json.dumps({"reflections": reflections}), 0.5
        n, ids = int(re.search(r"Name (\d+) changes", ask).group(1)), re.findall(r"^# Program (\w+) \(score", user, re.M)
        return json.dumps({"reflections": reflections, "ideas": [
            {"parent": ids[k % len(ids)], "borrows": [], "optimisation_method": f"m{k}", "modification_plan": f"k {k}",
             "prediction": {"speedup": 1.1}} for k in range(n)]}), 0.5


class Writer:
    """Each call makes one change, `a = <parent's a>*100 + k` (k counting on over every Writer), and keeps its prompts."""
    made, lock = [0], threading.Lock()

    def __init__(self, tag="sonnet"):
        self.model, self.calls = tag, []

    def __call__(self, system, user):
        cur = int(re.search(r"```python\na = (\d+)", user).group(1))
        with Writer.lock:
            k = Writer.made[0] % 90
            Writer.made[0] += 1
            self.calls.append((system, user))
        return f"<<<<<<< SEARCH\na = {cur}\n=======\na = {cur * 100 + k + 1}\n>>>>>>> REPLACE", 0.1


def stage(code, n):
    """Every program builds; one that says `wrong` is wrong in its timed run, one that says `shaky` only in the full
    check; the k-th change scores k/10, its parents' changes a little more."""
    a = int(re.search(r"a = (\d+)", code).group(1))
    if n == 5:
        return {"metrics": {"correct": 1.0, "stage": n}, "artifacts": {}}
    if n in (8, 9):
        ok = "shaky" not in code
        return {"metrics": {"correct": float(ok), "stage": n}, "artifacts": {"gates": json.dumps({
            "correctness_seen": {"passed": ok, "reason": "seed 2 differs"}})}}
    if "wrong" in code:
        return {"metrics": {"correct": 0.0, "stage": n}, "artifacts": {"error": "mismatch"}}
    return {"metrics": {"correct": 1.0, "combined_score": (a % 100) / 10 + (a // 100) / 1000, "stage": 2}, "artifacts": {}}


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def run(out, namer=None, writer=None, seeds=None, start=START, **cfg):
    Writer.made[0] = 0
    namer, writer = namer or Namer(), writer or Writer()
    pop, cost = search(start, seeds or {}, "SETTING", namer, writer, Evaluator(stage), diversity.Novelty(),
                       diversity.Similar(), out,
                       Config(**{"generations": 2, "ideas": 6, "merges": 2, "ports": 0, "parents": 2, "workers": 2,
                                 **cfg}))
    return pop, cost, lines(out / "hypotheses.jsonl"), namer, writer


def program(i, score, parent=None, code=None, **artifacts):
    return Program(f"p{i}", parent, code or f"a = {i}\n", {"correct": 1.0, "combined_score": score}, artifacts, 0)


def entry(text="fuse loops", vector=(1.0, 0.0), gain=1.2):
    return archive.Entry("idea-1", "toy", text, {"optimisation_method": text}, list(vector),
                         [archive.Outcome("r", "p", 1.0, "c", gain, True)])


def test_a_generation_is_one_naming_call_one_review_call_and_a_writer_call_per_change(tmp_path):
    pop, cost, rows, namer, writer = run(tmp_path)
    assert [(r["named"], r["reviewed"], r["written"]) for r in rows] == [(6, 6, 6)] * 2
    assert cost == {"usd": 3.7, "model_calls": 17, "evals": 13, "generations": 2}     # 2 x (name, review, 6 writes) + reflect
    assert len(namer.calls) == 5 and len(writer.calls) == 12
    assert len(rows[0]["parents"]) == 1 and len(rows[1]["parents"]) == 2          # the start alone, then two parents
    by = {p.id: p for p in pop}
    assert pop[0].parent is None and all(by[p.parent].generation < p.generation for p in pop[1:])
    assert rows[1]["parents"][0]["id"] == max(pop[:7], key=lambda p: p.score).id   # the best is always a parent
    trace = lines(tmp_path / "trace.jsonl")
    assert [t["child_id"] for t in trace] == [p.id for p in pop] and trace[1]["child_code"] == pop[1].code
    x = rows[0]["ideas"][0]
    assert x["fate"] == EXAM and x["improved"] and x["idea"]["parent_id"] == pop[0].id and x["diff"].count("\n") >= 3


def test_every_idea_and_every_model_call_is_kept_whole(tmp_path):
    pop, _, _, _, _ = run(tmp_path)
    ideas, calls = lines(tmp_path / "ideas.jsonl"), {c["id"]: c for c in lines(tmp_path / "calls.jsonl")}
    texts = {t["key"]: t["text"] for t in lines(tmp_path / "texts.jsonl")}
    assert len(ideas) == 12 and [c["role"] for c in calls.values()].count("write") == 12
    i = ideas[0]
    assert i["plan"]["prediction"] == {"speedup": 1.1} and i["merged_from"] == ["n1"] and i["writer"] == "sonnet"
    assert i["parent_id"] == pop[0].id and i["child_id"] == pop[1].id and i["gain"] == round(pop[1].score / pop[0].score, 4)
    assert texts[i["program"]] == pop[1].code and i["reflection"] == f"r {i['idea_id']}"
    named, reviewed, written = (calls[i["calls"][k]] for k in ("name", "review", "write"))
    assert (named["role"], named["model"], written["model"]) == ("name", "opus", "sonnet")
    assert texts[named["system"][0]] == "SETTING" and named["user"][0] == reviewed["user"][0]   # the review reads it cached
    assert json.loads(texts[named["answer"]])["ideas"][0]["optimisation_method"] == "m0"
    assert "# The change to make" in texts[written["user"][-1]] and "<<<<<<< SEARCH" in texts[written["answer"]]
    assert all(x["reflection"] for x in ideas) and list(calls.values())[-1]["role"] == "reflect"


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
    Writer.made[0] = 0
    _, cost = search(START, {}, "S", Namer(), Writer(), Evaluator(flat), diversity.Novelty(), diversity.Similar(),
                     tmp_path / "flat", Config(generations=3, ideas=2, ports=0, parents=1, workers=2))
    assert cost["generations"] == 3 and cost["model_calls"] == 3 * 4 + 1
    pop, cost, rows, _, _ = run(tmp_path / "dead", start="a = 1  # wrong\n")
    assert len(pop) == 1 and rows == [] and cost["generations"] == 0 and cost["model_calls"] == 0


def test_seeds_are_judged_beside_the_start_and_can_be_parents(tmp_path):
    pop, _, rows, _, _ = run(tmp_path, seeds={"vllm": "a = 7\n"}, generations=1)
    assert pop[1].kind == "competitor" and pop[1].parent == pop[0].id and pop[1].plan == {"competitor": "vllm"}
    assert {p["id"] for p in rows[0]["parents"]} == {pop[0].id, pop[1].id}
    assert {x["idea"]["parent_id"] for x in rows[0]["ideas"]} == {pop[0].id, pop[1].id}


def test_the_review_merges_and_adds_changes_and_an_unreadable_one_leaves_the_named_changes(tmp_path):
    def review(named):
        return [{**named[0], "from": ["n1", "n2"]}, *[{**p, "from": [f"n{k}"]} for k, p in enumerate(named[2:], 3)],
                {**named[2], "optimisation_method": "both", "from": ["n3", "n4"]}, {"parent": "nobody", "from": ["n5"]}]
    _, _, rows, _, _ = run(tmp_path / "a", namer=Namer(review=review), generations=1)
    ideas = lines(tmp_path / "a" / "ideas.jsonl")
    assert rows[0]["named"] == 6 and rows[0]["reviewed"] == 6                     # 6 named, 2 merged, 1 added, 1 of no parent
    assert ideas[0]["merged_from"] == ["n1", "n2"] and ideas[-1]["merged_from"] == ["n3", "n4"]
    assert ideas[-1]["plan"]["optimisation_method"] == "both" and "from" not in ideas[-1]["plan"]
    _, _, rows, _, _ = run(tmp_path / "b", namer=Namer(review=lambda named: None), generations=1)
    assert rows[0]["reviewed"] == 6 and all(i["merged_from"] == [] for i in lines(tmp_path / "b" / "ideas.jsonl"))
    _, _, rows, _, _ = run(tmp_path / "c", namer=Namer(review=lambda named: named * 5), generations=1)
    assert rows[0]["reviewed"] == 6 + 2                                           # at most `ideas` + `merges`


def test_a_naming_call_reflects_on_the_last_generation_and_one_call_reflects_on_the_final_one(tmp_path):
    _, _, _, namer, _ = run(tmp_path)
    asks = [u.split(CACHE_SPLIT)[-1] for _, u in namer.calls]
    assert "last generation" not in asks[0] and "final list" in asks[1]
    assert "# The last generation's ideas" in asks[2] and '"prediction": {"speedup": 1.1}' in asks[2] and '"gain"' in asks[2]
    assert asks[4].startswith("# The last generation's ideas") and "Reflect on each of these ideas" in asks[4]
    ideas = lines(tmp_path / "ideas.jsonl")
    assert all(i["idea_id"] in asks[2] for i in ideas[:6]) and all(i["idea_id"] in asks[4] for i in ideas[6:])


class Marked(Writer):
    """A Writer whose first program carries a mark (and, with `a`, that value)."""

    def __init__(self, mark, a=None):
        super().__init__()
        self.mark, self.a = mark, a

    def __call__(self, system, user):
        text, usd = super().__call__(system, user)
        if len(self.calls) > 1:
            return text, usd
        new = re.search(r"=======\n(a = \d+)", text).group(1)
        return text.replace(new, f"{f'a = {self.a}' if self.a else new}  # {self.mark}"), usd


def test_a_program_wrong_in_its_timed_run_never_joins_and_its_idea_keeps_why(tmp_path):
    pop, cost, rows, _, _ = run(tmp_path, writer=Marked("wrong"), generations=1, workers=1)
    fates = [x["fate"] for x in rows[0]["ideas"]]
    assert fates.count("failed correctness") == 1 and fates.count(EXAM) == 5 and len(pop) == 6 and cost["evals"] == 7
    assert "child_id" not in next(x for x in rows[0]["ideas"] if x["fate"] != EXAM)
    failed = next(i for i in lines(tmp_path / "ideas.jsonl") if i["fate"] != EXAM)
    assert failed["child_id"] is None and failed["gain"] is None and failed["program"]   # its program is kept


def test_a_picked_parent_that_fails_the_full_check_is_dropped_and_never_picked_again(tmp_path):
    checked = []

    def st(code, n):
        checked.extend([code] if n == 8 else [])
        return stage(code, n)
    Writer.made[0] = 0
    pop, _ = search(START, {}, "S", Namer(), Marked("shaky", a=190), Evaluator(st), diversity.Novelty(),
                    diversity.Similar(), tmp_path, Config(generations=3, ideas=6, ports=0, parents=2, workers=1))
    rows, shaky = lines(tmp_path / "hypotheses.jsonl"), next(p for p in pop if "shaky" in p.code)
    assert shaky.score == max(p.score for p in pop if p.generation == 1) and rows[0]["dropped"] == {}
    assert rows[1]["dropped"] == {shaky.id: "correctness_seen: seed 2 differs"}
    assert [len(r["parents"]) for r in rows] == [1, 1, 2] and all(shaky.id != p["id"] for r in rows for p in r["parents"])
    assert not shaky.usable and rows[2]["dropped"] == {} and len(checked) == len(set(checked))   # each program once
    idea = next(i for i in lines(tmp_path / "ideas.jsonl") if i["child_id"] == shaky.id)
    assert idea["fate"] == "failed the full check as a parent: correctness_seen: seed 2 differs"
    assert idea["gain"] is None and idea["child_score"] == shaky.score


def test_public_implementations_are_ported_before_the_first_generation_and_judged_beside_the_start(tmp_path):
    pop, cost, rows, namer, _ = run(tmp_path, seeds={"vllm": "a = 7\n"}, generations=1, ports=2)
    assert rows[0]["iteration"] == 0 and rows[0]["named"] == 2 and rows[0]["examined"] == 2
    assert [p.kind for p in pop[:4]] == ["", "competitor", "competitor", "competitor"]
    assert pop[2].parent == pop[0].id and pop[2].plan == {"source": "lib.k0", "modification_plan": "port 0"}
    calls = lines(tmp_path / "calls.jsonl")
    assert [(c["role"], c["model"]) for c in calls[:3]] == [("ports", "opus"), ("write", "opus"), ("write", "opus")]
    ask = namer.calls[0][1].split(CACHE_SPLIT)[-1]
    assert "Name up to 2 public implementations" in ask and "Already ported and judged: vllm." in ask
    assert namer.calls[0][0] == "SETTING" + CACHE_SPLIT + "\n\n" + context.PORTER
    assert "last generation" not in namer.calls[1][1]                             # ports carry no prediction
    ports = [i for i in lines(tmp_path / "ideas.jsonl") if i["generation"] == 0]
    assert len(ports) == 2 and ports[0]["calls"].keys() == {"name", "write"} and ports[0]["reflection"] == ""


def test_builds_run_before_any_timing_and_the_compilers_report_stays_with_the_program():
    order = []

    def st(code, n):
        order.append(n)
        if n == 5:
            return {"metrics": {"correct": 0.0 if "nobuild" in code else 1.0}, "artifacts": {"compile_report": "R", "precheck": "ld"}}
        return stage(code, n)
    ev = Evaluator(st)
    ok, wrong, nobuild = ev.evaluate(["a = 3\n", "a = 4  # wrong\n", "a = 5  # nobuild\n"])
    assert order == [5, 5, 5, 2, 2] and ev.evals == 2                             # no correctness run before timing
    assert ok.examined and ok.artifacts["compile_report"] == "R" and ok.metrics["combined_score"] == 0.3
    assert wrong.fate == "failed correctness" and nobuild.fate == "precheck: ld" and not nobuild.examined
    assert ev.fails("a = 3\n") is None and ev.fails("a = 3  # shaky\n", 9) == "correctness_seen: seed 2 differs"
    assert Evaluator(lambda code, n: {"metrics": {"correct": 0.0}, "artifacts": {"error": "load failed"}}).fails("x") \
        == "load failed"


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


def test_a_named_change_is_matched_in_its_list_in_this_search_and_in_earlier_runs():
    class Embedder:
        def passages(self, texts):
            return [[1.0, 0.0] if "fuse" in t else [0.0, 1.0] if "tile" in t else [-1.0, 0.0] for t in texts]
    past = entry()
    s = diversity.Similar(Embedder(), [past])
    named = [{"optimisation_method": "fuse a"}, {"optimisation_method": "tile b"}, {"optimisation_method": "fuse c"},
             {"optimisation_method": "other"}]
    assert [(m["named"], m["where"], m["like"]) for m in s.match(named)] == [
        (0, "an earlier run", past), (2, "this list", 0), (2, "an earlier run", past)]
    done = Idea("i1", "p0", 1, {"optimisation_method": "tile x"})
    s.add([done])
    assert [(m["where"], m["like"], m["cosine"]) for m in s.match(named[1:2])] == [("this search", done, 1.0)]
    assert diversity.Similar().match(named) == [] and diversity.Similar(Embedder()).match([]) == []


def test_the_naming_and_review_calls_read_the_setting_every_idea_and_the_parents_with_their_facts():
    timings = json.dumps([{"shape": [4], "speedup": 1.2, "weight": 1, "pct_of_roofline": 40.0, "secret": 1}])
    root, mid = program(0, 1.0), program(1, 1.5, "p0", timings=timings, compile_report="REPORT", ncu_missing="no ncu")
    other, past = program(3, 2.0, "p0"), entry()
    pop = [root, mid, other]
    ideas = [Idea("i1", "p0", 1, {"optimisation_method": "fuse", "prediction": {"speedup": 1.3}}, fate=EXAM, child=mid),
             Idea("i2", "p0", 1, {"optimisation_method": "unroll"}, fate="not applied: no block")]
    system, user = context.name("SETTING", [mid, other], pop, ideas, [past], 2, 5)
    stable, rest = system.split(CACHE_SPLIT)
    assert stable.startswith("SETTING\n\n# Ideas tried on this task in earlier runs")
    assert '{"idea": "fuse loops", "tried": 1, "best_gain": 1.2}' in stable
    assert '"idea": "fuse", "gain": 1.5' in rest and '"furthest": "not applied: no block"' in rest
    assert rest.endswith(context.NAMER)
    shown, asked = user.split(CACHE_SPLIT)
    assert shown.startswith("The best score in the search is 2.0000.") and "# Program p1 (score 1.5000)" in shown
    assert "# Program p3 (score 2.0000)" in shown and "p0 (1.0000) -> p1 (1.5000)" in shown
    assert "# The starting program p0" in shown and "40.0% of its roofline bound" in shown and "REPORT" in shown
    assert "no ncu" in shown and "secret" not in shown
    assert '{"id": "i1", "idea": "fuse", "prediction": {"speedup": 1.3}, "gain": 1.5}' in asked
    assert "Name 5 changes" in asked and '"reflections"' in asked
    first = context.name("S", [root], [root], [], [], 1, 5)
    assert first[0] == "S" + CACHE_SPLIT + "\n\n" + context.NAMER
    assert not any(t in first[1] for t in ("# The starting program", "reflections", "last generation"))

    named = [{"parent": "p1", "optimisation_method": "a"}, {"parent": "p3", "optimisation_method": "b"}]
    matches = [{"named": 1, "like": 0, "where": "this list", "cosine": 0.95},
               {"named": 0, "like": ideas[0], "where": "this search", "cosine": 0.93},
               {"named": 0, "like": past, "where": "an earlier run", "cosine": 0.94}]
    again, user = context.review("SETTING", [mid, other], pop, ideas, [past], named, matches, 4)
    assert again == system and user.split(CACHE_SPLIT)[0] == shown               # read from the naming call's cache
    asked = user.split(CACHE_SPLIT)[1]
    assert 'n1: {"parent": "p1", "optimisation_method": "a"}' in asked and "Add up to 4 changes" in asked
    assert '{"named": "n2", "where": "this list", "cosine": 0.95, "like": "n1"}' in asked
    assert '"like": "fuse", "gain": 1.5}' in asked and '"like": "fuse loops", "tried": 1, "best_gain": 1.2}' in asked
    assert "close to another" not in context.review("S", [mid], pop, [], [], named, [], 4)[1]


def test_a_writer_call_reads_the_program_its_facts_the_change_and_the_programs_it_takes_from():
    mid, other = program(1, 1.5, "p0", compile_report="REPORT"), program(3, 2.0, "p0")
    system, user = context.write("SETTING", mid, {"optimisation_method": "a", "borrows": ["p3"]}, [other])
    assert system == "SETTING" + CACHE_SPLIT + "\n\n" + context.WRITER + "\n\n" + context.FORMAT
    shown, asked = user.split(CACHE_SPLIT)
    assert shown == context.program(mid) and "REPORT" in shown
    assert "# Program p3 (score 2.0000), which the change takes from\n```python\na = 3\n" in asked
    assert '# The change to make\n{"optimisation_method": "a", "borrows": ["p3"]}' in asked and asked.endswith(context.WRITE)


def test_earlier_runs_ideas_are_shown_the_most_tried_first_and_counted_past_the_cap():
    never = archive.Entry("idea-2", "toy", "t", {"optimisation_method": "tile"}, None,
                          [archive.Outcome("r", "p", 1.0, None, error="not applied: no block")] * 2)
    block = context.earlier([entry(), never])
    assert block.splitlines()[1] == '{"idea": "tile", "tried": 2, "furthest": "not applied: no block"}'
    assert context.earlier([]) == "" and "(5 more ideas not shown)" in context.earlier([entry()] * (context.EARLIER_MAX + 5))


def test_an_answers_json_is_read_best_effort():
    assert context.read('```json\n{"ideas": [{"parent": "p"}]}\n```') == {"ideas": [{"parent": "p"}]}
    assert context.read("just words") == {} and context.read("") == {} and context.read("[1]") == {}


if __name__ == "__main__":
    import inspect
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(*[Path(tempfile.mkdtemp()) for _ in inspect.signature(fn).parameters]); print("ok", name)
