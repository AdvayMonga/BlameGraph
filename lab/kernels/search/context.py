"""Context: everything a model call reads, built in one place, and how its answer is read back. Facts only.

Five calls: `ports` (once, before the first generation: the public implementations to port), `name` (one a
generation: the changes to make), `review` (one a generation: the final list, duplicates merged and combinations
added), `write` (one a change or port: the program it makes) and `reflect` (once, after the last generation). `name` and `review` share their system prompt and the programs shown, so the second reads them cached."""
from __future__ import annotations

import json
import re

from lab.kernels.search.llm import CACHE_SPLIT
from lab.kernels.search.program import Idea, Program

NAMER = ("You are directing a search for a faster GPU kernel program. You are shown the programs this generation "
         "builds on, each with its measurements, the starting program and every idea tried so far.")
PORTER = ("You are choosing starting programs for a search for a faster GPU kernel program. You are shown the program "
          "the search starts from and its measurements.")
WRITER = ("You are making one change to a GPU kernel program in a search. You are shown the program, its measurements "
          "and the change to make.")
FORMAT = """Answer with SEARCH/REPLACE blocks against the program shown. Each SEARCH section must match the program
exactly once, or the answer is not applied. Any number of blocks:

<<<<<<< SEARCH
old lines
=======
new lines
>>>>>>> REPLACE"""
CHANGE = ('"parent": "<the id of the program to change>", "borrows": [<ids of other programs shown whose changes it '
          'takes; may be empty>], "bottleneck": "<max 30 words>", "optimisation_method": "<max 35 words>", '
          '"modification_plan": "<max 60 words>", "prediction": {"speedup": <the changed program\'s score / its '
          'parent\'s>, "mechanism": [{"counter": "<an ncu metric name from the measurements>", "change_pct": '
          '<predicted % change, summed over the program\'s kernels>}]}')
REFLECTIONS = ('"reflections": [{"id": "<the idea\'s id>", "reflection": "<why what was measured matched or differed '
               'from its prediction, max 40 words>"}]')
WRITE = ("Make this change to the program. Answer with SEARCH/REPLACE blocks against the program shown. It is applied "
         "and measured.")
TIMING_KEYS = ("shape", "candidate_ms", "baseline_ms", "speedup", "speedup_p25", "speedup_p75", "beyond_noise",
               "pct_of_roofline", "bound", "pct_of_peak_bandwidth", "pct_of_peak_flops")
HISTORY_MAX = 400            # rows of the run's ideas shown; the rest are counted
TEXT = 240                   # characters of an idea's text shown


def ports(setting: str, start: Program, ported: list[str], n: int) -> tuple[str, str]:
    """(system, user) for the call before the first generation: the public implementations to port, `ported` aside."""
    ask = (f"Name up to {n} public implementations of this operation to port as further starting programs of the "
           "search: those shown above, or others you find. A port is this program with its kernel replaced by that "
           "implementation's, computing the same result through the same entry point. Each is written, applied and "
           "judged beside this program." + (f" Already ported and judged: {', '.join(ported)}." if ported else "")
           + ' Answer with one JSON object and nothing else: {"ideas": [{"parent": "<this program\'s id>", "source": '
           '"<the library, its kernel and where its source is>", "modification_plan": "<what replaces what, max 60 '
           'words>"}]}')
    return setting + CACHE_SPLIT + "\n\n" + PORTER, program(start) + CACHE_SPLIT + "\n\n" + ask


def name(setting: str, parents: list[Program], population: list[Program], ideas: list[Idea], generation: int,
         n: int) -> tuple[str, str]:
    """(system, user) for a generation's naming call: reflections on the last generation's ideas, then `n` changes."""
    done = last(ideas, population, generation)
    ask = (f"Name {n} changes for a writer to make, each to one of the programs shown. Each must differ from the "
           "others and from every idea already tried in this search. A change may take from other programs shown "
           '("borrows"). Each is written, applied and measured.'
           + (" Before them, reflect on each of the last generation's ideas." if done else "")
           + " Answer with one JSON object and nothing else: {" + (REFLECTIONS + ", " if done else "")
           + '"ideas": [{' + CHANGE + "}]}")
    return _system(setting, ideas, population), programs(parents, population) + CACHE_SPLIT + "".join(
        f"\n\n{t}" for t in (done, ask) if t)


def review(setting: str, parents: list[Program], population: list[Program], ideas: list[Idea],
           named: list[dict], matches: list[dict], merges: int) -> tuple[str, str]:
    """(system, user) for a generation's review call: the named changes, and those whose text is close to another
    idea's, to be written as the final list."""
    score = {p.id: p.score for p in population}
    rows = "\n".join(f"n{k}: {json.dumps(p)}" for k, p in enumerate(named, 1))
    near = "\n".join(_match(m, score) for m in matches)
    ask = ("Write this generation's final list of changes. Keep each named change; where several are the same change, "
           f"write it once. Add up to {merges} changes that each combine named changes which can be made to one "
           'program together. Give every change the named changes it keeps or combines in "from". Answer with one '
           'JSON object and nothing else: {"ideas": [{"from": ["n1"], ' + CHANGE + "}]}")
    return _system(setting, ideas, population), programs(parents, population) + CACHE_SPLIT + "".join(
        f"\n\n{t}" for t in (f"# The changes named for this generation\n{rows}",
                             "# Named changes whose text is close to another idea's (cosine of their embeddings)\n"
                             + near if near else "", ask) if t)


def write(setting: str, parent: Program, plan: dict, borrowed: list[Program]) -> tuple[str, str]:
    """(system, user) for one writer call: the program, the change, and the programs the change takes from."""
    others = "".join(f"\n\n# Program {b.id} (score {b.score:.4f}), which the change takes from\n"
                     f"```python\n{b.code}\n```" for b in borrowed)
    return (setting + CACHE_SPLIT + "\n\n" + WRITER + "\n\n" + FORMAT,
            program(parent) + CACHE_SPLIT + others + f"\n\n# The change to make\n{json.dumps(plan)}\n\n{WRITE}")


def reflect(setting: str, population: list[Program], ideas: list[Idea], generation: int) -> tuple[str, str]:
    """(system, user) for the call after the last generation: reflections on its ideas, and one on the run."""
    return _system(setting, ideas, population), last(ideas, population, generation) + (
        "\n\nReflect on each of these ideas, and on the run. Answer with one JSON object and nothing else: {"
        + REFLECTIONS + ', "run": "<what the run gained and what limited it, max 60 words>"}')


def read(text: str) -> dict:
    """The answer's JSON object; of one cut off, the reflections and ideas that are whole; {} when it holds none."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        v = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        v = {k: items for k in ("reflections", "ideas") if (items := _whole(text, k))}
    return v if isinstance(v, dict) else {}


def _whole(text: str, key: str) -> list:
    """The complete items of the answer's list `key`."""
    m, items = re.search(rf'"{key}"\s*:\s*\[', text), []
    at = m.end() if m else 0
    while m:
        try:
            v, at = json.JSONDecoder().raw_decode(text, at + re.match(r"[\s,]*", text[at:]).end())
        except json.JSONDecodeError:
            break
        items.append(v)
    return items


def short(plan: dict) -> str:
    t = " / ".join(str(plan[k]) for k in ("optimisation_method", "modification_plan") if plan.get(k)) \
        or json.dumps(plan, sort_keys=True)
    return t if len(t) <= TEXT else t[:TEXT] + "..."


def outcome(i: Idea, score: dict) -> dict:
    """What was measured of an idea: its gain over its parent, or how far it got."""
    measured = i.child and i.child.metrics.get("correct") == 1.0 and score.get(i.parent)
    return {"gain": round(i.child.score / score[i.parent], 4)} if measured else {"furthest": i.fate[:120]}


def tried(e) -> dict:
    """What earlier runs measured of one distinct idea (an archive entry)."""
    gains = [o.child_score / o.parent_score for o in e.outcomes if o.child_score and o.parent_score]
    return {"tried": len(e.outcomes), **({"best_gain": round(max(gains), 4)} if gains else
                                         {"furthest": next((o.error for o in e.outcomes[::-1] if o.error),
                                                           "not measured")[:120]})}


def _system(setting: str, ideas: list[Idea], population: list[Program]) -> str:
    """The setting (stable for the run), then this run's ideas and the role."""
    past_run = history(ideas, population)
    return setting + CACHE_SPLIT + "\n\n" + (f"{past_run}\n\n" if past_run else "") + NAMER


def history(ideas: list[Idea], population: list[Program]) -> str:
    """Every idea of the run so far with what was measured."""
    if not ideas:
        return ""
    score = {p.id: p.score for p in population}
    rows = [json.dumps({"generation": i.generation, "idea": short(i.plan), **outcome(i, score)}) for i in ideas]
    more = [f"({len(rows) - HISTORY_MAX} more ideas not shown)"] if len(rows) > HISTORY_MAX else []
    return "\n".join(["# Every idea tried in this search, with what was measured (gain = child score / parent score)",
                      *rows[:HISTORY_MAX], *more])


def last(ideas: list[Idea], population: list[Program], generation: int) -> str:
    """The generation before `generation`: each idea with its prediction and what was measured (ports have none)."""
    score = {p.id: p.score for p in population}
    rows = [json.dumps({"id": i.id, "idea": short(i.plan), "prediction": i.plan.get("prediction"), **outcome(i, score)})
            for i in ideas if i.generation == generation - 1 > 0]
    return "\n".join(["# The last generation's ideas, each with its prediction and what was measured", *rows]) \
        if rows else ""


def _match(m: dict, score: dict) -> str:
    """One close pair: the named change, where the idea like it is from, and what was measured of that idea."""
    x = m["like"]
    like = {"like": f"n{x + 1}"} if isinstance(x, int) else {"like": short(x.plan), **outcome(x, score)} \
        if isinstance(x, Idea) else {"like": short(x.idea), **tried(x)}
    return json.dumps({"named": f"n{m['named'] + 1}", "where": m["where"], "cosine": m["cosine"], **like})


def lineage(node: Program, population: list[Program]) -> list[Program]:
    by, out = {p.id: p for p in population}, []
    while node:
        out.append(node)
        node = by.get(node.parent)
    return out[::-1]


def programs(parents: list[Program], population: list[Program]) -> str:
    """The generation's parents, each with its measurements, lineage and children, and the starting program."""
    root, best = population[0], max(p.score for p in population)

    def one(node: Program) -> str:
        path = " -> ".join(f"{n.id} ({n.score:.4f})" for n in lineage(node, population))
        kids = "\n".join(f"- {c.id}: score {c.score:.4f}" for c in population if c.parent == node.id) or "none"
        return (f"{program(node)}\n\n# Its lineage (program, score)\n{path}\n\n"
                f"# Programs already derived from it\n{kids}")
    start = "" if root in parents else \
        f"\n\n# The starting program {root.id} (score {root.score:.4f})\n```python\n{root.code}\n```"
    return f"The best score in the search is {best:.4f}.\n\n" + "\n\n".join(one(p) for p in parents) + start


def program(node: Program) -> str:
    """A program and its measurements."""
    return (f"# Program {node.id} (score {node.score:.4f})\n```python\n{node.code}\n```\n\n"
            f"# Its measurements\n{facts(node)}")


def facts(node: Program) -> str:
    """A program's measurements as the evaluator returned them: gates, per-shape timings, error against the
    reference, ncu counters, the compiler's report."""
    a, out = node.artifacts, [f"metrics: {json.dumps(node.metrics)}"]
    if a.get("error"):
        out.append(f"error: {a['error'][-2000:]}")
    if a.get("gates"):
        out.append("gates: " + "; ".join(f"{k} {'passed' if v['passed'] else 'FAILED: ' + v['reason']}"
                                         for k, v in json.loads(a["gates"]).items()))
    if a.get("timings"):
        rows = json.loads(a["timings"])
        w = sum(r.get("weight") or 0 for r in rows if r.get("pct_of_roofline") is not None)
        if w:
            pct = sum((r.get("weight") or 0) * r["pct_of_roofline"] for r in rows
                      if r.get("pct_of_roofline") is not None) / w
            out.append(f"{pct:.1f}% of its roofline bound, call-weighted")
        out.append("timings (per seen shape, paired with the baseline):")
        out += [json.dumps({k: r.get(k) for k in TIMING_KEYS}) for r in rows]
    if a.get("error_vs_reference"):
        out.append(f"error against the reference: {a['error_vs_reference']}")
    if counters(node):
        out.append("ncu counters (one call each on the smallest and the largest seen shape, longest kernels first):\n"
                   + counters(node))
    if a.get("compile_report"):
        out.append(f"the compiler's report:\n{a['compile_report']}")
    return "\n".join(out)


def counters(node: Program) -> str:
    """The top kernels of one call with their ncu counters, or why there are none."""
    a = node.artifacts
    if a.get("ncu"):
        d, lines = json.loads(a["ncu"]), []
        for p in d.get("shapes") or [{"shape": None, **d}]:
            head = f"shape {p['shape']}: " if p.get("shape") is not None else ""
            lines += [head + json.dumps(k) for k in p.get("kernels") or []] or [head + (p.get("error") or "none recorded")]
        return "\n".join(lines)
    if a.get("ncu_missing"):
        return a["ncu_missing"]
    return "ncu failed: " + a["ncu_error"][-300:] if a.get("ncu_error") else ""
