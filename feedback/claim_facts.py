"""Claim vs evidence in a lab ledger: the agent's `claim` text checked against recorded results. Regex only, no LLM.

A claim is about the snapshots and record ids it names, else about its own record's snapshot. Fact types:
  unrecorded_number  a number with a unit or metric name that matches no value in this run's recorded results
  unknown_record     a cited record id that is not a record of this run
  contradicted       a verification word about a snapshot whose completed records of that kind all fail to show it
  no_record          a verification word about a snapshot with no completed record of that kind
"""
from __future__ import annotations

import re

from diagnostics.claims_audit import _matches
from feedback.lab_verdict import metrics, passed, refused

REC_RE = re.compile(r"\bev-\d{8}-[0-9a-f]{12}\b")
HEX_RE = re.compile(r"\b[0-9a-f]{7,64}\b")
NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w]|\.\d)")
UNIT_RE = re.compile(r"\s*(%|ms\b|s\b|sec\b|x\b|×|tok(?:ens)?/s|req/s|rps\b|gb\b|mb\b)", re.I)
METRIC_WORDS = r"ttft|tpot|itl|latency|throughput|goodput|p50|p90|p99|speedup|delta"
VERIFY_RE = re.compile(r"\b(verified|tested|pass(?:es|ed)?|equivalent|faster|speed-?up)\b", re.I)
NEG_RE = re.compile(r"(\bnot|\bnever|\bno|n't|\bwithout)\W+(\w+\W+)?$", re.I)
KINDS_OF = {"verified": ("test", "equiv", "bench"), "tested": ("test",), "passes": ("test", "equiv"),
            "equivalent": ("equiv",), "faster": ("bench",), "speedup": ("bench",)}
SHOWS = {"test": "passed lint and tests", "equiv": "passed", "bench": "has a metric with verdict 'win'"}


def _text(c) -> str:
    if isinstance(c, dict):
        return " ; ".join(f"{k}: {_text(v)}" for k, v in c.items())
    if isinstance(c, list):
        return " ; ".join(map(_text, c))
    return "" if c is None else str(c)


def _leaves(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    elif isinstance(x, list):
        for v in x:
            yield from _leaves(v)
    else:
        yield x


def _recorded(recs: list[dict]) -> list[float]:
    """Absolute values of every number in recorded results (also base/new ratios and fractions as percent)."""
    out = []
    for r in recs:
        for v in _leaves([r.get("result"), r.get("metrics")]):
            if isinstance(v, str):
                out += [float(n) for n in NUM_RE.findall(v)]
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(abs(v))
        for m in metrics(r).values():
            b, n = m.get("base"), m.get("new")
            if isinstance(b, (int, float)) and isinstance(n, (int, float)) and b and n:
                out += [abs(b / n), abs(n / b)]
    return out + [v * 100 for v in out]


def _supports(r: dict, named: set[str]) -> bool:
    if r["kind"] == "bench":
        return any(m.get("verdict") == "win" for k, m in metrics(r).items() if not named or k in named)
    return passed(r) is True


def claim_facts(recs: list[dict]) -> list[dict]:
    """Facts about every agent claim in `recs` (one run). Human-authored findings are skipped."""
    by_id = {r.get("id"): r for r in recs}
    known = {r.get("snapshot") for r in recs if r.get("snapshot")}
    names = {k for r in recs for k in metrics(r)}
    metric_re = re.compile("|".join([METRIC_WORDS] + [re.escape(n) for n in sorted(names)]), re.I)
    seen = _recorded(recs)
    out = []
    for r in recs:
        text = _text(r.get("claim"))
        if not text.strip() or r.get("author") == "human":
            continue
        rid = r.get("id")
        cited = REC_RE.findall(text)
        bare = REC_RE.sub(" ", text)
        snaps = {s for tok in HEX_RE.findall(bare) for s in known if s.startswith(tok)}
        snaps |= {by_id[i].get("snapshot") for i in cited if i in by_id}
        snaps = sorted(s for s in snaps if s) or [r.get("snapshot")]
        for i in cited:
            if i not in by_id:
                out.append({"type": "unknown_record", "record": rid, "snapshots": snaps,
                            "fact": f"{rid} cites {i}, which is not a record of this run"})
        t = HEX_RE.sub(" ", bare)
        for m in NUM_RE.finditer(t):
            unit = UNIT_RE.match(t, m.end())
            u = unit.group(1).lower() if unit else ""
            if not unit and not metric_re.search(t[max(0, m.start() - 25):m.start()]):
                continue
            if u in ("x", "×", "s") and float(m.group()) in (0.0, 1.0, 2.0):
                continue   # "2x H100", "1 s": too generic to audit
            if not _matches(float(m.group()), m.group(), seen):
                out.append({"type": "unrecorded_number", "record": rid, "snapshots": snaps,
                            "fact": f"{m.group()}{u} in the claim of {rid} matches no value in this run's recorded results"})
        named = {n for n in names if n.lower() in text.lower()}
        done = set()
        for w in VERIFY_RE.finditer(text):
            if NEG_RE.search(text[max(0, w.start() - 25):w.start()]):
                continue
            word = w.group(1).lower().replace("-", "")
            word = "passes" if word.startswith("pass") else word
            for s in snaps:
                if (word, s) in done:
                    continue
                done.add((word, s))
                kinds = KINDS_OF[word]
                ev = [x for x in recs if x["kind"] in kinds and x.get("snapshot") == s and not refused(x)
                      and (x["kind"] != "bench" or passed(x))]
                if not ev:
                    out.append({"type": "no_record", "record": rid, "snapshots": [s],
                                "fact": f"{rid} says '{w.group(1)}' about snapshot {s}; no completed "
                                        f"{' or '.join(kinds)} record exists for it"})
                elif not any(_supports(x, named) for x in ev):
                    out.append({"type": "contradicted", "record": rid, "snapshots": [s], "evidence": [x.get("id") for x in ev],
                                "fact": f"{rid} says '{w.group(1)}' about snapshot {s}; none of its {len(ev)} "
                                        f"{'/'.join(kinds)} record(s) {' or '.join(SHOWS[k] for k in kinds)}"})
    return out
