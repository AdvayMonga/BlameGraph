"""Evals: every program is built and timed; only a program picked as a parent, or about to be promoted, takes the
full correctness check (the jailed evaluator's stages)."""
from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from lab.kernels.search.program import Program

TIMED, COUNTERS, PRECHECK, CHECK, WINNER = 2, 3, 5, 8, 9   # the jailed evaluator's stages
EXAM = "full exam"
REFUSED = "ERR_NVGPUCTRPERM"                     # ncu: this host does not let the jail read GPU performance counters
JAILS = 4                                        # build checks or full checks at once (no timing runs meanwhile)


def jailed(code: str, stage: int) -> dict:
    """{metrics, artifacts} for one stage of one program."""
    from lab.kernels.eval import evaluator
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    try:
        return evaluator.judge_stage(f.name, stage)
    finally:
        Path(f.name).unlink()


@dataclass
class Result:
    metrics: dict
    artifacts: dict
    fate: str                                    # EXAM, or why the program never reached it

    @property
    def examined(self) -> bool:
        return self.fate == EXAM


class Evaluator:
    def __init__(self, stage: Callable[[str, int], dict] = jailed, counters: bool = False):
        self.stage, self.counters, self.evals = stage, counters, 0

    def evaluate(self, codes: list[str]) -> list[Result]:
        """Each program's result, in order: build checks in parallel jails first, then timings one at a time. The
        timing run checks one seed of every seen shape and each timed output, so a wrong program is not scored."""
        with ThreadPoolExecutor(max_workers=JAILS) as ex:
            built = list(ex.map(self._build, codes))
        return [r if r.fate else self._time(code, r) for code, r in zip(codes, built)]

    def _build(self, code: str) -> Result:
        r = self.stage(code, PRECHECK)
        if r["metrics"].get("correct") == 0.0:
            return Result(r["metrics"], r["artifacts"], "precheck: " + r["artifacts"].get("precheck", "")[:300])
        report = r["artifacts"].get("compile_report")
        return Result({}, {"compile_report": report} if report else {}, "")

    def _time(self, code: str, built: Result) -> Result:
        r = self.stage(code, TIMED)
        self.evals += 1
        return Result(r["metrics"], {**r["artifacts"], **built.artifacts},
                      EXAM if r["metrics"].get("correct") == 1.0 else "failed correctness")

    def fails(self, code: str, stage: int = CHECK) -> str | None:
        """Why a program fails the full check (every seen shape on three seeds, twenty at WINNER; edge shapes,
        layouts, other value scales), or None."""
        r = self.stage(code, stage)
        if r["metrics"].get("correct") == 1.0:
            return None
        gates = json.loads(r["artifacts"].get("gates") or "{}")
        return next((f"{k}: {g['reason']}" for k, g in gates.items() if not g["passed"]),
                    r["artifacts"].get("error") or "the check did not pass")[:300]

    def check(self, programs: list[Program]) -> dict[str, str]:
        """{id: why} for the programs that fail the full check, in parallel jails; each program takes it once, and
        one that fails can no longer be a parent."""
        new = [p for p in programs if "parent_check" not in p.artifacts]
        with ThreadPoolExecutor(max_workers=JAILS) as ex:
            for p, why in zip(new, ex.map(lambda p: self.fails(p.code), new)):
                p.artifacts["parent_check"] = why or "passed"
                if why:
                    p.metrics["correct"] = 0.0
        return {p.id: p.artifacts["parent_check"] for p in programs if p.artifacts["parent_check"] != "passed"}

    def profile(self, program: Program) -> None:
        """ncu counters into the program's artifacts, once; never asked for again after the host refuses them."""
        if not self.counters or any(k in program.artifacts for k in ("ncu", "ncu_missing", "ncu_error")):
            return
        a = self.stage(program.code, COUNTERS)["artifacts"]
        if REFUSED in json.dumps(a):
            self.counters = False
            a = {"ncu_missing": f"ncu counters unavailable on this host ({REFUSED}); not collected this run"}
        program.artifacts.update(a)
