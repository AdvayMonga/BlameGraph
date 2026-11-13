"""Evals: one call judges a batch of programs through the jailed evaluator (build check, correctness, timing)."""
from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from generate.kernels.search.program import Program

CORRECT, TIMED, COUNTERS, PRECHECK = 1, 2, 3, 5   # the jailed evaluator's stages
EXAM = "full exam"
REFUSED = "ERR_NVGPUCTRPERM"                     # ncu: this host does not let the jail read GPU performance counters
JAILS = 4                                        # build and correctness checks at once (no timing runs meanwhile)


def jailed(code: str, stage: int) -> dict:
    """{metrics, artifacts} for one stage of one program."""
    from evals.kernels import evaluator
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
        """Each program's result, in order: checks in parallel jails first, then timings one at a time."""
        with ThreadPoolExecutor(max_workers=JAILS) as ex:
            checked = list(ex.map(self._check, codes))
        return [r if r.fate else self._time(code, r) for code, r in zip(codes, checked)]

    def _check(self, code: str) -> Result:
        built = self.stage(code, PRECHECK)
        if built["metrics"].get("correct") == 0.0:
            return Result(built["metrics"], built["artifacts"], "precheck: " + built["artifacts"].get("precheck", "")[:300])
        r = self.stage(code, CORRECT)
        if r["metrics"].get("correct") != 1.0:
            return Result(r["metrics"], r["artifacts"], "failed correctness")
        report = built["artifacts"].get("compile_report")
        return Result({}, {"compile_report": report} if report else {}, "")

    def _time(self, code: str, checked: Result) -> Result:
        r = self.stage(code, TIMED)
        self.evals += 1
        return Result(r["metrics"], {**r["artifacts"], **checked.artifacts}, EXAM)

    def profile(self, program: Program) -> None:
        """ncu counters into the program's artifacts, once; never asked for again after the host refuses them."""
        if not self.counters or any(k in program.artifacts for k in ("ncu", "ncu_missing", "ncu_error")):
            return
        a = self.stage(program.code, COUNTERS)["artifacts"]
        if REFUSED in json.dumps(a):
            self.counters = False
            a = {"ncu_missing": f"ncu counters unavailable on this host ({REFUSED}); not collected this run"}
        program.artifacts.update(a)
