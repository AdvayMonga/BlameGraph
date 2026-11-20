"""The loop with fakes of the worker, the model, the ways, the pull request and the VM: a win is pushed and becomes
the best, everything else is recorded and left, the turn and budget rules, and the VM stopped at every end."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lab.budget import Budget
from lab.loop import calibrate as C
from lab.loop import loop, record
from lab.loop.pick import pick
from lab.loop.push import push
from lab.loop.turn import CHANGE, KERNEL, turn
from lab.loop.types import Candidate, Gap, Measurement, Pushed
from lab.records import ledger
from lab.records.remote import DirRemote
from lab.task import Task

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_fixtures import make_repo  # noqa: E402
from loop_fakes import FakeWorker, make_peaks, make_target  # noqa: E402

TASK = Task("faster", ("saturated",))


class Run:
    """One loop run over fakes; `made`: the tree name each turn's way returns (None: no candidate)."""

    def __init__(self, tmp_path, monkeypatch, made, values=None, failing=(), **plan):
        self.tmp, self.made, self.pushed, self.stopped, self.gaps = tmp_path, list(made), [], 0, []
        self.wk = FakeWorker(values, failing)
        self.t = make_target(tmp_path, make_repo(tmp_path))
        self.remote, self.root = DirRemote(tmp_path / "store"), tmp_path / "ledger"
        peaks_dir = make_peaks(tmp_path)
        real = C.calibrate
        monkeypatch.setattr(loop, "calibrate", lambda *a: real(*a, peaks_dir))
        self.plan = loop.plan("run-1", plan.get("n"), plan.get("budget"), plan.get("rate", 0.0))

    def _pick(self, gaps, best, budget):
        self.gaps.append([g.layer for g in gaps])
        return gaps[0]

    def _turn(self, gap, best, work, budget):
        name = self.made.pop(0)
        if name is None:
            return None
        if name == "boom":
            raise RuntimeError("the way broke")
        budget.spent_usd += 2.0
        (self.tmp / name).mkdir(exist_ok=True)
        return Candidate(self.tmp / name, f"diff of {name}", f"make it {name} (on {best.tree.name})")

    def _push(self, c, body, branch, onto):
        self.pushed.append((c.note, branch, onto))
        return Pushed(branch, f"https://example.test/pr/{len(self.pushed)}", "blobs/x.patch")

    def _stop(self):
        self.stopped += 1

    def go(self):
        parts = loop.Parts(self._pick, self._turn, self._push, self._stop)
        return loop.run(self.plan, TASK, self.t, self.wk, parts, self.remote, self.tmp / "work", "main", self.root)


def test_a_win_is_pushed_and_the_next_turn_builds_on_it(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, ["fast", "same", "faster"], {"fast": 150.0, "same": 150.5, "faster": 200.0}, n="3")
    recs = r.go()
    assert [x["kind"] for x in recs] == ["baseline", "turn", "turn", "turn"]
    assert [x["result"]["win"] for x in recs[1:]] == [True, False, True]
    assert r.pushed == [("make it fast (on base)", "perf/run-1-1", "main"),
                        ("make it faster (on fast)", "perf/run-1-3", "perf/run-1-1")]
    assert recs[1]["patch"] == "diff of fast" and recs[1]["result"]["pr"]["url"].endswith("/1")
    assert recs[2]["result"]["score"]["verdicts"] == {"saturated": "within_band"}
    assert recs[3]["result"]["score"]["gains_pct"]["saturated"] == pytest.approx((200 - 150) / 150 * 100)
    assert recs[1]["cost"]["usd"] == 2.0 and r.stopped == 1
    assert [x["id"] for x in ledger.records(r.root, run="run-1")] == [x["id"] for x in recs]
    st = record.status(r.remote, "run-1")
    assert st["state"] == "done" and st["wins"] == 2 and st["turn"] == 3 and record.last_run(r.remote) == "run-1"
    text = (tmp_path / "store" / record.PREFIX / "run-1" / "report.md").read_text()
    assert "turn 1 [kernels]: make it fast (on base): saturated +50.0%: win, https://example.test/pr/1" in text
    assert "turn 2" in text and "no win" in text and r.remote.keys("ledger/")


def test_wrong_slower_missing_and_broken_candidates_are_recorded_and_not_pushed(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, ["wrong", "slow", None, "boom"], {"wrong": 500.0, "slow": 50.0},
            failing=("wrong",), n="4")
    recs = r.go()
    wrong, slow, none, boom = (x["result"] for x in recs[1:])
    assert not wrong["correct"] and "score" not in wrong and not wrong["win"]
    assert slow["score"]["verdicts"] == {"saturated": "regressed"} and not slow["win"]
    assert "candidate" not in none and none["gap"]["layer"] == "kernels"
    assert boom["error"] == "RuntimeError: the way broke"
    assert r.pushed == [] and r.stopped == 1 and record.status(r.remote, "run-1")["wins"] == 0
    text = (tmp_path / "store" / record.PREFIX / "run-1" / "report.md").read_text()
    assert "incorrect: kl too high" in text and "no candidate" in text and "failed: RuntimeError" in text


def test_no_flags_is_one_turn_and_loop_runs_until_the_budget_is_spent(tmp_path, monkeypatch):
    assert loop.plan("r", None, None).turns == 1 and loop.plan("r", None, 500.0).turns == 1
    assert loop.plan("r", "4", None).turns == 4 and loop.plan("r", "loop", 5.0).turns is None
    for bad in (("loop", None), ("0", None), ("many", None)):
        with pytest.raises(ValueError):
            loop.plan("r", *bad)
    r = Run(tmp_path, monkeypatch, ["a"] * 10, n="loop", budget=5.0)
    recs = r.go()                                           # each turn's way spends $2
    assert len(recs) == 1 + 3 and record.status(r.remote, "run-1")["stopped"] == "budget spent"


def test_a_budget_stops_a_counted_run_early(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, ["a"] * 10, n="8", budget=3.0)
    assert len(r.go()) == 1 + 2 and r.stopped == 1


def test_a_run_that_fails_is_finished_and_its_vm_stopped(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, [])
    monkeypatch.setattr(loop, "calibrate", lambda *a: (_ for _ in ()).throw(RuntimeError("no GPU")))
    with pytest.raises(RuntimeError):
        r.go()
    st = record.status(r.remote, "run-1")
    assert st["state"] == "failed" and st["reason"] == "RuntimeError: no GPU" and r.stopped == 1

    class Down(DirRemote):
        def put(self, key, path):
            raise OSError("store down")
    r = Run(tmp_path / "b", monkeypatch, [])
    r.remote = Down(tmp_path / "b" / "store")
    with pytest.raises(OSError):
        r.go()
    assert r.stopped == 1


def test_turn_takes_the_kernel_way_for_the_kernels_gap_and_the_change_way_otherwise(tmp_path):
    ways = {KERNEL: lambda g, b, w, bu: Candidate(w, note="kernel"), CHANGE: lambda g, b, w, bu: Candidate(w, note="change")}
    base, budget = Candidate(tmp_path), Budget(1.0)
    assert turn(Gap("kernels", 40, None, 40), base, tmp_path, budget, ways).note == "kernel"
    assert turn(Gap("runtime", 40, None, 40), base, tmp_path, budget, ways).note == "change"


def test_pick_takes_the_gap_the_model_names_and_charges_the_call():
    gs = (Gap("kernels", 40, None, 40), Gap("runtime", 30, None, 30))
    best, budget, seen = Measurement(True, {}, {"saturated": {"value": 1.0}}), Budget(10.0), []

    def ask(system, user):
        seen.append(user)
        return "`runtime`\n", 0.25
    assert pick(gs, best, budget, ask).layer == "runtime" and budget.spent_usd == 0.25
    assert '"recoverable_pct": 30' in seen[0] and '"saturated"' in seen[0]
    assert pick(gs, best, budget, lambda s, u: ("the serving layer", 0.0)).layer == "kernels"


def test_push_commits_the_patch_on_a_branch_from_the_last_win_and_opens_a_labelled_pr(tmp_path):
    repo = make_repo(tmp_path)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["branch", "-M", "main"], check=True)
    (repo / "src/inference_server/engine.py").write_text("def speed():\n    return 2\n")
    patch = subprocess.run(git + ["diff"], capture_output=True, text=True, check=True).stdout
    subprocess.run(git + ["checkout", "-q", "."], check=True)
    calls = []

    def sh(argv, cwd, stdin=None):
        calls.append(argv)
        if argv[0] == "gh":
            return "https://github.com/o/r/pull/7"
        if argv[:2] == ["git", "push"]:
            return ""
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                               *argv[1:]], cwd=cwd, input=stdin, text=True, capture_output=True, check=True).stdout
    got = push(Candidate(tmp_path, patch, "speed 2"), "body", repo, "perf/run-1-1", "main", tmp_path / "ledger", sh)
    assert got.branch == "perf/run-1-1" and got.url.endswith("/pull/7")
    assert (tmp_path / "ledger" / got.patch).read_text() == patch
    shown = subprocess.run(git + ["show", "perf/run-1-1:src/inference_server/engine.py"], capture_output=True, text=True)
    assert "return 2" in shown.stdout
    head = subprocess.run(git + ["rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == "main" and "worktree" not in subprocess.run(git + ["worktree", "list"], capture_output=True, text=True).stdout.split("\n", 1)[1]
    gh = next(c for c in calls if c[0] == "gh")
    assert gh[gh.index("--base") + 1] == "main" and gh[gh.index("--label") + 1] == "perf" and "merge" not in gh
