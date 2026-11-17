"""lab run: the commands of a run in order, nothing started when something is absent, the VM always deleted."""
import os

os.environ.setdefault("LAB_TARGET", "evals/targets/inference-server.toml")

from lab import run


def test_a_run_is_its_commands_in_order_and_the_vm_is_always_deleted(monkeypatch):
    done: list = []
    monkeypatch.setattr(run, "absent", lambda budget: [])
    monkeypatch.setattr(run, "lab", lambda step: done.append(step) or 0)
    monkeypatch.delenv("LAB_WORKER", raising=False)
    assert run.main(["--task", "t.toml", "--budget", "20"]) == 0
    run_id = done[3][-1]
    assert done == [["vm", "start", "--budget", "20"], ["vm", "setup"], ["setup", "--task", "t.toml"],
                    ["session", "--task", "t.toml", "--budget", "20", "--run", run_id],
                    ["feedback", str(run.ledger.ROOT), "--run", run_id], ["store", "push"], ["vm", "stop"]]
    assert run_id.startswith("run-") and os.environ["LAB_WORKER"] == "ssh"
    # a failing step ends the run with its code; the VM is still deleted
    done.clear()
    monkeypatch.setattr(run, "lab", lambda step: done.append(step) or (3 if step[0] == "setup" else 0))
    assert run.main(["--task", "t.toml", "--budget", "20"]) == 3
    assert [s[0] for s in done] == ["vm", "vm", "setup", "vm"] and done[-1] == ["vm", "stop"]
    # so is it when a step is interrupted
    done.clear()

    def interrupted(step):
        done.append(step)
        if step[0] == "session":
            raise KeyboardInterrupt
        return 0
    monkeypatch.setattr(run, "lab", interrupted)
    try:
        run.main(["--task", "t.toml", "--budget", "20"])
    except KeyboardInterrupt:
        pass
    assert done[-1] == ["vm", "stop"]


def test_nothing_is_started_when_something_is_absent(monkeypatch, capsys):
    done: list = []
    monkeypatch.setattr(run, "lab", lambda step: done.append(step) or 0)
    monkeypatch.setattr(run, "absent", lambda budget: ["no GPU rate: set LAB_GPU_USD_PER_HOUR"])
    assert run.main(["--task", "t.toml", "--budget", "20"]) == 2 and done == []
    assert "no GPU rate" in capsys.readouterr().err


def test_absent_adds_the_gpu_rate_to_what_a_session_needs(monkeypatch):
    from lab import session
    monkeypatch.setattr(session, "preflight", lambda provider=None: ["a"])
    monkeypatch.delenv("LAB_GPU_USD_PER_HOUR", raising=False)
    assert run.absent(20)[0] == "a" and "no GPU rate" in run.absent(20)[1]
    monkeypatch.setenv("LAB_GPU_USD_PER_HOUR", "4")
    assert run.absent(20) == ["a"]
