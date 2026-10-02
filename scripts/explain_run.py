"""Human-readable narrative of one run's experiment log. Usage: explain_run.py run_0003 [--text]"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blamegraph.assertions import evaluate  # noqa: E402
from blamegraph.blame import blame  # noqa: E402
from blamegraph.experiment import build_log, unwrap_shell  # noqa: E402
from blamegraph.exploration import knob_values  # noqa: E402
from blamegraph.extract import obs_by_run  # noqa: E402
from blamegraph.noise import _primary  # noqa: E402
from blamegraph.traces import BASELINE_METRIC, DATA_ROOT, load_run  # noqa: E402


def main(run_id: str, show_text: bool = False):
    run = load_run(DATA_ROOT / "runs" / run_id)
    log = build_log(run, llm_obs=obs_by_run().get(run_id))
    steps = {s.i: s for s in run.steps()}
    base = BASELINE_METRIC[run.scenario]
    print(f"# {run_id}  {run.agent} ({run.harness})  scenario {run.scenario}  "
          f"scored={run.scored} speedup={run.speedup:.2f}x  flagged={run.flagged}  gate={run.gate_passed}\n")

    # timeline
    events = []
    for c in log.configs:
        kv = knob_values(c)
        events.append((c.step_i, f"CONFIG v{c.idx} [{c.engine}] " + ", ".join(f"{k}={v}" for k, v in sorted(kv.items()) if v) ))
    for si, idx in log.server_starts:
        events.append((si, f"START server (config v{idx})"))
    for e in log.evals:
        tag = "quick" if e.quick else "FULL"
        flags = [t for t, on in (("stale", e.config_stale), ("non-standard", not e.standard), ("warm", e.warm),
                                 ("killed", e.killed), ("unobserved", not e.observed)) if on]
        events.append((e.step_i, f"EVAL {tag} on config v{e.config_idx}" + (f"  ({', '.join(flags)})" if flags else "")))
    for o in log.observations:
        v = _primary(o, run.scenario)
        parts = [f"{v/base:.2f}x" if v else "", f"ttft {o.ttft_p50:.3f}s" if o.ttft_p50 else "", f"tpot {o.tpot_p50*1000:.1f}ms" if o.tpot_p50 else "",
                 f"{o.rps:.2f} req/s" if o.rps else "", f"fail {o.failure_rate:.0%}" if o.failure_rate else "",
                 f"mmlu {o.mmlu_ratio:.3f}" if o.mmlu_ratio else ""]
        events.append((o.step_i, f"  observed (config v{o.config_idx}): " + " ".join(p for p in parts if p)))
    for si, m in log.timer_marks:
        events.append((si, f"timer: minute {m:.0f}"))
    events.sort()
    last_min = None
    for si, txt in events:
        if txt.startswith("timer"):
            last_min = txt
            continue
        print(f"{si:5d}  {txt}")
        if show_text:
            s = steps.get(si)
            if s:
                print(f"       $ {unwrap_shell(s.cmd)[:160]!r}")
    if last_min:
        print(f"\nlast {last_min}")

    # verdicts
    print("\n## assertions")
    for k, v in evaluate(run, log).items():
        print(f"  {'PASS' if v else ('FAIL' if v is False else ' n/a'):5s} {k}")
    b = blame(run, log, full_only=True)
    print(f"\n## blame: found={b.found and round(b.found,2)}x shipped_obs={b.shipped_obs and round(b.shipped_obs,2)}x final={b.final:.2f}x  phase_of_loss={b.phase_of_loss}")
    for e in b.events:
        print(f"  step {e.step_i:5d}  {e.kind:18s} {e.note}")
    if log.eval_script_modified:
        print("\n!! evaluate.py was modified during this run")


if __name__ == "__main__":
    main(sys.argv[1], show_text="--text" in sys.argv)
