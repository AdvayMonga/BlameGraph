"""Session (ledger) -> ExperimentLog, so assertions, blame and noise analyses run on any loop's sessions."""
from __future__ import annotations

from .experiment import ConfigVersion, EvalLaunch, ExperimentLog, Observation
from .session import Ledger


def log_from_ledger(led: Ledger, run_id: str = "ledger") -> ExperimentLog:
    log = ExperimentLog(run_id=run_id)
    idx_by_hash: dict[str, int] = {}
    step = 0
    for e in led.events:
        step += 1
        if e["kind"] == "config":
            idx = len(log.configs)
            idx_by_hash[e["hash"]] = idx
            log.configs.append(ConfigVersion(idx=idx, step_i=step, content=e.get("content", ""), minute=e.get("minute")))
        elif e["kind"] == "launch":
            log.server_starts.append((step, idx_by_hash.get(e.get("config_hash"))))
        elif e["kind"] == "measure":
            cfg = idx_by_hash.get(e.get("live_hash"))
            ev = EvalLaunch(step_i=step, quick=e["mode"] == "quick", config_idx=cfg, config_stale=bool(e.get("stale")),
                            minute=e.get("minute"), error=bool(e.get("error")), standard=bool(e.get("standard")),
                            warm=e.get("cache_state") == "warm", observed=e.get("metrics") is not None)
            log.evals.append(ev)
            m = e.get("metrics") or {}
            profiles = m.get("profiles") or {}
            b = profiles.get("burst") or {}
            if m:
                o = Observation(step_i=step, config_idx=cfg, ttft_p50=(b.get("ttft") or {}).get("p50"), tpot_p50=(b.get("tpot") or {}).get("p50"),
                                rps=b.get("request_throughput_req_per_s"), gen_tps=b.get("generation_throughput_tokens_per_s"),
                                failure_rate=b.get("failure_rate"), quality_pass=(m.get("quality_check") or {}).get("pass"),
                                quick=ev.quick, standard=ev.standard, warm=ev.warm)
                rps = [profiles.get(n, {}).get("request_throughput_req_per_s") for n in ("burst", "poisson", "constant")]
                if all(rps):
                    o.rps_geomean = (rps[0] * rps[1] * rps[2]) ** (1 / 3)
                log.observations.append(o)
        if e.get("minute") is not None:
            log.timer_marks.append((step, e["minute"]))
    start, end = led.grader_hashes()
    log.eval_script_modified = bool(start and end and any(start.get(k) != v for k, v in end.items()))
    return log
