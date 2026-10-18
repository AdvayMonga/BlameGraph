"""python -m sim run CLASS [--synthetic] [--seed N] [--env JSON] [--cost FILE] [--horizon S] [--out FILE]
Simulate one corpus class's traffic (the target's corpus; `--synthetic` generates it fresh, regimes/synth.py)
through the engine's scheduler under `--env` (the engine's scheduler env vars, e.g. a profile's) and `--cost`
(a CostModel JSON; defaults are unfitted stand-ins). Prints the regimes summary under the conversational limits.
Run with the engine's python."""
from __future__ import annotations

import argparse
import json
import sys

from lab import target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim", description=__doc__.split("\n")[1])
    target.add_argument(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("cls", help="a corpus class (steady_interactive, spike, long_context, mixed_1, ...)")
    r.add_argument("--synthetic", action="store_true", help="traffic generated fresh from the class's seen split")
    r.add_argument("--split", choices=("seen", "heldout"), default="seen")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--env", default="{}", help="scheduler env vars as JSON, e.g. '{\"MAX_BATCH_SIZE\": \"32\"}'")
    r.add_argument("--cost", help="CostModel JSON (fitted coefficients)")
    r.add_argument("--horizon", type=float, help="stop after this many simulated seconds")
    r.add_argument("--out", help="write rows and the summary here")
    a = ap.parse_args(argv)
    from regimes import workload
    from regimes.runner import CONVERSATIONAL, summarize
    from sim.scheduler import CostModel, Settings, simulate
    t = target.load(a.target) if getattr(a, "target", None) else target.load()
    corpus = str(t.corpus_dir)
    if a.synthetic:
        from regimes import synth
        records = synth.records(corpus, a.cls, a.seed, a.split)
    else:
        records = workload.corpus(corpus, a.cls, a.split)
    res = simulate(records, Settings.from_env(json.loads(a.env)), CostModel.load(a.cost) if a.cost else None,
                   seed=a.seed, horizon_s=a.horizon)
    summary = summarize(res.rows, CONVERSATIONAL, res.elapsed_s)
    report = {"class": a.cls, "synthetic": a.synthetic, "split": a.split, "seed": a.seed, "requests": len(records),
              "simulated_s": res.elapsed_s, "steps": res.steps, "decode_steps": res.decode_steps,
              "prefill_waves": res.prefill_waves, "cost_source": res.cost.source, "summary": summary}
    if a.out:
        with open(a.out, "w") as f:
            json.dump({**report, "rows": [x.to_dict() for x in res.rows]}, f, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
