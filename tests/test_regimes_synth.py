"""regimes.synth: traffic generated fresh from a corpus's seen split. Keeps the seen shapes (sizes exactly per pass,
the timing pattern), repeats no prompt, never reads the real held-out split, writes nothing, and the seed regenerates
it. Run: python tests/test_regimes_synth.py"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.corpus import TraceRequest, WorkloadClass, build_manifest, load_trace, write_trace  # noqa: E402
from regimes import synth  # noqa: E402


def _real(d: Path) -> Path:
    """A two-class corpus: single requests with conversations, and one multi-turn session; then its held-out traces
    are deleted, so a generator that read them would fail."""
    words = "alpha beta gamma delta epsilon zeta. Eta theta iota kappa! Lambda mu nu xi? Omicron pi rho sigma.".split()
    reqs, t = [], 0.0
    for k in range(40):
        t += [1.0, 1.0, 0.0, 5.0][k % 4]
        user = " ".join(words[(k + j) % len(words)] for j in range(5 + k % 13))
        msgs = [{"role": "user", "content": "earlier " + user}, {"role": "assistant", "content": "answer " * (1 + k % 5)},
                {"role": "user", "content": user}]
        reqs.append(TraceRequest(arrival_s=t, session_id=f"s{k}", turn_index=0, prompt="\n\n".join(m["content"] for m in msgs),
                                 max_tokens=64 + 32 * (k % 3), messages=msgs, build_prompt_tokens=10 + 3 * len(user.split())))
    multi = [TraceRequest(arrival_s=2.0 + 30 * i, session_id="m", turn_index=i, prompt=f"turn {i} " * 20, max_tokens=128,
                          build_prompt_tokens=40) for i in range(3)]
    classes = {}
    for name, trace in (("steady", reqs), ("multi", multi)):
        for split in ("seen", "heldout"):
            write_trace(d / name / f"{split}.jsonl", trace)
        classes[name] = WorkloadClass(name, "test", 1000.0, 100.0, 1.0, f"{name}/seen.jsonl", f"{name}/heldout.jsonl")
    (d / "manifest.json").write_text(json.dumps(build_manifest(classes, d).to_dict()))
    for name in classes:
        (d / name / "heldout.jsonl").unlink()
    return d


def test_a_generated_trace_keeps_the_seen_shapes_and_repeats_no_prompt():
    with tempfile.TemporaryDirectory() as tmp:
        real = _real(Path(tmp) / "real")                         # its held-out traces are gone: never read
        _, src = load_trace("steady", "seen", real)
        one = synth.trace("steady", seed=0, corpus_dir=real, min_requests=0)
        assert len(one) == 40 and {r.prompt for r in one}.isdisjoint({r.prompt for r in src})
        assert sorted(r.max_tokens for r in one) == sorted(r.max_tokens for r in src)          # each session once
        assert all(len(a.messages) == 3 and a.messages[1]["role"] == "assistant" for a in one)
        cmp = synth.compare(real)
        assert cmp["steady"]["ks_prompt_tokens"] <= 0.15
        g_real, g_syn = cmp["steady"]["real"]["gap_s"], cmp["steady"]["synthetic"]["gap_s"]
        assert abs(g_syn["mean"] / g_real["mean"] - 1) < 0.15                                 # the same pattern
        multi = synth.trace("multi", seed=0, corpus_dir=real, min_requests=0)
        assert [r.turn_index for r in multi] == [0, 1, 2] and multi[2].arrival_s - multi[0].arrival_s == 60.0
        big = synth.trace("steady", seed=0, corpus_dir=real, min_requests=100)
        assert len(big) == 120                                                                # whole passes
        again = synth.trace("steady", seed=0, corpus_dir=real, min_requests=0)
        other = synth.trace("steady", seed=1, corpus_dir=real, min_requests=0)
        held = synth.trace("steady", seed=0, split="heldout", corpus_dir=real, min_requests=0)
        assert [r.prompt for r in again] == [r.prompt for r in one]                          # the seed regenerates it
        assert [r.prompt for r in other] != [r.prompt for r in one] != [r.prompt for r in held]


def test_regimes_draw_generated_pools_with_synthetic():
    from regimes import suite
    with tempfile.TemporaryDirectory() as tmp:
        real = _real(Path(tmp) / "real")
        ctx = suite.Ctx("http://x", "m", corpus=str(real), synthetic=True, seed=3)
        pool = ctx.pool("steady")
        assert len(pool) >= synth.MIN_REQUESTS and all(r["id"].startswith("syn-steady-seen-3-") for r in pool)
        assert len(ctx.pool(None)) > len(pool)                                                # every class
        real_prompts = {r.prompt for r in load_trace("steady", "seen", real)[1]}
        assert sum(r["prompt"] in real_prompts for r in pool) / len(pool) < 0.05      # a 25-word corpus repeats by chance
        assert not list(Path(tmp).rglob("syn*"))                                              # nothing written


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
