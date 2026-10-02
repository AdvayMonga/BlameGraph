# BlameGraph

Feedback for autoresearch loops that optimize inference systems: did the agent earn its result, and what actually happened in the session?

- **It reads a finished session and reports on it.** Input is a session log (or an InferenceBench-format trace); output is an integrity verdict, a set of facts, and researcher diagnostics.
- **Integrity is a verdict with reasons.** The grader was untouched, the submission is something that was actually measured on a fresh setup, and the numbers are physically possible. Anything else voids the result.
- **Facts go to the agent; judgments don't.** The agent-facing half says what was measured, what was shipped, what was best, and which measurements were stale, non-standard, or never read. It never says what to do next.
- **Diagnostics stay with the researcher.** Self-consistency and methodology assertions and a blame breakdown (found, kept, executed) show how the session went without being fed back as advice.
- **Assertions are tested by breaking things.** Known failures are injected into real traces and each assertion must flip when its failure is present.
- **Equivalence checks are next.** For kernels and servers, "faster" only counts if the output is the same: randomized and held-out inputs against a reference, side-effect and timing checks, token-level agreement for model servers.
- **Usage:** `python -m blamegraph feedback <session.jsonl | run_dir>` (add `--agent-text` for the agent-facing half only). Tests: `python tests/test_feedback.py`, `python tests/test_flip.py`.
- The full research snapshot on public InferenceBench traces (analyses, judge, dashboard) is preserved at the `research-v1` tag.
