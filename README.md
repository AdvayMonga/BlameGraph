# BlameGraph

Referee for autoresearch loops that optimize LLM inference.

- **Integrity verdict:** did the agent earn its result
- **Session feedback:** facts for the agent, diagnostics for the researcher
- **Correctness gate:** optimized outputs still match the original model
- **Canaries:** planted cheats that must get caught
- **Validity checks:** held-out budget, cache audit, short-vs-full test agreement
- **Kernel checker:** rewritten kernels compute the same thing
