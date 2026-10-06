# BlameGraph

Referee for autoresearch loops that optimize LLM inference.

- **Integrity verdict:** did the agent earn its result
- **Session feedback:** facts for the agent, diagnostics for the researcher
- **Load regimes:** drives a server the eight ways it gets used and finds its goodput
- **Correctness gate:** optimized outputs still match the original model
- **Canaries:** planted cheats that must get caught
- **Validity checks:** held-out budget, cache audit, short-vs-full test agreement
- **Kernel checker:** rewritten kernels compute the same thing

Diagnostics are developed on the public [InferenceBench trajectories](https://huggingface.co/datasets/aisa-group/InferenceBench-Trajectories).
