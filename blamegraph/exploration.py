"""How an agent explored the config space, measured against the vLLM search space InferenceBench's search baselines use."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .experiment import ConfigVersion, ExperimentLog

# src/baselines/search_spaces/vllm.yaml in aisa-group/InferenceBench (11 parameters)
SEARCH_SPACE = {
    "max_num_seqs": r"--max-num-seqs",
    "max_num_batched_tokens": r"--max-num-batched-tokens",
    "gpu_memory_utilization": r"--gpu-memory-utilization",
    "block_size": r"--block-size",
    "enable_chunked_prefill": r"--(no-)?enable-chunked-prefill",
    "enable_prefix_caching": r"--(no-)?enable-prefix-caching",
    "enforce_eager": r"--enforce-eager",
    "quantization": r"--quantization",
    "kv_cache_dtype": r"--kv-cache-dtype",
    "attention_backend": r"VLLM_ATTENTION_BACKEND|--attention-backend",
    "num_speculative_tokens": r"--speculative-config|--num-speculative-tokens",
}
# other knobs agents commonly touch that are outside the baseline's space
EXTRA_KNOBS = {
    "max_model_len": r"--max-model-len", "dtype": r"--dtype", "performance_mode": r"--performance-mode|PERFORMANCE_MODE",
    "tokenizer_mode": r"--tokenizer-mode|TOKENIZER_MODE", "cuda_graph": r"--cuda-graph|--compilation-config|cudagraph",
    "async_scheduling": r"--async-scheduling", "scheduler": r"--scheduling-policy|--scheduler",
    "env_tuning": r"export (VLLM_|CUDA_|NCCL_|TORCH)",
}


DEFAULT_RE = re.compile(r"\$\{\w+:-([^}]*)\}")


def resolve(value: str) -> str:
    """`${VAR:-8192}` -> `8192`; a value that still references a variable is unresolved."""
    return DEFAULT_RE.sub(r"\1", value)


def is_parameterized(values: dict[str, str]) -> bool:
    return any("$" in v for v in values.values())


def knob_values(cfg: ConfigVersion) -> dict[str, str]:
    """Value (or presence) of each known knob in a config; env-var knobs read `export X=val`.
    `${VAR:-default}` resolves to the default; other `$VAR` references stay as-is (see is_parameterized)."""
    body = re.sub(r"\\\n", " ", cfg.content)
    flags = cfg.flags
    out: dict[str, str] = {}
    for name, pat in {**SEARCH_SPACE, **EXTRA_KNOBS}.items():
        m = re.search(pat, body)
        if not m:
            continue
        tok = m.group(0)
        if tok.startswith("--"):
            out[name] = resolve(flags.get(tok, ""))
        else:
            env = re.search(re.escape(tok.split("|")[0]) + r"\w*\s*=\s*['\"]?([^'\"\s;]+)", body)
            out[name] = resolve(env.group(1)) if env else "set"
    return out


@dataclass
class Exploration:
    measured_configs: list[ConfigVersion]
    knobs_varied: set[str]           # knobs whose value differs across measured configs
    space_knobs_varied: set[str]     # subset inside the baseline search space
    knobs_touched: set[str]          # knobs set in any measured config
    n_transitions: int               # consecutive measured-config pairs
    n_single_factor: int             # transitions that changed exactly one knob
    n_multi_factor: int              # transitions that changed 2+ knobs
    n_unknown_change: int            # transitions where no known knob changed (edit elsewhere)

    @property
    def ofat_rate(self) -> float | None:
        return self.n_single_factor / self.n_transitions if self.n_transitions else None


def analyze(log: ExperimentLog) -> Exploration:
    h = {c.idx: c for c in log.configs}
    seq: list[ConfigVersion] = []
    for e in log.evals:
        if e.config_idx in h and (not seq or seq[-1].hash != h[e.config_idx].hash):
            seq.append(h[e.config_idx])
    vals = [knob_values(c) for c in seq]
    touched = set().union(*vals) if vals else set()
    varied = {k for k in touched if len({v.get(k) for v in vals}) > 1}
    single = multi = unknown = 0
    for a, b in zip(vals, vals[1:]):
        changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        if len(changed) == 1:
            single += 1
        elif len(changed) >= 2:
            multi += 1
        else:
            unknown += 1
    return Exploration(
        measured_configs=seq, knobs_varied=varied, space_knobs_varied=varied & set(SEARCH_SPACE),
        knobs_touched=touched, n_transitions=max(0, len(seq) - 1), n_single_factor=single,
        n_multi_factor=multi, n_unknown_change=unknown,
    )
