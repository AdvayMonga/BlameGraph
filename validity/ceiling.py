"""Physical ceilings for an LLM engine, from the model's HF config and the GPU's measured peaks (validity/peaks.py).
No GPU, no weights. Everything here is a bound, not a target: a step cannot be faster than streaming the weights
and KV it needs, or than the tensor cores can multiply.

Per decode step the engine must read every weight a token touches (attention, router, lm_head, and the experts
the batch routes to), read each sequence's KV cache, and do 2 FLOP per active parameter per token. Prefill of P
tokens reads the weights once and does 2·P FLOP per active parameter plus the attention FLOPs.
step floor = max(bytes / bandwidth, FLOPs / peak). The gap from a floor to a measurement is overhead."""
from __future__ import annotations

from dataclasses import dataclass

from validity.peaks import Peaks

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
ALIASES = {"bfloat16": "bf16", "float16": "fp16", "half": "fp16", "float8": "fp8", "fp8_e4m3": "fp8", "fp8_e5m2": "fp8"}


def dtype(name: str) -> str:
    """A dtype name as DTYPE_BYTES knows it; ValueError for one it does not."""
    d = ALIASES.get(str(name).lower(), str(name).lower())
    if d not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype {name!r}; one of {sorted(DTYPE_BYTES)} or {sorted(ALIASES)}")
    return d


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_layers: int
    hidden: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    intermediate: int                 # dense MLP width (0 for a pure MoE)
    n_experts: int = 0
    top_k: int = 0
    moe_intermediate: int = 0
    shared_intermediate: int = 0      # always-on shared expert width (DeepSeek-style), 0 if none
    tie_embeddings: bool = False

    @classmethod
    def from_hf_config(cls, cfg: dict, name: str = "") -> "ModelSpec":
        """Llama-style dense and Qwen3/Mixtral/DeepSeek-style MoE configs."""
        heads = cfg["num_attention_heads"]
        n_exp = cfg.get("num_experts") or cfg.get("num_local_experts") or cfg.get("n_routed_experts") or 0
        return cls(name or cfg.get("_name_or_path", ""), cfg["num_hidden_layers"], cfg["hidden_size"], heads,
                   cfg.get("num_key_value_heads", heads), cfg.get("head_dim") or cfg["hidden_size"] // heads,
                   cfg["vocab_size"], 0 if n_exp else cfg["intermediate_size"], n_exp,
                   cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_token") or 0,
                   cfg.get("moe_intermediate_size") or (cfg["intermediate_size"] if n_exp else 0),
                   (cfg.get("moe_intermediate_size") or 0) * (cfg.get("n_shared_experts") or 0),
                   bool(cfg.get("tie_word_embeddings", False)))

    # -- parameter counts ----------------------------------------------------------------
    @property
    def attn_per_layer(self) -> int:
        return self.hidden * (self.n_q_heads + 2 * self.n_kv_heads) * self.head_dim + self.n_q_heads * self.head_dim * self.hidden

    @property
    def expert_params(self) -> int:
        """One expert: gate, up and down projections."""
        return 3 * self.hidden * self.moe_intermediate

    @property
    def dense_mlp_per_layer(self) -> int:
        return 3 * self.hidden * (self.intermediate + self.shared_intermediate)

    @property
    def router_per_layer(self) -> int:
        return self.hidden * self.n_experts

    @property
    def lm_head(self) -> int:
        return self.vocab * self.hidden

    @property
    def always_on(self) -> int:
        """Parameters every token reads regardless of routing (the embedding table is gathered, not streamed)."""
        return self.n_layers * (self.attn_per_layer + self.dense_mlp_per_layer + self.router_per_layer) + self.lm_head

    @property
    def active_params(self) -> int:
        return self.always_on + self.n_layers * self.top_k * self.expert_params

    @property
    def total_params(self) -> int:
        return self.always_on + self.n_layers * self.n_experts * self.expert_params + (0 if self.tie_embeddings else self.vocab * self.hidden)

    def kv_bytes_per_token(self, kv_dtype: str = "bf16") -> float:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * DTYPE_BYTES[kv_dtype]

    def experts_touched(self, tokens: int) -> float:
        """Expected distinct experts a layer reads for `tokens` routed tokens, uniform routing (the engine can do no
        better than reading each touched expert once)."""
        if not self.n_experts:
            return 0.0
        return self.n_experts * (1 - (1 - self.top_k / self.n_experts) ** tokens)

    def weight_bytes(self, tokens: int, w_dtype: str = "bf16") -> float:
        """Bytes of weights one forward over `tokens` tokens must stream."""
        return (self.always_on + self.n_layers * self.experts_touched(tokens) * self.expert_params) * DTYPE_BYTES[w_dtype]


# -- floors (seconds) ---------------------------------------------------------------------
def decode_step_floor(m: ModelSpec, hw: Peaks, batch: int, context: float, w_dtype: str = "bf16",
                      kv_dtype: str = "bf16") -> dict:
    """One decode step for `batch` sequences at mean `context` tokens."""
    w = m.weight_bytes(batch, w_dtype)
    kv = batch * context * m.kv_bytes_per_token(kv_dtype)
    flops = 2 * m.active_params * batch + 4 * m.n_layers * m.n_q_heads * m.head_dim * context * batch
    t_mem, t_compute = (w + kv) / hw.bw_bytes_s, flops / hw.bf16_flops
    return {"floor_s": max(t_mem, t_compute), "t_mem_s": t_mem, "t_compute_s": t_compute,
            "bound": "memory" if t_mem >= t_compute else "compute", "weight_bytes": w, "kv_bytes": kv, "flops": flops}


def prefill_floor(m: ModelSpec, hw: Peaks, prompt_tokens: int, w_dtype: str = "bf16", kv_dtype: str = "bf16") -> dict:
    """Prefill of one prompt: weights once, KV written once, quadratic attention."""
    w = m.weight_bytes(prompt_tokens, w_dtype)
    kv = prompt_tokens * m.kv_bytes_per_token(kv_dtype)
    flops = 2 * m.active_params * prompt_tokens + 2 * m.n_layers * m.n_q_heads * m.head_dim * prompt_tokens * (prompt_tokens + 1)
    t_mem, t_compute = (w + kv) / hw.bw_bytes_s, flops / hw.bf16_flops
    return {"floor_s": max(t_mem, t_compute), "t_mem_s": t_mem, "t_compute_s": t_compute,
            "bound": "memory" if t_mem >= t_compute else "compute", "weight_bytes": w, "kv_bytes": kv, "flops": flops}


def capacity_batch(m: ModelSpec, hw: Peaks, context: float, w_dtype: str = "bf16", kv_dtype: str = "bf16") -> int:
    """Sequences of `context` tokens whose KV fits in HBM beside all the weights (no activations or fragmentation:
    an upper bound)."""
    free = hw.mem_bytes - m.total_params * DTYPE_BYTES[w_dtype]
    return max(0, int(free // (context * m.kv_bytes_per_token(kv_dtype)))) if context > 0 else 1 << 16


def max_batch_under(m: ModelSpec, hw: Peaks, tpot_limit_s: float, context: float, w_dtype: str = "bf16",
                    kv_dtype: str = "bf16", cap: int = 1 << 16) -> int:
    """Largest batch whose decode-step floor still meets the TPOT limit and whose KV fits in HBM."""
    lo, hi = 0, min(cap, capacity_batch(m, hw, context, w_dtype, kv_dtype))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if decode_step_floor(m, hw, mid, context, w_dtype, kv_dtype)["floor_s"] <= tpot_limit_s:
            lo = mid
        else:
            hi = mid - 1
    return lo


def tokens_per_s_ceiling(m: ModelSpec, hw: Peaks, tpot_limit_s: float, context: float, w_dtype: str = "bf16",
                         kv_dtype: str = "bf16") -> float:
    """Output tokens/s at the largest batch the TPOT limit allows; 0 when even one sequence cannot meet it."""
    b = max_batch_under(m, hw, tpot_limit_s, context, w_dtype, kv_dtype)
    return b / decode_step_floor(m, hw, b, context, w_dtype, kv_dtype)["floor_s"] if b else 0.0


# -- regime ceilings ------------------------------------------------------------------------
@dataclass(frozen=True)
class Shape:
    """What a regime's requests looked like: the ceiling depends on them."""
    prompt_tokens_mean: float
    prompt_tokens_p99: float
    output_tokens_mean: float
    concurrency: float = 1.0          # mean sequences in flight
    uncached_prompt_tokens_mean: float | None = None   # what the server reported it had to prefill; None = unknown


def regime_ceiling(objective: str, m: ModelSpec, hw: Peaks, shape: Shape, limits: dict,
                   w_dtype: str = "bf16", kv_dtype: str = "bf16", usd_per_hour: float | None = None) -> dict | None:
    """The physical bound on a regime's objective, or None where none exists (speedup ratios, cold start, a $
    objective without a price)."""
    ctx = shape.prompt_tokens_mean + shape.output_tokens_mean / 2
    if objective == "tpot_p99_s":
        f = decode_step_floor(m, hw, max(1, round(shape.concurrency)), ctx, w_dtype, kv_dtype)
        return {"ceiling": f["floor_s"], "better": "lower", "bound": f["bound"],
                "basis": f"decode step floor at batch {max(1, round(shape.concurrency))}, context {ctx:.0f}; one "
                         f"token per sequence per step, uniform expert routing, {w_dtype} weights, {kv_dtype} KV"}
    if objective == "ttft_p99_s":
        f = prefill_floor(m, hw, int(shape.prompt_tokens_p99), w_dtype, kv_dtype)
        return {"ceiling": f["floor_s"], "better": "lower", "bound": f["bound"],
                "basis": f"prefill floor at the p99 prompt, {shape.prompt_tokens_p99:.0f} tokens, {w_dtype} weights"}
    if objective == "usd_per_m_output_tokens":
        g = regime_ceiling("goodput_req_per_s", m, hw, shape, limits, w_dtype, kv_dtype)
        tps = g["ceiling"] * shape.output_tokens_mean
        if usd_per_hour is None or not tps:
            return None
        return {"ceiling": usd_per_hour / 3600 / tps * 1e6, "better": "lower", "bound": g["bound"],
                "basis": f"${usd_per_hour:g}/GPU-hour over the goodput ceiling's {tps:.0f} output tokens/s: " + g["basis"]}
    if objective in ("goodput_req_per_s", "good_req_per_s"):
        # each request costs its prefill FLOPs at peak (weights amortised over the batch) plus its output tokens at
        # the decode ceiling; one GPU-second holds at most 1 / (that) requests
        # a prefix cache can skip any prompt token, so only what the server reports as uncached counts; when it
        # reports nothing, prefill is left out and the bound is decode alone (still a true bound, a looser one)
        tps = tokens_per_s_ceiling(m, hw, limits["tpot_s"], ctx, w_dtype, kv_dtype)
        uncached = shape.uncached_prompt_tokens_mean or 0.0
        prefill_s = prefill_floor(m, hw, int(uncached), w_dtype, kv_dtype)["t_compute_s"] if uncached >= 1 else 0.0
        per_req = prefill_s + (shape.output_tokens_mean / tps if tps else float("inf"))
        known = shape.uncached_prompt_tokens_mean is not None
        return {"ceiling": 1 / per_req if per_req and per_req != float("inf") else 0.0, "better": "higher",
                "bound": "decode" if shape.output_tokens_mean / max(tps, 1e-12) > prefill_s else "prefill",
                "basis": (f"1 / (prefill compute of the {uncached:.0f} uncached prompt tokens per request the server "
                          f"reported" if known else "1 / (no prefill: the server reported no prefix-cache counts, so "
                          "every prompt token may have been cached") + " + "
                         f"{shape.output_tokens_mean:.0f} output tokens at the decode ceiling: the largest batch "
                         f"meeting TPOT <= {limits['tpot_s']} s whose KV fits in HBM, context {ctx:.0f}); "
                         f"one token per sequence per step, uniform expert routing, {w_dtype} weights, {kv_dtype} KV"}
    return None


def pct_of(value: float | None, ceiling: float | None, better: str) -> float | None:
    """How close a measurement is to its bound, 100 = at the bound; None without both numbers."""
    if value is None or ceiling is None or not value or not ceiling:
        return None
    return (ceiling / value if better == "lower" else value / ceiling) * 100


def ceiling_gain(share_of_total: float, pct_of_ceiling: float | None) -> float | None:
    """Fraction of the total that disappears if this part reached its ceiling (Amdahl); None without a ceiling."""
    if pct_of_ceiling is None or pct_of_ceiling <= 0:
        return None
    return share_of_total * (1 - min(pct_of_ceiling, 100.0) / 100)
