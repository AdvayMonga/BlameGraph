"""Where a measurement's time goes and how far each part is from its ceiling. Arithmetic on the profile only."""
from __future__ import annotations

from lab.infra.context.ceiling import ceiling_gain
from lab.loop.types import Gap, Measurement

LAYERS = ("kernels", "model_execution", "runtime", "serving")


def gaps(m: Measurement) -> tuple[Gap, ...]:
    """Each profiled layer with a share of the window, the most recoverable time first; empty without a profile."""
    layers = m.profile.get("layers") or {}
    out = []
    for name in LAYERS:
        share = (layers.get(name) or {}).get("share")
        if share is None:
            continue
        pct = layers[name].get("pct_of_ceiling")
        gain = ceiling_gain(share, pct)
        out.append(Gap(name, share * 100, pct, (share if gain is None else gain) * 100))
    return tuple(sorted(out, key=lambda g: -g.recoverable_pct))
