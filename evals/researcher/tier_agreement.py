"""Short-tier validity: the cheap tier counts only if it ranks known changes the way the full tier does."""
from __future__ import annotations

import math
import random


def _ranks(xs: list[float]) -> list[float]:
    """1-based ranks, ties get the average rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    """Pearson correlation of average ranks; nan if either side is constant."""
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb) if va and vb else float("nan")


def kendall_tau_b(a: list[float], b: list[float]) -> float:
    """Kendall tau-b; nan if either side is constant."""
    c = d = ta = tb = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            s = (a[i] > a[j]) - (a[i] < a[j]); t = (b[i] > b[j]) - (b[i] < b[j])
            ta += s == 0; tb += t == 0
            c += s * t > 0; d += s * t < 0
    n0 = len(a) * (len(a) - 1) // 2
    den = math.sqrt((n0 - ta) * (n0 - tb))
    return (c - d) / den if den else float("nan")


def _quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs); p = q * (len(xs) - 1); lo = int(p)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (p - lo)


def tier_agreement(changes: list[dict], min_n: int = 8, min_rho: float = 0.8, min_ci_low: float = 0.5,
                   n_boot: int = 2000, seed: int = 0) -> dict:
    """`changes`: [{"name", "short", "full", optional "short_band"/"full_band"}] → rank agreement + verdict."""
    n = len(changes)
    short = [float(c["short"]) for c in changes]; full = [float(c["full"]) for c in changes]
    rho = spearman(short, full) if n >= 2 else float("nan")
    tau = kendall_tau_b(short, full) if n >= 2 else float("nan")

    boots = []
    if n >= 2:
        rng = random.Random(seed)
        for _ in range(n_boot):
            idx = [rng.randrange(n) for _ in range(n)]
            r = spearman([short[i] for i in idx], [full[i] for i in idx])
            if not math.isnan(r):
                boots.append(r)
    ci = (_quantile(boots, 0.025), _quantile(boots, 0.975)) if boots else (float("nan"), float("nan"))

    disagreements = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = changes[i], changes[j]
            df, ds = full[i] - full[j], short[i] - short[j]
            full_band = a.get("full_band", 0.0) + b.get("full_band", 0.0)
            if abs(df) > full_band and (ds > 0) - (ds < 0) != (df > 0) - (df < 0):
                disagreements.append({
                    "a": a["name"], "b": b["name"], "full_diff": df, "full_band": full_band, "short_diff": ds,
                    "short_band": a.get("short_band", 0.0) + b.get("short_band", 0.0)})

    reasons = []
    if n < min_n:
        reasons.append(f"n={n} < {min_n} changes")
    if math.isnan(rho):
        reasons.append("rho undefined (fewer than 2 changes or a tier has no variation)")
    elif rho < min_rho:
        reasons.append(f"rho={rho:.3f} < {min_rho}")
    if not math.isnan(rho) and not (ci[0] >= min_ci_low):
        reasons.append(f"rho 95% CI lower bound {ci[0]:.3f} < {min_ci_low}")
    return {"n": n, "spearman_rho": rho, "kendall_tau_b": tau, "rho_ci95": ci, "n_boot_valid": len(boots),
            "disagreements": disagreements, "verdict": {"usable": not reasons, "reasons": reasons}}
