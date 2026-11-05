"""The gap register's facts from a compare/profile.py dir: per load point, every engine's client numbers, GPU busy share of
the nsys window, kernel time, launches, the top kernels by time, and the top Python frames by samples per process.

    python -m compare.gap DIR [--top 12] [--out gap.md]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def kernels(db: Path, top: int) -> dict:
    """GPU facts of one nsys window: busy = union of kernel intervals over the window's span."""
    con = sqlite3.connect(db)
    q = lambda sql: con.execute(sql).fetchall()
    if not q("select name from sqlite_master where name='CUPTI_ACTIVITY_KIND_KERNEL'"):
        return {"error": "no kernels in the window"}
    every = q("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName "
              "order by k.start")
    rows = [r for r in every if 0 <= r[0] <= r[1] and r[1] - r[0] < 1e9]   # nsys gives some graph nodes a bogus start
    span = (max(e for _, e, _ in rows) - rows[0][0]) / 1e9
    busy, cur_s, cur_e = 0, None, None
    for s, e, _ in rows:
        if cur_e is None or s > cur_e:
            busy += (cur_e - cur_s) if cur_e is not None else 0
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    by = Counter()
    calls = Counter()
    for s, e, name in rows:
        by[name] += e - s
        calls[name] += 1
    total = sum(by.values())
    launches = dict(q("select s.value, count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId "
                      "where s.value like '%LaunchKernel%' or s.value like '%GraphLaunch%' or s.value like 'cuLaunch%' "
                      "group by s.value"))
    copy = q("select coalesce(sum(end - start), 0) from CUPTI_ACTIVITY_KIND_MEMCPY")[0][0] \
        if q("select name from sqlite_master where name='CUPTI_ACTIVITY_KIND_MEMCPY'") else 0
    return {"span_s": span, "gpu_busy": busy / 1e9 / span, "kernel_s": total / 1e9, "kernels": len(rows),
            "dropped": len(every) - len(rows),
            "memcpy_s": copy / 1e9, "launch_calls": launches,
            "top": [(n, by[n] / total, calls[n], by[n] / calls[n] / 1e3) for n, _ in by.most_common(top)]}


def frames(path: Path, top: int) -> list:
    """py-spy speedscope: the frames with the most self samples, per profile (process or thread), with their share."""
    d = json.loads(path.read_text())
    names = [f"{f['name']} ({Path(f.get('file') or '?').name}:{f.get('line')})" for f in d["shared"]["frames"]]
    out = []
    for p in d["profiles"]:
        c = Counter()
        for stack, w in zip(p["samples"], p["weights"]):
            if stack:
                c[names[stack[-1]]] += w
        n = sum(c.values())
        if n:
            out.append((p["name"], n, [(f, v / n) for f, v in c.most_common(top)]))
    return sorted(out, key=lambda x: -x[1])


def render(d: Path, top: int = 12) -> str:
    lines = [f"# Gap facts ({d.name})", ""]
    cfgs = sorted(p for p in d.iterdir() if p.is_dir())
    points = sorted({f.stem for c in cfgs for f in c.glob("*.json") if not f.name.endswith(".pyspy.json")})
    for point in points:
        lines += [f"## {point}", "", "| engine | out tok/s | TPOT p50 ms | TTFT p99 s | ok/n | GPU busy | kernels in window | "
                  "kernel launches | memcpy s |", "|---|---|---|---|---|---|---|---|---|"]
        detail = []
        for c in cfgs:
            f = c / f"{point}.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())["summary"]
            db = c / f"{point}.sqlite"
            k = kernels(db, top) if db.exists() else {"error": "no nsys export"}
            lc = sum((k.get("launch_calls") or {}).values())
            lines.append(f"| {c.name} | {s['output_tok_per_s']:.0f} | {(s['tpot_p50_s'] or 0) * 1e3:.1f} | "
                         f"{(s['ttft_p99_s'] or 0):.2f} | {s['ok']}/{s['n']} | "
                         + (f"{k['gpu_busy']:.0%} | {k['kernels']}" + (f" ({k['dropped']} bad dropped)" if k["dropped"] else "")
                            + f" | {lc} | {k['memcpy_s']:.2f} |" if "error" not in k
                            else f"{k['error']} | | | |"))
            if "top" in k:
                detail += [f"**{c.name}** top kernels (share of kernel time, calls, mean us):", ""]
                detail += [f"- {share:.1%} {n[:90]} ({calls}, {us:.1f})" for n, share, calls, us in k["top"]]
                detail.append("")
            py = c / f"{point}.pyspy.json"
            if py.exists():
                for name, n, fr in frames(py, 6)[:3]:
                    detail += [f"**{c.name}** py-spy {name} ({n:.1f} s sampled): "
                               + "; ".join(f"{share:.0%} {f}" for f, share in fr), ""]
        lines += [""] + detail
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    text = render(a.dir, a.top)
    print(text)
    if a.out:
        a.out.write_text(text)
