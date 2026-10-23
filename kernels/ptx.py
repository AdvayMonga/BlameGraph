"""The PTX comparison (decided 2026-10-22, after Standard Kernel's method): several implementations of one operation
lowered to PTX and condensed into comparable feature cards, so the judge can see what the fastest does that ours
does not. The cards are counted from the PTX itself (instruction classes, memory spaces, vector widths, shared memory,
the Hopper features in use), not written by a model: facts, not a reading of them. A card may carry its measured time
on our shapes (`timing_ms`, from a GPU run) beside it.

  python -m kernels.ptx card --name NAME PTX [--name NAME PTX ...] [--timing NAME=MS ...] --out CARDS.txt
  (a file of `cuobjdump -sass` output is read as SASS: libraries that ship compiled code without PTX)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Instruction classes by opcode prefix (the first match wins), the features that tell kernels apart on Hopper.
CLASSES = (
    ("wgmma", "wgmma."),                      # warpgroup tensor-core MMA (Hopper)
    ("mma", "mma."),                          # warp-level tensor-core MMA
    ("tma", "cp.async.bulk.tensor"),          # TMA bulk tensor copies (Hopper)
    ("bulk_copy", "cp.async.bulk"),
    ("cp_async", "cp.async"),                 # async global -> shared copies (Ampere+)
    ("ldmatrix", "ldmatrix"),
    ("mbarrier", "mbarrier."),
    ("setmaxnreg", "setmaxnreg"),             # register reallocation between warpgroups: warp specialization
    ("barrier", "bar."),
    ("shuffle", "shfl."),
    ("atomic", "atom."),
    ("red", "red."),
    ("ld_global", "ld.global"),
    ("st_global", "st.global"),
    ("ld_shared", "ld.shared"),
    ("st_shared", "st.shared"),
    ("fma", "fma."),
    ("convert", "cvt."),
    ("branch", "bra"),
)
_ENTRY = re.compile(r"\.(?:visible\s+)?\.entry\s+(\S+?)\s*\(")
_OP = re.compile(r"^\s*(?:@!?%\w+\s+)?([a-z][a-z0-9_.]*)", re.M)
_SHARED = re.compile(r"\.shared\s+(?:\.align\s+\d+\s+)?\.\w+\s+\S+?\[(\d*)\]")
_REGS = re.compile(r"\.reg\s+\.(\w+)\s+%\w+<(\d+)>")
_VEC = re.compile(r"\.v(2|4|8)\.")


def entries(ptx: str) -> dict[str, str]:
    """{kernel name: its body} for every .entry in the PTX."""
    starts = [(m.group(1), m.start()) for m in _ENTRY.finditer(ptx)]
    return {name: ptx[s:(starts[i + 1][1] if i + 1 < len(starts) else len(ptx))] for i, (name, s) in enumerate(starts)}


def features(body: str) -> dict:
    """One kernel's card: instruction counts by class, the widest vector access, declared registers and static shared
    memory, and which Hopper features it uses."""
    ops = Counter(_OP.findall(body))
    counts: Counter = Counter()
    for op, n in ops.items():
        for name, prefix in CLASSES:
            if op.startswith(prefix):
                counts[name] += n
                break
    regs = sum(int(n) for _, n in _REGS.findall(body))
    shared = sum(int(n) for n in _SHARED.findall(body) if n)
    vec = max((int(v) for v in _VEC.findall(body)), default=1)
    return {"instructions": sum(ops.values()), **{k: counts[k] for k, _ in CLASSES if counts[k]},
            "widest_vector": vec, "declared_registers": regs, "static_shared_bytes": shared,
            "uses": sorted(k for k in ("wgmma", "tma", "setmaxnreg", "mbarrier", "cp_async", "mma", "ldmatrix")
                           if counts[k])}


# The same classes in SASS (`cuobjdump -sass`), for libraries that ship compiled code without PTX.
SASS = (("wgmma", "HGMMA"), ("mma", "HMMA"), ("tma", "UTMALDG"), ("tma", "UTMASTG"), ("bulk_copy", "UBLKCP"),
        ("cp_async", "LDGSTS"), ("ldmatrix", "LDSM"), ("mbarrier", "SYNCS"), ("setmaxnreg", "USETMAXREG"),
        ("barrier", "BAR"), ("shuffle", "SHFL"), ("atomic", "ATOM"), ("red", "RED"), ("ld_global", "LDG"),
        ("st_global", "STG"), ("ld_shared", "LDS"), ("st_shared", "STS"), ("fma", "FFMA"), ("convert", "F2F"),
        ("branch", "BRA"))
_FUNCTION = re.compile(r"Function : (\S+)")
_SASS_OP = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)")


def sass_entries(sass: str) -> dict[str, str]:
    starts = [(m.group(1), m.start()) for m in _FUNCTION.finditer(sass)]
    return {name: sass[s:(starts[i + 1][1] if i + 1 < len(starts) else len(sass))] for i, (name, s) in enumerate(starts)}


def sass_features(body: str) -> dict:
    """A kernel's card from its SASS: instruction counts by the same classes and the Hopper features in use."""
    ops = Counter(_SASS_OP.findall(body))
    counts: Counter = Counter()
    for op, n in ops.items():
        for name, prefix in SASS:
            if op.startswith(prefix):
                counts[name] += n
                break
    return {"instructions": sum(ops.values()), **{k: counts[k] for k, _ in CLASSES if counts[k]},
            "uses": sorted(k for k in ("wgmma", "tma", "setmaxnreg", "mbarrier", "cp_async", "mma", "ldmatrix")
                           if counts[k]), "from": "sass"}


def card(name: str, ptx: str, timing_ms: float | None = None) -> dict:
    """An implementation's card from its PTX, or from SASS when the text is `cuobjdump -sass` output."""
    ks = ({k: sass_features(b) for k, b in sass_entries(ptx).items()} if "Function : " in ptx
          else {k: features(b) for k, b in entries(ptx).items()})
    return {"implementation": name, **({"timing_ms": timing_ms} if timing_ms is not None else {}),
            "kernels": len(ks), "per_kernel": ks}


def render(cards: list[dict]) -> str:
    """The cards as the judge's slice: one line per kernel per implementation, fastest implementation first."""
    lines = []
    for c in sorted(cards, key=lambda c: c.get("timing_ms", float("inf"))):
        head = f"{c['implementation']}" + (f" ({c['timing_ms']} ms on our shapes)" if "timing_ms" in c else "")
        lines.append(f"{head}: {c['kernels']} kernels")
        lines += [f"  {k}: {json.dumps(f)}" for k, f in c["per_kernel"].items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernels.ptx")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("card")
    c.add_argument("--name", action="append", required=True)
    c.add_argument("ptx", nargs="+")
    c.add_argument("--timing", action="append", default=[], help="NAME=MS, a measured time on our shapes")
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.name) != len(a.ptx):
        ap.error("one --name per PTX file")
    times = {k: float(v) for k, _, v in (t.partition("=") for t in a.timing)}
    cards = [card(n, Path(p).read_text(), times.get(n)) for n, p in zip(a.name, a.ptx)]
    Path(a.out).write_text(render(cards) + "\n")
    print(json.dumps({"cards": len(cards), "out": a.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
