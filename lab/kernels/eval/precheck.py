"""The precheck: a program's import with no device (building its CUDA C++) and each built kernel's
resource use, run jailed with no GPU.

  python -m lab.kernels.eval.precheck CONFIG PROGRAM     (prints one JSON line)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

BUILD_FAILED = ("Error building extension", "ninja: build stopped", "nvcc fatal")
_COMPILE_ERROR = re.compile(r"\.(cu|cpp|cuh|h)\(?\d+\)?[:)].{0,20}error", re.I)     # nvcc / gcc diagnostics
_FUNC = re.compile(r"Function (\S+):")
_RES = re.compile(r"(REG|STACK|SHARED|LOCAL):(\d+)")


def resources(text: str) -> list[dict]:
    """`cuobjdump --dump-resource-usage` -> [{kernel, registers, stack, shared, local}]."""
    out, name = [], None
    for line in text.splitlines():
        m = _FUNC.search(line)
        if m:
            name = m.group(1)
            continue
        got = dict(_RES.findall(line))
        if name and "REG" in got:
            out.append({"kernel": name, "registers": int(got["REG"]), "stack": int(got.get("STACK", 0)),
                        "shared": int(got.get("SHARED", 0)), "local": int(got.get("LOCAL", 0))})
            name = None
    return out


def report(build_dir: Path) -> list[dict]:
    """The resource use of every kernel in the extensions built under `build_dir`."""
    exe = shutil.which("cuobjdump") or (Path(os.environ.get("CUDA_HOME", "")) / "bin" / "cuobjdump")
    if not Path(exe).exists():
        return []
    out = []
    for so in sorted(Path(build_dir).rglob("*.so")):
        p = subprocess.run([str(exe), "--dump-resource-usage", str(so)], capture_output=True, text=True, timeout=60)
        out += resources(p.stdout)
    return out


def precheck(program: Path, build_dir: Path) -> dict:
    """{passed: True|False|None, reason, compile_report}: None when the import failed but not on a build."""
    from lab.kernels.eval.core import load_program
    try:
        load_program(program)
    except Exception as e:                       # noqa: BLE001
        text = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
        built = any(s in text for s in BUILD_FAILED) or bool(_COMPILE_ERROR.search(text))
        return {"passed": False if built else None,
                "reason": ("the CUDA C++ build failed: " if built else "import failed without a device: ") + text[-1500:]}
    kernels = report(build_dir)
    return {"passed": True, "reason": f"imported; {len(kernels)} built kernels", "compile_report": kernels}


def main() -> int:
    cfg, program = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
    key = None
    if cfg.get("result_key"):                    # read and removed before the candidate loads
        key = Path(cfg["result_key"]).read_text()
        Path(cfg["result_key"]).unlink()
    r = precheck(program, Path(os.environ.get("TORCH_EXTENSIONS_DIR", program.parent)))
    sys.stdout.write("\n" + json.dumps({**r, "task": cfg.get("task"), **({"result_key": key} if key else {})}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
