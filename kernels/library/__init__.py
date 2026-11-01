"""Other implementations of our kernel tasks, in one place: the index (kernels/library/index.toml) names each library's
kernel for a task and the upstream source it is (files kept verbatim under src/), and `block` is what every search
prompt shows of them: their source and, when built on a GPU host, their compiled code's PTX/SASS card. No timings.

  python -m kernels.library fetch                       (re-download every indexed file at its library's version)
  python -m kernels.library cards --task T --out knowledge/ptx/T.txt [--dump DIR]   (on the GPU host: each entry's card
      from its library's .so (`sass` in the index) or from DIR, a `kernels.competitors` dump (<name>.ptx|.sass,
      ours.ptx for the engine's kernel))
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.toml"
SRC = HERE / "src"
CARDS = HERE.parents[1] / "knowledge" / "ptx"         # the cards, one file per task (kernels/ptx.py's format)
ENGINE = "engine"                                       # the starting program's card in a task's card file


@dataclass(frozen=True)
class Kernel:
    name: str                                           # <library>.<kernel>
    repo: str
    version: str
    tasks: tuple[str, ...]
    source: tuple[str, ...]                             # upstream paths, each optionally "#L<a>-L<b>"
    sass: dict | None = None

    @property
    def library(self) -> str:
        return self.name.split(".", 1)[0]


def _part(s: str) -> tuple[str, tuple[int, int] | None]:
    path, _, lines = s.partition("#")
    m = re.fullmatch(r"L(\d+)-L(\d+)", lines) if lines else None
    if lines and not m:
        raise ValueError(f"bad line range in {s!r}")
    return path, (int(m[1]), int(m[2])) if m else None


def index(path: Path = INDEX) -> list[Kernel]:
    raw = tomllib.loads(path.read_text())
    libs = raw["libraries"]
    return [Kernel(k["name"], libs[k["name"].split(".", 1)[0]]["repo"], libs[k["name"].split(".", 1)[0]]["version"],
                   tuple(k["tasks"]), tuple(k["source"]), k.get("sass")) for k in raw["kernel"]]


def for_task(task: str, path: Path = INDEX) -> list[Kernel]:
    return [k for k in index(path) if task in k.tasks]


def url(k: Kernel, part: str) -> str:
    p, r = _part(part)
    return f"https://github.com/{k.repo}/blob/{k.version}/{p}" + (f"#L{r[0]}-L{r[1]}" if r else "")


def text(k: Kernel, src: Path = SRC) -> str:
    """The entry's source: each part's lines from the vendored file."""
    out = []
    for part in k.source:
        p, r = _part(part)
        lines = (src / k.library / p).read_text().splitlines()
        out.append("\n".join(lines[r[0] - 1:r[1]] if r else lines))
    return "\n\n".join(out)


def cards(task: str, cards_dir: Path = CARDS) -> dict[str, list[str]]:
    """{implementation: its card's lines} from the task's card file, any timing in a heading dropped."""
    f = cards_dir / f"{task}.txt"
    out: dict[str, list[str]] = {}
    name = None
    for line in (f.read_text().splitlines() if f.exists() else []):
        m = re.match(r"(\S+)(?: \([^)]*\))?: (\d+ kernels)$", line)
        if m:
            name = m[1]
            out[name] = [f"{m[2]}"]
        elif name and line.strip():
            out[name].append(line.strip())
    return out


def block(task: str, path: Path = INDEX, src: Path = SRC, cards_dir: Path = CARDS) -> str:
    """The prompt's part on other implementations: each one's upstream source and, where built, its card, and the
    starting program's card; empty when the task has neither."""
    ks, cs = for_task(task, path), cards(task, cards_dir)
    if not ks and not cs:
        return ""
    out = ["# Other implementations of this operation",
           "Their upstream source as published, and, where built, a card counted from the compiled code of each "
           "(kernels/ptx.py: instruction classes, widest vector access, registers, static shared memory, Hopper "
           "features in use). They are not timed on our shapes."]
    for k in ks:
        lang = "python" if k.source[0].split("#")[0].endswith(".py") else "cuda"
        out.append(f"## {k.name} ({k.library} {k.version}; " + ", ".join(url(k, p) for p in k.source) + ")")
        out.append(f"```{lang}\n{text(k, src)}\n```")
        out.append("Card: " + ("\n".join(cs[k.name]) if k.name in cs else "not built"))
    if ENGINE in cs:
        out.append("## The starting program's card\n" + "\n".join(cs[ENGINE]))
    return "\n\n".join(out)


def fetch(path: Path = INDEX, src: Path = SRC) -> list[Path]:
    """Download every indexed file at its library's version into src/ (whole files: ranges are applied when read)."""
    got = []
    for k in index(path):
        for part in k.source:
            p, _ = _part(part)
            dest = src / k.library / p
            if dest in got:
                continue
            with urllib.request.urlopen(f"https://raw.githubusercontent.com/{k.repo}/{k.version}/{p}", timeout=60) as r:
                data = r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            got.append(dest)
    return got


def _package_sass(package: str, pattern: str, arch: str | None) -> str:
    """SASS of the functions matching `pattern` in every .so of an installed package."""
    from kernels.competitors import sass
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(f"package {package} is not installed")
    texts = []
    for d in spec.submodule_search_locations:
        for so in sorted(Path(d).rglob("*.so")):
            try:
                texts.append(sass(str(so), pattern, arch=arch))
            except RuntimeError:
                continue                          # a .so with no device code
    return "".join(texts)


def build_cards(task: str, dump: Path | None, arch: str | None) -> tuple[list[dict], dict[str, str]]:
    """Each entry's card (from the dump when it has the entry's code, else from its package's .so) and the engine's
    from the dump's ours.ptx; with why each missing one is missing."""
    from kernels.ptx import card
    built, missing = [], {}
    for k in for_task(task):
        code = ""
        for ext in (".ptx", ".sass"):
            if dump and (dump / f"{k.name}{ext}").exists():
                code = (dump / f"{k.name}{ext}").read_text()
                break
        if not code and k.sass:
            try:
                code = _package_sass(k.sass["package"], k.sass["pattern"], arch)
            except Exception as e:                # noqa: BLE001  one entry's failure keeps the others' cards
                missing[k.name] = f"{type(e).__name__}: {e}"[:300]
                continue
        c = card(k.name, code) if code else None
        if c and c["kernels"]:
            built.append(c)
        else:
            missing.setdefault(k.name, "no compiled code found")
    if dump and (dump / "ours.ptx").exists():
        c = card(ENGINE, (dump / "ours.ptx").read_text())
        if c["kernels"]:
            built.insert(0, c)
        else:
            missing[ENGINE] = "no compiled code in ours.ptx"
    return built, missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernels.library")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    c = sub.add_parser("cards")
    c.add_argument("--task", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--dump", help="a kernels.competitors out dir")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        print(json.dumps({"files": len(fetch())}))
        return 0
    from kernels.ptx import render
    import torch
    arch = "sm_%d%d" % torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    built, missing = build_cards(a.task, Path(a.dump) if a.dump else None, arch)
    Path(a.out).write_text(render(built) + "\n")
    print(json.dumps({"cards": [c["implementation"] for c in built], "missing": missing, "out": a.out}))
    return 0
