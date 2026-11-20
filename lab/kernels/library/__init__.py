"""Other implementations of the engine's stations: index.toml names each library's kernel and its upstream source
(downloaded to local/library on first read); `block` is the search prompt's part on them: their source and PTX/SASS cards.

  lab kernels library cards --name NAME --out FILE [--dump DIR]   (on the GPU host: the card of each entry whose
      `cards` names NAME, from its library's .so (`sass` in the index) or from DIR (<entry name>.ptx|.sass, ours.ptx for
      the engine's kernel); FILE is the card file ptx/NAME.txt here)
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.toml"
SRC = HERE.parents[2] / "local" / "library"   # <library>/<version>/<path>: each indexed file as downloaded
CARDS = HERE / "ptx"         # the card files (lab/kernels/library/ptx.py's format)
ENGINE = "engine"                                       # the engine's own kernel's card in a card file


@dataclass(frozen=True)
class Kernel:
    name: str                                           # <library>.<kernel>
    repo: str
    version: str
    cards: tuple[str, ...]                              # the card files (ptx/<name>.txt) its card is in
    source: tuple[str, ...]                             # upstream paths, each optionally "#L<a>-L<b>"
    sass: dict | None = None
    stations: tuple[str, ...] = ()                      # the engine stations it implements (a region's prompt)

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
                   tuple(k.get("cards", ())), tuple(k["source"]), k.get("sass"), tuple(k.get("stations", ())))
            for k in raw["kernel"]]


def for_stations(names, path: Path = INDEX) -> list[Kernel]:
    """The entries implementing any of the engine stations `names` (a region's)."""
    return [k for k in index(path) if set(k.stations) & set(names)]


def url(k: Kernel, part: str) -> str:
    p, r = _part(part)
    return f"https://github.com/{k.repo}/blob/{k.version}/{p}" + (f"#L{r[0]}-L{r[1]}" if r else "")


def _file(k: Kernel, path: str, src: Path) -> Path:
    """The upstream file at the library's version, downloaded the first time it is read."""
    dest = src / k.library / k.version / path
    if not dest.exists():
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/{k.repo}/{k.version}/{path}", timeout=60) as r:
            data = r.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return dest


def text(k: Kernel, src: Path = SRC) -> str:
    """The entry's source: each part's lines of its upstream file."""
    out = []
    for part in k.source:
        p, r = _part(part)
        lines = _file(k, p, src).read_text().splitlines()
        out.append("\n".join(lines[r[0] - 1:r[1]] if r else lines))
    return "\n\n".join(out)


def cards(name: str, cards_dir: Path = CARDS) -> dict[str, list[str]]:
    """{implementation: its card's lines} from the card file `name`, any timing in a heading dropped."""
    f = cards_dir / f"{name}.txt"
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


def block(stations, path: Path = INDEX, src: Path = SRC, cards_dir: Path = CARDS) -> str:
    """The prompt's part on other implementations of a region's `stations`: each one's upstream source and card;
    empty when no entry implements one of them."""
    ks = for_stations(stations, path)
    if not ks:
        return ""
    cs = {}
    for name in {n for k in ks for n in k.cards}:
        cs.update(cards(name, cards_dir))
    out = ["# Other implementations of these operations",
           "Their upstream source as published, and, where built, a card counted from the compiled code of each "
           "(lab/kernels/library/ptx.py: instruction classes, widest vector access, registers, static shared memory, Hopper "
           "features in use). They are not timed on our shapes."]
    for k in ks:
        lang = "python" if k.source[0].split("#")[0].endswith(".py") else "cuda"
        out.append(f"## {k.name} ({k.library} {k.version}; " + ", ".join(url(k, p) for p in k.source) + ")")
        out.append(f"```{lang}\n{text(k, src)}\n```")
        out.append("Card: " + ("\n".join(cs[k.name]) if k.name in cs else "not built"))
    return "\n\n".join(out)


def _tool(name: str) -> str:
    return shutil.which(name) or str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / name)


def cuobjdump(module_file: str) -> str:
    """`cuobjdump -sass` of a file. Raises if cuobjdump fails."""
    p = subprocess.run([_tool("cuobjdump"), "-sass", module_file], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cuobjdump exited {p.returncode} on {module_file}: {p.stderr.strip()[:300]}")
    return p.stdout


def sass(module_file: str, pattern: str, arch: str | None = None, dump=cuobjdump) -> str:
    """The SASS of the functions in a library's extension whose names match `pattern`, only `arch`'s code when given.
    `dump(file)` disassembles. Raises if cuobjdump fails."""
    code = dump(module_file)
    if arch:                                     # cuobjdump heads each architecture's code with "code for sm_XY"
        code = "".join(c for c in re.split(r"(?=\n\s*code for sm_)", code) if re.match(rf"\s*code for {arch}[a-z]?\s", c))
    return "".join(b for b in re.split(r"(?=\n\s+Function : )", code) if re.search(r"Function : \S*(" + pattern + r")", b))


@functools.lru_cache(maxsize=None)
def _dump(so: str) -> str:
    """A .so's disassembly, once per process: every entry of a package reads the same files."""
    return cuobjdump(so)


def _names(so: Path, pattern: str) -> bool:
    """Whether `so`'s bytes hold a match of `pattern` (a kernel's mangled name is a plain string in the file)."""
    import mmap
    if not so.stat().st_size:
        return False
    with open(so, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        return re.search(pattern.encode(), m) is not None


def _package_sass(package: str, pattern: str, arch: str | None) -> str:
    """SASS of the functions matching `pattern` in every .so of an installed package."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(f"package {package} is not installed")
    texts = []
    for d in spec.submodule_search_locations:
        for so in sorted(Path(d).rglob("*.so")):
            if not _names(so, pattern):
                continue
            try:
                texts.append(sass(str(so), pattern, arch=arch, dump=_dump))
            except RuntimeError:
                continue                          # a .so with no device code
    return "".join(texts)


def build_cards(name: str, dump: Path | None, arch: str | None) -> tuple[list[dict], dict[str, str]]:
    """The card of each entry of the card file `name` (from the dump when it has the entry's code, else from its
    package's .so) and the engine's from the dump's ours.ptx; with why each missing one is missing."""
    from lab.kernels.library.ptx import card
    built, missing = [], {}
    for k in (k for k in index() if name in k.cards):
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
    ap = argparse.ArgumentParser(prog="lab kernels library")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cards")
    c.add_argument("--name", required=True, help="the card file's name")
    c.add_argument("--out", required=True)
    c.add_argument("--dump", help="a dir of <entry name>.ptx|.sass files and optionally ours.ptx")
    a = ap.parse_args(argv)
    from lab.kernels.library.ptx import render
    import torch
    arch = "sm_%d%d" % torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    built, missing = build_cards(a.name, Path(a.dump) if a.dump else None, arch)
    Path(a.out).write_text(render(built) + "\n")
    print(json.dumps({"cards": [c["implementation"] for c in built], "missing": missing, "out": a.out}))
    return 0
