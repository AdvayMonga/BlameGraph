"""Push the controller's data to the remote and pull it back. Everything goes up zstd-compressed, under keys that
are never rewritten:

  ledger/<start>-<end>.jsonl.zst      byte range [start, end) of ledger.jsonl, whole lines only
  blobs/<name>.zst | .tar.zst         one ledger blob (file | directory), named as in the ledger
  archive/<name>-<hash>.tar.zst       anything else kept (GPU run reports, correctness outputs)
  heldout/guard/state-<n>.json.zst    the holdout guard's state after its n-th query
  heldout/private/<run>/<dir>.tar.zst submit's held-out passive data (lab/LEDGER.md)
  heldout/corpus/<version>/<class>/heldout.jsonl.zst   the held-out traces, which are not in git
  heldout/corpus/<version>/seed.zst   the corpus build seed, which reproduces them
  heldout/archive/...                 archive of held-out material
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

import zstandard

from lab import ledger
from store.remote import Remote

SEGMENT = re.compile(r"ledger/(\d+)-(\d+)\.jsonl\.zst$")


def pack(src: Path, out: Path) -> None:
    """A file as zstd, a directory as a zstd tar holding it under its own name."""
    with open(out, "wb") as fh, zstandard.ZstdCompressor(level=10, threads=-1).stream_writer(fh) as w:
        if Path(src).is_dir():
            with tarfile.open(fileobj=w, mode="w|") as tar:
                tar.add(src, arcname=Path(src).name)
        else:
            with open(src, "rb") as r:
                shutil.copyfileobj(r, w, 1 << 20)


def unpack(packed: Path, dest: Path) -> None:
    """Back to `dest` (a .tar.zst to the directory it held); appears whole or not at all."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".unpack-", dir=dest.parent))
    try:
        with open(packed, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as r:
            if str(packed).endswith(".tar.zst"):
                with tarfile.open(fileobj=r, mode="r|") as tar:
                    tar.extractall(tmp, filter="data")
                os.replace(next(tmp.iterdir()), dest)
            else:
                with open(tmp / "f", "wb") as w:
                    shutil.copyfileobj(r, w, 1 << 20)
                os.replace(tmp / "f", dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _put(remote: Remote, key: str, src: Path, have: set[str]) -> bool:
    if key in have:
        return False
    with tempfile.TemporaryDirectory() as d:
        packed = Path(d) / Path(key).name
        pack(src, packed)
        return remote.put(key, packed)


def _get(remote: Remote, key: str, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as d:
        packed = Path(d) / Path(key).name
        remote.get(key, packed)
        unpack(packed, dest)


def blob_key(p: Path) -> str:
    return f"blobs/{p.name}.tar.zst" if p.is_dir() else f"blobs/{p.name}.zst"


def _segments(remote: Remote) -> list[tuple[int, int, str]]:
    segs = sorted((int(m[1]), int(m[2]), k) for k in remote.keys("ledger/") if (m := SEGMENT.search(k)))
    pos = 0
    for start, end, k in segs:
        if start != pos:
            raise RuntimeError(f"remote ledger has a gap or overlap at {k} (expected start {pos})")
        pos = end
    return segs


def push_ledger(root: Path, remote: Remote) -> int:
    """Upload the whole lines appended since the last push; returns bytes uploaded."""
    f = Path(root) / "ledger.jsonl"
    if not f.exists():
        return 0
    data = f.read_bytes()
    end = data.rfind(b"\n") + 1
    segs = _segments(remote)
    done = segs[-1][1] if segs else 0
    if end < done:
        raise RuntimeError(f"local ledger ({end} bytes) is behind the remote ({done}); pull first")
    if end == done:
        return 0
    with tempfile.TemporaryDirectory() as d:
        part = Path(d) / "part.jsonl"
        part.write_bytes(data[done:end])
        _put(remote, f"ledger/{done:012d}-{end:012d}.jsonl.zst", part, set())
    return end - done


def pull_ledger(root: Path, remote: Remote) -> int:
    """Append the remote segments the local ledger lacks; returns bytes added. Refuses a diverged local ledger."""
    f = Path(root) / "ledger.jsonl"
    size = f.stat().st_size if f.exists() else 0
    segs = _segments(remote)
    if size and not any(end == size for _, end, _ in segs) and size < (segs[-1][1] if segs else 0):
        raise RuntimeError(f"local ledger ({size} bytes) ends inside a remote segment: it has diverged")
    added = 0
    f.parent.mkdir(parents=True, exist_ok=True)
    for start, end, key in segs:
        if start < size:
            continue
        with tempfile.TemporaryDirectory() as d:
            _get(remote, key, Path(d) / "part")
            with ledger._lock(root), open(f, "ab") as out:
                out.write((Path(d) / "part").read_bytes())
        added += end - start
    return added


def push(root: Path, remote: Remote, runs: Path | None = None, corpus: Path | None = None) -> dict:
    """Ledger, blobs, guard state, held-out passive data and held-out traces; returns what went up."""
    root = Path(root)
    out = {"ledger_bytes": push_ledger(root, remote), "blobs": 0, "guard": 0, "heldout_private": 0, "corpus": 0}
    have = set(remote.keys("blobs/"))
    for p in sorted((root / "blobs").iterdir()) if (root / "blobs").is_dir() else []:
        if ".tmp-" not in p.name:
            out["blobs"] += _put(remote, blob_key(p), p, have)
    state = root / "holdout_state.json"
    if state.exists():
        n = json.loads(state.read_text())["n_queries"]
        out["guard"] += _put(remote, f"heldout/guard/state-{n:06d}.json.zst", state, set(remote.keys("heldout/guard/")))
    if runs and Path(runs).is_dir():
        have = set(remote.keys("heldout/private/"))
        for d in sorted(Path(runs).glob("*/heldout-private/*")):
            if (d / "client_rows.jsonl").exists():        # written last: the held-out serve is complete
                out["heldout_private"] += _put(remote, f"heldout/private/{d.parents[1].name}/{d.name}.tar.zst", d, have)
    if corpus and Path(corpus).is_dir():
        version = corpus_version(Path(corpus))
        have = set(remote.keys(f"heldout/corpus/{version}/"))
        for p in sorted(Path(corpus).glob("*/heldout.jsonl")):
            out["corpus"] += _put(remote, f"heldout/corpus/{version}/{p.parent.name}/heldout.jsonl.zst", p, have)
        if (Path(corpus) / ".seed").exists():
            out["corpus"] += _put(remote, f"heldout/corpus/{version}/seed.zst", Path(corpus) / ".seed", have)
    return out


def pull(root: Path, remote: Remote, blobs: bool = False, corpus: Path | None = None) -> dict:
    """The ledger; with `blobs`, every blob missing locally; with `corpus`, its held-out traces; the guard state
    only when there is none locally (an older copy must never replace a newer one)."""
    root = Path(root)
    out = {"ledger_bytes": pull_ledger(root, remote), "blobs": 0, "guard": 0, "corpus": 0}
    if blobs:
        for key in remote.keys("blobs/"):
            name = Path(key).name.removesuffix(".zst").removesuffix(".tar")
            if not (root / "blobs" / name).exists():
                _get(remote, key, root / "blobs" / name)
                out["blobs"] += 1
    state = root / "holdout_state.json"
    guard = remote.keys("heldout/guard/")
    if guard and not state.exists():
        _get(remote, guard[-1], state)
        out["guard"] = 1
    if corpus:
        version = corpus_version(Path(corpus))
        for key in remote.keys(f"heldout/corpus/{version}/"):
            dest = (Path(corpus) / ".seed" if key.endswith("/seed.zst")
                    else Path(corpus) / key.split("/")[-2] / "heldout.jsonl")
            if not dest.exists():
                _get(remote, key, dest)
                out["corpus"] += 1
    return out


def archive(path: Path, remote: Remote, heldout: bool = False) -> str:
    """Keep a file or directory as one compressed object named by its content hash; returns the key."""
    path = Path(path)
    ext = ".tar.zst" if path.is_dir() else ".zst"
    key = f"{'heldout/' if heldout else ''}archive/{path.name}-{ledger._hash(path)[:16]}{ext}"
    _put(remote, key, path, set(remote.keys(key)))
    return key


def corpus_version(corpus: Path) -> str:
    return json.loads((Path(corpus) / "manifest.json").read_text())["corpus_version"]
