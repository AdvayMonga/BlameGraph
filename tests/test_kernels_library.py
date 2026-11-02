"""The library of other implementations (kernels/library): the index, the vendored source, the prompt block and the
cards, no network or GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernels import library as L  # noqa: E402
from kernels.tasks import TASKS  # noqa: E402

INDEX = """[libraries]
a = { repo = "o/a", version = "v1" }

[[kernel]]
name = "a.k"
tasks = ["t"]
source = ["x/k.cu#L2-L3", "x/k.cu#L5-L5"]
sass = { package = "a", pattern = "k" }

[[kernel]]
name = "a.py"
tasks = ["t", "u"]
source = ["y.py"]
"""


def make(tmp_path):
    (tmp_path / "index.toml").write_text(INDEX)
    (tmp_path / "src" / "a" / "x").mkdir(parents=True)
    (tmp_path / "src" / "a" / "x" / "k.cu").write_text("1\n2\n3\n4\n5\n6\n")
    (tmp_path / "src" / "a" / "y.py").write_text("def f(): pass\n")
    (tmp_path / "cards").mkdir()
    (tmp_path / "cards" / "t.txt").write_text('engine: 1 kernels\n  e: {"mma": 1}\na.k (0.5 ms on our shapes): 1 kernels\n'
                                              '  k: {"mma": 2}\n')
    return tmp_path / "index.toml", tmp_path / "src", tmp_path / "cards"


def test_every_indexed_file_is_vendored_its_ranges_hold_and_its_tasks_exist():
    for k in L.index():
        assert set(k.tasks) <= set(TASKS), k.name
        for part in k.source:
            path, r = L._part(part)
            lines = (L.SRC / k.library / path).read_text().splitlines()
            assert r is None or 1 <= r[0] <= r[1] <= len(lines), part
    assert {t for k in L.index() for t in k.tasks} == set(TASKS) - {"toy_rmsnorm"}     # every real task has some


def test_the_block_shows_each_source_by_range_with_its_url_and_card_and_no_timings(tmp_path):
    idx, src, cards = make(tmp_path)
    b = L.block("t", idx, src, cards)
    assert "```cuda\n2\n3\n\n5\n```" in b and "```python\ndef f(): pass\n```" in b
    assert "https://github.com/o/a/blob/v1/x/k.cu#L2-L3" in b and 'k: {"mma": 2}' in b
    assert "ms" not in b.replace("instruction classes", "")            # a card's timing never reaches the prompt
    assert b.index("## a.py") < b.index("Card: not built") and "## The starting program's card" in b
    assert L.block("u", idx, src, cards).count("## ") == 1 and L.block("none", idx, src, cards) == ""


def test_cards_come_from_a_dump_before_the_package(tmp_path, monkeypatch):
    idx, *_ = make(tmp_path)
    monkeypatch.setattr(L, "INDEX", idx)
    monkeypatch.setattr(L, "for_task", lambda task, path=idx: [k for k in L.index(idx) if task in k.tasks])
    dump = tmp_path / "dump"
    dump.mkdir()
    (dump / "a.k.ptx").write_text(".visible .entry k(\n)\n{\n\tmma.sync.aligned.m16n8k16 x;\n\tret;\n}\n")
    (dump / "ours.ptx").write_text(".visible .entry e(\n)\n{\n\tret;\n}\n")
    monkeypatch.setattr(L, "_package_sass", lambda *a: (_ for _ in ()).throw(AssertionError("not reached")))
    built, missing = L.build_cards("t", dump, "sm_90")
    assert [c["implementation"] for c in built] == ["engine", "a.k"] and built[1]["per_kernel"]["k"]["mma"] == 1
    assert missing == {"a.py": "no compiled code found"}
