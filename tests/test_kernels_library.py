"""The library of other implementations (kernels/library): the index, the downloaded source, the prompt block and the
cards, no network or GPU."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.kernels import library as L  # noqa: E402

INDEX = """[libraries]
a = { repo = "o/a", version = "v1" }

[[kernel]]
name = "a.k"
cards = ["t"]
stations = ["st1"]
source = ["x/k.cu#L2-L3", "x/k.cu#L5-L5"]
sass = { package = "a", pattern = "k" }

[[kernel]]
name = "a.py"
cards = ["t", "u"]
stations = ["st1", "st2"]
source = ["y.py"]
"""


def make(tmp_path):
    (tmp_path / "index.toml").write_text(INDEX)
    (tmp_path / "src" / "a" / "v1" / "x").mkdir(parents=True)
    (tmp_path / "src" / "a" / "v1" / "x" / "k.cu").write_text("1\n2\n3\n4\n5\n6\n")
    (tmp_path / "src" / "a" / "v1" / "y.py").write_text("def f(): pass\n")
    (tmp_path / "cards").mkdir()
    (tmp_path / "cards" / "t.txt").write_text('engine: 1 kernels\n  e: {"mma": 1}\na.k (0.5 ms on our shapes): 1 kernels\n'
                                              '  k: {"mma": 2}\n')
    return tmp_path / "index.toml", tmp_path / "src", tmp_path / "cards"


def test_every_entrys_ranges_are_well_formed_and_a_file_is_downloaded_once(tmp_path, monkeypatch):
    for k in L.index():
        for part in k.source:
            r = L._part(part)[1]
            assert r is None or 1 <= r[0] <= r[1], part
    asked = []

    class Reply:
        def __init__(self, url):
            asked.append(url)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return b"1\n2\n3\n4\n5\n6\n"
    monkeypatch.setattr(L.urllib.request, "urlopen", lambda url, timeout: Reply(url))
    (tmp_path / "index.toml").write_text(INDEX)
    k = L.index(tmp_path / "index.toml")[0]
    assert L.text(k, tmp_path / "src") == L.text(k, tmp_path / "src") == "2\n3\n\n5"
    assert asked == ["https://raw.githubusercontent.com/o/a/v1/x/k.cu"]
    assert (tmp_path / "src" / "a" / "v1" / "x" / "k.cu").exists()


def test_the_block_shows_each_source_by_range_with_its_url_and_card_and_no_timings(tmp_path):
    idx, src, cards = make(tmp_path)
    b = L.block(["st1"], idx, src, cards)
    assert b.startswith("# Other implementations of these operations")
    assert "```cuda\n2\n3\n\n5\n```" in b and "```python\ndef f(): pass\n```" in b
    assert "https://github.com/o/a/blob/v1/x/k.cu#L2-L3" in b and 'k: {"mma": 2}' in b
    assert "ms" not in b.replace("instruction classes", "")            # a card's timing never reaches the prompt
    assert b.index("## a.py") < b.index("Card: not built") and '{"mma": 1}' not in b      # not the engine's card
    assert L.block(["st2"], idx, src, cards).count("## ") == 1
    assert L.block(["none"], idx, src, cards) == L.block([], idx, src, cards) == ""


def test_cards_come_from_a_dump_before_the_package(tmp_path, monkeypatch):
    idx, *_ = make(tmp_path)
    monkeypatch.setattr(L, "INDEX", idx)
    monkeypatch.setattr(L, "index", lambda path=idx, index=L.index: index(path))
    dump = tmp_path / "dump"
    dump.mkdir()
    (dump / "a.k.ptx").write_text(".visible .entry k(\n)\n{\n\tmma.sync.aligned.m16n8k16 x;\n\tret;\n}\n")
    (dump / "ours.ptx").write_text(".visible .entry e(\n)\n{\n\tret;\n}\n")
    monkeypatch.setattr(L, "_package_sass", lambda *a: (_ for _ in ()).throw(AssertionError("not reached")))
    built, missing = L.build_cards("t", dump, "sm_90")
    assert [c["implementation"] for c in built] == ["engine", "a.k"] and built[1]["per_kernel"]["k"]["mma"] == 1
    assert missing == {"a.py": "no compiled code found"}
    assert L.build_cards("u", None, "sm_90") == ([], {"a.py": "no compiled code found"})   # by card file name


def test_sass_keeps_the_named_kernels_of_the_running_gpus_code_and_raises_when_cuobjdump_fails(monkeypatch):
    def proc(stdout="", returncode=0, stderr=""):
        return types.SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)
    dump = ("\n\tcode for sm_80\n\t\tFunction : _Z15rms_norm_kernelv\n\t/*0000*/ OLD ;\n"
            "\n\tcode for sm_90a\n\t\tFunction : _Z15rms_norm_kernelv\n\t/*0000*/ NEW ;\n"
            "\n\t\tFunction : _Z9otherKernv\n\t/*0000*/ BRA 0x0 ;\n")
    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: proc(dump))
    got = L.sass("lib.so", r"rms_norm_kernel", arch="sm_90")
    assert "NEW" in got and "OLD" not in got and "otherKern" not in got
    assert "OLD" in L.sass("lib.so", r"rms_norm_kernel", arch="sm_80") and "OLD" in L.sass("lib.so", r"rms_norm_kernel")
    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: proc("", 1, "no device code"))
    try:
        L.sass("lib.so", r"rms_norm_kernel")
        raise AssertionError("a failed cuobjdump returned")
    except RuntimeError as e:
        assert "no device code" in str(e)


ENGINE_STATIONS = {"input_norm", "qkv", "qk_norm_rope", "kv_append", "attention", "o_proj", "add_norm", "router", "route",
                   "gate_up", "silu_mul", "down", "residual", "p_input_norm", "p_qkv", "p_qk_norm_rope", "p_kv_store",
                   "p_attention", "p_o_proj", "p_add_norm", "p_moe"}       # inference-server's DECODE and PREFILL


def test_every_entry_names_engine_stations_and_a_region_gets_the_entries_of_its_stations():
    assert all(k.stations and set(k.stations) <= ENGINE_STATIONS for k in L.index())
    assert {k.name for k in L.for_stations(["p_attention"])} == {"flashinfer.batch_prefill"}
    assert {"vllm.fused_moe", "sglang.fused_moe"} <= {k.name for k in L.for_stations(["router", "route", "gate_up"])}
