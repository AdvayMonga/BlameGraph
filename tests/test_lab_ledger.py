"""lab.ledger appends stamped records, stores blobs by hash, guards held-out detail, and merges another ledger."""

from __future__ import annotations

import json


import pytest

from lab import ledger


def test_append_stamps_and_reads_back(tmp_path):
    a = ledger.append({"kind": "test", "session": "s1", "snapshot": "x", "pass": True}, tmp_path)
    b = ledger.append({"kind": "bench", "session": "s1", "snapshot": "y"}, tmp_path)
    assert a["schema"] == ledger.SCHEMA and a["id"].startswith("ev-") and a["at"]
    assert [r["id"] for r in ledger.records(tmp_path)] == [a["id"], b["id"]]
    assert [r["id"] for r in ledger.records(tmp_path, snapshot="y")] == [b["id"]]


@pytest.mark.parametrize("record", [
    {"kind": "nope"},
    {"kind": "test", "id": "forged"},
    {"kind": "submit", "config": {"split": "heldout"}, "raw": {"per_request": "blobs/x"}},
    {"kind": "submit", "config": {"split": "heldout"},
     "metrics": {"ttft_p50_ms": {"delta_pct": -3, "by_class": {"cold_start": -9}}}},
    # held-out data nested where the guard used to look away: under result, args, or a list
    {"kind": "submit", "result": {"split": "heldout", "per_request": [1, 2, 3]}},
    {"kind": "bench", "args": {"split": "heldout"}, "result": {"metrics": {}}},
    {"kind": "submit", "result": {"runs": [{"split": "heldout", "rows": "blobs/x"}]}},
])
def test_rejects(tmp_path, record):
    with pytest.raises(ledger.LedgerError):
        ledger.append(record, tmp_path)
    assert list(ledger.records(tmp_path)) == []


def test_a_tool_record_shaped_like_tools_record_cannot_carry_heldout(tmp_path):
    record = {"kind": "submit", "run": "r", "session": "s", "tool": "submit", "args": {}, "snapshot": "x",
              "snapshot_blob": "blobs/a", "patch": "blobs/b", "cost": {"usd": 0},
              "result": {"split": "heldout", "metrics": {"ttft_p50_ms": {"delta_pct": -3}}}}
    with pytest.raises(ledger.LedgerError):
        ledger.append(record, tmp_path)


def test_claim_may_name_the_split(tmp_path):
    ledger.append({"kind": "note", "claim": {"text": "checked", "split": "heldout"}}, tmp_path)


def test_heldout_aggregate_is_allowed(tmp_path):
    m = {"ttft_p50_ms": {"base": 800, "new": 700, "delta_pct": -12.5, "band_pct": 3, "verdict": "win"}}
    ledger.append({"kind": "submit", "config": {"split": "heldout"}, "metrics": m}, tmp_path)


def test_blobs_dedupe_by_content(tmp_path):
    root = tmp_path / "ledger"
    d = tmp_path / "bundle"
    d.mkdir()
    (d / "meta.json").write_text("{}")
    rel = ledger.put_blob(d, root)
    assert ledger.put_blob(d, root) == rel and (root / rel / "meta.json").exists()
    assert ledger.put_blob(b"diff", root, ".patch").endswith(".patch")
    assert ledger.put_blob(b"diff", root, ".patch") == ledger.put_blob(b"diff", root, ".patch")


def test_merge_adds_only_new_records_and_blobs(tmp_path):
    root, vm = tmp_path / "local", tmp_path / "vm"
    mine = ledger.append({"kind": "note", "run": "a"}, root)
    d = tmp_path / "bundle"; d.mkdir(); (d / "meta.json").write_text("{}")
    rel = ledger.put_blob(d, vm)
    theirs = [ledger.append({"kind": "profile", "run": "b", "result": {"bundle": rel}}, vm),
              ledger.append({"kind": "note", "run": "b"}, vm)]
    assert ledger.merge(vm, root) == 2 and ledger.merge(vm, root) == 0
    assert [r["id"] for r in ledger.records(root)] == [mine["id"]] + [r["id"] for r in theirs]
    assert (root / rel / "meta.json").exists()


def test_merge_refuses_a_record_the_ledger_would_not_accept(tmp_path):
    root, vm = tmp_path / "local", tmp_path / "vm"
    ledger.append({"kind": "note", "run": "b"}, vm)
    with open(vm / "ledger.jsonl", "a") as f:       # written around append, as agent code on the VM could
        f.write(json.dumps({"id": "ev-x", "at": "t", "schema": 1, "kind": "submit", "config": {"split": "heldout"},
                            "metrics": {"ttft": {"new": {"per_request": [1, 2]}}}}) + "\n")
    with pytest.raises(ledger.LedgerError, match="ev-x refused, nothing merged"):
        ledger.merge(vm, root)
    assert not list(ledger.records(root))


def test_merge_keeps_legacy_findings_readable(tmp_path):
    root, vm = tmp_path / "local", tmp_path / "vm"
    ledger.append({"kind": "note", "run": "b"}, vm)
    with open(vm / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"id": "ev-f", "at": "t", "schema": 1, "kind": "finding", "claim": "old"}) + "\n")
    assert ledger.merge(vm, root) == 2
    with pytest.raises(ledger.LedgerError):
        ledger.append({"kind": "finding", "claim": "new"}, root)       # never written any more


def test_root_is_anchored_on_the_repo_not_cwd():
    assert ledger.ROOT.is_absolute() and ledger.ROOT.parts[-2:] == ("lab", "ledger")


def test_heldout_detail_cannot_hide_in_nested_or_extra_fields(tmp_path):
    nested = {"kind": "submit", "config": {"split": "heldout"},
              "metrics": {"ttft": {"base": 1, "new": {"per_request": [1, 2]}}}}
    extra = {"kind": "submit", "config": {"split": "heldout"}, "gates": {"per_request": [1, 2]}}
    for record in (nested, extra, {"kind": "submit", "config": "heldout"},
                   {"kind": "submit", "config": {"split": "heldout"}, "metrics": [1, 2]}):
        with pytest.raises(ledger.LedgerError):
            ledger.append(record, tmp_path)


def test_non_json_values_are_refused_not_coerced(tmp_path):
    with pytest.raises(ledger.LedgerError, match="lossless"):
        ledger.append({"kind": "test", "claim": {"s": {1, 2}}}, tmp_path)
    assert list(ledger.records(tmp_path)) == []


def test_torn_final_line_is_skipped_but_corruption_raises(tmp_path):
    a = ledger.append({"kind": "test"}, tmp_path)
    path = tmp_path / "ledger.jsonl"
    path.write_text(path.read_text() + '{"kind": "te')
    assert [r["id"] for r in ledger.records(tmp_path)] == [a["id"]]
    path.write_text('{"kind": "te\n' + path.read_text())
    with pytest.raises(ledger.LedgerError):
        list(ledger.records(tmp_path))


def test_dir_blob_hash_distinguishes_layouts_and_empty(tmp_path):
    a, b, empty = tmp_path / "a", tmp_path / "b", tmp_path / "e"
    for d in (a, b, empty):
        d.mkdir()
    (a / "a").write_bytes(b"bc")
    (b / "ab").write_bytes(b"c")
    root = tmp_path / "ledger"
    rels = {ledger.put_blob(a, root), ledger.put_blob(b, root), ledger.put_blob(empty, root),
            ledger.put_blob(b"", root)}
    assert len(rels) == 4
    assert not list((root / "blobs").glob("*.tmp-*"))


def test_ids_use_utc_date_and_long_suffix(tmp_path):
    r = ledger.append({"kind": "test"}, tmp_path)
    date, suffix = r["id"].split("-")[1:]
    assert r["at"].startswith(f"{date[:4]}-{date[4:6]}-{date[6:]}") and len(suffix) == 12


@pytest.mark.parametrize("bundle,ok", [(["h2"], True), ({"rows": [1, 2]}, False), ([{"row": 1}], False)])
def test_a_heldout_experiment_bundle_is_a_list_of_ids(tmp_path, bundle, ok):
    m = {"saturated": {"base": 6, "new": 7, "delta_pct": 16.7, "band_pct": 3, "verdict": "improved"}}
    exp = {"hypothesis": "h1", "bundle": bundle, "independence": "x",
           "expect": {"metric": "saturated", "value": 10, "unit": "change_pct", "why": "w"}}
    rec = {"kind": "submit", "config": {"split": "heldout"}, "metrics": m, "experiment": exp}
    if ok:
        ledger.append(rec, tmp_path)
    else:
        with pytest.raises(ledger.LedgerError):
            ledger.append(rec, tmp_path)
