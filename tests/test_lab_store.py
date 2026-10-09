"""store: push/pull round trips through a directory and a (moto) S3 bucket, write-once keys, held-out material only
under heldout/, and the SQL index with the agent's copy locked down and free of held-out rows."""

from __future__ import annotations

import json
import types

import pytest

from lab import ledger
from lab.tools import Toolbox
from store import index, sync
from store.remote import DirRemote, S3Remote


def _lab(root, runs):
    """A ledger with a bench (client rows + device samples), a profile bundle, a submit, a noise record, a guard
    state and one held-out serve under runs/."""
    rows = root.parent / "rows.jsonl"
    rows.write_text("".join(json.dumps({"id": i, "status": "ok", "ttft_s": 0.1 * i}) + "\n" for i in range(3)))
    dev = root.parent / "device"; dev.mkdir()
    (dev / "meta.json").write_text(json.dumps({"available": True, "source": "nvidia-smi"}))
    (dev / "samples.csv").write_text("epoch_s,timestamp,utilization.gpu\n1.0,t,90\n1.1,t,95\n")
    bundle = root.parent / "bundle"; bundle.mkdir()
    (bundle / "kernels.json").write_text(json.dumps({"kernels": [{"name": "gemm", "count": 4, "total_us": 10.0}]}))
    art = {"client_rows": ledger.put_blob(rows, root, ".jsonl"), "device": ledger.put_blob(dev, root),
           "telemetry": None, "serve_log": None}
    ledger.append({"kind": "bench", "run": "r1", "snapshot": "s1", "config": {"split": "seen", "tier": "short"},
                   "result": {"verdict": "ok", "metrics": {"saturated": {"objective": "goodput", "value": 18.4,
                              "better": "higher", "valid": True, "invalid_reasons": []}}, "artifacts": art}}, root)
    ledger.append({"kind": "profile", "run": "r1", "snapshot": "s1", "result": {"bundle": ledger.put_blob(bundle, root)}}, root)
    ledger.append({"kind": "submit", "run": "r1", "snapshot": "s1", "config": {"split": "heldout", "tier": "full"},
                   "metrics": {"saturated": {"base": 16.0, "new": 18.0, "delta_pct": 12.5, "band_pct": 4.0,
                                             "verdict": "improved"}}}, root)
    ledger.append({"kind": "noise", "result": {"regime": "saturated", "band_pct": 4.0, "values": [1, 2], "target": "t"}}, root)
    (root / "holdout_state.json").write_text(json.dumps({"n_queries": 3, "budget_left": 97}))
    private = runs / "r1" / "heldout-private" / "s1-20261009T120000"
    private.mkdir(parents=True)
    (private / "client_rows.jsonl").write_text(json.dumps({"id": 0, "ttft_s": 9.9}) + "\n")


@pytest.fixture
def lab(tmp_path):
    root, runs = tmp_path / "ledger", tmp_path / "runs"
    _lab(root, runs)
    return root, runs


def test_pack_round_trips_files_and_dirs(tmp_path):
    d = tmp_path / "d"; (d / "sub").mkdir(parents=True)
    (d / "a.txt").write_text("a" * 10_000); (d / "sub" / "b.bin").write_bytes(bytes(range(256)))
    sync.pack(d, tmp_path / "d.tar.zst"); sync.unpack(tmp_path / "d.tar.zst", tmp_path / "out" / "d2")
    assert (tmp_path / "out/d2/a.txt").read_text() == "a" * 10_000
    assert (tmp_path / "out/d2/sub/b.bin").read_bytes() == bytes(range(256))
    assert (tmp_path / "d.tar.zst").stat().st_size < 2_000          # compressed
    sync.pack(d / "a.txt", tmp_path / "a.zst"); sync.unpack(tmp_path / "a.zst", tmp_path / "a2")
    assert (tmp_path / "a2").read_text() == "a" * 10_000


def test_push_then_pull_restores_everything(lab, tmp_path):
    root, runs = lab
    remote = DirRemote(tmp_path / "bucket")
    corpus = tmp_path / "corpus"; (corpus / "c1").mkdir(parents=True)
    (corpus / "manifest.json").write_text(json.dumps({"corpus_version": "v9"}))
    (corpus / "c1" / "heldout.jsonl").write_text('{"secret": 1}\n')
    out = sync.push(root, remote, runs=runs, corpus=corpus)
    assert out["blobs"] == 3 and out["guard"] == 1 and out["heldout_private"] == 1 and out["corpus"] == 1
    assert sync.push(root, remote, runs=runs, corpus=corpus) == {"ledger_bytes": 0, "blobs": 0, "guard": 0,
                                                                "heldout_private": 0, "corpus": 0}
    keys = remote.keys("")
    assert "heldout/corpus/v9/c1/heldout.jsonl.zst" in keys and "heldout/guard/state-000003.json.zst" in keys
    assert "heldout/private/r1/s1-20261009T120000.tar.zst" in keys
    assert not [k for k in keys if not k.startswith("heldout/") and "secret" in k]

    fresh = tmp_path / "fresh"
    (tmp_path / "corpus2").mkdir(); (tmp_path / "corpus2" / "manifest.json").write_text(json.dumps({"corpus_version": "v9"}))
    got = sync.pull(fresh, remote, blobs=True, corpus=tmp_path / "corpus2")
    assert (fresh / "ledger.jsonl").read_bytes() == (root / "ledger.jsonl").read_bytes()
    assert got["blobs"] == 3 and got["guard"] == 1 and got["corpus"] == 1
    assert sorted(p.name for p in (fresh / "blobs").iterdir()) == sorted(p.name for p in (root / "blobs").iterdir())
    assert (tmp_path / "corpus2" / "c1" / "heldout.jsonl").read_text() == '{"secret": 1}\n'


def test_ledger_goes_up_in_write_once_segments(lab, tmp_path):
    root, _ = lab
    remote = DirRemote(tmp_path / "bucket")
    sync.push_ledger(root, remote)
    ledger.append({"kind": "note", "run": "r1"}, root)
    with open(root / "ledger.jsonl", "a") as f:
        f.write('{"torn')                                     # a writer killed mid-line: not uploaded
    assert sync.push_ledger(root, remote) > 0 and sync.push_ledger(root, remote) == 0
    assert len(remote.keys("ledger/")) == 2
    fresh = tmp_path / "fresh"
    sync.pull_ledger(fresh, remote)
    assert [r["id"] for r in ledger.records(fresh)] == [r["id"] for r in ledger.records(root)]
    # the guard state is never replaced by an older remote copy
    (fresh / "holdout_state.json").write_text(json.dumps({"n_queries": 50}))
    sync.push(root, remote); sync.pull(fresh, remote)
    assert json.loads((fresh / "holdout_state.json").read_text())["n_queries"] == 50


def test_pull_refuses_a_diverged_ledger(lab, tmp_path):
    root, _ = lab
    remote = DirRemote(tmp_path / "bucket")
    sync.push_ledger(root, remote)
    other = tmp_path / "other"
    ledger.append({"kind": "note", "run": "x", "claim": "a different history"}, other)
    with pytest.raises(RuntimeError, match="diverged"):
        sync.pull_ledger(other, remote)


def test_s3_remote_round_trip(lab, tmp_path, monkeypatch):
    moto = pytest.importorskip("moto")
    import boto3
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x"); monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    root, runs = lab
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="bg-data")
        remote = S3Remote("bg-data", client=s3)
        sync.push(root, remote, runs=runs)
        (tmp_path / "x").write_bytes(b"other content")
        assert remote.put(remote.keys("blobs/")[0], tmp_path / "x") is False          # write-once
        fresh = tmp_path / "fresh"
        sync.pull(fresh, remote, blobs=True)
        assert (fresh / "ledger.jsonl").read_bytes() == (root / "ledger.jsonl").read_bytes()
        assert len(list((fresh / "blobs").iterdir())) == 3


def test_index_tables_and_the_agent_never_sees_heldout_rows(lab):
    root, runs = lab
    out = index.build(root, root / "index" / "all.duckdb", heldout_runs=runs)
    assert out["tables"]["client_rows"] == 3 and out["tables"]["device_samples"] == 2
    assert out["tables"]["kernels"] == 1 and out["tables"]["heldout_rows"] == 1 and not out["missing_blobs"]
    q = index.query(root / "index" / "all.duckdb", "SELECT regime, value FROM regimes")
    assert q["rows"] == [["saturated", 18.4]]
    q = index.query(root / "index" / "all.duckdb", "SELECT delta_pct, verdict FROM heldout_aggregates")
    assert q["rows"] == [[12.5, "improved"]]

    agent = index.agent_db(root)
    tables = {r[0] for r in index.query(agent, "SHOW TABLES")["rows"]}
    assert "heldout_rows" not in tables and {"records", "client_rows", "noise"} <= tables
    for sql in (f"SELECT * FROM read_json('{runs}/r1/heldout-private/s1-20261009T120000/client_rows.jsonl')",
                f"ATTACH '{root / 'index' / 'all.duckdb'}' AS a", "SET enable_external_access = true",
                "COPY records TO 'leak.csv'", "CREATE TABLE x AS SELECT 1"):
        with pytest.raises(Exception):
            index.query(agent, sql)


def test_agent_index_follows_the_ledger_and_the_tool_reports_errors(lab):
    root, _ = lab
    tb = Toolbox(types.SimpleNamespace(ledger_root=root))
    n = json.loads(tb.query({"sql": "SELECT count(*) FROM records"}))["rows"][0][0]
    ledger.append({"kind": "note", "run": "r2"}, root)
    assert json.loads(tb.query({"sql": "SELECT count(*) FROM records"}))["rows"][0][0] == n + 1
    assert tb.query({"sql": "SELECT * FROM read_csv('/etc/passwd')"}).startswith("query failed")
