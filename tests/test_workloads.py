"""Converters on synthetic copies of each source's shape (no network): record shape, template stripping, few-shot
turns, manifest. Run: python tests/test_workloads.py"""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import workloads as W  # noqa: E402
from workloads import mlperf_deepseek_r1, mlperf_gpt_oss_120b, mlperf_llama2_70b, mlperf_llama3_1_8b, mlperf_mixtral_8x7b, sharegpt  # noqa: E402

import pandas as pd  # noqa: E402

TMP = Path(tempfile.mkdtemp())
FILES: dict[str, Path] = {}


def fake_download(url, md5=None):
    return FILES[url.rsplit("/", 1)[1]]


def put(name, writer):
    p = TMP / name
    writer(p)
    FILES[name] = p


def check_shape(recs, n):
    assert len(recs) == n, len(recs)
    for r in recs:
        assert ("prompt" in r) != ("messages" in r), r
        assert r["max_tokens"] > 0 and r["id"] and r["subset"]
        if "messages" in r:
            roles = [m["role"] for m in r["messages"]]
            assert roles[-1] == "user" and all(a != b for a, b in zip(roles, roles[1:])), roles
            assert all(m["content"] for m in r["messages"])
        else:
            assert r["prompt"].strip()


def test_llama3_1_8b():
    put("cnn_eval.json", lambda p: p.write_text(json.dumps([{"input": f"article {k}", "output": "s", "tok_input": [1] * (k + 3),
                                                             "instruction": "x"} for k in range(4)])))
    recs = mlperf_llama3_1_8b.records()
    check_shape(recs, 4)
    assert recs[1]["prompt"].startswith("Summarize the following news article in 128 tokens") and "article 1" in recs[1]["prompt"]
    assert recs[1]["build_prompt_tokens"] == 4 and recs[1]["max_tokens"] == 128


def test_llama2_70b():
    df = pd.DataFrame({"id": ["cot.1", "flan.2"], "system_prompt": ["be helpful", ""], "question": ["q1", "q2"],
                       "tok_input_length": [10, 20], "origin": ["cot", "flan"]})
    put("open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz", lambda p: df.to_pickle(p, compression="gzip"))
    recs = mlperf_llama2_70b.records()
    check_shape(recs, 2)
    assert [m["role"] for m in recs[0]["messages"]] == ["system", "user"] and [m["role"] for m in recs[1]["messages"]] == ["user"]
    assert recs[0]["subset"] == "cot" and recs[1]["build_prompt_tokens"] == 20


def test_mixtral():
    shots = "".join(f"[INST] Q: ex {i} [/INST] A: ans {i} " for i in range(5))
    df = pd.DataFrame({"dataset": ["OpenOrca", "GSM8K", "MBXP"], "id": ["a", "b", "c"], "tok_input_len": [5, 6, 7],
                       "input": ["<s>[INST] tell me [/INST]", f"<s> {shots}[INST] Q: final [/INST]",
                                 "<s> [INST] Complete the code\ndef f(): [/INST]Here's the completed code:\n\n```python"]})
    put("09292024_mixtral_15k_mintoken2_v1.pkl", lambda p: df.to_pickle(p))
    recs = mlperf_mixtral_8x7b.records()
    check_shape(recs, 3)
    assert recs[0]["prompt"] == "tell me"
    assert len(recs[1]["messages"]) == 11 and recs[1]["messages"][-1]["content"] == "Q: final" and recs[1]["messages"][1]["content"] == "A: ans 0"
    assert recs[2]["prompt"] == "Complete the code\ndef f():"
    assert not any("[INST]" in json.dumps(r) for r in recs)


def test_deepseek():
    df = pd.DataFrame({"dataset": ["gpqa", "aime1983"], "text_input": ["which?", "sum?"], "tok_input_len": [50, 60]})
    put("mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl", lambda p: df.to_pickle(p))
    recs = mlperf_deepseek_r1.records()
    check_shape(recs, 2)
    assert recs[0]["prompt"] == "which?" and recs[0]["max_tokens"] == 20_000 and recs[1]["subset"] == "aime1983"


def test_gpt_oss():
    harmony = ("<|start|>system<|message|>You are ChatGPT<|end|><|start|>developer<|message|># Instructions\nsummarize<|end|>"
               "<|start|>user<|message|>paper text<|end|><|start|>assistant")
    put("perf_eval_ref.parquet", lambda p: pd.DataFrame({"dataset": ["pubmed_summarization"], "text_input": [harmony],
                                                         "num_tokens": [8000]}).to_parquet(p))
    put("acc_eval_ref.parquet", lambda p: pd.DataFrame({"dataset": ["aime25"], "num_tokens": [100],
                                                        "original_messages": [[{"role": "user", "content": "solve"}]]}).to_parquet(p))
    recs = mlperf_gpt_oss_120b.records()
    check_shape(recs, 2)
    assert recs[0]["messages"] == [{"role": "system", "content": "# Instructions\nsummarize"}, {"role": "user", "content": "paper text"}]
    assert recs[0]["max_tokens"] == 10_240 and recs[0]["subset"] == "perf:pubmed_summarization"
    assert recs[1]["max_tokens"] == 32_768 and recs[1]["subset"] == "acc:aime25" and recs[1]["messages"][0]["content"] == "solve"


def test_sharegpt():
    convs = [{"id": "ok1", "conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]},
             {"id": "ok2", "conversations": [{"from": "human", "value": "a"}, {"from": "gpt", "value": "b"},
                                             {"from": "human", "value": "c"}, {"from": "gpt", "value": "d"}]},
             {"id": "bad-start", "conversations": [{"from": "gpt", "value": "x"}, {"from": "human", "value": "y"}]},
             {"id": "bad-double", "conversations": [{"from": "human", "value": "x"}, {"from": "human", "value": "y"}]},
             {"id": "bad-empty", "conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": "y"}]},
             {"id": "bad-role", "conversations": [{"from": "system", "value": "x"}, {"from": "gpt", "value": "y"}]}]
    p = TMP / "sharegpt.json"
    p.write_text(json.dumps(convs))
    with mock.patch("huggingface_hub.hf_hub_download", return_value=str(p)):
        recs = sharegpt.records()
    check_shape(recs, 2)
    assert recs[0]["prompt"] == "hi" and recs[0]["subset"] == "single" and "build_prompt_tokens" not in recs[0]
    assert [m["content"] for m in recs[1]["messages"]] == ["a", "b", "c"] and recs[1]["subset"] == "multi"


def test_fetch_manifest():
    with mock.patch.object(W, "DATA", TMP / "out"):
        out = W.fetch(mlperf_deepseek_r1)
        man = json.loads((TMP / "out" / "manifest.json").read_text())["mlperf_deepseek_r1"]
    assert out.exists() and man["n"] == 2 and man["limits"]["server"]["tpot_s"] == 0.08 and man["source_md5"] == mlperf_deepseek_r1.MD5
    assert man["mlperf_commit"] == W.MLPERF_COMMIT and len(man["sha256"]) == 64


if __name__ == "__main__":
    for mod in (mlperf_deepseek_r1, mlperf_gpt_oss_120b, mlperf_llama2_70b, mlperf_llama3_1_8b, mlperf_mixtral_8x7b):
        mod.download = fake_download        # each module bound the name at import
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
