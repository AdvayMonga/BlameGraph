"""How much load the client itself can drive before its own timing goes wrong, measured against a fake server that
streams every token on a fixed schedule and logs when it really sent each one. Per request, the client's TPOT minus
the server's (from the server's own send times) is the client's error; a level is clean when that error's p99 stays
under `max_error_s`, the send lag under the runner's limit and no client process is saturated. Run it on the machine
and cores the comparison will use, with nothing on the GPU:

    python -m regimes.clientcheck [--client-procs 8] [--client-cpus 8-15] [--server-cpus 0-7] [--out F]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import socket
import tempfile
import time
from pathlib import Path

from . import runner, workload as W

CLOSED = (64, 256, 1024, 2048, 4096)       # concurrency levels
OPEN = (50.0, 100.0, 200.0, 400.0)         # req/s


async def _handle(reader, writer, tpot_s: float, log):
    try:
        head = await reader.readuntil(b"\r\n\r\n")
        length = next(int(h.split(b":")[1]) for h in head.split(b"\r\n") if h.lower().startswith(b"content-length"))
        trace = next((h.split(b":", 1)[1].strip().decode() for h in head.split(b"\r\n")
                      if h.lower().startswith(b"x-trace-id")), "")
        body = json.loads(await reader.readexactly(length))
        n = int(body["max_tokens"])
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")

        def chunk(obj):
            line = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
            writer.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
        start, sent = time.perf_counter(), []
        for i in range(n):
            await asyncio.sleep(max(0.0, start + i * tpot_s - time.perf_counter()))
            chunk({"choices": [{"index": 0, "delta": {"content": f"t{i} "}}]})
            await writer.drain()
            sent.append(time.perf_counter())
        chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
        chunk({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": n}})
        chunk(b"[DONE]")
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        log.write(json.dumps({"trace_id": trace, "tpot_s": (sent[-1] - sent[0]) / (n - 1) if n > 1 else None}) + "\n")
    except (asyncio.IncompleteReadError, ConnectionError, StopIteration):
        pass
    finally:
        writer.close()


def _serve(port: int, tpot_s: float, log_path: str, cpus):
    runner._pin(cpus)
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(("127.0.0.1", port))
    log = open(log_path, "a", buffering=1)

    async def main():
        srv = await asyncio.start_server(lambda r, w: _handle(r, w, tpot_s, log), sock=sock, backlog=8192)
        await srv.serve_forever()
    asyncio.run(main())


def _server(procs: int, tpot_s: float, cpus, logdir: Path):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=_serve, args=(port, tpot_s, str(logdir / f"server{k}.jsonl"), cpus), daemon=True)
          for k in range(procs)]
    for p in ps:
        p.start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            return ps, f"http://127.0.0.1:{port}"
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("the check's server did not start")


def _judge(rows, logdir: Path, elapsed: float, max_error_s: float) -> dict:
    truth = {}
    for f in logdir.glob("server*.jsonl"):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            truth[d["trace_id"]] = d["tpot_s"]
        f.write_text("")
    err = sorted(r.tpot_s - truth[f"bg-{r.id}"] for r in rows
                 if r.status == "ok" and r.tpot_s is not None and truth.get(f"bg-{r.id}") is not None)
    s = runner.summarize(rows, runner.Limits(1e9, 1e9), elapsed)
    p99 = err[min(len(err) - 1, int(0.99 * len(err)))] if err else None
    return {"n": s["n"], "ok": s["ok"], "output_tok_per_s": round(s["output_tok_per_s"]),
            "tpot_error_p50_ms": err and round(err[len(err) // 2] * 1e3, 3),
            "tpot_error_p99_ms": p99 is not None and round(p99 * 1e3, 3),
            "client_lag_p99_ms": s["client_lag_p99_s"] is not None and round(s["client_lag_p99_s"] * 1e3, 2),
            "client_cpu_max": s["client_cpu_max"],
            "clean": s["valid"] and p99 is not None and p99 <= max_error_s, "invalid_reasons": s["invalid_reasons"]}


def check(client_procs: int = 8, client_cpus=None, server_procs: int = 8, server_cpus=None, tpot_s: float = 0.02,
          tokens: int = 128, duration_s: float = 20.0, max_error_s: float = 0.001) -> dict:
    """Closed loop at each of CLOSED and open loop at each of OPEN; each level's result, and the highest clean one."""
    pool = [{"id": f"cc-{k}", "prompt": "x", "max_tokens": tokens, "expected_output_tokens": tokens} for k in range(64)]
    with tempfile.TemporaryDirectory() as d:
        logdir = Path(d)
        ps, url = _server(server_procs, tpot_s, server_cpus, logdir)
        try:
            closed = {}
            for c in CLOSED:
                rows, el = runner.run_closed(url, "m", W.Cycle(pool, f"c{c}"), c, duration_s, procs=client_procs,
                                             cpus=client_cpus)
                closed[c] = _judge(rows, logdir, el, max_error_s)
            opened = {}
            for rate in OPEN:
                reqs = W.poisson(pool, rate, duration_s, seed=0)
                rows = runner.run_open(url, "m", reqs, procs=client_procs, cpus=client_cpus)
                opened[rate] = _judge(rows, logdir, duration_s, max_error_s)
        finally:
            for p in ps:
                p.kill()
    best = lambda res: max((k for k, v in res.items() if v["clean"]), default=None)
    return {"closed": closed, "open": opened, "max_clean_concurrency": best(closed), "max_clean_rate": best(opened),
            "settings": {"client_procs": client_procs, "client_cpus": client_cpus, "server_procs": server_procs,
                         "server_cpus": server_cpus, "tpot_s": tpot_s, "tokens": tokens, "duration_s": duration_s,
                         "max_error_s": max_error_s, "cpu_count": os.cpu_count()}}


if __name__ == "__main__":
    from .__main__ import _cpus
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--client-procs", type=int, default=8)
    ap.add_argument("--client-cpus")
    ap.add_argument("--server-procs", type=int, default=8)
    ap.add_argument("--server-cpus")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = check(a.client_procs, _cpus(a.client_cpus), a.server_procs, _cpus(a.server_cpus), duration_s=a.duration)
    text = json.dumps(res, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
