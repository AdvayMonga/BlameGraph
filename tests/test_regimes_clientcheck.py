"""The client-capacity check against its own fixed-schedule server, at small levels. Run: python tests/test_regimes_clientcheck.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from regimes import clientcheck as C  # noqa: E402


def test_the_check_measures_the_clients_own_tpot_error():
    C.CLOSED, C.OPEN = (8,), (20.0,)
    res = C.check(client_procs=2, server_procs=2, tpot_s=0.01, tokens=16, duration_s=1.0)
    for level in (res["closed"][8], res["open"][20.0]):
        assert level["ok"] == level["n"] > 0 and level["clean"], level
        assert abs(level["tpot_error_p99_ms"]) < 1.0
    assert res["max_clean_concurrency"] == 8 and res["max_clean_rate"] == 20.0


if __name__ == "__main__":
    test_the_check_measures_the_clients_own_tpot_error(); print("ok")
