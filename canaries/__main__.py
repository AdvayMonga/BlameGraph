"""python -m canaries --upstream URL [--port 8100] --cheat NAME [--seed 0]"""
import argparse
import threading

from canaries.cheat_proxy import CHEATS, serve

ap = argparse.ArgumentParser(prog="canaries")
ap.add_argument("--upstream", required=True); ap.add_argument("--port", type=int, default=8100)
ap.add_argument("--cheat", required=True, choices=CHEATS); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
s = serve(a.upstream, a.port, a.cheat, a.seed)
print(f"cheat proxy '{a.cheat}' on http://127.0.0.1:{a.port} -> {a.upstream}")
threading.Event().wait()
