#!/usr/bin/env bash
# The engine comparison's extra setup on a GPU VM that lab/vm-setup.sh has bootstrapped (engine venv, docker, NVIDIA
# Container Toolkit): the three engines' images and both checkpoints, pulled once. Idempotent. From the laptop:
#   LAB_VM_PROVIDER=nebius LAB_TARGET=targets/inference-server.toml python -m lab.vm start --minutes 720
#   python -m lab.vm setup                                  # engine venv + docker (lab/vm-setup.sh)
#   rsync data/equiv/ref to ~/BlameGraph/data/equiv/ref       # equiv's reference outputs (gitignored)
#   ssh: bash ~/BlameGraph/compare/vm-setup.sh
#   ssh: cd ~/BlameGraph && ulimit -n 65536 && setsid nohup ~/inference-server/.venv/bin/python -m compare.run \
#        --out ~/compare > ~/compare.log 2>&1 < /dev/null &
#   fetch ~/compare, then: python -m compare.report DIR --gpu-usd-per-hour <price>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
DOCKER=$(docker info >/dev/null 2>&1 && echo docker || echo "$SUDO docker")

echo "== images"
images=$(python3 -c 'import sys, tomllib; e = tomllib.load(open(sys.argv[1], "rb"))["engines"]
print(" ".join(v["image"] for v in e.values() if v.get("image")))' "$here/engines.toml")
for img in $images; do $DOCKER pull -q "$img" & done
wait

echo "== checkpoints"
export PATH="$HOME/.local/bin:$PATH"
checkpoints=$(python3 -c 'import sys, tomllib; d = tomllib.load(open(sys.argv[1], "rb"))
print(" ".join(sorted({p["checkpoint"] for p in d["precision"].values()})))' "$here/engines.toml")
for ck in $checkpoints; do
  HF_HUB_ENABLE_HF_TRANSFER=1 uvx --from 'huggingface_hub[cli,hf_transfer]' hf download "$ck" >/dev/null
done

echo "== reference outputs for equiv"
[ -f "$here/../data/equiv/ref/outputs.jsonl" ] && echo ok || { echo "missing data/equiv/ref: rsync it from the laptop"; exit 1; }

echo "== facts"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader
for img in $images; do echo "$img $($DOCKER image inspect --format '{{index .RepoDigests 0}}' "$img")"; done
echo "compare setup done"
