#!/usr/bin/env bash
# One-time bootstrap of a fresh GPU VM for the lab. Idempotent; run via `python -m lab.vm setup`.
# Arg: the engine tree on the VM (default ~/inference-server).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)           # this environment's lab/
cd "${1:-$HOME/inference-server}"            # the engine tree: its venv is the one every GPU command runs with
SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)   # Verda images log in as root

echo "== GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== uv + venv"
# Triton builds its launcher against Python.h at first kernel launch.
$SUDO apt-get update -qq >/dev/null   # a fresh image has no package lists
$SUDO apt-get install -y -qq python3-dev >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# pyproject pins torch to the CPU index on linux (CI). Here we want the CUDA wheel at the SAME
# locked version, so the rest of the lock still matches. Skipped when the venv already has it;
# never run `uv sync` on this box afterwards, it would put CPU torch back. Proper fix: a cuda
# extra with its own index in pyproject (tracked as a follow-up).
if ! .venv/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  uv sync --locked --extra dev
  torch_version=$(awk '/^name = "torch"$/{getline; sub(/version = "/,""); sub(/"/,""); print; exit}' uv.lock)
  uv pip install --reinstall "torch==${torch_version}" --index-url https://download.pytorch.org/whl/cu130
fi

echo "== CUDA toolkit"
# The image's nvcc decides what JIT-compiling libraries can do: vLLM's DeepGEMM FP8 path needs nvcc >= 12.9.
nvcc --version 2>/dev/null | tail -1 || ls /usr/local/cuda/bin/nvcc 2>/dev/null || echo "nvcc: not on PATH"
# A C++ extension (the kernel loop's CUDA candidates, the engine's megakernel) loads only when nvcc matches the CUDA
# torch was built with: install that toolkit beside the image's (Verda's is 12.8 beside torch cu130).
tv=$(.venv/bin/python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || true)
if [ -n "$tv" ] && [ ! -x "/usr/local/cuda-$tv/bin/nvcc" ]; then
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "cuda-toolkit-${tv/./-}" >/dev/null \
    || echo "cuda-toolkit-$tv: not in apt (finding, not failure)"
fi
[ -n "$tv" ] && { "/usr/local/cuda-$tv/bin/nvcc" --version 2>/dev/null | tail -1 || true; }

echo "== Nsight"
# CUDA images keep ncu in /usr/local/cuda/bin, on no login PATH, and that one is only a launcher for the install under
# /opt/nvidia/nsight-compute/<version>: link the real one, so the tool's jail mounts its whole install root.
ncu_real=$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)
if [ -n "$ncu_real" ]; then $SUDO ln -sf "$ncu_real" /usr/local/bin/ncu
elif ! command -v ncu >/dev/null && [ -x /usr/local/cuda/bin/ncu ]; then $SUDO ln -sf /usr/local/cuda/bin/ncu /usr/local/bin/ncu; fi
if ! command -v nsys >/dev/null || ! command -v ncu >/dev/null; then
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq nsight-systems nsight-compute 2>/dev/null \
    || echo "Nsight not in apt; install from the CUDA toolkit repo (finding, not failure)"
fi
nsys --version 2>/dev/null || true
ncu --version 2>/dev/null | head -1 || true

echo "== py-spy (hostprof) and srt, the agent CLI's jail until lab/RUNTIME.md's split (Node >= 20, bubblewrap, socat, ripgrep)"
uv pip install -q --python .venv/bin/python py-spy && .venv/bin/py-spy --version
uv pip install -q --python .venv/bin/python duckdb zstandard    # the agent's query tool (store/index.py)
if ! command -v srt >/dev/null; then
  $SUDO apt-get install -y -qq bubblewrap socat ripgrep xz-utils >/dev/null
  if ! node -e 'process.exit(+process.versions.node.split(".")[0] < 20)' 2>/dev/null; then
    curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz | $SUDO tar -xJ -C /usr/local --strip-components=1
  fi
  $SUDO npm i -g -s @anthropic-ai/sandbox-runtime >/dev/null
fi
# Ubuntu 24.04 blocks unprivileged user namespaces, so bwrap cannot start; this VM is single-purpose.
$SUDO sysctl -qw kernel.apparmor_restrict_unprivileged_userns=0 2>/dev/null || true
srt -c true >/dev/null 2>&1 && echo "srt: ok" || echo "srt: cannot start"

echo "== jail: docker + NVIDIA Container Toolkit + the lab-jail image (lab/safety/container.py)"
command -v docker >/dev/null || $SUDO apt-get install -y -qq docker.io socat >/dev/null
command -v socat >/dev/null || $SUDO apt-get install -y -qq socat >/dev/null
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq nvidia-container-toolkit >/dev/null
  $SUDO nvidia-ctk runtime configure --runtime=docker >/dev/null && $SUDO systemctl restart docker
fi
[ "$(id -u)" = 0 ] || $SUDO usermod -aG docker "$USER"   # takes effect at the next login
$SUDO docker build -q -t lab-jail -f "$here/safety/jail.Dockerfile" "$here/safety" >/dev/null
$SUDO docker run --rm --gpus all --network none lab-jail nvidia-smi -L && echo "jail: gpu ok" || echo "jail: no GPU in the container"
# The agent's workbench has the internet but not the cloud's metadata service (instance credentials, user-data).
# Only that address: a cloud's DNS resolver can sit in 169.254.0.0/16 too, and the workbench needs names.
$SUDO iptables -C DOCKER-USER -d 169.254.169.254 -j DROP 2>/dev/null || $SUDO iptables -I DOCKER-USER -d 169.254.169.254 -j DROP
$SUDO docker run --rm lab-jail python3 -c "import socket; socket.create_connection(('169.254.169.254', 80), timeout=3)" \
  2>/dev/null && echo "workbench: METADATA REACHABLE" || echo "workbench: metadata blocked"
$SUDO docker run --rm lab-jail python3 -c "import socket; socket.create_connection(('pypi.org', 443), timeout=10)" \
  2>/dev/null && echo "workbench: internet ok (DNS and egress)" || echo "workbench: NO INTERNET (DNS or egress)"

echo "== counters: can a non-admin process read GPU performance counters?"
# ncu needs NVreg_RestrictProfilingToAdminUsers=0 (or root). Recorded as a fact about this venue.
if [ -r /proc/driver/nvidia/params ]; then
  grep -o 'RestrictProfilingToAdminUsers: [0-9]' /proc/driver/nvidia/params || echo "RestrictProfilingToAdminUsers: unknown"
fi

echo "== clocks: can we lock them?"
$SUDO nvidia-smi -pm 1 >/dev/null 2>&1 && echo "persistence mode: on" || echo "persistence mode: refused"
max_sm=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits -i 0 2>/dev/null || echo "")
if [ -n "$max_sm" ] && $SUDO nvidia-smi -lgc "$max_sm" >/dev/null 2>&1; then
  echo "clock lock: ok (sm ${max_sm} MHz)"
  $SUDO nvidia-smi -rgc >/dev/null 2>&1 || true
else
  echo "clock lock: refused"
fi

echo "== peaks: HBM bandwidth and bf16 GEMM, measured (validity/peaks.py; the ceilings' denominators)"
ENV_DIR="${LAB_VM_ENV_DIR:-$HOME/BlameGraph}"; ENGINE_DIR="$PWD"
(cd "$ENV_DIR" && "$ENGINE_DIR/.venv/bin/python" -m validity peaks --seconds 3) || echo "peaks: not measured (finding, not failure)"

echo "== model cache: the target's model and the Hub repos its tests read (every jail is offline)"
mkdir -p "$HOME/.cache/huggingface"
.venv/bin/python - "$here/../${LAB_TARGET:-targets/inference-server.toml}" <<'PY' \
  || echo "model cache: download failed (finding, not failure)"
import sys, tomllib
from huggingface_hub import snapshot_download
t = tomllib.load(open(sys.argv[1], "rb"))
print("cached", snapshot_download(t["model"]["name"]))
for repo in t["engine"].get("hub", []):              # what tests read is tokenizers and configs, not weights
    print("cached", snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken"]))
PY
chmod -R a+rX "$HOME/.cache/huggingface"      # the jails' own user reads it (hub metadata is written 0600)
du -sh "$HOME/.cache/huggingface" 2>/dev/null || true

echo "== smoke"
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "setup done"
