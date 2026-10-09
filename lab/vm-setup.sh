#!/usr/bin/env bash
# One-time bootstrap of a fresh GPU VM for the lab. Idempotent; run via `python -m lab.vm setup`.
# Arg: the engine tree on the VM (default ~/inference-server).
set -euo pipefail
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

echo "== Nsight"
if ! command -v nsys >/dev/null || ! command -v ncu >/dev/null; then
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq nsight-systems nsight-compute 2>/dev/null \
    || echo "Nsight not in apt; install from the CUDA toolkit repo (finding, not failure)"
fi
nsys --version 2>/dev/null || true
ncu --version 2>/dev/null | head -1 || true

echo "== py-spy (hostprof) and the jail (srt needs Node >= 20, bubblewrap, socat, ripgrep)"
uv pip install -q --python .venv/bin/python py-spy && .venv/bin/py-spy --version
if ! command -v srt >/dev/null; then
  $SUDO apt-get install -y -qq bubblewrap socat ripgrep xz-utils >/dev/null
  if ! node -e 'process.exit(+process.versions.node.split(".")[0] < 20)' 2>/dev/null; then
    curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz | $SUDO tar -xJ -C /usr/local --strip-components=1
  fi
  $SUDO npm i -g -s @anthropic-ai/sandbox-runtime >/dev/null
fi
# Ubuntu 24.04 blocks unprivileged user namespaces, so bwrap cannot start; this VM is single-purpose.
$SUDO sysctl -qw kernel.apparmor_restrict_unprivileged_userns=0 2>/dev/null || true
srt -c true >/dev/null 2>&1 && echo "jail: ok" || echo "jail: srt cannot start"

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

echo "== model cache"
mkdir -p "$HOME/.cache/huggingface"
du -sh "$HOME/.cache/huggingface" 2>/dev/null || true

echo "== smoke"
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "setup done"
