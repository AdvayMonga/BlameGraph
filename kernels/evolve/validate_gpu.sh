#!/usr/bin/env bash
# Layer-1 GPU validation, run on the VM from this environment's dir with the engine's venv on PATH:
#   python -m lab.vm run --env --keep -- bash kernels/evolve/validate_gpu.sh [ITERATIONS] [TASK]   (TASK: paged_decode)
# 1. capture the engine's calls to the task's entry point under two corpus classes (real shapes, real inputs), CUDA graphs off:
#    replayed graphs never reach Python, so with them on most calls would be invisible
# 2. split them into seen / held-out regions
# 3. judge the engine's own kernel against the reference on both (correctness on the GPU, the baseline timing)
# 4. a short OpenEvolve run on the seen shapes (model calls: ANTHROPIC_API_KEY must be set; 0 iterations skips it)
# 5. the held-out gate on the best program
# Everything lands in lab/runs/kernels-validate/ for `--fetch`.
set -euo pipefail
ITER="${1:-10}"
TASK="${2:-paged_decode}"
OUT=lab/runs/kernels-validate$([ "$TASK" = paged_decode ] || echo "-$TASK")
ENTRY=$(python -c "from kernels.tasks import load; print(load('$TASK').entry)")
export LAB_TARGET="${LAB_TARGET:-targets/inference-server.toml}"
mkdir -p "$OUT"
# The engine's serving env (BACKEND, MODEL_NAME, ...) from the target spec, as `serve` would set it.
eval "$(python -c 'from lab import target; import shlex; print(" ".join(f"export {k}={shlex.quote(v)};" for k, v in target.load().engine.env.items()))')"
ENGINE="${LAB_ENGINE_REPO:-$HOME/inference-server}"
export PYTHONPATH="$PWD:$ENGINE/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import openevolve" 2>/dev/null || "$HOME/.local/bin/uv" pip install -q --python "$(command -v python)" "openevolve==0.4.0"

echo "== 1. capture"
for spec in steady_interactive:64 long_context:24; do     # long_context's seen split holds ~30 requests
  cls=${spec%%:*}; n=${spec##*:}
  ls -d "$OUT/profile-$cls"/*/captures >/dev/null 2>&1 && continue      # already captured
  CUSTOM_BACKEND_CUDA_GRAPH=0 \
  python -m lab.profile --corpus-class "$cls" --requests "$n" --max-tokens 64 --capture "$ENTRY" --out "$OUT/profile-$cls" \
    > "$OUT/profile-$cls.log" 2>&1 || { tail -20 "$OUT/profile-$cls.log"; exit 1; }
done
CAPS=$(ls -d "$OUT"/profile-*/*/captures | sed 's/^/--captures /' | tr '\n' ' ')

echo "== 2. split"
python -m kernels.evolve split --task $TASK $CAPS --write "$OUT/shapes.json"
python -c "import json; d=json.load(open('$OUT/shapes.json')); print('seen', len(d['seen']), 'heldout', len(d['heldout']), 'cut', d.get('cut'))"
python - <<EOF
import json, glob
calls = [json.loads(l) for f in glob.glob("$OUT/profile-*/*/captures/calls.jsonl") for l in open(f)]
print("calls", len(calls), "all contiguous:", all(a.get("contiguous", True) for c in calls for a in c["args"] if "shape" in a))
EOF

echo "== 3. the engine's kernel on the GPU"
python - <<EOF
from kernels.evolve.run import initial_program
from kernels.tasks import load
open("$OUT/engine_kernel.py", "w").write(initial_program(load("$TASK")))
EOF
python -m kernels.evolve gate --task $TASK --shapes "$OUT/shapes.json" --program "$OUT/engine_kernel.py" \
  > "$OUT/gate_engine.json" || true
python -c "import json; d=json.load(open('$OUT/gate_engine.json')); print(d['metrics']); print(d['artifacts'].get('gates','')[:3000])"

if [ "$ITER" -gt 0 ]; then
  echo "== 4. OpenEvolve, $ITER iterations"
  python -m kernels.evolve run --task $TASK --shapes "$OUT/shapes.json" --iterations "$ITER" \
    --initial "$OUT/engine_kernel.py" --out "$OUT/evolve" > "$OUT/evolve.log" 2>&1 || tail -30 "$OUT/evolve.log"
  echo "== 5. held-out gate on the best program"
  python -m kernels.evolve gate --task $TASK --shapes "$OUT/shapes.json" --program "$OUT/evolve/best_program.py" \
    > "$OUT/gate_best.json" || true
  python -c "import json; d=json.load(open('$OUT/gate_best.json')); print(d['metrics'])"
fi
echo "validate done"
