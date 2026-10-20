#!/usr/bin/env bash
# The layer-1 validation session, run on the GPU VM from this environment's dir after `lab.vm setup`:
#   bash kernels/evolve/validate_all.sh SHIPPED_PROGRAM.py
# 0. what the jails need: GPU counters open, a CUDA C++ extension built and judged, ncu counters reaching the judge
# 1. every task's entry point captured under real corpus traffic at several concurrencies (CAPTURE_SPECS), graphs off
# 2. a served profile (graphs on) for the tasks' shares of time
# 3. per task: seen/held-out split, then the engine's own kernel gated (it must pass; it scores ~1.0 by definition)
# 4. tasks ranked by share x headroom (kernels.evolve share)
# 5. the shipped paged_decode kernel gated at the engine's block-table width
# Each step skips what it already has. Everything lands in lab/runs/kernels-validate-all/ for `--fetch`. The long
# search runs afterwards from the controller (kernels.search over SSH), on the task step 4 ranks first.
set -euo pipefail
SHIPPED="${1:?the shipped program to re-gate}"
TASKS="${TASKS:-paged_decode moe_gemm_fp8 moe_gemm_scatter_fp8 rmsnorm add_rmsnorm rmsnorm_rope silu_mul topk_softmax}"
OUT=lab/runs/kernels-validate-all
export LAB_TARGET="${LAB_TARGET:-targets/inference-server.toml}"
eval "$(python -c 'from lab import target; import shlex; print(" ".join(f"export {k}={shlex.quote(v)};" for k, v in target.load().engine.env.items()))')"
ENGINE="${LAB_ENGINE_REPO:-$HOME/inference-server}"
export PYTHONPATH="$PWD:$ENGINE/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

echo "== 0. jail facts"
grep -o 'RestrictProfilingToAdminUsers: [0-9]' /proc/driver/nvidia/params || true
python -m pytest -q -s tests/test_kernels_evolve.py -k "cuda_extension or ncu_counters" 2>&1 | tail -5

echo "== 1. capture under real traffic"
ENTRIES=$(python -c "from kernels.tasks import load; print(' '.join(sorted({'--capture ' + load(t).entry for t in '$TASKS'.split()})))")
for spec in ${CAPTURE_SPECS:-mixed_1:256 mixed_3:64 long_context:60}; do
  cls=${spec%%:*}; n=${spec##*:}
  ls -d "$OUT/profile-$cls"/*/captures >/dev/null 2>&1 && continue
  CUSTOM_BACKEND_CUDA_GRAPH=0 python -m lab.profile --corpus-class "$cls" --requests "$n" --max-tokens 128 $ENTRIES \
    --out "$OUT/profile-$cls" > "$OUT/profile-$cls.log" 2>&1 || { tail -20 "$OUT/profile-$cls.log"; exit 1; }
done
CAPS=$(ls -d "$OUT"/profile-*/*/captures | sed 's/^/--captures /' | tr '\n' ' ')

echo "== 2. a served profile for the shares"
ls -d "$OUT/served"/*/ >/dev/null 2>&1 || python -m lab.profile --corpus-class mixed_1 --requests 64 --max-tokens 128 \
  --out "$OUT/served" > "$OUT/served.log" 2>&1 || { tail -20 "$OUT/served.log"; exit 1; }

echo "== 3. per task: split, the engine's own kernel gated"
GATES=""
for t in $TASKS; do
  [ -s "$OUT/shapes-$t.json" ] || python -m kernels.evolve split --task "$t" $CAPS --write "$OUT/shapes-$t.json" >/dev/null
  python -c "import json; d=json.load(open('$OUT/shapes-$t.json')); print('$t', 'seen', len(d['seen']), 'heldout', len(d['heldout']), 'cut', d.get('cut'))"
  [ -s "$OUT/start-$t.py" ] || python -c "from kernels.evolve.run import initial_program; from kernels.tasks import load; \
open('$OUT/start-$t.py', 'w').write(initial_program(load('$t')))"
  [ -s "$OUT/gate-start-$t.json" ] || python -m kernels.evolve gate --task "$t" --shapes "$OUT/shapes-$t.json" \
    --program "$OUT/start-$t.py" --device cuda > "$OUT/gate-start-$t.json" || true
  python -c "import json; d=json.load(open('$OUT/gate-start-$t.json')); g=json.loads(d['artifacts'].get('gates') or '{}'); \
print('  engine kernel:', d['metrics'], [k for k, v in g.items() if not v['passed']] or 'all gates pass')"
  GATES="$GATES --gate $t=$OUT/gate-start-$t.json"
done

echo "== 4. tasks ranked by share x headroom"
python -m kernels.evolve share $(ls -d "$OUT"/served/*/ | sed 's/^/--profile /') $GATES > "$OUT/share.json"
python -c "import json; d=json.load(open('$OUT/share.json')); [print(' ', r) for r in d['tasks'][:8]]"

echo "== 5. the shipped paged_decode kernel at the engine's width"
python -m kernels.evolve gate --task paged_decode --shapes "$OUT/shapes-paged_decode.json" --program "$SHIPPED" \
  --device cuda > "$OUT/gate-shipped.json" || true
python -c "import json; d=json.load(open('$OUT/gate-shipped.json')); print(d['metrics'])"
echo "validate-all done"
