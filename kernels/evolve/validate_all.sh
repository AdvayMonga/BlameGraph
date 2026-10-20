#!/usr/bin/env bash
# The layer-1 validation session, on the GPU VM from this environment's dir (~/BlameGraph) after `lab.vm setup`:
#   scp SHIPPED.py VM:~/shipped.py        # outside both trees: lab.vm's push deletes what it does not know
#   ssh VM 'cd ~/BlameGraph && PATH=$HOME/inference-server/.venv/bin:$PATH setsid nohup \
#           bash kernels/evolve/validate_all.sh ~/shipped.py > ~/va.log 2>&1 < /dev/null &'   # then poll ~/va.log
# 0. what the jails need: GPU counters open, a CUDA C++ extension built and judged, ncu counters reaching the judge
# 1. every task's entry point captured under real corpus traffic at several concurrencies (CAPTURE_SPECS), CUDA graphs
#    off, no warmup traffic (random ids would skew the call counts the split weighs by), no profiler
# 2. a served profile (graphs on, as the engine runs) for the tasks' shares of time
# 3. per task: seen/held-out split (in parallel), then the engine's own kernel gated (it must pass, ~1.0x)
# 4. the gated tasks ranked by share x headroom (kernels.evolve share)
# 5. the shipped paged_decode kernel gated at the engine's block-table width
# Each step reuses only complete results. Then fetch the small files (rsync -a --include='*.json' --include='*.py'
# --include='*.log' --include='*/' --exclude='*' VM:~/BlameGraph/lab/runs/kernels-validate-all/ ./...) and run the
# search from the controller: python -m kernels.search --task T --shapes shapes-T.json --initial START.py --out DIR
# --device cuda --evals N --usd X, with KERNEL_EVAL_SSH / KERNEL_EVAL_REMOTE (LAB_TARGET, the engine venv on PATH)
# set; then gate its winner on the VM (python -m kernels.evolve gate ... --device cuda).
set -euo pipefail
SHIPPED="${1:?the shipped program to re-gate}"
TASKS="${TASKS:-paged_decode moe_gemm_fp8 moe_gemm_scatter_fp8 rmsnorm add_rmsnorm rmsnorm_rope silu_mul topk_softmax}"
OUT=lab/runs/kernels-validate-all
export LAB_TARGET="${LAB_TARGET:-targets/inference-server.toml}"
eval "$(python -c 'from lab import target; import shlex; print(" ".join(f"export {k}={shlex.quote(v)};" for k, v in target.load().engine.env.items()))')"
ENGINE="${LAB_ENGINE_REPO:-$HOME/inference-server}"
export PYTHONPATH="$PWD:$ENGINE/src${PYTHONPATH:+:$PYTHONPATH}"
# The engine JIT-builds its megakernel with the toolkit at CUDA_HOME: the one matching torch's CUDA (as the jails use).
export CUDA_HOME="${CUDA_HOME:-$(python -c 'from kernels.evolve.jailed import cuda_home; print(cuda_home() or "")')}"
echo "CUDA_HOME=$CUDA_HOME"
mkdir -p "$OUT"
done_json() { python -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if '$2' in d else 1)" "$1" 2>/dev/null; }
gated() {      # a gate result whose gates ran (a jail error or timeout has none, and is run again)
  python -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if (d.get('artifacts') or {}).get('gates') else 1)" "$1" 2>/dev/null; }

echo "== 0. jail facts"
grep -o 'RestrictProfilingToAdminUsers: [0-9]' /proc/driver/nvidia/params || true
LAB_NO_JAIL=0 python -m pytest -q -s tests/test_kernels_evolve.py -k "cuda_extension or ncu_counters" > "$OUT/step0.log" 2>&1 || true
grep -E "passed|failed|skip:|Error" "$OUT/step0.log" | tail -8

echo "== 1. capture under real traffic"
ENTRIES=$(python -c "from kernels.tasks import load; print(' '.join(sorted({'--capture ' + load(t).entry for t in '$TASKS'.split()})))")
for spec in ${CAPTURE_SPECS:-mixed_1:256 mixed_3:64 long_context:60}; do
  cls=${spec%%:*}; n=${spec##*:}
  ls "$OUT/profile-$cls"/*/captures/calls.jsonl >/dev/null 2>&1 && continue
  rm -rf "$OUT/profile-$cls"
  CUSTOM_BACKEND_CUDA_GRAPH=0 python -m lab.profile --corpus-class "$cls" --requests "$n" --max-tokens 128 --warmup 0 \
    --profiler cuda-range $ENTRIES --out "$OUT/profile-$cls" > "$OUT/profile-$cls.log" 2>&1 \
    || { tail -20 "$OUT/profile-$cls.log"; exit 1; }
  grep -o "megakernel[^)]*" "$OUT/profile-$cls.log" | sort -u | head -2 || true
done
CAPS=$(ls -d "$OUT"/profile-*/*/captures | sed 's/^/--captures /' | tr '\n' ' ')

echo "== 2. a served profile for the shares"
if ! ls "$OUT"/served/*/kernels.json >/dev/null 2>&1; then
  rm -rf "$OUT/served"
  python -m lab.profile --corpus-class mixed_1 --requests 64 --max-tokens 128 --out "$OUT/served" > "$OUT/served.log" 2>&1 \
    || { tail -20 "$OUT/served.log"; exit 1; }
fi

echo "== 3. per task: split (in parallel), the engine's own kernel gated"
for t in $TASKS; do
  done_json "$OUT/shapes-$t.json" seen || python -m kernels.evolve split --task "$t" $CAPS --write "$OUT/shapes-$t.json" \
    > "$OUT/split-$t.log" 2>&1 &
done
wait
GATES=""
for t in $TASKS; do
  python -c "import json; d=json.load(open('$OUT/shapes-$t.json')); print('$t', d.get('source'), 'seen', len(d['seen']), \
'heldout', len(d['heldout']), 'cut', d.get('cut'))"
  [ -s "$OUT/start-$t.py" ] || python -c "from kernels.evolve.run import initial_program; from kernels.tasks import load; \
open('$OUT/start-$t.py', 'w').write(initial_program(load('$t')))"
  gated "$OUT/gate-start-$t.json" || python -m kernels.evolve gate --task "$t" --shapes "$OUT/shapes-$t.json" \
    --program "$OUT/start-$t.py" --device cuda > "$OUT/gate-start-$t.json" 2> "$OUT/gate-start-$t.log" || true
  python - "$OUT/gate-start-$t.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:                           # noqa: BLE001
    print("  engine kernel: no gate result:", e)
    sys.exit(0)
g = json.loads((d.get("artifacts") or {}).get("gates") or "{}")
bad = [k for k, v in g.items() if not v["passed"]] if g else ["no gates ran: " + (d.get("artifacts") or {}).get("error", "")[-300:]]
print("  engine kernel:", d.get("metrics"), bad or "all gates pass")
PY
  gated "$OUT/gate-start-$t.json" && GATES="$GATES --gate $t=$OUT/gate-start-$t.json"
done

echo "== 4. the gated tasks ranked by share x headroom"
python -m kernels.evolve share $(ls -d "$OUT"/served/*/ | sed 's/^/--profile /') $GATES > "$OUT/share.json"
python -c "import json; d=json.load(open('$OUT/share.json')); [print(' ', r) for r in d['tasks'][:8]]"

echo "== 5. the shipped paged_decode kernel at the engine's width"
python -m kernels.evolve gate --task paged_decode --shapes "$OUT/shapes-paged_decode.json" --program "$SHIPPED" \
  --device cuda > "$OUT/gate-shipped.json" 2> "$OUT/gate-shipped.log" || true
python -c "import json; d=json.load(open('$OUT/gate-shipped.json')); print(d['metrics'])" || tail -5 "$OUT/gate-shipped.log"
echo "validate-all done"
