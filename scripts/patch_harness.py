"""Patch the InferenceBench fork's run_task.sh so every session is recorded at the source and validated before scoring.
Idempotent. Usage: patch_harness.py [.]   (this repo)

Changes:
 1. task-side evaluate.py becomes a thin wrapper: BlameGraph records the measurement, then runs the pristine grader.
 2. the read-only eval bundle (/opt/inference_eval) ships blamegraph/tool as blamegraph_tool/ and grader_entry.py.
 3. a session-start event is written when the task dir is prepared.
 4. after the agent finishes and before final eval: submission is recorded, validated, and
    blamegraph_ledger.jsonl + blamegraph_validation.json are copied to EVAL_DIR.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WRAPPER = r'''    cat > "${JOB_DIR}/task/evaluate.py" <<'PY'
#!/usr/bin/env python3
"""evaluate.py: runs the official benchmark. BlameGraph records each measurement to .blamegraph/ledger.jsonl
(config hash, live server state, flags, metrics) before handing off to the pristine grader."""
import os
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, "/opt")
os.environ.setdefault("BLAMEGRAPH_TASK_DIR", str(TASK_DIR))
from inference_eval.blamegraph_tool.cli import main as blamegraph_main  # noqa: E402

if __name__ == "__main__":
    blamegraph_main(["measure", "--task-dir", str(TASK_DIR), "--grader", "/opt/inference_eval/grader_entry.py", "--", *sys.argv[1:]])
PY
    # record the session start and the pristine wrapper hash (outside the agent's reach)
    mkdir -p "${JOB_DIR}/task/.blamegraph"
    BLAMEGRAPH_GRADER_HASH="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()[:16])' "${JOB_DIR}/task/evaluate.py")"
    export BLAMEGRAPH_GRADER_HASH
    python3 -m blamegraph.tool.cli session-start --task-dir "${JOB_DIR}/task" --scenario "${EVALUATION_TASK}" || true
fi
'''

BUNDLE = '''chmod +x "${INFERENCE_EVAL_BUNDLE}/bin/launch_supervised_server.sh"
# BlameGraph: measurement ledger + validator, read-only inside the container
cp -r blamegraph/tool "${INFERENCE_EVAL_BUNDLE}/blamegraph_tool"
cp src/eval/inference/grader_entry.py "${INFERENCE_EVAL_BUNDLE}/grader_entry.py"
'''

SUBMIT = '''capture_agent_runtime_for_final_eval
# BlameGraph: record the submission and validate the session before anything is scored
if [[ "${EVALUATION_TASK}" == inference_scenario_* ]]; then
    python3 -m blamegraph.tool.cli submit --task-dir "${JOB_DIR}/task" --pristine-grader-hash "${BLAMEGRAPH_GRADER_HASH:-}" \\
        > "${EVAL_DIR}/blamegraph_validation.json" 2>> "${EVAL_LOG}" || echo "[blamegraph] submission INVALID (see blamegraph_validation.json)" | tee -a "${EVAL_LOG}"
    copy_if_exists "${JOB_DIR}/task/.blamegraph/ledger.jsonl" "${EVAL_DIR}/blamegraph_ledger.jsonl"
fi
'''


def main(repo: str = "."):
    p = (ROOT / repo).resolve() / "src" / "run_task.sh"
    s = p.read_text()
    if "blamegraph_tool" in s:
        print("already patched:", p); return
    # 1 + 3: replace the task-side evaluate.py heredoc (keep its enclosing `if ... fi`)
    start = s.index('    cat > "${JOB_DIR}/task/evaluate.py" <<\'PY\'')
    end = s.index("PY\nfi\n", start) + len("PY\nfi\n")
    s = s[:start] + WRAPPER + s[end:]
    # 2: bundle
    s = s.replace('chmod +x "${INFERENCE_EVAL_BUNDLE}/bin/launch_supervised_server.sh"\n', BUNDLE, 1)
    # 4: submission + validation before final eval (the second occurrence is the call site, after the function definition)
    k = s.rfind("capture_agent_runtime_for_final_eval\n")
    s = s[:k] + SUBMIT + s[k + len("capture_agent_runtime_for_final_eval\n"):]
    p.write_text(s)
    print("patched", p)


if __name__ == "__main__":
    main(*sys.argv[1:])
