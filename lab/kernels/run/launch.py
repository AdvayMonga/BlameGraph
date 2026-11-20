"""The laptop's part of a VM run: the VM up with the lab's service account, set up, the job started there detached."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from lab.infra.cloud import vm
from lab.targets import engine

SECRETS = ("ANTHROPIC_API_KEY", "LAB_STORE", "LAB_STORE_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
LAUNCH_TIMEOUT_S = 120


def missing() -> list[str]:
    """What this host needs before anything is started."""
    out = [f"{k} is not set" for k in ("ANTHROPIC_API_KEY", "LAB_STORE") if not os.environ.get(k)]
    spec = Path(os.environ.get("LAB_TARGET", "")).resolve()
    if not os.environ.get("LAB_TARGET") or not spec.is_relative_to(engine.ENV_ROOT):
        out.append("LAB_TARGET must name a target spec inside this repo")
    return out


def env() -> dict:
    """What the job gets on stdin: the secrets, the target and the provider."""
    spec = Path(os.environ["LAB_TARGET"]).resolve().relative_to(engine.ENV_ROOT).as_posix()
    return {**{k: os.environ[k] for k in SECRETS if os.environ.get(k)}, "LAB_TARGET": spec,
            "LAB_VM_PROVIDER": os.environ.get("LAB_VM_PROVIDER", "nebius")}


def go(v: vm.VM, run: str, task: str) -> None:
    """The VM running and set up (created with the service account, or a stopped one started) and the job started on
    it; a VM this call created or started is deleted when a step fails."""
    p = v.provider
    try:
        record = vm.start(v, account=p.account())
        vm.wait_ssh(v, record)
        if vm.left_s(v) < 60:
            vm.arm(v, record, vm.MAX_MINUTES)
        p.allow_delete(v.name)
        if vm.setup(v):
            raise RuntimeError(f"`lab vm setup` failed on {v.name}")
        record = v.provider.get(v.name)
        deadline = time.time() + vm.left_s(v)
        venv_bin = v.remote_dir.replace("~", "$HOME", 1) + "/.venv/bin"
        cmd = (f'cd {v.env_dir} && PATH="{venv_bin}:$PATH" python -m lab.kernels.run.job --run {shlex.quote(run)} '
               f'--task {shlex.quote(task)} --vm {shlex.quote(v.name)} --instance {p.instance_id(v.name)} '
               f'--deadline {deadline:.0f}')
        out = subprocess.run(["ssh", *v.ssh_opts, vm.ssh_target(v, record), cmd], input=json.dumps(env()), text=True,
                             capture_output=True, timeout=LAUNCH_TIMEOUT_S)
        if out.returncode != 0 or f"started {run}" not in out.stdout:
            raise RuntimeError(f"the job did not start on {v.name}: {(out.stderr or out.stdout).strip()[-2000:]}")
    except BaseException:
        vm.stop(v)
        raise
