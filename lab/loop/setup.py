"""What a run needs on this machine before anything is started or billed, each missing thing with its fix."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from lab.infra.cloud import vm
from lab.infra.cloud.providers import ProviderError
from lab.targets import engine


@dataclass(frozen=True)
class Problem:
    what: str
    fix: str


def setup(v: vm.VM, env=os.environ) -> tuple[Problem, ...]:
    """The problems found: the target, the model key, the GitHub token, the store, the cloud login, the VM's account.
    The account is asked for only when the login works."""
    from lab.records.remote import NoRemote, open_remote
    out = []
    spec = env.get("LAB_TARGET")
    if not spec or not Path(spec).resolve().is_relative_to(engine.ENV_ROOT):
        out.append(Problem("LAB_TARGET names no target spec inside this repo",
                           "set LAB_TARGET=evals/targets/<name>.toml in .env"))
    if not env.get("ANTHROPIC_API_KEY"):
        out.append(Problem("no model key", "set ANTHROPIC_API_KEY in .env"))
    if not env.get("GH_TOKEN"):
        out.append(Problem("no GitHub token: a win could not be pushed",
                           "set GH_TOKEN in .env to a token that may push branches and open pull requests on the "
                           "engine repo"))
    try:
        open_remote(env.get("LAB_STORE"))
    except NoRemote as e:
        out.append(Problem(f"no store: {e}", "set LAB_STORE (and its credentials) in .env"))
    try:
        v.provider.get(v.name)
    except ProviderError as e:
        return (*out, Problem(f"the cloud cannot be asked: {e}", "install the provider's CLI and log in"))
    try:
        v.provider.account()
    except ProviderError as e:
        out.append(Problem(f"no service account for the VM to stop itself as: {e}",
                           "log in as a user who may create service accounts, groups and access permits"))
    return tuple(out)
