"""Cloud adapters for lab.infra.cloud.vm: each is a `Provider` over one cloud's API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Record:
    """What lab.infra.cloud.vm needs to know about a VM, whichever cloud it is on."""
    name: str
    state: str          # running | stopped | absent | anything else = transitional
    ip: str | None
    type: str
    detail: str = ""    # provider wording, for status output
    spot: bool | None = None   # preemptible; None where the provider has no such option


def pricing(spot: bool | None) -> str:
    return "spot" if spot else "on-demand"


class ProviderError(RuntimeError):
    """The provider refused; its message is here."""


def env_field(name: str, default: str):
    """A dataclass default read from env at construction, not import."""
    return field(default_factory=lambda: os.environ.get(name, default))


class Provider(Protocol):
    """One cloud. `name` is the VM's name there; a refusal raises ProviderError."""
    type: str            # the GPU machine type `create` asks for
    default_user: str    # the ssh login of a VM it created

    def get(self, name: str) -> Record | None: ...
    def create(self, name: str, keyfile: str, spot: bool, account: str | None = None) -> None: ...
    # spot: preemptible, at the spot price; account: a service account the VM acts as
    def start(self, name: str) -> None: ...
    def stop(self, name: str) -> None: ...      # the disk is kept
    def delete(self, name: str) -> None: ...    # the VM and its disk
    def types(self) -> str: ...
    def account(self) -> str: ...              # the service account a VM stops itself as
    def allow_stop(self, name: str) -> None: ...     # that account may stop VM `name`
    def instance_id(self, name: str) -> str: ...
    def stop_id(self, instance_id: str) -> None: ...     # run on the VM, as the account


def load(name: str) -> Provider:
    if name == "nebius":
        from lab.infra.cloud.providers import nebius
        return nebius.Nebius()
    raise ProviderError(f"unknown provider {name!r}; LAB_VM_PROVIDER is nebius")
