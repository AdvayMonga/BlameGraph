"""Nebius AI Cloud through the `nebius` CLI (auth: `nebius profile create`)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lab.infra.cloud.providers import ProviderError, Record, env_field, pricing

USER = "lab"                        # the login cloud-init creates
IMAGE = "ubuntu24.04-cuda13.0"      # an image family
DISK_GB = 400
CLI_DIR = Path.home() / ".nebius" / "bin"      # where the CLI's installer puts it; searched after PATH


@dataclass(frozen=True)
class Nebius:
    type: str = env_field("LAB_VM_TYPE", "gpu-h200-sxm/1gpu-16vcpu-200gb")   # platform/preset, as `types` lists them
    default_user: str = USER

    def _cli(self, *args: str) -> str:
        exe = shutil.which("nebius") or shutil.which("nebius", path=str(CLI_DIR))
        if exe is None:
            raise ProviderError("`nebius` CLI not installed: https://docs.nebius.com/cli/install")
        out = subprocess.run([exe, *args], capture_output=True, text=True)
        if out.returncode != 0:
            raise ProviderError(f"nebius {' '.join(args[:4])}: {out.stderr.strip() or out.stdout.strip()}")
        return out.stdout

    def _json(self, *args: str):
        text = self._cli(*args, "--format", "json")
        try:
            return json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            raise ProviderError(f"nebius {' '.join(args[:4])}: not JSON: {text[:200]}")

    def _raw(self, name: str) -> dict | None:
        items = self._json("compute", "instance", "list").get("items") or []
        return next((i for i in items if (i.get("metadata") or {}).get("name") == name), None)

    def get(self, name: str) -> Record | None:
        raw = self._raw(name)
        if raw is None:
            return None
        status = raw.get("status") or {}
        state = str(status.get("state", "")).lower()
        ip = None
        for nic in status.get("network_interfaces") or []:
            addr = (nic.get("public_ip_address") or {}).get("address")
            if addr:
                ip = addr.split("/")[0]
                break
        res = (raw.get("spec") or {}).get("resources") or {}
        kind = f"{res.get('platform', '')}/{res.get('preset', '')}".strip("/")
        spot = bool((raw.get("spec") or {}).get("preemptible"))
        return Record(name=name, state=state, ip=ip, type=kind, detail=f"{state} {pricing(spot)}", spot=spot)

    def _subnet(self) -> str:
        """The first subnet of the CLI profile's project."""
        items = self._json("vpc", "subnet", "list").get("items") or []
        if not items:
            raise ProviderError("no VPC subnet in this project")
        return items[0]["metadata"]["id"]

    def create(self, name: str, keyfile: str, spot: bool) -> None:
        platform, slash, preset = self.type.partition("/")
        if not (platform and slash and preset):
            raise ProviderError(f"LAB_VM_TYPE={self.type!r}: PLATFORM/PRESET, as `lab vm types` lists them")
        try:
            public = Path(os.path.expanduser(keyfile)).read_text().strip()
        except OSError as e:
            raise ProviderError(f"ssh public key {keyfile}: {e.strerror}; set LAB_VM_KEYFILE")
        user_data = ("#cloud-config\nusers:\n"
                     f"  - name: {USER}\n    sudo: ALL=(ALL) NOPASSWD:ALL\n    shell: /bin/bash\n"
                     f"    ssh_authorized_keys:\n      - {public}\n")
        nics = json.dumps([{"name": "eth0", "ip_address": {}, "public_ip_address": {}, "subnet_id": self._subnet()}])
        self._cli("compute", "instance", "create", "--name", name,
                  "--resources-platform", platform, "--resources-preset", preset,
                  "--boot-disk-managed-disk-name", f"{name}-boot",
                  "--boot-disk-managed-disk-type", "network_ssd",
                  "--boot-disk-managed-disk-size-gibibytes", str(DISK_GB),
                  "--boot-disk-managed-disk-source-image-family-image-family", IMAGE,
                  "--boot-disk-attach-mode", "READ_WRITE",
                  "--network-interfaces", nics, "--cloud-init-user-data", user_data,
                  *(("--follows-spot-price", "--preemptible-on-preemption", "stop", "--recovery-policy", "fail")
                    if spot else ()))    # spot: stop on preemption, no recovery

    def _id(self, name: str) -> str:
        raw = self._raw(name)
        if raw is None:
            raise ProviderError(f"{name}: no instance")
        return raw["metadata"]["id"]

    def start(self, name: str) -> None:
        self._cli("compute", "instance", "start", self._id(name))

    def stop(self, name: str) -> None:
        self._cli("compute", "instance", "stop", self._id(name))

    def delete(self, name: str) -> None:
        """Delete the instance and its managed boot disk."""
        self._cli("compute", "instance", "delete", self._id(name))

    def types(self) -> str:
        items = self._json("compute", "platform", "list").get("items") or []
        lines = []
        for p in items:
            presets = [x.get("name") for x in (p.get("spec") or {}).get("presets") or []]
            lines.append(f"{(p.get('metadata') or {}).get('name')}: {', '.join(filter(None, presets))}")
        return "\n".join(lines) + "\n"
