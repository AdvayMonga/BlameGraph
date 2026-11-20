"""pytest setup for the lab's tests; tests that drive engine code skip when it can't be imported."""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import pytest

# Every lab test runs against the example target unless the caller chose one.
os.environ.setdefault("LAB_TARGET", str(pathlib.Path(__file__).resolve().parents[1] / "evals" / "targets" / "inference-server.toml"))

from lab.targets import engine  # noqa: E402

_src = engine.repo() / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
# The grader runs lint and tests with this interpreter.
os.environ.setdefault("LAB_ENGINE_PYTHON", sys.executable)


def pytest_configure(config):
    config.addinivalue_line("markers", "heavy: loads a real model (GBs of RAM); run alone")
    config.addinivalue_line("markers", "needs_host: needs git, sockets or the held-out corpus")


@pytest.fixture(scope="session")
def stub_backend_cls():
    """The engine's own test StubBackend, loaded from the engine repo's tests/."""
    pytest.importorskip("inference_server.server", reason="needs the engine and its deps", exc_type=ImportError)
    path = engine.repo() / "tests" / "stub_backend.py"
    if not path.exists():
        pytest.skip(f"no {path}")
    spec = importlib.util.spec_from_file_location("engine_stub_backend", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StubBackend


@pytest.fixture(autouse=True)
def no_installed_cloud_cli(monkeypatch, tmp_path):
    """No test reaches the machine's own `nebius` CLI: only one a test puts on PATH is found."""
    from lab.infra.cloud.providers import nebius
    monkeypatch.setattr(nebius, "CLI_DIR", tmp_path / "no-nebius")
