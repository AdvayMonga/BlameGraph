"""Launch profiles read from the engine tree's own file: which launches a measurement makes."""
from __future__ import annotations

import pytest

from lab import profiles, target

SPEC = '''
[model]
name = "fake"
[engine]
repo = "."
launch = "python serve.py --port {port}"
test = "true"
lint = "true"
'''


def _target(tmp_path, extra: str = 'profiles = "profiles.toml"\nprofile_env = "ENGINE_PROFILE"\n'):
    p = tmp_path / "t.toml"
    p.write_text(SPEC + extra)
    return target.parse(p)


def test_no_file_or_no_target_setting_means_one_general_launch(tmp_path):
    assert profiles.launches(tmp_path, _target(tmp_path), ["single_stream"]) == [(None, ["single_stream"])]
    (tmp_path / "profiles.toml").write_text("[single_stream]\nMAX_BATCH_SIZE = 1\n")
    assert profiles.launches(tmp_path, _target(tmp_path, ""), ["single_stream"]) == [(None, ["single_stream"])]


def test_each_regime_with_its_own_table_gets_one_more_launch(tmp_path):
    (tmp_path / "profiles.toml").write_text("[general]\nMAX_BATCH_SIZE = 64\n[single_stream]\nMAX_BATCH_SIZE = 1\n"
                                            "[saturated]\n[bursty]\nMAX_BATCH_SIZE = 128\n")
    t = _target(tmp_path)
    got = profiles.launches(tmp_path, t, ["single_stream", "saturated", "overload"])
    assert got == [(None, ["single_stream", "saturated", "overload"]), ("single_stream", ["single_stream"])]
    assert profiles.env(t, "single_stream") == {"ENGINE_PROFILE": "single_stream"} and profiles.env(t, None) == {}


def test_a_file_that_does_not_parse_fails_the_measurement(tmp_path):
    (tmp_path / "profiles.toml").write_text("[single_stream\n")
    with pytest.raises(ValueError):
        profiles.launches(tmp_path, _target(tmp_path), ["single_stream"])


def test_the_file_and_its_env_var_go_together(tmp_path):
    with pytest.raises(ValueError, match="go together"):
        _target(tmp_path, 'profiles = "profiles.toml"\n')


def test_refusal_names_why_and_never_raises(tmp_path):
    t = _target(tmp_path)
    (tmp_path / "profiles.toml").write_text("[cold_start]\nMAX_BATCH_SIZE = 1\n[single_stream]\nMAX_BATCH_SIZE = 1\n")
    assert profiles.refusal(tmp_path, t, None) is None and profiles.refusal(tmp_path, t, "single_stream") is None
    assert "is not a regime" in profiles.refusal(tmp_path, t, "cold_open")
    assert profiles.refusal(tmp_path, t, "saturated") == "no [saturated] table with settings in profiles.toml"
    (tmp_path / "profiles.toml").write_text("[single_stream\n")
    assert profiles.refusal(tmp_path, t, "single_stream").startswith("profiles.toml does not parse")
