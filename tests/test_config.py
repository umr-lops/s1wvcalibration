"""Unit tests for the configuration loader (replaces shared_information)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import s1wvcalibration.config as config_mod
from s1wvcalibration.config import (
    CONFIG_ENV_VAR,
    _load_yaml,
    data_path,
    find_local_config,
    get_config,
    load_config,
    reset_config_cache,
)


@pytest.fixture(autouse=True)
def _clean_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


def test_defaults_loaded():
    cfg = load_config()
    assert cfg["calibration"]["high_lat_threshold"] == 50
    assert cfg["calibration"]["wind_ranges"]["classical"] == 7
    assert cfg["calibration"]["wind_ranges"]["noise_free"] == 10
    assert cfg["calibration"]["dev_max_measu"] == 3
    assert "root_directory" in cfg["paths"]
    assert cfg["satellites"]["S1A"] == "sentinel-1a"


def test_get_config_is_alias():
    assert get_config() is load_config()


def test_find_local_config_none():
    assert find_local_config() is None


def test_find_local_config_env_missing_warns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    missing = str(tmp_path / "does_not_exist.yaml")
    monkeypatch.setenv(CONFIG_ENV_VAR, missing)
    with caplog.at_level("WARNING"):
        assert find_local_config() is None
    assert any(missing in rec.message for rec in caplog.records)


def test_find_local_config_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = tmp_path / "override.yaml"
    cfg.write_text("paths:\n  root_directory: /tmp/override-root\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg))
    assert find_local_config() == cfg


def test_find_local_config_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    (tmp_path / "localconfig.yaml").write_text("calibration:\n  dev_max_measu: 9\n")
    monkeypatch.chdir(tmp_path)
    assert find_local_config() == tmp_path / "localconfig.yaml"


def test_deep_merge_single_leaf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Overriding one leaf keeps the sibling defaults.
    cfg = tmp_path / "override.yaml"
    cfg.write_text("paths:\n  root_directory: /tmp/new-root\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg))
    merged = load_config()
    assert merged["paths"]["root_directory"] == "/tmp/new-root"
    # an untouched sibling default survives
    assert merged["paths"]["analysis_dir"]


def test_override_nested_mapping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = tmp_path / "override.yaml"
    cfg.write_text("calibration:\n  wind_ranges:\n    classical: 5\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg))
    merged = load_config()
    assert merged["calibration"]["wind_ranges"]["classical"] == 5
    # the other wind range default is preserved by the deep merge
    assert merged["calibration"]["wind_ranges"]["noise_free"] == 10


def test_load_yaml_non_mapping_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n- b\n")
    with pytest.raises(TypeError, match="must contain a mapping"):
        _load_yaml(bad)


def test_data_path_root_as_is():
    root = data_path("root_directory")
    assert Path(root).is_absolute()


def test_data_path_relative_joined():
    cfg = load_config()
    root = cfg["paths"]["root_directory"]
    assert data_path("analysis_dir") == str(Path(root) / cfg["paths"]["analysis_dir"])


def test_data_path_absolute_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = tmp_path / "override.yaml"
    cfg.write_text("paths:\n  ww3_satglob: /abs/escape/path\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg))
    reset_config_cache()
    assert data_path("ww3_satglob") == "/abs/escape/path"


def test_config_yaml_parses():
    with config_mod.CONFIG_YAML.open(encoding="utf-8") as fid:
        data = yaml.safe_load(fid)
    assert set(data) >= {"paths", "calibration", "gmf", "satellites"}
