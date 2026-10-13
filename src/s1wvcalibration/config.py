"""Configuration loading for s1wvcalibration.

This replaces the legacy ``shared_information`` module. A packaged
``config.yaml`` holds the defaults; a ``localconfig.yaml`` (in the current
working directory, or pointed to by the ``S1WVCALIB_CONFIG`` environment
variable) overrides any key without editing the packaged file.

The merge is *deep*: a mapping value in the override is merged into the
default mapping key by key, so e.g. overriding only
``paths.root_directory`` keeps every other default path. A non-mapping
override value (string, int, ...) replaces the default outright.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import TypedDict, cast

import yaml

logger = logging.getLogger(__name__)


class _WindRanges(TypedDict):
    """Center wind speed (m/s) per sweet-spot (``classical`` / ``noise_free``)."""

    classical: int
    noise_free: int


class _Calibration(TypedDict):
    """Per-SAFE calibration constants."""

    high_lat_threshold: int
    wind_ranges: _WindRanges
    dev_max_measu: int
    time_unit: str


class _Gmf(TypedDict):
    """GMF (xsarsea) settings."""

    nc_luts_path: str


class _Paths(TypedDict):
    """Data locations on the cluster (all paths as ``str`` for portability).

    ``root_directory`` is the single machine-specific value; the other entries
    are sub-paths relative to it (or absolute, to escape the root). See
    :func:`data_path`.
    """

    root_directory: str
    project_dir: str
    analysis_dir: str
    ecmwf_0100_light: str
    ecmwf_0100_full: str
    ecmwf_0125: str
    land_mask: str
    classif_wv_cls: str
    ww3_coloc: str
    ww3_satglob: str


class Config(TypedDict):
    """Merged top-level configuration (see ``config.yaml`` for defaults)."""

    paths: _Paths
    calibration: _Calibration
    gmf: _Gmf
    satellites: dict[str, str]


#: Directory of the packaged default configuration.
_PACKAGE_DIR = Path(__file__).parent
#: Packaged default configuration file.
CONFIG_YAML = _PACKAGE_DIR / "config.yaml"
#: Environment variable pointing to a full override config file.
CONFIG_ENV_VAR = "S1WVCALIB_CONFIG"


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as fid:
        data = yaml.safe_load(fid) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file {path} must contain a mapping at top level.")
    return data


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> None:
    """Recursively merge *override* into *base* in place.

    For a key present in both: if both values are mappings they are merged
    recursively; otherwise the override value wins. Keys only in *override*
    are added. This lets an override supply a single leaf (e.g.
    ``paths.root_directory``) without dropping its sibling defaults.
    """
    for key, value in override.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge(
                cast("dict[str, object]", existing), cast("dict[str, object]", value)
            )
        else:
            base[key] = value


def find_local_config() -> Path | None:
    """Return the path of a local override config, or ``None``.

    Resolution order:
      1. ``$S1WVCALIB_CONFIG`` if set and the file exists.
      2. ``./localconfig.yaml`` if it exists.
    """
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        p = Path(env)
        if p.is_file():
            return p
        logger.warning(
            "%s set to %s which does not exist; ignoring.", CONFIG_ENV_VAR, env
        )
    local = Path.cwd() / "localconfig.yaml"
    if local.is_file():
        return local
    return None


@lru_cache(maxsize=1)
def load_config() -> Config:
    """Load the merged configuration (defaults + local override).

    The merge is deep (see :func:`_deep_merge`): an override mapping is merged
    into the default mapping key by key, so a single-leaf override (e.g.
    ``paths.root_directory``) keeps the other defaults.
    """
    config = _load_yaml(CONFIG_YAML)
    local_path = find_local_config()
    if local_path is not None:
        local = _load_yaml(local_path)
        logger.info(
            "Merging local config from %s (keys: %s).", local_path, sorted(local)
        )
        _deep_merge(config, local)
    return cast("Config", config)


def get_config() -> Config:
    """Alias of :func:`load_config` for callers that prefer a get_* name."""
    return load_config()


def data_path(key: str) -> str:
    """Resolve a ``paths`` entry to an absolute path string.

    ``root_directory`` is returned as-is. Any other ``paths`` key that is a
    relative path is joined onto ``root_directory``; an absolute sub-path is
    returned unchanged so a location can escape the root when it does not follow
    the standard layout.

    Args:
        key: key of a value in the ``paths`` mapping (e.g. ``"analysis_dir"``).

    Returns:
        The resolved absolute path as a ``str``.
    """
    paths = cast("dict[str, str]", load_config()["paths"])
    if key == "root_directory":
        return paths[key]
    value = paths[key]
    p = Path(value)
    if p.is_absolute():
        return value
    return str(Path(paths["root_directory"]) / p)


def reset_config_cache() -> None:
    """Clear the cached config (used by tests that monkeypatch the environment)."""
    load_config.cache_clear()
