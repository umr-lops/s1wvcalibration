"""Configuration loading for s1wvcalibration.

This replaces the legacy ``shared_information`` module. A packaged
``config.yaml`` holds the defaults; a ``localconfig.yaml`` (in the current
working directory, or pointed to by the ``S1WVCALIB_CONFIG`` environment
variable) overrides any top-level key without editing the packaged file.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

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
        logger.warning("%s set to %s which does not exist; ignoring.", CONFIG_ENV_VAR, env)
    local = Path.cwd() / "localconfig.yaml"
    if local.is_file():
        return local
    return None


@lru_cache(maxsize=1)
def load_config() -> dict[str, object]:
    """Load the merged configuration (defaults + local override).

    The merge is shallow: any top-level key present in the local override
    replaces the whole packaged sub-mapping for that key.
    """
    config = _load_yaml(CONFIG_YAML)
    local_path = find_local_config()
    if local_path is not None:
        local = _load_yaml(local_path)
        logger.info("Merging local config from %s (keys: %s).", local_path, sorted(local))
        config.update(local)
    return config


def get_config() -> dict[str, object]:
    """Alias of :func:`load_config` for callers that prefer a get_* name."""
    return load_config()


def reset_config_cache() -> None:
    """Clear the cached config (used by tests that monkeypatch the environment)."""
    load_config.cache_clear()
