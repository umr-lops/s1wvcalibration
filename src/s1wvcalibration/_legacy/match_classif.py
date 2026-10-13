"""Match a S-1 WV measurement file with its Deep Learning classification XML.

Ported from ``qualitycheck/match_measu_wv_with_classif_xml.py``. The
``shared_information.CLASSIF_WV_CLS`` constant is replaced by the
``paths.classif_wv_cls`` configuration entry.
"""

from __future__ import annotations

import datetime as dt
import logging
from datetime import timezone
from pathlib import Path

from s1wvcalibration.config import load_config

logger = logging.getLogger(__name__)

__all__ = ["get_xml_path_from_measu", "get_xml_path_from_safe"]

_UTC = timezone.utc


def _classif_root() -> str:
    return str(load_config()["paths"]["classif_wv_cls"])


def get_xml_path_from_measu(full_path_measu: str) -> str | None:
    """Return the DL_Summary.xml path for a measurement, or None if not found."""
    base_safe = Path(full_path_measu).parent.parent.name
    logger.debug("basesafe = %s", base_safe)
    base, star_name = _classif_glob_pattern(base_safe)
    logger.debug("pattern = %s", base / star_name)
    poten = list(base.glob(star_name))
    if len(poten) >= 1:
        return str(poten[0])
    return None


def get_xml_path_from_safe(base_safe: str) -> str | None:
    """Return the DL_Summary.xml path for a SAFE name, or None if not found."""
    logger.debug("basesafe = %s", base_safe)
    base, star_name = _classif_glob_pattern(base_safe)
    logger.debug("pattern = %s", base / star_name)
    poten = list(base.glob(star_name))
    if len(poten) >= 1:
        return str(poten[0])
    return None


def _classif_glob_pattern(base_safe: str) -> tuple[Path, str]:
    """Return ``(concrete_base_dir, pattern)`` for the DL_Summary.xml lookup.

    The wildcard (replacing the SAFE unique id) lives in the directory name, so
    the pattern is anchored on the concrete ``.../YYYY/MM/DD`` base directory.
    """
    date_dt = dt.datetime.strptime(base_safe.split("_")[5], "%Y%m%dT%H%M%S").replace(
        tzinfo=_UTC
    )
    unique_id = base_safe.split("_")[9]
    base = (
        Path(_classif_root())
        / date_dt.strftime("%Y")
        / date_dt.strftime("%m")
        / date_dt.strftime("%d")
    )
    if "OCN" in base_safe:
        star_dir = base_safe.replace("OCN", "DL").replace(unique_id, "*")
    else:  # SLC
        star_dir = (
            base_safe.replace("SLC", "DL").replace(unique_id, "*").replace("1SS", "2SS")
        )
    return base, f"{star_dir}/DL_Summary.xml"
