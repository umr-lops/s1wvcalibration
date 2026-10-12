"""Locate the WW3 spectra file colocated with a S-1 L2 measurement.

Ported from ``colocation/get_ww3spectrum_file_fromL2S1measu.py``. The
``shared_information.PROJECT_DIR_DATARMOR`` and ``sats_acro`` references are
replaced by the ``paths.ww3_coloc`` and ``satellites`` configuration entries.
"""

from __future__ import annotations

import datetime as dt
import logging
from datetime import timezone
from pathlib import Path

from s1wvcalibration.config import load_config

logger = logging.getLogger(__name__)

__all__ = ["get_ww3spectra_l2_filepath"]

_UTC = timezone.utc


def _satellite_name(sat_acro: str) -> str:
    return str(load_config()["satellites"][sat_acro.upper()])


def _ww3_coloc_dir() -> str:
    return str(load_config()["paths"]["ww3_coloc"])


def _rejeu_2016_dirs() -> dict:
    coloc = Path(_ww3_coloc_dir())
    s1a = {
        "201601": "sentinel-1a/sar-model/ww3spectra/rejeu_2016_01_07_L2/netcdf/",
        "201602": "sentinel-1a/sar-model/ww3spectra/rejeu_2016_01_07_L2/netcdf/",
        "201603": "sentinel-1a/sar-model/ww3spectra/rejeu_2016_01_07_L2/netcdf/",
        "201604": "sentinel-1a/sar-model/ww3spectra/rejeu_2016_01_07_L2/netcdf/",
        "201605": "sentinel-1a/sar-model/ww3spectra/rejeu_2016_01_07_L2/netcdf/",
        "201606": "sentinel-1a/sar-model/ww3spectra/rejeu_2016_01_07_L2/netcdf/",
        "201607": "sentinel-1a/sar-model/ww3spectra/rejeu_2016_01_07_L2/netcdf/",
        "201608": "sentinel-1a/sar-model/ww3spectra/rejeu_2016_08_09_L2/netcdf/",
        "201609": "sentinel-1a/sar-model/ww3spectra/rejeu_2016_08_09_L2/netcdf/",
        "201610": "sentinel-1a/sar-model/ww3spectra/rejeu_2016_10_11_L2/netcdf/",
        "201611": "sentinel-1a/sar-model/ww3spectra/rejeu_2016_10_11_L2/netcdf/",
    }
    s1b = {
        "201606": "sentinel-1b/sar-model/ww3spectra/rejeu_2016_06_11_L2/netcdf/",
        "201607": "sentinel-1b/sar-model/ww3spectra/rejeu_2016_06_11_L2/netcdf/",
        "201608": "sentinel-1b/sar-model/ww3spectra/rejeu_2016_06_11_L2/netcdf/",
        "201609": "sentinel-1b/sar-model/ww3spectra/rejeu_2016_06_11_L2/netcdf/",
        "201610": "sentinel-1b/sar-model/ww3spectra/rejeu_2016_06_11_L2/netcdf/",
        "201611": "sentinel-1b/sar-model/ww3spectra/rejeu_2016_06_11_L2/netcdf/",
    }
    return {"S1A": {k: coloc / v for k, v in s1a.items()}, "S1B": {k: coloc / v for k, v in s1b.items()}}


def get_ww3spectra_l2_filepath(
    measu_filepath: str, use_new_procedure: str | None = None
) -> tuple[str, str]:
    """Return ``(filepath, procedure)`` of the colocated WW3 spectra netCDF.

    Args:
        measu_filepath: full path of the S-1 L2 OCN measurement.
        use_new_procedure: REJEU2016 or SATGLOB or SENTINEL, None => cron.
    """
    basename = Path(measu_filepath).name
    satellite = _satellite_name(basename[:3])
    yearmonth = basename.split("-")[4][:6]
    datedt = dt.datetime.strptime(basename.split("-")[4], "%Y%m%dt%H%M%S").replace(tzinfo=_UTC)
    logger.debug("provenance of the coloc: %s", use_new_procedure)
    if use_new_procedure is None:
        procedure_coloc = "run_datarmor_satglob"
        directo = (
            Path(_ww3_coloc_dir())
            / satellite
            / "sar-model/ww3spectra"
            / yearmonth
            / "netCDF_L2_daily"
            / datedt.strftime("%Y%m%d")
        )
    elif use_new_procedure == "SATGLOB":
        procedure_coloc = "run_ope_satglob"
        logger.debug("enter patch new procedure")
        directo = Path(
            f"/home/cercache/users/agrouaze/temporaire/sentinel1/coloc/new_recomposition_ww3spectra_{yearmonth}",
            basename[:3].upper(),
            "L2/",
        )
    elif use_new_procedure == "SENTINEL":
        procedure_coloc = "run_ope_sentinel"
        raise NotImplementedError
    elif use_new_procedure == "REJEU2016":
        procedure_coloc = "rejeu2016"
        directo = _rejeu_2016_dirs()[basename[:3].upper()][yearmonth]
        logger.debug("which coloc dir to look at: %s", directo)
    else:
        raise ValueError(f"cannot find the coloc dir WW3 {use_new_procedure}")
    ww3base = "ww3" + basename[3:].replace("-roughness.png", ".nc")
    filepath = Path(directo) / ww3base
    logger.debug("WW3 that should be present: %s", filepath)
    return str(filepath), procedure_coloc
