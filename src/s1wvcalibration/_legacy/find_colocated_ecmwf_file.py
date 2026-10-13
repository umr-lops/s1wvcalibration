"""Find the ECMWF 0.125 deg 3-hourly wind file closest to a given date.

Ported from ``colocation/find_colocated_ecmwf_file.py``. The hard cluster path
is replaced by the ``paths.ecmwf_0125`` entry of the package configuration.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from s1wvcalibration.config import data_path

logger = logging.getLogger(__name__)

__all__ = ["find_colocated_ecmwf_file"]

_UTC = timezone.utc


def find_colocated_ecmwf_file(
    date: datetime, path_wind: str | None = None, hour_step: int | None = None
) -> dict:
    """Locate the METEOFRANCE_CEP netCDF file closest to *date*.

    Args:
        date: SAR acquisition date.
        path_wind: Root directory of the 3-hourly ECMWF archive. Defaults to
            the ``paths.ecmwf_0125`` configuration entry.
        hour_step: Forecast cadence in hours (default 3).

    Returns:
        A dict with ``status`` (1 if found) and ``filename`` (path or None).
    """
    res: dict = {"status": 0, "filename": None}

    if path_wind is None:
        path_wind = data_path("ecmwf_0125")
    path_wind_p = Path(path_wind)
    logger.debug("path_wind : %s", path_wind)
    if hour_step is None:
        hour_step = 3

    date_moins = date - timedelta(1)
    date_plus = date + timedelta(1)

    files_wind: list[str] = []
    for day in (date, date_moins, date_plus):
        path_wind_day = path_wind_p / str(day.year) / f"{day.timetuple().tm_yday:03d}"
        files_wind += [str(p) for p in path_wind_day.glob("METEOFRANCE_CEP_*.nc")]

    # initial threshold (seconds): accept any file closer than this.
    best_dti = float(hour_step) * 24 * 60 * 60.0
    file_wind_coloc: str | None = None
    for file_wind in files_wind:
        date_fic = Path(file_wind).name[16:27]
        date_ecmwf = datetime.strptime(date_fic, "%Y%m%dT%H").replace(tzinfo=_UTC)
        dti = abs((date_ecmwf - date).total_seconds())
        if dti < best_dti:
            file_wind_coloc = file_wind
            best_dti = dti

    if file_wind_coloc is not None:
        logger.info("Successful ECMWF Wind colocation")
        logger.debug("colocated Wind field: %s", Path(file_wind_coloc).name)
        res["status"] = 1
        res["filename"] = file_wind_coloc
    else:
        logger.error("failed ECMWF colocation")
    return res
