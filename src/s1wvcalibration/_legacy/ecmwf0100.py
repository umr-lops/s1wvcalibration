"""Reader for the ECMWF 0.100 deg hourly wind product.

Ported from ``colocation/reader_ecmwf_0p100_hourly_nc.py``. The two hard
cluster paths (light / full) are now function parameters defaulting to the
``paths.ecmwf_0100_light`` / ``paths.ecmwf_0100_full`` configuration entries.
"""

from __future__ import annotations

import datetime as dt
import logging
from datetime import timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np

from s1wvcalibration.config import load_config

logger = logging.getLogger(__name__)

__all__ = ["findfile", "read_full_file", "readfile"]

ecmwf_hours = np.arange(0, 24)

# Product dates/times are UTC; keep every datetime in this module tz-aware UTC
# so the ECMWF colocation (nearest file by wall-clock) is timezone-independent.
_UTC = timezone.utc


def readfile(fname: str) -> dict:
    """Read the light netCDF (U and V only) into a dict of 2D arrays."""
    data = nc.Dataset(fname)
    ecmwf_data: dict = {}
    ecmwf_data["lat"] = data.variables["Latitude"][::].squeeze()
    ecmwf_data["lon"] = data.variables["Longitude"][::].squeeze()
    ecmwf_data["time"] = nc.num2date(data.variables["time"][:], data.variables["time"].units)
    ecmwf_data["uwind"] = data.variables["10U"][::].squeeze()
    ecmwf_data["vwind"] = data.variables["10V"][::].squeeze()
    ecmwf_data["zon"] = ecmwf_data["uwind"]
    ecmwf_data["mer"] = ecmwf_data["vwind"]
    ecmwf_data["speed"] = np.sqrt(ecmwf_data["uwind"] ** 2 + ecmwf_data["vwind"] ** 2)
    return ecmwf_data


def read_full_file(fname: str) -> dict:
    """Read the full netCDF (U, V and MSL) into a dict of 2D arrays."""
    data = nc.Dataset(fname)
    ecmwf_data: dict = {}
    ecmwf_data["lat"] = data.variables["lat"][::].squeeze()
    ecmwf_data["lon"] = data.variables["lon"][::].squeeze()
    _time = dt.datetime.strptime(
        Path(fname).name.split("_")[3].replace(".nc", ""), "%Y%m%d%H%M"
    ).replace(tzinfo=_UTC)
    ecmwf_data["time"] = _time
    ecmwf_data["uwind"] = data.variables["10u"][::].squeeze()
    ecmwf_data["vwind"] = data.variables["10v"][::].squeeze()
    ecmwf_data["msl"] = data.variables["msl"][::].squeeze()
    return ecmwf_data


def findfile(fltdate, path_light: str | None = None, path_full: str | None = None, typee: str = "light") -> str:
    """Return the path of the nearest hourly ECMWF file for *fltdate*.

    Args:
        fltdate: datetime or timestamp (float).
        path_light: Root dir of the light files (default from config).
        path_full: Root dir of the full files (default from config).
        typee: ``"light"`` or ``"full"``.
    """
    logger.debug("fltdate = %s val= %s", type(fltdate), fltdate)
    if path_light is None:
        path_light = str(load_config()["paths"]["ecmwf_0100_light"])
    if path_full is None:
        path_full = str(load_config()["paths"]["ecmwf_0100_full"])

    if isinstance(fltdate, float):  # case timestamp (epoch seconds -> UTC)
        date = dt.datetime.fromtimestamp(fltdate, tz=_UTC)
    else:
        date = fltdate
    logger.debug("reader_ecmwf_0100 ... findfile | %s", date)

    datefile = [
        dt.datetime(date.year, date.month, date.day, hour, tzinfo=_UTC) for hour in ecmwf_hours
    ]
    datefile.append(datefile[0] + dt.timedelta(days=1))

    offsets = [abs(date - datefilei) for datefilei in datefile]
    nearest_date = datefile[offsets.index(min(offsets))]

    doy = nearest_date.timetuple().tm_yday
    year = nearest_date.year
    nomencl = nearest_date.strftime("%Y%m%d%H%M")
    if typee == "light":
        ecmwf_file = Path(path_light) / str(year) / f"{doy:03d}" / f"ECMWF_FORECAST_0100_{nomencl}_10U_10V.nc"
    else:  # full
        ecmwf_file = Path(path_full) / str(year) / f"{doy:03d}" / f"ECMWF_FORECAST_0100_{nomencl}.nc"
    return str(ecmwf_file)
