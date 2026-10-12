"""Reader for the ECMWF 0.125 deg 3-hourly (METEOFRANCE_CEP) wind product.

Ported from ``colocation/get_meteofrance_cep_0125_wind.py``.
"""

from __future__ import annotations

import datetime as dt
import logging

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from netCDF4 import Dataset

logger = logging.getLogger(__name__)

__all__ = ["get_meteofrance_cep_0125_wind"]


def get_meteofrance_cep_0125_wind(
    file: str, area: object = None, verbose: bool = False, control_plot: bool = False
) -> dict:
    """Read a METEOFRANCE_CEP netCDF wind file into a dict of 2D arrays.

    The output dict contains ``lat``, ``lon``, ``zon``, ``mer``, ``speed`` and
    ``time``.
    """
    data = Dataset(file)

    # reverse latitudes so they go from south to north
    reslat = float(data.latitude_resolution)
    latmin = float(data.southernmost_latitude)
    latmax = float(data.northernmost_latitude)
    reslon = float(data.longitude_resolution)
    lonmin = float(data.westernmost_longitude)
    lonmax = float(data.easternmost_longitude)
    latitudes = (np.arange(latmin, latmax + reslat, reslat))[::-1]
    longitudes = np.arange(lonmin, lonmax + reslon, reslon)

    if verbose:
        logger.info(
            "Native product lon min/max: %s %s, lat min/max: %s %s",
            np.min(longitudes),
            np.max(longitudes),
            np.min(latitudes),
            np.max(latitudes),
        )

    # 10-m wind data
    uin = (data.variables["u10m"][:].squeeze())[::-1, :]
    vin = (data.variables["v10m"][:].squeeze())[::-1, :]
    sin = np.sqrt(np.power(uin, 2) + np.power(vin, 2))

    lon, lat = np.meshgrid(longitudes, latitudes)

    wind: dict = {
        "lat": lat,
        "lon": lon,
        "zon": uin,
        "mer": vin,
        "speed": sin,
    }

    _times = netCDF4.num2date(data.variables["time"][:], data.variables["time"].getncattr("units"))
    gregtime = _times[0] if isinstance(_times, np.ndarray) else _times
    wind["time"] = dt.datetime(
        gregtime.year, gregtime.month, gregtime.day, gregtime.hour, gregtime.minute, gregtime.second,
        tzinfo=dt.timezone.utc,
    )

    if control_plot:
        fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)
        axes[0, 0].imshow(sin)
        axes[0, 0].set_title("speed")
        axes[1, 0].imshow(uin)
        axes[1, 0].set_title("zonal component")
        axes[2, 0].imshow(vin)
        axes[2, 0].set_title("meridional component")
        axes[0, 1].imshow(lon)
        axes[0, 1].set_title("longitude")
        axes[1, 1].imshow(lat)
        axes[1, 1].set_title("latitude")
        plt.savefig("control_plot_meteofrance_cep_0125_wind.png")
        plt.close(fig)

    if verbose:
        logger.info(
            "Output product lon shape/min/max: %s %s %s, lat shape/min/max: %s %s %s, speed min/max: %s %s",
            np.shape(lon),
            np.min(lon),
            np.max(lon),
            np.shape(lat),
            np.min(lat),
            np.max(lat),
            np.min(sin),
            np.max(sin),
        )

    if area is not None:  # kept for signature compatibility
        logger.debug("area argument not used: %s", area)

    return wind
