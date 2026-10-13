"""Wind azimuth helpers (ported from qualitycheck/compute_wind_azimuth.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .format_angle import format_angle

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_azimuth_angle(
    track_angle: NDArray,
    wind_zonal_component: NDArray,
    wind_meridional_component: NDArray,
) -> NDArray:
    """Radar azimuth angle relative to geographic North, minus the wind angle.

    The S-1 C-band sensor points 90 deg to the right of the trajectory; this is
    compensated by the cyclonic ``arctan2`` convention (method of A. Mouche,
    January 2016).

    Args:
        track_angle: satellite track angle relative to North (deg).
        wind_zonal_component: wind zonal component (m/s).
        wind_meridional_component: wind meridional component (m/s).

    Returns:
        The antenna-referenced wind azimuth angle (deg).
    """
    wdi = np.mod(
        np.degrees(np.arctan2(wind_zonal_component, wind_meridional_component)), 360.0
    )
    wdi = np.mod(wdi + 180.0, 360)
    return abs(format_angle(track_angle + 90 - wdi, compass=False))


def method_wind_azi_range(
    wind_u: NDArray, wind_v: NDArray, trackangle: NDArray
) -> NDArray:
    """Wind azimuth relative to range direction (from the IPF owi_lop_application)."""
    ext_ancillary_wind_direction = 90.0 - np.rad2deg(np.arctan2(wind_v, wind_u))
    return -format_angle(90.0 + trackangle - ext_ancillary_wind_direction + 180.0)
