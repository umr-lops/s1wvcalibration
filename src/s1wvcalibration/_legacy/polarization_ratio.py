"""Polarization ratio (ported from qualitycheck/get_polarization_ratio.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["get_polarization_ratio"]


def get_polarization_ratio(
    u10: NDArray, inc: NDArray, phi: Any = None, name: str = "mouche1"
) -> NDArray:
    """HH/VV polarization ratio used to convert the VV GMF to HH.

    Args:
        u10: wind speed (m/s) -- kept for signature compatibility.
        inc: incidence angle (deg).
        phi: wind azimuth (deg) -- only used by ``mouche2``.
        name: ``"mouche1"`` (default) or ``"mouche2"``.
    """
    _ = (u10, phi)
    if name == "mouche2":
        A0 = 0.00650704
        B0 = 0.128983
        C0 = 0.992839
        Api2 = 0.00782194
        Bpi2 = 0.121405
        Cpi2 = 0.992839
        Api = 0.00598416
        Bpi = 0.140952
        Cpi = 0.992885

        P0_theta = A0 * np.exp(B0 * inc) + C0
        Ppi2_theta = Api2 * np.exp(Bpi2 * inc) + Cpi2
        Ppi_theta = Api * np.exp(Bpi * inc) + Cpi

        C0_theta = (P0_theta + Ppi_theta + 2 * Ppi2_theta) / 4
        C1_theta = (P0_theta - Ppi_theta) / 2
        C2_theta = (P0_theta + Ppi_theta - 2 * Ppi2_theta) / 4

        return (
            C0_theta
            + C1_theta * np.cos(phi * np.pi / 180.0)
            + C2_theta * np.cos(2 * phi * np.pi / 180.0)
        )

    A = 0.00799793
    B = 0.125465
    C = 0.997379
    return A * np.exp(B * inc) + C
