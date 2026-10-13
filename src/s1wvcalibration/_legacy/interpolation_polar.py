"""Interpolate a polar (phi, k) spectrum onto a new grid.

Ported from ``graphics/interpolation_polar_spectrum.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from scipy import interpolate

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = ["interp_polar_spectrum"]


def interp_polar_spectrum(
    k_out: NDArray,
    phi_out: NDArray,
    spec_in: NDArray,
    k_in: NDArray,
    phi_in: NDArray,
    order: str = "C",
) -> NDArray:
    """Interpolate the input polar spectrum onto the (k_out, phi_out) grid.

    Args:
        phi_out: output phi in degrees.
        k_out: output wave number (m-1).
        spec_in: input 2D spectrum (phi, k) order.
        k_in: input wave number.
        phi_in: input phi in degrees.
    """
    if spec_in.shape[0] == len(phi_in) and spec_in.shape[1] == len(k_in):
        logger.debug("shape input spectra seems ok")
    else:
        logger.warning("shape input spectra does not match (phi,k) expected dimensions")
        msg = f"wrong input spectra shape {spec_in.shape[0]}x{spec_in.shape[1]} it needs phi,k order"
        raise ValueError(msg)

    dim = (phi_in.size, k_in.size)
    kgrid = np.tile(k_in, (dim[0], 1))
    phigrid = np.tile(phi_in, (dim[1], 1)).transpose()
    x = (kgrid * np.cos(phigrid * np.pi / 180)).flatten()
    y = (kgrid * np.sin(phigrid * np.pi / 180)).flatten()
    z = spec_in.flatten(order=cast("Literal['C', 'F']", order))

    dimout = (phi_out.size, k_out.size)
    koutgrid = np.tile(k_out, (dimout[0], 1))
    phioutgrid = np.tile(phi_out, (dimout[1], 1)).transpose()
    xout = koutgrid * np.cos(phioutgrid * np.pi / 180)
    yout = koutgrid * np.sin(phioutgrid * np.pi / 180)
    return interpolate.griddata((x, y), z, (xout, yout), method="linear")
