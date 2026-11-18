"""Geophysical model functions (GMF).

Three GMFs feed the WV geophysical calibration. Each is exposed in two
implementations where relevant:

* the **legacy** one (``*_legacy``) — exact port of what the current production
  runs, kept numerically identical (this is what the operational ``sig_mod`` is);
* the **xsarsea** one (``*_xsarsea`` / default) — the modern standard
  implementation, stored alongside during the migration so the two can be
  tracked in production before the legacy one is retired.

Legacy argument semantics are preserved:

* ``v``    -- wind speed [m/s]
* ``phi``  -- wind azimuth relative to the antenna [deg]
* ``theta``/``inc`` -- incidence angle [deg]

All functions return a linear (unitless) sigma-0 ``numpy.ndarray``.

PARITY NOTES
------------
* ``cmod5n_legacy`` vs ``cmod5n_xsarsea``: the legacy model applies its low-wind
  ``S -> S0`` clamp per-array (see :mod:`s1wvcalibration._legacy.cmod5n_ipf`),
  xsarsea element-wise. Measured difference up to ~1.4 dB on the classical
  7 m/s sweet spot (mostly WV2 / higher incidence). See ``test_gmf_parity``.
* ``cmod_ifr2``: the xsarsea implementation was verified **exactly equal** to
  the legacy ``cmod_ifremer.cmod_ifr2`` (max diff 0.0 dB), so a single
  implementation is enough.
* ``cmod5``: the legacy production called ``sar.utils.gmf.cmod5(u10, phi, inc)``
  whose signature is ``cmod5(u10, phi, inc, neutral=True)``: despite its name it
  used the **CMOD5.N** coefficients (element-wise, standard formulation). The
  ``correction_wv*_cmod5*`` variables and the ``wind_vs_sig_calibrated_cmod5``
  figure have therefore always been CMOD5.N. :func:`cmod5` reproduces this with
  xsarsea ``cmod5n`` (verified against the legacy source to < 1e-5 dB, see
  ``test_gmf_parity``). Calling the true CMOD5 (xsarsea ``cmod5``) shifts the
  constants by +0.3 to +0.75 dB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from xsarsea.windspeed import get_model
from xsarsea.windspeed.gmfs_impl import gmf_cmodifr2

from .cmod5n_ipf import cmod5n_legacy

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "cmod5",
    "cmod5n_forward",
    "cmod5n_legacy",
    "cmod5n_xsarsea",
    "cmod_ifr2",
    "get_cmod5n",
]


def _to_linear_ndarray(x: Any) -> NDArray:
    """Convert a model output (xarray.DataArray / ndarray / scalar) to a float ndarray."""
    if hasattr(x, "values"):
        x = x.values
    return np.asarray(x, dtype=np.float64)


def get_cmod5n() -> Any:
    """Return the xsarsea CMOD5_N (neutral wind) model object."""
    return get_model("cmod5n")


def cmod5n_xsarsea(v: NDArray, phi: NDArray, theta: NDArray) -> NDArray:
    """Standard CMOD5_N (element-wise clamp) linear backscatter via xsarsea.

    Args:
        v: wind speed [m/s].
        phi: wind azimuth relative to antenna [deg].
        theta: incidence angle [deg].
    """
    model = get_cmod5n()
    # xsarsea Model.__call__ signature: (inc, wspd, phi, broadcast).
    res = model(theta, v, phi, broadcast=True)
    return _to_linear_ndarray(res)


def cmod5n_forward(v: NDArray, phi: NDArray, theta: NDArray) -> NDArray:
    """CMOD5_N as used by the current production (legacy, batch-corrected).

    This is the value written in the operational ``sig_mod``. See
    :mod:`s1wvcalibration._legacy.cmod5n_ipf` for the "correction vector
    agrouaze June 2019" batch behaviour and the ~1.4 dB gap with the standard
    model.
    """
    return cmod5n_legacy(v, phi, theta)


def cmod5(v: NDArray, phi: NDArray, theta: NDArray) -> NDArray:
    """Legacy "cmod5" linear backscatter -- which is in fact CMOD5.N.

    Reproduces ``sar.utils.gmf.cmod5(u10, phi, inc)`` as called by the legacy
    production, i.e. with its default ``neutral=True`` (CMOD5.N coefficients,
    element-wise). Kept under this name so that the ``*_cmod5`` NetCDF
    variables and figure stay continuous with the historical archive.

    Args:
        v: wind speed [m/s].
        phi: wind azimuth relative to antenna [deg].
        theta: incidence angle [deg].
    """
    return cmod5n_xsarsea(v, phi, theta)


def cmod_ifr2(inc: NDArray, v: NDArray, phi: NDArray) -> NDArray:
    """CMOD-IFR2 linear backscatter.

    Verified exactly equal to the legacy ``cmod_ifremer.cmod_ifr2`` (max diff
    0.0 dB), so only the xsarsea implementation is kept.

    Args:
        inc: incidence angle [deg].
        v: wind speed [m/s].
        phi: wind azimuth relative to antenna [deg].
    """
    res = gmf_cmodifr2(inc, v, phi)
    return _to_linear_ndarray(res)
