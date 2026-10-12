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
* ``cmod5``: only the xsarsea version is available (the legacy
  ``sar.utils.gmf.cmod5`` source is not in the sandbox); kept for reference
  (CMOD5 is not the GMF used for reporting/interpretation, only CMOD5n is).
"""

from __future__ import annotations

import numpy as np
from xsarsea.windspeed import get_model
from xsarsea.windspeed.gmfs_impl import gmf_cmodifr2

from .cmod5n_ipf import cmod5n_legacy

__all__ = [
    "cmod5",
    "cmod5n_forward",
    "cmod5n_legacy",
    "cmod5n_xsarsea",
    "cmod_ifr2",
    "get_cmod5n",
]


def _to_linear_ndarray(x) -> np.ndarray:
    """Convert a model output (xarray.DataArray / ndarray / scalar) to a float ndarray."""
    if hasattr(x, "values"):
        x = x.values
    return np.asarray(x, dtype=np.float64)


def get_cmod5n():
    """Return the xsarsea CMOD5_N (neutral wind) model object."""
    return get_model("cmod5n")


def cmod5n_xsarsea(v, phi, theta) -> np.ndarray:
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


def cmod5n_forward(v, phi, theta) -> np.ndarray:
    """CMOD5_N as used by the current production (legacy, batch-corrected).

    This is the value written in the operational ``sig_mod``. See
    :mod:`s1wvcalibration._legacy.cmod5n_ipf` for the "correction vector
    agrouaze June 2019" batch behaviour and the ~1.4 dB gap with the standard
    model.
    """
    return cmod5n_legacy(v, phi, theta)


def cmod5(v, phi, theta) -> np.ndarray:
    """CMOD5 linear backscatter (xsarsea).

    Args:
        v: wind speed [m/s].
        phi: wind azimuth relative to antenna [deg].
        theta: incidence angle [deg].
    """
    model = get_model("cmod5")
    res = model(theta, v, phi, broadcast=True)
    return _to_linear_ndarray(res)


def cmod_ifr2(inc, v, phi) -> np.ndarray:
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
