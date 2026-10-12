"""Legacy CMOD5_N forward model — exact port of the production implementation.

This is a faithful, dependency-free (numpy only) port of
``calvalsentinel1/.../qualitycheck/cmod5n.py::cmod5n_forward`` (the version that
produces the operational ``sig_mod`` since June 2019). It is kept *as is* so that
the new library stays numerically identical to the current production.

.. warning:: batch "correction vector agrouaze June 2019"
   Unlike the standard CMOD5_N (e.g. ``xsarsea`` ``gmf_cmod5n``) which applies the
   low-wind ``S -> S0`` clamp **element-wise**, this legacy version applies it
   **per-array**: if *any* imagette of the batch has ``S < S0``, the *whole*
   ``S_vec`` is set to ``S0`` and the ``(S/S0)**(S0*(1-A3))`` factor is then applied
   to *every* element. The same batch behaviour is used for the B2 ``V2`` clamp.
   Do NOT "fix" this to element-wise: it is what the production ``sig_mod`` is.
   (The standard element-wise model is available as :func:`cmod5n_xsarsea` in
   :mod:`s1wvcalibration._legacy.gmf`; the two differ by up to ~1.4 dB on the
   classical 7 m/s sweet spot, mostly in the WV2 / higher-incidence region.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["cmod5n_legacy"]


def cmod5n_legacy(v: NDArray, phi: NDArray, theta: NDArray) -> NDArray:
    """Legacy CMOD5_N (batch-corrected) linear backscatter.

    Args:
        v: wind speed [m/s].
        phi: wind azimuth relative to antenna [deg].
        theta: incidence angle [deg].

    Returns:
        Linear CMOD5_N backscatter (``numpy.ndarray``), same semantics as the
        historical production ``cmod5n_forward``.
    """
    v = np.asarray(v, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    DTOR = 57.29577951
    THETM = 40.0
    THETHR = 25.0
    ZPOW = 1.6

    # NB: leading 0 keeps 1-based Fortran indexing.
    C = [
        0, -0.6878, -0.7957, 0.3380, -0.1728, 0.0000, 0.0040, 0.1103, 0.0159,
        6.7329, 2.7713, -2.2885, 0.4971, -0.7250, 0.0450,
        0.0066, 0.3222, 0.0120, 22.7000, 2.0813, 3.0000, 8.3659,
        -3.3428, 1.3236, 6.2437, 2.3893, 0.3249, 4.1590, 1.6930,
    ]

    Y0 = C[19]
    PN = C[20]
    A = C[19] - (C[19] - 1) / C[20]
    B = 1.0 / (C[20] * (C[19] - 1.0) ** (3 - 1))

    # Angles
    FI = phi / DTOR
    CSFI = np.cos(FI)
    CS2FI = 2.0 * CSFI * CSFI - 1.0

    X = (theta - THETM) / THETHR
    XX = X * X

    # B0: function of wind speed and incidence angle
    A0 = C[1] + C[2] * X + C[3] * XX + C[4] * X * XX
    A1 = C[5] + C[6] * X
    A2 = C[7] + C[8] * X
    GAM = C[9] + C[10] * X + C[11] * XX
    S0 = C[12] + C[13] * X

    S = A2 * v

    # "correction vector agrouaze June 2019": batch clamp of S onto S0.
    S_vec = S
    if (S_vec < S0).any():
        S_vec = S0
    A3 = 1.0 / (1.0 + np.exp(-S_vec))
    if (S < S0).any():
        A3 = A3 * (S / S0) ** (S0 * (1.0 - A3))
    B0 = (A3 ** GAM) * 10.0 ** (A0 + A1 * v)

    # B1: function of wind speed and incidence angle
    B1 = C[15] * v * (0.5 + X - np.tanh(4.0 * (X + C[16] + C[17] * v)))
    B1 = C[14] * (1.0 + X) - B1
    B1 = B1 / (np.exp(0.34 * (v - C[18])) + 1.0)

    # B2: function of wind speed and incidence angle
    V0 = C[21] + C[22] * X + C[23] * XX
    D1 = C[24] + C[25] * X + C[26] * XX
    D2 = C[27] + C[28] * X
    V2 = v / V0 + 1.0
    if (V2 < Y0).any():
        V2 = A + B * (V2 - 1.0) ** PN
    B2 = (-D1 + D2 * V2) * np.exp(-V2)

    # Combine the three Fourier terms.
    return B0 * (1.0 + B1 * CSFI + B2 * CS2FI) ** ZPOW
