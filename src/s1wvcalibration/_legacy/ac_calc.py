"""Theoretical S-1 azimuth cutoff (Justin Stopa ``acCalc`` model).

Ported from ``qualitycheck/theorical_azimuth_cutoff_justinStopa_acCalc.py``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["acCalc"]


def acCalc(s, f, th, ia, ta, hv):
    """Compute the azimuth cutoff (m) from a wave spectrum.

    Args:
        s: spectra [nf, nth, ntime] in m^2/Hz/rad.
        f: spectrum frequency (length nf).
        th: spectrum direction in radians (length nth), from due north.
        ia: incidence angle in radians (length ntime).
        ta: track angle in radians (length ntime), from due north.
        hv: slant range-to-velocity ratio of the satellite.
    """
    mF = 5  # maximum frequency HZ
    nf, nth, ts = np.shape(s)
    df = np.zeros(len(f))
    df[0] = f[1] - f[0]
    df[-1] = f[-1] - f[-2]
    df[1:-1] = (f[2:] - f[0:-2]) / 2.0

    dth = np.ones((nf, nth)) * np.abs(th[1] - th[0])
    dfdth = np.dot(df.reshape(-1, 1), np.ones((1, nth)) * np.abs(th[1] - th[0]))

    vg = np.ones((ts, 1))  # m2 with geometry
    vk = np.ones((ts, 1))  # m2 using Kerbaol
    for ij in np.arange(ts):
        th = np.tile(th.reshape(1, -1), (nf, 1))
        u_var1 = np.sum(np.cos(th) ** 2 * dth * s[:, :, ij], axis=1)
        v_var1 = np.sum(np.sin(th) ** 2 * dth * s[:, :, ij], axis=1)
        uv_var1 = np.sum(np.sin(th) * np.cos(th) * dth * s[:, :, ij], axis=1)
        w_var1 = np.sum(dth * s[:, :, ij], axis=1)

        u_var = add_spc_tail(u_var1, f, 2, mF)
        v_var = add_spc_tail(v_var1, f, 2, mF)
        uv_var = add_spc_tail(uv_var1, f, 2, mF)
        w_var = add_spc_tail(w_var1, f, 2, mF)

        vg[ij] = (
            w_var * (np.cos(ia[ij])) ** 2
            + (
                u_var * (np.cos(ta[ij])) ** 2
                + v_var * (np.sin(ta[ij])) ** 2
                - 2 * uv_var * np.cos(ta[ij]) * np.sin(ta[ij])
            )
            * (np.sin(ia[ij])) ** 2
        )
        vk[ij] = np.sum(dfdth * s[:, :, ij] * np.tile(f.reshape(-1, 1) ** 2, (1, nth)))

    return (2 * np.pi**2 * hv) * np.sqrt(vk)  # azimuth-cutoff (m)


def add_spc_tail(f_arr, f, m, mF):
    """Add a f^-5 high-frequency tail to spectrum moment (from Justin Stopa)."""
    tailpower = 5
    ddf = 0.01

    df = np.zeros(len(f))
    df[0] = f[1] - f[0]
    df[-1] = f[-1] - f[-2]
    df[1:-1] = (f[2:] - f[0:-2]) / 2.0

    eps = np.finfo(float).eps
    idd = np.where(np.abs(f_arr) < eps)
    idd = len(f) - 1 if idd[0].size == 0 else np.amin(idd) - 1
    ff = np.arange(1, np.round(mF / ddf)) * ddf + f[-1]
    f_arr = np.append(f_arr[1:idd], [f_arr[idd] * (f[idd] / ff) ** tailpower])
    dff = np.append(df[1:idd], np.tile(ddf, (len(ff), 1)))
    ff = np.append(f[1:idd], ff)
    return np.sum(ff**m * dff * f_arr)
