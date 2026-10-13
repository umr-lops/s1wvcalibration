"""IFREMER CMOD-IFR2 (and CMOD2-I3) geophysical models.

Ported verbatim from ``qualitycheck/cmod_ifremer.py`` (cleaned).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import arange, cos, pi, sqrt, tanh

if TYPE_CHECKING:
    from numpy.typing import NDArray


def cmod_ifr2(T: NDArray, wind: NDArray, dir: NDArray) -> NDArray:
    """CMOD-IFR2 backscatter (linear) for incidence *T*, wind *wind*, azimuth *dir*.

    Args:
        T: incidence angle (deg).
        wind: wind speed (m/s).
        dir: angular direction, 0 is the central antenna (deg).
    """
    angsz = len(T)
    arange(angsz, dtype="f4")

    C = arange(26, dtype="f4")
    C[0] = 0.0
    C[1] = -2.437597
    C[2] = -1.5670307
    C[3] = 0.3708242
    C[4] = -0.040590
    C[5] = 0.404678
    C[6] = 0.188397
    C[7] = -0.027262
    C[8] = 0.064650
    C[9] = 0.054500
    C[10] = 0.086350
    C[11] = 0.055100
    C[12] = -0.058450
    C[13] = -0.096100
    C[14] = 0.412754
    C[15] = 0.121785
    C[16] = -0.024333
    C[17] = 0.072163
    C[18] = -0.062954
    C[19] = 0.015958
    C[20] = -0.069514
    C[21] = -0.062945
    C[22] = 0.035538
    C[23] = 0.023049
    C[24] = 0.074654
    C[25] = -0.014713

    def funcb1(Cc: NDArray, x: NDArray, V: NDArray) -> NDArray:
        tetanor = (2.0 * x - (18.0 + 58.0)) / (58.0 - 18.0)
        vitnor = (2.0 * V - (25.0 + 3.0)) / (25.0 - 3.0)
        pv1 = vitnor
        pv2 = 2 * vitnor * pv1 - 1.0
        2 * vitnor * pv2 - pv1
        pt1 = tetanor
        pt2 = 2 * tetanor * pt1 - 1.0
        return (
            Cc[8]
            + Cc[9] * pv1
            + (Cc[10] + Cc[11] * pv1) * pt1
            + (Cc[12] + Cc[13] * pv1) * pt2
        )

    def funcb2(Cc: NDArray, x: NDArray, V: NDArray) -> NDArray:
        tetanor = (2.0 * x - (18.0 + 58.0)) / (58.0 - 18.0)
        vitnor = (2.0 * V - (25.0 + 3.0)) / (25.0 - 3.0)
        pv1 = vitnor
        pv2 = 2 * vitnor * pv1 - 1.0
        pv3 = 2 * vitnor * pv2 - pv1
        pt1 = tetanor
        pt2 = 2 * tetanor * pt1 - 1.0
        return (
            Cc[14]
            + Cc[15] * pt1
            + Cc[16] * pt2
            + (Cc[17] + Cc[18] * pt1 + Cc[19] * pt2) * pv1
            + (Cc[20] + Cc[21] * pt1 + Cc[22] * pt2) * pv2
            + (Cc[23] + Cc[24] * pt1 + Cc[25] * pt2) * pv3
        )

    v = wind
    tetai = (T - 36.0) / 19.0
    xSQ = tetai * tetai
    P1 = tetai
    P2 = (3.0 * xSQ - 1.0) / 2.0
    P3 = (5.0 * xSQ - 3.0) * tetai / 2.0
    ALPHA = C[1] + C[2] * P1 + C[3] * P2 + C[4] * P3
    BETA = C[5] + C[6] * P1 + C[7] * P2
    ang = dir
    cosi = cos(ang * pi / 180.0)
    cos2i = 2.0 * cosi * cosi - 1.0
    b1 = funcb1(C, T, v)
    b2 = funcb2(C, T, v)

    vbiais = 0.0
    SPD_Wind = v - vbiais
    b0 = pow(10.0, (ALPHA + BETA * sqrt(SPD_Wind)))
    return b0 * (1.0 + b1 * cosi + tanh(b2) * cos2i)


def cmod_ifremer_matrix(
    T: NDArray, wind: NDArray, dir: NDArray, model: str = "CMOD-IFR2"
) -> NDArray:
    """Dispatch to :func:`cmod_ifr2` for the *model* name."""
    if model == "CMOD-IFR2":
        return cmod_ifr2(T, wind, dir)
    raise ValueError(f"unknown model {model}")
