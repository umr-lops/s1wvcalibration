"""Effective (2D-ellipsoid-cutoff) significant wave height from S-1 and WW3.

Ported from ``qualitycheck/get_hs_effective_2D_elipsoid_cutoff.py``. The helper
modules (2D cutoff, low-frequency mask, polar interpolation, Stopa acCalc) now
live as relative imports inside ``s1wvcalibration._legacy``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from .hs_2d_cutoff import get_WV_2D_cutoff
from .interpolation_polar import interp_polar_spectrum

logger = logging.getLogger(__name__)

__all__ = ["get_S1_hs_eff_2D_cutoff", "get_ww3_hs_eff_2D_cutoff"]


def get_S1_hs_eff_2D_cutoff(
    file_handler: Any, lowfreqfilter: bool = True, plot_dev: bool = False
) -> tuple[float, float, float, float, float, float]:
    """S-1 effective Hs using the 2D ellipsoid cutoff.

    Args:
        file_handler: netCDF4 dataset of a S-1 L2 OCN WV product.
        lowfreqfilter: remove the low-frequency (partition -1) bins.
        plot_dev: kept for signature compatibility.

    Returns:
        ``(swh_eff_2d, swh_azc, wavelength_peak, direction_peak,
        wavelength_peak_eff, direction_peak_eff)``.
    """
    logger.debug("lowfreqfilter %s", lowfreqfilter)
    s1a_spec = file_handler.variables["oswPolSpec"][:].squeeze()
    k_sar = file_handler.variables["oswK"][:]
    toto_ksar = copy.copy(k_sar)
    logger.debug("k_sar shape: %s", k_sar.shape)
    az_cutoff = file_handler.variables["oswAzCutoff"][0][0]
    if isinstance(az_cutoff, np.ma.core.MaskedConstant) or isinstance(
        s1a_spec, np.float64
    ):
        logger.debug("AzCutOf is masked")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    mask = s1a_spec * 0.0
    mask_eff, _, phi_sar = get_WV_2D_cutoff(
        file_handler, lowfreqfilter=lowfreqfilter, plot_dev=plot_dev
    )

    dk_sar = np.ma.zeros(k_sar.size)
    dk_sar[0] = toto_ksar[1] - toto_ksar[0]
    dk_sar[-1] = toto_ksar[-1] - toto_ksar[-2]
    dk_sar[1:-1] = (toto_ksar[2:] - toto_ksar[0:-2]) / 2
    dphi_sar = np.array(abs(phi_sar[1] - phi_sar[0]) * np.pi / 180.0)
    kdphi_sar = np.array(np.tile(toto_ksar * dphi_sar, (phi_sar.size, 1)))
    kdkdphi_sar = np.array(np.tile(toto_ksar * dk_sar * dphi_sar, (phi_sar.size, 1)))
    spk_ocn = np.sum(s1a_spec * mask * kdphi_sar, 0)
    spk_ocn_eff = np.sum(s1a_spec * mask_eff * kdphi_sar, 0)

    kdk_sar = np.array(np.tile(toto_ksar * dk_sar, (phi_sar.size, 1)))
    spp_ocn = np.sum(s1a_spec * kdk_sar, 1)
    spp_ocn_eff = np.sum(s1a_spec * mask_eff * kdk_sar, 1)

    swh_azc_s1a = 4 * np.sqrt(np.sum(s1a_spec * kdkdphi_sar * mask))
    swh_eff_2d_s1 = 4 * np.sqrt(np.sum(s1a_spec * kdkdphi_sar * mask_eff))
    logger.debug("np.nanmax(spk_ocn) = %s", np.nanmax(spk_ocn))

    if not np.isnan(np.max(spk_ocn)):
        _ind = np.where(spk_ocn == np.max(spk_ocn))[0][0]
        ocnwavelengthpeak = 2 * np.pi / toto_ksar[_ind]
        if isinstance(ocnwavelengthpeak, np.ma.core.MaskedConstant):
            ocnwavelengthpeak = np.nan
    else:
        ocnwavelengthpeak = np.nan

    if not np.isnan(np.nanmax(spp_ocn)):
        _ind = np.where(spp_ocn == np.max(spp_ocn))[0][0]
        ocndirectionpeak = phi_sar[_ind]
        if isinstance(ocndirectionpeak, np.ma.core.MaskedConstant):
            ocndirectionpeak = np.nan
    else:
        ocndirectionpeak = np.nan

    if not np.isnan(np.max(spk_ocn_eff)):
        _ind = np.where(spk_ocn_eff == np.max(spk_ocn_eff))[0][0]
        ocnwavelengthpeak_eff = 2 * np.pi / toto_ksar[_ind]
        if isinstance(ocnwavelengthpeak_eff, np.ma.core.MaskedConstant):
            ocnwavelengthpeak_eff = np.nan
    else:
        ocnwavelengthpeak_eff = np.nan

    if not np.isnan(np.max(spp_ocn_eff)):
        _ind = np.where(spp_ocn_eff == np.max(spp_ocn_eff))[0][0]
        ocndirectionpeak_eff = phi_sar[_ind]
    else:
        ocndirectionpeak_eff = np.nan

    return (
        swh_eff_2d_s1,
        swh_azc_s1a,
        ocnwavelengthpeak,
        ocndirectionpeak,
        ocnwavelengthpeak_eff,
        ocndirectionpeak_eff,
    )


def get_ww3_hs_eff_2D_cutoff(
    file_handler_s1: Any,
    file_handler_ww3: Any,
    osw: Any = None,  # noqa: ARG001 - legacy signature
) -> tuple[float, ...]:
    """WW3 spectrum regridded on the S-1 grid and masked with the S-1 2D cutoff.

    Returns:
        ``(swh_eff_2d_ww3, swh_ww3, ww3_wavelength_peak, ww3_wavelength_peak_eff,
        ww3_direction_peak, ww3_direction_peak_eff, ww3_wl_partSAR1_weighted)``.
    """
    k_eff = 2 * np.pi / file_handler_s1.variables["oswSpecRes"][:].squeeze()
    k_sar = file_handler_s1.variables["oswK"][:]
    phi_sar = np.array(file_handler_s1.variables["oswPhi"][:].squeeze())
    k_cutoff = 2 * np.pi / file_handler_s1.variables["oswAzCutoff"][0][0]
    s1a_spec = file_handler_s1.variables["oswPolSpec"][:].squeeze()
    partoch = file_handler_s1.variables["oswPartitions"][:].squeeze()

    if isinstance(k_cutoff, np.ma.core.MaskedConstant):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    mask = s1a_spec * 0.0
    toto_ksar = copy.copy(k_sar)
    k_sar_2D = np.array(np.tile(toto_ksar, (phi_sar.size, 1)))
    mask[k_sar_2D < k_cutoff] = 1

    ww3_spec = file_handler_ww3.variables["polSpec"][:]
    k_ww3 = file_handler_ww3.variables["k"][:]
    phi_ww3 = file_handler_ww3.variables["phi"][:]
    ww3_spec_interp = interp_polar_spectrum(
        toto_ksar, phi_sar, ww3_spec, k_ww3, phi_ww3
    )

    k_eff_2D = np.array((np.tile(k_eff, (toto_ksar.size, 1))).T)
    mask_eff = s1a_spec * 0.0
    mask_eff[k_sar_2D < k_eff_2D] = 1

    dk_sar = np.ma.zeros(k_sar.size)
    logger.debug("k_sar %s dk_sar %s ", type(k_sar), type(dk_sar))
    dk_sar[0] = k_sar[1] - k_sar[0]
    dk_sar[-1] = k_sar[-1] - k_sar[-2]
    dk_sar[1:-1] = (k_sar[2:] - k_sar[0:-2]) / 2

    kdk_sar = np.array(np.tile(k_sar * dk_sar, (phi_sar.size, 1)))
    dphi_sar = np.array(abs(phi_sar[1] - phi_sar[0]) * np.pi / 180.0)
    kdphi_sar = np.array(np.tile(k_sar * dphi_sar, (phi_sar.size, 1)))
    kdkdphi_sar = np.array(np.tile(k_sar * dk_sar * dphi_sar, (phi_sar.size, 1)))

    spk_ww3 = np.sum(ww3_spec_interp * mask * kdphi_sar, 0)
    spk_ww3_eff = np.sum(ww3_spec_interp * mask_eff * kdphi_sar, 0)
    spp_ww3 = np.sum(ww3_spec_interp * mask * kdk_sar, 1)
    spp_ww3_eff = np.sum(ww3_spec_interp * mask_eff * kdk_sar, 1)
    swh_ww3 = 4 * np.sqrt(np.sum(ww3_spec_interp * kdkdphi_sar))
    swh_eff_2d_ww3 = 4 * np.sqrt(np.sum(ww3_spec_interp * kdkdphi_sar * mask_eff))

    if not np.isnan(np.max(spk_ww3)):
        _ind = np.where(spk_ww3 == np.nanmax(spk_ww3))[0][0]
        ww3wavelengthpeak = 2 * np.pi / k_sar[_ind]
    else:
        ww3wavelengthpeak = np.nan
    logger.debug("spp_ww3 = %s", spp_ww3)
    logger.debug("np.nanmax(spp_ww3) = %s", np.nanmax(spp_ww3))
    # Legacy behaviour preserved verbatim: ``nanmax(x) != nan`` is always True
    # (even when the value is NaN), so ``ww3directionpeak`` is always NaN. Do not
    # "fix" this to ``np.isnan`` -- it would flip the (parity-validated) output.
    if np.nanmax(spp_ww3) != np.nan:  # noqa: PLW0177 - see note above
        ww3directionpeak = np.nan
    else:
        _ind = np.where(spp_ww3 == np.nanmax(spp_ww3))[0][0]
        ww3directionpeak = phi_sar[_ind]

    if not np.isnan(np.max(spk_ww3_eff)):
        _ind = np.where(spk_ww3_eff == np.nanmax(spk_ww3_eff))[0][0]
        ww3wavelengthpeak_eff = 2 * np.pi / k_sar[_ind]
    else:
        ww3wavelengthpeak_eff = np.nan

    energy_not_sumed = ww3_spec_interp * mask_eff * kdphi_sar
    maskpartition = partoch != 1
    energy_not_sumed[maskpartition] = 0
    denominator = np.sum(energy_not_sumed)
    ww3wl_partSAR1_weightened = np.divide(
        np.sum(energy_not_sumed * 2 * np.pi / k_sar),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )

    logger.debug("spp_ww3_eff = %s", spp_ww3_eff)
    logger.debug("np.nanmax(spp_ww3_eff) = %s", np.nanmax(spp_ww3_eff))
    if not np.isnan(np.max(spp_ww3_eff)):
        _ind = np.where(spp_ww3_eff == np.nanmax(spp_ww3_eff))[0][0]
        ww3directionpeak_eff = phi_sar[_ind]
    else:
        ww3directionpeak_eff = np.nan

    return (
        swh_eff_2d_ww3,
        swh_ww3,
        ww3wavelengthpeak,
        ww3wavelengthpeak_eff,
        ww3directionpeak,
        ww3directionpeak_eff,
        ww3wl_partSAR1_weightened,
    )
