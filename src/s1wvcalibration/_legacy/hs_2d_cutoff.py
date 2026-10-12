"""2D (ellipsoid) azimuth/range cutoff mask for S-1 WV spectra.

Ported from ``qualitycheck/compute_2D_cutoff.py``. The legacy
``reference_oswk`` import is dropped: the reference wavenumber array is always
overwritten by the one read from the netCDF handler.
"""

from __future__ import annotations

import logging

import numpy as np

from .low_frequency_energy import get_low_frequencies_mask

logger = logging.getLogger(__name__)

__all__ = ["get_WV_2D_cutoff"]


def get_WV_2D_cutoff(handlerNC, lowfreqfilter: bool = True, plot_dev: bool = False):
    """Return the mask to filter out the SAR azimuth/range cut-off.

    Args:
        handlerNC: netCDF4 dataset (S-1 L2 OCN WV).
        lowfreqfilter: remove the low-frequency (partition -1) bins.
        plot_dev: (unused, kept for signature compatibility)

    Returns:
        ``(mask_eff, k_sar, phi_sar)``. ``mask_eff`` is 1 where the spectral
        value can be kept.
    """
    mask_eff = None
    logger.debug("enter get_WV_2D_cutoff lowfreqfilter %s plot_dev %s", lowfreqfilter, plot_dev)
    s1a_spec = handlerNC.variables["oswPolSpec"][:].squeeze()
    k_sar = handlerNC.variables["oswK"][:]
    k_cutoff = 2 * np.pi / handlerNC.variables["oswAzCutoff"][0][0]
    k_eff = 2 * np.pi / handlerNC.variables["oswSpecRes"][:].squeeze()
    if isinstance(k_eff, np.ma.core.MaskedConstant):
        logger.info("compute_2D_cutoff|corrupted file containing oswAngularBinSize dimension=1")
    else:
        k_eff_2D = np.array((np.tile(k_eff, (k_sar.size, 1))).T)
        mask = s1a_spec * 0.0
        phi_sar = np.array(handlerNC.variables["oswPhi"][:].squeeze())
        logger.debug("phi_sar.size = %s", phi_sar.size)
        k_sar_2D = np.ma.array(np.tile(k_sar, (phi_sar.size, 1)))
        mask[k_sar_2D < k_cutoff] = 1
        mask_eff = s1a_spec * 0.0
        mask_eff[k_sar_2D < k_eff_2D] = 1
        if lowfreqfilter:
            mask_low_freq = np.ones_like(mask_eff)
            filterout_low_freq_condition = get_low_frequencies_mask(handlerNC)
            logger.debug(
                "any bins with low frequencies??? %s (%s %s%%)",
                filterout_low_freq_condition.any(),
                np.sum(filterout_low_freq_condition),
                np.sum(filterout_low_freq_condition) * 100 / filterout_low_freq_condition.size,
            )
            mask_low_freq[filterout_low_freq_condition] = 0
            mask_eff *= mask_low_freq
    return mask_eff, k_sar, phi_sar
