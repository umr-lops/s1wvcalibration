"""Low-frequency mask helpers (subset of qualitycheck/compute_low_frequency_energy.py).

Only the symbols used by the 2D-cutoff chain are ported.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = ["get_low_frequencies_mask", "get_low_frequencies_mask_core"]


def get_low_frequencies_mask_core(partitions: NDArray) -> NDArray[np.bool_]:
    """Boolean mask of low-frequency bins (partition value -1)."""
    return partitions == -1


def get_low_frequencies_mask(handler: Any) -> NDArray[np.bool_]:
    """Return the 2D (phi, k) low-frequency mask from an xarray Dataset."""
    partitions = np.asarray(handler["oswPartitions"].values).squeeze()
    mat_low = get_low_frequencies_mask_core(partitions)
    logger.debug("any low-frequency bins: %s", bool(mat_low.any()))
    return mat_low
