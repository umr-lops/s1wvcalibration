"""Angle formatting helpers (ported from qualitycheck/format_angle.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import mod

if TYPE_CHECKING:
    from numpy.typing import NDArray


def format_angle(angle: float | NDArray, compass: bool = False) -> Any:
    """Wrap *angle* (deg) to [-180, 180] or [0, 360] if ``compass``."""
    theta = mod(angle, 360.0)
    try:
        # array-like input
        idx = np.where(theta < -180)
        if len(idx) > 0:
            theta[idx] += 360
        idx = np.where(theta > 180)
        if len(idx) > 0:
            theta[idx] -= 360
        if compass:
            idx = np.where(theta < 0)
            if len(idx) > 0:
                theta[idx] += 360
    except (IndexError, TypeError):
        if theta < -180:
            theta += 360
        if theta > 180:
            theta -= 360
        if compass and theta < 0:
            theta += 360
    return theta


def format_angle_v2(
    angle: float | NDArray, cycle: float = 360, compass_format: bool = False
) -> Any:
    """Wrap *angle* (deg) to [-cycle/2, cycle/2] or [0, cycle] if compass_format."""
    theta = np.mod(angle, cycle)
    try:
        theta[theta <= cycle / 2] += cycle
        theta[theta > cycle / 2] -= cycle
        if compass_format:
            theta[theta < 0] += cycle
    except TypeError:
        if theta <= cycle / 2:
            theta += cycle
        if theta > cycle / 2:
            theta -= cycle
        if compass_format and theta < 0:
            theta += cycle
    return theta
