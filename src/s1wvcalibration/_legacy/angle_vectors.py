"""Angle-difference helpers (ported from qualitycheck/angle_between_2Ndimensial_vectors.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .small_methods import decode_ecmwf_uv_if_necessary

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


def unit_vector(vector: NDArray) -> NDArray:
    """Return the unit vector of *vector*."""
    return vector / np.linalg.norm(vector)


def angle_between(v1: NDArray, v2: NDArray) -> float:
    """Return the angle in radians between vectors *v1* and *v2*."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return float(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def wave_direction_comparison_s1_ww3(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``diff_s1_ww3_wave_dir_peak_deg`` to the calval index dataframe."""
    mdecoded_dir_s1 = df["s1_peak_wave_direction_2Dcutoff"].apply(decode_ecmwf_uv_if_necessary)
    mdecoded_dir_ww3 = df["ww3wavedirpeak_2Dcutoff"].apply(decode_ecmwf_uv_if_necessary)
    u_component_s1 = np.cos(np.radians(mdecoded_dir_s1))
    v_component_s1 = np.sin(np.radians(mdecoded_dir_s1))
    u_component_ww3 = np.cos(np.radians(mdecoded_dir_ww3))
    v_component_ww3 = np.sin(np.radians(mdecoded_dir_ww3))
    vec_s1 = np.array([(u, v) for u, v in zip(u_component_s1, v_component_s1, strict=False)])
    vec_ww3 = np.array([(u, v) for u, v in zip(u_component_ww3, v_component_ww3, strict=False)])
    diff_rad = [angle_between(vec_s1[i], vec_ww3[i]) for i in range(len(vec_s1))]
    diff_deg = np.degrees(diff_rad)
    df["diff_s1_ww3_wave_dir_peak_deg"] = diff_deg
    return df


def wind_direction_comparison_s1_ecmwf(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``diff_s1_ecmwf_wind_dir_rad`` to the calval index dataframe."""
    df["EcmwfWindDirection2"] = np.mod(90.0 - df["ecmwf_wind_direction"], 360)
    diff = wind_direction_comparison_core(
        wdir1=df.oswWindDirection.values, wdir2=df["EcmwfWindDirection2"].values
    )
    df["diff_s1_ecmwf_wind_dir_deg"] = diff
    return df


def wind_direction_comparison_core(wdir1: NDArray, wdir2: NDArray, trigotrick: bool = False) -> NDArray:
    """Angle difference (deg) between two direction vectors (0..pi)."""
    u1 = np.cos(np.radians(wdir1))
    v1 = np.sin(np.radians(wdir1))
    u2 = np.cos(np.radians(wdir2))
    v2 = np.sin(np.radians(wdir2))
    vec1 = np.array([(u, v) for u, v in zip(u1, v1, strict=False)])
    vec2 = np.array([(u, v) for u, v in zip(u2, v2, strict=False)])
    diff = [angle_between(vec1[i], vec2[i]) for i in range(len(vec1))]
    return 90.0 - np.degrees(diff) if trigotrick else np.degrees(diff)
