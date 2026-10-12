"""Read a S-1 WV SLC imagette with ``xsar`` (replaces cerbere/SARImage).

This is a drop-in replacement for the legacy ``level == 'L1'`` block that used
``cerbere.mapper.safegeotifffile.SAFEGeoTiffFile`` and ``sarimage.SARImage``. It
returns the same quantities that the legacy code extracted, so that the
surrounding calibration logic can stay unchanged.

Numeric parity (vs the production NetCDF produced by the legacy mapper) is
verified by :mod:`tests.test_l1_parity` to float32 precision for every
calibration-relevant quantity:

* ``sigma0_matrix``      = xsar ``sigma0_raw`` (legacy ``sigma0``)
* ``sigma0_denoised``    = xsar ``sigma0`` (legacy ``sigma0_corrected``)
* ``noise_linear``       = xsar ``nesz`` (legacy ``noise_linear_equivalent``)
* ``lon/lat/inc/elev``   = full-res (spline-interpolated) grid, centre pixel --
  identical to the legacy ``SARImage.get_data(..., extent=center)``.
* ``track_angle``        = ``meta.platform_heading`` (legacy ``get_info``).
* ``I``                  = ``|digital_number|^2`` (feeds nv/skew/kurt).
* ``doppler_centroid``   = mean over the image of the Doppler-Centroid estimate
  (``frequency``) spline-interpolated over (azimuth, slant-range), exactly as
  ``cerbere.SAFEGeoTiffFile.read_values('doppler_centroid')`` does.

Notes
-----
* For **SLC** (incl. WV) xsar does *not* auto-compute the calibration and the
  high-resolution variables (only GRD does), so this module requests them
  explicitly: ``add_high_resolution_variables`` then
  ``apply_calibration_and_denoising``.
* The measurement tiff name embeds the imagette index, which is turned into the
  xsar multi-dataset id ``SENTINEL1_DS:<SAFE>:WV_NNN``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from scipy import interpolate
from xsar import Sentinel1Dataset, Sentinel1Meta

logger = logging.getLogger(__name__)

__all__ = ["read_slc_imagette"]


def _wv_subdataset_id(measu_path: str) -> str:
    """Return the xsar dataset id of one WV SLC measurement.

    A WV SLC SAFE is a *multidataset* of ``WV_001``..``WV_NNN`` subdatasets, and
    the tiff filename embeds the imagette index (e.g.
    ``s1d-wv2-slc-vv-...-00378e-002.tiff`` -> ``WV_002``). The legacy cerbere
    mapper opened the tiff directly, xsar needs the ``SENTINEL1_DS:<SAFE>:WV_NNN``
    identifier instead.
    """
    measu = Path(measu_path)
    base = measu.name
    match = re.search(r"-(\d+)\.tiff$", base)
    if match is None:
        raise ValueError(f"cannot extract imagette index from measurement name: {base}")
    safe = str(measu.parent.parent.resolve())
    return f"SENTINEL1_DS:{safe}:WV_{int(match.group(1)):03d}"


def _open_calibrated(measu_path: str):
    """Open the imagette with calibration, denoising and HR geoloc computed."""
    dsid = _wv_subdataset_id(measu_path)
    meta = Sentinel1Meta(dsid)
    sar = Sentinel1Dataset(dsid)
    # SLC (incl. WV): these are not auto-computed (GRD only), request explicitly.
    sar.add_high_resolution_variables(patch_variable=True, luts=False, lazy_loading=True)
    sar.apply_calibration_and_denoising()
    return meta, sar


def _doppler_from_meta(meta, data):
    """Mean Doppler-centroid estimate and its quality flag.

    Mirrors ``cerbere.mapper.safegeotiff.File.read_values('doppler_centroid')``:
    the per-(azimuth, slant-range) ``frequency`` DC estimates are
    spline-interpolated onto the full image grid and averaged.
    """
    try:
        dop = meta._doppler_estimate
    except Exception as exc:
        logger.debug("no doppler estimate available: %s", exc)
        return float("nan"), float("nan")

    freq = np.asarray(dop["frequency"].values, dtype=np.float64)      # (n_dce, n_srt)
    srt = np.asarray(dop["slantRangeTime"].values, dtype=np.float64)
    base = np.asarray(dop["azimuthTime"].values, dtype="datetime64[s]")[0]
    t_stop = np.asarray(dop["fineDceAzimuthStopTime"].values, dtype="datetime64[s]")[0]

    # cerbere builds a 2-point azimuth axis when there is a single DC estimate.
    x_val = np.array([0.0, (t_stop - base) / np.timedelta64(1, "s")])
    val = np.repeat(freq, 2, axis=0)
    y_val = srt[0, :]
    func = interpolate.RectBivariateSpline(
        x_val, y_val, val, kx=min(x_val.size - 1, 3), ky=min(y_val.size - 1, 3)
    )

    az = np.asarray(data["azimuth_time"].values, dtype="datetime64[s]")
    sl = np.asarray(data["slant_range_time"].values, dtype=np.float64)
    n_line, n_samp = az.shape
    x_int = ((az - base) / np.timedelta64(1, "s"))[:, n_samp // 2 : n_samp // 2 + 1]
    y_int = sl[n_line // 2 : n_line // 2 + 1, :]
    mean = float(np.mean(func(x_int, y_int)))

    quality = int(bool(np.asarray(dop["dataDcRmsErrorAboveThreshold"].values).ravel()[-1]))
    return mean, quality


def read_slc_imagette(measu_path: str) -> dict:
    """Read one S-1 WV SLC measurement and return the calibration quantities.

    Args:
        measu_path: full path of the measurement tiff (one WV imagette).

    Returns:
        A dict with keys: ``lon``, ``lat``, ``inc``, ``elev``, ``noise_linear``,
        ``sigma0_matrix``, ``sigma0_denoised_matrix``, ``doppler_centroid_mean``,
        ``doppler_centroid_quality``, ``track_angle`` and ``I`` (``|DN|^2``).
    """
    meta, sar = _open_calibrated(measu_path)
    data = sar.dataset

    n_line = data["digital_number"].sizes["line"]
    n_samp = data["digital_number"].sizes["sample"]
    cl, cs = n_line // 2, n_samp // 2

    # Centre of the full-res (interpolated) grid == legacy SARImage extent centre.
    lon = float(data["longitude"].isel(line=cl, sample=cs))
    lat = float(data["latitude"].isel(line=cl, sample=cs))
    inc = float(data["incidence"].isel(line=cl, sample=cs))
    elev = float(data["elevation"].isel(line=cl, sample=cs))

    sigma0_matrix = np.asarray(data["sigma0_raw"].isel(pol=0).values, dtype=np.float64)
    sigma0_denoised = np.asarray(data["sigma0"].isel(pol=0).values, dtype=np.float64)
    noise_linear = np.asarray(data["nesz"].isel(pol=0).values, dtype=np.float64)

    dn = np.asarray(data["digital_number"].isel(pol=0).values)
    i_sq = np.abs(dn) ** 2

    doppler_mean, doppler_quality = _doppler_from_meta(meta, data)
    track_angle = float(meta.platform_heading)

    return {
        "lon": lon,
        "lat": lat,
        "inc": inc,
        "elev": elev,
        "noise_linear": noise_linear,
        "sigma0_matrix": sigma0_matrix,
        "sigma0_denoised_matrix": sigma0_denoised,
        "doppler_centroid_mean": doppler_mean,
        "doppler_centroid_quality": doppler_quality,
        "track_angle": track_angle,
        "I": i_sq,
    }
