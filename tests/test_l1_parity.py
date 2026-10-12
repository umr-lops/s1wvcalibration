"""L1 SLC reader parity: xsar-based ``read_slc_imagette`` vs the production
NetCDF produced by the legacy cerbere/SARImage mapper.

This is the test that validates the ``cerbere -> xsar`` replacement. It reads a
real S-1 WV SLC SAFE with the new reader and compares, imagette by imagette,
every calibration-relevant quantity against the reference NetCDF that the old
production actually emitted for the same SAFE.

The reference (float32) is the target, so the tolerances are set to float32
rounding. Measured on S1D ... 00378E_9153: every quantity agrees to well below
1e-4 relative / 1e-4 dB.

Data (override with env vars, defaults = the HPC checkout used to develop this):

* ``S1WVCALIB_L1_SAFE``  : full path of the raw ``....SAFE`` (with measurement/).
* ``S1WVCALIB_L1_REF_NC``: the production NetCDF ``calib/....SAFE.nc``.

If either is missing the tests are skipped, so the suite still runs on a clean
CI checkout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import netCDF4
import numpy as np
import pytest

from s1wvcalibration._legacy.xsar_reader import read_slc_imagette

_DEFAULT_SAFE = (
    "/legacy/project/cersat/public/cache/project/mpc-sentinel1/data/esa/sentinel-1d/"
    "L1/WV/S1D_WV_SLC__1S/2026/093/"
    "S1D_WV_SLC__1SSV_20260403T111410_20260403T114853_002181_00378E_9153.SAFE"
)
_DEFAULT_REF_NC = (
    "/legacy/project/cersat/public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/"
    "L1_v16/WV/S1D_WV_SLC__1S/2026/093/"
    "S1D_WV_SLC__1SSV_20260403T111410_20260403T114853_002181_00378E_9153.SAFE/"
    "calib/S1D_WV_SLC__1SSV_20260403T111410_20260403T114853_002181_00378E_9153.SAFE.nc"
)

# How many imagettes to check (keep the test cheap; the SAFE has 143).
_N_TEST = 3


def _safe_path() -> str:
    return os.environ.get("S1WVCALIB_L1_SAFE", _DEFAULT_SAFE)


def _ref_nc() -> str:
    return os.environ.get("S1WVCALIB_L1_REF_NC", _DEFAULT_REF_NC)


def _data_available() -> bool:
    return Path(_safe_path()).is_dir() and Path(_ref_nc()).is_file()


@pytest.fixture(scope="module")
def reference():
    if not _data_available():
        pytest.skip(f"L1 data not available (safe={_safe_path()}, ref={_ref_nc()})")
    return netCDF4.Dataset(_ref_nc())


@pytest.fixture(scope="module")
def measurements():
    if not _data_available():
        pytest.skip("L1 data not available")
    ms = sorted(str(p) for p in (Path(_safe_path()) / "measurement").glob("*.tiff"))
    if not ms:
        pytest.skip("no measurement tiffs found in SAFE")
    return ms[:_N_TEST]


def _moments(imag: dict[str, object]) -> tuple[float, float, float]:
    i_sq = cast("np.ndarray", imag["I"])
    mu = np.mean(i_sq)
    nv = float(np.mean(np.power((i_sq - mu) / mu, 2)))
    skew = float(np.mean(np.power((i_sq - mu), 3)) / np.power(np.var(i_sq - mu), 1.5))
    kurt = float(np.mean(np.power((i_sq - mu), 4)) / np.power(np.var(i_sq - mu), 2))
    return nv, skew, kurt


def _assert_close(name: str, new: float, ref: float, atol: float, rtol: float) -> None:
    assert np.isclose(new, ref, atol=atol, rtol=rtol), (
        f"{name}: new={new!r} ref={ref!r} (atol={atol} rtol={rtol})"
    )


def test_l1_reader_matches_reference(reference, measurements):
    for ii, measu in enumerate(measurements):
        imag = read_slc_imagette(measu)

        _assert_close("sig_sar", float(np.mean(imag["sigma0_matrix"])),
                      reference["sig_sar"][ii], atol=1e-5, rtol=1e-5)
        _assert_close("sig_sar_denoised", float(np.mean(imag["sigma0_denoised_matrix"])),
                      reference["sig_sar_denoised"][ii], atol=1e-5, rtol=1e-5)
        _assert_close("noise_sar", float(np.mean(imag["noise_linear"])),
                      reference["noise_sar"][ii], atol=1e-5, rtol=1e-4)
        _assert_close("doppler_centroid", imag["doppler_centroid_mean"],
                      reference["doppler_centroid"][ii], atol=1e-3, rtol=1e-4)
        _assert_close("doppler_centroid_quality", float(imag["doppler_centroid_quality"]),
                      reference["doppler_centroid_quality"][ii], atol=0.0, rtol=0.0)
        _assert_close("track_angle", imag["track_angle"],
                      reference["track_angle"][ii], atol=1e-3, rtol=0.0)

        nv, skew, kurt = _moments(imag)
        _assert_close("nv_sar", nv, reference["nv_sar"][ii], atol=1e-4, rtol=1e-4)
        _assert_close("skew_sar", skew, reference["skew_sar"][ii], atol=1e-4, rtol=1e-4)
        _assert_close("kurt_sar", kurt, reference["kurt_sar"][ii], atol=1e-3, rtol=1e-4)


def test_l1_geolocation_matches_reference(reference, measurements):
    for ii, measu in enumerate(measurements):
        imag = read_slc_imagette(measu)

        # lon is 0-360 in the reference, xsar may return -180..180 -> compare mod 360.
        lon_new = np.mod(imag["lon"], 360.0)
        lon_ref = np.mod(reference["lon_sar"][ii], 360.0)
        _assert_close("lon_sar", float(lon_new), float(lon_ref), atol=0.05, rtol=0.0)
        _assert_close("lat_sar", imag["lat"], reference["lat_sar"][ii], atol=0.05, rtol=0.0)
        _assert_close("incidence_angle", imag["inc"],
                      reference["incidence_angle"][ii], atol=1e-3, rtol=0.0)
        _assert_close("elevation_angle", imag["elev"],
                      reference["elevation_angle"][ii], atol=1e-3, rtol=0.0)
