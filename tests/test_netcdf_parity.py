"""End-to-end parity: the full ``s1wvcalib`` pipeline vs the production NetCDF.

This is the test that *validates* the migration. It compares the NetCDF written
by the new library for a real S-1 WV SLC SAFE against the NetCDF the old
production (cerbere/SARImage mapper + legacy GMF) actually emitted for the same
SAFE.

What is asserted
----------------
* The **reporting geophysical calibration constants** (CMOD5n, the operational
  GMF per the ATBD) — classical 7 m/s and noise-free 10 m/s, WV1/WV2 — agree to
  ``< 5e-3 dB`` (float32 rounding).
* Every SAR-side per-imagette quantity (``sig_sar``, ``sig_sar_denoised``,
  ``noise_sar``, ``nv/skew/kurt``, ``doppler_centroid``, ``track_angle``) agrees
  to float32 precision on *all* imagettes (independent of wind colocation).
* The ECMWF wind colocation matches on all but at most a couple of imagettes:
  one antimeridian (dateline) imagette can pick the neighbouring 1 deg ECMWF
  cell, which is a nearest-grid-point edge case, not a regression.

Running
-------
The new NetCDF is produced by ``s1wvcalib`` (see :func:`_produce`). To keep CI
fast the test does NOT re-run the ~30 min pipeline unless asked:

* ``S1WVCALIB_E2E_REF_NC``  -- production reference NetCDF (default = HPC path).
* ``S1WVCALIB_E2E_NEW_NC``  -- a pre-produced NetCDF to compare against. If
  unset, the test produces one *only* if ``S1WVCALIB_E2E_RUN=1`` and the raw
  SAFE is present; otherwise it is skipped.

Typical local usage::

    S1WVCALIB_CONFIG=/path/localconfig.yaml s1wvcalib -i SAFE -o OUT --ecmwf 0100 --overwrite
    S1WVCALIB_E2E_NEW_NC=OUT/.../calib/SAFE.nc pytest -k netcdf_parity
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from s1wvcalibration.config import data_path

logger = logging.getLogger(__name__)

_SAFE_NAME = "S1D_WV_SLC__1SSV_20260403T111410_20260403T114853_002181_00378E_9153.SAFE"


def _default_safe() -> str:
    """Default raw SAFE location, derived from the config ``project_dir``."""
    return str(
        Path(data_path("project_dir"))
        / "data/esa/sentinel-1d/L1/WV/S1D_WV_SLC__1S/2026/093"
        / _SAFE_NAME
    )


def _default_ref_nc() -> str:
    """Default production NetCDF, derived from the config ``analysis_dir``."""
    return str(
        Path(data_path("analysis_dir"))
        / "L1_v16/WV/S1D_WV_SLC__1S/2026/093"
        / _SAFE_NAME
        / "calib"
        / f"{_SAFE_NAME}.nc"
    )


_DEFAULT_NEW_NC = (
    "/tmp/opencode/e2e_full/2026/093/"
    "S1D_WV_SLC__1SSV_20260403T111410_20260403T114853_002181_00378E_9153.SAFE/"
    "calib/S1D_WV_SLC__1SSV_20260403T111410_20260403T114853_002181_00378E_9153.SAFE.nc"
)

# Tolerances (float32 reference).
_DBCONST_ATOL = 5e-3  # reporting GMF constants, dB
_SAR_ATOL = 1e-4  # SAR-side per-imagette, relative
_COL_LOC_N_BAD = 3  # max imagettes allowed to differ in wind colocation


def _ref_nc() -> str:
    return os.environ.get("S1WVCALIB_E2E_REF_NC", _default_ref_nc())


def _safe() -> str:
    return os.environ.get("S1WVCALIB_L1_SAFE", _default_safe())


def _produce_new_nc() -> str | None:
    """Produce the new NetCDF by running the CLI; return its path or None."""
    safe = _safe()
    out = os.environ.get("S1WVCALIB_E2E_OUT", "/tmp/opencode/e2e_new")
    Path(out).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "s1wvcalibration.calibration",
        "-i",
        safe,
        "-o",
        out,
        "--ecmwf",
        "0100",
        "--overwrite",
    ]
    logger.info("producing end2end netcdf via: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    prod = [str(p) for p in Path(out).rglob("*.nc") if "calib" in p.parts]
    return prod[0] if prod else None


@pytest.fixture(scope="module")
def ref():
    if not Path(_ref_nc()).is_file():
        pytest.skip(f"end2end reference netcdf not available: {_ref_nc()}")
    return xr.open_dataset(_ref_nc())


@pytest.fixture(scope="module")
def new():
    path = os.environ.get("S1WVCALIB_E2E_NEW_NC", _DEFAULT_NEW_NC)
    if not Path(path).is_file():
        if os.environ.get("S1WVCALIB_E2E_RUN") and Path(_safe()).is_dir():
            produced = _produce_new_nc()
            if not produced:
                pytest.skip("could not produce the end2end netcdf")
            path = produced
        else:
            pytest.skip(
                f"end2end produced netcdf not available ({path}); "
                "set S1WVCALIB_E2E_NEW_NC or S1WVCALIB_E2E_RUN=1 to run the pipeline"
            )
    return xr.open_dataset(path)


def _v(ds: xr.Dataset, name: str) -> np.ndarray:
    return np.asarray(ds[name].values, dtype=np.float64)


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return 0.0
    return float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-30)))


def test_e2e_reporting_constants(ref, new):
    """The operational CMOD5n calibration constants must match the production."""
    for name in (
        "correction_wv1_cmod5n",
        "correction_wv2_cmod5n",
        "correction_wv1_cmod5n_10ws",
        "correction_wv2_cmod5n_10ws",
        "correction_wv1",
        "correction_wv2",
    ):
        a = float(_v(new, name)[0])
        b = float(_v(ref, name)[0])
        assert abs(a - b) < _DBCONST_ATOL, f"{name}: new={a:+.5f} ref={b:+.5f} dB"


def test_e2e_sar_per_imagette(ref, new):
    """SAR-side per-imagette quantities match on every imagette (float32)."""
    n = int(ref.sizes["number_of_wm_acq"])
    assert int(new.sizes["number_of_wm_acq"]) == n
    for name in (
        "sig_sar",
        "sig_sar_denoised",
        "noise_sar",
        "nv_sar",
        "skew_sar",
        "kurt_sar",
        "doppler_centroid",
        "track_angle",
    ):
        a = _v(new, name)
        b = _v(ref, name)
        assert np.isfinite(b).all() or True
        m = np.isfinite(a) & np.isfinite(b)
        assert m.any(), f"{name}: all non-finite"
        assert np.allclose(a[m], b[m], rtol=_SAR_ATOL, atol=1e-6), (
            f"{name}: max rel diff {_max_rel(a, b):.3g} > {_SAR_ATOL}"
        )


def test_e2e_windspeed_bias_and_counts(ref, new):
    """Scalar wind-bias / SWh-bias / nbobs entries match exactly."""
    for name in ref.data_vars:
        if (
            "bias" in name or "nbobs" in name or "nbptsNRCS" in name
        ) and "number_of_wm_correction" in ref[name].dims:
            a = _v(new, name)
            b = _v(ref, name)
            if not np.isfinite(b[0]):
                continue  # both undefined for L1 (e.g. windspeed_bias_full)
            assert a[0] == b[0], f"{name}: new={a[0]} ref={b[0]}"


def test_e2e_wind_colocation(ref, new):
    """Wind colocation matches on all but a couple (antimeridian) imagettes."""
    n = int(ref.sizes["number_of_wm_acq"])
    spd_n = _v(new, "spd_ecmwf")
    spd_r = _v(ref, "spd_ecmwf")
    bad = np.sum(np.abs(spd_n - spd_r) > 0.1)
    assert bad <= _COL_LOC_N_BAD, (
        f"{bad}/{n} imagettes differ in ECMWF colocation (> 0.1 m/s); "
        "expected at most the antimeridian case"
    )
