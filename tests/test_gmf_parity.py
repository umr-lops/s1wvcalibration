"""GMF parity tests: new library vs the legacy production source.

The legacy reference implementations are loaded from the old
``mpcsentinellibs/qualitycheck`` tree (default: the sandbox checkout, override
with ``S1WVCALIB_LEGACY_DIR``). If the tree is not available the tests are
skipped, so the suite still runs on a clean CI checkout.

Measured results (grid: v 1-25 m/s, phi 0-180 deg, inc 15-45 deg):

* ``cmod5n_legacy`` (ported, batch-clamped) vs the legacy ``cmod5n.py`` source:
  **exact** (max abs diff 0.0).
* ``cmod5n_xsarsea`` (standard, element-wise clamp) vs the legacy source:
  up to ~1.3 dB on the classical 7 m/s sweet spot, entirely in the WV2
  (inc > 30 deg) region; WV1 is identical to float noise. On the full
  1-25 m/s grid the gap grows to ~6.7 dB because the legacy batch clamp of
  ``S -> S0`` is applied to the *whole* array when *any* imagette is clamped.
* ``cmod_ifr2`` (xsarsea, float64) vs the legacy ``cmod_ifremer.cmod_ifr2``
  (float32 coefficients): ~2e-6 dB, i.e. float32 rounding only.
* ``compute_azimuth_angle`` vs the legacy source: **exact** (0.0).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from s1wvcalibration._legacy.compute_wind_azimuth import compute_azimuth_angle
from s1wvcalibration._legacy.gmf import cmod5n_forward, cmod5n_xsarsea, cmod_ifr2

_DEFAULT_LEGACY = (
    "/scratch/agrouaze/opencode_sandbox/calvalsentinel1/"
    "mpc-sentinel/mpcsentinellibs/qualitycheck"
)

logging.getLogger().setLevel(logging.WARNING)


def _legacy_dir() -> Path:
    return Path(os.environ.get("S1WVCALIB_LEGACY_DIR", _DEFAULT_LEGACY))


def _load(name: str, filename: str):
    path = _legacy_dir() / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def legacy_cmod5n():
    path = _legacy_dir() / "cmod5n.py"
    if not path.exists():
        pytest.skip(f"legacy source not available: {path}")
    sys.path.insert(0, str(_legacy_dir()))
    try:
        module = _load("legacy_cmod5n", "cmod5n.py")
    finally:
        sys.path.remove(str(_legacy_dir()))
    return module


@pytest.fixture(scope="module")
def legacy_cmod_ifremer():
    path = _legacy_dir() / "cmod_ifremer.py"
    if not path.exists():
        pytest.skip(f"legacy source not available: {path}")
    return _load("legacy_cmod_ifremer", "cmod_ifremer.py")


@pytest.fixture(scope="module")
def legacy_wind_azimuth():
    path = _legacy_dir() / "compute_wind_azimuth.py"
    if not path.exists():
        pytest.skip(f"legacy source not available: {path}")
    sys.path.insert(0, str(_legacy_dir()))
    try:
        module = _load("legacy_compute_wind_azimuth", "compute_wind_azimuth.py")
    finally:
        sys.path.remove(str(_legacy_dir()))
    return module


@pytest.fixture(scope="module")
def wv_grid():
    """Typical Wave-mode grid: v 1-25 m/s, phi 0-180 deg, inc 15-45 deg."""
    v = np.arange(1.0, 25.01, 0.5)
    phi = np.arange(0.0, 181.0, 1.0)
    inc = np.arange(15.0, 45.01, 1.0)
    return np.meshgrid(v, phi, inc, indexing="ij")


def _db(x) -> np.ndarray:
    return 10.0 * np.log10(np.asarray(x, dtype=np.float64))


def test_cmod5n_port_is_exact(legacy_cmod5n, wv_grid):
    """The ported legacy CMOD5_N must be numerically identical to the source."""
    v, phi, theta = wv_grid
    ported = cmod5n_forward(v, phi, theta)
    source = legacy_cmod5n.cmod5n_forward(v, phi, theta)
    assert np.max(np.abs(ported - source)) == 0.0


def test_cmod5n_xsarsea_vs_legacy_sweet_spot(wv_grid):
    """Standard vs batch-clamped CMOD5_N on the classical 7 m/s sweet spot."""
    _v, phi, theta = wv_grid
    v7 = np.full_like(theta, 7.0)
    db_xsarsea = _db(cmod5n_xsarsea(v7, phi, theta))
    db_legacy = _db(cmod5n_forward(v7, phi, theta))
    diff = np.abs(db_xsarsea - db_legacy)
    # WV1 (inc < 30 deg): no low-wind clamp involved -> float noise only.
    assert diff[theta < 30].max() < 1e-6
    # WV2 (inc > 30 deg): the batch S->S0 clamp bites, measured ~1.3 dB.
    assert diff[theta > 30].max() < 1.5


def test_cmod5n_xsarsea_vs_legacy_full_grid(wv_grid):
    """Sanity bound on the full grid (batch clamp can diverge at low wind)."""
    v, phi, theta = wv_grid
    diff = np.abs(_db(cmod5n_xsarsea(v, phi, theta)) - _db(cmod5n_forward(v, phi, theta)))
    # Measured 6.7 dB at the low-wind edge of the grid; keep the bound loose.
    assert diff.max() < 7.0


def test_cmod_ifr2_matches_legacy(legacy_cmod_ifremer):
    """xsarsea CMOD-IFR2 (float64) vs legacy (float32 coefficients)."""
    inc = np.arange(15.0, 45.01, 1.0)
    v = np.arange(1.0, 25.01, 0.5)
    phi = np.arange(0.0, 181.0, 5.0)
    i, vv, p = np.meshgrid(inc, v, phi, indexing="ij")
    new = cmod_ifr2(i.ravel(), vv.ravel(), p.ravel())
    legacy = legacy_cmod_ifremer.cmod_ifr2(i.ravel(), vv.ravel(), p.ravel())
    diff_db = np.abs(_db(new) - _db(legacy))
    # float32 rounding in the legacy coefficients only.
    assert diff_db.max() < 1e-5


def test_wind_azimuth_is_exact(legacy_wind_azimuth):
    track = np.arange(0.0, 360.0, 3.0)
    u = np.arange(-20.0, 20.01, 0.5)
    mer = np.arange(-20.0, 20.01, 0.5)
    tr, uuu, mermer = np.meshgrid(track, u, mer, indexing="ij")
    new = compute_azimuth_angle(tr, uuu, mermer)
    legacy = legacy_wind_azimuth.compute_azimuth_angle(tr, uuu, mermer)
    assert np.max(np.abs(new - legacy)) == 0.0
