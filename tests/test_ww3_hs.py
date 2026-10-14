"""Unit tests for the effective-significant-wave-height (2D-cutoff) chain and
the WW3 spectra file locator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

from s1wvcalibration._legacy.hs_2d_cutoff import get_WV_2D_cutoff
from s1wvcalibration._legacy.hs_eff_2d import (
    get_S1_hs_eff_2D_cutoff,
    get_ww3_hs_eff_2D_cutoff,
)
from s1wvcalibration._legacy.interpolation_polar import interp_polar_spectrum
from s1wvcalibration._legacy.low_frequency_energy import (
    get_low_frequencies_mask,
    get_low_frequencies_mask_core,
)
from s1wvcalibration._legacy.ww3_spectra import get_ww3spectra_l2_filepath
from s1wvcalibration.config import CONFIG_ENV_VAR, reset_config_cache

# A peaked (in k) polar spectrum on a (phi, k) grid.
K = np.linspace(0.05, 0.4, 24)  # 1/m
PHI = np.linspace(0.0, 359.0, 36)


def _spec(k: np.ndarray, peak_k: float = 0.2, width: float = 0.04) -> np.ndarray:
    """(phi, k) spectrum peaking at *peak_k*, azimuthally uniform."""
    prof = np.exp(-0.5 * ((k - peak_k) / width) ** 2)
    return np.tile(prof, (PHI.size, 1))


def _make_ocn_nc(
    path: Path, lowfreq: bool = False, masked_azcutoff: bool = False
) -> None:
    # the code indexes oswAzCutoff[0][0] -> a 2D variable. Fully masked when
    # ``masked_azcutoff``; otherwise a single 10 m value (k_cutoff = 2pi/10,
    # above all K -> full mask).
    azc: np.ma.MaskedArray = np.ma.masked_all((1, 1), dtype=np.float64)
    if not masked_azcutoff:
        azc[0, 0] = 10.0
    part = np.ones((PHI.size, K.size), dtype="i4")
    if lowfreq:
        part[:, :2] = -1  # two low-frequency (partition -1) bins
    ds = xr.Dataset(
        {
            "oswPolSpec": (("phi", "k"), _spec(K)),
            "oswK": (("k",), K),
            "oswPhi": (("phi",), PHI),
            "oswAzCutoff": (("pol", "acq"), azc),
            # spectral resolution is a scalar (the code tiles 2pi/res across k)
            "oswSpecRes": ((), np.float64(0.01)),
            "oswPartitions": (("phi", "k"), part),
        }
    )
    ds.to_netcdf(str(path), engine="h5netcdf")


def _make_ww3_nc(path: Path) -> None:
    ds = xr.Dataset(
        {
            "polSpec": (("phi", "k"), _spec(K, peak_k=0.2)),
            "k": (("k",), K),
            "phi": (("phi",), PHI),
        }
    )
    ds.to_netcdf(str(path), engine="h5netcdf")


def test_low_frequency_mask_core():
    partitions = np.array([[1, -1, 1], [-1, 1, -1]])
    np.testing.assert_array_equal(
        get_low_frequencies_mask_core(partitions), partitions == -1
    )


def test_low_frequency_mask_from_handler(tmp_path):
    p = tmp_path / "ocn.nc"
    _make_ocn_nc(p, lowfreq=True)
    with xr.open_dataset(p) as ds:
        m = get_low_frequencies_mask(ds)
    assert m.shape == (PHI.size, K.size)
    assert m[:, :2].all()
    assert not m[:, 2:].any()


def test_2d_cutoff_returns_mask_and_grid(tmp_path):
    p = tmp_path / "ocn.nc"
    _make_ocn_nc(p)
    with xr.open_dataset(p) as ds:
        mask_eff, k_sar, phi_sar = get_WV_2D_cutoff(ds, lowfreqfilter=True)
    assert mask_eff.shape == (PHI.size, K.size)
    np.testing.assert_array_equal(k_sar, K)
    np.testing.assert_array_equal(phi_sar, PHI)
    assert set(np.unique(mask_eff).tolist()) <= {0.0, 1.0}


def test_interp_polar_spectrum_identity():
    # interpolating onto the same grid returns (approx) the input
    spec = _spec(K, peak_k=0.2)
    out = interp_polar_spectrum(K, PHI, spec, K, PHI)
    assert out.shape == spec.shape
    # peak wavelength is preserved (0.2 -> 2pi/0.2)
    peak_k = K[np.argmax(np.nanmean(out, axis=0))]
    assert abs(peak_k - 0.2) < 0.05


def test_interp_polar_spectrum_bad_shape():
    with pytest.raises(ValueError, match="wrong input spectra shape"):
        interp_polar_spectrum(K, PHI, np.ones((K.size, PHI.size)), K, PHI)


def test_s1_hs_eff_2d_cutoff(tmp_path):
    p = tmp_path / "ocn.nc"
    _make_ocn_nc(p)
    with xr.open_dataset(p) as ds:
        swh_eff, swh_azc, wl, _dirp, _wl_eff, _dirp_eff = get_S1_hs_eff_2D_cutoff(ds)
    assert np.isfinite(swh_eff)
    assert np.isfinite(swh_azc)
    assert swh_eff > 0.0
    # the k-peak of the spectrum lies within [min(K), max(K)]
    assert np.isfinite(wl)
    assert 2 * np.pi / np.max(K) <= wl <= 2 * np.pi / np.min(K)


def test_s1_hs_eff_2d_masked_azcutoff(tmp_path):
    # masked azimuth cutoff -> all NaN (corrupted-file branch)
    p = tmp_path / "ocn.nc"
    _make_ocn_nc(p, masked_azcutoff=True)
    with xr.open_dataset(p) as ds:
        res = get_S1_hs_eff_2D_cutoff(ds)
    assert all(np.isnan(r) for r in res)


def test_ww3_hs_eff_2d_cutoff(tmp_path):
    p_s1 = tmp_path / "ocn.nc"
    p_ww3 = tmp_path / "ww3.nc"
    _make_ocn_nc(p_s1)
    _make_ww3_nc(p_ww3)
    with xr.open_dataset(p_s1) as ds_s1, xr.open_dataset(p_ww3) as ds_ww3:
        (
            swh_eff_ww3,
            swh_ww3,
            wl,
            _wl_eff,
            dirp,
            _dirp_eff,
            _wl_weighted,
        ) = get_ww3_hs_eff_2D_cutoff(ds_s1, ds_ww3)
    assert np.isfinite(swh_eff_ww3)
    assert np.isfinite(swh_ww3)
    assert swh_ww3 > 0.0
    assert np.isfinite(wl)
    # legacy: direction peak is always NaN (see note in hs_eff_2d.py)
    assert np.isnan(dirp)


# ---------------------------------------------------------------------------
# WW3 spectra file path finder
# ---------------------------------------------------------------------------


@pytest.fixture
def ww3_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path, None, None]:
    root = tmp_path / "coloc"
    root.mkdir()
    cfg = tmp_path / "override.yaml"
    cfg.write_text(f"paths:\n  ww3_coloc: {root}\n  ww3_satglob: {tmp_path}/satglob\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg))
    reset_config_cache()
    yield tmp_path
    reset_config_cache()


def _l2_ocn_name() -> str:
    # L2 OCN measurement names end in "-roughness.png" (replaced by ".nc").
    return (
        "S1D-WV1-OCN-VV-20200101t000000-20200101t003443-012345-00378e-0001"
        "-roughness.png"
    )


def test_ww3_filepath_default(ww3_config):  # noqa: ARG001  # side-effect fixture
    fp, proc = get_ww3spectra_l2_filepath("some/dir/" + _l2_ocn_name())
    assert proc == "run_datarmor_satglob"
    # satellite dir + yearmonth + daily date
    assert "sentinel-1d" in fp
    assert "202001" in fp
    # "ww3" + basename[3:] with "-roughness.png" -> ".nc"
    assert fp.endswith(
        "ww3-WV1-OCN-VV-20200101t000000-20200101t003443-012345-00378e-0001.nc"
    )


def test_ww3_filepath_satglob(ww3_config):  # noqa: ARG001  # side-effect fixture
    fp, proc = get_ww3spectra_l2_filepath(
        "some/dir/" + _l2_ocn_name(), use_new_procedure="SATGLOB"
    )
    assert proc == "run_ope_satglob"
    assert "new_recomposition_ww3spectra_202001" in fp
    assert "/S1D/L2/" in fp


def test_ww3_filepath_sentinel_not_implemented(ww3_config):  # noqa: ARG001  # side-effect fixture
    with pytest.raises(NotImplementedError):
        get_ww3spectra_l2_filepath(
            "some/dir/" + _l2_ocn_name(), use_new_procedure="SENTINEL"
        )


def test_ww3_filepath_rejeu2016(ww3_config):  # noqa: ARG001  # side-effect fixture
    # 201601 is in the S1A rejeu table
    name = "S1A-WV1-OCN-VV-20160101t000000-20160101t003443-012345-00378e-0001"
    fp, proc = get_ww3spectra_l2_filepath(
        "some/dir/" + name, use_new_procedure="REJEU2016"
    )
    assert proc == "rejeu2016"
    assert "sentinel-1a" in fp
    assert "rejeu_2016_01_07_L2" in fp


def test_ww3_filepath_invalid_procedure(ww3_config):  # noqa: ARG001  # side-effect fixture
    with pytest.raises(ValueError, match="cannot find the coloc dir WW3"):
        get_ww3spectra_l2_filepath(
            "some/dir/" + _l2_ocn_name(), use_new_procedure="BOGUS"
        )
