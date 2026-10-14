"""Tests for the NRCS-vs-wind figure replot entrypoint (``s1wvfig``)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from s1wvcalibration.figure import (
    _pol_str,
    make_nrcs_vs_wind_figure,
    replot_calibration,
)

SAFE = "S1D_WV_SLC__1SSV_20200101T000000_20200101T003443_012345_00378E_1234.SAFE"

# Per-imagette geometry: a mix of WV1 (inc < 30) and WV2 (inc > 30), all ocean,
# moderate latitude, with wind spread across the 7 +/- 3 m/s sweet spot.
N = 8
_INC = np.array([18.0, 22.0, 25.0, 28.0, 32.0, 35.0, 38.0, 42.0])
_LAT = np.full(N, 20.0)
_SPD = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 12.0, 15.0])
_ZON = np.full(N, 7.0)
_MER = np.zeros(N)
_TRACK = np.full(N, -90.0)
_SIG = np.array([0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.02, 0.015])


def _make_calib_nc(path, pol: str = "V") -> None:
    ds = xr.Dataset(
        {
            "sig_sar": (("number_of_wm_acq",), _SIG.astype(np.float32)),
            "sig_sar_denoised": (("number_of_wm_acq",), _SIG.astype(np.float32)),
            "noise_sar": (("number_of_wm_acq",), np.zeros(N, dtype=np.float32)),
            "incidence_angle": (("number_of_wm_acq",), _INC.astype(np.float32)),
            "lat_sar": (("number_of_wm_acq",), _LAT.astype(np.float32)),
            "land": (("number_of_wm_acq",), np.zeros(N, dtype=np.float32)),
            "track_angle": (("number_of_wm_acq",), _TRACK.astype(np.float32)),
            "spd_ecmwf": (("number_of_wm_acq",), _SPD.astype(np.float32)),
            "zon_ecmwf": (("number_of_wm_acq",), _ZON.astype(np.float32)),
            "mer_ecmwf": (("number_of_wm_acq",), _MER.astype(np.float32)),
            "sar_imagette_polarization": (
                ("number_of_wm_correction",),
                np.array([pol.encode("utf-8")], dtype="|S1"),
            ),
        },
        coords={
            "number_of_wm_acq": np.arange(N),
            "number_of_wm_correction": np.arange(1),
            "classifchars": np.arange(3),
        },
    )
    ds.to_netcdf(str(path), engine="h5netcdf")


def _figure_names(out) -> set[str]:
    return {p.name for p in out.glob("wind_vs_sig_calibrated_*.png")}


def test_replot_writes_three_figures(tmp_path):
    p = tmp_path / f"{SAFE}.nc"
    _make_calib_nc(p)
    written = replot_calibration(str(p))
    assert len(written) == 3
    assert all(w.is_file() for w in written)
    # named after the bare SAFE stem (the trailing ".SAFE"/".nc" are stripped,
    # matching the engine's ``PlotNRCSvsWindPerSafe`` naming)
    stem = SAFE[:-5]  # drop ".SAFE"
    assert all(stem in w.name and ".nc" not in w.name for w in written)
    # one figure per GMF
    assert {
        g for g in ("cmod5", "cmod-ifr2", "cmod5n") if any(g in w.name for w in written)
    } == {"cmod5", "cmod-ifr2", "cmod5n"}


def test_replot_default_output_dir(tmp_path):
    p = tmp_path / f"{SAFE}.nc"
    _make_calib_nc(p)
    written = replot_calibration(str(p))  # out_dir=None -> nc's directory
    assert all(w.parent == tmp_path for w in written)
    assert len(list(tmp_path.glob("wind_vs_sig_calibrated_*.png"))) == 3


def test_make_alias_is_replot(tmp_path):
    p = tmp_path / f"{SAFE}.nc"
    _make_calib_nc(p)
    assert make_nrcs_vs_wind_figure(str(p)) == replot_calibration(str(p))


def test_replot_h_pol_divides_by_ratio(tmp_path):
    p = tmp_path / f"{SAFE}.nc"
    _make_calib_nc(p, pol="H")
    written = replot_calibration(str(p))
    assert len(written) == 3
    assert all(w.is_file() for w in written)


def test_replot_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        replot_calibration(str(tmp_path / "does_not_exist.nc"))


def test_pol_str_variants():
    assert _pol_str(b"V") == "V"
    assert _pol_str("H") == "H"
    assert _pol_str(np.bytes_(b"V")) == "V"
