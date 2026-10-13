"""End-to-end test of the calibration engine on a synthetic L2 (OCN) SAFE.

Builds a minimal but complete L2 OCN SAFE (two WV imagettes, one per sub-swath),
a colocated ECMWF 0.100 light wind field, a colocated WW3 spectra file and a
land mask, then drives :meth:`WVcalibrator.AnalyseOneSAFE` and the
``s1wvcalib`` CLI through the full L2 path (SAR read -> ECMWF colocation ->
GMF -> filters -> bias -> NetCDF + PNG). No HPC data required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

import s1wvcalibration.calibration as calib
from s1wvcalibration._legacy.ecmwf0100 import findfile
from s1wvcalibration._legacy.ww3_spectra import get_ww3spectra_l2_filepath
from s1wvcalibration.config import CONFIG_ENV_VAR, reset_config_cache

# is_land() calls DataArray.argmin/argmax() without an axis; recent xarray
# emits a FutureWarning about the (future) dict-of-indices behaviour. The
# behaviour is unchanged for our use, so ignore it (the suite runs
# filterwarnings=error).
pytestmark = [
    pytest.mark.filterwarnings("ignore:Behaviour of argmin/argmax:FutureWarning"),
    # a sub-swath can legitimately have no points in the SWH/WW3 filter; the
    # legacy code means over an empty slice -> NaN. Keep the (parity) behaviour.
    pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning"),
    # np.divide(out=, where=denominator!=0) still evaluates the 0-division and
    # warns even when `where` guards it (legacy parity behaviour).
    pytest.mark.filterwarnings(
        "ignore:invalid value encountered in scalar divide:RuntimeWarning"
    ),
]

_UTC = timezone.utc
SAFE = "S1D_WV_OCN__2SSV_20200101T000000_20200101T003443_012345_000001_1234"


def _calib_dir(out: Path) -> Path:
    """Locate the per-SAFE calib dir (legacy nests it under ``YYYY/DOY/SAFE``)."""
    matches = list(out.glob(f"*/*/{SAFE}/calib"))
    assert len(matches) == 1
    return matches[0]


# (subswath, imagette index, incidence, lon, lat, nrcs dB)
_IMAGETTES = [
    ("wv1", "0001", 20.0, 10.0, 20.0, -30.0),
    ("wv2", "0002", 35.0, 11.0, 21.0, -32.0),
]
_WIND = 7.0  # m/s, inside the classical 7 +/- 3 sweet spot

K = np.linspace(0.05, 0.4, 24)
PHI = np.linspace(0.0, 359.0, 36)


def _spec(k: np.ndarray, peak_k: float = 0.2, width: float = 0.04) -> np.ndarray:
    prof = np.exp(-0.5 * ((k - peak_k) / width) ** 2)
    return np.tile(prof, (PHI.size, 1))


def _make_ocn_meas(path: Path, inc: float, lon: float, lat: float, nrcs_db: float):
    ds = nc.Dataset(str(path), "w", format="NETCDF4")
    ds.createDimension("a", 1)
    ds.createDimension("b", 1)
    ds.createDimension("phi", PHI.size)
    ds.createDimension("k", K.size)
    ds.createDimension("pol", 1)
    ds.createDimension("acq", 1)
    scalar2d = {
        "oswLon": lon,
        "oswLat": lat,
        "oswIncidenceAngle": inc,
        "oswHeading": 0.0,
        "oswNrcs": nrcs_db,
        "oswNrcsNeszCorr": nrcs_db - 1.0,
        "oswNv": 0.02,
        "oswWindSpeed": _WIND,
        "oswWindDirection": 180.0,
    }
    for name, val in scalar2d.items():
        v = ds.createVariable(name, "f8", ("a", "b"))
        v[0, 0] = val
    # spectral variables (for the 2D-cutoff SWH branch)
    ds.createVariable("oswPolSpec", "f8", ("phi", "k"))[:] = _spec(K)
    ds.createVariable("oswK", "f8", ("k",))[:] = K
    ds.createVariable("oswPhi", "f8", ("phi",))[:] = PHI
    ds.createVariable("oswAzCutoff", "f8", ("pol", "acq"))[0, 0] = 10.0
    ds.createVariable("oswSpecRes", "f8")[:] = 0.01
    ds.createVariable("oswPartitions", "i4", ("phi", "k"))[:] = 1
    ds.close()


def _make_ww3(path: Path):
    ds = nc.Dataset(str(path), "w", format="NETCDF4")
    ds.createDimension("phi", PHI.size)
    ds.createDimension("k", K.size)
    ds.createVariable("polSpec", "f8", ("phi", "k"))[:] = _spec(K)
    ds.createVariable("k", "f8", ("k",))[:] = K
    ds.createVariable("phi", "f8", ("phi",))[:] = PHI
    ds.close()


def _make_ecmwf_light(path: Path, lat: float, lon: float):
    """A single-grid-point 0.100 light file with a uniform _WIND east wind."""
    ds = nc.Dataset(str(path), "w", format="NETCDF4")
    ds.createDimension("time", 1)
    ds.createDimension("lat", 1)
    ds.createDimension("lon", 1)
    t = ds.createVariable("time", "f8", ("time",))
    t.units = "hours since 2000-01-01 00:00:00"
    t[:] = [0.0]
    ds.createVariable("Latitude", "f8", ("lat",))[:] = [lat]
    ds.createVariable("Longitude", "f8", ("lon",))[:] = [lon]
    u = ds.createVariable("10U", "f8", ("time", "lat", "lon"))
    v = ds.createVariable("10V", "f8", ("time", "lat", "lon"))
    u[0] = [[_WIND]]
    v[0] = [[0.0]]
    ds.close()


def _make_land_mask(path: Path, all_ocean: bool = True):
    flag = 0.0 if all_ocean else 1.0
    ds = nc.Dataset(str(path), "w", format="NETCDF4")
    ds.createDimension("latitude", 2)
    ds.createDimension("longitude", 2)
    ds.createVariable("latitude", "f8", ("latitude",))[:] = [15.0, 25.0]
    ds.createVariable("longitude", "f8", ("longitude",))[:] = [5.0, 15.0]
    ds.createVariable("quality_flag", "f8", ("latitude", "longitude"))[:] = flag
    ds.close()


@pytest.fixture
def l2_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a synthetic L2 OCN SAFE + ECMWF + WW3 + land mask; override config."""
    safe_dir = tmp_path / "data" / SAFE
    meas_dir = safe_dir / "measurement"
    meas_dir.mkdir(parents=True)
    meas_paths = []
    for sub, idx, inc, lon, lat, nrcs in _IMAGETTES:
        name = (
            f"s1d-{sub}-ocn-vv-20200101t000000-20200101t003443-012345-00378e-{idx}.nc"
        )
        p = meas_dir / name
        _make_ocn_meas(p, inc, lon, lat, nrcs)
        meas_paths.append(p)

    ecmwf_root = tmp_path / "ecmwf"
    ecmwf_root.mkdir()
    ww3_root = tmp_path / "coloc"
    ww3_root.mkdir()
    land_root = tmp_path / "landmask"
    land_root.mkdir()

    # Set the override config FIRST so the locators below resolve under tmp.
    cfg = tmp_path / "override.yaml"
    cfg.write_text(
        "paths:\n"
        f"  root_directory: {tmp_path}\n"
        f"  ecmwf_0100_light: {ecmwf_root}\n"
        f"  ecmwf_0100_full: {ecmwf_root}\n"
        f"  ww3_coloc: {ww3_root}\n"
        f"  land_mask: {land_root}\n"
        f"  classif_wv_cls: {tmp_path}/classif\n"
        f"  analysis_dir: {tmp_path}/analysis\n"
    )
    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg))
    reset_config_cache()

    # colocated WW3 spectra (created where the locator expects it)
    for p in meas_paths:
        ww3_fp, _ = get_ww3spectra_l2_filepath(str(p), use_new_procedure=None)
        ww3_path = Path(ww3_fp)
        ww3_path.parent.mkdir(parents=True, exist_ok=True)
        _make_ww3(ww3_path)

    # colocated ECMWF light field at the path findfile() resolves
    date_sar1 = datetime(2020, 1, 1, 0, 0, tzinfo=_UTC)
    ecmwf_path = Path(
        findfile(date_sar1, path_light=str(ecmwf_root), path_full=str(ecmwf_root))
    )
    ecmwf_path.parent.mkdir(parents=True, exist_ok=True)
    _make_ecmwf_light(ecmwf_path, lat=20.0, lon=10.0)

    _make_land_mask(land_root / "mwf-ers-mask.nc", all_ocean=True)
    yield {
        "safe": safe_dir,
        "land_mask": land_root / "mwf-ers-mask.nc",
        "analysis_dir": tmp_path / "analysis",
    }
    reset_config_cache()


def _land_mask_dataset(all_ocean: bool = True) -> xr.Dataset:
    flag = 0.0 if all_ocean else 1.0
    return xr.Dataset(
        {
            "quality_flag": (
                ("latitude", "longitude"),
                np.full((2, 2), flag),
            )
        },
        coords={"latitude": [15.0, 25.0], "longitude": [5.0, 15.0]},
    )


def test_analyse_one_safe_l2(l2_safe):
    out = l2_safe["analysis_dir"] / "calib_out"
    final_result, flag = calib.WVcalibrator().AnalyseOneSAFE(
        str(l2_safe["safe"]),
        _land_mask_dataset(),
        str(out),
        max_latitude=50,
        ecmwf_version="0100",
        include_zero_s0=True,
    )
    assert flag is True
    # biases are finite for both sub-swaths / GMFs
    for key in (
        "cor_wv1_cmodifr",
        "cor_wv2_cmodifr",
        "cor_wv1_cmod5",
        "cor_wv2_cmod5",
    ):
        assert key in final_result
        assert np.isfinite(next(iter(final_result[key].values())))
    # the per-SAFE NetCDF is written
    calib_dir = _calib_dir(out)
    nc_path = calib_dir / (SAFE + ".nc")
    assert nc_path.exists()
    with nc.Dataset(nc_path) as ds:
        assert "sig_sar" in ds.variables
        assert "correction_wv1_cmod5n" in ds.variables
        assert "correction_wv2_cmod5n" in ds.variables
    # diagnostic PNGs are written next to the NetCDF
    pngs = list(calib_dir.glob("wind_vs_sig_calibrated_*.png"))
    assert len(pngs) >= 2


def test_analyse_one_safe_l2_land_rejected(l2_safe):
    # A land mask flagging every pixel does NOT clear flag_data_available: that
    # flag counts the measurement files read (legacy semantics, ``tpt``), and the
    # files are read fine. The land filter only removes pixels from the sweet
    # spot, so every bias ends up NaN while the per-SAFE NetCDF is still written.
    final_result, flag = calib.WVcalibrator().AnalyseOneSAFE(
        str(l2_safe["safe"]),
        _land_mask_dataset(all_ocean=False),
        str(l2_safe["analysis_dir"] / "land_out"),
        max_latitude=50,
        ecmwf_version="0100",
        include_zero_s0=True,
    )
    assert flag is True
    for key in ("cor_wv1_cmodifr", "cor_wv2_cmodifr", "cor_wv1_cmod5", "cor_wv2_cmod5"):
        for value in final_result[key].values():
            assert not np.isfinite(value)


def test_sequential_list_of_safe_analyse(l2_safe):
    out = l2_safe["analysis_dir"] / "seq_out"
    calib.sequentialListOfSAfeAnalyse(
        [str(l2_safe["safe"])], _land_mask_dataset(), str(out)
    )
    assert (_calib_dir(out) / (SAFE + ".nc")).exists()


def test_main_cli_l2(l2_safe):
    out = l2_safe["analysis_dir"] / "cli_out"
    rc = calib.main(
        [
            "-i",
            str(l2_safe["safe"]),
            "-o",
            str(out),
            "--ecmwf",
            "0100",
        ]
    )
    assert rc == 0
    assert (_calib_dir(out) / (SAFE + ".nc")).exists()


def test_is_land():
    msk = _land_mask_dataset(all_ocean=True)
    assert calib.is_land(10.0, 20.0, msk) is False
    msk_land = _land_mask_dataset(all_ocean=False)
    assert calib.is_land(10.0, 20.0, msk_land) is True


def test_config_helpers():
    assert calib._high_lat_threshold() == 50
    wr = calib._wind_ranges()
    assert wr == {"classical": 7, "noise_free": 10}
    assert calib._dev_max_measu() == 3
    assert isinstance(calib._time_unit(), str)
