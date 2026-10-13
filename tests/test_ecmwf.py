"""Unit tests for the ECMWF wind readers and colocation (0.100 / 0.125 deg)."""

from __future__ import annotations

import datetime as dt
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np

from s1wvcalibration._legacy.ecmwf0100 import findfile, readfile
from s1wvcalibration._legacy.ecmwf0125 import get_meteofrance_cep_0125_wind
from s1wvcalibration._legacy.find_colocated_ecmwf_file import (
    find_colocated_ecmwf_file,
)

_UTC = timezone.utc


def _make_light_nc(path: Path) -> None:
    ds = nc.Dataset(str(path), "w", format="NETCDF4")
    ds.createDimension("time", 1)
    ds.createDimension("lat", 4)
    ds.createDimension("lon", 5)
    t = ds.createVariable("time", "f8", ("time",))
    t.units = "hours since 2000-01-01 00:00:00"
    t[:] = [0.0]
    lat = ds.createVariable("Latitude", "f8", ("lat",))
    lat[:] = [-10.0, -5.0, 0.0, 5.0]
    lon = ds.createVariable("Longitude", "f8", ("lon",))
    lon[:] = [10.0, 20.0, 30.0, 40.0, 50.0]
    u = ds.createVariable("10U", "f8", ("time", "lat", "lon"))
    v = ds.createVariable("10V", "f8", ("time", "lat", "lon"))
    u[0] = np.arange(20.0).reshape(4, 5)
    v[0] = np.arange(20.0, 40.0).reshape(4, 5)
    ds.close()


def test_readfile_light(tmp_path):
    p = tmp_path / "ECMWF_FORECAST_0100_202001010000_10U_10V.nc"
    _make_light_nc(p)
    d = readfile(str(p))
    assert d["lat"].shape == (4,)
    assert d["lon"].shape == (5,)
    assert d["uwind"].shape == (4, 5)
    assert d["vwind"].shape == (4, 5)
    # zon/mer are aliases for uwind/vwind
    np.testing.assert_array_equal(d["zon"], d["uwind"])
    np.testing.assert_array_equal(d["mer"], d["vwind"])
    expected = np.sqrt(d["uwind"] ** 2 + d["vwind"] ** 2)
    np.testing.assert_allclose(d["speed"], expected)
    # num2date returns datetime / cftime objects
    assert hasattr(d["time"][0], "year")


def test_findfile_light_nearest_hour(tmp_path):
    # nearest hourly file to 00:30 is 00:00
    path = findfile(
        datetime(2020, 1, 1, 0, 30, tzinfo=_UTC),
        path_light=str(tmp_path),
        path_full=str(tmp_path),
        typee="light",
    )
    assert path.endswith("ECMWF_FORECAST_0100_202001010000_10U_10V.nc")
    assert "/2020/001/" in path


def test_findfile_full_nomenclature(tmp_path):
    path = findfile(
        datetime(2020, 1, 1, 0, 0, tzinfo=_UTC),
        path_light=str(tmp_path),
        path_full=str(tmp_path),
        typee="full",
    )
    assert path.endswith("ECMWF_FORECAST_0100_202001010000.nc")
    assert "/2020/001/" in path


def test_findfile_from_timestamp(tmp_path):
    epoch = dt.datetime(2020, 6, 15, 12, 0, tzinfo=_UTC).timestamp()
    path = findfile(
        epoch,
        path_light=str(tmp_path),
        path_full=str(tmp_path),
        typee="light",
    )
    assert "202006151200" in path
    assert "/2020/167/" in path  # day of year for 2020-06-15


def _make_cep_nc(path: Path) -> None:
    ds = nc.Dataset(str(path), "w", format="NETCDF4")
    ds.latitude_resolution = 1.0
    ds.southernmost_latitude = 0.0
    ds.northernmost_latitude = 3.0
    ds.longitude_resolution = 1.0
    ds.westernmost_longitude = 0.0
    ds.easternmost_longitude = 4.0
    ds.createDimension("time", 1)
    ds.createDimension("y", 4)
    ds.createDimension("x", 5)
    t = ds.createVariable("time", "i4", ("time",))
    t.units = "hours since 2000-01-01 00:00:00"
    t[:] = [100]
    u = ds.createVariable("u10m", "f8", ("time", "y", "x"))
    v = ds.createVariable("v10m", "f8", ("time", "y", "x"))
    u[0] = np.ones((4, 5))
    v[0] = np.zeros((4, 5))
    ds.close()


def test_get_meteofrance_cep_0125_wind(tmp_path):
    p = tmp_path / "METEOFRANCE_CEP_20200101T00.nc"
    _make_cep_nc(p)
    wind = get_meteofrance_cep_0125_wind(str(p), verbose=True)
    assert wind["lat"].shape == (4, 5)
    assert wind["lon"].shape == (4, 5)
    assert wind["zon"].shape == (4, 5)
    # latitudes span the configured range (reversed to south->north)
    assert set(np.unique(wind["lat"]).tolist()) == {0.0, 1.0, 2.0, 3.0}
    assert wind["speed"].min() >= 0.0
    assert wind["time"].tzinfo is not None


def test_get_meteofrance_cep_0125_wind_control_plot(tmp_path, monkeypatch):
    p = tmp_path / "METEOFRANCE_CEP_20200101T00.nc"
    _make_cep_nc(p)
    monkeypatch.chdir(tmp_path)
    wind = get_meteofrance_cep_0125_wind(str(p), control_plot=True)
    assert (tmp_path / "control_plot_meteofrance_cep_0125_wind.png").exists()
    assert wind["speed"].shape == (4, 5)


def _make_cep_tree(root: str):
    d = Path(root) / "2020" / "001"
    d.mkdir(parents=True, exist_ok=True)
    # two candidate files: one near 00:00, one near 03:00
    for name in (
        "METEOFRANCE_CEP_20200101T00.nc",
        "METEOFRANCE_CEP_20200101T03.nc",
    ):
        p = d / name
        ds = nc.Dataset(str(p), "w", format="NETCDF4")
        ds.close()


def test_find_colocated_ecmwf_file_found(tmp_path):
    _make_cep_tree(str(tmp_path))
    res = find_colocated_ecmwf_file(
        datetime(2020, 1, 1, 0, 10, tzinfo=_UTC),
        path_wind=str(tmp_path),
        hour_step=3,
    )
    assert res["status"] == 1
    assert "METEOFRANCE_CEP_20200101T00.nc" in res["filename"]


def test_find_colocated_ecmwf_file_nearest_of_two(tmp_path):
    _make_cep_tree(str(tmp_path))
    # 01:00 is closer to the 03:00 file than to 00:00? No: 60 min vs 120 min.
    res = find_colocated_ecmwf_file(
        datetime(2020, 1, 1, 1, 0, tzinfo=_UTC),
        path_wind=str(tmp_path),
        hour_step=3,
    )
    assert res["status"] == 1
    assert "METEOFRANCE_CEP_20200101T00.nc" in res["filename"]


def test_find_colocated_ecmwf_file_not_found(tmp_path):
    res = find_colocated_ecmwf_file(
        datetime(2020, 1, 1, 0, 0, tzinfo=_UTC),
        path_wind=str(tmp_path / "does_not_exist"),
        hour_step=3,
    )
    assert res["status"] == 0
    assert res["filename"] is None
