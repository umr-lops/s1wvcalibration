"""Unit tests for the pure (non-IO) numeric helpers: angle formatting,
polarisation ratio, wind-bias ranges, and azimuth computation."""

from __future__ import annotations

import numpy as np

from s1wvcalibration._legacy.compute_wind_azimuth import compute_azimuth_angle
from s1wvcalibration._legacy.format_angle import format_angle, format_angle_v2
from s1wvcalibration._legacy.polarization_ratio import get_polarization_ratio
from s1wvcalibration._legacy.wind_bias import wind_plages, wind_pref_directions

# ---------------------------------------------------------------------------
# format_angle
# ---------------------------------------------------------------------------


def test_format_angle_array():
    out = format_angle(np.array([0.0, 200.0, -200.0, 360.0]))
    np.testing.assert_allclose(out, [0.0, -160.0, 160.0, 0.0])


def test_format_angle_array_compass():
    out = format_angle(np.array([-90.0, 90.0]), compass=True)
    np.testing.assert_allclose(out, [270.0, 90.0])


def test_format_angle_scalar():
    # single-element array (scalar floats hit a numpy-2 0-d issue in the
    # legacy branch; the engine always passes arrays)
    assert float(format_angle(np.array([200.0]))[0]) == -160.0
    assert float(format_angle(np.array([-200.0]))[0]) == 160.0
    assert float(format_angle(np.array([360.0]))[0]) == 0.0


def test_format_angle_v2_array():
    out = format_angle_v2(np.array([0.0, 200.0, -200.0]))
    np.testing.assert_allclose(out, [0.0, -160.0, 160.0])


def test_format_angle_v2_compass():
    assert float(format_angle_v2(np.array([-90.0]), compass_format=True)[0]) == 270.0


# ---------------------------------------------------------------------------
# polarisation ratio
# ---------------------------------------------------------------------------


def test_polarization_ratio_mouche1():
    inc = np.array([20.0, 30.0, 40.0])
    r = get_polarization_ratio(np.full_like(inc, 7.0), inc, name="mouche1")
    assert r.shape == inc.shape
    # ratio is a slowly-varying positive factor (~1.3 at these angles)
    assert np.all(r > 0.0)
    assert np.all(np.isfinite(r))


def test_polarization_ratio_mouche2():
    inc = np.array([20.0, 30.0, 40.0])
    phi = np.array([0.0, 90.0, 180.0])
    r = get_polarization_ratio(np.full_like(inc, 7.0), inc, phi=phi, name="mouche2")
    assert r.shape == inc.shape
    assert np.all(np.isfinite(r))


# ---------------------------------------------------------------------------
# wind bias ranges
# ---------------------------------------------------------------------------


def test_wind_bias_ranges_present():
    assert set(wind_plages) == {"lowwind", "mediumwind", "highwind"}
    assert wind_plages["lowwind"] == (0, 4)
    assert set(wind_pref_directions) == {
        "upwind",
        "downwind",
        "right_crosswind",
        "left_crosswind",
    }


# ---------------------------------------------------------------------------
# azimuth
# ---------------------------------------------------------------------------


def test_azimuth_scalar():
    # track along north (0), wind from the east (u=10 east, v=0)
    az = compute_azimuth_angle(np.array([0.0]), np.array([10.0]), np.array([0.0]))
    assert az.shape == (1,)
    assert 0.0 <= float(az[0]) <= 180.0


def test_azimuth_array():
    track = np.arange(0.0, 360.0, 45.0)
    u = np.full_like(track, 10.0)
    v = np.full_like(track, 0.0)
    az = compute_azimuth_angle(track, u, v)
    assert az.shape == track.shape
    assert np.all((az >= 0.0) & (az <= 180.0))
