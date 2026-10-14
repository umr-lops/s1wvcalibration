"""Core geophysical calibration engine for Sentinel-1 Wave Mode (WV).

This module is the re-packaged version of the legacy production script
``qualitycheck/computeWVconstantecalibration.py``. The public surface is:

- :class:`WVcalibrator` -- computes the per-SAFE calibration constant and
  writes the NetCDF + diagnostic PNG.
- :func:`entrypoint` -- the ``s1wvcalib`` CLI.

Dependencies that used to be flat modules in ``mpcsentinellibs`` now live in
the :mod:`s1wvcalibration._legacy` package. The hard cluster paths (land mask,
ECMWF archives, WW3 colocation, output dir) come from
:func:`~s1wvcalibration.config.load_config` instead of ``shared_information``.
"""

from __future__ import annotations

import argparse
import collections
import logging
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats
from scipy.special import exp10

from ._legacy.classification import get_classification_infos
from ._legacy.compute_wind_azimuth import compute_azimuth_angle
from ._legacy.ecmwf0100 import findfile, readfile
from ._legacy.ecmwf0125 import get_meteofrance_cep_0125_wind
from ._legacy.find_colocated_ecmwf_file import find_colocated_ecmwf_file
from ._legacy.gmf import cmod5, cmod5n_forward, cmod5n_xsarsea, cmod_ifr2
from ._legacy.hs_eff_2d import get_S1_hs_eff_2D_cutoff, get_ww3_hs_eff_2D_cutoff
from ._legacy.polarization_ratio import get_polarization_ratio
from ._legacy.wind_bias import wind_plages, wind_pref_directions
from ._legacy.ww3_spectra import get_ww3spectra_l2_filepath
from ._legacy.xsar_reader import read_slc_imagette
from .config import data_path, load_config

logger = logging.getLogger(__name__)

# Sentinel-1 SAFE times are UTC; keep datetimes tz-aware UTC so the ECMWF
# colocation (nearest file by wall-clock) is timezone-independent.
_UTC = timezone.utc


def _high_lat_threshold() -> int:
    """|latitude| (deg) above which data is rejected (config)."""
    return int(load_config()["calibration"]["high_lat_threshold"])


def _wind_ranges() -> dict[str, int]:
    """Center wind speed (m/s) per sweet-spot (config)."""
    return {
        k: int(v)
        for k, v in cast(
            "dict[str, int]", load_config()["calibration"]["wind_ranges"]
        ).items()
    }


def _dev_max_measu() -> int:
    """Number of imagettes processed in ``--dev`` mode (config)."""
    return int(load_config()["calibration"]["dev_max_measu"])


def _time_unit() -> str:
    """NetCDF time reference (config)."""
    return str(load_config()["calibration"]["time_unit"])


def is_land(lon: float, lat: float, dsmask: xr.Dataset) -> bool:
    """Return ``True`` if the grid point closest to (lon, lat) is land.

    Defined for the ``mwf-ers-mask.nc`` land mask.

    Args:
        lon: longitude (deg).
        lat: latitude (deg).
        dsmask: xarray.Dataset with a ``quality_flag`` variable.
    """
    lons = dsmask.longitude
    lats = dsmask.latitude
    # Nearest grid-point index. Call argmin on the raw values: DataArray.argmin()
    # with no dim/axis is deprecated (and will change to return a dict of per-
    # dim indices), which would break this and, under a warnings-as-errors suite,
    # raise instead of computing the index.
    lon_idx = int(np.argmin(np.abs(lons.values - lon)))
    lat_idx = int(np.argmin(np.abs(lats.values - lat)))
    flag = dsmask.quality_flag.isel(longitude=lon_idx, latitude=lat_idx).item()
    return flag != 0


class WVcalibrator:
    """Compute the WV calibration constant from one SAFE (L1 or L2)."""

    def AnalyseOneSAFE(
        self,
        in_safe_file: str,
        msk: xr.Dataset,
        path_out: str,
        max_latitude: int | None = None,
        suffix_output: str = "",
        save_calib_file: bool = True,
        include_zero_s0: bool = False,
        ecmwf_version: str = "0100",
        redo: bool = False,
        dev: bool = False,
    ) -> tuple[dict, bool]:
        """Compute the WV calibration constant from sigma0 and colocated wind.

        Args:
            in_safe_file: full path of the SAFE WV to analyse (L1 or L2).
            msk: xarray.Dataset of the land mask.
            path_out: output directory for the png and netcdf files.
            max_latitude: |latitude| (deg) to reject (default from config).
            suffix_output: suffix appended to the generated files.
            include_zero_s0: take sigma0==0 pixels into account in the mean.
            ecmwf_version: ``0125`` (3-hourly) or ``0100`` (hourly).
            redo: overwrite an existing NetCDF output.
            dev: process only a few imagettes (faster).

        Returns:
            ``(final_result, flag_data_available)``.
        """
        if max_latitude is None:
            max_latitude = _high_lat_threshold()
        wind_ranges = _wind_ranges()
        logger.info("ecmwf version = %s", ecmwf_version)
        logger.info("outputdir will be : %s", path_out)
        final_result: dict = {}
        flag_data_available = True  # by default we consider there is data available
        if in_safe_file[-1] == "/":
            in_safe_file = in_safe_file[:-1]
        if "OCN" in in_safe_file:
            level = "L2"
            ext = "nc"
        elif "SLC" in in_safe_file:
            level = "L1"
            ext = "tiff"
        cpt_counter: dict[str, int] = collections.defaultdict(int)
        logger.info("input file %s", in_safe_file)
        self.cpt_wind_bias: dict[str, int] = collections.defaultdict(int)
        self.orbit_name = Path(in_safe_file).name
        logger.info("==========================")
        logger.info("Sentinel-1 SAFE file:%s ", self.orbit_name)
        self.orbit_name = self.orbit_name.replace("\n", "")

        # nominal case product
        special = False
        date_orbit1 = self.orbit_name.split("_")[5][0:8]
        hour_orbit1 = self.orbit_name.split("_")[5][9:]
        date_orbit2 = self.orbit_name.split("_")[6][0:8]
        hour_orbit2 = self.orbit_name.split("_")[6][9:]
        logger.debug(
            "date_orbit1 %s hour_orbit1 %s date_orbit2 %s hour_orbit2 %s",
            date_orbit1,
            hour_orbit1,
            date_orbit2,
            hour_orbit2,
        )
        self.polarisation = self.orbit_name[15]
        self.date_sar1 = datetime.strptime(
            date_orbit1 + "_" + hour_orbit1, "%Y%m%d_%H%M%S"
        ).replace(tzinfo=_UTC)
        _ = datetime.strptime(date_orbit2 + "_" + hour_orbit2, "%Y%m%d_%H%M%S").replace(
            tzinfo=_UTC
        )
        dayofyear = self.date_sar1.timetuple().tm_yday
        self.path_out = (
            Path(path_out)
            / date_orbit1[0:4]
            / str(dayofyear).zfill(3)
            / self.orbit_name
            / "calib"
        )
        self.file_out_netcdf = self.orbit_name + suffix_output + ".nc"
        finalpath = self.path_out / self.file_out_netcdf
        if finalpath.exists() and redo is False:
            logger.info("file %s already exists, no over-writing", finalpath)
            return final_result, flag_data_available

        # ===================
        # Ancillary data
        # ===================
        if ecmwf_version == "0125":
            _file = find_colocated_ecmwf_file(
                self.date_sar1, path_wind=None, hour_step=None
            )
        else:
            tmpres_colocecm0100 = findfile(self.date_sar1, typee="light")
            _file = {"status": 1, "filename": tmpres_colocecm0100}
        if _file["status"] == 1:
            coloc_wind = True
            file_wind_coloc = _file["filename"]
        else:
            coloc_wind = False
            file_wind_coloc = None
        tpt = 0
        if coloc_wind:
            if ecmwf_version == "0125":
                anc_wind = get_meteofrance_cep_0125_wind(
                    file_wind_coloc, verbose=False, control_plot=False
                )
            else:  # case ecmwf 0.100 hourly
                anc_wind = readfile(file_wind_coloc)
                lonwnd, latwnd = np.meshgrid(anc_wind["lon"], anc_wind["lat"])
                anc_wind["lon"] = lonwnd
                anc_wind["lat"] = latwnd
            lat_wind = anc_wind["lat"]
            lon_wind = anc_wind["lon"]
            zon_wind = anc_wind["zon"]
            mer_wind = anc_wind["mer"]
            spd_wind = np.sqrt(np.power(zon_wind, 2) + np.power(mer_wind, 2))

            if special:
                measurements = [
                    str(Path(root) / fifi)
                    for root, _dirs, files in os.walk(in_safe_file)
                    for fifi in files
                    if "ocn" in fifi and ".nc" in fifi
                ]
            else:
                measu_dir = Path(in_safe_file) / "measurement"
                logger.info("pattern %s", measu_dir / ("*" + ext))
                measurements = [str(p) for p in measu_dir.glob("*" + ext)]
            if dev:
                logger.info(
                    "dev/test mode is True, limit number of measurement to %i (originally: %i)",
                    _dev_max_measu(),
                    len(measurements),
                )
                measurements = measurements[0 : _dev_max_measu()]
            measurements = sorted(measurements)
            self.nbimagestiff = len(measurements)
            nb_measurements = len(measurements)
            _land = np.zeros(nb_measurements, dtype=bool)
            dates_sar = np.zeros(nb_measurements, dtype=object)
            _highlat_flag = np.ones(nb_measurements, dtype=bool)
            _tra_sar = np.zeros(nb_measurements)
            _inc_sar = np.zeros(nb_measurements)
            _elev_sar = np.zeros(nb_measurements)
            _lon_sar = np.zeros(nb_measurements)
            _lat_sar = np.zeros(nb_measurements)
            _sig_sar = np.zeros(nb_measurements)
            _mom_stat3_sar_nrcs = np.zeros(nb_measurements)
            _mom_stat4_sar_nrcs = np.zeros(nb_measurements)
            _noise_sar = np.zeros(nb_measurements)
            _quality_doppler_centroid = np.zeros(nb_measurements)
            _doppler_centroid = np.zeros(nb_measurements)
            _classification_sar: list[str] = []
            _sig_sar_denoised = np.zeros(nb_measurements)
            _nv_sar = np.zeros(nb_measurements)
            _hseff2d_sar = np.zeros(nb_measurements)
            _hseff2d_ww3 = np.zeros(nb_measurements)
            _windspeed_sar = np.zeros(nb_measurements)
            _winddirection_sar = np.zeros(nb_measurements)
            _spd_coloc_ecmwf = np.zeros(nb_measurements)
            _dir_coloc_ecmwf = np.zeros(nb_measurements)
            _zon_coloc_ecmwf = np.zeros(nb_measurements)
            _mer_coloc_ecmwf = np.zeros(nb_measurements)
            _lon_coloc_ecmwf = np.zeros(nb_measurements)
            _lat_coloc_ecmwf = np.zeros(nb_measurements)
            dlon_coloc_ecmwf = np.zeros(nb_measurements)
            dlat_coloc_ecmwf = np.zeros(nb_measurements)
            _measu_fname: list[str] = []
            _indices_ecmwf_coloc = np.zeros(nb_measurements)
            cpt_not_finite_position = 0

            err: dict[str, int] = collections.defaultdict(int)
            cnt: dict[str, int] = collections.defaultdict(int)
            logger.info("Nber of measurements: %s", nb_measurements)
            for ii, in_file_meas in enumerate(measurements):
                _measu_fname.append(in_file_meas)
                content: dict[str, object] = {}
                content, _errors, _counters = get_classification_infos(
                    in_file_meas, content, err, cnt
                )
                if "class_1" in content:
                    _classification_sar.append(str(content["class_1"]).ljust(3))
                else:
                    _classification_sar.append("   ")
                if ii % 15 == 0:
                    logger.info(
                        "Sentinel-1 file: %s %s/%s",
                        Path(in_file_meas).name,
                        ii + 1,
                        len(measurements),
                    )
                try:
                    if level == "L1":
                        imag = read_slc_imagette(in_file_meas)
                        # Naive on purpose: this feeds the ``time_sar`` NetCDF
                        # variable; xarray cannot serialise tz-aware datetimes
                        # in an object array (see ``Write2NetCDFWVsentinel1CalValParams3``).
                        dates_sar[ii] = datetime.strptime(  # noqa: DTZ007
                            Path(in_file_meas).name.split("-")[4], "%Y%m%dt%H%M%S"
                        )
                        noise = imag["noise_linear"]
                        _lon_sar[ii] = imag["lon"]
                        _lat_sar[ii] = imag["lat"]
                        _inc_sar[ii] = imag["inc"]
                        _elev_sar[ii] = imag["elev"]
                        _quality_doppler_centroid[ii] = imag["doppler_centroid_quality"]
                        _doppler_centroid[ii] = imag["doppler_centroid_mean"]
                        sig = imag["sigma0_matrix"]
                        if not include_zero_s0:
                            _sig_sar[ii] = np.mean(sig[sig > 0])
                        else:
                            _sig_sar[ii] = np.mean(sig)
                            logger.debug(
                                "zero value taken into account in the imagette"
                            )
                        sig_denoised = imag["sigma0_denoised_matrix"]
                        if not include_zero_s0:
                            _sig_sar_denoised[ii] = np.mean(
                                sig_denoised[sig_denoised > 0]
                            )
                        else:
                            _sig_sar_denoised[ii] = np.mean(sig_denoised)
                        _noise_sar[ii] = np.mean(noise)  # in linear
                        intensity = imag["I"]
                        _mu = np.mean(intensity)
                        _nv_sar[ii] = np.mean(np.power(((intensity - _mu) / _mu), 2))
                        _tra_sar[ii] = imag["track_angle"]
                        my_skewness = np.mean(
                            np.power((intensity - _mu), 3)
                        ) / np.power(np.var(intensity - _mu), (3 / 2))
                        my_kurtosis = np.mean(
                            np.power((intensity - _mu), 4)
                        ) / np.power(np.var(intensity - _mu), 2)
                        _mom_stat3_sar_nrcs[ii] = my_skewness
                        _mom_stat4_sar_nrcs[ii] = my_kurtosis
                        normvar = np.var(intensity, ddof=1) / _mu**2
                        logger.debug("check Nv mine %s scipy %s", _nv_sar[ii], normvar)
                        skewness = np.mean(stats.skew(intensity - _mu))
                        kurtosis = np.mean(
                            stats.kurtosis(intensity - _mu, axis=1, fisher=True)
                        )
                        logger.debug(
                            "kurtosis check mine %s scipy %s ", my_kurtosis, kurtosis
                        )
                        logger.debug(
                            "skewness check mine %s scipy %s ", my_skewness, skewness
                        )
                    elif level == "L2":
                        dsocn = xr.open_dataset(in_file_meas)
                        # Naive on purpose: feeds the ``time_sar`` NetCDF variable
                        # (xarray cannot serialise tz-aware datetimes in object arrays).
                        dates_sar[ii] = datetime.strptime(  # noqa: DTZ007
                            Path(in_file_meas).name.split("-")[4], "%Y%m%dt%H%M%S"
                        )
                        ww3file, _procedure = get_ww3spectra_l2_filepath(
                            in_file_meas, use_new_procedure=None
                        )
                        if Path(ww3file).exists():
                            cpt_counter["ww3file_found"] += 1
                            ds_ww3 = xr.open_dataset(ww3file, decode_times=False)
                            try:
                                (
                                    hsww3_2D,
                                    _hsww3_1D,
                                    _ww3wavelengthpeak,
                                    _ww3wavelengthpeak_eff,
                                    _ww3directionpeak,
                                    _ww3directionpeak_eff,
                                    _ww3wl_partSAR1_weighted,
                                ) = get_ww3_hs_eff_2D_cutoff(dsocn, ds_ww3)
                                (
                                    hscutoff2d,
                                    _hscutoff1d,
                                    _ocnwavelengthpeak,
                                    _ocndirectionpeak,
                                    _ocnwavelengthpeak_eff,
                                    _ocndirectionpeak_eff,
                                ) = get_S1_hs_eff_2D_cutoff(dsocn)
                            finally:
                                ds_ww3.close()
                            _hseff2d_sar[ii] = hscutoff2d
                            _hseff2d_ww3[ii] = hsww3_2D
                        else:
                            _hseff2d_sar[ii] = np.nan
                            _hseff2d_ww3[ii] = np.nan
                            cpt_counter["ww3file_absent"] += 1
                        _lon_sar[ii] = dsocn["oswLon"].values[0][0]
                        _lat_sar[ii] = dsocn["oswLat"].values[0][0]
                        _inc_sar[ii] = dsocn["oswIncidenceAngle"].values[0][0]
                        _doppler_centroid[ii] = np.nan
                        _elev_sar[ii] = np.nan
                        _quality_doppler_centroid[ii] = np.nan
                        _sig_sar[ii] = exp10(dsocn["oswNrcs"].values[0][0] / 10.0)
                        if "oswNrcsNeszCorr" in dsocn.variables:
                            _sig_sar_denoised[ii] = exp10(
                                dsocn["oswNrcsNeszCorr"].values[0][0] / 10.0
                            )
                        else:
                            _sig_sar_denoised[ii] = exp10(
                                dsocn["oswNrcs"].values[0][0] / 10.0
                            )
                        _noise_sar[ii] = np.nan
                        _nv_sar[ii] = dsocn["oswNv"].values[0][0]
                        _tra_sar[ii] = dsocn["oswHeading"].values[0][0]
                        _windspeed_sar[ii] = dsocn["oswWindSpeed"].values[0][0]
                        _winddirection_sar[ii] = dsocn["oswWindDirection"].values[0][0]
                        dsocn.close()

                    if np.isfinite(_lat_sar[ii]) and np.isfinite(_lon_sar[ii]):
                        if np.abs(_lat_sar[ii]) < max_latitude:
                            _highlat_flag[ii] = False
                        if not _highlat_flag[ii]:
                            _land[ii] = is_land(
                                lon=_lon_sar[ii], lat=_lat_sar[ii], dsmask=msk
                            )
                        else:
                            _land[ii] = False
                        if np.min(lon_wind) >= 0:
                            _lon_sar = np.mod(_lon_sar, 360)
                            cpt_counter["bad_longitude_value"] += 1

                        diff = np.abs(_lat_sar[ii] - np.ravel(lat_wind)) + np.abs(
                            _lon_sar[ii] - np.ravel(lon_wind)
                        )
                        ind = np.where(diff == np.min(diff))
                        _indices_ecmwf_coloc[ii] = ind[0][0]
                        _dir = np.mod(
                            (180.0 / np.pi)
                            * (
                                math.atan2(
                                    (np.ravel(mer_wind))[ind[0][0]],
                                    (np.ravel(zon_wind))[ind[0][0]],
                                )
                            )
                            - 90,
                            360,
                        )
                        _spd_coloc_ecmwf[ii] = np.ravel(spd_wind)[ind[0][0]]
                        _dir_coloc_ecmwf[ii] = _dir
                        _zon_coloc_ecmwf[ii] = np.ravel(zon_wind)[ind[0][0]]
                        _mer_coloc_ecmwf[ii] = np.ravel(mer_wind)[ind[0][0]]
                        _lon_coloc_ecmwf[ii] = np.ravel(lon_wind)[ind[0][0]]
                        _lat_coloc_ecmwf[ii] = np.ravel(lat_wind)[ind[0][0]]
                        dlon_coloc_ecmwf[ii] = _lon_coloc_ecmwf[ii] - _lon_sar[ii]
                        dlat_coloc_ecmwf[ii] = _lat_coloc_ecmwf[ii] - _lat_sar[ii]
                    else:
                        logger.warning(
                            "file %s contains lon=%s and lat=%s",
                            in_file_meas,
                            _lon_sar[ii],
                            _lat_sar[ii],
                        )
                        cpt_not_finite_position += 1
                    tpt = tpt + 1
                except KeyboardInterrupt:
                    sys.exit(-1)
                except Exception:
                    logger.exception("can't read the file: %s ", in_file_meas)
                    logger.exception("%s", traceback.format_exc())
                if dev and ii >= _dev_max_measu():
                    logger.warning("stop the processing after 3 images to go quicker")
                    break
        if tpt > 0:
            self._cor_wv1_cmod5 = {}
            self._cor_wv2_cmod5 = {}
            self._cor_wv1_cmod5n = {}
            self._cor_wv2_cmod5n = {}
            self._cor_wv1_cmod5n_xsarsea = {}
            self._cor_wv2_cmod5n_xsarsea = {}
            self._cor_wv1 = {}
            self._cor_wv2 = {}
            self.mean_inc_1_corr = {}
            self.mean_inc_2_corr = {}
            self.mean_sig_1_corr = {}
            self.mean_sig_2_corr = {}
            self.nb_inc1_corr = {}
            self.nb_inc2_corr = {}
            for windrange_i in wind_ranges:
                self._dates_sar = dates_sar
                self._mer_coloc_ecmwf = _mer_coloc_ecmwf
                self._lon_coloc_ecmwf = _lon_coloc_ecmwf
                self._lat_coloc_ecmwf = _lat_coloc_ecmwf
                self._zon_coloc_ecmwf = _zon_coloc_ecmwf
                self._spd_coloc_ecmwf = _spd_coloc_ecmwf
                self._dir_coloc_ecmwf = _dir_coloc_ecmwf
                self.dlat_coloc_ecmwf = dlat_coloc_ecmwf
                self.dlon_coloc_ecmwf = dlon_coloc_ecmwf
                self._measu_fname = np.array(_measu_fname)
                self._indices_ecmwf_coloc = _indices_ecmwf_coloc
                self._sig_sar = _sig_sar
                self._sig_sar_denoised = _sig_sar_denoised
                self._noise_sar = _noise_sar
                logger.info("Nber of nan in sigmaO SAR: %s", np.sum(np.isnan(_sig_sar)))
                logger.info("Nber of negative sigmaO SAR: %s", np.sum(_sig_sar < 0))
                self._nv_sar = _nv_sar
                self._mom_stat3_sar_nrcs = _mom_stat3_sar_nrcs
                self._mom_stat4_sar_nrcs = _mom_stat4_sar_nrcs
                self._lon_sar = _lon_sar
                self._lat_sar = _lat_sar
                self._inc_sar = _inc_sar
                self._elev_sar = _elev_sar
                self._quality_doppler_centroid = _quality_doppler_centroid
                self._doppler_centroid = _doppler_centroid
                self._tra_sar = _tra_sar
                self._land = _land
                self._classification_sar = np.array(_classification_sar)
                self._azi_sar_coloc_ecmwf = compute_azimuth_angle(
                    self._tra_sar, self._zon_coloc_ecmwf, self._mer_coloc_ecmwf
                )
                sigma0_cmod5 = cmod5(
                    _spd_coloc_ecmwf, self._azi_sar_coloc_ecmwf, _inc_sar
                )
                sigma0_cmod_ifr2 = cmod_ifr2(
                    _inc_sar, _spd_coloc_ecmwf, self._azi_sar_coloc_ecmwf
                )
                # Two CMOD5n implementations kept side by side during the migration:
                #  - legacy (batch "correction vector agrouaze 2019") -> operational `sig_mod`
                #  - xsarsea standard CMOD5_N (element-wise) -> `sig_mod_xsarsea`
                sigma0_cmod5n = cmod5n_forward(
                    _spd_coloc_ecmwf, self._azi_sar_coloc_ecmwf, _inc_sar
                )
                sigma0_cmod5n_xsarsea = cmod5n_xsarsea(
                    _spd_coloc_ecmwf, self._azi_sar_coloc_ecmwf, _inc_sar
                )
                if self.polarisation == "H":
                    pr = get_polarization_ratio(
                        _spd_coloc_ecmwf,
                        _inc_sar,
                        phi=self._azi_sar_coloc_ecmwf,
                        name="mouche1",
                    )
                    sigma0_cmod5 = sigma0_cmod5 / pr
                    sigma0_cmod_ifr2 = sigma0_cmod_ifr2 / pr
                    sigma0_cmod5n = sigma0_cmod5n / pr
                    sigma0_cmod5n_xsarsea = sigma0_cmod5n_xsarsea / pr
                self.sigma0_cmod5 = sigma0_cmod5
                self.sigma0_cmod_ifr2 = sigma0_cmod_ifr2
                self.sigma0_cmod5n = sigma0_cmod5n
                self.sigma0_cmod5n_xsarsea = sigma0_cmod5n_xsarsea

                logger.info(
                    " Nber of value on land: %s on ocean: %s",
                    np.sum(_land),
                    np.sum(~_land),
                )
                logger.info(
                    " Nber of value high latitude : %s moderate latitudes: %s",
                    np.sum(_highlat_flag),
                    np.sum(~_highlat_flag),
                )
                filter_wv2 = (
                    (_inc_sar > 30)
                    & (_sig_sar > 0)
                    & (np.abs(_spd_coloc_ecmwf - wind_ranges[windrange_i]) < 3)
                    & ~_land
                    & ~_highlat_flag
                )
                filter_wv1 = (
                    (_inc_sar < 30)
                    & (_sig_sar > 0)
                    & (np.abs(_spd_coloc_ecmwf - wind_ranges[windrange_i]) < 3)
                    & ~_land
                    & ~_highlat_flag
                )
                self.inc_2_ = (_inc_sar > 30) & (_sig_sar > 0) & ~_land & ~_highlat_flag
                self.inc_1_ = (_inc_sar < 30) & (_sig_sar > 0) & ~_land & ~_highlat_flag

                self._cor_wv1_cmod5[windrange_i] = np.mean(
                    10 * np.log10(sigma0_cmod5[filter_wv1])
                    - 10 * np.log10(_sig_sar[filter_wv1])
                )
                self._cor_wv2_cmod5[windrange_i] = np.mean(
                    10 * np.log10(sigma0_cmod5[filter_wv2])
                    - 10 * np.log10(_sig_sar[filter_wv2])
                )
                self._cor_wv1_cmod5n[windrange_i] = np.mean(
                    10 * np.log10(sigma0_cmod5n[filter_wv1])
                    - 10 * np.log10(_sig_sar[filter_wv1])
                )
                self._cor_wv2_cmod5n[windrange_i] = np.mean(
                    10 * np.log10(sigma0_cmod5n[filter_wv2])
                    - 10 * np.log10(_sig_sar[filter_wv2])
                )
                self._cor_wv1_cmod5n_xsarsea[windrange_i] = np.mean(
                    10 * np.log10(sigma0_cmod5n_xsarsea[filter_wv1])
                    - 10 * np.log10(_sig_sar[filter_wv1])
                )
                self._cor_wv2_cmod5n_xsarsea[windrange_i] = np.mean(
                    10 * np.log10(sigma0_cmod5n_xsarsea[filter_wv2])
                    - 10 * np.log10(_sig_sar[filter_wv2])
                )
                logger.info(
                    "10*np.log10(sigma0_cmod_ifr2[filter_wv1]: %s",
                    10 * np.log10(sigma0_cmod_ifr2[filter_wv1]),
                )
                logger.info(
                    "10*np.log10(_sig_sar[filter_wv1])): %s",
                    10 * np.log10(_sig_sar[filter_wv1]),
                )
                self._cor_wv1[windrange_i] = np.mean(
                    10 * np.log10(sigma0_cmod_ifr2[filter_wv1])
                    - 10 * np.log10(_sig_sar[filter_wv1])
                )
                self._cor_wv2[windrange_i] = np.mean(
                    10 * np.log10(sigma0_cmod_ifr2[filter_wv2])
                    - 10 * np.log10(_sig_sar[filter_wv2])
                )

                windfilter1 = (_inc_sar > 30) & (_sig_sar > 0) & ~_land & ~_highlat_flag
                windfilter2 = (_inc_sar < 30) & (_sig_sar > 0) & ~_land & ~_highlat_flag
                self.cpt_wind_bias = collections.defaultdict(int)
                self.wind_bias = collections.defaultdict(int)
                if level == "L2":
                    tmp1 = []
                    tmp2 = []
                    safe_windspeed_biasfull1 = (
                        _spd_coloc_ecmwf[windfilter1] - _windspeed_sar[windfilter1]
                    )
                    safe_windspeed_biasfull2 = (
                        _spd_coloc_ecmwf[windfilter2] - _windspeed_sar[windfilter2]
                    )
                    safe_winddirection_biasfull1 = self._azi_sar_coloc_ecmwf[
                        windfilter1
                    ]
                    safe_winddirection_biasfull2 = self._azi_sar_coloc_ecmwf[
                        windfilter2
                    ]
                    logger.debug(
                        "length of safe_windspeed_biasfull2 %s",
                        len(safe_windspeed_biasfull2),
                    )
                    logger.debug(
                        "length of safe_windspeed_biasfull1 %s",
                        len(safe_windspeed_biasfull1),
                    )
                    for plage in wind_plages:
                        ws_filterd1 = safe_windspeed_biasfull1[
                            (_spd_coloc_ecmwf[windfilter1] > wind_plages[plage][0])
                            & (_spd_coloc_ecmwf[windfilter1] < wind_plages[plage][1])
                        ]
                        ws_filterd2 = safe_windspeed_biasfull2[
                            (_spd_coloc_ecmwf[windfilter2] > wind_plages[plage][0])
                            & (_spd_coloc_ecmwf[windfilter2] < wind_plages[plage][1])
                        ]
                        wd_filterd1 = safe_winddirection_biasfull1[
                            (_spd_coloc_ecmwf[windfilter1] > wind_plages[plage][0])
                            & (_spd_coloc_ecmwf[windfilter1] < wind_plages[plage][1])
                        ]
                        wd_filterd2 = safe_winddirection_biasfull2[
                            (_spd_coloc_ecmwf[windfilter2] > wind_plages[plage][0])
                            & (_spd_coloc_ecmwf[windfilter2] < wind_plages[plage][1])
                        ]
                        for wdd in wind_pref_directions:
                            bias_ws_dd_1 = ws_filterd1[
                                (wd_filterd1 > wind_pref_directions[wdd][0])
                                & (wd_filterd1 < wind_pref_directions[wdd][1])
                            ]
                            bias_ws_dd_2 = ws_filterd2[
                                (wd_filterd2 > wind_pref_directions[wdd][0])
                                & (wd_filterd2 < wind_pref_directions[wdd][1])
                            ]
                            mean_ws_dd_1 = np.nanmean(bias_ws_dd_1)
                            mean_ws_dd_2 = np.nanmean(bias_ws_dd_2)
                            self.wind_bias[plage + "-" + wdd + "1"] = mean_ws_dd_1
                            self.wind_bias[plage + "-" + wdd + "2"] = mean_ws_dd_2
                            self.cpt_wind_bias["1"] += len(bias_ws_dd_1)
                            self.cpt_wind_bias["2"] += len(bias_ws_dd_2)
                            self.cpt_wind_bias[plage + "-" + wdd + "1"] = len(
                                bias_ws_dd_1
                            )
                            self.cpt_wind_bias[plage + "-" + wdd + "2"] = len(
                                bias_ws_dd_2
                            )
                            tmp1.append(
                                self.cpt_wind_bias[plage + "-" + wdd + "1"]
                                * self.wind_bias[plage + "-" + wdd + "1"]
                            )
                            tmp2.append(
                                self.cpt_wind_bias[plage + "-" + wdd + "2"]
                                * self.wind_bias[plage + "-" + wdd + "2"]
                            )
                            logger.debug("bias_ws_dd_1 %s", bias_ws_dd_1)
                            logger.debug(
                                "wind speed bias for plage %s in %s, wv1=%s m/s wv2=%sm/s with (resp. %s and %s obs)",
                                plage,
                                wdd,
                                mean_ws_dd_1,
                                mean_ws_dd_2,
                                len(bias_ws_dd_1),
                                len(bias_ws_dd_2),
                            )

                    logger.debug(
                        "total obs for windbias %s",
                        self.cpt_wind_bias["1"] + self.cpt_wind_bias["2"],
                    )
                    logger.debug("cpt_wind_bias = %s", self.cpt_wind_bias)
                    logger.debug(
                        "ponderated wind bias wv1 %s m/s",
                        np.nansum(tmp1) / self.cpt_wind_bias["1"],
                    )
                    logger.debug(
                        "ponderated wind bias wv2 %s m/s",
                        np.nansum(tmp2) / self.cpt_wind_bias["2"],
                    )
                    logger.debug(
                        "previous bias (not ponderated) wv1 %s",
                        np.nanmean(
                            _spd_coloc_ecmwf[filter_wv1] - _windspeed_sar[filter_wv1]
                        ),
                    )
                    logger.debug(
                        "previous bias (not ponderated) wv2 %s",
                        np.nanmean(
                            _spd_coloc_ecmwf[filter_wv2] - _windspeed_sar[filter_wv2]
                        ),
                    )
                if level == "L2":
                    self.safe_windspeed_biasfull1 = np.nanmean(
                        _spd_coloc_ecmwf[windfilter1] - _windspeed_sar[windfilter1]
                    )
                    self.safe_windspeed_biasfull2 = np.nanmean(
                        _spd_coloc_ecmwf[windfilter2] - _windspeed_sar[windfilter2]
                    )
                else:
                    self.safe_windspeed_biasfull1 = np.nan
                    self.safe_windspeed_biasfull2 = np.nan

                swhfilter1 = (
                    (_inc_sar > 30)
                    & (_sig_sar > 0)
                    & ~_land
                    & ~_highlat_flag
                    & (np.isfinite(_hseff2d_ww3))
                )
                swhfilter2 = (
                    (_inc_sar < 30)
                    & (_sig_sar > 0)
                    & ~_land
                    & ~_highlat_flag
                    & (np.isfinite(_hseff2d_ww3))
                )
                self.cpt_swh_bias = collections.defaultdict(int)
                self.swh_bias = collections.defaultdict(int)
                logger.debug(
                    "data swh: %s", _hseff2d_ww3[swhfilter1] - _hseff2d_sar[swhfilter1]
                )
                self.swh_bias["wv1"] = np.mean(
                    _hseff2d_ww3[swhfilter1] - _hseff2d_sar[swhfilter1]
                )
                self.swh_bias["wv2"] = np.mean(
                    _hseff2d_ww3[swhfilter2] - _hseff2d_sar[swhfilter2]
                )
                self.cpt_swh_bias["wv1"] = len(_hseff2d_ww3[swhfilter1])
                self.cpt_swh_bias["wv2"] = len(_hseff2d_ww3[swhfilter2])
                logger.debug('self.cpt_swh_bias["wv1"] %s', self.cpt_swh_bias["wv1"])

                self.mean_inc_1_corr[windrange_i] = np.mean(_inc_sar[filter_wv1])
                self.mean_inc_2_corr[windrange_i] = np.mean(_inc_sar[filter_wv2])
                self.mean_sig_1_corr[windrange_i] = np.mean(_sig_sar[filter_wv1])
                self.mean_sig_2_corr[windrange_i] = np.mean(_sig_sar[filter_wv2])
                self.nb_inc1 = len(_inc_sar[self.inc_1_])
                self.nb_inc2 = len(_inc_sar[self.inc_2_])
                self.nb_inc1_corr[windrange_i] = len(_inc_sar[filter_wv1])
                self.nb_inc2_corr[windrange_i] = len(_inc_sar[filter_wv2])

                logger.info(
                    "Nber of points in the wind sweet spot: WV1 = %s WV2= %s",
                    self.nb_inc1_corr,
                    self.nb_inc2_corr,
                )
                logger.info("WV1 cmod ifr: %s", self._cor_wv1)
                logger.info("WV2 cmod ifr: %s", self._cor_wv2)
                logger.info("WV1 _cmod5: %s", self._cor_wv1_cmod5)
                logger.info("WV2 _cmod5: %s", self._cor_wv2_cmod5)

                final_result["cor_wv1_cmodifr"] = self._cor_wv1
                final_result["cor_wv2_cmodifr"] = self._cor_wv2
                final_result["cor_wv1_cmod5"] = self._cor_wv1_cmod5
                final_result["cor_wv2_cmod5"] = self._cor_wv2_cmod5

            self.path_out.mkdir(parents=True, mode=0o775, exist_ok=True)
            save_calib_file = True
            self.finalpath = self.path_out / self.file_out_netcdf
            if self.finalpath.exists() and redo is False:
                save_calib_file = False
                logger.info(
                    "redo option is False and file already exist -> skip re writing the netcdf output"
                )
            if save_calib_file:
                self.Write2NetCDFWVsentinel1CalValParams3()
                logger.debug("step:plot NRCS vs wind")
                filout, fig = PlotNRCSvsWindPerSafe(
                    self._spd_coloc_ecmwf,
                    self.inc_1_,
                    self.inc_2_,
                    self._cor_wv1_cmod5["classical"],
                    self._cor_wv2_cmod5["classical"],
                    self.orbit_name,
                    self.sigma0_cmod5,
                    self._sig_sar,
                    self.path_out,
                    self.polarisation,
                    suffix_output=suffix_output,
                    gmf_name="cmod5",
                )
                SaveNRCSvsWindPlot(filout, fig)
                filout, fig = PlotNRCSvsWindPerSafe(
                    self._spd_coloc_ecmwf,
                    self.inc_1_,
                    self.inc_2_,
                    self._cor_wv1["classical"],
                    self._cor_wv2["classical"],
                    self.orbit_name,
                    self.sigma0_cmod_ifr2,
                    self._sig_sar,
                    self.path_out,
                    self.polarisation,
                    suffix_output=suffix_output,
                    gmf_name="cmod-ifr2",
                )
                SaveNRCSvsWindPlot(filout, fig)
                filout, fig = PlotNRCSvsWindPerSafe(
                    self._spd_coloc_ecmwf,
                    self.inc_1_,
                    self.inc_2_,
                    self._cor_wv1_cmod5n["classical"],
                    self._cor_wv2_cmod5n["classical"],
                    self.orbit_name,
                    self.sigma0_cmod5n,
                    self._sig_sar,
                    self.path_out,
                    self.polarisation,
                    suffix_output=suffix_output,
                    gmf_name="cmod5n",
                )
                SaveNRCSvsWindPlot(filout, fig)
            logger.info(
                "There is %s measurement that do not have finite lon/lat",
                cpt_not_finite_position,
            )
            logger.info("counters: %s", cpt_counter)
        else:
            flag_data_available = False

        return final_result, flag_data_available

    def Write2NetCDFWVsentinel1CalValParams3(self) -> None:
        """Write the calibration parameters to a NetCDF file using xarray."""
        nber = len(self._sig_sar)
        finalpath = self.path_out / self.file_out_netcdf
        logger.info("Attempting to write NetCDF file: %s", finalpath)

        if finalpath.exists():
            logger.debug("Removing existing file: %s", finalpath)
            finalpath.unlink()

        data_vars: dict = {}
        data_vars["time_sar"] = (("number_of_wm_acq",), self._dates_sar)
        data_vars["sig_sar"] = (("number_of_wm_acq",), self._sig_sar.astype(np.float32))
        data_vars["sig_sar_denoised"] = (
            ("number_of_wm_acq",),
            self._sig_sar_denoised.astype(np.float32),
        )
        data_vars["noise_sar"] = (
            ("number_of_wm_acq",),
            self._noise_sar.astype(np.float32),
        )
        data_vars["sig_mod"] = (
            ("number_of_wm_acq",),
            self.sigma0_cmod5n.astype(np.float32),
        )
        data_vars["sig_mod_xsarsea"] = (
            ("number_of_wm_acq",),
            self.sigma0_cmod5n_xsarsea.astype(np.float32),
        )
        data_vars["land"] = (("number_of_wm_acq",), self._land.astype(np.float32))
        data_vars["track_angle"] = (
            ("number_of_wm_acq",),
            self._tra_sar.astype(np.float32),
        )
        data_vars["incidence_angle"] = (
            ("number_of_wm_acq",),
            self._inc_sar.astype(np.float32),
        )
        data_vars["elevation_angle"] = (
            ("number_of_wm_acq",),
            self._elev_sar.astype(np.float32),
        )
        data_vars["doppler_centroid"] = (
            ("number_of_wm_acq",),
            self._doppler_centroid.astype(np.float32),
        )
        data_vars["doppler_centroid_quality"] = (
            ("number_of_wm_acq",),
            # The legacy wrote this float array into a netCDF4 byte variable,
            # where unavailable (NaN, e.g. OCN SAFEs) became 0. Reproduce that
            # explicitly: a bare .astype(np.int8) of NaN warns (and errors
            # under -Werror) and is not guaranteed to yield 0.
            np.nan_to_num(self._quality_doppler_centroid, nan=0.0).astype(np.int8),
            {
                "descr": "True means the RMS of doppler centroid fit with estimates is below the threshold"
            },
        )
        data_vars["lon_sar"] = (("number_of_wm_acq",), self._lon_sar.astype(np.float32))
        data_vars["lat_sar"] = (("number_of_wm_acq",), self._lat_sar.astype(np.float32))
        data_vars["nv_sar"] = (("number_of_wm_acq",), self._nv_sar.astype(np.float32))
        data_vars["skew_sar"] = (
            ("number_of_wm_acq",),
            self._mom_stat3_sar_nrcs.astype(np.float32),
        )
        data_vars["kurt_sar"] = (
            ("number_of_wm_acq",),
            self._mom_stat4_sar_nrcs.astype(np.float32),
            {"descr": "kurtosis"},
        )

        rough_class_sar_bytes = np.array(
            [s.encode("utf-8").ljust(3)[:3] for s in self._classification_sar],
            dtype="|S3",
        )
        data_vars["rough_class_sar"] = (("number_of_wm_acq",), rough_class_sar_bytes)

        data_vars["spd_ecmwf"] = (
            ("number_of_wm_acq",),
            self._spd_coloc_ecmwf.astype(np.float32),
        )
        data_vars["dir_ecmwf"] = (
            ("number_of_wm_acq",),
            self._dir_coloc_ecmwf.astype(np.float32),
        )
        data_vars["zon_ecmwf"] = (
            ("number_of_wm_acq",),
            self._zon_coloc_ecmwf.astype(np.float32),
        )
        data_vars["mer_ecmwf"] = (
            ("number_of_wm_acq",),
            self._mer_coloc_ecmwf.astype(np.float32),
        )
        data_vars["lon_ecmwf"] = (
            ("number_of_wm_acq",),
            self._lon_coloc_ecmwf.astype(np.float32),
        )
        data_vars["lat_ecmwf"] = (
            ("number_of_wm_acq",),
            self._lat_coloc_ecmwf.astype(np.float32),
        )
        data_vars["dlon_ecmwf"] = (
            ("number_of_wm_acq",),
            self.dlon_coloc_ecmwf.astype(np.float32),
        )
        data_vars["dlat_ecmwf"] = (
            ("number_of_wm_acq",),
            self.dlat_coloc_ecmwf.astype(np.float32),
        )

        if self.wind_bias is not None:
            for plage in wind_plages:
                for wdd in wind_pref_directions:
                    data_vars[f"windspeed_bias_{plage}_{wdd}_wv1"] = (
                        ("number_of_wm_correction",),
                        np.array([self.wind_bias[f"{plage}-{wdd}1"]]).astype(
                            np.float32
                        ),
                    )
                    data_vars[f"windspeed_bias_{plage}_{wdd}_wv2"] = (
                        ("number_of_wm_correction",),
                        np.array([self.wind_bias[f"{plage}-{wdd}2"]]).astype(
                            np.float32
                        ),
                    )
                    data_vars[f"nbobs_windspeed_bias_{plage}_{wdd}_wv1"] = (
                        ("number_of_wm_correction",),
                        np.array([self.cpt_wind_bias[f"{plage}-{wdd}1"]]).astype(
                            np.float32
                        ),
                    )
                    data_vars[f"nbobs_windspeed_bias_{plage}_{wdd}_wv2"] = (
                        ("number_of_wm_correction",),
                        np.array([self.cpt_wind_bias[f"{plage}-{wdd}2"]]).astype(
                            np.float32
                        ),
                    )

        data_vars["windspeed_bias_full_wv1"] = (
            ("number_of_wm_correction",),
            np.array([self.safe_windspeed_biasfull1]).astype(np.float32),
        )
        data_vars["windspeed_bias_full_wv2"] = (
            ("number_of_wm_correction",),
            np.array([self.safe_windspeed_biasfull2]).astype(np.float32),
        )

        data_vars["mean_swh_2d_eff_bias_ww3spec_wv1"] = (
            ("number_of_wm_correction",),
            np.array([self.swh_bias["wv1"]]).astype(np.float32),
        )
        data_vars["mean_swh_2d_eff_bias_ww3spec_ww3spec_wv2"] = (
            ("number_of_wm_correction",),
            np.array([self.swh_bias["wv2"]]).astype(np.float32),
        )
        data_vars["nb_swh_2d_eff_bias_ww3spec_wv1"] = (
            ("number_of_wm_correction",),
            np.array([self.cpt_swh_bias["wv1"]]).astype(np.float32),
        )
        data_vars["nb_swh_2d_eff_bias_ww3spec_wv2"] = (
            ("number_of_wm_correction",),
            np.array([self.cpt_swh_bias["wv2"]]).astype(np.float32),
        )

        data_vars["correction_wv1"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv1["classical"]]).astype(np.float32),
        )
        data_vars["correction_wv2"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv2["classical"]]).astype(np.float32),
        )
        data_vars["correction_wv1_cmod5"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv1_cmod5["classical"]]).astype(np.float32),
        )
        data_vars["correction_wv2_cmod5"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv2_cmod5["classical"]]).astype(np.float32),
        )
        data_vars["correction_wv1_cmod5n"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv1_cmod5n["classical"]]).astype(np.float32),
            {"descr": "GMF - SAR obs", "unit": "dB"},
        )
        data_vars["correction_wv2_cmod5n"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv2_cmod5n["classical"]]).astype(np.float32),
            {"descr": "GMF - SAR obs", "unit": "dB"},
        )
        data_vars["correction_wv1_cmod5n_xsarsea"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv1_cmod5n_xsarsea["classical"]]).astype(np.float32),
            {"descr": "GMF(xsarsea) - SAR obs", "unit": "dB"},
        )
        data_vars["correction_wv2_cmod5n_xsarsea"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv2_cmod5n_xsarsea["classical"]]).astype(np.float32),
            {"descr": "GMF(xsarsea) - SAR obs", "unit": "dB"},
        )
        data_vars["nbptsNRCSbias_wv1"] = (
            ("number_of_wm_correction",),
            np.array([self.nb_inc1_corr["classical"]]).astype(np.float32),
        )
        data_vars["nbptsNRCSbias_wv2"] = (
            ("number_of_wm_correction",),
            np.array([self.nb_inc2_corr["classical"]]).astype(np.float32),
        )

        data_vars["correction_wv1_10ws"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv1["noise_free"]]).astype(np.float32),
        )
        data_vars["correction_wv2_10ws"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv2["noise_free"]]).astype(np.float32),
        )
        data_vars["correction_wv1_cmod5_10ws"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv1_cmod5["noise_free"]]).astype(np.float32),
        )
        data_vars["correction_wv2_cmod5_10ws"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv2_cmod5["noise_free"]]).astype(np.float32),
        )
        data_vars["correction_wv1_cmod5n_10ws"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv1_cmod5n["noise_free"]]).astype(np.float32),
        )
        data_vars["correction_wv2_cmod5n_10ws"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv2_cmod5n["noise_free"]]).astype(np.float32),
        )
        data_vars["correction_wv1_cmod5n_xsarsea_10ws"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv1_cmod5n_xsarsea["noise_free"]]).astype(np.float32),
        )
        data_vars["correction_wv2_cmod5n_xsarsea_10ws"] = (
            ("number_of_wm_correction",),
            np.array([self._cor_wv2_cmod5n_xsarsea["noise_free"]]).astype(np.float32),
        )
        data_vars["nbptsNRCSbias_wv1_10ws"] = (
            ("number_of_wm_correction",),
            np.array([self.nb_inc1_corr["noise_free"]]).astype(np.float32),
        )
        data_vars["nbptsNRCSbias_wv2_10ws"] = (
            ("number_of_wm_correction",),
            np.array([self.nb_inc2_corr["noise_free"]]).astype(np.float32),
        )

        polarisation_bytes = np.array([self.polarisation.encode("utf-8")], dtype="|S1")
        data_vars["sar_imagette_polarization"] = (
            ("number_of_wm_correction",),
            polarisation_bytes,
        )

        ds = xr.Dataset(
            data_vars,
            coords={
                "number_of_wm_acq": np.arange(nber),
                "number_of_wm_correction": np.arange(1),
                "classifchars": np.arange(3),
            },
        )
        ds.attrs["producer"] = "CERSAT"
        ds.attrs["project"] = "MPC-sentinel1"
        ds.attrs["time_unit"] = _time_unit()
        ds.to_netcdf(finalpath, engine="h5netcdf")
        logger.info("NetCDF file successfully written to %s", finalpath)
        try:
            finalpath.chmod(0o0775)
        except PermissionError as e:
            logger.info("Could not change file permissions on %s: %s", finalpath, e)


def PlotNRCSvsWindPerSafe(
    _spd_coloc_ecmwf: Any,
    inc_1_: Any,
    inc_2_: Any,
    _cor_wv1: float,
    _cor_wv2: float,
    orbit_name: str,
    sigma0_gmf: Any,
    _sig_sar: Any,
    path_out_netcdf: Any,
    pol: str,
    fighandler: Any = None,
    suffix_output: str = "",
    gmf_name: str = "",
) -> tuple[Path, Any]:
    """Constant calval SAR figure (published aside of the correction constant)."""
    logger.info("NRCS plot Sentinel-1 SAFE file: %s", orbit_name)
    satellite = orbit_name[0:3]
    date_orbit1 = orbit_name[17:25]
    hour_orbit1 = orbit_name[26:32]
    date_orbit2 = orbit_name[33:41]
    hour_orbit2 = orbit_name[42:48]

    # Naive: display only (figure title); keeps the title identical to legacy.
    date_sar1 = datetime.strptime(date_orbit1 + "_" + hour_orbit1, "%Y%m%d_%H%M%S")  # noqa: DTZ007
    date_sar2 = datetime.strptime(date_orbit2 + "_" + hour_orbit2, "%Y%m%d_%H%M%S")  # noqa: DTZ007
    fig = fighandler if fighandler is not None else plt.figure()
    plt.plot(
        _spd_coloc_ecmwf[inc_1_],
        10 * np.log10(sigma0_gmf[inc_1_]),
        ".b",
        label="model WV1",
    )
    plt.plot(
        _spd_coloc_ecmwf[inc_2_],
        10 * np.log10(sigma0_gmf[inc_2_]),
        ".r",
        label="model WV2",
    )
    plt.plot(
        _spd_coloc_ecmwf[inc_1_],
        10 * np.log10(_sig_sar[inc_1_]),
        "ob",
        label="SAR WV1",
        markeredgecolor="k",
    )
    plt.plot(
        _spd_coloc_ecmwf[inc_2_],
        10 * np.log10(_sig_sar[inc_2_]),
        "or",
        label="SAR WV2",
        markeredgecolor="k",
    )
    plt.legend(loc="lower left")
    plt.grid()
    plt.axvspan(7.0 - 3.0, 7.0 + 3.0, facecolor="g", alpha=0.25)

    plt.annotate(
        f"diff model-SAR WV2: {_cor_wv2:0.5} dB", xy=(11, -32.5), color=("red")
    )
    plt.annotate(
        f"diff model-SAR WV1: {_cor_wv1:0.5} dB", xy=(11, -35.0), color=("blue")
    )

    plt.xlabel("Wind Speed ECMWF [m/s]")
    # S-1 WV is co-polarized: single-char "V"/"H" -> full "VV"/"HH" in the label.
    plt.ylabel(pol + pol + " NRCS [dB]")
    plt.title(
        satellite + " - " + str(date_sar1) + " - " + str(date_sar2) + " " + gmf_name
    )
    plt.ylim((-40, 5))
    plt.xlim((0, max([max(_spd_coloc_ecmwf), 15])))

    file_out = Path(path_out_netcdf) / (
        "wind_vs_sig_calibrated_"
        + str(gmf_name)
        + "_"
        + orbit_name[:-5]
        + str(suffix_output)
        + ".png"
    )
    return file_out, fig


def SaveNRCSvsWindPlot(file_out: Path, fig: Any, dpi: int = 150) -> None:
    """Save the NRCS vs wind figure and set group-readable permissions."""
    fig.savefig(file_out, dpi=dpi)
    try:
        Path(file_out).chmod(0o0775)
    except PermissionError:
        logger.info("cannot change permission on %s", file_out)


def sequentialListOfSAfeAnalyse(
    listSAFE: list[str], msk: xr.Dataset, path_out: str
) -> None:
    """Analyse a list of SAFE sequentially (writes NetCDF only, no pickle)."""
    cptu: dict[str, int] = collections.defaultdict(int)
    logger.info(
        "sequentialListOfSAfeAnalyse | number of safe to analyse %s", len(listSAFE)
    )
    for safe_file in listSAFE:
        clean_safe = safe_file.strip("\n") if "\n" in safe_file else safe_file
        logger.debug("safe %s", clean_safe)
        obj = WVcalibrator()
        _data_res, flag_data_available = obj.AnalyseOneSAFE(clean_safe, msk, path_out)
        if not flag_data_available:
            cptu["KO_SAFE"] += 1
        else:
            cptu["OK_SAFE"] += 1
    logger.info("sequential processing score: %s", cptu)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the WV geophysical calibration constant for a Sentinel-1 SAFE."
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Output directory where to save the PNG and NetCDF files (optional).",
    )
    parser.add_argument(
        "-i",
        "--input-safe",
        dest="input",
        required=True,
        help="Input SAFE product to analyze.",
    )
    parser.add_argument(
        "--maxlat",
        dest="maxlat",
        type=int,
        help="Maximum |latitude| (deg) to exclude marginal ice areas (optional).",
    )
    parser.add_argument(
        "--ecmwf",
        dest="ecmwf",
        type=str,
        choices=["0125", "0100"],
        help="ECMWF version: '0125' (3-hourly) or '0100' (hourly) [default: 0100].",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite an existing NetCDF calibration file (default False).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Development mode: process only a few imagettes (faster).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose (DEBUG) logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        int: Process exit code (0 on success).
    """
    args = _build_parser().parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    in_safe_file = args.input
    logger.info("in_safe_file = %s", in_safe_file)
    if in_safe_file[-1] == "/":
        in_safe_file = in_safe_file[0:-1]
    satellite_acro = Path(in_safe_file).name[0:3]

    if args.output is not None:
        path_out = args.output
    else:
        analysis_dir = Path(data_path("analysis_dir"))
        path_out = analysis_dir / "L1_v16/WV/" / (satellite_acro + "_WV_SLC__1S")
        if "OCN" in in_safe_file:
            path_out = analysis_dir / "L2/WV/" / (satellite_acro + "_WV_OCN__2S")
        logger.info("default output dir (from config analysis_dir): %s", path_out)

    land_mask = Path(data_path("land_mask"))
    msk = xr.open_dataset(land_mask / "mwf-ers-mask.nc")
    instance = WVcalibrator()

    suffix = "maxlat_" + str(args.maxlat) if args.maxlat else ""
    latlat = _high_lat_threshold() if args.maxlat is None else args.maxlat
    ecmwf_vers = args.ecmwf or "0100"
    if args.ecmwf is None:
        logger.info("default ECMWF version is 0100")

    _data_res, flag_data_available = instance.AnalyseOneSAFE(
        in_safe_file,
        msk,
        path_out,
        max_latitude=latlat,
        suffix_output=suffix,
        include_zero_s0=True,
        ecmwf_version=ecmwf_vers,
        redo=args.overwrite,
        dev=args.dev,
    )
    logger.info("netcdf and plot(png) have been generated")
    logger.info("flag_data_available = %s", flag_data_available)
    return 0


def entrypoint() -> None:
    """Console-script wrapper for ``s1wvcalib``."""
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
