"""Regenerate the per-SAFE NRCS-vs-wind-speed figures from a calibration NetCDF.

The calibration NetCDF produced by :func:`s1wvcalibration.calibration.entrypoint`
stores every per-imagette quantity needed to re-plot the published
``wind_vs_sig_calibrated_<gmf>_<safe>.png`` figures without re-running the
whole (expensive) pipeline: the SAR backscatter (``sig_sar``), the colocated
ECMWF wind (``spd_ecmwf``, ``zon_ecmwf``, ``mer_ecmwf``), the geometry
(``incidence_angle``, ``lat_sar``, ``track_angle``) and the quality flags
(``land``). This module recomputes the model backscatter (cmod5 / cmod-ifr2 /
cmod5n, including the H-polarisation ratio) from those fields and re-issues the
same three figures via :func:`~s1wvcalibration.calibration.PlotNRCSvsWindPerSafe`.

Exposed as the ``s1wvfig`` console script (see ``[project.scripts]``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ._legacy.compute_wind_azimuth import compute_azimuth_angle
from ._legacy.gmf import cmod5, cmod5n_forward, cmod_ifr2
from ._legacy.polarization_ratio import get_polarization_ratio
from .calibration import (
    PlotNRCSvsWindPerSafe,
    SaveNRCSvsWindPlot,
    _high_lat_threshold,
    _wind_ranges,
)

logger = logging.getLogger(__name__)

__all__ = ["entrypoint", "main", "make_nrcs_vs_wind_figure", "replot_calibration"]

#: gmf_name -> (GMF callable, output sigma0 variable used in the figure).
_GMF_CALLABLES: dict[str, Any] = {
    "cmod5": cmod5,
    "cmod-ifr2": cmod_ifr2,
    "cmod5n": cmod5n_forward,
}


def _pol_str(raw: object) -> str:
    """Decode the ``sar_imagette_polarization`` ``|S1`` byte to a single char."""
    if isinstance(raw, (bytes, bytearray)):
        return raw[:1].decode("utf-8")
    if isinstance(raw, str):
        return raw[:1]
    return str(np.asarray(raw).reshape(-1)[0][:1])


def _sweet_spot_correction(
    sigma0_gmf: np.ndarray,
    sig_sar: np.ndarray,
    filter_mask: np.ndarray,
) -> float:
    """Mean ``GMF - SAR`` (dB) over the sweet-spot points (empty -> NaN)."""
    m = filter_mask & (np.asarray(sigma0_gmf) > 0) & (np.asarray(sig_sar) > 0)
    if int(np.sum(m)) == 0:
        return float("nan")
    gmf_db = 10.0 * np.log10(np.asarray(sigma0_gmf)[m])
    sar_db = 10.0 * np.log10(np.asarray(sig_sar)[m])
    return float(np.mean(gmf_db - sar_db))


def replot_calibration(nc_path: str, out_dir: str | None = None) -> list[Path]:
    """Regenerate the NRCS-vs-wind figures for a calibration NetCDF.

    Args:
        nc_path: Path to a per-SAFE calibration NetCDF (output of ``s1wvcalib``).
        out_dir: Directory for the PNGs. Defaults to the directory of *nc_path*.

    Returns:
        The list of written figure paths (one per GMF).
    """
    ds = xr.open_dataset(nc_path)
    try:
        sig_sar = np.asarray(ds["sig_sar"].values, dtype=np.float64)
        inc = np.asarray(ds["incidence_angle"].values, dtype=np.float64)
        lat = np.asarray(ds["lat_sar"].values, dtype=np.float64)
        land = np.asarray(ds["land"].values, dtype=np.float64) > 0.5
        spd = np.asarray(ds["spd_ecmwf"].values, dtype=np.float64)
        zon = np.asarray(ds["zon_ecmwf"].values, dtype=np.float64)
        mer = np.asarray(ds["mer_ecmwf"].values, dtype=np.float64)
        track = np.asarray(ds["track_angle"].values, dtype=np.float64)
        pol = _pol_str(ds["sar_imagette_polarization"].values[0])
    finally:
        ds.close()

    # The NetCDF is named ``<SAFE><suffix>.nc``; the engine's figures are named
    # from the bare SAFE. Truncate back to the SAFE (ends in ".SAFE" when
    # present) so the reproduced figure paths match the engine's exactly.
    orbit_name = Path(nc_path).name.removesuffix(".nc")
    idx = orbit_name.rfind(".SAFE")
    if idx != -1:
        orbit_name = orbit_name[: idx + len(".SAFE")]

    path_out = Path(out_dir) if out_dir is not None else Path(nc_path).parent

    # Reconstruct the engine filters (identical to WVcalibrator.AnalyseOneSAFE).
    max_latitude = _high_lat_threshold()
    highlat = np.abs(lat) >= max_latitude
    valid = (sig_sar > 0) & ~land & ~highlat
    inc_1 = (inc < 30) & valid
    inc_2 = (inc > 30) & valid

    # Recompute the antenna-referenced wind azimuth (same call as the engine).
    azi = compute_azimuth_angle(track, zon, mer)

    # Same 7 +/- 3 m/s sweet-spot band as the engine (classical wind range).
    classical = float(_wind_ranges()["classical"])
    sweet = (np.abs(spd - classical) < 3) & valid

    # Polarisation ratio is applied to the model backscatter for H pol.
    pr: np.ndarray | None = None
    if pol == "H":
        pr = get_polarization_ratio(spd, inc, phi=azi, name="mouche1")

    def _gmf(name: str, fn: Any) -> np.ndarray:
        # cmod_ifr2 takes (inc, v, phi); the others take (v, phi, inc).
        sigma = fn(inc, spd, azi) if name == "cmod-ifr2" else fn(spd, azi, inc)
        return sigma / pr if pr is not None else sigma

    written: list[Path] = []
    for gmf_name, fn in _GMF_CALLABLES.items():
        sigma0 = _gmf(gmf_name, fn)
        cor_wv1 = _sweet_spot_correction(sigma0, sig_sar, inc_1 & sweet)
        cor_wv2 = _sweet_spot_correction(sigma0, sig_sar, inc_2 & sweet)
        filout, fig = PlotNRCSvsWindPerSafe(
            spd,
            inc_1,
            inc_2,
            cor_wv1,
            cor_wv2,
            orbit_name,
            sigma0,
            sig_sar,
            path_out,
            pol,
            gmf_name=gmf_name,
        )
        SaveNRCSvsWindPlot(filout, fig)
        plt.close(fig)
        logger.info("wrote figure: %s", filout)
        written.append(filout)
    return written


def make_nrcs_vs_wind_figure(nc_path: str, out_dir: str | None = None) -> list[Path]:
    """Alias of :func:`replot_calibration` (kept for a shorter, greppable name)."""
    return replot_calibration(nc_path, out_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the NRCS-vs-wind-speed figures (cmod5, cmod-ifr2, cmod5n) "
            "from a per-SAFE calibration NetCDF."
        )
    )
    parser.add_argument(
        "-i",
        "--input-nc",
        dest="input",
        required=True,
        help="Input calibration NetCDF (output of s1wvcalib).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Output directory for the PNGs (default: the NetCDF's directory).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose (DEBUG) logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``s1wvfig``.

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

    in_nc = args.input.removesuffix("/")
    if not Path(in_nc).is_file():
        logger.error("input NetCDF not found: %s", in_nc)
        return 1

    written = replot_calibration(in_nc, args.output)
    logger.info("wrote %d figure(s)", len(written))
    return 0


def entrypoint() -> None:
    """Console-script wrapper for ``s1wvfig``."""
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
