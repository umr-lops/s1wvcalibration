# s1wvcalibration

Geophysical calibration of Sentinel-1 Wave Mode (WV) observations.

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![Coverage][coverage-badge]][coverage-link]

<!-- SPHINX-START -->

## Purpose

**s1wvcalibration** computes a geophysical calibration constant (a bias in dB)
for Sentinel-1 Wave Mode acquisitions. WV mode has no corner reflectors or
transponders, so the calibration is _relative_: each 20x20 km WV imagette is
compared against an empirical Geophysical Model Function (GMF) — CMOD5n — fed
with colocated ECMWF winds. The difference between the (denoised) SAR NRCS and
the GMF-predicted NRCS is the calibration bias, computed per sub-swath (WV1 /
WV2).

The library is a clean-room re-packaging of the long-running Ifremer
MPC-Sentinel1 WV calibration pipeline (`computeWVconstantecalibration.py`).

### Reading the products

- **Level-1 SLC** (`.tiff` measurement) — read with
  [xsar](https://pypi.org/project/xsar/), which replaces the legacy
  `cerbere`/`sarimage` mapper.
- **Level-2 OCN** (`.nc` measurement) — read directly with xarray.

### GMF (NRCS model)

The GMF NRCS is computed with [xsarsea](https://pypi.org/project/xsarsea/)
(`windspeed.get_model("cmod5n")`), which replaces the legacy `sar.utils.gmf`.

### Configuration

All cluster-specific paths and calibration constants live in a packaged
`config.yaml`, overridable by a `localconfig.yaml` (or `$S1WVCALIB_CONFIG`).
This replaces the legacy `shared_information` module.

### Dependencies

| Package                                      | Role                                            |
| -------------------------------------------- | ----------------------------------------------- |
| [xsar](https://pypi.org/project/xsar/)       | Sentinel-1 Level-1 SLC reader                   |
| [xsarsea](https://pypi.org/project/xsarsea/) | GMF (cmod5n / cmod5 / cmodifremer) NRCS models  |
| numpy, scipy                                 | numerical core                                  |
| xarray, cftime, h5netcdf                     | NetCDF I/O (all NetCDF reads/writes via xarray) |

## Installation

```bash
pip install s1wvcalibration
```

## Usage

Run the calibration for a single SAFE:

```bash
s1wvcalib \
  --input-safe /path/to/S1A_WV_SLC__1SSV_...SAFE \
  --ecmwf 0100 \
  --overwrite
```

Full CLI reference:

```
usage: s1wvcalib [-h] [-o OUTPUT] -i INPUT [--maxlat MAXLAT]
                 [--ecmwf {0125,0100}] [--overwrite] [--dev] [-v]
```

Per-SAFE NetCDF calibration files are written under the configured
`analysis_dir` (see `config.yaml`).

### Regenerating the figures from a calibration NetCDF

The NRCS-vs-wind-speed figures (one per GMF: cmod5, cmod-ifr2, cmod5n) are
written next to the NetCDF by `s1wvcalib`. To re-issue them without re-running
the (expensive) calibration, use `s1wvfig` on an existing per-SAFE NetCDF:

```bash
s1wvfig \
  --input-nc /path/to/calib/S1A_WV_SLC__1SSV_...SAFE.nc \
  --output /path/to/figures
```

It reads the per-imagette wind / incidence / backscatter stored in the NetCDF,
recomputes the model backscatter (including the H-polarisation ratio), and
writes the same `wind_vs_sig_calibrated_<gmf>_<safe>.png` files.

## Configuration

The packaged `config.yaml` holds the `root_directory` (under which all the
`paths.*` entries are resolved) and the calibration constants (high-latitude
threshold, wind ranges, ECMWF version, ...). To point the library at your
cluster's data tree, drop a `localconfig.yaml` next to it (or set
`$S1WVCALIB_CONFIG` to its path); it is deep-merged over the defaults, so a
single-leaf override is enough:

```yaml
# localconfig.yaml
paths:
  root_directory: /path/to/mpc-sentinel1
```

## Project structure

```
s1wvcalibration/
├── src/
│   └── s1wvcalibration/
│       ├── __init__.py        # Package entry point, exposes __version__
│       ├── calibration.py     # Core calibration engine + `s1wvcalib` CLI
│       ├── figure.py          # Re-plot figures from a NetCDF + `s1wvfig` CLI
│       ├── config.py          # config.yaml loading (replaces shared_information)
│       ├── config.yaml        # Default paths & calibration constants
│       └── _legacy/           # Modules ported from mpcsentinellibs
│                              # (gmf, xsar_reader, ecmwf, ww3, ...)
├── tests/
│   ├── test_package.py        # Package metadata tests
│   ├── test_gmf_parity.py     # GMF / azimuth parity vs the legacy code
│   ├── test_figure.py         # Figure replot entrypoint (s1wvfig)
│   ├── test_l1_parity.py      # L1 reader parity vs a production NetCDF
│   └── test_netcdf_parity.py  # End-to-end NetCDF parity
├── docs/                      # Sphinx documentation
├── pyproject.toml
└── README.md
```

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/umr-lops/s1wvcalibration/workflows/CI/badge.svg
[actions-link]:             https://github.com/umr-lops/s1wvcalibration/actions
[pypi-link]:                https://pypi.org/project/s1wvcalibration/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/s1wvcalibration
[pypi-version]:             https://img.shields.io/pypi/v/s1wvcalibration
[rtd-badge]:                https://readthedocs.org/projects/s1wvcalibration/badge/?version=latest
[rtd-link]:                 https://s1wvcalibration.readthedocs.io/en/latest/?badge=latest
[coverage-badge]:           https://codecov.io/github/umr-lops/s1wvcalibration/branch/main/graph/badge.svg
[coverage-link]:            https://codecov.io/github/umr-lops/s1wvcalibration

<!-- prettier-ignore-end -->
