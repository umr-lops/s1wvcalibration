# s1wvcalibration

```{toctree}
:maxdepth: 2
:hidden:

atbd
api
```

```{include} ../README.md
:start-after: <!-- SPHINX-START -->
```

## What a calibration figure looks like

For every analysed SAFE the pipeline stores, alongside the per-SAFE NetCDF
calibration file, a PNG figure of the (denoised) SAR NRCS and the GMF-predicted
NRCS against the colocated ECMWF wind speed, per sub-swath (WV1 / WV2). The
vertical green band marks the classical 7 m/s analysis window. The vertical
offset between the model and SAR points is the calibration constant.

```{image} _static/wind_vs_sig_calibrated_cmod5n.png
:width: 640px
:alt: NRCS vs wind speed for the S1C_WV_OCN__2SSV SAFE of 2026-05-15
```

_Real operational figure (CMOD5n) produced by `s1wvfig` from the calibration
NetCDF of `S1C_WV_OCN__2SSV_20260515T140206` (V-polarized, VV). The y-axis
reports the measured co-polarized NRCS._

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
