# s1wvcalibration

```{toctree}
:maxdepth: 2
:hidden:

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
:alt: NRCS vs wind speed for a Sentinel-1 WV SAFE (illustrative)
```

_Illustrative figure (synthetic SAR points around the real xsarsea CMOD5n
curve); the operational figure uses the measured imagettes of the SAFE._

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
