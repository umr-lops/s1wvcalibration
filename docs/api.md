# API reference

## Public API

### `s1wvcalibration.calibration`

The calibration engine and the `s1wvcalib` CLI entry point.

```{autoclass} s1wvcalibration.calibration.WVcalibrator
   :members:
   :member-order: bysource
```

```{autofunction} s1wvcalibration.calibration.is_land
```

```{autofunction} s1wvcalibration.calibration.PlotNRCSvsWindPerSafe
```

```{autofunction} s1wvcalibration.calibration.SaveNRCSvsWindPlot
```

```{autofunction} s1wvcalibration.calibration.sequentialListOfSAfeAnalyse
```

```{autofunction} s1wvcalibration.calibration.main
```

### `s1wvcalibration.config`

Configuration loading (replaces the legacy `shared_information` module).

```{autofunction} s1wvcalibration.config.load_config
```

```{autofunction} s1wvcalibration.config.get_config
```

```{autofunction} s1wvcalibration.config.data_path
```

```{autofunction} s1wvcalibration.config.reset_config_cache
```

## Internal (legacy) modules

The `s1wvcalibration._legacy` package holds the modules ported from the legacy
Ifremer `mpcsentinellibs` scripts. They are implementation details of
`WVcalibrator`; use the public API above instead.

| Module | Purpose |
| ------ | ------- |
| `xsar_reader` | Sentinel-1 Level-1 SLC reader (replaces `cerbere`/`sarimage`) |
| `gmf` | GMF NRCS models via xsarsea (cmod5n / cmod5 / cmod-ifremer) |
| `cmod5n_ipf` | legacy CMOD5n (batch-corrected, "IPF 2.9" behaviour) |
| `ecmwf0100` | read ECMWF 0.1° hourly wind fields |
| `ecmwf0125` | read Météo-France CEP 0.125° 3-hourly wind fields |
| `find_colocated_ecmwf_file` | pick the ECMWF file colocated with a SAFE |
| `compute_wind_azimuth` | wind azimuth relative to the antenna boresight |
| `format_angle` | angle normalisation helpers |
| `classification` | ocean/ice/land classification of each imagette |
| `classif_xml` | read the SAFE classification XML |
| `match_classif` | locate the classification XML for a measurement file |
| `polarization_ratio` | co/pol NRCS ratio |
| `wind_bias` | wind-speed / wind-direction filter ranges |
| `hs_eff_2d` | effective significant wave height (2D ellipsoid cutoff) |
| `hs_2d_cutoff` | 2D-ellipsoid wave-number/direction cutoff |
| `low_frequency_energy` | low-frequency spectral energy helpers |
| `interpolation_polar` | polar interpolation of a wave spectrum |
| `ww3_spectra` | locate the WW3 spectra file colocated with an L2 measurement |
