# Algorithm Theoretical Basis Document (ATBD)

Geophysical Calibration of Sentinel-1 Wave Mode (WV) Observations — Ifremer, MPC
Sentinel-1.

This page is the living ATBD for **s1wvcalibration**. It follows the structure
of the reference document _S1_geophysical_calibration_WV_2026_ and is kept in
sync with the code. Where the two differ, the code is authoritative and the
discrepancy is flagged.

## 1. Scope and objectives

Sentinel-1 Wave Mode has no corner reflectors or transponders, so the
calibration is **relative** (NWP-Ocean Calibration): each 20×20 km² WV imagette
is compared against an empirical Geophysical Model Function (GMF) fed with
colocated ECMWF winds. The calibration constant is a **bias in dB** between the
measured Normalised Radar Cross-Section (NRCS, σ⁰) and the GMF-predicted NRCS,
for given wind and geometry.

- Done per SAR unit, on **VV** polarisation (the IW/WV baseline since 2014).
- Two sub-swaths are calibrated separately: **WV1** (incidence < 30°) and
  **WV2** (incidence > 30°).

## 2. Inputs

| Input              | Source / format                          | Description                                        |
| ------------------ | ---------------------------------------- | -------------------------------------------------- |
| Sentinel-1 WV SAFE | L1 SLC (`.tiff`) or L2 OCN (`.nc`)       | One or more 20×20 km² WV imagettes                 |
| ECMWF wind         | NetCDF, 0.100° hourly or 0.125° 3-hourly | 10 m wind (U/V components)                         |
| Land mask          | `mwf-ers-mask.nc`                        | Quality flag per grid point (0 = ocean, ≠0 = land) |
| Output dir         | user-defined                             | PNG figures + per-SAFE NetCDF                      |

**ECMWF versions** (config `calibration`, CLI `--ecmwf`):

- `0100` — 0.100°, hourly (default).
- `0125` — 0.125°, 3-hourly (IFS); used before 2019-08-21.

### 2.1 GMFs

Three GMFs are implemented (via [xsarsea]); **CMOD5n is the one used for
reporting and interpretation** in 2026 (see §5.4):

| GMF       | Function                            | Note                                   |
| --------- | ----------------------------------- | -------------------------------------- |
| CMOD5     | `cmod5`                             | standard C-band scatterometer GMF (VV) |
| CMOD5n    | `cmod5n_forward` / `cmod5n_xsarsea` | tuned CMOD5, default operational       |
| CMOD-IFR2 | `cmod_ifr2`                         | IFREMER alternative                    |

## 3. Processing steps overview

```
Sentinel-1 WV SAFE
  │  for each imagette
  ├─► read SAR (L1 or L2); extract lon/lat/inc/elev, sigma0, noise, heading
  ├─► co-locate nearest ECMWF grid point (wind speed, direction)
  ├─► compute GMF NRCS (CMOD5 / CMOD5n / CMOD-IFR2); pol-ratio if HH
  ├─► write per-SAFE NetCDF + diagnostic PNG
  │
  │  across a set of SAFE NetCDF files (offline)
  ├─► apply filters (land, |lat|, finite, sigma0>0, wind sweet-spot)
  └─► compute the mean NRCS bias (GMF − SAR) per sub-swath / GMF / wind range
```

## 4. Detailed algorithm

### 4.1 SAR data extraction per imagette

WV mode is designed to be homogeneous, so each imagette is summarised by single
values.

**L1 (SLC)** — read with [xsar] (`_legacy/xsar_reader.read_slc_imagette`):

- `sigma0` (raw, linear) and `sigma0_corrected` (denoised, linear);
- `noise_linear_equivalent` (mean noise, linear);
- `incidence`, `elevation`, `lon`, `lat` at the full-resolution spline centre
  (`line//2`, `sample//2`);
- `doppler_centroid` (image mean of the DC estimates, spline-interpolated) and
  `doppler_centroid_quality` (`dataDcRmsErrorAboveThreshold`);
- `track_angle` (platform heading);
- normalised variance (Nv), skewness, kurtosis of the intensity image.

The denoised sigma0 removes the NEZ: `σ⁰_corrected = σ⁰_raw − nesz`. Pixels with
σ⁰ ≤ 0 are **removed before averaging** (the `include_zero_s0` option is `True`
in the L1 production path, `False` in the test/comparison path — see design
notes).

**L2 (OCN)** — read directly with xarray:

- `oswNrcs` → linear: `σ⁰ = 10^(oswNrcs/10)`;
- `oswNrcsNeszCorr` (denoised, available since IPF 003.51 / Mar 2022);
- `oswLon`, `oswLat`, `oswIncidenceAngle`, `oswHeading`;
- `oswWindSpeed`, `oswWindDirection` (product wind);
- `oswNv` (normalised variance).

OWI/RVL components expose the same variables; NRCS differs slightly between
components but is consistent enough for a joint interpretation.

### 4.2 ECMWF colocation

For each imagette centre `(lon_sar, lat_sar)` (all datetimes UTC):

1. Pick the nearest ECMWF time step (by wall clock, tz-independent).
2. Pick the nearest grid point by Manhattan distance:

   ```
   diff = |lat_sar − lat_wind| + |lon_sar − lon_wind|
   ```

3. Retrieve the colocated wind:

   ```
   spd_ecmwf = sqrt(zon² + mer²)
   dir_ecmwf = mod(degrees(atan2(mer, zon)) − 90, 360)   # meteorological
   ```

4. Antenna-referenced wind azimuth (Mouche, Jan 2016):

   ```
   azi = |format_angle(track_angle + 90 − wdi, compass=False)|
   wdi = mod(degrees(atan2(u, v)) + 180, 360)
   ```

   `azi` is the wind-direction input to the GMF.

### 4.3 Filters

An imagette is **rejected** if any of these hold:

| Filter              | Condition                               | Reason              |
| ------------------- | --------------------------------------- | ------------------- |
| Land                | `is_land(lon, lat) == True`             | only ocean scenes   |
| High latitude       | `abs(lat) > max_latitude` (default 50°) | avoid sea ice / MIZ |
| Non-finite position | `not (isfinite(lon) and isfinite(lat))` | invalid geo         |
| Non-positive sigma0 | `σ⁰ ≤ 0`                                | invalid NRCS        |

It is **kept** for the bias when additionally:

| Selection             | Condition                 | Use                   |
| --------------------- | ------------------------- | --------------------- |
| WV1                   | `incidence < 30°`         | first sub-swath       |
| WV2                   | `incidence > 30°`         | second sub-swath      |
| Classical sweet spot  | `abs(spd − 7) < 3` (m/s)  | main calibration      |
| Noise-free sweet spot | `abs(spd − 10) < 3` (m/s) | secondary calibration |

> The historical slide mentions ±55°; the code uses `high_lat_threshold = 50`
> (changed 01/02/2017). Override with `--maxlat`.

### 4.4 GMFs

- `σ⁰_cmod5 = cmod5(spd, azi, inc)`
- `σ⁰_cmod5n = cmod5n_forward(spd, azi, inc)` (default)
- `σ⁰_cmod_ifr2 = cmod_ifr2(inc, spd, azi)`

**Polarisation ratio (H-pol only):** if the acquisition is HH, each VV-based GMF
is divided by the ratio
`pr = get_polarization_ratio(spd, inc, phi=azi, name="mouche1")`. In practice
the WV offset-gain tuning has only been run on VV.

### 4.5 NRCS bias (calibration constant)

For a wind range (e.g. the classical sweet spot) and sub-swath, the bias Δ (dB)
is the mean over the kept imagettes of

```
Δ = 10·log10(σ⁰_GMF) − 10·log10(σ⁰_SAR)
```

> **Sign / input convention (code).** In this repository the per-SAFE NetCDF
> `correction_*` is computed on the **raw** sigma0 (`_sig_sar`), i.e.
> `GMF − SAR_raw` (GMF-obs sense). The published decision figures (`s1reports`)
> instead plot `SAR_denoised − GMF` (opposite sign, denoised). These two are
> intentionally different — do not harmonise without a decision. A positive Δ
> here means the GMF overestimates NRCS relative to SAR.

Separate biases are computed for WV1 / WV2, for each GMF, and for each wind
range (`classical` / `noise_free`).

### 4.6 Interpretation figures

Since the start of the mission, MPC Sentinel-1 estimates a single WV gain from
the per-SAFE biases using (a) rolling time-averaged time-series and (b) scatter
plots of GMF vs SAR NRCS per sub-swath, with a `numpy.polyfit` linear fit. A
final **distance-to-coast ≥ 300 km** filter is applied to avoid
land/island/atoll contamination and because NWP performs better in the open
ocean. The regression not lying exactly on the 1:1 line is nominal: the strategy
targets a single gain over a specific wind-speed (σ⁰) range, ECMWF has
phase-shift issues, and CMOD5n was derived from coarser scatterometer
observations.

## 5. Mathematical formulation summary

| Quantity         | Formula                                     | Unit   |
| ---------------- | ------------------------------------------- | ------ |
| NRCS (linear)    | σ⁰                                          | –      |
| NRCS (dB)        | `10·log10(σ⁰)`                              | dB     |
| GMF prediction   | `σ⁰_GMF = GMF(ws, φ, θ, pol)`               | linear |
| Calibration bias | `Δ = ⟨10·log10(σ⁰_GMF) − 10·log10(σ⁰_SAR)⟩` | dB     |
| Relative azimuth | `φ = azimuth_angle(track, u, v)`            | deg    |

## 6. Output NetCDF variables

**Per-imagette** (dimension `number_of_wm_acq`):

`time_sar`, `sig_sar`, `sig_sar_denoised`, `noise_sar`, `land`, `track_angle`,
`incidence_angle`, `elevation_angle`, `doppler_centroid`,
`doppler_centroid_quality`, `lon_sar`, `lat_sar`, `nv_sar`, `skew_sar`,
`kurt_sar`, `rough_class_sar`, `spd_ecmwf`, `dir_ecmwf`, `zon_ecmwf`,
`mer_ecmwf`, `lon_ecmwf`, `lat_ecmwf`, `dlon_ecmwf`, `dlat_ecmwf`, `sig_mod`
(CMOD5n GMF NRCS), `sig_mod_xsarsea`.

**Per-SAFE** (dimension `number_of_wm_correction` = 1):

- Biases: `correction_wv1`, `correction_wv2` (CMOD-IFR2),
  `correction_wv1_cmod5`/`_wv2_cmod5`, `correction_wv1_cmod5n`/`_wv2_cmod5n`,
  `…_cmod5n_xsarsea_…`; each with a `_10ws` suffix for the noise-free range.
- `nbptsNRCSbias_wv1`/`_wv2` (and `_10ws`) — number of points used.
- `sar_imagette_polarization` (`'V'` or `'H'`).
- Significant-wave-height (2D-ellipsoid-cutoff) bias vs WW3 spectra
  (`mean_swh_2d_eff_bias_ww3spec_*`, L2 path only).

## 7. Assumptions and limitations

1. **Homogeneous scene** — each 20×20 km² imagette is assumed uniform; real
   ocean/atmospheric features introduce inhomogeneities.
2. **ECMWF as truth** — the GMF uses ECMWF winds (calibrated in situ,
   scatterometer-assimilating); the model can be shifted in time and space.
3. **C-band GMF reliability** is maximised in the 4–10 m/s sweet spot.
4. **No absolute calibration** — this is a relative geophysical calibration,
   complementary to transponder-based absolute calibration (WV↔IW alignment: Hi
   et al. 2019).
5. **Land mask** ~0.125° resolution: coastal points within ~10 km are flagged
   land.
6. **High-latitude filter** default 50°, configurable.

## 8. Default constants (from `config.yaml`)

| Constant                 | Value                               | Meaning                             |
| ------------------------ | ----------------------------------- | ----------------------------------- |
| `high_lat_threshold`     | 50                                  | degrees latitude                    |
| `dev_max_measu`          | 3                                   | imagettes processed in `--dev` mode |
| `wind_ranges.classical`  | 7 m/s                               | centre wind, main calibration       |
| `wind_ranges.noise_free` | 10 m/s                              | centre wind, secondary calibration  |
| `time_unit`              | `seconds since 2014-01-01 00:00:00` | NetCDF time reference               |

## 9. References

- NWP Ocean Calibration methodology: <https://doi.org/10.1029/2020EA001606>
- CMOD5 / CMOD5n: Hersbach et al. (2007), _J. Atmos. Oceanic Technol._,
  doi:10.1029/2006JC003743
- CMOD-IFR2: Quilfen et al. (1998), IFREMER report
- Sentinel-1 WV product specification: ESA Sentinel-1 Product Definition
- Li et al. 2019, <https://doi.org/10.1109/TGRS.2018.2867035>
