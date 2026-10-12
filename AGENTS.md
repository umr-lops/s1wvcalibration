# AGENTS.md

`s1wvcalibration` est une lib Python qui calcule la **calibration geophysique
du mode Wave (WV) de Sentinel-1**. Elle est un re-packaging propre (python
3.10+, ruff/mypy strict, packaging `flit-scm`/uv/nox) de l'ancienne production
Ifremer MPC-Sentinel1 (`computeWVconstantecalibration.py`, héritée de
`calvalsentinel1`).

> **Ceci est un fichier de travail "en cours"** : il trace ce qui est fait et
> ce qui reste, pour que n'importe quel agent puisse reprendre la tâche.
> A nettoyer/retirer avant publication (ou à garder comme doc de dev).

## Mission / contexte

- **Objectif** : remplacer les 3 dépendances legacy par des deps modernes, en
  gardant la production **numériquement identique** :
  1. `cerbere`/`sarimage` (lecture SLC L1) → **xsar**
  2. `sar.utils.gmf` / `cmod5n.py` (GMF NRCS) → **xsarsea**
  3. `shared_information.py` (chemins/constants) → **`config.yaml` + `config.py`**
- **Source legacy** : `/scratch/agrouaze/opencode_sandbox/calvalsentinel1/mpc-sentinel/mpcsentinellibs/`
- **Source de référence pour les API** : `/scratch/agrouaze/opencode_sandbox/xsar/` et `/scratch/agrouaze/opencode_sandbox/xsarsea/`
- **ATBD à valider** : `S1_geophysical_calibration_WV_2026.docx` (voir note
  "Incohérence ATBD" plus bas).

## Environnement de dev

- **Pas de `uv`** sur cette machine. Utiliser l'env micromamba existant
  `s1wvcalib` (créé via conda-forge, GDAL prébuilt) :
  ```bash
  MICRO=~/.local/bin/micromamba
  $MICRO run -n s1wvcalib python ...          # exécuter
  $MICRO run -n s1wvcalib s1wvcalib --help    # CLI
  $MICRO run -n s1wvcalib pytest              # tests
  $MICRO run -n s1wvcalib ruff check src/     # lint
  ```
- Installer le paquet en editable (sans deps lourdes) :
  ```bash
  $MICRO run -n s1wvcalib pip install -e /scratch/agrouaze/opencode_sandbox/s1wvcalibration --no-deps
  ```
- **xsar / xsarsea / GDAL / dask / xarray doivent rester en conda-forge**
  (prébuilt). **NE PAS** `pip install xsar/xsarsea` (ça casse la chaîne
  numpy/dask/xarray). `xarray-safe-s1` (= `safe_s1`) est installé en
  `--no-deps`. Si l'env casse, réaligner :
  ```bash
  $MICRO install -y -n s1wvcalib -c conda-forge "xarray<2026" "dask>=2025" "numpy" "xsar" "xsarsea"
  ```
- ⚠️ L'utilisateur est **limité sur l'espace `/home/agrouaze/`** : ne pas créer
  d'envs de plus en plus lourds, réutiliser `s1wvcalib`.

## Ce qui est FAIT

### 1. Squelette du repo (calqué sur `turboblast`)
- `pyproject.toml` (flit-scm, dynamic version, deps, config ruff/mypy/pytest/coverage),
  `[project.scripts] s1wvcalib = "s1wvcalibration.calibration:entrypoint"`.
- `noxfile.py`, `.pre-commit-config.yaml`, `.readthedocs.yaml`,
  `.github/workflows/{ci,cd}.yml`, `docs/` (sphinx + furo), `LICENSE`,
  `.gitignore`, `.git_archival.txt`, `README.md`.
- Repo ciblé sur **GitHub** `umr-lops/s1wvcalibration` (pas GitLab).
- ✅ Vérifié : `pip install -e`, `s1wvcalib --help`, `import`, `pytest` (test
  de version), `python -m build --wheel`.

### 2. Config (remplace `shared_information`)
- `src/s1wvcalibration/config.yaml` : `paths` (project_dir, analysis_dir,
  ecmwf_0100_light/full, ecmwf_0125, land_mask, classif_wv_cls, ww3_coloc),
  `calibration` (high_lat_threshold=50, wind_ranges classical=7/noise_free=10,
  dev_max_measu=3, time_unit), `gmf.nc_luts_path`, `satellites`.
- `src/s1wvcalibration/config.py` : `load_config()` (lru_cache) merge
  `config.yaml` + override `localconfig.yaml` ou `$S1WVCALIB_CONFIG`.
  `reset_config_cache()` pour les tests.
- ✅ Tous les chemins durcis legacy remplacés par la config.

### 3. Migration du cœur (`calibration.py` + `_legacy/`)
- `calibration.py` : `WVcalibrator`, `is_land`, `PlotNRCSvsWindPerSafe`,
  `SaveNRCSvsWindPlot`, `sequentialListOfSAfeAnalyse`, CLI (`main`/
  `entrypoint`/`_build_parser`) identique à la legacy.
- `_legacy/` (20 modules portés, imports relatifs, sans `pdb`/`print`) :
  `ecmwf0100`, `ecmwf0125`, `find_colocated_ecmwf_file`, `cmod_ifremer`,
  `compute_wind_azimuth` (+`format_angle`,`angle_vectors`,`small_methods`),
  `classification` (+`match_classif`,`classif_xml`), `polarization_ratio`,
  `wind_bias`, `hs_eff_2d` (+`hs_2d_cutoff`,`low_frequency_energy`,
  `interpolation_polar`,`ac_calc`), `ww3_spectra`, **`gmf`**, **`xsar_reader`**.

### 4. Remplacement GMF → xsarsea (`_legacy/gmf.py`)
- `get_cmod5n()` = `xsarsea.windspeed.get_model("cmod5n")`, `cmod5`,
  `cmod_ifr2` (= `xsarsea.windspeed.gmfs_impl.gmf_cmodifr2`).
- Sémantique d'args legacy conservée `(v, phi, theta)` ; le `__call__` xsarsea
  est `(inc, wspd, phi, broadcast=True)` → on appelle `model(theta, v, phi)`.
- Retourne `numpy.ndarray` linéaire (`_to_linear_ndarray`).

### 5. Remplacement SLC → xsar (`_legacy/xsar_reader.py`)
- `read_slc_imagette(measu_path) -> dict` via `Sentinel1Meta` +
  `Sentinel1Dataset` + `apply_calibration_and_denoising()`.
- **Mapping de noms confirmé** dans le code xsar (`_add_denoised`) :
  - legacy `sigma0` (raw)   → xsar `sigma0_raw`
  - legacy `sigma0_corrected` (denoised) → xsar `sigma0` (= `sigma0_raw - nesz`)
  - legacy `noise_linear_equivalent` → xsar `nesz`
  - geoloc : `longitude`/`latitude`/`incidenceAngle`/`elevationAngle`
    (centre du geoloc grid).
- ⚠️ `safe_s1` doit être installé pour exécuter `Sentinel1Meta` (pas dispo dans
  le sandbox initial ; installé en `--no-deps`).

### 6. Parités VÉRIFIÉES (max diff = 0.0)
- `cmod_ifr2` (xsarsea) vs legacy `cmod_ifremer.cmod_ifr2`.
- `compute_wind_azimuth` (+ 3 modules angle) vs legacy.

## Ce qui RESTE À FAIRE (ordre)

### A. Tests de parité (étape 3 de la mission) — **EN COURS**
1. **test_gmf_parity.py** ✅ FAIT (`tests/test_gmf_parity.py`, 6 tests, passe).
   Charge les sources legacy depuis le checkout `calvalsentinel1`
   (override env `S1WVCALIB_LEGACY_DIR`), skip si absent. Résultats mesurés
   sur la grille WV (inc 15–45°, v 1–25 m/s, phi 0–180°) :
   - port `cmod5n_ipf.cmod5n_legacy` vs source legacy `cmod5n.py` : **exact** (0.0).
   - xsarsea vs legacy au sweet spot 7 m/s : WV1 identique (<1e-6 dB),
     WV2 écart max **~1.3 dB** (le clamp batch `S->S0` du legacy bite) ;
     borne testée < 1.5 dB. Sur grille complète 1–25 m/s : écart max ~6.7 dB
     (borne testée < 7 dB) — comportement attendu, documenté.
   - `cmod_ifr2` xsarsea (f64) vs legacy (coeffs f32) : ~2e-6 dB (float32).
   - `compute_azimuth_angle` vs legacy : **exact** (0.0).
   - `cmod5` : ⚠️ la source legacy `from sar.utils.gmf import cmod5` **n'est
     pas dans le sandbox** (package `sar` non local) → parité non testable
     localement ; à faire sur l'ancienne instance HPC.
2. **test_l1_parity.py** ✅ FAIT (`tests/test_l1_parity.py`, 2 tests, passe).
   **SAFE réel** (dispo dans le sandbox, montée `/legacy`) :
   `data/esa/sentinel-1d/L1/WV/S1D_WV_SLC__1S/2026/093/S1D_WV_SLC__1SSV_..._9153.SAFE`
   (override env `S1WVCALIB_L1_SAFE` / `S1WVCALIB_L1_REF_NC`), 143 tiff.
   Compare `read_slc_imagette` (xsar) vs le NetCDF de prod (f32) imagette par
   imagette — **tout passe à précision float32** (sig/sigd/noise/dc/trk/nv/
   skew/kurt, lat/lon/inc/elev). Conventions validées :
   - xsar **SLC (incl. WV)** ne fait PAS auto la calibration (GRD seulement) →
     appeler `add_high_resolution_variables()` **puis** `apply_calibration_and_denoising()`.
     (`lazy_loading=False` est buggy pour WV → toujours `True`.)
   - `sigma0_raw`=legacy `sigma0`, `sigma0`=legacy `sigma0_corrected`, `nesz`=legacy
     `noise_linear_equivalent` (isel `pol=0`).
   - lon/lat/inc/elev = **full-res (spline) centre** `line//2, sample//2` →
     lat/inc/elev **exactes**, lon à 360 près.
   - `track_angle` = `meta.platform_heading`.
   - `doppler_centroid` = moyenne image du **`frequency`** (estimates DC)
     interpolé `RectBivariateSpline` sur (azimuth, slant-range) — même méthode
     que `cerbere.SAFEGeoTiffFile.read_values('doppler_centroid')` (pas le
     `dataDcPolynomial` qui diverge). `quality` = `dataDcRmsErrorAboveThreshold`.
   - ordre des mesures = `sorted(glob(measurement/*.tiff))` = ordre legacy
     (tous wv1 avant wv2 en lex). Réf = `calib/....SAFE.nc`.
3. **test_netcdf_parity.py** (END2END) — **EN COURS** : lancer
   `WVcalibrator.AnalyseOneSAFE` (CLI `s1wvcalib`) sur le SAFE entier (143) et
   comparer variable par variable vs le NetCDF de prod
   `calib/S1D_WV_SLC__1SSV_..._9153.SAFE.nc`.
   - Données montées dans `/legacy` (land mask, ecmwf_0100 light, classif DL,
     SAFE). Config pointant vers `/legacy` : `/tmp/opencode/localconfig_legacy.yaml`
     via `S1WVCALIB_CONFIG` (voir aussi `localconfig.yaml` projet si créé).
   - Smoke-test `--dev` (3 imagettes) ✅ OK : pipeline complet (classif, ecmwf
     colocalisation, land mask, GMF, écriture nc + 3 png) fonctionne.
   - **Repo GitHub créé** : `umr-lops/s1wvcalibration` (vide).
   - ⚠️ la prod est **L1 par défaut** (`include_zero_s0=True` dans le DAG) ;
     comparer avec les mêmes params que la prod (ecmwf `0100`, no maxlat).
   comparer `read_slc_imagette()` (xsar) vs l'ancien extracteur cerbere/
   `SARImage` : `sigma0`, `sigma0_corrected`, `noise`, `doppler_centroid`,
   `lon/lat/inc/elev`, `I=|DN|²` (nv/skew/kurt). **Exige un SAFE** → à faire
   sur l'ancienne instance (`ssh datarmor-access`).
3. **test_netcdf_parity.py** : lancer `WVcalibrator.AnalyseOneSAFE` sur un
   SAFE réel (L1 et L2) et comparer variable par variable le NetCDF produit
   vs celui de la production actuelle (tolérance stricte, ex. `atol=1e-5` en
   dB / relatif en linéaire). C'est le test de bout en bout qui "valide".
   - **SAFE de référence L1** (HPC, pour l'end2end) :
     `/legacy/project/cersat/public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/L1_v16/WV/S1D_WV_SLC__1S/2026/093/S1D_WV_SLC__1SSV_20260403T111410_20260403T114853_002181_00378E_9153.SAFE`
     NetCDF production de référence :
     `.../S1D_WV_SLC__1SSV_20260403T111410_20260403T114853_002181_00378E_9153.SAFE/calib/S1D_WV_SLC__1SSV_20260403T111410_20260403T114853_002181_00378E_9153.SAFE.nc`

### B. Lint / qualité (étape 4)
- `$MICRO run -n s1wvcalib ruff check src/ tests/` et `ruff format` —
  corriger toutes les violations (règles dans `pyproject.toml`).
- `mypy` (config `strict` sur `s1wvcalibration.*`, `ignore_missing_imports`) :
  type-annoter les signatures, notamment `calibration.py`.
- `nox -s lint` / `pre-commit run --all-files` (prek) une fois réseau dispo.

### C. Divers / finition
- **README** : documenter `config.yaml` / `localconfig.yaml`, le fait que la
  production est **L1 par défaut** (le DAG `compute_WVcstcalval_prun.sh` fait
  de la **L1 SLC** — cf. listing `S1A_..._wv_slc_...`).
- Vérifier que la constante `HIGH_LAT_THRESHOLD` (50°) et les `WIND_RANGES`
  bien issues de la config et non hardcodées.
- Décider du **nommage** : le script legacy s'appelle
  `computeWVconstantecalibration.py` (sic, typo) ; on a `calibration.py`.
- **Commit / branch** : créer une branche `feat/initial-migration` et commit
  **atomiques** (skeleton / config / gmf / xsar / parity-tests). **Ne pas
  pousser/commuter sans accord explicite de l'utilisateur.**
- Optionnel : ajouter un `Dockerfile`/`apptainer.def` (modèle
  `s1reports/apptainer_s1reports.def`) pour le HPC.

## Notes de design importantes (à ne pas "corriger")

- **`include_zero_s0`** : la production L1 passe `include_zero_s0=True`
  (moyenne sur tous les pixels, 0 inclus). Le test/comparaison xsar utilise
  `False`. Conserver les deux chemins (comme la legacy).
- **Biais de calibration** : le NetCDF par-SAFE `correction_wv*_cmod5n` est
  `GMF − SAR_raw` (sens GMF obs), calculé sur **`_sig_sar` (raw, avec bruit)**
  et moyenné sur le SAFE. Les **figures publiées** (`s1reports`) utilisent
  **`sigma0_denoised`** avec le **sens inverse** (`SAR_denoised − GMF`). Les
  deux existent et sont volontairement différents — ne pas harmoniser sans
  instruction.
- **WV1 / WV2** : split par `incidence < 30°` (WV1) / `> 30°` (WV2).
- **GMF en prod = CMOD5n** (les autres GMF cmod5 / cmodifremer sont calculés
  et sauvegardés mais CMOD5n est celui "used for reporting and interpretation"
  — cf. ATBD §5.4).

## Incohérence ATBD à signaler (docx `S1_geophysical_calibration_WV_2026`)
La §5.5/§6 décrit `Δ = ⟨σ⁰_GMF − σ⁰_SAR⟩` (sens GMF obs, et pas de mention de
débruitage). Or les figures de décision publiées sont en `SAR_denoised − GMF`
(sens inverse + débruité). Le §3.1 liste `sig_sar` **et** `sig_sar_denoised`
(mais le biais §5.5 n'utilise que `sig_sar`). À clarifier dans le doc.

## Arborescence
```
s1wvcalibration/
├── pyproject.toml            # build (flit-scm), deps, ruff/mypy/pytest/coverage, scripts
├── noxfile.py                # lint / tests / docs / build
├── .pre-commit-config.yaml   # ruff, mypy, prettier, codespell, ...
├── .github/workflows/        # ci.yml (pre-commit + pytest matrix), cd.yml (pypi)
├── docs/                     # sphinx (furo), index.md
├── src/s1wvcalibration/
│   ├── __init__.py           # __version__
│   ├── _version.py           # généré (flit-scm) — ne pas éditer
│   ├── config.py             # load_config() — remplace shared_information
│   ├── config.yaml           # chemins + constantes calibration
│   ├── calibration.py        # cœur : WVcalibrator + CLI `s1wvcalib`
│   └── _legacy/              # 20 modules portés (gmf, xsar_reader, ecmwf, ...)
└── tests/
    ├── test_package.py       # version test
    └── test_gmf_parity.py    # GMF/azimuth parity vs legacy (à compléter : l1, netcdf)
```
