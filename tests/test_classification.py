"""Unit tests for the Deep-Learning WV roughness classification chain."""

from __future__ import annotations

import collections
from pathlib import Path

import pytest

from s1wvcalibration._legacy.classif_xml import read_xml_classif_ID
from s1wvcalibration._legacy.classification import (
    get_classification_infos,
    get_classification_infos_safe_granularity,
)
from s1wvcalibration._legacy.match_classif import (
    _classif_glob_pattern,
    get_xml_path_from_measu,
    get_xml_path_from_safe,
)
from s1wvcalibration.config import CONFIG_ENV_VAR, reset_config_cache

# A realistic S1D WV OCN SAFE name (underscore split: [5]=date, [9]=unique id).
SAFE = "S1D_WV_OCN__2SSV_20200101T000000_20200101T003443_012345_000001_1234"
# L2 OCN measurement: dash split, field [8] is "0001.nc" (imagette index).
MEASU_NAME = "s1d-wv1-ocn-vv-20200101t000000-20200101t003443-012345-00378e-0001.nc"

_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    "<DL_Summary><imagettes>"
    "<id>1<lonlat_center>10.0 20.0</lonlat_center>"
    "<all_prob>0.5 0.2 0.1 0.07 0.05 0.03 0.02 0.01 0.007 0.003</all_prob>"
    "</id></imagettes></DL_Summary>"
)


def _xml_path_for(safe: str) -> Path:
    """Concrete DL_Summary.xml path for a SAFE (the * is replaced by AAAA)."""
    base, star = _classif_glob_pattern(safe)
    return Path(base) / star.replace("*", "AAAA")


@pytest.fixture
def classif_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "classif"
    root.mkdir()
    cfg = tmp_path / "override.yaml"
    cfg.write_text(f"paths:\n  classif_wv_cls: {root}\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg))
    reset_config_cache()
    yield root
    reset_config_cache()


def _measu_path(safe_root: Path) -> Path:
    """Build <safe_root>/SAFE/measurement/<meas>."""
    safe_dir = safe_root / SAFE
    (safe_dir / "measurement").mkdir(parents=True)
    return safe_dir / "measurement" / MEASU_NAME


def test_read_xml_classif_id(classif_root):  # noqa: ARG001  # side-effect fixture
    xml = _xml_path_for(SAFE)
    xml.parent.mkdir(parents=True)
    xml.write_text(_XML)
    df = read_xml_classif_ID(str(xml))
    # one imagette id per column (DataFrame is transposed vs the doc string)
    assert "1" in df.columns
    row = df["1"]
    for i in range(1, 11):
        assert f"prob_{i}" in row.index
        assert f"class_{i}" in row.index
    assert float(row["lon"]) == 10.0
    assert float(row["lat"]) == 20.0
    # probabilities are sorted descending
    assert float(row["prob_1"]) >= float(row["prob_10"])


def test_get_xml_path_from_measu_found(classif_root, tmp_path):  # noqa: ARG001  # side-effect fixture
    measu = _measu_path(tmp_path)
    measu.write_text("")
    xml = _xml_path_for(SAFE)
    xml.parent.mkdir(parents=True)
    xml.write_text(_XML)
    found = get_xml_path_from_measu(str(measu))
    assert found is not None
    assert Path(found).name == "DL_Summary.xml"


def test_get_xml_path_from_measu_missing(classif_root, tmp_path):  # noqa: ARG001  # side-effect fixture
    measu = _measu_path(tmp_path)
    measu.write_text("")
    assert get_xml_path_from_measu(str(measu)) is None


def test_get_xml_path_from_safe(classif_root, tmp_path):  # noqa: ARG001  # side-effect fixture
    xml = _xml_path_for(SAFE)
    xml.parent.mkdir(parents=True)
    xml.write_text(_XML)
    found = get_xml_path_from_safe(SAFE)
    assert found is not None
    assert Path(found).name == "DL_Summary.xml"
    # a different (non-existing) SAFE returns None
    other = "S1B_WV_SLC__1SSV_20200102T000000_20200102T003443_012345_000001_1234"
    assert get_xml_path_from_safe(other) is None


def test_classification_found(classif_root, tmp_path):  # noqa: ARG001  # side-effect fixture
    measu = _measu_path(tmp_path)
    measu.write_text("")
    xml = _xml_path_for(SAFE)
    xml.parent.mkdir(parents=True)
    xml.write_text(_XML)
    content: dict[str, object] = {}
    errors = collections.defaultdict(int)
    counters = collections.defaultdict(int)
    content, errors, counters = get_classification_infos(
        str(measu), content, errors, counters
    )
    assert counters["classif_OK"] == 1
    assert content["class_1"] == "AF"  # highest prob -> first ordered class


def test_classification_missing_imagette(classif_root, tmp_path):  # noqa: ARG001  # side-effect fixture
    # XML with a non-matching id (id "999" != imagette "1")
    measu = _measu_path(tmp_path)
    measu.write_text("")
    xml = _xml_path_for(SAFE)
    xml.parent.mkdir(parents=True)
    xml.write_text(_XML.replace("<id>1", "<id>999"))
    content: dict[str, object] = {}
    errors = collections.defaultdict(int)
    counters = collections.defaultdict(int)
    get_classification_infos(str(measu), content, errors, counters)
    assert errors["classif_missing_imagette"] == 1
    assert content["class_1"] == " "


def test_classification_empty_xml(classif_root, tmp_path):  # noqa: ARG001  # side-effect fixture
    measu = _measu_path(tmp_path)
    measu.write_text("")
    xml = _xml_path_for(SAFE)
    xml.parent.mkdir(parents=True)
    xml.write_text("")  # zero bytes
    content: dict[str, object] = {}
    errors = collections.defaultdict(int)
    counters = collections.defaultdict(int)
    get_classification_infos(str(measu), content, errors, counters)
    assert errors["classification_file_empty"] == 1


def test_classification_no_xml(classif_root, tmp_path):  # noqa: ARG001  # side-effect fixture
    measu = _measu_path(tmp_path)
    measu.write_text("")
    content: dict[str, object] = {}
    errors = collections.defaultdict(int)
    counters = collections.defaultdict(int)
    get_classification_infos(str(measu), content, errors, counters)
    assert errors["no_classification"] == 1
    assert content["class_1"] == " "
    assert content["prob_1"] == -1


def test_classification_bad_extension(classif_root, tmp_path):  # noqa: ARG001  # side-effect fixture
    # XML must exist so the code reaches the extension check.
    measu = (
        tmp_path
        / SAFE
        / "measurement"
        / ("s1d-wv1-ocn-vv-20200101t000000-20200101t003443-012345-00378e-0001.txt")
    )
    measu.parent.mkdir(parents=True)
    measu.write_text("")
    xml = _xml_path_for(SAFE)
    xml.parent.mkdir(parents=True)
    xml.write_text(_XML)
    with pytest.raises(ValueError, match="unknown measurement extension"):
        get_classification_infos(
            str(measu), {}, collections.defaultdict(int), collections.defaultdict(int)
        )


def test_classification_safe_granularity_found():
    data_from_xml: dict[str, object] = {"1": {}}
    row: dict[str, object] = {}
    for i in range(2, 11):
        row[f"prob_{i}"] = 0.0
        row[f"class_{i}"] = "WS"
    row["prob_1"] = 1.0
    row["class_1"] = "AF"
    for tag in ("AF", "BS", "IB", "LWA", "MCC", "OF", "POS", "RC", "SI", "WS"):
        row[tag] = 0.0
    data_from_xml["1"] = row
    content: dict[str, object] = {}
    errors = collections.defaultdict(int)
    counters = collections.defaultdict(int)
    get_classification_infos_safe_granularity(
        f"SAFE/measurement/{MEASU_NAME}", data_from_xml, content, errors, counters
    )
    assert counters["classif_OK"] == 1
    assert content["class_1"] == "AF"


def test_classification_safe_granularity_missing():
    # imagette "1" not present in the dict
    data_from_xml: dict[str, object] = {}
    content: dict[str, object] = {}
    errors = collections.defaultdict(int)
    counters = collections.defaultdict(int)
    get_classification_infos_safe_granularity(
        f"SAFE/measurement/{MEASU_NAME}", data_from_xml, content, errors, counters
    )
    assert errors["classif_missing_imagette"] == 1


def test_classif_glob_pattern_slc():
    safe = "S1A_WV_SLC__1SSV_20200101T000000_20200101T003443_012345_000001_1234"
    base, star = _classif_glob_pattern(safe)
    assert str(base).endswith("2020/01/01")
    assert "DL" in star
    assert "*" in star
    # SLC -> 2SS (DL products use 2 look angles)
    assert "2SS" in star
