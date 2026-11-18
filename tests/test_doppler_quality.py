"""Unit tests for the Doppler-centroid quality flag read from the annotation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from s1wvcalibration._legacy.xsar_reader import _doppler_uncertain_flag

if TYPE_CHECKING:
    from pathlib import Path

MEASU = "s1c-wv2-slc-vv-20260907t031116-20260907t031119-009335-012911-006.tiff"


def _annotation(uncertain: str, rms_above: str = "false") -> str:
    """Minimal annotation with both flags, which may disagree in real products."""
    return (
        "<product><dopplerCentroid><dcEstimateList><dcEstimate>"
        f"<dataDcRmsErrorAboveThreshold>{rms_above}</dataDcRmsErrorAboveThreshold>"
        "</dcEstimate></dcEstimateList></dopplerCentroid>"
        "<qualityInformation><qualityDataList><qualityData>"
        "<dopplerCentroidQuality>"
        f"<dopplerCentroidUncertainFlag>{uncertain}</dopplerCentroidUncertainFlag>"
        "</dopplerCentroidQuality></qualityData></qualityDataList>"
        "</qualityInformation></product>"
    )


def _make_safe(tmp_path: Path, xml: str | None) -> Path:
    safe = tmp_path / "S1C_WV_SLC__1SSV.SAFE"
    (safe / "measurement").mkdir(parents=True)
    (safe / "annotation").mkdir()
    measu = safe / "measurement" / MEASU
    measu.write_text("")
    if xml is not None:
        (safe / "annotation" / MEASU.replace(".tiff", ".xml")).write_text(xml)
    return measu


@pytest.mark.parametrize(
    ("uncertain", "rms_above", "expected"),
    [
        ("true", "false", 1.0),  # real case S1C 2026-09-07 imagette 006
        ("false", "true", 0.0),
        ("false", "false", 0.0),
    ],
)
def test_flag_follows_doppler_centroid_uncertain_flag(
    tmp_path: Path, uncertain: str, rms_above: str, expected: float
) -> None:
    measu = _make_safe(tmp_path, _annotation(uncertain, rms_above))
    assert _doppler_uncertain_flag(str(measu)) == expected


def test_missing_annotation_gives_nan(tmp_path: Path) -> None:
    measu = _make_safe(tmp_path, None)
    assert math.isnan(_doppler_uncertain_flag(str(measu)))


def test_missing_flag_gives_nan(tmp_path: Path) -> None:
    measu = _make_safe(tmp_path, "<product><qualityInformation/></product>")
    assert math.isnan(_doppler_uncertain_flag(str(measu)))
