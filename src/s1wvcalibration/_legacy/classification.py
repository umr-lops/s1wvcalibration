"""Retrieve the Deep Learning WV roughness classification for a measurement.

Ported from ``qualitycheck/get_classification_infos.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .classif_xml import read_xml_classif_ID
from .match_classif import get_xml_path_from_measu

logger = logging.getLogger(__name__)

__all__ = ["get_classification_infos"]


def get_classification_infos(i, content, errors, counters):
    """Fill *content* with the classification variables for measurement *i*.

    Args:
        i: full path of the L2 OCN (``.nc``) or L1 SLC (``.tiff``) measurement.
        content: dict updated in place.
        errors: counter of errors (defaultdict int).
        counters: counter of infos (defaultdict int).

    Returns:
        ``(content, errors, counters)``.
    """
    path_pot_xml = get_xml_path_from_measu(i)
    var_to_index = []
    check_sum = 0
    for yy in range(1, 11):
        var_to_index.append("class_" + str(yy))
        var_to_index.append("prob_" + str(yy))
    for var_i in var_to_index:
        if "class" in var_i:
            content[var_i] = " "
        else:
            content[var_i] = -1

    if path_pot_xml is not None and Path(path_pot_xml).exists():
        if Path(path_pot_xml).stat().st_size == 0:
            logger.info("%s is empty", path_pot_xml)
            errors["classification_file_empty"] += 1
        else:
            df_classif = read_xml_classif_ID(path_pot_xml)
            logger.debug("key classif %s", df_classif.keys())
            measu_stem = Path(i).name.split("-")[8].rsplit(".", 1)[0]
            if not (".nc" in i or ".tiff" in i):
                raise ValueError(f"unknown measurement extension in {i}")
            imagette_number = str(int(measu_stem))
            logger.debug("imagette number %s", imagette_number)
            if imagette_number in df_classif:
                for var_i in var_to_index:
                    content[var_i] = df_classif[imagette_number][var_i]
                    if "prob_" in var_i:
                        check_sum += df_classif[imagette_number][var_i]
                counters["classif_OK"] += 1
            else:
                errors["classif_missing_imagette"] += 1
                logger.info("imagette %s is not present in %s", imagette_number, path_pot_xml)
    else:
        logger.debug("no classif file for pat %s", path_pot_xml)
        errors["no_classification"] += 1
    logger.debug("check_sum : %s", check_sum)
    assert check_sum <= 1.1, f"check_sum {check_sum} > 1.1"
    return content, errors, counters


def get_classification_infos_safe_granularity(measu, data_from_xml, content, errors, counters):
    """Fill *content* using a pre-read classif dict (SAFE granularity)."""
    var_to_index = []
    check_sum = 0
    for yy in range(1, 11):
        var_to_index.append("class_" + str(yy))
        var_to_index.append("prob_" + str(yy))
    var_to_index += ["AF", "BS", "IB", "LWA", "MCC", "OF", "POS", "RC", "SI", "WS"]
    for var_i in var_to_index:
        if "class" in var_i:
            content[var_i] = " "
        else:
            content[var_i] = -1
    if not (".nc" in measu or ".tiff" in measu):
        raise ValueError(f"unknown measurement extension in {measu}")
    imagette_number = str(int(Path(measu).name.split("-")[8].rsplit(".", 1)[0]))
    if imagette_number in data_from_xml:
        for var_i in var_to_index:
            content[var_i] = data_from_xml[imagette_number][var_i]
            if "prob_" in var_i:
                check_sum += data_from_xml[imagette_number][var_i]
        counters["classif_OK"] += 1
    else:
        errors["classif_missing_imagette"] += 1
        logger.info("imagette %s is not present in the xml file", imagette_number)
    assert check_sum <= 1.1, f"check_sum {check_sum} > 1.1"
    return content, errors, counters
