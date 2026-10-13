"""Reader for the Deep Learning S-1 WV classification XML files.

Ported from ``qualitycheck/reader_classif_wv_cls_xml.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lxml import etree

logger = logging.getLogger(__name__)

__all__ = ["read_xml_classif_ID"]

ORDERED_CLASSES_XML = np.array(
    ["AF", "BS", "IB", "LWA", "MCC", "OF", "POS", "RC", "SI", "WS"]
)


def read_xml_classif_ID(full_path_xml: str) -> pd.DataFrame:
    """Return a dict-of-dict (indexed by imagette number) of the classif infos.

    Keys per imagette: ``prob_1..10``, ``class_1..10``, ``lon``, ``lat`` and
    the raw XML child tags.
    """
    data: dict = {}
    logger.debug("full_path_xml %s", full_path_xml)
    tree = etree.parse(full_path_xml)
    for imagette in tree.xpath("/DL_Summary/imagettes/id"):
        im_id = imagette.text
        data[im_id] = {}
        for chi in imagette.getchildren():
            if chi.tag in ("prob_1", "prob_2", "prob_3"):
                data[im_id][chi.tag] = float(chi.text)
            elif chi.tag == "lonlat_center":
                lon_v, lat_v = chi.text.split(" ")
                data[im_id]["lon"] = float(lon_v)
                data[im_id]["lat"] = float(lat_v)
            elif chi.tag == "all_prob":
                all_probs = chi.text.split(" ")
                for jj, ju in enumerate(all_probs):
                    ju_flo = float(ju)
                    data[im_id][f"proba_disordered_{jj + 1}"] = ju_flo
                    data[im_id][ORDERED_CLASSES_XML[jj]] = ju_flo
                all_possible_prob = np.array(
                    [
                        data[im_id][prob]
                        for prob in data[im_id]
                        if "proba_disordered_" in prob
                    ]
                )
                inds_sorted = np.argsort(all_possible_prob)[::-1]
                all_possible_prob_ordered = np.sort(all_possible_prob)[::-1]
                decrease_belong_order = ORDERED_CLASSES_XML[inds_sorted]
                for xx, probx in enumerate(all_possible_prob_ordered):
                    if f"prob_{xx + 1}" not in data[im_id]:
                        data[im_id][f"prob_{xx + 1}"] = probx
                        data[im_id][f"class_{xx + 1}"] = decrease_belong_order[xx]
            else:
                data[im_id][chi.tag] = chi.text
    return pd.DataFrame(data)
