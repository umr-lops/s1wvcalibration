"""Small dataframe helpers (subset of qualitycheck/small_methods_to_decode_vectorize_dataframe.py).

Only the symbols actually used by the calibration engine are ported. The
``msgpack_numpy`` dependency is imported lazily so the module stays importable
in environments that do not need the HBase pickle decoding path.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def notisnan(x: float) -> bool:
    """Return ``True`` when *x* is not NaN."""
    return not np.isnan(x)


def decode_ecmwf_uv_if_necessary(df_element: Any) -> Any:
    """Decode a possibly msgpack-encoded ECMWF u/v value, else return as-is."""
    if isinstance(df_element, dict):
        import msgpack_numpy as msgnum  # noqa: PLC0415 - lazy: optional heavy dep

        msgnum.patch()
        return msgnum.decode(df_element)
    return df_element
