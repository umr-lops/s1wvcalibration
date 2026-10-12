from __future__ import annotations

import importlib.metadata

import s1wvcalibration as m


def test_version():
    assert importlib.metadata.version("s1wvcalibration") == m.__version__
