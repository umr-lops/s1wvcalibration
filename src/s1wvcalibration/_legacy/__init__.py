"""Legacy dependencies of the WV calibration engine, ported to Python 3.

These modules were extracted from the ``mpcsentinellibs`` production scripts
(``colocation/`` and ``qualitycheck/``). They are re-packaged with relative
imports, cleaned up (no ``pdb``, no ``print``, logging only) and with the hard
cluster paths moved into the package :func:`~s1wvcalibration.config.load_config`.
"""
