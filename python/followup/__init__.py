"""Follow-up study package.

BLAS thread pinning happens HERE, before any submodule imports numpy.  The
matrices in this study are small (m x m with m <= 600, M x M with M = 300), so
multi-threaded BLAS is pure overhead: the threads spend their time
synchronising rather than working, and when several cells run in parallel the
machine is oversubscribed many times over.  Setting the variables after numpy
is imported has no effect, which is the mistake this module exists to prevent.

Override by exporting the variables yourself before starting Python.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
del os, _v
