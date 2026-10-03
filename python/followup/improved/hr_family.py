"""The scale-heterogeneity family and seed namespaces of the bandwidth-
flexibility experiment (final_package/experiments/bandwidth_flexibility),
reproduced here so this study is self-contained.  Same definitions, same seed
keys, so datasets are IDENTICAL to the ones the saved rows there were fitted
on and the comparison is paired.
"""
from __future__ import annotations

import zlib

import numpy as np

from akde.densities import MixtureSpec

R_VALUES = (1, 2, 4, 8)
NAMESPACES = {"bf_dev": 43, "bf_main": 41}
MASTERS = {"bf_dev": 20261001, "bf_main": 20260930}
DATA_STREAM, TEST_STREAM, FOLD_STREAM = 0, 1, 100


def target_name(r):
    return "hr%d" % int(r)


def r_of(name):
    return int(name[2:])


def make_target(r):
    r = int(r)
    return MixtureSpec(name=target_name(r), weights=np.array([0.75, 0.25]),
                       means=np.array([[0.0], [2.0]]), variances=np.array([[1.0], [1.0 / r ** 2]]),
                       label="0.75 N(0,1) + 0.25 N(2, 1/%d^2)" % r,
                       meta={"suite": "bandwidth_flexibility", "r": r})


def narrow_region(r):
    return (2.0 - 2.0 / r, 2.0 + 2.0 / r)


def _key(ns, target, n, rep, stream):
    return [MASTERS[ns], NAMESPACES[ns], zlib.crc32(target.encode()) & 0x7FFFFFFF, int(n), int(rep), int(stream)]


def _rng(key):
    return np.random.default_rng(np.random.SeedSequence(key))


def data_rng(ns, t, n, rep):
    return _rng(_key(ns, t, n, rep, DATA_STREAM))


def test_rng(ns, t, n, rep):
    return _rng(_key(ns, t, n, rep, TEST_STREAM))


def fold_rng(ns, t, n, rep):
    return _rng(_key(ns, t, n, rep, FOLD_STREAM))
