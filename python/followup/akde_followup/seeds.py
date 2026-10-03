"""Seed namespaces for the follow-up study, and the checks that keep them honest.

Three rules, all checkable:

1. **No salted hashing.**  Target names enter a seed key through ``crc32``
   (the existing ``akde.runner.name_key``), which is stable across processes,
   machines and Python versions.  ``hash()`` is not used anywhere.

2. **Disjoint namespaces.**  Development (Stage A/B) and confirmation
   (Stage C) draw from different master seeds *and* carry a distinct
   ``namespace`` integer inside every key, so no confirmation dataset can
   coincide with a development dataset even by accident.  V2's namespace is
   recorded too, so follow-up confirmation data is disjoint from the V2
   benchmark that was used for diagnosis.

3. **Pairing across methods.**  Within one (namespace, target, N, rep) the
   *data* stream is fixed, so every method sees the identical realisation.
   Estimator-internal randomness uses a separate stream indexed by method, so
   adding or reordering methods cannot perturb another method's data.

``validate_seed_keys`` checks 1-3 empirically over the grid actually used.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

__all__ = ["NAMESPACES", "name_key", "data_rng", "fit_rng", "seed_key",
           "validate_seed_keys"]

#: Namespace integers. V2 used the implicit namespace 0 (its key had no
#: namespace field); the follow-up namespaces are non-zero and distinct, and
#: the key carries one extra field, so V2 keys can never be regenerated here.
NAMESPACES = {
    "v2_main": 0,          # historical, for reference only - never regenerated
    "development": 11,     # Stage A + Stage B
    "confirmation": 23,    # Stage C
    "figure_dev": 31,      # Figure A development reference fits
}

#: Master seeds, one per namespace. Different from V2's 20260915.
MASTERS = {
    "development": 20260927,
    "confirmation": 20260928,
    "figure_dev": 20260929,
}

DATA_STREAM = 0
TEST_STREAM = 1
FIT_STREAM_BASE = 100


def name_key(name: str) -> int:
    """Stable integer key for a target name (crc32, never ``hash()``)."""
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def seed_key(namespace: str, target: str, n: int, rep: int, stream: int) -> list[int]:
    if namespace not in NAMESPACES:
        raise KeyError(f"unknown seed namespace {namespace!r}")
    return [int(MASTERS[namespace]), int(NAMESPACES[namespace]),
            name_key(target), int(n), int(rep), int(stream)]


def _rng(key: list[int]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(key))


def data_rng(namespace: str, target: str, n: int, rep: int) -> np.random.Generator:
    """The dataset stream.  Identical for every method -> pairing holds."""
    return _rng(seed_key(namespace, target, n, rep, DATA_STREAM))


def test_rng(namespace: str, target: str, n: int, rep: int) -> np.random.Generator:
    return _rng(seed_key(namespace, target, n, rep, TEST_STREAM))


def fit_rng(namespace: str, target: str, n: int, rep: int,
            method_index: int) -> np.random.Generator:
    """Estimator-internal stream, separate per method."""
    return _rng(seed_key(namespace, target, n, rep, FIT_STREAM_BASE + int(method_index)))


@dataclass
class SeedReport:
    n_keys: int
    n_unique_keys: int
    n_unique_states: int
    pairing_ok: bool
    dev_conf_disjoint: bool
    v2_disjoint: bool
    collisions: list


def validate_seed_keys(targets, sizes, reps, n_methods: int,
                       namespaces=("development", "confirmation")) -> SeedReport:
    """Check key uniqueness, pairing, and namespace disjointness empirically."""
    keys, states = [], []
    for ns in namespaces:
        for t in targets:
            for n in sizes:
                for r in range(reps):
                    for s in [DATA_STREAM, TEST_STREAM] + \
                             [FIT_STREAM_BASE + i for i in range(n_methods)]:
                        k = tuple(seed_key(ns, t, n, r, s))
                        keys.append(k)
                        states.append(int(_rng(list(k)).integers(0, 2 ** 62)))
    n = len(keys)
    uk, us = len(set(keys)), len(set(states))
    collisions = []
    if us != n:
        seen: dict[int, tuple] = {}
        for k, st in zip(keys, states):
            if st in seen and seen[st] != k:
                collisions.append((seen[st], k))
            seen[st] = k

    # pairing: the data stream must not depend on the method index
    t0, n0 = targets[0], sizes[0]
    a = data_rng("development", t0, n0, 0).standard_normal(8)
    b = data_rng("development", t0, n0, 0).standard_normal(8)
    pairing_ok = bool(np.array_equal(a, b))

    # disjointness: same (target, N, rep) must give different data across namespaces
    d = data_rng("development", t0, n0, 0).standard_normal(8)
    c = data_rng("confirmation", t0, n0, 0).standard_normal(8)
    dev_conf_disjoint = not np.allclose(d, c)

    # V2 used SeedSequence([20260915, crc32(target), N, rep, stream]) - 5 fields
    v2 = np.random.default_rng(np.random.SeedSequence(
        [20260915, name_key(t0), n0, 0, 0])).standard_normal(8)
    v2_disjoint = (not np.allclose(v2, d)) and (not np.allclose(v2, c))

    return SeedReport(n, uk, us, pairing_ok, dev_conf_disjoint, v2_disjoint, collisions)
