# `akde_v2_original` — frozen V2 snapshot

A byte-for-byte copy of `Adaptive_KDE/python/akde/` as it stood when the V2
manuscript's results were produced (`results/main_full/`, run 2026-09-22/23).
Taken 2026-09-27 before any follow-up change.

It exists so that:

* the original method and its results stay recoverable no matter what the
  live `akde/` package becomes;
* every follow-up claim of the form "unchanged from V2" can be checked by
  running both copies on the same input and comparing outputs bit-for-bit
  (`followup/scripts/check_v2_identity.py`);
* file hashes recorded in `followup/results/manifest/*.json` pin exactly which
  source produced which result row.

**Do not edit anything in this directory.** Import it as

```python
from followup.akde_v2_original import akde as akde_v2_original
```

The live package is `akde`. Follow-up work adds opt-in flags there whose
defaults reproduce this snapshot exactly; `check_v2_identity.py` is the test
that keeps that promise honest.
