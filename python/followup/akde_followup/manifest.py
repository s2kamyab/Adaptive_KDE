"""Run manifests: what code, what configuration, what inputs, what machine.

A manifest is written next to every result file this follow-up produces.  It
records enough to answer, months later, "which source tree and which settings
produced this row, and can I tell whether a cached row is still valid?"

Hashing rules
-------------
* Never ``hash()``: Python salts string hashes per interpreter process.  All
  hashes here are SHA-256 over canonical bytes.
* Configurations are hashed after JSON canonicalisation (sorted keys, no
  whitespace variance), so key order cannot change a hash.
* Result files are hashed over their raw bytes.
* The *dependency hash* of a result is the hash of (code hash, config hash,
  protocol hash).  Two rows are only interchangeable when their dependency
  hashes agree, which is the rule the brief asks for before reusing a cached
  row.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "sha256_bytes", "sha256_file", "canonical_json", "config_hash",
    "code_hash", "environment", "write_manifest", "dependency_hash",
]

PY_ROOT = Path(__file__).resolve().parents[2]          # .../Adaptive_KDE/python
REPO_ROOT = PY_ROOT.parent                             # .../Adaptive_KDE


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj) -> bytes:
    def default(o):
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=default).encode("utf-8")


def config_hash(cfg) -> str:
    return sha256_bytes(canonical_json(cfg))


def code_hash(roots=("akde", "followup/akde_followup")) -> dict:
    """Per-file SHA-256 of every tracked source file, plus one combined digest."""
    files: dict[str, str] = {}
    for root in roots:
        base = PY_ROOT / root
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            files[str(p.relative_to(PY_ROOT)).replace("\\", "/")] = sha256_file(p)
    combined = sha256_bytes(canonical_json(files))
    return {"files": files, "combined": combined}


def _git_commit() -> dict:
    out = {"repo": str(REPO_ROOT), "commit": None, "dirty": None,
           "note": "the python/ tree is untracked in this repository; "
                   "the commit below describes the surrounding repo only"}
    try:
        c = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        if c.returncode == 0:
            out["commit"] = c.stdout.strip()
        s = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=20)
        if s.returncode == 0:
            out["dirty"] = bool(s.stdout.strip())
            out["n_dirty_paths"] = len([l for l in s.stdout.splitlines() if l.strip()])
        t = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "python"],
                           capture_output=True, text=True, timeout=20)
        if t.returncode == 0:
            out["python_tree_tracked"] = bool(t.stdout.strip())
    except Exception as exc:                                   # pragma: no cover
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def environment() -> dict:
    import numpy as np
    mods = {}
    for name in ("numpy", "scipy", "pandas", "sklearn", "numba", "joblib",
                 "matplotlib", "statsmodels"):
        try:
            mods[name] = __import__(name).__version__
        except Exception:
            mods[name] = None
    try:
        import numba
        numba_threads = numba.get_num_threads()
    except Exception:
        numba_threads = None
    try:
        from akde.solvers import HAVE_NUMBA
    except Exception:
        HAVE_NUMBA = None
    thread_env = {k: os.environ.get(k) for k in
                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                   "NUMBA_NUM_THREADS", "PYTHONHASHSEED")}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "logical_cpus": os.cpu_count(),
        "modules": mods,
        "solver_backend": "numba-jit" if HAVE_NUMBA else "numpy-fallback",
        "numba_threads": numba_threads,
        "thread_env": thread_env,
        "numpy_blas": _blas_info(),
    }


def _blas_info() -> dict:
    try:
        import numpy as np
        cfg = np.__config__.CONFIG                      # numpy >= 2
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        return {k: blas.get(k) for k in ("name", "version", "pc file dir")}
    except Exception:
        return {}


def dependency_hash(code: dict, cfg_hash: str, protocol_hash: str | None = None) -> str:
    return sha256_bytes(canonical_json({
        "code": code["combined"], "config": cfg_hash, "protocol": protocol_hash,
    }))


def write_manifest(path: str | Path, *, stage: str, config, inputs=None,
                   outputs=None, protocol_hash: str | None = None,
                   extra: dict | None = None) -> dict:
    """Write ``path`` (JSON) describing one executed stage.  Returns the manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    code = code_hash()
    cfgh = config_hash(config)
    man = {
        "stage": stage,
        "config": config if isinstance(config, dict) else str(config),
        "config_sha256": cfgh,
        "code_sha256_combined": code["combined"],
        "code_sha256_files": code["files"],
        "dependency_sha256": dependency_hash(code, cfgh, protocol_hash),
        "protocol_sha256": protocol_hash,
        "git": _git_commit(),
        "environment": environment(),
        "inputs": {str(k): (sha256_file(v) if Path(v).exists() else None)
                   for k, v in (inputs or {}).items()},
        "outputs": {str(k): (sha256_file(v) if Path(v).exists() else None)
                    for k, v in (outputs or {}).items()},
    }
    if extra:
        man["extra"] = extra
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    return man
