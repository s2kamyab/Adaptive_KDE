"""Layout-independent resolution of the results and figures roots.

The study was developed in a repository where results live at
``<python_root>/followup/results``. The distributed package instead has
``code/`` and ``results/`` as siblings. Rather than hard-coding either, every
script resolves its roots here, in this order:

1. the environment variables ``AKDE_RESULTS_ROOT`` / ``AKDE_FIGURES_ROOT``,
   if set (absolute or relative to the current working directory);
2. the repository layout ``<python_root>/followup/results``, if it exists;
3. the package layout ``<python_root>/../results``, if it exists;

and fall back to (2) so a fresh run creates the repository layout.

This is why the package can be unpacked anywhere and run without editing
paths, and why no script contains a machine-specific directory.
"""
from __future__ import annotations

import os
from pathlib import Path

PY_ROOT = Path(__file__).resolve().parents[1]      # .../python  or  .../code


def _populated(root: Path) -> bool:
    """A results root counts only if it actually holds raw confirmation rows.

    Merely existing is not enough: a script that writes diagnostics can create
    the directory, and an empty directory must not win the search and shadow
    the real data.
    """
    return ((root / "stage_c" / "all_rows.csv").exists()
            or (root / "raw" / "stage_c" / "all_rows.csv").exists())


def _resolve(env_var: str, repo_rel: str, pkg_rel: str,
             require_data: bool = False) -> Path:
    v = os.environ.get(env_var)
    if v:
        return Path(v).expanduser().resolve()
    repo = (PY_ROOT / repo_rel).resolve()
    pkg = (PY_ROOT.parent / pkg_rel).resolve()
    if require_data:
        if _populated(repo):
            return repo
        if _populated(pkg):
            return pkg
    if repo.exists():
        return repo
    if pkg.exists():
        return pkg
    return repo


RESULTS_ROOT = _resolve("AKDE_RESULTS_ROOT", "followup/results", "results",
                        require_data=True)
FIGURES_ROOT = _resolve("AKDE_FIGURES_ROOT", "followup/paper_figures", "figures")

#: Where the raw per-fit rows live. In the repository these are directly under
#: RESULTS_ROOT; in the package they are under RESULTS_ROOT/raw.
RAW_ROOT = (RESULTS_ROOT / "raw") if (RESULTS_ROOT / "raw").exists() else RESULTS_ROOT

#: Diagnostics and summaries: same idea.
DIAG_ROOT = ((RESULTS_ROOT / "diagnostics")
             if (RESULTS_ROOT / "diagnostics").exists()
             else (RESULTS_ROOT / "verification"))
SUMMARY_ROOT = ((RESULTS_ROOT / "summaries")
                if (RESULTS_ROOT / "summaries").exists()
                else (RESULTS_ROOT / "statistics"))


def describe() -> dict:
    return {"python_root": str(PY_ROOT), "results_root": str(RESULTS_ROOT),
            "raw_root": str(RAW_ROOT), "diagnostics_root": str(DIAG_ROOT),
            "summary_root": str(SUMMARY_ROOT), "figures_root": str(FIGURES_ROOT),
            "layout": "package" if RAW_ROOT != RESULTS_ROOT else "repository"}


__all__ = ["PY_ROOT", "RESULTS_ROOT", "RAW_ROOT", "DIAG_ROOT", "SUMMARY_ROOT",
           "FIGURES_ROOT", "describe"]
