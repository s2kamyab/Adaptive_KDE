"""Assemble final_package/ : code snapshot, results, configs, manifests.

Idempotent. Copies rather than moves, so nothing historical is disturbed.
Writes a file manifest with SHA-256 for every included file and a
fit-to-table/figure mapping.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PY = Path(__file__).resolve().parents[2]
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

import followup                                    # noqa: E402,F401
import pandas as pd                                # noqa: E402

from followup.akde_followup.manifest import (      # noqa: E402
    code_hash, environment, sha256_file)

PROJ = PY.parents[1]
FP = PROJ / "final_package"
RES = PY / "followup" / "results"

EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}


def copytree(src: Path, dst: Path, patterns=None):
    n = 0
    for p in sorted(src.rglob("*")):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        if patterns and not any(p.match(pat) for pat in patterns):
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    return n


def main() -> int:
    FP.mkdir(parents=True, exist_ok=True)
    for d in ("environment", "configs", "code", "results/raw",
              "results/diagnostics", "results/summaries", "figures/data",
              "paper/figures", "paper/tables", "manifests"):
        (FP / d).mkdir(parents=True, exist_ok=True)

    log = {}

    # ---------------- code snapshot -------------------------------------
    log["code_akde"] = copytree(PY / "akde", FP / "code" / "akde", ["*.py"])
    log["code_followup"] = copytree(PY / "followup" / "akde_followup",
                                    FP / "code" / "followup" / "akde_followup", ["*.py"])
    log["code_scripts"] = copytree(PY / "followup" / "scripts",
                                   FP / "code" / "followup" / "scripts", ["*.py"])
    log["code_verify"] = copytree(PY / "followup" / "verify",
                                  FP / "code" / "followup" / "verify", ["*.py"])
    log["code_v2_frozen"] = copytree(PY / "followup" / "akde_v2_original",
                                     FP / "code" / "followup" / "akde_v2_original",
                                     ["*.py", "*.md"])
    log["code_tests"] = copytree(PY / "tests", FP / "code" / "tests", ["*.py"])
    # top-level modules of the followup package (__init__.py pins BLAS,
    # paths.py resolves the results roots for either layout)
    n_top = 0
    for p in sorted((PY / "followup").glob("*.py")):
        shutil.copy2(p, FP / "code" / "followup" / p.name)
        n_top += 1
    log["code_followup_toplevel"] = n_top
    for extra in ("requirements.txt",):
        if (PY / extra).exists():
            shutil.copy2(PY / extra, FP / "environment" / extra)

    # ---------------- configs -------------------------------------------
    for src, name in ((PROJ / "protocol_followup_locked.yaml",
                       "protocol_followup_locked.yaml"),
                      (RES / "finalist.json", "finalist.json"),
                      (RES / "power" / "replication_choice.json",
                       "replication_choice.json"),
                      (PY / "results" / "main_full" / "config.json",
                       "v2_benchmark_config.json"),
                      (RES / "stage_c" / "stage_config.json"
                       if (RES / "stage_c" / "stage_config.json").exists()
                       else RES / "stage_a" / "stage_config.json",
                       "stage_config_example.json"),
                      (PY / "followup" / "verify" / "check_config.py",
                       "frozen_numerical_check_config.py"),
                      (RES / "statistics" / "bootstrap_config.json",
                       "bootstrap_config.json"),
                      (RES / "verification" / "cost_measurement_config.json",
                       "cost_measurement_config.json")):
        if Path(src).exists():
            shutil.copy2(src, FP / "configs" / name)

    # ---------------- results -------------------------------------------
    for stage in ("stage_a", "stage_b", "stage_b_extra", "stage_c",
                  "dev_references"):
        src = RES / stage
        if not src.exists():
            continue
        log["raw_" + stage] = copytree(src, FP / "results" / "raw" / stage,
                                       ["*.csv", "*.json"])
        fits = src / "fits"
        if fits.exists():
            log["fits_" + stage] = copytree(fits,
                                            FP / "results" / "raw" / stage / "fits",
                                            ["*.npz"])
    # the historical V2 benchmark rows, preserved read-only
    log["raw_v2_benchmark"] = copytree(PY / "results" / "main_full",
                                       FP / "results" / "raw" / "v2_benchmark_2026-09-22",
                                       ["*.csv", "*.json"])
    log["raw_v2_moment_count"] = copytree(PY / "results" / "moment_count",
                                          FP / "results" / "raw" / "v2_moment_count",
                                          ["*.csv"])

    log["diagnostics"] = copytree(RES / "verification",
                                  FP / "results" / "diagnostics", ["*.csv", "*.json"])
    for extra in ("oracle_controls", "criterion_geometry", "solver_audit", "cost",
                  "power"):
        if (RES / extra).exists():
            log["diag_" + extra] = copytree(RES / extra,
                                            FP / "results" / "diagnostics" / extra,
                                            ["*.csv", "*.json"])
    log["summaries"] = copytree(RES / "statistics", FP / "results" / "summaries",
                                ["*.csv", "*.json", "*.tex"])

    # ---------------- figures -------------------------------------------
    log["figures"] = copytree(PY / "followup" / "paper_figures", FP / "figures",
                              ["*.pdf", "*.png", "*.svg", "*.md", "*.npz",
                               "*.csv", "*.json"])
    for p in sorted((PY / "followup" / "paper_figures").glob("fig*.pdf")):
        shutil.copy2(p, FP / "paper" / "figures" / p.name)
    for p in sorted((RES / "statistics" / "tables").glob("*.tex")):
        shutil.copy2(p, FP / "paper" / "tables" / p.name)

    # ---------------- top-level reports ---------------------------------
    for name in ("audit_findings.md", "diagnosis_report.md", "method_changes.md",
                 "results_summary.md", "theory_checks.md",
                 "manuscript_patch.md", "EXPERIMENT_LOG.md",
                 "ADDITIONAL_EXPERIMENTS.md", "REPRODUCE.md"):
        if (PROJ / name).exists():
            shutil.copy2(PROJ / name, FP / "results" / "summaries" / name)

    # ---------------- environment ---------------------------------------
    env = environment()
    with open(FP / "environment" / "environment.json", "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2)
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=180)
        if freeze.returncode == 0:
            (FP / "environment" / "pip_freeze.txt").write_text(freeze.stdout,
                                                               encoding="utf-8")
            log["pip_freeze_lines"] = len(freeze.stdout.splitlines())
    except Exception as exc:                                    # pragma: no cover
        log["pip_freeze_error"] = str(exc)

    # ---------------- manifests ------------------------------------------
    rows = []
    for p in sorted(FP.rglob("*")):
        if not p.is_file() or "manifests" in p.relative_to(FP).parts:
            continue
        rows.append({"path": str(p.relative_to(FP)).replace("\\", "/"),
                     "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    man = pd.DataFrame(rows)
    man.to_csv(FP / "manifests" / "file_manifest.csv", index=False)

    mapping = {
        "Table: benchmark mean ISE": {
            "tex": "paper/tables/tab_benchmark_mean.tex",
            "from": "results/summaries/benchmark_means.csv",
            "built_by": "code/followup/verify/statistics.py",
            "raw": "results/raw/stage_c/all_rows.csv"},
        "Table: fixed-benchmark ratio": {
            "tex": "paper/tables/tab_fixed_benchmark_ratio.tex",
            "from": "results/summaries/fixed_benchmark.csv",
            "built_by": "code/followup/verify/statistics.py",
            "raw": "results/raw/stage_c/all_rows.csv"},
        "Table: cost": {
            "tex": "paper/tables/tab_cost.tex",
            "from": "results/summaries/per_cell_summary.csv",
            "built_by": "code/followup/verify/statistics.py",
            "raw": "results/raw/stage_c/all_rows.csv"},
        "Tables: per-target supplement": {
            "tex": "paper/tables/tab_per_target_N{200,1000,5000}.tex",
            "from": "results/summaries/per_cell_summary.csv",
            "built_by": "code/followup/verify/statistics.py",
            "raw": "results/raw/stage_c/all_rows.csv"},
        "Figure 1 fitted densities": {
            "pdf": "figures/fig1_fitted_densities.pdf",
            "data": "figures/data/fig1_fitted_densities_curves.npz + _metrics.csv",
            "built_by": "code/followup/verify/make_paper_figures.py --figure 1",
            "raw": "results/raw/stage_c/fits/*_r0_*.npz"},
        "Figure 2 variability": {
            "pdf": "figures/fig2_variability_N1000.pdf",
            "data": "figures/data/fig2_variability_N1000_curves.npz",
            "built_by": "code/followup/verify/make_paper_figures.py --figure 2",
            "raw": "results/raw/stage_c/fits/*_N1000_*.npz"},
        "Figure 3 accuracy": {
            "pdf": "figures/fig3_accuracy_vs_N.pdf",
            "data": "figures/data/fig3_accuracy_vs_N_metrics.csv",
            "built_by": "code/followup/verify/make_paper_figures.py --figure 3",
            "raw": "results/summaries/*.csv"},
        "Figure 4 mechanism": {
            "pdf": "figures/fig4_mechanism.pdf",
            "data": "figures/data/fig4_mechanism_metrics.csv",
            "built_by": "code/followup/verify/make_paper_figures.py --figure 4",
            "raw": "results/diagnostics/oracle_ladder_check.csv, "
                   "ambiguity_cone_witnesses.csv, solver_verification.csv"},
        "Figure 5 cost": {
            "pdf": "figures/fig5_cost_and_compactness.pdf",
            "data": "figures/data/fig5_cost_and_compactness_metrics.csv",
            "built_by": "code/followup/verify/make_paper_figures.py --figure 5",
            "raw": "results/diagnostics/model_size_and_query_cost.csv"},
    }
    with open(FP / "manifests" / "fit_to_output_mapping.json", "w",
              encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2)

    prov = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(man), "total_bytes": int(man.bytes.sum()),
        "code": code_hash(roots=("akde", "followup/akde_followup",
                                 "followup/scripts", "followup/verify")),
        "environment": env,
        "copy_log": log,
        "notes": [
            "Historical results are copied, never moved or overwritten.",
            "results/raw/v2_benchmark_2026-09-22/ is the earlier 27,000-fit "
            "benchmark and belongs to the frozen implementation snapshot in "
            "code/followup/akde_v2_original/.",
            "__pycache__ and .git are excluded.",
        ],
    }
    with open(FP / "manifests" / "provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)

    print("package assembled at", FP)
    for k, v in sorted(log.items()):
        print("  %-28s %s" % (k, v))
    print("  %-28s %d files, %.1f MB" % ("TOTAL", len(man), man.bytes.sum() / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
