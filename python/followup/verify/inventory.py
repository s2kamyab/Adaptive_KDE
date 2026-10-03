"""Task 1 - establish the authoritative inputs and record provenance.

Produces, in ``followup/results/verification/``:

    input_inventory.csv      every input/output artefact with size, mtime, sha256
    provenance_manifest.json code/config/environment versions + the verified
                             structural facts about the confirmation grid
    grid_verification.json   counts, uniqueness, pairing, missing rows, the
                             three distinct status families, seed disjointness

Nothing here trusts a prose report: every number is recomputed from the saved
rows or the files themselves.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PY = Path(__file__).resolve().parents[2]          # .../Adaptive_KDE/python
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

import followup                                    # noqa: E402,F401  (pins BLAS)
import numpy as np                                 # noqa: E402
import pandas as pd                                # noqa: E402

from followup.akde_followup.manifest import (      # noqa: E402
    code_hash, config_hash, environment, sha256_file)

PROJ = PY.parents[1]                               # .../Adaptive_KDE_project
from followup.paths import DIAG_ROOT, RAW_ROOT   # noqa: E402
OUT = DIAG_ROOT

# Artefacts that this study depends on or produces.  (role, path)
ARTEFACTS = [
    ("manuscript_v2_pdf", PROJ / "Adaptive_KDE_V2 (1).pdf"),
    ("manuscript_v2_tex", PROJ / "Adaptive_KDE_GMM_revised.tex"),
    ("manuscript_v2_text_dump", PROJ / "paper_text.txt"),
    ("report_audit", PROJ / "audit_findings.md"),
    ("report_diagnosis", PROJ / "diagnosis_report.md"),
    ("report_method_changes", PROJ / "method_changes.md"),
    ("report_results_summary", PROJ / "results_summary.md"),
    ("report_manuscript_patch", PROJ / "manuscript_patch.md"),
    ("report_experiment_log", PROJ / "EXPERIMENT_LOG.md"),
    ("report_additional_experiments", PROJ / "ADDITIONAL_EXPERIMENTS.md"),
    ("report_reproduce", PROJ / "REPRODUCE.md"),
    ("protocol_locked", PROJ / "protocol_followup_locked.yaml"),
    ("design_recommendations", PY.parent / "Experimental_comparison_recommendations.md"),
    ("v2_benchmark_rows", PY / "results" / "main_full" / "all_results.csv"),
    ("v2_benchmark_config", PY / "results" / "main_full" / "config.json"),
    ("v2_moment_count_rows", PY / "results" / "moment_count" / "raw.csv"),
    ("finalist_config", PY / "followup" / "results" / "finalist.json"),
    ("replication_choice", PY / "followup" / "results" / "power" / "replication_choice.json"),
    ("stage_a_rows", PY / "followup" / "results" / "stage_a" / "all_rows.csv"),
    ("stage_b_rows", PY / "followup" / "results" / "stage_b" / "all_rows.csv"),
    ("stage_b_extra_rows", PY / "followup" / "results" / "stage_b_extra" / "all_rows.csv"),
    ("stage_c_rows", PY / "followup" / "results" / "stage_c" / "all_rows.csv"),
    ("stage_c_manifest", PY / "followup" / "results" / "stage_c" / "manifest.json"),
    ("dev_reference_rows", PY / "followup" / "results" / "dev_references" / "all_rows.csv"),
    ("oracle_ladder_base", PY / "followup" / "results" / "oracle_controls" / "ladder_base_rho0.001.csv"),
    ("oracle_ladder_matched", PY / "followup" / "results" / "oracle_controls" / "ladder_matched_rho0.001.csv"),
    ("oracle_ambiguity_base", PY / "followup" / "results" / "oracle_controls" / "ambiguity_base_rho0.001.csv"),
    ("oracle_ambiguity_matched", PY / "followup" / "results" / "oracle_controls" / "ambiguity_matched_rho0.001.csv"),
    ("criterion_geometry", PY / "followup" / "results" / "criterion_geometry" / "criterion_geometry.csv"),
    ("solver_audit", PY / "followup" / "results" / "solver_audit" / "stage1_solver_comparison.csv"),
    ("cost_profile", PY / "followup" / "results" / "cost" / "phase_profile.csv"),
    ("study_manifest", PY / "followup" / "results" / "manifest_study.json"),
]

DIRS = [
    ("stage_a_fits", PY / "followup" / "results" / "stage_a" / "fits"),
    ("stage_b_fits", PY / "followup" / "results" / "stage_b" / "fits"),
    ("stage_b_extra_fits", PY / "followup" / "results" / "stage_b_extra" / "fits"),
    ("stage_c_fits", PY / "followup" / "results" / "stage_c" / "fits"),
    ("dev_reference_fits", PY / "followup" / "results" / "dev_references" / "fits"),
    ("figures", PY / "followup" / "figures"),
    ("figure_data", PY / "followup" / "figures" / "data"),
    ("v2_frozen_snapshot", PY / "followup" / "akde_v2_original" / "akde"),
    ("live_akde_package", PY / "akde"),
    ("followup_package", PY / "followup" / "akde_followup"),
    ("followup_scripts", PY / "followup" / "scripts"),
    ("user_backup_snapshot", PROJ / "Adaptive_KDE copy"),
]


def inventory_rows():
    rows = []
    for role, p in ARTEFACTS:
        p = Path(p)
        rows.append({
            "role": role, "kind": "file", "path": str(p.relative_to(PROJ)) if
            str(p).startswith(str(PROJ)) else str(p),
            "exists": p.exists(),
            "bytes": p.stat().st_size if p.exists() else 0,
            "mtime_utc": (datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                          .isoformat(timespec="seconds") if p.exists() else ""),
            "sha256": sha256_file(p) if p.exists() else "",
            "n_files": 1 if p.exists() else 0,
        })
    for role, d in DIRS:
        d = Path(d)
        files = sorted(f for f in d.rglob("*") if f.is_file()
                       and "__pycache__" not in f.parts) if d.exists() else []
        h = hashlib.sha256()
        total = 0
        for f in files:
            h.update(str(f.relative_to(d)).replace("\\", "/").encode())
            h.update(sha256_file(f).encode())
            total += f.stat().st_size
        rows.append({
            "role": role, "kind": "directory",
            "path": str(d.relative_to(PROJ)) if str(d).startswith(str(PROJ)) else str(d),
            "exists": d.exists(), "bytes": total,
            "mtime_utc": (datetime.fromtimestamp(d.stat().st_mtime, timezone.utc)
                          .isoformat(timespec="seconds") if d.exists() else ""),
            "sha256": h.hexdigest() if files else "",
            "n_files": len(files),
        })
    return pd.DataFrame(rows)


def verify_grid():
    """Everything structural about the confirmation, recomputed from the rows."""
    p = PY / "followup" / "results" / "stage_c" / "all_rows.csv"
    d = pd.read_csv(p, low_memory=False)
    err = d["error"].fillna("").astype(str)
    targets = sorted(d.target.unique())
    sizes = sorted(int(x) for x in d.N.unique())
    methods = sorted(d.variant.unique())
    reps = sorted(int(x) for x in d.rep.unique())

    key = d[["target", "N", "rep", "variant"]]
    full = pd.MultiIndex.from_product([targets, sizes, reps, methods],
                                      names=["target", "N", "rep", "variant"])
    missing = full.difference(pd.MultiIndex.from_frame(key))

    # --- three distinct status families, never conflated -------------------
    valid = d["is_valid_density"].astype(str).isin(["True"])
    stat = {
        "execution": {
            "n_rows": int(len(d)),
            "n_execution_failures": int((err != "").sum()),
            "n_ise_nonfinite": int((~np.isfinite(d.ise)).sum()),
            "n_ise_nonpositive": int((d.ise <= 0).sum()),
        },
        "density_feasibility": {
            "n_valid_density": int(valid.sum()),
            "n_invalid_density": int((~valid).sum()),
            "max_abs_mass_error": float(d["mass_error"].max()),
            "n_fallback_used": int(pd.to_numeric(
                d.get("fallback_used", pd.Series(dtype=float))
                .astype(str).map({"True": 1, "False": 0}), errors="coerce").fillna(0).sum()),
            "n_support_capped": int(pd.to_numeric(
                d.get("support_capped", pd.Series(dtype=float))
                .astype(str).map({"True": 1, "False": 0}), errors="coerce").fillna(0).sum()),
        },
        "optimisation_convergence": {},
    }
    for m in ("v2_akde", "finalist", "spades_simplex"):
        sub = d[d.variant == m]
        if "stage1_converged" not in sub or sub["stage1_converged"].isna().all():
            continue
        conv = sub["stage1_converged"].astype(str).isin(["True"])
        stat["optimisation_convergence"][m] = {
            "n_fits": int(len(sub)),
            "n_stage1_converged": int(conv.sum()),
            "frac_stage1_converged": float(conv.mean()),
            "median_stage1_rel_kkt": float(sub["stage1_kkt"].median()),
            "max_stage1_rel_kkt": float(sub["stage1_kkt"].max()),
            "median_stage2_rel_kkt": float(sub["stage2_kkt"].median()),
            "max_stage2_rel_kkt": float(sub["stage2_kkt"].max()),
        }

    return {
        "source": str(p.relative_to(PROJ)),
        "sha256": sha256_file(p),
        "n_rows": int(len(d)),
        "n_targets": len(targets), "targets": targets,
        "sample_sizes": sizes,
        "n_replications": len(reps), "replications": [reps[0], reps[-1]],
        "n_methods": len(methods), "methods": methods,
        "expected_rows": len(targets) * len(sizes) * len(reps) * len(methods),
        "grid_complete": len(missing) == 0,
        "n_missing_cells": int(len(missing)),
        "n_duplicate_keys": int(key.duplicated().sum()),
        "statuses": stat,
    }


def verify_pairing_and_seeds():
    """Pairing across methods, and disjointness of the three seed namespaces."""
    from akde.densities import get_target
    from followup.akde_followup.seeds import (MASTERS, NAMESPACES, data_rng,
                                              validate_seed_keys)
    from akde.runner import _child_seed

    out = {}
    # (a) within a (target, N, rep) the dataset stream is method-independent
    a = get_target("f9").sample(6, data_rng("confirmation", "f9", 1000, 3))
    b = get_target("f9").sample(6, data_rng("confirmation", "f9", 1000, 3))
    out["pairing_dataset_stream_method_independent"] = bool(np.array_equal(a, b))

    # (b) key uniqueness / disjointness over the grid actually run
    rep = validate_seed_keys([f"f{i}" for i in range(15)], [200, 1000, 5000], 30, 6)
    out["seed_keys_checked"] = rep.n_keys
    out["seed_keys_unique"] = rep.n_unique_keys == rep.n_keys
    out["seed_first_draws_unique"] = rep.n_unique_states == rep.n_keys
    out["dev_confirmation_disjoint"] = rep.dev_conf_disjoint
    out["v2_disjoint_from_followup"] = rep.v2_disjoint

    # (c) explicit dataset-level disjointness on a sample of cells
    coll = 0
    checked = 0
    for t in ["f0", "f5", "f9"]:
        for N in [200, 1000, 5000]:
            for r in range(5):
                dev = get_target(t).sample(min(N, 50), data_rng("development", t, N, r))
                con = get_target(t).sample(min(N, 50), data_rng("confirmation", t, N, r))
                v2 = get_target(t).sample(min(N, 50), _child_seed(20260915, t, N, r, 0))
                checked += 3
                coll += int(np.allclose(dev, con)) + int(np.allclose(dev, v2)) \
                    + int(np.allclose(con, v2))
    out["dataset_level_pairs_checked"] = checked
    out["dataset_level_collisions"] = coll

    out["namespaces"] = {k: {"id": NAMESPACES[k], "master": MASTERS.get(k)}
                         for k in NAMESPACES}
    return out


def protocol_evidence():
    """What the filesystem can and cannot establish about prospective locking."""
    prot = PROJ / "protocol_followup_locked.yaml"
    fin = PY / "followup" / "results" / "finalist.json"
    rep = PY / "followup" / "results" / "power" / "replication_choice.json"
    stage_c_first = sorted((PY / "followup" / "results" / "stage_c").glob("f*_N*.csv"))

    def mt(p):
        return (datetime.fromtimestamp(Path(p).stat().st_mtime, timezone.utc)
                .isoformat(timespec="seconds")) if Path(p).exists() else None

    first_conf = min((f.stat().st_mtime for f in stage_c_first), default=None)
    ev = {
        "protocol_file_mtime_utc": mt(prot),
        "finalist_frozen_utc_claimed": (json.load(open(fin, encoding="utf-8"))
                                        .get("frozen_utc") if fin.exists() else None),
        "replication_decided_utc_claimed": (json.load(open(rep, encoding="utf-8"))
                                            .get("decided_utc") if rep.exists() else None),
        "first_confirmation_shard_mtime_utc": (
            datetime.fromtimestamp(first_conf, timezone.utc).isoformat(timespec="seconds")
            if first_conf else None),
        "caveat": (
            "Filesystem mtimes and content hashes establish ORDERING ON THIS "
            "MACHINE ONLY. They do not constitute third-party timestamping, and "
            "a hash does not by itself prove prospective locking. The freeze is "
            "supported by: (i) finalist.json and replication_choice.json both "
            "carry UTC stamps earlier than the first confirmation shard; (ii) "
            "the confirmation code reads the finalist from finalist.json rather "
            "than choosing it; (iii) the confirmation seed namespace is disjoint "
            "from development. It is NOT supported by any external registry, and "
            "is described in the manuscript as pre-specified, not preregistered."),
    }
    if ev["finalist_frozen_utc_claimed"] and ev["first_confirmation_shard_mtime_utc"]:
        ev["freeze_precedes_first_confirmation_shard"] = (
            ev["finalist_frozen_utc_claimed"] < ev["first_confirmation_shard_mtime_utc"])
    return ev


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inv = inventory_rows()
    inv.to_csv(OUT / "input_inventory.csv", index=False)
    missing = inv[~inv.exists]
    print("inventory: %d entries, %d missing" % (len(inv), len(missing)))
    if len(missing):
        print(missing[["role", "path"]].to_string(index=False))

    grid = verify_grid()
    pair = verify_pairing_and_seeds()
    prot = protocol_evidence()
    with open(OUT / "grid_verification.json", "w", encoding="utf-8") as fh:
        json.dump({"grid": grid, "pairing_and_seeds": pair,
                   "protocol_evidence": prot}, fh, indent=2)

    man = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(PROJ),
        "environment": environment(),
        "code": code_hash(roots=("akde", "followup/akde_followup", "followup/scripts",
                                 "followup/verify")),
        "inventory_sha256": config_hash(inv.to_dict(orient="records")),
        "grid_verification": grid,
        "pairing_and_seeds": pair,
        "protocol_evidence": prot,
    }
    with open(OUT / "provenance_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)

    print("\n== confirmation grid ==")
    print(" rows %d (expected %d), complete=%s, missing=%d, duplicates=%d"
          % (grid["n_rows"], grid["expected_rows"], grid["grid_complete"],
             grid["n_missing_cells"], grid["n_duplicate_keys"]))
    for fam, v in grid["statuses"].items():
        print(" %s: %s" % (fam, json.dumps(v)))
    print("\n== pairing / seeds ==")
    for k in ("pairing_dataset_stream_method_independent", "seed_keys_unique",
              "seed_first_draws_unique", "dev_confirmation_disjoint",
              "v2_disjoint_from_followup", "dataset_level_collisions"):
        print("  %-42s %s" % (k, pair[k]))
    print("\n== protocol evidence ==")
    for k, v in prot.items():
        if k != "caveat":
            print("  %-42s %s" % (k, v))
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
