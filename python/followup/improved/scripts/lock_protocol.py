"""Lock the improved-AKDE study protocol BEFORE the development ablation runs.

    python followup/improved/scripts/lock_protocol.py

Writes followup/improved/protocol_improved.yaml, protocol_improved.md and
protocol_improved_lock.json (sha256 + UTC time).  The runner refuses to run
any stage if the yaml's hash differs from the lock.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1]))
import followup.improved  # noqa: E402,F401
from followup.improved.estimator import VARIANTS  # noqa: E402
from followup.improved.runner import code_hash  # noqa: E402

PROTOCOL = {
    "version": "1", "created_utc": "2026-10-02",
    "status": "prospective development-then-confirmation protocol; NOT preregistered: "
              "the repairs were chosen from the findings of the earlier studies",
    "question": "do the recommended repairs (metric-consistent weighting, converged solver, "
                "full-simplex L2 fit, scale-aware penalty, extended lambda grid + 1-SE rule, "
                "CV-anchored scales, post-fit merge) make AKDE more accurate and/or more compact, "
                "and does the repaired estimator still fall behind the conventional comparators?",
    "development": {
        "namespace": "development", "targets": ["f0", "f2", "f5", "f9"], "sizes": [500, 5000],
        "n_reps": 5, "fold_stream": "method index 0 for every variant (shared folds, as Stage A)",
        "hr_namespace": "bf_dev", "hr_targets": ["hr1", "hr8"], "hr_sizes": [500, 5000], "hr_reps": 2,
        "hr_fold_stream": "100 shared",
        "variants": ["I0_finalist"] + list(VARIANTS),
        "n_fits": (4 * 2 * 5 + 2 * 2 * 2) * (1 + len(VARIANTS)),
        "purpose": "ablation and finalist selection only; never reported as confirmation",
    },
    "selection_rule": {
        "primary": "equal-weight-per-cell geometric mean, over the 8 Marron-Wand dev cells, of the "
                   "paired ISE ratio variant / I0_finalist (lower is better)",
        "gates": ["geometric-mean ratio vs I0 on the 4 hr dev cells <= 1.0",
                  "stage-1 AND stage-2 KKT at tolerance in >= 90% of dev fits (stage-1 n/a for simplex_l2)",
                  "no execution error; all fits valid densities"],
        "tie_break": "among variants whose primary ratio is within 10% of the best, prefer the one with "
                     "fewer tuned hyperparameters, then lower mean wall time",
        "ablation_to_confirm": "the finalist; plus I4_simplex_l2 if the finalist is not I4 (the "
                               "parameter-free core), so the contribution of anchor/merge is confirmed too",
        "decided_before_dev_results": True,
    },
    "confirmation": {
        "namespace": "confirmation", "targets": ["f%d" % i for i in range(15)], "sizes": [200, 1000, 5000],
        "n_reps": 30, "fold_stream": "method index 6 (finalist) and 7 (ablation): Stage-C policy, "
                                     "datasets identical to Stage C so comparisons with the saved "
                                     "Stage-C rows of AKDE-M, AKDE-V2, SPADES-simplex, KDE-SJ, Abramson, "
                                     "GMM-EM are paired on data",
        "comparators_reused": "results/stage_c/all_rows.csv (not re-run)",
        "hr_namespace": "bf_main", "hr_targets": ["hr1", "hr2", "hr4", "hr8"], "hr_sizes": [500, 5000],
        "hr_reps": 20, "hr_comparators_reused": "final_package/experiments/bandwidth_flexibility/results/main/all_rows.csv",
        "n_reps_fixed_in_advance": True,
    },
    "metrics": ["ISE (analytic)", "narrow-region ISE on hr", "mass error, validity", "n_components, n_active",
                "minimal storage 24 bytes/component", "tune/fit/wall time", "query latency on saved models "
                "(10000 identical points, batch 2000, 2 warmup, 7 repeats, median within model)",
                "stage-1/2 KKT, iterations, objective, convergence", "selected lambda/rho/anchor and grid-edge flags",
                "merge counts and training-loss change", "large-error flag: ISE > 5x own cell median"],
    "analysis": {"per_cell": "mean +- SD (ddof=1), median", "benchmark": "equal-weight-per-target mean ISE; "
                 "fixed-benchmark exp(T) paired ratio vs AKDE-M and vs SPADES-simplex / KDE-SJ / Abramson / GMM",
                 "bootstrap": {"B": 10000, "seed": 20261002, "unit": "replicate id within (target, N) cell",
                               "targets_never_resampled": True},
                 "tables": "mean +- SD; bootstrap intervals only in labelled tables/figures"},
    "solver_tolerances": {"stage1_fista_kkt": 1e-8, "stage1_max_iter": 20000, "stage2_kkt": 1e-7,
                          "simplex_l2": "fully corrective Frank-Wolfe, 400 rounds, active-set subproblems"},
    "seeds": {"development": "master 20260927 id 11", "confirmation": "master 20260928 id 23",
              "bf_dev": "master 20261001 id 43", "bf_main": "master 20260930 id 41"},
}


def to_yaml(obj, indent=0):
    pad = "  " * indent
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                out.append("%s%s:" % (pad, k)); out.append(to_yaml(v, indent + 1))
            elif isinstance(v, (list, tuple)):
                out.append("%s%s: [%s]" % (pad, k, ", ".join(json.dumps(x) for x in v)))
            else:
                out.append("%s%s: %s" % (pad, k, json.dumps(v)))
    return "\n".join(out)


def main():
    lock = HERE / "protocol_improved_lock.json"
    if lock.exists():
        print("already locked:", json.load(open(lock))["protocol_yaml_sha256"]); return 0
    y = HERE / "protocol_improved.yaml"
    y.write_text("# locked protocol of the improved-AKDE study (rendered by lock_protocol.py)\n"
                 + to_yaml(PROTOCOL) + "\n", encoding="utf-8")
    (HERE / "protocol_improved.json").write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    md = ["# Improved-AKDE study: protocol (locked)", "", PROTOCOL["status"] + ".", "",
          "**Question.** " + PROTOCOL["question"], "",
          "**Development.** namespace %s, targets %s, N %s, R=%d (+ hr dev %s x %s x %d), variants: %s."
          % (PROTOCOL["development"]["namespace"], PROTOCOL["development"]["targets"],
             PROTOCOL["development"]["sizes"], PROTOCOL["development"]["n_reps"],
             PROTOCOL["development"]["hr_targets"], PROTOCOL["development"]["hr_sizes"],
             PROTOCOL["development"]["hr_reps"], ", ".join(PROTOCOL["development"]["variants"])), "",
          "**Selection rule (fixed before the development run).** Primary: %s. Gates: %s. Tie-break: %s. %s."
          % (PROTOCOL["selection_rule"]["primary"], "; ".join(PROTOCOL["selection_rule"]["gates"]),
             PROTOCOL["selection_rule"]["tie_break"], PROTOCOL["selection_rule"]["ablation_to_confirm"]), "",
          "**Confirmation.** 15 Marron-Wand targets x N in {200, 1000, 5000} x R=30 on the confirmation "
          "namespace (datasets identical to Stage C, comparator rows reused, fold stream = new method index), "
          "plus the hr family (4 r x 2 N x 20) on bf_main with comparator rows reused.", "",
          "**Analysis.** " + json.dumps(PROTOCOL["analysis"]), ""]
    (HERE / "protocol_improved.md").write_text("\n".join(md), encoding="utf-8")
    info = {"locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "protocol_yaml_sha256": hashlib.sha256(y.read_bytes()).hexdigest(),
            "code_sha256_at_lock": code_hash()}
    lock.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
