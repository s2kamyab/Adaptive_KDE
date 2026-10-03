"""Run one stage of the improved-AKDE study.

    python followup/improved/scripts/run_stage.py --stage dev --jobs 12
    python followup/improved/scripts/run_stage.py --stage confirm --jobs 12
    python followup/improved/scripts/run_stage.py --stage hr --jobs 12

'confirm' and 'hr' read the locked finalist from followup/improved/finalist_improved.json
(written by select_finalist.py).  Resumable per fit; --force recomputes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1]))
import followup.improved  # noqa: E402,F401
import pandas as pd  # noqa: E402
from followup.improved.estimator import VARIANTS  # noqa: E402
from followup.improved.runner import RESULTS, collect, run_units, code_hash  # noqa: E402


def protocol_hash():
    lock = json.load(open(HERE / "protocol_improved_lock.json", encoding="utf-8"))
    cur = hashlib.sha256((HERE / "protocol_improved.yaml").read_bytes()).hexdigest()
    if cur != lock["protocol_yaml_sha256"]:
        raise SystemExit("protocol_improved.yaml changed since lock")
    return cur


def units_for(stage):
    P = json.load(open(HERE / "protocol_improved.json", encoding="utf-8"))
    if stage == "dev":
        d = P["development"]
        methods = d["variants"]
        units = []
        for m in methods:
            for t in d["targets"]:
                for N in d["sizes"]:
                    for r in range(d["n_reps"]):
                        units.append((d["namespace"], t, N, r, m, 0))
            for t in d["hr_targets"]:
                for N in d["hr_sizes"]:
                    for r in range(d["hr_reps"]):
                        units.append((d["hr_namespace"], t, N, r, m, 0))
        return units
    fin = json.load(open(HERE / "finalist_improved.json", encoding="utf-8"))
    methods = [(fin["finalist"], 6)] + ([(fin["ablation"], 7)] if fin.get("ablation") else [])
    c = P["confirmation"]
    units = []
    if stage == "confirm":
        for m, mi in methods:
            for t in c["targets"]:
                for N in c["sizes"]:
                    for r in range(c["n_reps"]):
                        units.append((c["namespace"], t, N, r, m, mi))
    elif stage == "hr":
        for m, mi in methods:
            for t in c["hr_targets"]:
                for N in c["hr_sizes"]:
                    for r in range(c["hr_reps"]):
                        units.append((c["hr_namespace"], t, N, r, m, 0))
    else:
        raise SystemExit("unknown stage")
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("dev", "confirm", "hr"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ph = protocol_hash()
    units = units_for(a.stage)
    units.sort(key=lambda u: (-u[2], u[1], u[3]))
    print("stage %s: %d fits, jobs %d" % (a.stage, len(units), a.jobs), flush=True)
    t0 = perf_counter()
    res = run_units(a.stage, units, protocol_hash=ph, n_jobs=a.jobs, force=a.force)
    el = perf_counter() - t0
    df = collect(a.stage)
    err = df["error"].fillna("").astype(str)
    summ = {"stage": a.stage, "n_fits": len(units), "n_rows": int(len(df)), "n_errors": int((err != "").sum()),
            "status_counts": pd.Series([r["status"] for r in res]).value_counts().to_dict(),
            "elapsed_s": el, "jobs": a.jobs, "protocol_sha256": ph, "code_sha256": code_hash()}
    (RESULTS / a.stage / "run_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    if summ["n_errors"]:
        print(df.loc[err != "", ["target", "N", "rep", "method", "error"]].to_string())
    ok = df[err == ""]
    print(ok.pivot_table(index="method", columns="N", values="ise", aggfunc="median")
          .to_string(float_format=lambda v: "%.4g" % v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
