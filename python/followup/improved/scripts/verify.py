"""Completion checks for the improved-AKDE study.

    python followup/improved/scripts/verify.py

Writes followup/results/improved/verification.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1]))
import followup.improved  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from followup.improved import PY_ROOT  # noqa: E402
from followup.improved.runner import RESULTS  # noqa: E402

PAPER = PY_ROOT.parents[1] / "final_package" / "paper"
BF = PY_ROOT.parents[1] / "final_package" / "experiments" / "bandwidth_flexibility" / "results" / "main" / "all_rows.csv"
checks = {}


def rec(name, ok, **info):
    checks[name] = {"ok": bool(ok), **info}
    print("%-44s %s  %s" % (name, "PASS" if ok else "FAIL", json.dumps(info, default=str)[:170]))


def main():
    fin = json.load(open(HERE / "finalist_improved.json"))
    n_methods = 1 + (1 if fin.get("ablation") else 0)
    for stage, n in (("dev", 528), ("confirm", 1350 * n_methods), ("hr", 160 * n_methods)):
        d = pd.read_csv(RESULTS / stage / "all_rows.csv", low_memory=False)
        err = d["error"].fillna("").astype(str)
        rec("%s_counts" % stage, len(d) == n and (err == "").sum() == n and d.is_valid_density.astype(bool).all(),
            rows=len(d), expected=n, errors=int((err != "").sum()), max_mass_error=float(d.mass_error.max()),
            code_hashes=int(d.code_sha256.nunique()), protocol_hashes=int(d.protocol_sha256.nunique()))
        if stage != "dev":
            rec("%s_stage2_at_tol" % stage, (d.stage2_kkt.astype(float) < 1e-7).all(),
                max_stage2_kkt=float(d.stage2_kkt.astype(float).max()))
    # hr datasets identical to the bandwidth-flexibility rows (dataset sha256 stored there)
    if BF.exists():
        h = pd.read_csv(RESULTS / "hr" / "all_rows.csv", low_memory=False)
        b = pd.read_csv(BF, low_memory=False)
        m = h.merge(b[["target", "N", "rep", "dataset_sha256"]].drop_duplicates(), on=["target", "N", "rep"], suffixes=("", "_bf"))
        rec("hr_datasets_identical_to_bf_rows", (m.dataset_sha256 == m.dataset_sha256_bf).all(), n=len(m))
    # confirmation datasets: the Stage-C rows store no hash; check by regenerating one dataset per cell
    from akde.densities import get_target
    from followup.akde_followup.seeds import data_rng
    c = pd.read_csv(RESULTS / "confirm" / "all_rows.csv", low_memory=False)
    import hashlib
    bad = 0
    for (t, N), s in c.groupby(["target", "N"]):
        X = get_target(t).sample(int(N), data_rng("confirmation", t, int(N), 0))
        hsh = hashlib.sha256(np.ascontiguousarray(X, np.float64).tobytes()).hexdigest()
        if not (s[s.rep == 0].dataset_sha256 == hsh).all():
            bad += 1
    rec("confirm_datasets_regenerate_from_seeds", bad == 0, cells_checked=int(c.groupby(["target", "N"]).ngroups), mismatches=bad)
    # tables recomputed
    po = pd.read_csv(HERE / "tables" / "improved_benchmark_mean_sd.csv")
    t = pd.read_csv(RESULTS / "analysis" / "tidy_confirm.csv", low_memory=False)
    worst = 0.0
    for s in po.itertuples():
        v = t[(t.method == s.method) & (t.N == s.N) & t.valid].ise.to_numpy(float)
        m = v.sum() / v.size
        sd = np.sqrt(((v - m) ** 2).sum() / (v.size - 1))
        worst = max(worst, abs(m - s.mean) / m, abs(sd - s.pooled_sd) / sd)
    rec("table_mean_sd_recomputed", worst < 1e-12, worst_rel_dev=worst)
    # macros and manuscript
    defs = set(re.findall(r"\\newcommand\{\\(\w+)\}", (HERE / "tables" / "improved_numbers.tex").read_text(encoding="utf-8")))
    sec = (HERE / "manuscript" / "section_improved.tex").read_text(encoding="utf-8")
    used = set(re.findall(r"\\(ir[A-Za-z]+)", sec))
    rec("section_macros_defined", not (used - defs), n_used=len(used), undefined=sorted(used - defs))
    tex = (PAPER / "main (2).tex").read_text(encoding="utf-8")
    integrated = "\\label{sec:repaired}" in tex
    if integrated:
        body = re.sub(r"(?m)^\s*%.*$", "", tex)
        refs = re.findall(r"\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]*)\}", body)
        missing = [r for r in refs if not (PAPER / r).exists() and not (PAPER / (r + ".tex")).exists()]
        badp = [r for r in refs if "/" in r or "\\" in r]
        allmac = defs | set(re.findall(r"\\newcommand\{\\(\w+)\}", (PAPER / "bandwidth_flexibility_numbers.tex").read_text(encoding="utf-8")))
        used_all = set(re.findall(r"\\((?:ir|bf)[A-Z][A-Za-z]*)", body))
        labels = set(re.findall(r"\\label\{([^}]*)\}", body)); rr = set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", body))
        begins = re.findall(r"\\begin\{(\w+\*?)\}", body); ends = re.findall(r"\\end\{(\w+\*?)\}", body)
        bib = (PAPER / "references.tex").read_text(encoding="utf-8")
        keys = set(re.findall(r"\\bibitem\{([^}]*)\}", bib))
        cites = set(k.strip() for cc in re.findall(r"\\cite\{([^}]*)\}", body) for k in cc.split(","))
        rec("manuscript_assets_and_macros", not missing and not badp and not (used_all - allmac),
            n_refs=len(refs), missing=missing, bad_paths=badp, undefined_macros=sorted(used_all - allmac))
        rec("manuscript_structure", sorted(begins) == sorted(ends) and not (rr - labels) and not (cites - keys)
            and body.count("{") == body.count("}"), dangling_refs=sorted(rr - labels), undefined_cites=sorted(cites - keys))
        diffs = [p.name for p in (HERE / "tables").glob("improved_*.tex") if (PAPER / p.name).read_bytes() != p.read_bytes()]
        diffs += [p.name for p in (HERE / "figures").glob("improved_*.pdf") if (PAPER / p.name).read_bytes() != p.read_bytes()]
        rec("paper_assets_match_generated", not diffs, differing=diffs)
    else:
        rec("manuscript_integrated", False, note="section not yet in main (2).tex")
    rec("plot_data_present", len(list((HERE / "figures" / "data").glob("*_curves.npz"))) >= 5)
    ok = all(v["ok"] for v in checks.values())
    (RESULTS / "verification.json").write_text(json.dumps({"all_passed": ok, "checks": checks}, indent=2, default=str), encoding="utf-8")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
