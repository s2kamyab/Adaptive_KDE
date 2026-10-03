"""Integrate the repaired-estimator section into final_package/paper/main (2).tex
and copy the generated tables, macros and figures next to it (flat paths).

    python followup/improved/scripts/integrate_paper.py

Idempotent: refuses to insert the section twice; always refreshes the copied
assets.  A pre-edit snapshot is kept in followup/improved/snapshot/.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PAPER = HERE.parents[3] / "final_package" / "paper"
TEX = PAPER / "main (2).tex"
SEC = HERE / "manuscript" / "section_improved.tex"


def main():
    t = TEX.read_text(encoding="utf-8")
    snap = HERE / "snapshot"
    snap.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if "\\label{sec:repaired}" not in t:
        (snap / ("main (2).tex.before-repaired-%s" % stamp)).write_text(t, encoding="utf-8")
        s = SEC.read_text(encoding="utf-8")

        def rep(old, new):
            nonlocal t
            if t.count(old) != 1:
                raise SystemExit("expected exactly one occurrence: %r" % old[:70])
            t = t.replace(old, new)

        rep("\\input{bandwidth_flexibility_numbers.tex}",
            "\\input{bandwidth_flexibility_numbers.tex}\n\\input{improved_numbers.tex}  % generated numbers for Section~\\ref{sec:repaired}")
        # abstract: after the bandwidth sentences
        rep("density-valid SPADES behaves the same way on the same dictionary. These findings",
            "density-valid SPADES behaves the same way on the same dictionary. Acting on these diagnoses, a "
            "repaired estimator that minimises the integrated-square loss directly over the simplex on a "
            "dictionary whose scales are anchored to a validated bandwidth, with no selection stage and one "
            "tuned hyperparameter, reduces the paired geometric-mean error to 0.26, 0.40 and 0.51 of AKDE-M's "
            "at $N=200$, 1000 and 5000 on a locked confirmation, beats the SPADES adaptation on the original "
            "dictionary, and closes the gap to the Sheather--Jones estimator to about 30\\% while keeping a few "
            "dozen atoms and microsecond queries. These findings")
        rep("Multiscale selection is not what makes the estimator useful; compactness is\n(Section~\\ref{sec:bandwidth}).\n\\end{enumerate}",
            "Multiscale selection is not what makes the estimator useful; compactness is\n(Section~\\ref{sec:bandwidth}).\n"
            "\\item \\textbf{A repaired estimator.} Implementing the resulting repairs (metric-consistent\n"
            "weighting, a converged or selection-free solve, validated dictionary scales) as switches of one\n"
            "estimator, selecting a configuration on development seeds by a rule fixed in advance and confirming\n"
            "it on the confirmation seeds yields AKDE-R: paired geometric-mean ISE ratios of $0.26$, $0.40$ and\n"
            "$0.51$ against AKDE-M, $0.32$--$0.58$ against SPADES-simplex, and $1.3$ against the Sheather--Jones\n"
            "estimator, with one tuned hyperparameter, no first stage and no catastrophic-replicate mechanism\n"
            "(Section~\\ref{sec:repaired}).\n\\end{enumerate}")
        rep("\\section{Analysis of estimation error}\n\\label{sec:mechanism}",
            s.rstrip() + "\n\n\\section{Analysis of estimation error}\n\\label{sec:mechanism}")
        rep("""The result bounds what the
dictionary design can be credited with; it does not identify
which of the penalty grid, the selection rule or the solver
is responsible, and it does not transfer to other dictionary
constructions without a fresh confirmation.""",
            """The result bounds what the
dictionary design can be credited with. The repaired estimator of
Section~\\ref{sec:repaired} shows that anchoring the scales to a validated
bandwidth recovers most of the loss, but that section's repairs were chosen
with these results in hand and confirmed on seeds that had already served
one confirmation, so they are a confirmed follow-up rather than an
independent replication.""")
        rep("""The estimator's demonstrated strength is therefore
a compact, valid mixture that can be fitted from the
moment criterion and queried cheaply, not a
multiscale representation that captures sharp and
broad features better than its alternatives.""",
            """The estimator's demonstrated strength is therefore
a compact, valid mixture that can be fitted from the
moment criterion and queried cheaply, not a
multiscale representation that captures sharp and
broad features better than its alternatives.

Acting on these diagnoses produces a markedly better
estimator (Section~\\ref{sec:repaired}): minimising the
integrated-square loss directly over the simplex, with
the dictionary scales chosen by validation, removes the
first stage, two of three hyperparameters and the
catastrophic-replicate mechanism, and cuts the paired
error to $0.26$--$0.51$ of AKDE-M's. The repaired
estimator remains about $30\\%$ behind a Sheather--Jones
kernel estimator and further behind the adaptive and
mixture comparators, so compactness, not accuracy, is
still its case; and the gain comes from the dictionary
design, which any sparse-mixture method can adopt, not
from the moment criterion.""")
        TEX.write_text(t, encoding="utf-8")
        print("section inserted; snapshot written")
    else:
        print("section already present; refreshing assets only")
    for p in (HERE / "tables").glob("improved_*.tex"):
        shutil.copy2(p, PAPER / p.name)
    for p in (HERE / "figures").glob("improved_*.pdf"):
        shutil.copy2(p, PAPER / p.name)
    print("assets copied to", PAPER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
