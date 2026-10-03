"""One-off factual corrections to manuscript/section_improved.tex after checking
every sentence against the saved rows (recorded in experiment_log.md)."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "manuscript" / "section_improved.tex"
s = p.read_text(encoding="utf-8")
reps = [
    ("exists because there is no $\\lambda$; the remaining cases are datasets on\nwhich the validated base bandwidth is small.",
     "exists because there is no $\\lambda$; all seven remaining cases occur at\n$N=200$ and none of them selected a base bandwidth at the edge of its grid."),
    ("ablation of that section, which already used a validated bandwidth, it is\nwithin a factor of two in every cell and better in three of eight;",
     "ablation of that section, which already used a validated bandwidth, it is\nwithin a factor of $2.6$ in every cell and better in three of eight;"),
    ("AKDE-R improves on AKDE-M by paired factors of\n$\\irHrRVsMROneNFiveH$ to $\\irHrRVsMREightNFiveH$ at $N=500$ and\n$\\irHrRVsMROneNLarge$ to $\\irHrRVsMREightNLarge$ at $N=5000$",
     "AKDE-R improves on AKDE-M by paired factors of\n$\\irHrRVsMROneNFiveH$ ($r=1$) to $\\irHrRVsMREightNFiveH$ ($r=8$) at $N=500$ and\n$\\irHrRVsMROneNLarge$ to $\\irHrRVsMREightNLarge$ at $N=5000$"),
    ("not help accuracy; the merge halves the component count at a\nmeasurable accuracy cost",
     "not help accuracy; the merge cuts the component count by half or more at a\nmeasurable accuracy cost"),
    ("Development and confirmation used\ndisjoint seeds and the selection rule was fixed in advance, but the repairs\nthemselves were chosen with the earlier results in hand, so this is a\nconfirmed follow-up, not an independent replication.",
     "Two limitations are specific to the repaired estimator. The validated base\nbandwidth sits at the top of its nine-point grid ($4h_{\\mathrm{Silverman}}$)\nin $57\\%$ of the confirmation fits, so the anchor grid is too narrow at the\nwide end and a wider grid may improve it further; it was not widened because\nthe protocol was locked. And development and confirmation used disjoint\nseeds with a selection rule fixed in advance, but the repairs themselves were\nchosen with the earlier results in hand, so this is a confirmed follow-up,\nnot an independent replication."),
]
n = 0
for a, b in reps:
    if a in s:
        s = s.replace(a, b); n += 1
    else:
        print("NOT FOUND:", a[:60])
p.write_text(s, encoding="utf-8")
print("applied %d/%d" % (n, len(reps)))
