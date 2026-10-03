"""One-off: normalise macro names in manuscript/section_improved.tex to the
names make_tables.py generates (prefix \\ir, digit-free, method tags)."""
import re
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "manuscript" / "section_improved.tex"
s = p.read_text(encoding="utf-8")
exact = {
    "\\bfDevIFourSimplexLTwo": "\\irDevC",
    "\\bfDevISixGramExtOneseFiveF": "\\irDevISixGramExtOneseFivef",
    "\\bfDevLoR": "\\irDevLoR", "\\bfDevHiR": "\\irDevHiR", "\\bfDevR": "\\irDevR",
}
for a, b in sorted(exact.items(), key=lambda kv: -len(kv[0])):
    s = s.replace(a, b)
s = s.replace("\\bfir", "\\ir").replace("\\bfDev", "\\irDev")
s = s.replace("ratios of $\\irDevIOneGramFista$--$\\irDevC$ against AKDE-M",
              "ratios of $\\irDevIOneGramFista$, $\\irDevITwoIdentityFista$ and $\\irDevC$ against AKDE-M")
p.write_text(s, encoding="utf-8")
print(sorted(set(re.findall(r"\\(ir[A-Za-z]+)", s))))
print("remaining bf-prefixed:", sorted(set(re.findall(r"\\(bf[A-Za-z]+)", s))))
