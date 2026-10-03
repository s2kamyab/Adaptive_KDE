"""Structural validation of the LaTeX sources when no TeX toolchain is present.

This does NOT claim a compiled PDF. It checks what can be checked statically:
balanced environments and braces, every \\input target present, every
\\includegraphics target present, every \\cite key defined, every \\ref target
labelled, no duplicate labels, and no obvious placeholder text.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

def _package_root():
    """Locate final_package/ whether we are in the repo or inside the package.

    In the package, code/ and manifests/ are siblings; in the repository the
    package sits at <project>/final_package. Both are checked, so no script
    contains a machine-specific path.
    """
    here = Path(__file__).resolve()
    sib = here.parents[2].parent            # .../code/..  -> package root
    if (sib / "manifests").exists() or (sib / "paper" / "main.tex").exists():
        return sib
    for up in here.parents:
        cand = up / "final_package"
        if cand.exists():
            return cand
    return sib


PAPER = _package_root() / "paper"
import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parents[2]))
import followup  # noqa: E402,F401
from followup.paths import DIAG_ROOT   # noqa: E402
OUT = DIAG_ROOT

PROBLEMS = []


def note(kind, msg):
    PROBLEMS.append({"kind": kind, "message": msg})
    print("  [%s] %s" % (kind, msg))


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def check(path: Path, bib_keys):
    print("\n== %s ==" % path.name)
    raw = path.read_text(encoding="utf-8")
    s = strip_comments(raw)

    # environments
    begins = re.findall(r"\\begin\{([^}]*)\}", s)
    ends = re.findall(r"\\end\{([^}]*)\}", s)
    from collections import Counter
    cb, ce = Counter(begins), Counter(ends)
    for env in set(cb) | set(ce):
        if cb[env] != ce[env]:
            note("ERROR", "%s: environment %r begin=%d end=%d"
                 % (path.name, env, cb[env], ce[env]))
    # braces
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{" and (i == 0 or s[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or s[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                note("ERROR", "%s: unbalanced closing brace near offset %d" % (path.name, i))
                depth = 0
    if depth != 0:
        note("ERROR", "%s: %d unclosed brace(s)" % (path.name, depth))

    # inputs
    for tgt in re.findall(r"\\input\{([^}]*)\}", s):
        t = tgt if tgt.endswith(".tex") else tgt + ".tex"
        if not (path.parent / t).exists():
            note("ERROR", "%s: \\input target missing: %s" % (path.name, t))

    # graphics
    gp = re.findall(r"\\graphicspath\{\{([^}]*)\}\}", s)
    roots = [path.parent / g.strip("/") for g in gp] or [path.parent]
    for tgt in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", s):
        cands = []
        for r in roots + [path.parent]:
            cands += [r / tgt, r / (tgt + ".pdf"), r / (tgt + ".png")]
        if not any(c.exists() for c in cands):
            note("ERROR", "%s: figure not found: %s" % (path.name, tgt))

    # labels and refs
    labels = re.findall(r"\\label\{([^}]*)\}", s)
    dup = [l for l in set(labels) if labels.count(l) > 1]
    for d in dup:
        note("ERROR", "%s: duplicate label %r" % (path.name, d))
    refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]*)\}", s))
    missing = sorted(refs - set(labels))
    for m in missing:
        note("ERROR", "%s: \\ref to undefined label %r" % (path.name, m))

    # citations
    used = set()
    for c in re.findall(r"\\cite[a-z]*\{([^}]*)\}", s):
        used.update(k.strip() for k in c.split(","))
    for k in sorted(used - bib_keys):
        note("ERROR", "%s: undefined citation %r" % (path.name, k))

    # placeholders
    for pat in (r"\bTODO\b", r"\bTBD\b", r"\bXXX\b", r"\?\?\?", r"\\todo"):
        for m in re.finditer(pat, s):
            note("ERROR", "%s: placeholder %r at offset %d"
                 % (path.name, m.group(0), m.start()))
    return {"labels": len(labels), "refs": len(refs), "citations": len(used),
            "environments": len(begins), "bytes": len(raw)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    refs = PAPER / "references.tex"
    bib_keys = set(re.findall(r"\\bibitem\{([^}]*)\}",
                              refs.read_text(encoding="utf-8"))) if refs.exists() else set()
    print("bibliography keys defined: %d" % len(bib_keys))
    stats = {}
    for name in ("main.tex", "supplement.tex"):
        p = PAPER / name
        if not p.exists():
            note("ERROR", "missing %s" % name)
            continue
        stats[name] = check(p, bib_keys)

    n_err = sum(1 for p in PROBLEMS if p["kind"] == "ERROR")
    print("\n%d structural problem(s)" % n_err)
    print("\nNOTE: no TeX toolchain is installed on this machine "
          "(pdflatex/latexmk/xelatex/tectonic all absent), so NO PDF has been "
          "produced and none is claimed. Build with:")
    print("    cd final_package/paper && latexmk -pdf main.tex && "
          "latexmk -pdf supplement.tex")
    with open(OUT / "latex_structure_check.json", "w", encoding="utf-8") as fh:
        json.dump({"problems": PROBLEMS, "n_errors": n_err, "stats": stats,
                   "bib_keys": len(bib_keys),
                   "toolchain_available": False,
                   "pdf_produced": False,
                   "build_command": "latexmk -pdf main.tex; latexmk -pdf supplement.tex"},
                  fh, indent=2)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
