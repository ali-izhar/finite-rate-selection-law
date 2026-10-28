"""Check the paper's tables and compiled PDF against saved results.

Build, then check, from the repository root:
    (cd paper && latexmk -pdf -outdir=build main.tex)
    python -m checks.paper_consistency [--paper paper/build] [--output FILE]
The source is paper/main.tex; the .aux, .log and .pdf come from --paper.
This checks saved results and the compiled document; it runs no simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr

# Words of the earlier protocol that the paper no longer uses (case-insensitive).
REMOVED = ("reheat", "filtered", "second selection", "dwell", "field grid", "834", "0.199", "0.201",
           "0.2101", "1.063", "reflected", r"\tau_{\mathrm{ref", "three estimators", "tab:schedule",
           "tab:counts")
ROWS = ((1024, 1), (2048, 1), (4096, 1), (2048, 2), (1024, 3), (2048, 3), (4096, 3), (2048, 4))


def check(root: Path, paper: Path) -> dict:
    sys.path.insert(0, str(root))
    from src.simple import design as SD
    from src.simple.analyse import counts_for_estimator
    from src.simple.estimate import estimate

    tex = (root / "paper/main.tex").read_text()
    flat = " ".join(tex.split())          # for phrases that a line break may split
    passed = []

    def require(ok, label):
        if not ok:
            raise AssertionError(label)
        passed.append(label)

    def read(path):
        return json.loads((root / path).read_text())

    def table(label):
        start = tex.rindex(r"\begin{table}", 0, tex.index(r"\label{" + label + "}"))
        return tex[start:tex.index(r"\end{table}", start)]

    def body_rows(label):
        """Cells of each data row (between \\midrule and \\bottomrule)."""
        t = table(label)
        body = t[t.index(r"\midrule") + len(r"\midrule"):t.index(r"\bottomrule")]
        return [[c.strip() for c in line.rstrip("\\ ").split("&")]
                for line in body.strip().splitlines() if "&" in line]

    def nums(cell):
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", cell.replace("{,}", ""))]

    # Table 1: the linear ramps.
    linear = read("results/experiments/cw_linear_sweep.json")
    rows = body_rows("tab:linear")
    require(len(rows) == 3, "Table 1 has three sizes")
    for cells in rows:
        n = int(cells[0])
        got = [x for c in cells[1:] for x in nums(c)]
        expected = []
        for gamma in (0.004, 0.001, 0.00025):
            z = next(z for z in linear if z["N"] == n and z["gamma"] == gamma)
            expected.extend([round(z["c_exact"], 5), round(100 * (z["ratio"] - 1), 2)])
        require(np.allclose(got, expected, rtol=0, atol=1e-8), f"Table 1 row N={n}")
    row = next(z for z in linear if z["N"] == 2048 and z["gamma"] == 0.001)
    amplitude = row["G"] * np.sqrt(row["N"]) * (np.pi / row["gamma"]) ** 0.25
    require(abs(ndtr(amplitude) - row["q_KN"]) < 1e-12 and abs(amplitude - 0.208) < 0.0005
            and "$a=0.208$" in tex, "Figure 1b Gaussian agrees with the saved linear-ramp formula")

    # Table 3: the Hopfield test of the written protocol.
    pred = {(r["N"], r["p"]): r for r in read("results/simple/predictions.json")["rows"]}
    analysis = read("results/simple/analysis.json")
    kinds = {"exact (gauge identity)": "exact chain", "exact two-block chain": "exact two-block",
             "A_p * c_CW": r"$A_{p}c_{\mathrm{CW}}$"}
    rows = body_rows("tab:transfer")
    require([(int(c[1]), int(c[0])) for c in rows] == list(ROWS), "Table 3 lists the eight rows in order")
    for cells in rows:
        p, n = int(cells[0]), int(cells[1])
        r, z = read(f"results/simple/row_N{n}_p{p}.json"), pred[(n, p)]
        e = estimate(counts_for_estimator(r), z["L_eq"], level=0.95)
        require(cells[2] == kinds[z["prediction"]].replace("{p}", str(p)), f"Table 3 prediction kind N={n}, p={p}")
        expected = [round(z["c_pred"], 4), round(e["c"], 4), round(e["lo"], 4), round(e["hi"], 4),
                    round(e["c"] / z["c_pred"], 3)]
        got = nums(cells[3]) + nums(cells[4]) + nums(cells[5])
        require(np.allclose(got, expected, rtol=0, atol=1e-9), f"Table 3 row N={n}, p={p}")
        runs, won = (int(nums(x)[0]) for x in cells[6:8])
        k = r["counts"]
        require((runs, won) == (int(np.sum(k["runs"])), int(np.sum(k["pattern1"])))
                and int(np.sum(k["tie"])) == 0 and r["complete"] and r["torch_seed"] == SD.torch_seed(n, p),
                f"Table 3 runs and runs won N={n}, p={p}")
        require(abs(analysis["rows"][f"N{n}_p{p}"]["estimate"]["c"] - e["c"]) < 1e-15,
                f"Table 3 N={n}, p={p} equals results/simple/analysis.json")
    require(analysis["gates"]["all_passed"] and analysis["T"]["reading"] == "(a)",
            "all gates passed; the mechanical reading at bootstrap seed 7 is (a)")
    require("Reading (c) applies" in flat and "We report reading (c), the less favorable one" in flat
            and "ends at $1.0005$" in flat and "$139$ of the $200$ bootstrap seeds $0$--$199$" in flat,
            "the paper reports reading (c) and that reading (a) holds at seed 7 but fails at 139 of 200 seeds")

    # Table 4: the post-hoc per-set analysis.
    per_set = read("results/posthoc/per_set.json")["rows"]
    rows = body_rows("tab:sets")
    require([(int(c[1]), int(c[0])) for c in rows] == [(1024, 3), (2048, 3), (4096, 3), (2048, 4)],
            "Table 4 lists the four A_p rows in order")
    for cells in rows:
        p, n = int(cells[0]), int(cells[1])
        z = per_set[f"N{n}_p{p}"]
        lo, hi = nums(cells[2])
        mant, _, expo = nums(cells[3])
        pval = z["homogeneity"]["p_value"]
        ok = (round(z["win_share"]["min"], 3) == lo and round(z["win_share"]["max"], 3) == hi
              and round(pval / 10.0 ** expo) == mant
              and round(z["spearman_c_vs_win_share"], 2) == nums(cells[4])[0]
              and round(100 * z["ratio_per_set"]["sd_beyond_binomial"], 1) == nums(cells[5])[0]
              and round(z["equal_weight"]["ratio"], 3) == nums(cells[6])[0])
        require(ok, f"Table 4 row N={n}, p={p}")

    paired = read("results/posthoc/per_set.json")["paired_N2048_p3_minus_p4"]
    require(round(paired["difference"], 3) == 0.055 and "$0.055$ lower" in flat
            and (round(paired["lo_seed7"], 3), round(paired["hi_seed7"], 3)) == (0.008, 0.099)
            and "$[0.008,0.099]$" in flat and paired["fraction_excluding_0"] == 1.0,
            "App. C paired difference of the p=3 and p=4 ratios at N=2048")

    # Printed design constants and seeds.
    for printed, label in ((r"$M=225N$", "updates per run"),
                           (r"the ramp lasts $225$ sweeps", "ramp length"),
                           (r"$NG=1.2544$", "N G"),
                           (r"$G=0.0006125\times2048/N$", "G"),
                           (r"\texttt{default\_rng}($[20261027,N,s]$)", "pattern-set seed rule"),
                           (r"$10{,}000$ cluster-bootstrap resamples", "bootstrap size"),
                           (r"(seed $7$)", "bootstrap seed")):
        require(printed in tex, f"methods print the {label}")
    require(SD.n_updates(2048) == 225 * 2048 and SD.NG == 1.2544 and SD.SEED == 20261027
            and (SD.N_BOOT, SD.BOOT_SEED, SD.GATE_LEVEL) == (10_000, 7, 0.9875),
            "printed constants match src/simple/design.py")

    # Exact-chain numbers printed in Sec. 3 and App. A.
    from scipy.special import logsumexp
    from src.barrier import log_mfpt, log_pi_closed_form
    k_mode = int(np.argmax(log_pi_closed_form(2048, 1.1)[:1024]))
    switch = log_mfpt(2048, 1.1, 0.0, k_mode, 2048 - k_mode) / np.log(10)
    require(abs(10**switch / 5e7 - 1) < 0.05 and r"$5\times10^{7}$" in tex,
            "Section 3 mode-switch time at beta=1.1 is about 5e7 sweeps")
    require(r"$10^{-36}$" in tex, "zero-magnetization mass bound is printed")
    for n in (512, 1024, 2048, 4096, 16384):
        lp = log_pi_closed_form(n, 1.6)
        require((lp[n // 2] - logsumexp(lp)) / np.log(10) < -36,
                f"zero-magnetization mass below 1e-36 at beta_f=1.6, N={n}")

    # Structure: the earlier protocol is gone; one sampler sentence on permutation sweeps remains.
    low = tex.lower()
    for word in REMOVED:
        require(word.lower() not in low, f"removed element absent: {word}")
    require(low.count("permutation") == 1 and low.index("permutation") > low.index(r"\label{app:exact}"),
            "permutation sweeps appear once, in the appendix")
    sections = re.findall(r"\\section\{([^}]*)\}", tex)
    require(sections[:5] == ["Introduction", r"Model, sampler and the fraction \texorpdfstring{$c$",
                             "The selection law", "Competition among several modes",
                             "Limitations and related work"], "five main sections in order")
    require(len(sections) == 8, "three appendices")

    sw = read("results/experiments/switching.json")
    late = max(max((z["window3"], z["beta1.1"]), key=lambda d: d["beta_split"])["switched"] for z in sw)
    slow = next(z for z in sw if z["N"] == 1024 and z["gamma"] == 0.00025)["window3"]["switched"]
    require(late < 0.01 and slow > 0.05 and "rarely switch once the modes have separated" in flat
            and "$12\\%$ of runs still change the sign of $m$" in flat and round(100 * slow) == 12,
            "Section 3 switching statements agree with results/experiments/switching.json")

    aux = (paper / "main.aux").read_text()
    require(bool(re.search(r"\\newlabel\{main-text-end\}\{\{5\}\{5\}", aux)), "main text ends on page five")
    require(bool(re.search(r"\\newlabel\{app:exact\}\{\{A\}\{6\}", aux)), "appendix starts on page six")
    log = (paper / "main.log").read_text()
    require(not re.search(r"Overfull|undefined|Citation .*Warning|Rerun to get|Fatal error", log),
            "no overflow, unresolved references or incomplete build")
    require("destination with the same identifier" not in log, "no duplicate PDF link destinations")
    pdf = paper / "main.pdf"
    proc = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, text=True, capture_output=True)
    require(not proc.stderr.strip(), "PDF text extraction has no warnings")
    pages = [p for p in proc.stdout.split("\f") if p.strip()]
    require(len(pages) <= 11, "at most eleven total pages")
    require(all(len(p.strip()) > 200 for p in pages), "no empty content pages")
    require(bool(re.search(r"^References\s*$", pages[5], re.M)), "references start on page six")
    require("??" not in proc.stdout, "no unresolved-reference markers in PDF")
    authors = re.findall(r"\\Name\{([^}]*)\}", tex)
    require(len(authors) == 3 and all(name in pages[0] for name in authors),
            "all authors in paper/main.tex present on first page")
    require([pages[0].index(name) for name in authors] == sorted(pages[0].index(name) for name in authors),
            "author order matches paper/main.tex")
    proc = subprocess.run(["pdffonts", str(pdf)], check=True, text=True, capture_output=True)
    require(not proc.stderr.strip(), "PDF font inspection has no warnings")
    require("Type 3" not in proc.stdout, "no Type 3 fonts")
    require(all(line.split()[-5] == "yes" for line in proc.stdout.splitlines()[2:] if line.strip()),
            "all PDF fonts embedded")
    return {"checks_passed": len(passed), "checks": passed,
            "pages": {"main": 5, "references_and_appendix": len(pages) - 5, "total": len(pages)},
            "source_sha256": hashlib.sha256(tex.encode()).hexdigest(),
            "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "scope": "Saved-result transcription and compiled-document checks; no GPU rerun"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--paper", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.root.resolve(), args.paper or args.root / "paper/build")
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(f"{result['checks_passed']} checks passed, exit 0", file=sys.stderr)
        print(text, end="")
