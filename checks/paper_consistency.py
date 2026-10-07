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


def check(root: Path, paper: Path) -> dict:
    sys.path.insert(0, str(root))
    from src.hopfield import protocol

    tex = (root / "paper/main.tex").read_text()
    passed = []

    def require(ok, label):
        if not ok:
            raise AssertionError(label)
        passed.append(label)

    def read(path):
        return json.loads((root / path).read_text())

    def table(label):
        start = tex.index(r"\label{" + label + "}")
        return tex[start:tex.index(r"\end{table}", start)]

    def numeric_rows(label):
        result = []
        for line in table(label).splitlines():
            if (re.match(r"\s*\$?\s*-?\d", line) and "&" in line
                    and re.match(r"\s*\$?\s*\d", line.split("&")[1])):
                result.append([float(x) for x in re.findall(r"-?(?:\d+(?:\.\d*)?|\.\d+)", line)])
        return result

    counts = numeric_rows("tab:counts")
    points = read("results/hopfield/field_grid_analysis.json")["points"]
    null = next(z for z in read("results/hopfield/sampler_checks.json") if z["name"].startswith("G2 "))["values"]
    require(len(counts) == 9 and len(points) == 8 and counts[4][0] == 0,
            "all eight bias rows and the G=0 row present")
    successes = round(null["q_model"] * null["n"])
    require(counts[4][:5] == [0, null["n"], successes, round(null["q_model"], 4), 0.5],
            "G=0 row matches the saved zero-field check (5560/11220)")
    require(f"$[{null['lower']:.4f},{null['upper']:.4f}]$" in tex,
            "G=0 Wilson interval in the Table 4 caption matches the saved check")
    for row, p in zip(counts[:4] + counts[5:], points):
        row = row[:5]
        expected = [p["field_g"] * 1e4, p["n"], p["successes"],
                    round(p["successes"] / p["n"], 4), round(p["q_eq"]["point"], 4)]
        require(np.allclose(row, expected, rtol=0, atol=1e-8), f"bias counts and references G={p['field_g']}")

    linear = read("results/experiments/cw_linear_sweep.json")
    rows = numeric_rows("tab:linear")
    require(len(rows) == 3, "linear-ramp table has three sizes")
    for row in rows:
        expected = [row[0]]
        for gamma in (0.004, 0.001, 0.00025):
            z = next(z for z in linear if z["N"] == row[0] and z["gamma"] == gamma)
            expected.extend([round(z["c_exact"], 5), round(100 * (z["ratio"] - 1), 2)])
        require(np.allclose(row, expected, rtol=0, atol=1e-8), f"full-formula table N={row[0]:.0f}")

    by_p = read("results/experiments/draw_level_analysis.json")
    by_n = read("results/experiments/size_scan_p3_analysis.json")
    rows = numeric_rows("tab:transfer")
    require(len(rows) == 6, "all six transfer rows present")
    for row in rows:
        p, n = map(int, row[:2])
        z = by_p[str(p)] if n == 2048 else by_n[str(n)]
        expected = [p, n, round(z["c_pred"], 3),
                    round(z["c_pooled"] if n == 2048 else z["c_meas"], 3),
                    *[round(x, 3) for x in z["draw_boot"]]]
        require(np.allclose(row, expected, rtol=0, atol=1e-8), f"transfer point and interval p={p},N={n}")

    schedule = table("tab:schedule")
    beta_line = next(x for x in schedule.splitlines() if x.lstrip().startswith(r"$\beta$ &"))
    sweep_line = next(x for x in schedule.splitlines() if x.lstrip().startswith("sweeps &"))
    beta = [float(x) for x in re.findall(r"\d*\.\d+", beta_line)]
    sweeps = [int(x) for x in re.findall(r"\d+", sweep_line)]
    require(np.allclose(beta, protocol.LADDER, rtol=0, atol=1e-12), "all ten temperatures match protocol")
    require(tuple(sweeps) == tuple(protocol.K_REDESCENT) and sum(sweeps) == 834,
            "all ten dwell counts match protocol and total 834")

    # Methods: compare printed inputs with saved results.
    from src.hopfield.stats import wilson, mover_sum
    for i in range(4):
        a, b = points[i], points[7-i]
        ci = mover_sum(wilson(a["successes"], a["n"]), wilson(b["successes"], b["n"]))
        printed = f"[{ci.lower:.5f},{ci.upper:.5f}]"
        require(printed in tex, f"paired-bias control interval {i+1} matches pooled Wilson combination")
    calibration = read("results/hopfield/overlap_run.json")
    require(len(calibration) == 20 and all(z["n_samples"] == 256*200 and z["n_excluded"] == 0
                                         for z in calibration), "reference-run counts and absence of exclusions match the saved runs")
    require(all(np.isclose(z["se_spec"], z["sd_in_well"] / np.sqrt(256*200/protocol.TAU_EX_SPEC[-1]),
                           rtol=1e-12, atol=0) for z in calibration),
            "calibration SE uses the stated reflected CW relaxation proxy")
    schedule_plot = read("visual/data/schedule.json")
    require(np.allclose(schedule_plot["ladder"], beta, rtol=0, atol=1e-12)
            and np.array_equal(np.ceil(6*np.array(schedule_plot["tau"])).astype(int), sweeps),
            "saved schedule data (docs/relaxation-times.md) reproduces all ten dwells")
    row = next(z for z in linear if z["N"] == 2048 and z["gamma"] == 0.001)
    amplitude = row["G"] * np.sqrt(row["N"]) * (np.pi/row["gamma"])**0.25
    require(abs(ndtr(amplitude)-row["q_KN"]) < 1e-12 and abs(amplitude-0.208)<0.0005,
            "Figure 1b Gaussian agrees with the saved linear-ramp formula")

    # Appendix A exact-chain numbers.
    from scipy.special import logsumexp
    from src.barrier import log_mfpt, log_pi_closed_form

    def log10_switch(beta, n=2048):
        k_mode = int(np.argmax(log_pi_closed_form(n, beta)[:n // 2]))
        return log_mfpt(n, beta, 0.0, k_mode, n - k_mode) / np.log(10)

    for beta, printed in ((1.1, "7.7"), (1.2, "22.9"), (1.6, "137.8")):
        require(abs(log10_switch(beta) - float(printed)) < 0.06 and f"10^{{{printed}}}" in tex,
                f"zero-field mode-center passage time at beta={beta} is 10^{printed} sweeps")
    require(abs(10**log10_switch(1.1) / 5e7 - 1) < 0.05 and r"$5\times10^{7}$" in tex,
            "Section 3 mode-switch time at beta=1.1 is about 5e7 sweeps")
    require(r"$10^{-36}$" in tex, "zero-magnetization mass bound is printed")
    for n in (512, 1024, 2048, 4096, 16384):
        lp = log_pi_closed_form(n, 1.6)
        require((lp[n // 2] - logsumexp(lp)) / np.log(10) < -36,
                f"zero-magnetization mass below 1e-36 at beta_f=1.6, N={n}")
    from src.gap import gap_full
    inverse = [1 / gap_full(n, 1.0) for n in (1024, 2048, 4096)]
    require([f"{t:.1f}" for t in inverse] == ["42.2", "59.0", "82.7"]
            and r"$42.2$, $59.0$, and $82.7$ sweeps" in tex,
            "Appendix A full-chain inverse gaps at beta=1 (N=1024, 2048, 4096)")
    require(np.allclose([inverse[1] / inverse[0], inverse[2] / inverse[1]], np.sqrt(2), rtol=0.02),
            "the beta=1 inverse gap grows as sqrt N")
    require(int(np.ceil(6 * inverse[1])) == 355 and protocol.K_REDESCENT[list(protocol.LADDER).index(1.0)] == 355,
            "the beta=1 dwell is ceil(6 x 59.0) = 355 sweeps")
    require(abs(2 * 0.343384 / 3.574685 - 0.19212) < 5e-6
            and all(x in tex for x in ("$0.343384$", r"$\logit\qeq=3.575$", "$c=0.19212$")),
            "p=2 equal-block estimate 2 x 0.343384 / 3.575 = 0.19212")

    start = tex.rindex(r"\begin{table}", 0, tex.index(r"\label{tab:runs}"))
    runs = tex[start:tex.index(r"\end{table}", start)]  # caption precedes the label
    for printed, label in ((r"$20260823+1009s$", "pattern-set seed rule"),
                           (r"$20260825$", "field-grid dynamics and bootstrap seed"),
                           (r"dynamics seed $20260823$", "equilibrium-reference dynamics seed"),
                           (r"$9000+p$", "pattern-count scan seed"),
                           (r"$4000+N$", "size scan seed"),
                           (r"$31337+N$", "one-pattern run seed"),
                           (r"NumPy seed $4$", "Figure 1a path seed"),
                           (r"$256$ runs per set", "reference runs per pattern set"),
                           (r"$2000$ initial runs per set and field", "field-grid runs per set and field"),
                           (r"$8000$ initial runs per set and field sign", "scan runs per set and sign"),
                           (r"$150{,}000$ initial runs per field sign", "one-pattern runs per sign")):
        require(printed in runs, f"Table 5 states the {label}")
    require(protocol.SEED_BASE == 20260823, "Table 5 pattern-set seed base matches src.hopfield.protocol")
    require(all(z["n_samples"] == 256*200 for z in calibration) and len(calibration) == 20,
            "Table 5 reference runs agree with the saved runs (20 sets x 256 x 200)")

    aux = (paper / "main.aux").read_text()
    require(bool(re.search(r"\\newlabel\{main-text-end\}\{\{6\}\{6\}", aux)), "main text ends on page six")
    require(bool(re.search(r"\\newlabel\{app:exact\}\{\{A\}\{8\}", aux)), "appendix starts on page eight")
    log = (paper / "main.log").read_text()
    require(not re.search(r"Overfull|undefined|Citation .*Warning|Rerun to get|Fatal error", log),
            "no overflow, unresolved references or incomplete build")
    require("destination with the same identifier" not in log,
            "no duplicate PDF link destinations")
    pdf = paper / "main.pdf"
    proc = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, text=True, capture_output=True)
    require(not proc.stderr.strip(), "PDF text extraction has no warnings")
    pages = [p for p in proc.stdout.split("\f") if p.strip()]
    require(len(pages) == 13, "thirteen total pages")
    require(all(len(p.strip()) > 200 for p in pages), "no empty content pages")
    require(bool(re.search(r"^References\s*$", pages[6], re.M)), "references start on page seven")
    require("??" not in proc.stdout, "no unresolved-reference markers in PDF")
    authors = re.findall(r"\\Name\{([^}]*)\}", (root / "paper/main.tex").read_text())
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
            "pages": {"main": 6, "references": 1, "appendix": 6, "total": 13},
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
