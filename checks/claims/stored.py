"""Category (b): numbers recomputed from saved results, with no rerun.

Sources: results/simple/ (the pre-registered Hopfield test: frozen predictions, per-row counts,
the G0 gate, the analysis), src/simple/design.py (its constants), docs/protocol-simple.md (the
pre-registration) and results/figures/mechanism.json (Fig. 1a).  The estimator is the paper's:
src.simple.estimate.estimate, a pooled logit over pattern sets and field signs with a cluster
bootstrap over the 50 pattern sets.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
from scipy.special import logit

from src.simple import design as SD
from src.simple.analyse import G1_ROWS, G2_ROW, T_ROWS, boot_logit, counts_for_estimator
from src.simple.estimate import estimate

from . import ROOT, Result, check_bool, check_interval, check_rounding, check_sci

S = "results/simple"
ROWS = ((1024, 1), (2048, 1), (4096, 1), (2048, 2), (1024, 3), (2048, 3), (4096, 3), (2048, 4))
# Table 1 as printed: predicted c, measured c, 95% interval, ratio
TABLE1 = {
    (1024, 1): ("0.0937", "0.0928", "0.0909", "0.0947", "0.990"),
    (2048, 1): ("0.0659", "0.0659", "0.0647", "0.0672", "1.001"),
    (4096, 1): ("0.0464", "0.0472", "0.0464", "0.0480", "1.015"),
    (2048, 2): ("0.0936", "0.0946", "0.0928", "0.0965", "1.011"),
    (1024, 3): ("0.1558", "0.1518", "0.1463", "0.1571", "0.975"),
    (2048, 3): ("0.1095", "0.1090", "0.1059", "0.1123", "0.995"),
    (4096, 3): ("0.0772", "0.0758", "0.0734", "0.0782", "0.982"),
    (2048, 4): ("0.1209", "0.1137", "0.1069", "0.1210", "0.940"),
}
# Table 3 as printed: runs, pattern-1 runs, m^1 = 0, GPU seed
TABLE3 = {
    (1024, 1): (320000, 320000, 0, 20271268), (2048, 1): (640000, 640000, 0, 20281508),
    (4096, 1): (1260000, 1260000, 0, 20301988), (2048, 2): (640000, 321598, 4318, 20281509),
    (1024, 3): (540000, 181324, 5752, 20271270), (2048, 3): (820000, 274222, 6619, 20281510),
    (4096, 3): (1460000, 504568, 6708, 20301990), (2048, 4): (940000, 254495, 7780, 20281511),
}


def _load(rel: str):
    return json.loads((ROOT / rel).read_text())


def checks(shared: dict) -> list[Result]:
    out: list[Result] = []
    pred = {(r["N"], r["p"]): r for r in _load(f"{S}/predictions.json")["rows"]}
    rows = {k: _load(f"{S}/row_N{k[0]}_p{k[1]}.json") for k in ROWS}
    analysis = _load(f"{S}/analysis.json")

    # ------------------------------------------ frozen predictions equal the live exact chains
    live = {n: shared["c_cw"][n] for n in (1024, 2048, 4096)}
    gap = max(abs(pred[(n, 1)]["c_CW"] - live[n]) for n in live)
    gap = max(gap, abs(pred[(2048, 2)]["c_pred"] - shared["c_p2"]),
              *(abs(pred[(n, 1)]["L_eq"] - shared["l_eq"][n]) for n in live))
    out.append(check_bool("B40", "Table 1, App. B", "predictions fixed before the runs",
                          gap < 1e-12, f"max |frozen - live| = {gap:.1e}", "< 1e-12",
                          f"{S}/predictions.json vs the live chains of checks/claims/exact_chain.py"))

    # ------------------------------------------ Table 1: measured c, 95% intervals, ratios
    est, ratio = {}, {}
    for k in ROWS:
        e = estimate(counts_for_estimator(rows[k]), pred[k]["L_eq"], level=0.95)
        est[k], ratio[k] = e, e["c"] / pred[k]["c_pred"]
        pc, mc, lo, hi, rt = TABLE1[k]
        tag = f"N={k[0]}, p={k[1]}"
        src = f"{S}/row_N{k[0]}_p{k[1]}.json: src.simple.estimate.estimate (95%, 10,000 resamples, seed 7)"
        out.append(check_rounding(f"B41p_{k[0]}_{k[1]}", f"Table 1, {tag}", pc, pred[k]["c_pred"],
                                  f"{S}/predictions.json c_pred"))
        out.append(check_rounding(f"B41m_{k[0]}_{k[1]}", f"Table 1, {tag}", mc, e["c"], src))
        out.append(check_interval(f"B41i_{k[0]}_{k[1]}", f"Table 1, {tag}", lo, hi, e["lo"], e["hi"], src))
        out.append(check_rounding(f"B41r_{k[0]}_{k[1]}", f"Table 1, {tag}", rt, ratio[k], "measured / predicted c"))
    stored = max(abs(analysis["rows"][f"N{n}_p{p}"]["estimate"]["c"] - est[(n, p)]["c"]) for n, p in ROWS)
    out.append(check_bool("B42", "Table 1", "point estimates equal results/simple/analysis.json",
                          stored < 1e-15, f"max diff {stored:.1e}", "< 1e-15", f"{S}/analysis.json"))

    # ------------------------------------------ Sec. 4 Result, Sec. 5, Abstract, Intro
    one = [ratio[(n, 1)] for n in (1024, 2048, 4096)]
    out.append(check_interval("B43", "Sec. 4", "0.990", "1.015", min(one), max(one),
                              "one-pattern ratios to the exact chain, three sizes"))
    out.append(check_rounding("B44", "Sec. 4", "1.1", 100.0 * (ratio[(2048, 2)] - 1.0),
                              "two-pattern runs above the exact two-block value, percent"))
    r3 = est[(2048, 3)]
    d3 = pred[(2048, 3)]["c_pred"]
    out.append(check_rounding("B45", "Sec. 4", "0.995", ratio[(2048, 3)], "c / (A_3 c_CW), N = 2048"))
    out.append(check_interval("B45i", "Sec. 4", "0.967", "1.026", r3["lo"] / d3, r3["hi"] / d3,
                              "95% interval of c / (A_3 c_CW), N = 2048"))
    t_rows = {k: (est[k]["lo"] / pred[k]["c_pred"], est[k]["hi"] / pred[k]["c_pred"]) for k in T_ROWS}
    out.append(check_bool(
        "B46", "Sec. 4, Sec. 5, App. B", "ratios 0.975 and 0.982 at N = 1024, 4096; every competition interval contains 1",
        f"{ratio[(1024, 3)]:.3f}" == "0.975" and f"{ratio[(4096, 3)]:.3f}" == "0.982"
        and all(lo <= 1.0 <= hi for lo, hi in t_rows.values()),
        "; ".join(f"N={k[0]},p={k[1]}: [{lo:.4f}, {hi:.4f}]" for k, (lo, hi) in t_rows.items()),
        "rounding; 1 inside each interval", "Table 1 rows p >= 3",
    ))
    a, b = counts_for_estimator(rows[(1024, 3)]), counts_for_estimator(rows[(4096, 3)])
    la, lb = pred[(1024, 3)]["L_eq"], pred[(4096, 3)]["L_eq"]
    expo = 0.5 * math.log2(est[(1024, 3)]["c"] / est[(4096, 3)]["c"])
    draws = np.log2((boot_logit(a, seed=SD.BOOT_SEED) / la) / (boot_logit(b, seed=SD.BOOT_SEED + 1) / lb)) / 2
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out.append(check_rounding("B47", "Sec. 4", "0.50", expo, "three-pattern size exponent, N = 1024 to 4096"))
    out.append(check_interval("B47i", "Sec. 4", "0.47", "0.53", float(lo), float(hi),
                              "independent cluster bootstraps of the two rows (src.simple.analyse)"))
    out.append(check_rounding("B48", "Abstract, Intro, Sec. 5", "6", 100.0 * (1.0 - ratio[(2048, 4)]),
                              "four patterns: measured below A_4 c_CW, percent"))
    d4 = pred[(2048, 4)]["c_pred"]
    out.append(check_interval("B48i", "Sec. 5", "0.884", "1.001", est[(2048, 4)]["lo"] / d4,
                              est[(2048, 4)]["hi"] / d4, "95% interval of c / (A_4 c_CW)"))
    worst3 = max(abs(ratio[k] - 1.0) for k in T_ROWS if k[1] == 3)
    out.append(check_bool("B49", "Abstract, Intro", "within 3% for three patterns at three sizes",
                          worst3 < 0.03, f"max |ratio - 1| = {100 * worst3:.2f}%", "< 3%", "Table 1, p = 3 rows"))
    deficit = {k: 100.0 * (1.0 - ratio[k]) for k in T_ROWS}
    out.append(check_bool(
        "B50", "Sec. 5, App. C", "all four A_p c_CW rows below their predictions, by 0.5-6%",
        all(d > 0 for d in deficit.values()) and f"{min(deficit.values()):.1f}" == "0.5"
        and f"{max(deficit.values()):.0f}" == "6",
        ", ".join(f"{d:.2f}%" for d in deficit.values()), "all positive; ends round as printed", "Table 1",
    ))
    png = {k: k[1] / (k[0] * SD.GAMMA) for k in T_ROWS}
    out.append(check_interval("B51", "Sec. 5", "0.18", "0.73", min(png.values()), max(png.values()),
                              "p / (N gamma) over the A_p c_CW rows"))
    out.append(check_bool(
        "B52", "Sec. 5", "largest deficit at p = 4, N = 2048, not at p = 3, N = 1024 where p/(N gamma) is largest",
        max(deficit, key=deficit.get) == (2048, 4) and max(png, key=png.get) == (1024, 3),
        f"largest deficit {max(deficit, key=deficit.get)}, largest p/(N gamma) {max(png, key=png.get)}",
        "equality", "Table 1",
    ))
    spearman = analysis["T"]["spearman_rho_deficit_vs_p_over_N_gamma"]
    out.append(check_bool("B53", "Sec. 5", "deficits do not follow p/(N gamma) closely",
                          spearman < 1.0 and analysis["T"]["reading"] == "(a)",
                          f"Spearman rho = {spearman:.2f}; reading {analysis['T']['reading']}",
                          "rho < 1; reading (a)", f"{S}/analysis.json"))

    # ------------------------------------------ App. B: gates, Table 3, cost
    g0 = _load(f"{S}/g0.json")["result"]
    gates = analysis["gates"]
    out.append(check_bool(
        "B54", "App. B", "G0 at N = 16, p = 3, 50 updates, 400 runs; overlaps carried = recomputed; all gates passed",
        (g0["N"], g0["p"], g0["updates"], g0["chains"]) == (16, 3, 50, 400) and g0["passed"]
        and all(r["drift_max"] == 0 for r in rows.values()) and gates["all_passed"],
        f"G0 {g0['N']},{g0['p']},{g0['updates']},{g0['chains']} passed={g0['passed']}; "
        f"G1 {gates['G1']['passed']}, G2 {gates['G2']['passed']}", "equality",
        f"{S}/g0.json, row_*.json drift_max, analysis.json gates",
    ))
    gate_ok = []
    for k in (*G1_ROWS, G2_ROW):
        e = estimate(counts_for_estimator(rows[k]), pred[k]["L_eq"], level=SD.GATE_LEVEL)
        gate_ok.append(e["lo"] <= pred[k]["c_pred"] <= e["hi"])
    out.append(check_bool("B55", "App. B", "each gate row read with a 98.75% interval; every one contains its exact value",
                          SD.GATE_LEVEL == 0.9875 and all(gate_ok), f"level {SD.GATE_LEVEL}, {gate_ok}",
                          "equality", "src/simple/design.py GATE_LEVEL; src.simple.estimate.estimate"))
    got = {k: (int(np.sum(r["counts"]["runs"])), int(np.sum(r["counts"]["pattern1"])),
               int(np.sum(r["counts"]["m1zero"])), int(r["torch_seed"])) for k, r in rows.items()}
    ties = sum(int(np.sum(r["counts"]["tie"])) for r in rows.values())
    out.append(check_bool("B56", "Table 3, App. B", "runs, pattern-1 runs, m^1 = 0 counts and GPU seeds; no tie",
                          got == TABLE3 and ties == 0 and all(r["complete"] for r in rows.values())
                          and all(got[k][3] == SD.torch_seed(*k) for k in got),
                          "all equal" if got == TABLE3 else str(got), "equality",
                          f"{S}/row_*.json counts; src.simple.design.torch_seed"))
    m1_pattern1 = all(got[k][1] + got[k][2] <= got[k][0] for k in got)
    out.append(check_bool("B56b", "App. B", "no pattern-1 run excluded (m^1 = 0 only in runs won by another pattern)",
                          m1_pattern1 and all(got[(n, 1)][2] == 0 for n in (1024, 2048, 4096)),
                          "m^1 = 0 is 0 in every one-pattern row", "equality",
                          f"{S}/row_*.json; with one pattern every run is a pattern-1 run"))
    updates = sum(r["runs"] * r["M_updates"] for r in rows.values())
    hours = sum(r["wall_seconds"] for r in rows.values()) / 3600.0
    out.append(check_sci("B57", "App. B", "4.1", 12, float(updates), f"{S}/row_*.json: runs x updates per run"))
    out.append(check_rounding("B57h", "App. B", "2.9", hours, f"{S}/row_*.json: sum of wall_seconds / 3600"))

    # ------------------------------------------ App. B: design constants and the pre-registration
    n_up = {n: SD.n_updates(n) for n in (1024, 2048, 4096)}
    ok = (SD.BETA0, SD.BETA_F, SD.GAMMA, SD.N_SETS, SD.SEED, SD.N_BOOT, SD.BOOT_SEED, SD.RUN_FACTOR) == \
        (0.7, 1.6, 0.004, 50, 20261027, 10_000, 7, 2) \
        and abs(SD.NG - 1.2544) < 1e-12 and abs(SD.field(2048) - 0.0006125) < 1e-15 \
        and all(n_up[n] == 225 * n for n in n_up) and all(r["M_updates"] == 225 * r["N"] for r in rows.values())
    out.append(check_bool(
        "B58", "Sec. 4, App. B", "beta 0.7 to 1.6 at gamma = 0.004, M = 225 N (225 sweeps), N G = 1.2544, 50 sets, "
        "seed 20261027, 10,000 resamples (seed 7), run counts doubled",
        ok, "all equal" if ok else "mismatch", "equality", "src/simple/design.py; row_*.json M_updates",
    ))
    out.append(check_rounding("B59", "App. B", "3", 100.0 * SD.HALF_WIDTH,
                              "pre-registered target half-width, percent of the predicted c (src/simple/design.py)"))
    text = (ROOT / "docs/protocol-simple.md").read_text()
    m = re.search(r"about -3% at \(1024, 3\), -1% at \(2048, 3\), -1\.5% at \(2048, 4\), -0\.5% at \(4096, 3\)", text)
    out.append(check_bool("B60", "Sec. 5, App. C", "expected deficits of 0.5-3% for the four A_p c_CW rows",
                          m is not None, "found in docs/protocol-simple.md Sec. 6" if m else "not found",
                          "text match", "docs/protocol-simple.md Sec. 6 (pre-stated expectation)"))
    eq3 = _load(f"{S}/eq_reference_p3.json")
    src_eq = f"{S}/eq_reference_p3.json: exact four-block Hopfield/CW equilibrium log-odds, p = 3, 50 pattern sets"
    for n, lo, hi, mean in (("512", "0.987", "0.997", "0.994"), ("1024", "0.991", "0.998", "0.997")):
        r = eq3[n]["ratios"]
        out.append(check_interval(f"B61_{n}", "App. B", lo, hi, min(r), max(r), f"{src_eq}, N = {n}"))
        out.append(check_rounding(f"B61m_{n}", "App. B", mean, float(np.mean(r)), f"{src_eq}, N = {n}, mean"))
    worst = max(1.0 - min(eq3[n]["ratios"]) for n in ("512", "1024"))
    worst = max(worst, 1.0 - _load(f"{S}/predictions.json")["checks"]["p2"]["L_eq_Hopfield_over_CW_mean"])
    out.append(check_rounding("B62", "Sec. 4", "1.3", 100.0 * worst,
                              "CW denominator within 1.3% of the Hopfield one where both were computed (p <= 3, N <= 1024), percent"))

    # ------------------------------------------ Fig. 1a
    fig1 = _load("results/figures/mechanism.json")
    out.append(check_bool(
        "B27", "Fig. 1 caption, App. A", "fourteen paths at N = 1024",
        fig1["trajectories"]["n"] == 14 and fig1["image"]["N"] == 1024,
        f"{fig1['trajectories']['n']} paths, N = {fig1['image']['N']}", "equality",
        "results/figures/mechanism.json",
    ))
    return out
