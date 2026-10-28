"""Category (b): the post-hoc per-set analysis (paper Sec. 5, App. B, Table 4), from saved results.

Recomputed live with src.posthoc from results/simple/ (no rerun), and compared with the saved
results/posthoc/per_set.json.  The analysis is not in the protocol; the paper labels it post hoc.
"""

from __future__ import annotations

import json
import re

from src import posthoc as PH

from . import ROOT, Result, check_bool, check_interval, check_rounding

S = "results/simple"
T_ROWS = ((1024, 3), (2048, 3), (4096, 3), (2048, 4))
# Table 4 as printed: win share range, P (mantissa, exponent), Spearman, excess SD %, equal-weight ratio
TABLE4 = {
    (1024, 3): ("0.075", "0.692", 2, -12, "-0.75", "8.0", "1.003"),
    (2048, 3): ("0.168", "0.566", 2, -6, "-0.69", "7.7", "1.013"),
    (4096, 3): ("0.229", "0.522", 2, -5, "-0.56", "6.6", "0.993"),
    (2048, 4): ("0.087", "0.713", 5, -40, "-0.72", "13.0", "1.002"),
}


def _load(rel: str):
    return json.loads((ROOT / rel).read_text())


def checks(shared: dict) -> list[Result]:
    out: list[Result] = []
    pred = {(r["N"], r["p"]): r for r in _load(f"{S}/predictions.json")["rows"]}
    live = {}
    for k in PH.ROWS:
        row = _load(f"{S}/row_N{k[0]}_p{k[1]}.json")
        live[k] = PH.per_set(row, pred[k]["c_pred"], pred[k]["L_eq"])
    saved = _load("results/posthoc/per_set.json")["rows"]
    gap = max(abs(saved[f"N{k[0]}_p{k[1]}"]["ratio_per_set"]["sd_beyond_binomial"]
                  - live[k]["ratio_per_set"]["sd_beyond_binomial"]) for k in PH.ROWS)
    gap = max(gap, *(abs(saved[f"N{k[0]}_p{k[1]}"]["homogeneity"]["chi2"] - live[k]["homogeneity"]["chi2"])
                     for k in PH.ROWS))
    out.append(check_bool("B70", "Table 4", "saved per-set analysis equals the live recomputation",
                          gap < 1e-12, f"max diff {gap:.1e}", "< 1e-12",
                          "results/posthoc/per_set.json vs src.posthoc.per_set"))

    # ------------------------------------------ Table 4, cell by cell
    src = f"{S}/row_N*_p*.json: src.posthoc.per_set"
    for k, (wlo, whi, mant, expo, rho, sd, eq) in TABLE4.items():
        r, tag = live[k], f"Table 4, N={k[0]}, p={k[1]}"
        out.append(check_interval(f"B71w_{k[0]}_{k[1]}", tag, wlo, whi,
                                  r["win_share"]["min"], r["win_share"]["max"], f"{src}: win share range"))
        pval = r["homogeneity"]["p_value"]
        out.append(check_bool(f"B71p_{k[0]}_{k[1]}", tag, f"{mant}e{expo}",
                              round(pval / 10.0 ** expo) == mant, f"{pval:.2e}",
                              "mantissa rounds to integer", f"{src}: chi-square, 49 df"))
        out.append(check_rounding(f"B71s_{k[0]}_{k[1]}", tag, rho, r["spearman_c_vs_win_share"],
                                  f"{src}: Spearman(c_s, win share)"))
        out.append(check_rounding(f"B71d_{k[0]}_{k[1]}", tag, sd, 100.0 * r["ratio_per_set"]["sd_beyond_binomial"],
                                  f"{src}: SD of c_s / c_pred beyond binomial, percent"))
        out.append(check_rounding(f"B71e_{k[0]}_{k[1]}", tag, eq, r["equal_weight"]["ratio"],
                                  f"{src}: mean logit q_s / L_eq / c_pred"))

    # ------------------------------------------ Sec. 5 and App. B text
    p4 = live[(2048, 4)]
    out.append(check_interval("B72", "Sec. 5", "0.09", "0.71", p4["win_share"]["min"], p4["win_share"]["max"],
                              "pattern-1 win share over the four-pattern sets"))
    rhos = [live[k]["spearman_c_vs_win_share"] for k in T_ROWS]
    out.append(check_bool("B73", "Sec. 5", "in every A_p c_CW row per-set c falls as the win share rises",
                          all(x < 0 for x in rhos), ", ".join(f"{x:.2f}" for x in rhos),
                          "every Spearman rho < 0", src))
    sds = [100.0 * live[k]["ratio_per_set"]["sd_beyond_binomial"] for k in T_ROWS]
    out.append(check_interval("B74", "Sec. 5", "7", "13", min(sds), max(sds),
                              "SD of per-set c / (A_p c_CW) beyond sampling noise, percent"))
    out.append(check_rounding("B75", "Sec. 5", "1.00", p4["equal_weight"]["ratio"],
                              "four patterns, sets weighted equally"))
    ctrl = [live[k]["homogeneity"]["p_value"] for k in PH.ROWS if k[1] <= 2]
    out.append(check_interval("B76", "App. B", "0.56", "0.94", min(ctrl), max(ctrl),
                              "chi-square P in the one- and two-pattern rows"))
    t_p = max(live[k]["homogeneity"]["p_value"] for k in T_ROWS)
    out.append(check_bool("B76b", "Abstract", "single pattern sets depart from the prediction by more than sampling noise",
                          t_p < 1e-4, f"largest chi-square P over the A_p rows {t_p:.1e}", "< 1e-4", src))

    # ------------------------------------------ App. B: bootstrap-seed sensitivity (200 seeds)
    boot = {k: PH.boot_seeds(_load(f"{S}/row_N{k[0]}_p{k[1]}.json"), pred[k]["c_pred"], pred[k]["L_eq"])
            for k in T_ROWS}
    b4 = boot[(2048, 4)]
    srcb = "src.posthoc.boot_seeds: estimate(level=0.95, 10,000 resamples) at seeds 0-199"
    out.append(check_interval("B77", "App. B", "0.997", "1.002", b4["ratio_hi_min"], b4["ratio_hi_max"],
                              f"{srcb}, upper end of c / (A_4 c_CW)"))
    below = int(round(200 * (1.0 - b4["fraction_containing_1"])))
    out.append(check_bool("B78", "App. B", "upper end below 1 for 139 of the 200 seeds", below == 139,
                          f"{below} of 200", "equality", srcb))
    others = [boot[k]["fraction_containing_1"] for k in T_ROWS if k != (2048, 4)]
    out.append(check_bool("B79", "App. B", "the other intervals contain 1 at every one of these seeds",
                          all(f == 1.0 for f in others), str(others), "all 1.0", srcb))
    saved4 = saved["N2048_p4"]["boot_seeds"]
    out.append(check_bool("B79b", "App. B", "saved seed sensitivity equals the live recomputation",
                          abs(saved4["ratio_hi_max"] - b4["ratio_hi_max"]) < 1e-12
                          and saved4["fraction_containing_1"] == b4["fraction_containing_1"],
                          f"saved {saved4['fraction_containing_1']}, live {b4['fraction_containing_1']}",
                          "equality", "results/posthoc/per_set.json"))
    text = (ROOT / "paper/main.tex").read_text()
    out.append(check_bool("B79c", "Sec. 5, App. B", "the per-set analysis is labeled post hoc in the paper",
                          len(re.findall(r"post[- ]hoc", text)) >= 2 and "is not in the protocol" in text,
                          "found", "text match", "paper/main.tex"))
    return out
