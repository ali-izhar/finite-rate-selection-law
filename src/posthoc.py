"""Post-hoc per-set analysis of the Hopfield test (not in the protocol)  -> results/posthoc/per_set.json.

    python -m src.posthoc

Reads the saved rows and frozen predictions in results/simple/ (unchanged) and runs no simulation.
The protocol's estimator pools the 50 pattern sets of a row.  The competition factor A_p
assumes independent early amplitudes of common variance, which makes the patterns exchangeable:
pattern 1 should win a share 1/p of the runs in every set, and every set should give the same c.
This module tests that per set (field signs pooled within a set):

  win_share        pattern 1's share of the runs it wins, per set (min, max)
  homogeneity      chi-square test that all sets share the pooled sign fraction q (49 df)
  spearman         rank correlation of per-set c with the win share
  sd_ratio         SD of per-set c / c_pred, raw and beyond binomial noise
  equal_weight     mean over sets of per-set logit q / L_eq, over c_pred (sets weighted equally)
  boot_seeds       for every row: the paper's 95% percentile cluster bootstrap (10,000 resamples)
                   under bootstrap seeds 0-199, i.e. the Monte Carlo spread of the interval ends

Paper: Sec. 5 ("Pattern sets, post hoc"), App. B (Table 4).  Checked by checks.claims (B70-B79c).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import logit
from scipy.stats import chi2, spearmanr

from src.simple.analyse import counts_for_estimator
from src.simple.estimate import estimate

ROOT = Path(__file__).resolve().parents[1]
SIMPLE = ROOT / "results" / "simple"
OUT = ROOT / "results" / "posthoc" / "per_set.json"
ROWS = ((1024, 1), (2048, 1), (4096, 1), (2048, 2), (1024, 3), (2048, 3), (4096, 3), (2048, 4))
BOOT_SEEDS = tuple(range(200))


def per_set(row: dict, c_pred: float, l_eq: float) -> dict:
    """Per-set statistics of one row; signs pooled within each set."""
    c = row["counts"]
    n = np.asarray(c["pattern1"], dtype=np.float64).sum(axis=1)      # pattern-1 runs per set
    k = np.asarray(c["correct"], dtype=np.float64).sum(axis=1)       # of those, field's sign
    runs = np.asarray(c["runs"], dtype=np.float64).sum(axis=1)
    wins = np.asarray(row["winners"], dtype=np.float64)[..., 0].sum(axis=1)
    sets = n.size
    q = k.sum() / n.sum()
    chi_sq = float((((k - n * q) ** 2) / (n * q * (1 - q))).sum())
    q_set = k / n
    c_set = logit(q_set) / l_eq
    r_set = c_set / c_pred
    var_binom = 1.0 / (n * q_set * (1 - q_set))                      # variance of logit q_set
    excess = float(np.var(r_set, ddof=1) - np.mean(var_binom) / (l_eq * c_pred) ** 2)
    share = wins / runs
    out = dict(
        sets=int(sets),
        win_share=dict(pooled=float(wins.sum() / runs.sum()), min=float(share.min()),
                       max=float(share.max()), per_set=share.tolist()),
        homogeneity=dict(chi2=chi_sq, df=int(sets - 1), p_value=float(chi2.sf(chi_sq, sets - 1))),
        ratio_per_set=dict(min=float(r_set.min()), max=float(r_set.max()),
                           sd_raw=float(np.std(r_set, ddof=1)),
                           sd_beyond_binomial=float(np.sqrt(max(excess, 0.0)))),
        equal_weight=dict(c=float(np.mean(logit(q_set)) / l_eq),
                          ratio=float(np.mean(logit(q_set)) / l_eq / c_pred)),
        c_per_set=c_set.tolist(),
    )
    if np.ptp(share) > 0:
        out["spearman_c_vs_win_share"] = float(spearmanr(share, c_set)[0])
    return out


def boot_seeds(row: dict, c_pred: float, l_eq: float) -> dict:
    """The paper's 95% interval of c / c_pred under 200 bootstrap seeds."""
    counts = counts_for_estimator(row)
    ends = np.array([[e["lo"], e["hi"]] for e in
                     (estimate(counts, l_eq, seed=s, level=0.95) for s in BOOT_SEEDS)]) / c_pred
    contains = (ends[:, 0] <= 1.0) & (ends[:, 1] >= 1.0)
    return dict(seeds=[BOOT_SEEDS[0], BOOT_SEEDS[-1]], resamples=10_000, level=0.95,
                ratio_lo_min=float(ends[:, 0].min()), ratio_lo_max=float(ends[:, 0].max()),
                ratio_hi_min=float(ends[:, 1].min()), ratio_hi_max=float(ends[:, 1].max()),
                fraction_containing_1=float(contains.mean()))


def main() -> dict:
    pred = {(r["N"], r["p"]): r for r in json.loads((SIMPLE / "predictions.json").read_text())["rows"]}
    result: dict = dict(step="posthoc", note="post hoc; not part of the pre-registered analysis",
                        inputs="results/simple/row_N*_p*.json, results/simple/predictions.json",
                        rows={})
    for n_spins, p in ROWS:
        row = json.loads((SIMPLE / f"row_N{n_spins}_p{p}.json").read_text())
        z = pred[(n_spins, p)]
        entry = per_set(row, z["c_pred"], z["L_eq"])
        entry["boot_seeds"] = boot_seeds(row, z["c_pred"], z["L_eq"])
        result["rows"][f"N{n_spins}_p{p}"] = entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    res = main()
    for key, r in res["rows"].items():
        b = r["boot_seeds"]
        print(f"{key}: win share {r['win_share']['min']:.3f}-{r['win_share']['max']:.3f}; "
              f"homogeneity chi2={r['homogeneity']['chi2']:.1f}/49 p={r['homogeneity']['p_value']:.2g}; "
              f"rho={r.get('spearman_c_vs_win_share', float('nan')):.2f}; "
              f"per-set R {r['ratio_per_set']['min']:.2f}-{r['ratio_per_set']['max']:.2f}, "
              f"SD beyond binomial {r['ratio_per_set']['sd_beyond_binomial']:.3f}; "
              f"equal-weight R={r['equal_weight']['ratio']:.3f}; "
              f"95% hi over seeds {b['ratio_hi_min']:.4f}-{b['ratio_hi_max']:.4f}, "
              f"contains 1 in {100 * b['fraction_containing_1']:.0f}%")
    print(f"wrote {OUT.relative_to(ROOT)}")
