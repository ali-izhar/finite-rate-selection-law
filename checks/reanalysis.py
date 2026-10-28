"""Reanalyse the saved counts behind the paper's estimates; no simulation is rerun.

    python -m checks.reanalysis [--output FILE]

Prints JSON: the field-grid slope under the paper's weights and three alternatives,
both pattern-set bootstraps, the logistic fit, the schedule, the linear ramps,
the schedule-scale factor, the one-pattern ratios, the p = 2 equal-block estimate
and block sizes, and the eigenvalues of the pattern overlap matrices (App. C).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


def audit(root: Path) -> dict:
    sys.path.insert(0, str(root))
    from src.hopfield import protocol as D, reference as H
    from src.hopfield.stats import logit_variance, weighted_slope_through_origin

    hashes = {}

    def read(relative):
        data = (root / relative).read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    analysis = read("results/hopfield/field_grid_analysis.json")
    grid = read("results/hopfield/field_grid.json")
    points = analysis["points"]
    assert len(points) == 8
    rows = []
    for p in points:
        assert p["successes"] == sum(grid["n_correct"][p["k"]])
        assert p["n"] == sum(grid["n_retained"][p["k"]])
        q, e = p["q_model"], p["q_eq"]
        keep = (0.5 < q["lower"] and q["upper"] < e["lower"]) or (
            e["upper"] < q["lower"] and q["upper"] < 0.5
        )
        rows.append(dict(k=p["k"], G=p["field_g"], n=p["n"],
                         successes=p["successes"], q=q["point"],
                         strictly_between=keep))

    x = np.array([logit(p["q_eq"]["point"]) for p in points])
    y = np.array([logit(p["q_model"]["point"]) for p in points])
    var = np.array([logit_variance(p["successes"], p["n"]) for p in points])
    pooled = weighted_slope_through_origin(x, y, var).point
    seeds = sorted({int(r[0]) for p in points for r in p["per_seed"]})
    slopes, wilson_slopes, weights = [], [], []
    for seed in seeds:
        r = np.array([next(r for r in p["per_seed"] if r[0] == seed) for p in points])
        n, successes = r[:, 2], r[:, 1]
        xx, yy = logit(r[:, 6]), logit(r[:, 3])
        w = 1 / np.array([logit_variance(s, t) for s, t in zip(successes, n)])
        slopes.append(float(np.sum(w * xx * yy) / np.sum(w * xx * xx)))
        fw = 1 / ((logit(r[:, 5]) - logit(r[:, 4])) / (2 * 1.96)) ** 2
        wilson_slopes.append(float(np.sum(fw * xx * yy) / np.sum(fw * xx * xx)))
        weights.append(n.sum())
    slopes, wilson_slopes, weights = map(np.asarray, (slopes, wilson_slopes, weights))

    def cluster_interval(cs, rng_seed):
        ids = np.random.default_rng(rng_seed).integers(0, len(seeds), (20_000, len(seeds)))
        bs = (cs[ids] * weights[ids]).sum(axis=1) / weights[ids].sum(axis=1)
        return np.percentile(bs, [2.5, 97.5]).tolist()

    success = np.array([p["successes"] for p in points])
    counts = np.array([p["n"] for p in points])
    matrix = np.column_stack((np.ones(len(x)), x))
    fit = minimize(
        lambda b: np.sum(counts * np.logaddexp(0, matrix @ b) - success * (matrix @ b)),
        [0, 0.2],
        jac=lambda b: matrix.T @ (counts * expit(matrix @ b) - success),
        method="BFGS",
        options={"gtol": 1e-6},
    )
    assert fit.success, fit.message

    linear = read("results/experiments/cw_linear_sweep.json")
    clock = read("results/experiments/sampler_clock.json")
    size1 = read("results/experiments/size_scan_p1.json")
    sizes = read("results/experiments/size_scan_p3_analysis.json")
    p2 = read("results/experiments/pattern_count_p2.json")
    ret, cor = np.array(p2["n_ret"]), np.array(p2["n_cor"])
    q2 = (cor[0].sum() + (ret[1] - cor[1]).sum()) / ret.sum()
    scaling = read("results/experiments/cw_scaling.json")
    stepped = {z["scale"]: z["c"] for z in scaling if z["N"] == 2048}
    checks = read("results/hopfield/sampler_checks.json")

    # Geometry of the same random patterns, not a trajectory covariance test.
    gram_rows, blocks = {}, []
    for n in (1024, 2048, 4096):
        shifts = []
        for s in range(3, 15):
            xi = H.patterns(D.seed_of(s), n, 3).astype(float)
            shifts.append(np.linalg.eigvalsh(xi @ xi.T / n) - 1)
        a = np.array(shifts)
        gram_rows[n] = dict(
            eigenvalue_shift_min=float(a.min()), eigenvalue_shift_max=float(a.max()),
            median_spectral_norm_times_sqrt_N=float(np.median(np.max(abs(a), axis=1)) * np.sqrt(n)),
        )
    for s in range(3, 15):
        xi = H.patterns(D.seed_of(s), 2048, 2)
        n_plus = int(((xi[0] * xi[1]) == 1).sum())
        blocks.append(dict(pattern_set=s, n_plus=n_plus, n_minus=2048 - n_plus))

    return dict(
        provenance=dict(
            method="saved counts and regenerated patterns; no simulation",
            source_sha256=hashes,
        ),
        bias_grid=dict(
            points=rows, n_not_strictly_between=sum(not z["strictly_between"] for z in rows),
            pooled_slope=float(pooled),
            delta_weight_slope=float(np.average(slopes, weights=weights)),
            delta_weight_slope_equal_sets=float(slopes.mean()),
            delta_weight_bootstrap_seed=2026, delta_weight_bootstrap_interval=cluster_interval(slopes, 2026),
            slope=float(np.average(wilson_slopes, weights=weights)),
            bootstrap_seed=20260825, bootstrap_interval=cluster_interval(wilson_slopes, 20260825),
            pooled_logistic_sensitivity=dict(intercept=float(fit.x[0]), slope=float(fit.x[1]),
                note="point estimates only"),
            calibration_G0=dict(successes=int(sum(grid["n_correct"][8])), n=int(sum(grid["n_retained"][8]))),
            argmax_ties_initial=grid["n_argmax_tie_scarce"], argmax_ties_final=grid["n_argmax_tie_final"],
        ),
        schedule=dict(beta=list(D.LADDER), sweeps=list(D.K_REDESCENT), total=sum(D.K_REDESCENT),
            size_scan="the N = 2048 dwell counts and G0 are used at every size"),
        linear_ramps=dict(rows=linear, max_relative_c_error=max(abs(z["ratio"] - 1) for z in linear),
            N_gamma_range=[min(z["N"] * z["gamma"] for z in linear), max(z["N"] * z["gamma"] for z in linear)]),
        stepped_budget=dict(c_by_scale=stepped, c_ratio_scale4_to_scale025=stepped[4.0] / stepped[0.25],
            note="final-cooling dwells scaled by 0.25 and 4; brief cooling and reheat unchanged"),
        clocks=clock, size_scan=sizes,
        p2=dict(equal_blocks_proxy_logit=2 * size1["1024"]["logit"], measured_logit=float(logit(q2)),
            equal_blocks_proxy_c=2 * size1["1024"]["logit"] / float(logit(0.9727397)),
            measured_c=float(logit(q2) / logit(0.9727397)), actual_blocks=blocks,
            note="equal-block estimate; block sizes, update rule and initialization are not matched"),
        pattern_gram=gram_rows,
        sampler_checks=checks,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.root.resolve()), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
