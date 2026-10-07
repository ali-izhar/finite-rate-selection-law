"""Category (b): numbers recomputed from saved results, with no rerun.

Sources: results/hopfield/ (the field grid), results/experiments/ (pattern-count,
size and one-pattern runs, the exact-chain schedule runs, the MNIST control
arm) and results/figures/mechanism.json.  The estimators are the paper's:
src/hopfield/seed_bootstrap.py for the field-grid slope, and per-draw +/-G
pooling with a 12-draw bootstrap for Table 1.
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.special import logit

from src.hopfield import protocol as D
from src.hopfield.equilibrium import qeq_route_a

from . import (
    ROOT, Result, check_bool, check_interval, check_rounding, check_sci,
)


def _load(rel: str):
    return json.loads((ROOT / rel).read_text())


def checks(shared: dict) -> list[Result]:
    out: list[Result] = []
    c_cw, amp = shared["c_cw"], shared["amp"]

    # ------------------------------------------ field-grid slope (Sec. 4, App. B)
    from src.hopfield.seed_bootstrap import N_BOOT, SEED, set_slopes
    slopes, weights = set_slopes()
    ids = np.random.default_rng(SEED).integers(0, len(slopes), (N_BOOT, len(slopes)))
    lo, hi = np.percentile((slopes[ids] * weights[ids]).sum(axis=1) / weights[ids].sum(axis=1), [2.5, 97.5])
    c_chain, c_equal = float(np.average(slopes, weights=weights)), float(slopes.mean())
    src_bs = "src/hopfield/seed_bootstrap.py on results/hopfield/field_grid_analysis.json"
    out.append(check_rounding("B01", "Abstract, Sec. 4, App. B", "0.1994", c_chain, src_bs,
                              "sample-weighted mean of per-set slopes"))
    out.append(check_rounding("B02", "Sec. 4", "0.2101", c_equal, src_bs, "equal-weight mean of per-set slopes"))
    out.append(check_interval("B03", "Sec. 4, App. B", "0.1908", "0.2095", float(lo), float(hi), src_bs,
                              "20,000 pattern-set resamples, seed 20260825"))

    # ------------------------------------------ field-grid details (Sec. 4, App. B, Table 4)
    analysis = _load("results/hopfield/field_grid_analysis.json")
    top = analysis["points"][0]
    out.append(check_bool("B31", "Sec. 4", "66.9% sampled, 97.26% reference at G = 0.0006125",
                          f"{100 * top['q_model']['point']:.1f}" == "66.9" and f"{100 * top['q_eq']['point']:.2f}" == "97.26",
                          f"{100 * top['q_model']['point']:.3f}%, {100 * top['q_eq']['point']:.3f}%", "rounds as printed",
                          "results/hopfield/field_grid_analysis.json, strongest positive field"))
    out.append(check_bool("B32", "Sec. 4", "c = 0.70/3.57", f"{logit(top['q_model']['point']):.2f}/{logit(top['q_eq']['point']):.2f}" == "0.70/3.57",
                          f"{logit(top['q_model']['point']):.4f}/{logit(top['q_eq']['point']):.4f}", "rounds as printed",
                          "logits of B31"))
    from checks.reanalysis import audit
    grid_fit = audit(ROOT)["bias_grid"]
    alt = [grid_fit["delta_weight_slope"], grid_fit["pooled_slope"], grid_fit["pooled_logistic_sensitivity"]["slope"]]
    out.append(check_interval("B33", "Sec. 4, App. B", "0.1994", "0.1997", min(alt), max(alt),
                              "checks/reanalysis.py: three alternative estimators"))
    out.append(check_rounding("B34", "App. B", "0.000944", grid_fit["pooled_logistic_sensitivity"]["intercept"],
                              "checks/reanalysis.py: intercept of the pooled logistic fit"))
    g4 = next(z for z in _load("results/hopfield/sampler_checks.json") if z["name"].startswith("G4 "))["values"]
    out.append(check_sci("B35", "App. B", "2.108", -8, g4["route_gap"],
                         "results/hopfield/sampler_checks.json G4: max |exact - weak-field q_eq| over the eight fields"))
    ref = _load("results/hopfield/overlap_run.json")
    ratio = [z["se_across_chains"] / z["se_spec"] for z in ref]
    out.append(check_interval("B36", "App. B", "0.85", "1.05", min(ratio), max(ratio),
                              "results/hopfield/overlap_run.json: SE of the 256 run means / reported SE, per set"))
    grid = _load("results/hopfield/field_grid.json")
    kept = [sum(x) for x in grid["n_miscommitted"]]
    out.append(check_bool("B37", "Table 4 caption", "32,467-34,178 reheated (33,331 at G = 0); ties 101 initial, 0 final",
                          (min(kept[:8]), max(kept[:8]), kept[8]) == (32467, 34178, 33331)
                          and (grid["n_argmax_tie_scarce"], grid["n_argmax_tie_final"]) == (101, 0),
                          f"{min(kept[:8])}..{max(kept[:8])}, {kept[8]}; ties {grid['n_argmax_tie_scarce']}, {grid['n_argmax_tie_final']}",
                          "equality", "results/hopfield/field_grid.json"))

    # ------------------------------------------ Table 1: pattern counts at N = 2048
    l_eq = float(logit(qeq_route_a(D.N_SPINS, D.BETA_DEEP, D.G0)))
    rng = np.random.default_rng(2026)
    printed_c = {1: ("0.127", "0.124", "0.131"), 2: ("0.188", "0.184", "0.191"),
                 3: ("0.201", "0.194", "0.210"), 4: ("0.211", "0.191", "0.239")}
    c_meas = {}
    for p in (1, 2, 3, 4):
        v = _load(f"results/experiments/pattern_count_p{p}.json")
        cor = np.array(v["n_cor"]); ret = np.array(v["n_ret"])
        s_d = cor[0] + (ret[1] - cor[1]); n_d = ret[0] + ret[1]
        c_meas[p] = float(logit(s_d.sum() / n_d.sum()) / l_eq)
        idx = rng.integers(0, len(s_d), size=(20000, len(s_d)))
        boot = np.array([logit(s_d[i].sum() / n_d[i].sum()) / l_eq for i in idx])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        src = (f"pattern_count_p{p}.json: pooled c = logit(q)/logit(q_eq); "
               f"bootstrap over the 12 pattern sets (20,000 resamples, seed 2026)")
        out.append(check_rounding(f"B0{3 + p}", f"Table 1, p={p}", printed_c[p][0], c_meas[p], src))
        out.append(check_interval(f"B0{3 + p}i", f"Table 1, p={p}", printed_c[p][1], printed_c[p][2],
                                  float(lo), float(hi), src, slack=5e-4))
    worst = max(abs(c_meas[p] / (c_cw * amp[p]) - 1.0) for p in (1, 2, 3, 4))
    out.append(check_rounding("B08", "Sec. 1, Sec. 5", "11", 100.0 * worst,
                              "largest |c_meas / (c_CW x A_p) - 1| over p = 1..4, percent"))
    out.append(check_bool("B09", "Sec. 1", "closely matches three-pattern simulations",
                          abs(c_meas[3] / (c_cw * amp[3]) - 1.0) < 0.02,
                          f"{100.0 * (c_meas[3] / (c_cw * amp[3]) - 1.0):+.1f} %", "|excess| < 2 %",
                          "c_meas(p=3) / (c_CW x A_3) - 1"))

    # ------------------------------------------ Table 1 and Sec. 5: sizes at p = 3
    ex = {int(k): v for k, v in _load("results/experiments/exact_protocol_sizes.json").items()}
    sc = {int(k): v for k, v in _load("results/experiments/size_scan_p3.json").items()}
    sc[2048] = _load("results/experiments/pattern_count_p3.json")
    cN, cP = {}, {}
    for N in (1024, 2048, 4096):
        v = sc[N]; cor = np.array(v["n_cor"]); ret = np.array(v["n_ret"])
        s_d = cor[0] + (ret[1] - cor[1]); n_d = ret[0] + ret[1]
        cN[N] = float(logit(s_d.sum() / n_d.sum()) / logit(ex[N]["q_eq"]))
        cP[N] = ex[N]["c_cw"] * amp[3]
    for N, pm, pp in ((1024, "0.263", "0.284"), (4096, "0.160", "0.150")):
        out.append(check_rounding(f"B10m{N}", "Table 1", pm, cN[N],
                                  f"size_scan_p3.json N={N}: logit q / logit q_eq(N)"))
        out.append(check_rounding(f"B10p{N}", "Table 1", pp, cP[N],
                                  f"exact_protocol_sizes.json: c_CW({N}) x A_3"))
    for N, printed in ((512, "0.227"), (1024, "0.171"), (2048, "0.120"), (4096, "0.090")):
        out.append(check_rounding(f"B11_{N}", "Table 1 caption, App. B", printed, ex[N]["c_cw"],
                                  f"exact_protocol_sizes.json: c_CW at N={N}"))
    out.append(check_rounding("B12m", "Sec. 5", "-0.36", math.log(cN[4096] / cN[1024]) / math.log(4.0),
                              "log(c_meas(4096)/c_meas(1024)) / log 4"))
    out.append(check_rounding("B12p", "Sec. 5", "-0.46", math.log(cP[4096] / cP[1024]) / math.log(4.0),
                              "log(c_pred(4096)/c_pred(1024)) / log 4"))

    # ------------------------------------------ one-pattern ratios (Sec. 5, App. B)
    rc = {int(k): v for k, v in _load("results/experiments/sampler_clock.json").items()}
    for N, printed in ((512, "1.065"), (1024, "1.126"), (2048, "1.063"), (4096, "1.020")):
        out.append(check_rounding(f"B13_{N}", "Sec. 5, App. B" if N != 512 else "App. B", printed,
                                  rc[N]["r"], f"sampler_clock.json: c_meas(p=1, N={N}) / c_CW(N)"))
    out.append(check_rounding("B14", "Sec. 5", "0.212", c_cw * amp[3] * rc[2048]["r"],
                              "c_CW x A_3 x r(2048)"))
    for N, printed in ((1024, "0.319"), (4096, "0.153")):
        out.append(check_rounding(f"B15_{N}", "Sec. 5", printed, cP[N] * rc[N]["r"],
                                  f"c_pred({N}) x r({N})"))
    for N, printed in ((1024, "18"), (4096, "4")):
        dev = 100.0 * abs(rc[N]["p3_meas_over_pred_times_r"] - 1.0)
        out.append(check_rounding(f"B16_{N}", "Sec. 5", printed, dev,
                                  f"sampler_clock.json: |c_meas / (c_pred x r) - 1| at N={N}, percent"))

    # ------------------------------------------ p = 2 two-block estimate (App. C)
    one = _load("results/experiments/size_scan_p1.json")["1024"]
    out.append(check_rounding("B17", "App. C", "0.343384", one["logit"],
                              "size_scan_p1.json['1024'].logit: pooled one-pattern logit at N=1024"))
    out.append(check_rounding("B18", "App. C", "0.19212", 2.0 * one["logit"] / shared["logit_q_eq"],
                              "2 x B17 / logit q_eq (A07)"))
    out.append(check_rounding("B19", "App. C", "0.18783", c_meas[2], "pattern_count_p2.json, as B05"))
    from src.hopfield.reference import patterns
    sizes = [int((patterns(D.seed_of(s), 2048, 2).prod(axis=0) > 0).sum()) for s in range(3, 15)]
    sizes += [2048 - n for n in sizes]
    out.append(check_bool("B20", "App. C", "blocks of 980-1068 spins", (min(sizes), max(sizes)) == (980, 1068),
                          f"{min(sizes)}..{max(sizes)}", "equality",
                          "pattern sets s = 3..14 at N=2048: counts of xi^1 xi^2 = +1 and -1"))

    # ------------------------------------------ schedule factor (Sec. 4, Table 3)
    scale = {(z["N"], z["scale"]): z["c"] for z in _load("results/experiments/cw_scaling.json")}
    slow, fast = scale[(2048, 0.25)], scale[(2048, 4.0)]
    out.append(check_rounding("B21", "Table 3 caption", "0.098978", slow, "cw_scaling.json: N=2048, scale 0.25"))
    out.append(check_rounding("B22", "Table 3 caption", "0.146625", fast, "cw_scaling.json: N=2048, scale 4"))
    out.append(check_rounding("B23", "Sec. 4, Table 3 caption", "1.4814", fast / slow, "B22 / B21"))
    fields = [z["c_pred"] for z in _load("results/experiments/exact_protocol.json")]
    out.append(check_interval("B24", "Table 3 caption", "0.119498", "0.119808", min(fields), max(fields),
                              "exact_protocol.json: c_CW over the eight fields"))

    # ------------------------------------------ protocol constants (Sec. 4, Tables 4-5, Fig. 1)
    out.append(check_bool(
        "B25", "Sec. 4, Table 4, Table 5", "eight nonzero fields; twenty pattern sets; 2000 initial runs per set and field",
        len(analysis["points"]) == 8 and len(grid["seed_indices"]) == 20 and grid["chains_per_seed"] == 2000,
        f"{len(analysis['points'])} fields, {len(grid['seed_indices'])} sets, {grid['chains_per_seed']} runs",
        "counts", "results/hopfield/field_grid_analysis.json, field_grid.json",
    ))
    out.append(check_bool(
        "B26", "Sec. 4, App. B", "ten temperatures 0.6 to 1.6; reheat 30 sweeps at beta = 0.7; N=2048, p=3, beta_f=1.6",
        len(D.LADDER) == 10 and D.LADDER[0] == 0.60 and D.LADDER[-1] == 1.60
        and D.K_REHEAT == 30 and D.BETA_REHEAT == 0.7
        and (D.N_SPINS, D.N_PATTERNS, D.BETA_DEEP) == (2048, 3, 1.6),
        f"{len(D.LADDER)} temperatures {D.LADDER[0]}..{D.LADDER[-1]}, reheat {D.K_REHEAT} at {D.BETA_REHEAT}, "
        f"N={D.N_SPINS}, p={D.N_PATTERNS}, beta={D.BETA_DEEP}", "equality", "src/hopfield/protocol.py",
    ))
    fig1 = _load("results/figures/mechanism.json")
    out.append(check_bool(
        "B27", "Fig. 1 caption, Table 5", "fourteen paths at N = 1024",
        fig1["trajectories"]["n"] == 14 and fig1["image"]["N"] == 1024,
        f"{fig1['trajectories']['n']} paths, N = {fig1['image']['N']}", "equality",
        "results/figures/mechanism.json",
    ))

    # ------------------------------------------ MNIST illustration (docs/mnist-illustration.md)
    MN = "docs/mnist-illustration.md"
    ca = _load("results/experiments/dbm_control_arm.json")
    fresh = [a["end_3"] for a in ca["fresh"] + ca["fresh_matched"]]
    rep = [a["end_3"] for a in ca["repair"]]
    out.append(check_bool("B28", MN, "reheated runs end as 3s in 20-24 of 24 over ten seeds",
                          len(rep) == 10 and (min(rep), max(rep)) == (20, 24), f"{sorted(rep)}", "min 20, max 24",
                          "dbm_control_arm.json, repair arm (src/experiments/dbm_control_arm.py)"))
    out.append(check_bool("B29", MN, "random-pixel starts end as 3s in 21-24 of 24 (both controls)",
                          len(fresh) == 20 and (min(fresh), max(fresh)) == (21, 24), f"{sorted(fresh)}",
                          "min 21, max 24", "dbm_control_arm.json, fresh and fresh_matched arms"))
    z = np.load(ROOT / "results/experiments/dbm_reheat.npz", allow_pickle=True)
    proto = json.loads(str(z["protocol"]))
    out.append(check_bool("B30", MN, "24 runs; reheat at beta = 0.55",
                          int(z["frames"].shape[0]) == 24 and proto["beta_reheat"] == 0.55,
                          f"{int(z['frames'].shape[0])} runs, beta {proto['beta_reheat']}", "equality",
                          "results/experiments/dbm_reheat.npz"))
    return out
