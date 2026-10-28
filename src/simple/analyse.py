"""Pre-registered analysis (docs/protocol-simple.md Sec. 6), CPU only  -> results/simple/analysis.json.

Order: G0 and the drift checks, then G1 (p = 1 rows) and G2 ((2048, 2)); the A_p rows are
interpreted only if all of them pass.  The reading of the T rows is reported mechanically where
the pre-registration fixes it ((a): every interval contains 1; (c): some interval lies above 1);
otherwise the per-row facts and the rank correlation of the deficits with p/(N gamma) are written
and the reading is left to the author, because the pre-registration does not define "order".
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.special import logit
from scipy.stats import spearmanr

from src.simple import design as D
from src.simple.estimate import estimate, reading
from src.simple.provenance import provenance
from src.simple.run import OUT_DIR, _dump, predictions

G1_ROWS = ((1024, 1), (2048, 1), (4096, 1))
G2_ROW = (2048, 2)
T_ROWS = ((2048, 3), (2048, 4), (1024, 3), (4096, 3))    # primary first


def load_row(n_spins: int, p: int, seed_offset: int = 0) -> dict | None:
    name = f"row_N{n_spins}_p{p}" + (f"_rerun{seed_offset}" if seed_offset else "") + ".json"
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def counts_for_estimator(row: dict) -> np.ndarray:
    """(K, 2, 2): [..., 0] pattern-1 runs, [..., 1] of those with the sign of the field."""
    c = row["counts"]
    return np.stack([np.asarray(c["pattern1"]), np.asarray(c["correct"])], axis=-1)


def boot_logit(counts: np.ndarray, n_boot: int = D.N_BOOT, seed: int = D.BOOT_SEED) -> np.ndarray:
    """Bootstrap draws of logit(q-hat), the same resampling as estimate.estimate."""
    per_set = np.asarray(counts, dtype=np.float64).sum(axis=1)
    rng = np.random.default_rng(seed)
    tot = per_set[rng.integers(0, per_set.shape[0], size=(n_boot, per_set.shape[0]))].sum(axis=1)
    return logit(tot[:, 1] / tot[:, 0])


def main(seed_offset: int = 0) -> dict:
    pred = {(r["N"], r["p"]): r for r in predictions()["rows"]}
    out: dict = dict(step="analyse", seed_offset=seed_offset, rows={}, gates={})
    g0_path = os.path.join(OUT_DIR, "g0.json")
    g0_ok = os.path.exists(g0_path) and json.load(open(g0_path))["result"]["passed"]
    rows, missing = {}, []
    for key in D.ROWS:
        r = load_row(*key, seed_offset)
        if r is None or not r.get("complete"):
            missing.append(key)
        else:
            rows[key] = r
    drift_ok = all(r["drift_max"] == 0 for r in rows.values())
    out["gates"]["G0"] = dict(passed=bool(g0_ok and drift_ok), g0_file=g0_ok, drift_zero=drift_ok,
                              missing_or_incomplete=[list(k) for k in missing])

    def row_result(key, level=0.95):
        r, pr = rows[key], pred[key]
        cts = counts_for_estimator(r)
        est = estimate(cts, pr["L_eq"], level=level)
        c = r["counts"]
        res = dict(estimate=est, c_pred=pr["c_pred"], prediction=pr["prediction"],
                   reading=reading(est, pr["c_pred"]),
                   excluded=dict(tie=int(np.sum(c["tie"])), m1zero=int(np.sum(c["m1zero"])),
                                 runs=int(np.sum(c["runs"]))),
                   win_share_pattern1=float(np.sum(c["pattern1"]) / np.sum(c["runs"])),
                   c_over_Ap_csel=est["c"] / pr["A_p_c_sel"])
        return res

    for key in (*G1_ROWS, G2_ROW):
        if key in rows:
            out["rows"][f"N{key[0]}_p{key[1]}"] = row_result(key, level=D.GATE_LEVEL)
    g1 = [out["rows"].get(f"N{n}_p{p}", {}).get("reading", {}).get("passed") for n, p in G1_ROWS]
    g2 = out["rows"].get("N2048_p2", {}).get("reading", {}).get("passed")
    out["gates"]["G1"] = dict(passed=bool(all(x is True for x in g1)), per_row=g1)
    out["gates"]["G2"] = dict(passed=bool(g2 is True))
    gates_ok = all(out["gates"][g]["passed"] for g in ("G0", "G1", "G2"))
    out["gates"]["all_passed"] = gates_ok

    if not gates_ok:
        out["T"] = "not interpreted: a gate failed or a row is missing (stop rule, Sec. 6)"
    else:
        t = {}
        for key in T_ROWS:
            if key not in rows:
                continue
            res = row_result(key)
            pr = pred[key]
            denom = pr["A_p"] * pr["c_CW"]           # c_pred for p >= 3
            est = res["estimate"]
            res["R"] = dict(R=est["c"] / denom, lo=est["lo"] / denom, hi=est["hi"] / denom,
                            p_over_N_gamma=key[1] / (key[0] * D.GAMMA))
            res["R"]["contains_1"] = bool(res["R"]["lo"] <= 1 <= res["R"]["hi"])
            res["R"]["below_1"] = bool(res["R"]["hi"] < 1)
            res["R"]["above_1"] = bool(res["R"]["lo"] > 1)
            t[f"N{key[0]}_p{key[1]}"] = res
            out["rows"][f"N{key[0]}_p{key[1]}"] = res
        rs = [v["R"] for v in t.values()]
        rho = (float(spearmanr([x["p_over_N_gamma"] for x in rs], [1 - x["R"] for x in rs])[0])
               if len(rs) >= 3 else None)
        if rs and all(x["contains_1"] for x in rs):
            rd = "(a)"
        elif any(x["above_1"] for x in rs):
            rd = "(c): an interval lies above 1"
        else:
            rd = "(b) or (c): decide from the ordering of the deficits; see spearman_rho"
        out["T"] = dict(reading=rd, spearman_rho_deficit_vs_p_over_N_gamma=rho,
                        rows_present=list(t), primary="N2048_p3")
        if (1024, 3) in rows and (4096, 3) in rows:
            a = counts_for_estimator(rows[(1024, 3)]); b = counts_for_estimator(rows[(4096, 3)])
            la, lb = pred[(1024, 3)]["L_eq"], pred[(4096, 3)]["L_eq"]
            ca = logit(a[..., 1].sum() / a[..., 0].sum()) / la
            cb = logit(b[..., 1].sum() / b[..., 0].sum()) / lb
            draws = np.log2((boot_logit(a, seed=D.BOOT_SEED) / la)
                            / (boot_logit(b, seed=D.BOOT_SEED + 1) / lb)) / 2
            lo, hi = np.percentile(draws, [2.5, 97.5])
            out["size_exponent_p3"] = dict(value=float(np.log2(ca / cb) / 2), lo=float(lo),
                                           hi=float(hi), exact_CW=0.506)
    out["provenance"] = provenance()
    _dump("analysis.json" if not seed_offset else f"analysis_rerun{seed_offset}.json", out)
    print(json.dumps(dict(gates=out["gates"], T=out.get("T")), indent=1))
    for name, r in out["rows"].items():
        e = r["estimate"]
        extra = f"  R={r['R']['R']:.4f} [{r['R']['lo']:.4f},{r['R']['hi']:.4f}]" if "R" in r else ""
        print(f"{name}: c={e['c']:.5f} [{e['lo']:.5f},{e['hi']:.5f}]  pred={r['c_pred']:.5f}  "
              f"ratio={r['reading']['ratio']:.4f}{extra}")
    return out
