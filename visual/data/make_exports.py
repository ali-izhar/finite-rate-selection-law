"""Export the paper numbers the visual edition loads into small JSON files in visual/data/.

Run from the repository root (a few seconds):
    python visual/data/make_exports.py
Every value is copied from ../results/ or recomputed with the code in ../src (read only).
fork_trajectories.json is copied from results/figures/ (written by paper/figures/make_fork.py).
visual/data/README.md describes each file.
"""
import json
import shutil
import sys

sys.path.insert(0, '.')
from src.simple import design as D
from src.simple.analyse import counts_for_estimator
from src.simple.estimate import estimate

OUT = 'visual/data/'
ROWS = ((1024, 1), (2048, 1), (4096, 1), (2048, 2), (1024, 3), (2048, 3), (4096, 3), (2048, 4))   # Table 1 order


def load(path):
    with open(path) as f:
        return json.load(f)


def dump(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, separators=(',', ':'))
    print(f"wrote {path}")


# 1. Table 1 and Fig. 2b: the Hopfield test of the written protocol ---------------------------------------
pred = {(x["N"], x["p"]): x for x in load('results/simple/predictions.json')["rows"]}
analysis = load('results/simple/analysis.json')
rows = []
for N, p in ROWS:
    pr = pred[(N, p)]
    e = estimate(counts_for_estimator(load(f'results/simple/row_N{N}_p{p}.json')), pr["L_eq"], level=0.95)
    rows.append({"N": N, "p": p, "prediction": pr["prediction"], "A_p": pr["A_p"], "c_CW": pr["c_CW"],
                 "c_pred": pr["c_pred"], "c": e["c"], "lo": e["lo"], "hi": e["hi"],
                 "ratio": e["c"] / pr["c_pred"], "ratio_lo": e["lo"] / pr["c_pred"], "ratio_hi": e["hi"] / pr["c_pred"],
                 "p_over_N_gamma": p / (N * D.GAMMA)})
x = analysis["size_exponent_p3"]
dump(OUT + 'table1.json', {
    "what": "Hopfield test of the written protocol on one linear ramp: predicted c, measured c with its 95% interval "
            "(cluster bootstrap over the 50 pattern sets), and their ratio, for the eight rows",
    "source": "results/simple/predictions.json, results/simple/row_N*_p*.json via src.simple.estimate.estimate "
              "(95%, 10,000 resamples, seed 7); size exponent from results/simple/analysis.json",
    "paper": "Table 1, Fig. 2b, Sec. 4 and 5",
    "design": {"beta0": D.BETA0, "beta_final": D.BETA_F, "gamma": D.GAMMA, "NG": D.NG, "pattern_sets": D.N_SETS},
    "rows": rows,
    "size_exponent_p3": {"value": x["value"], "lo": x["lo"], "hi": x["hi"]},
})

# 2. The nine linear ramps of the exact chain (Table 2, Fig. 2a) ------------------------------------
dump(OUT + 'cw_linear.json', {
    "what": "exact-chain c on linear ramps (beta 0.7 to 1.6, equilibrium start, N G = 1.2544) against the selection law",
    "source": "results/experiments/cw_linear_sweep.json (python -m src.cw_ramp)",
    "paper": "Table 2, Fig. 2a, Eq. 3 and 4",
    "rows": [dict(row, N2gamma=row["N"] ** 2 * row["gamma"]) for row in load('results/experiments/cw_linear_sweep.json')],
})

# 3. Fork trajectories (paper Fig. 1a): a copy of the paper's input (python paper/figures/make_fork.py) --
shutil.copyfile('results/figures/fork_trajectories.json', OUT + 'fork_trajectories.json')   # the site serves visual/ only
print(f"wrote {OUT}fork_trajectories.json")
