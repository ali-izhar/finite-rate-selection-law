"""Export the paper numbers for the visual edition into small JSON files in visual/data/
(including c_cw.json) and the DBM repair filmstrip into visual/assets/.

Run from the repository root (about 20 s):
    python visual/data/make_exports.py
Every value is either copied from ../results/**
(JSON and dbm_reheat.npz), or recomputed with the code in ../src.
See visual/data/README.md for a per-file description.
"""
import json
import sys

import numpy as np
from PIL import Image
from scipy.special import logit

sys.path.insert(0, '.')
from src.gap import gap_well, gap_full                     # noqa: E402
from src.hopfield import protocol as D                            # noqa: E402
import src.experiments.exact_protocol as E                   # noqa: E402

OUT = 'visual/data/'


def r(x, sig=6):
    """Round to `sig` significant digits for compact JSON."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, sig) for v in x]
    x = float(x)
    if x == 0 or not np.isfinite(x):
        return x
    return float(f"{x:.{sig}g}")


def dump(name, obj):
    with open(OUT + name, 'w') as f:
        json.dump(obj, f, separators=(',', ':'))
    print(f"wrote {OUT + name}")


def load(path):
    return json.load(open(path))


# 0. c_CW = 0.120 at N = 2048 on the paper's schedule (Sec. 4, Table 1) -----
cw = {"N": D.N_SPINS, "G": D.G0, "ladder": list(D.LADDER), "sweeps_per_level": list(D.K_REDESCENT),
      "scarce_sweeps_per_level": D.K_SCARCE, "beta_reheat": D.BETA_REHEAT, "reheat_sweeps": D.K_REHEAT,
      "beta_final": D.BETA_DEEP, "runs": []}
for dt in (0.02, 0.01):
    f, q, _ = E.protocol(D.G0, dt=dt)
    qe = E.q_eq(D.G0)
    cw["runs"].append(dict(dt=dt, f_mis=float(f), q_chain=float(q), q_eq=float(qe),
                           c_cw=float(logit(q) / logit(qe))))
cw["c_cw"] = cw["runs"][-1]["c_cw"]
cw["source"] = "src.experiments.exact_protocol.protocol(G0, dt) and .q_eq(G0); logit(q_chain)/logit(q_eq)"
cw["paper"] = "Sec. 4 (c_CW = 0.120 at N = 2048), Table 1 caption"
cw["command"] = "python visual/data/make_exports.py (from the repository root)"
json.dump(cw, open(OUT + 'c_cw.json', 'w'), indent=1)
print(f"wrote {OUT}c_cw.json  c_cw = {cw['c_cw']:.6f}")

# 1. tracking-time curves (docs/relaxation-times.md) --------------------------------------------------
fig1 = load('results/figures/mechanism.json')
BETAS = np.linspace(0.70, 1.35, 360)                       # the beta grid of paper Fig. 1a
curves = {}
for n in (256, 1024, 4096, 16384):
    curves[str(n)] = r([1.0 / gap_well(n, float(b)) for b in BETAS], 6)
dump('tracking_time.json', {
    "what": "tracking time tau(beta) = 1/gap_well(N, beta, h=0), sweeps; chain reflected at m=0",
    "source": "src.gap.gap_well on the beta grid of paper Fig. 1a",
    "paper": "docs/relaxation-times.md (peak 76.94 sweeps at beta = 1.012, N = 16384)",
    "h": 0.0, "beta": r(BETAS, 7), "tau": curves,
    "peak_scan": {k: fig1["peak_scan"][k] for k in ("N", "peaks", "peak_betas", "loglog_slope")},
    "peak_N16384": {"beta": fig1["cost_curves"]["peak_beta_N16384"], "tau": fig1["cost_curves"]["peak_sweeps_N16384"]},
})

# 2. fork trajectories (paper Fig. 1a), NumPy seed 4 --------
N_IMG, N_TRAJ, TRAJ_SEED, TRAJ_DT, SPB = 1024, 14, 4, 0.02, 6.0
rng = np.random.default_rng(TRAJ_SEED)
m = np.zeros(N_TRAJ)
traj = np.zeros((N_TRAJ, BETAS.size))
for j, b in enumerate(BETAS):
    for _ in range(int(SPB / TRAJ_DT)):
        m += (-m + np.tanh(b * m)) * TRAJ_DT + np.sqrt(2 * TRAJ_DT / N_IMG) * rng.standard_normal(N_TRAJ)
    traj[:, j] = m
assert int((traj[:, -1] > 0).sum()) == fig1["trajectories"]["ended_up"], "trajectory recompute drifted from Fig. 1"
dump('fork_trajectories.json', {
    "what": "14 runs of dm = (-m + tanh(beta m)) dt + sqrt(2 dt / N) dW, N=1024, 6 sweeps per beta step, dt=0.02, seed 4",
    "source": "drift-plus-noise paths, NumPy seed 4 (paper Table 5); 9 end at m > 0",
    "paper": "Fig. 1a",
    "N": N_IMG, "beta": r(BETAS, 7), "m": [r(t, 4) for t in traj],
    "ended_up": int((traj[:, -1] > 0).sum()),
})

# 3. switching (mean first-passage) curves (visual edition) ---------------------------------------
A = load('results/figures/switching_times.json')
dump('switching_times.json', {
    "what": "log10 of the exact mean first-passage time (sweeps) from the wrong mode to the right one, h=0.05",
    "source": "results/figures/switching_times.json (src.barrier.log_mfpt)",
    "paper": "not in the paper (visual edition only)",
    "h": A["h"], "beta": r(A["betas_plotted"], 7),
    "log10_T": {k: r(v, 6) for k, v in A["curves_log10_sweeps"].items()},
    "log10_T_at_beta2": A["digits_at_beta_hi_exact"],
    "spinodal_beta": A["onset_curve_drawable_per_N"]["16384"],
})

# 4. Figure 2b points ---------------------------------------------------------------------
B = load('results/figures/field_grid_points.json')
pts = [{"k": p["k"], "G": p["field_G"], "n_retained": p["n_retained"],
        "q_eq": p["q_eq"], "q_chain": p["q_model"], "logit_q_eq": p["logit_q_eq"],
        "logit_q_chain": p["logit_q_model"]} for p in B["points"]]
rr = B["recovery_ratio_plotted"]
dump('field_grid_points.json', {
    "what": "Hopfield N=2048, p=3: logit q_chain (reheated chains) vs logit q_eq at 8 bias values; 20 pattern draws x 2000 chains",
    "source": "results/figures/field_grid_points.json (results/hopfield/field_grid_analysis.json)",
    "paper": "Fig. 2b; Sec. 4 (c = 0.199 [0.1908, 0.2095])",
    "points": pts,
    "c_measured": {"point": rr["point"], "lower": rr["lower"], "upper": rr["upper"]},
    "c_predicted": B["prediction_drawn"]["c_pred"],
    "c_CW": next(x["c_cw"] for x in cw["runs"] if x["dt"] == 0.02),
    "intervals": "q_chain: Wilson 95%; q_eq: the error envelope of App. B",
})

# 5. Table 1 ------------------------------------------------------------------------------
DL = load('results/experiments/draw_level_analysis.json')
rows = []
for p in ("1", "2", "3", "4"):
    v = DL[p]
    rows.append({"p": int(p), "amplification": v["amplification"], "c_predicted": v["c_pred"],
                 "c_measured": v["c_pooled"], "ci95": v["draw_boot"], "n_chains": v["n"]})
dump('table1.json', {
    "what": "Table 1: predicted c = c_CW x A_p vs measured c (Hopfield N=2048, 12 pattern draws x 8000 chains per sign of G)",
    "source": "results/experiments/draw_level_analysis.json (CI: 20,000-resample bootstrap over draws)",
    "paper": "Table 1", "c_CW": round(load('visual/data/c_cw.json')["c_cw"], 4), "rows": rows,
})

# 6. size scan ----------------------------------------------------------------------------
SS = load('results/experiments/size_scan_p3_analysis.json')
dump('size_scan.json', {
    "what": "p=3 size scan: measured c (draw bootstrap 95%) vs predicted c = c_CW(N) x 1.662",
    "source": "results/experiments/size_scan_p3_analysis.json",
    "paper": "Table 1; Sec. 5",
    "rows": [{"N": int(n), "c_measured": v["c_meas"], "ci95": v["draw_boot"], "c_predicted": v["c_pred"],
              "c_CW": v["c_cw"], "q_eq": v["q_eq"], "n_chains": v["n"]} for n, v in SS.items()],
})

# 7. sampler clock ------------------------------------------------------------------------
SC = load('results/experiments/sampler_clock.json')
dump('sampler_clock.json', {
    "what": "r(N) = c_measured(p=1, permutation-sweep sampler) / c_CW(N, continuous-time exact chain)",
    "source": "results/experiments/sampler_clock.json",
    "paper": "Sec. 5, App. B (1.063 at N=2048; p=3 prediction x r = 0.212)",
    "rows": [dict(N=int(n), **v) for n, v in SC.items()],
    "p3_clock_corrected_prediction_N2048": 0.120 * 1.662 * 1.063,
})

# 8. c scaling on the exact chain ---------------------------------------------------------
CS = load('results/experiments/cw_scaling.json')
LS = load('results/experiments/cw_linear_sweep.json')
dump('cw_scaling.json', {
    "what": "c on the exact chain: (a) paper schedule at N=512..8192 (G = G0*2048/N) and at budget scales 0.25..8 (N=2048); (b) linear sweeps vs the KN law",
    "source": "results/experiments/cw_scaling.json, results/experiments/cw_linear_sweep.json",
    "paper": "Eq. 4; Table 2 (linear ramps); Table 3 caption (schedule scale)",
    "schedule": CS,
    "linear_sweep": [dict(row, N2gamma=row["N"] ** 2 * row["gamma"]) for row in LS],
})

# 9. schedule ----------------------------------------------------------------------------
tau = [1.0 / (gap_full(2048, b) if b <= 1.0 else gap_well(2048, b)) for b in D.LADDER]
K = [int(np.ceil(6 * t)) for t in tau]
assert K == list(D.K_REDESCENT) and sum(K) == 834
ccw = load('visual/data/c_cw.json')
dump('schedule.json', {
    "what": "the paper's cooling schedule at N=2048",
    "source": "src/hopfield/protocol.py; tau recomputed with src.gap (1/gap_full for beta<=1, 1/gap_well above)",
    "paper": "Sec. 4; Table 3",
    "N": 2048, "ladder": list(D.LADDER), "tau": tau, "K": K, "K_total": sum(K),
    "beta_reheat": D.BETA_REHEAT, "reheat_sweeps": D.K_REHEAT, "beta_final": D.BETA_DEEP,
    "G0": D.G0, "c_CW": ccw["c_cw"],
})

# 10. MNIST illustration filmstrip ----------------------------------------------------------------
Z = np.load('results/experiments/dbm_reheat.npz')
frames = Z['frames']                                   # (24, 141, 784) uint8 in {0,1}
nC, nF, _ = frames.shape
sheet = frames.reshape(nC, nF, 28, 28).transpose(0, 2, 1, 3).reshape(nC * 28, nF * 28)
img = Image.fromarray((sheet * 255).astype(np.uint8), mode='L').convert('1')
img.save('visual/assets/mnist_illustration_frames.png', optimize=True)
print("wrote visual/assets/mnist_illustration_frames.png", img.size)
probs = Z['probs']                                     # (24, 701, 10)
betas = Z['betas']
step = 2
ctrl = load('results/experiments/dbm_control_arm.json')
proto = json.loads(str(Z['protocol']))
dump('mnist_illustration.json', {
    "what": "2-layer MNIST DBM: 24 chains started from '8' images, committed at beta=2 (300 sweeps), reheated to beta=0.55 (100), re-descended under a bias toward '3' (200 + 100)",
    "source": "results/experiments/dbm_reheat.npz (src/experiments/dbm_reheat.py); control arm results/experiments/dbm_control_arm.json",
    "paper": "docs/mnist-illustration.md (20-24 of 24 end as '3' over ten seeds; random-start controls 21-24)",
    "illustrative": True,
    "protocol": proto,
    "sprite": {"file": "assets/mnist_illustration_frames.png", "tile": 28, "rows_are": "chains (24)", "cols_are": "snapshots (141)",
               "pixel": "white = 1 (on), black = 0"},
    "snap_sweep": Z['snap_sweep'].tolist(),
    "beta_per_sweep": r(betas, 5),
    "probe_sweep_step": step,
    "p3": [r(probs[c, ::step, 3], 3) for c in range(nC)],
    "p8": [r(probs[c, ::step, 8], 3) for c in range(nC)],
    "label_at_snap": [[int(probs[c, s].argmax()) for s in Z['snap_sweep']] for c in range(nC)],
    "label_after": Z['label_all_after'].tolist(),
    "beta_star": 0.9333,
    "control_arm": ctrl,
})
