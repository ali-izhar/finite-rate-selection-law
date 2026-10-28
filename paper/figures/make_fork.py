"""Input of Fig. 1a: 14 drift-plus-noise paths of the magnetization, N = 1024, NumPy seed 4.

Run from the repository root (CPU, seconds):
    python paper/figures/make_fork.py [--out FILE]
The default --out is results/figures/fork_trajectories.json, which make_figures.py reads.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

OUT = "results/figures/fork_trajectories.json"
BETAS = np.linspace(0.70, 1.35, 360)
N_IMG, N_TRAJ, TRAJ_SEED, TRAJ_DT, SPB = 1024, 14, 4, 0.02, 6.0


def r(x, sig=6):
    """Round to `sig` significant digits for compact JSON."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, sig) for v in x]
    x = float(x)
    return x if x == 0 or not np.isfinite(x) else float(f"{x:.{sig}g}")


def fork_trajectories():
    rng = np.random.default_rng(TRAJ_SEED)
    m = np.zeros(N_TRAJ)
    traj = np.zeros((N_TRAJ, BETAS.size))
    for j, b in enumerate(BETAS):
        for _ in range(int(SPB / TRAJ_DT)):
            m += (-m + np.tanh(b * m)) * TRAJ_DT + np.sqrt(2 * TRAJ_DT / N_IMG) * rng.standard_normal(N_TRAJ)
        traj[:, j] = m
    assert int((traj[:, -1] > 0).sum()) == 9, "trajectory recompute drifted from Fig. 1 (9 of 14 paths end at m > 0)"
    return {
        "what": "14 runs of dm = (-m + tanh(beta m)) dt + sqrt(2 dt / N) dW, N=1024, 6 sweeps per beta step, dt=0.02, seed 4",
        "source": "drift-plus-noise paths, NumPy seed 4 (paper Fig. 1a, App. A); 9 end at m > 0",
        "paper": "Fig. 1a",
        "N": N_IMG, "beta": r(BETAS, 7), "m": [r(t, 4) for t in traj],
        "ended_up": int((traj[:, -1] > 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT, help=f"output JSON (default {OUT})")
    args = parser.parse_args()
    with open(args.out, "w") as f:
        json.dump(fork_trajectories(), f, separators=(",", ":"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
