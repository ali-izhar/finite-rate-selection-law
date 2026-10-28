"""Random-start controls for the MNIST illustration (docs/mnist-illustration.md; needs raw MNIST).

Arm A (reheat, as in the figure): 24 runs start from training 8s, 300 sweeps at
beta = 2.0, reheat to 0.55 for 100 sweeps with no bias, final cooling
0.55 -> 2.0 over 200 sweeps under the bias toward 3, 100 settling sweeps.
Arm B (control): 24 runs start from random pixels (each on with probability 1/2)
and run only the biased cooling and settling (300 sweeps).
Arm C (control, matched length): as B, after 400 unbiased sweeps at beta = 0.55,
so the total sweep count matches arm A.

Reports how many runs the linear classifier (models/) labels 3 at the end of each
arm, over ten seeds -> results/experiments/dbm_control_arm.json.

    python -m src.experiments.dbm_control_arm
"""
import sys, json, time
import numpy as np
sys.path.insert(0, '.')
from src.dbm import DBM, MODEL, load_mnist_binary, load_probe

Xb, y = load_mnist_binary()
d = np.load(MODEL)
coef, inter = load_probe()
label = lambda v: (v @ coef.T + inter).argmax(1)
field3 = Xb[y == 3].mean(0) - Xb.mean(0)
n, BC, BR, G = 24, 2.0, 0.55, 2.0
S_COMMIT, S_REHEAT, S_DESC, S_SETTLE = 300, 100, 200, 100
m = DBM(d['W1'], d['W2'], d['b'], d['c1'], d['c2'])
mt = DBM(d['W1'], d['W2'], d['b'] + G * field3, d['c1'], d['c2'])

def descend_settle(v, h1, h2, rng):
    for bb in np.linspace(BR, BC, S_DESC):
        v, h1, h2 = mt.step(v, h1, h2, float(bb), rng)
    for _ in range(S_SETTLE):
        v, h1, h2 = mt.step(v, h1, h2, BC, rng)
    return v, h1, h2

def arm_repair(seed):
    rng = np.random.default_rng(seed)
    sel = rng.choice(np.where(y == 8)[0], n, replace=False)
    v = Xb[sel].copy(); h1, h2 = m._init_hidden(v, BC, rng)
    for _ in range(S_COMMIT): v, h1, h2 = m.step(v, h1, h2, BC, rng)
    lb = label(v)
    for _ in range(S_REHEAT): v, h1, h2 = m.step(v, h1, h2, BR, rng)
    v, h1, h2 = descend_settle(v, h1, h2, rng)
    la = label(v)
    return lb, la

def arm_fresh(seed, pre_sweeps):
    rng = np.random.default_rng(seed)
    v = (rng.random((n, 784)) < 0.5).astype(float); h1, h2 = m._init_hidden(v, BR, rng)
    for _ in range(pre_sweeps): v, h1, h2 = m.step(v, h1, h2, BR, rng)
    v, h1, h2 = descend_settle(v, h1, h2, rng)
    return label(v)

out = {'seeds': [], 'repair': [], 'fresh': [], 'fresh_matched': [], 'protocol': dict(
    beta_commit=BC, beta_reheat=BR, tilt_g=G, sweeps=[S_COMMIT, S_REHEAT, S_DESC, S_SETTLE], n_chains=n)}
seeds = [int(s) for s in (sys.argv[1].split(',') if len(sys.argv) > 1 else [str(i) for i in range(10)])]
for s in seeds:
    t0 = time.time()
    lb, la = arm_repair(s)
    rep = int(((lb == 8) & (la == 3)).sum()); end3 = int((la == 3).sum())
    lf = arm_fresh(s, 0); lfm = arm_fresh(s, S_COMMIT + S_REHEAT)
    out['seeds'].append(s)
    out['repair'].append(dict(committed_8=int((lb == 8).sum()), repaired_8_to_3=rep, end_3=end3,
                              end_labels=np.bincount(la, minlength=10).tolist()))
    out['fresh'].append(dict(end_3=int((lf == 3).sum()), end_labels=np.bincount(lf, minlength=10).tolist()))
    out['fresh_matched'].append(dict(end_3=int((lfm == 3).sum()), end_labels=np.bincount(lfm, minlength=10).tolist()))
    print(f"seed {s}: repair 8->3 {rep}/{n} (committed 8: {(lb==8).sum()}, end 3: {end3}) | "
          f"fresh end 3: {(lf==3).sum()}/{n} | fresh matched-length end 3: {(lfm==3).sum()}/{n}   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open('results/experiments/dbm_control_arm.json', 'w'), indent=1)
