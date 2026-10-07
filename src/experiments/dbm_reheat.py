"""The 24-run MNIST reheating ensemble of docs/mnist-illustration.md (needs raw MNIST).

An illustration, not evidence for the paper's results.  Protocol:
  300 sweeps at beta = 2.0 from 24 training images of digit 8;
  reheat: 100 sweeps at beta = 0.55, no bias;
  final cooling 0.55 -> 2.0 over 200 sweeps under a visible-bias field toward 3,
  then 100 settling sweeps.
The field acts during the final cooling only, as G does in the paper's protocol.

The saved results/experiments/dbm_reheat.npz came from an earlier implementation
of the sampler; src.dbm uses the same weights but a different update order and
random stream, so a rerun reproduces the ensemble only statistically.  The file keeps:

  frames    the visible units every SNAP_EVERY sweeps, all 24 runs
  probs     the linear classifier's (models/) class probabilities at every sweep
  betas     the schedule, per sweep

    python -m src.experiments.dbm_reheat
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for src.dbm
from src.dbm import DBM, MODEL, load_mnist_binary, load_probe  # noqa: E402
OUT = Path(__file__).resolve().parents[2] / 'results/experiments/dbm_reheat.npz'


def main():
    Xb, y = load_mnist_binary()
    print(f'MNIST {Xb.shape}, labels {y.shape}')

    d = np.load(MODEL)
    m = DBM(d['W1'], d['W2'], d['b'], d['c1'], d['c2'])
    coef, inter = load_probe()

    def post(v):
        """The linear probe's class posterior on the visible sample."""
        z = v @ coef.T + inter
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    label = lambda v: (v @ coef.T + inter).argmax(1)

    mubar = Xb.mean(0)
    field3 = Xb[y == 3].mean(0) - mubar
    rng = np.random.default_rng(0)
    n, BC, BR, G = 24, 2.0, 0.55, 2.0
    S_COMMIT, S_REHEAT, S_DESC, S_SETTLE = 300, 100, 200, 100
    SNAP_EVERY = 5
    sel = rng.choice(np.where(y == 8)[0], n, replace=False)
    v = Xb[sel].copy()

    mt = DBM(d['W1'], d['W2'], d['b'] + G * field3, d['c1'], d['c2'])

    frames, snap_sweep, probs, betas = [], [], [], []

    def record(v_, beta, t):
        """No RNG is drawn here -- the chain is unaffected."""
        probs.append(post(v_).astype(np.float32))
        betas.append(beta)
        if t % SNAP_EVERY == 0:
            frames.append((v_ > 0.5).astype(np.uint8))
            snap_sweep.append(t)

    h1, h2 = m._init_hidden(v, BC, rng)
    t = 0
    record(v, BC, t)                       # t = 0, the real MNIST 8 seed
    for _ in range(S_COMMIT):
        v, h1, h2 = m.step(v, h1, h2, BC, rng); t += 1
        record(v, BC, t)
    before = v.copy(); lb = label(before)

    for _ in range(S_REHEAT):                          # reheat, NO tilt
        v, h1, h2 = m.step(v, h1, h2, BR, rng); t += 1
        record(v, BR, t)

    for bb in np.linspace(BR, BC, S_DESC):             # re-descend, TILTED to 3
        v, h1, h2 = mt.step(v, h1, h2, float(bb), rng); t += 1
        record(v, float(bb), t)

    for _ in range(S_SETTLE):
        v, h1, h2 = mt.step(v, h1, h2, BC, rng); t += 1
        record(v, BC, t)
    after = v.copy(); la = label(after)

    frames = np.stack(frames, 1)                       # (n_chain, n_snap, 784)
    probs = np.stack(probs, 1)                         # (n_chain, n_sweep+1, 10)
    snap_sweep = np.asarray(snap_sweep, np.int32)
    betas = np.asarray(betas, np.float64)

    ok = np.array([i for i in range(n) if lb[i] == 8 and la[i] == 3])
    bad = np.array([i for i in range(n) if not (lb[i] == 8 and la[i] == 3)])
    print(f'committed to 8 then recommitted to 3: {len(ok)}/{n}')
    print(f'excluded chains {bad.tolist()}: committed to {lb[bad].tolist()}, ended at {la[bad].tolist()}')
    print(f'ALL 24 end labels: {np.bincount(la, minlength=10).tolist()}')

    np.savez_compressed(
        OUT,
        before=before[ok][:8], after=after[ok][:8],           # kept for compatibility
        label_before=lb[ok][:8], label_after=la[ok][:8],
        frames=frames, snap_sweep=snap_sweep, probs=probs, betas=betas,
        label_all_before=lb, label_all_after=la,
        idx_repaired=ok, idx_failed=bad,
        protocol=json.dumps(dict(
            beta_commit=BC, beta_reheat=BR, tilt_g=G, tilt_toward=3,
            sweeps=[S_COMMIT, S_REHEAT, S_DESC, S_SETTLE], snap_every=SNAP_EVERY,
            model='boltzmann_machine', probe='digit_classifier',
            n_chains=n, n_repaired=int(len(ok)), illustrative=True)))
    print(f'saved {OUT}',
          'frames', frames.shape, 'probs', probs.shape)


if __name__ == '__main__':
    main()
