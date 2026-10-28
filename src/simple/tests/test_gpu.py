"""The torch sampler (src.simple.sampler): skipped where torch is absent (the host venv).

Run in the container (docs/protocol-simple.md):
1. G0: bit-for-bit equal to the NumPy reference on a supplied stream (N = 16, p = 3, 50 updates).
2. (a) on the torch path: p_+ for every (state, site) at N = 8, 10 equals the energy heat bath.
3. Site choice is uniform: chi-square of 2e7 torch.randint draws over N = 64 sites, p > 1e-4.
4. Labels: ties, m^1 = 0 and the winner on hand-built projections.
5. (b) on the torch path: p = 1, N = 256, 100,000 chains over the design ramp; pooled q against
   the exact birth-death chain, |z| < 4 (pre-set).
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

torch = pytest.importorskip("torch")

from scipy.stats import chisquare  # noqa: E402

from src.simple import design as D  # noqa: E402
from src.simple.predict import ramp_laws, sign_probs  # noqa: E402
from src.simple.reference import all_states, heat_bath_conditionals  # noqa: E402
from src.simple.run import g0  # noqa: E402
from src.simple.sampler import Sampler  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_g0_bit_for_bit():
    res = g0()
    assert res["passed"], res
    assert res["spins_changed"] > 100


@pytest.mark.parametrize("n,p", [(8, 1), (8, 3), (10, 2)])
@pytest.mark.parametrize("beta", [0.7, 1.6])
def test_torch_conditionals_equal_heat_bath(n, p, beta):
    rng = np.random.default_rng([n, p])
    xi = (2 * rng.integers(0, 2, size=(p, n)) - 1).astype(np.int8)
    for g in (D.field(n), -D.field(n)):
        x = all_states(n)
        s = x.shape[0]
        smp = Sampler(xi[None], np.zeros(s * n, dtype=int), np.full(s * n, g), device=DEV)
        smp.set_state(np.repeat(x, n, axis=0))
        site = torch.from_numpy(np.tile(np.arange(n), s)).to(smp.device)
        got = smp.p_plus_at(site, beta).cpu().numpy().reshape(s, n)
        assert_allclose(got, heat_bath_conditionals(xi, g, beta), rtol=1e-12, atol=1e-14)


def test_site_choice_uniform():
    gen = torch.Generator(device=DEV); gen.manual_seed(5)
    n = 64
    draws = torch.randint(0, n, (2000, 10_000), generator=gen, dtype=torch.int64, device=DEV)
    freq = torch.bincount(draws.view(-1), minlength=n).cpu().numpy()
    assert chisquare(freq).pvalue > 1e-4


def test_labels():
    xi = np.ones((1, 3, 4), dtype=np.int8)
    smp = Sampler(xi, np.zeros(5, dtype=int), np.array([1.0, -1.0, 1.0, 1.0, -1.0]), device=DEV)
    smp.proj = torch.tensor([[4, 2, 0], [-4, 4, 0], [0, 2, -2], [-2, 0, 0], [-6, 2, 4]],
                            dtype=torch.int64, device=smp.device)
    lab = {k: v.cpu().numpy() for k, v in smp.label_counts().items()}
    assert lab["tie"].tolist() == [False, True, True, False, False]
    assert lab["m1zero"].tolist() == [False, False, True, False, False]
    assert lab["pattern1"].tolist() == [True, False, False, True, True]
    assert lab["correct"].tolist() == [True, False, False, False, True]


def test_p1_sign_probability_matches_exact_chain():
    n, b, k = 256, 100_000, 50
    gen = torch.Generator(device=DEV); gen.manual_seed(12345)
    xi = np.stack([D.patterns(n, s, 1) for s in range(k)])
    c = np.arange(b)
    sign = np.where((c // k) % 2 == 0, 1.0, -1.0)
    smp = Sampler(xi, c % k, sign * D.field(n), device=DEV, generator=gen)
    smp.start()
    smp.run(0, D.n_updates(n))
    assert smp.projection_drift() == 0
    m1 = smp.projections()[:, 0]
    keep = m1 != 0
    hit = np.sign(m1[keep]) == sign[keep]
    qp, qm = sign_probs(ramp_laws([n], [1.0 / (n - 1)], n, D.field(n))[0], n)
    q = qp / (qp + qm)
    z = (hit.mean() - q) / np.sqrt(q * (1 - q) / keep.sum())
    print(f"N={n}: q_hat={hit.mean():.5f} exact={q:.5f} z={z:+.2f}")
    assert abs(z) < 4, (hit.mean(), q, z)
