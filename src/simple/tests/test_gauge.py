"""(b) p = 1 gauge equivalence: the Hopfield sampler's sign law equals the exact CW chain.

Two checks against src.simple.predict.ramp_laws (the exact birth-death chain of the ramp; the
frozen predictions come from it):

1. exact, N = 8: the full 2^8-state Markov chain of the sampler (kernel built from the sampler's own
   conditionals, update by update over the whole design ramp, independent unbiased start) gives
   P(m^1 > 0), P(m^1 < 0) equal to the birth-death chain to rtol 1e-10 (two propagations of ~1800
   stochastic maps; each step loses ~eps, so ~1e-13 is expected).
2. Monte Carlo, N = 64: the NumPy reference sampler run over the design ramp with random patterns
   and both field signs; the pooled q = P(sign m^1 = sign G | m^1 != 0) agrees with the exact q to
   |z| < 4 (pre-set; false-alarm probability 6e-5).  The torch sampler has the same test on the
   GPU in test_gpu.py.
"""
from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from src.simple import design as D
from src.simple.predict import ramp_laws, sign_probs
from src.simple.reference import ReferenceSampler, all_states, energy, kernel_from_conditionals


def exact_q(n):
    qp, qm = sign_probs(ramp_laws([n], [1.0 / (n - 1)], n, D.field(n))[0], n)
    return qp, qm


def test_full_state_chain_equals_birth_death_chain():
    n = 8
    xi = np.array([[1, -1, -1, 1, 1, -1, 1, -1]], dtype=np.int8)      # not the gauge-trivial all-ones
    g = D.field(n)
    x = all_states(n)
    de = np.empty((2**n, n))                                         # E(x_i=-1) - E(x_i=+1)
    e = energy(x, xi, g)
    idx = np.arange(2**n)
    for i in range(n):
        de[:, i] = e[idx & ~(1 << i)] - e[idx | (1 << i)]
    law = np.full(2**n, 2.0**-n)                                     # independent unbiased start
    for beta in D.beta_of_update(np.arange(D.n_updates(n)), n):
        law = law @ kernel_from_conditionals(1.0 / (1.0 + np.exp(-beta * de)))
    m1 = x.astype(int) @ xi[0].astype(int)
    qp, qm = exact_q(n)
    assert_allclose([law[m1 > 0].sum(), law[m1 < 0].sum()], [qp, qm], rtol=1e-10)


def test_monte_carlo_sign_probability_matches_exact_chain():
    n, b, k = 64, 40_000, 10
    rng = np.random.default_rng(20261027)
    xi = (2 * rng.integers(0, 2, size=(k, 1, n)) - 1).astype(np.int8)
    c = np.arange(b)
    sign = np.where((c // k) % 2 == 0, 1.0, -1.0)
    smp = ReferenceSampler(xi, c % k, sign * D.field(n))
    smp.start(rng)
    smp.run(0, D.n_updates(n), rng=rng)
    m1 = smp.proj[:, 0]
    keep = m1 != 0
    hit = np.sign(m1[keep]) == sign[keep]
    qp, qm = exact_q(n)
    q = qp / (qp + qm)
    z = (hit.mean() - q) / np.sqrt(q * (1 - q) / keep.sum())
    assert abs(z) < 4, (hit.mean(), q, z)
    # the m^1 = 0 share also matches the exact chain
    p0 = 1 - qp - qm
    z0 = ((~keep).mean() - p0) / np.sqrt(p0 * (1 - p0) / b)
    assert abs(z0) < 4, ((~keep).mean(), p0, z0)
