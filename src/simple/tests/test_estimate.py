"""(c) The one estimator (src.simple.estimate) on synthetic counts.

1. Arithmetic: c-hat = logit(pooled q-hat) / L_eq on a hand-built table.
2. Property tests (Hypothesis, 1000 examples): invariance under set permutation, under swapping
   the sign axis, under scaling every count by an integer; lo <= hi; reading() passes iff the
   interval holds the prediction.
3. Recovery and coverage: K = 50 sets with a between-set spread of the logit, binomial counts at
   the design's scale; over 400 synthetic experiments the bootstrap 95% interval covers the
   population value in [0.91, 0.98] of them (binomial sd of the coverage at 0.95 is 0.011) and the
   mean of c-hat is within 1% of it.
"""
from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose
from scipy.special import expit, logit

from src.simple.estimate import c_hat, estimate, reading

L_EQ = 3.574685


def test_arithmetic_on_a_hand_table():
    counts = np.zeros((3, 2, 2))
    counts[:, 0] = [[100, 60], [200, 110], [50, 30]]
    counts[:, 1] = [[120, 70], [80, 40], [90, 50]]
    q = (60 + 110 + 30 + 70 + 40 + 50) / (100 + 200 + 50 + 120 + 80 + 90)
    assert_allclose(c_hat(counts, L_EQ), logit(q) / L_EQ, rtol=1e-15)
    est = estimate(counts, L_EQ, n_boot=500)
    assert_allclose(est["q_plus"], 200 / 350, rtol=1e-15)
    assert_allclose(est["q_minus"], 160 / 290, rtol=1e-15)
    assert est["n_pattern1"] == 640


def _tables():
    """(K, 2, 2) tables with 1 <= correct <= n - 1 (finite logit in every resample)."""
    n = arrays(np.int64, st.tuples(st.integers(2, 60), st.just(2)), elements=st.integers(2, 10**6))
    frac = arrays(np.float64, st.just((60, 2)), elements=st.floats(0.01, 0.99))

    @st.composite
    def build(draw):
        nn = draw(n)
        ff = draw(frac)[: nn.shape[0]]
        cor = np.clip(np.round(ff * nn), 1, nn - 1)
        return np.stack([nn, cor], axis=-1).astype(np.int64)
    return build()


@settings(max_examples=1000, deadline=None, database=None)
@given(_tables(), st.integers(0, 2**31 - 1), st.integers(2, 7))
def test_invariances(counts, perm_seed, scale):
    c0 = c_hat(counts, L_EQ)
    perm = np.random.default_rng(perm_seed).permutation(counts.shape[0])
    assert_allclose(c_hat(counts[perm], L_EQ), c0, rtol=1e-12)
    assert_allclose(c_hat(counts[:, ::-1], L_EQ), c0, rtol=1e-12)
    assert_allclose(c_hat(counts * scale, L_EQ), c0, rtol=1e-12)
    est = estimate(counts, L_EQ, n_boot=200, seed=perm_seed)
    assert est["lo"] <= est["hi"]
    if est["lo"] == 0 or est["hi"] == 0:        # reading() divides by the prediction; c_pred > 0
        return
    assert reading(est, est["lo"])["passed"] and reading(est, est["hi"])["passed"]
    if est["hi"] > est["lo"]:
        assert not reading(est, est["hi"] + 1e-9 * (1 + abs(est["hi"])))["passed"]


def test_recovery_and_bootstrap_coverage():
    rng = np.random.default_rng(11)
    k, per_cell = 50, 1000
    c_true_mean, sd_rel = 0.11, 0.08            # the (2048, 3) scale; 2x the assumed disorder
    pop = rng.normal(c_true_mean * L_EQ, sd_rel * c_true_mean * L_EQ, size=2_000_000)
    c_pop = logit(expit(pop).mean()) / L_EQ     # the estimand: logit of the population mean q
    reps, cover, chats = 400, 0, []
    for _ in range(reps):
        lg = rng.normal(c_true_mean * L_EQ, sd_rel * c_true_mean * L_EQ, size=k)
        n1 = rng.binomial(per_cell, 1 / 3, size=(k, 2))            # pattern 1 wins ~1/3
        cor = rng.binomial(n1, expit(lg)[:, None])
        est = estimate(np.stack([n1, cor], axis=-1), L_EQ, n_boot=2000,
                       seed=int(rng.integers(2**31)))
        cover += est["lo"] <= c_pop <= est["hi"]
        chats.append(est["c"])
    assert 0.91 <= cover / reps <= 0.98, cover / reps
    assert abs(np.mean(chats) / c_pop - 1) < 0.01, np.mean(chats) / c_pop
