"""(a) The sampler's one-update kernel IS the random-site heat-bath kernel of the Hopfield energy.

Oracle: the energy evaluated from its definition (explicit sum over ordered pairs i != j, coupling
1/(N-1)) on all 2^N states; conditionals from energy differences only.  The sampler side uses the
carried-projection formula (reference.p_plus), the code path of every run.

Tolerance: E is a sum of O(N^2) terms of size 1/(N-1); its rounding error is ~N eps ~ 1e-14, so
the conditionals agree to ~1e-14.  rtol = 1e-12, atol = 1e-14 (p_+ in (1e-8, 1) here).
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.simple import design as D
from src.simple.reference import (ReferenceSampler, all_states, gibbs, heat_bath_conditionals,
                                  kernel_from_conditionals)

CASES = [(8, 1), (8, 2), (8, 3), (9, 4), (10, 2), (10, 3)]


def _xi(n, p, seed):
    rng = np.random.default_rng([seed, n, p])
    return (2 * rng.integers(0, 2, size=(p, n)) - 1).astype(np.int8)


def sampler_conditionals(xi, g, beta):
    """p_+ for every (state, site) through ReferenceSampler, shape (2^N, N)."""
    p, n = xi.shape
    x = all_states(n)
    s = x.shape[0]
    smp = ReferenceSampler(xi[None], np.zeros(s * n, dtype=int), np.full(s * n, g))
    smp.set_state(np.repeat(x, n, axis=0))
    site = np.tile(np.arange(n), s)
    return smp.p_plus_at(site, beta).reshape(s, n)


@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("beta", [0.0, 0.7, 1.15, 1.6, 3.0])
@pytest.mark.parametrize("sign", [+1, -1])
def test_conditionals_equal_heat_bath(n, p, beta, sign):
    xi = _xi(n, p, 1)
    g = sign * D.field(n)
    assert_allclose(sampler_conditionals(xi, g, beta), heat_bath_conditionals(xi, g, beta),
                    rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("n,p", CASES)
def test_kernel_is_reversible_for_gibbs(n, p):
    """K from the sampler's conditionals: rows sum to 1, pi K = pi, pi_x K_xy = pi_y K_yx."""
    xi = _xi(n, p, 2)
    g, beta = D.field(n), 1.3
    k = kernel_from_conditionals(sampler_conditionals(xi, g, beta))
    pi = gibbs(xi, g, beta)
    assert_allclose(k.sum(axis=1), 1.0, rtol=0, atol=1e-14)
    assert_allclose(pi @ k, pi, rtol=1e-12, atol=1e-16)
    flow = pi[:, None] * k
    assert_allclose(flow, flow.T, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("n,p", [(8, 3), (10, 2)])
def test_update_moves_one_spin_and_keeps_projections_exact(n, p):
    """An update changes only the chosen spin, to +1 iff u < p_+; the carried c stays exact."""
    rng = np.random.default_rng(3)
    xi = np.stack([_xi(n, p, s) for s in range(4)])
    b = 500
    smp = ReferenceSampler(xi, rng.integers(0, 4, b), rng.choice([-1, 1], b) * D.field(n))
    smp.start(rng)
    for t in range(300):
        before = smp.state.copy()
        site = rng.integers(0, n, b)
        u = rng.random(b)
        beta = 0.5 + t / 100
        pp = smp.p_plus_at(site, beta)
        smp.update(site, u, beta)
        changed = before != smp.state
        changed[np.arange(b), site] = False
        assert not changed.any()
        assert np.array_equal(smp.state[np.arange(b), site], np.where(u < pp, 1, -1))
    assert np.array_equal(smp.proj, smp.recompute_projections())
