"""NumPy reference of the simplified sampler, and brute-force oracles at tiny N.

Model (docs/protocol-simple.md Sec. 2), x in {-1,+1}^N, patterns xi^mu, mu = 1..p:

    E(x) = -(1/(2(N-1))) sum_{i != j} sum_mu xi^mu_i xi^mu_j x_i x_j  -  G sum_i xi^1_i x_i

One update: pick site i uniformly at random, set x_i = +1 with probability

    P(x_i = +1 | x_{-i}) = sigmoid(2 beta h_i),   h_i = sum_{j != i} J_ij x_j + G xi^1_i,
    J_ij = (1/(N-1)) sum_mu xi^mu_i xi^mu_j,

which is the heat-bath (Gibbs) conditional of exp(-beta E), since E(x_i=-1) - E(x_i=+1) = 2 h_i.
With carried projections c_mu = xi^mu . x (exact integers) the field is

    h_i = num_i / (N-1) + G xi^1_i,   num_i = sum_mu xi^mu_i c_mu - p x_i   (an integer),

so the only rounding is one division, one product, one sum and the sigmoid, all float64.

``p_plus`` below is the ONE formula both the NumPy reference and the torch sampler evaluate, in
the same operation order, so that the G0 check can compare their states bit for bit.
"""
from __future__ import annotations

import itertools

import numpy as np

from src.simple import design as D


def p_plus(num: np.ndarray, xi1: np.ndarray, beta: float, g: np.ndarray, n_spins: int) -> np.ndarray:
    """sigmoid(2 beta h), h = num/(N-1) + g xi1; float64, in the operation order of the sampler.

    |2 beta h| <= 2 * 1.6 * (p + |G|) < 20 here, so 1/(1+exp(-x)) neither overflows nor loses
    the tail that matters (p_plus near 0 is computed with relative accuracy ~eps; near 1 the
    complement 1 - p_plus is only needed through ``u < p_plus``, absolute accuracy suffices).
    """
    h = num.astype(np.float64) / (n_spins - 1) + g * xi1.astype(np.float64)
    x = h * (2.0 * beta)
    return 1.0 / (1.0 + np.exp(-x))


class ReferenceSampler:
    """B independent chains, vectorised over chains; float64 field, int64 projections.

    xi_sets   (K, p, N) int8   pattern sets
    set_idx   (B,)             which set each chain uses
    g         (B,) float64     the signed field of each chain (+|G| or -|G|)
    """

    def __init__(self, xi_sets: np.ndarray, set_idx: np.ndarray, g: np.ndarray) -> None:
        self.xi_sets = np.asarray(xi_sets, dtype=np.int8)
        self.k_sets, self.p, self.n = self.xi_sets.shape
        self.set_idx = np.asarray(set_idx, dtype=np.int64)
        self.g = np.asarray(g, dtype=np.float64)
        self.b = self.set_idx.size
        # xit[s, i, mu] = xi^mu_i of set s, int64 so the field numerator is exact
        self.xit = np.ascontiguousarray(self.xi_sets.transpose(0, 2, 1)).astype(np.int64)
        self.state = np.ones((self.b, self.n), dtype=np.int8)
        self.proj = np.zeros((self.b, self.p), dtype=np.int64)
        self._rows = np.arange(self.b)

    # ---------------------------------------------------------------- state
    def set_state(self, state: np.ndarray) -> None:
        self.state = np.array(state, dtype=np.int8, copy=True)
        self.proj = self.recompute_projections()

    def start(self, rng: np.random.Generator) -> None:
        """Every spin independent and unbiased."""
        self.set_state(2 * rng.integers(0, 2, size=(self.b, self.n), dtype=np.int8) - 1)

    def recompute_projections(self) -> np.ndarray:
        """c_mu = xi^mu . x from scratch, exact integers, shape (B, p)."""
        xi = self.xi_sets[self.set_idx].astype(np.int64)            # (B, p, N)
        return np.einsum("bmn,bn->bm", xi, self.state.astype(np.int64))

    # --------------------------------------------------------------- update
    def field_numerator(self, site: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(num, xi_i, x_i) at ``site`` (B,): num = sum_mu xi^mu_i c_mu - p x_i."""
        xi_i = self.xit[self.set_idx, site]                          # (B, p) int64
        s_old = self.state[self._rows, site].astype(np.int64)        # (B,)
        num = (xi_i * self.proj).sum(axis=1) - self.p * s_old
        return num, xi_i, s_old

    def p_plus_at(self, site: np.ndarray, beta: float) -> np.ndarray:
        num, xi_i, _ = self.field_numerator(site)
        return p_plus(num, xi_i[:, 0], beta, self.g, self.n)

    def update(self, site: np.ndarray, u: np.ndarray, beta: float) -> None:
        """One heat-bath update per chain at ``site`` with uniform ``u`` (both (B,))."""
        num, xi_i, s_old = self.field_numerator(site)
        pp = p_plus(num, xi_i[:, 0], beta, self.g, self.n)
        s_new = np.where(u < pp, 1, -1).astype(np.int64)
        self.state[self._rows, site] = s_new
        self.proj += xi_i * (s_new - s_old)[:, None]

    def run(self, n0: int, n1: int, *, rng: np.random.Generator | None = None,
            sites: np.ndarray | None = None, unif: np.ndarray | None = None) -> None:
        """Updates n0..n1-1 of the design ramp at size N.

        Either ``rng`` (site ~ Uniform{0..N-1}, u ~ Uniform[0,1) per chain and update) or the
        supplied streams ``sites``/``unif`` of shape (n1-n0, B).
        """
        betas = D.beta_of_update(np.arange(n0, n1), self.n)
        for t, beta in enumerate(betas):
            if sites is None:
                site = rng.integers(0, self.n, size=self.b)
                u = rng.random(self.b)
            else:
                site, u = sites[t].astype(np.int64), unif[t]
            self.update(site, u, float(beta))


# ------------------------------------------------------------------------ brute-force oracles
def all_states(n_spins: int) -> np.ndarray:
    """Every x in {-1,+1}^N, shape (2^N, N) int8; row r has bit i of r at spin i (1 -> +1)."""
    r = np.arange(2**n_spins)[:, None]
    return (2 * ((r >> np.arange(n_spins)[None, :]) & 1) - 1).astype(np.int8)


def energy(x: np.ndarray, xi: np.ndarray, g: float) -> np.ndarray:
    """E(x) straight from the definition: explicit pair sum over i != j, shape (S,)."""
    p, n = xi.shape
    xf = x.astype(np.float64)
    e = np.zeros(x.shape[0])
    for i, j in itertools.permutations(range(n), 2):               # ordered pairs, i != j
        jij = sum(float(xi[mu, i]) * float(xi[mu, j]) for mu in range(p)) / (n - 1)
        e -= 0.5 * jij * xf[:, i] * xf[:, j]
    e -= g * (xf @ xi[0].astype(np.float64))
    return e


def heat_bath_conditionals(xi: np.ndarray, g: float, beta: float) -> np.ndarray:
    """P(x_i = +1 | x_{-i}) for every state and site from energies only, shape (2^N, N).

    p_+ = exp(-beta E+) / (exp(-beta E+) + exp(-beta E-)) = 1 / (1 + exp(-beta (E- - E+))).
    """
    n = xi.shape[1]
    x = all_states(n)
    e = energy(x, xi, g)
    idx = np.arange(2**n)
    out = np.empty((2**n, n))
    for i in range(n):
        up = idx | (1 << i)
        dn = idx & ~(1 << i)
        out[:, i] = 1.0 / (1.0 + np.exp(-beta * (e[dn] - e[up])))
    return out


def kernel_from_conditionals(cond: np.ndarray) -> np.ndarray:
    """Random-site heat-bath kernel K[x, y] = (1/N) sum_i P_i(x -> y), shape (2^N, 2^N)."""
    s, n = cond.shape
    idx = np.arange(s)
    k = np.zeros((s, s))
    for i in range(n):
        up = idx | (1 << i)
        dn = idx & ~(1 << i)
        np.add.at(k, (idx, up), cond[:, i] / n)
        np.add.at(k, (idx, dn), (1.0 - cond[:, i]) / n)
    return k


def gibbs(xi: np.ndarray, g: float, beta: float) -> np.ndarray:
    e = energy(all_states(xi.shape[1]), xi, g)
    w = np.exp(-beta * (e - e.min()))
    return w / w.sum()
