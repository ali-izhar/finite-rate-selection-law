"""Batched torch sampler of the simplified protocol: random-site heat bath, coupling 1/(N-1).

The same chain as ``src.simple.reference.ReferenceSampler`` (see its docstring for the model and
the field formula), on the GPU.  What is kept from ``src.hopfield.sampler``: the carried integer
projections c_mu = xi^mu . x (O(p) work per update, exact int64), the exact chunked recompute
used for the drift check, and float64 field arithmetic.  What changed, each by the locked design:

* site: one uniformly random site per update and chain (``torch.randint``), not a permutation
  sweep;  one sweep = N updates;
* coupling 1/(N-1), so p = 1 is the CW chain of paper Eq. (1) exactly (gauge y = xi^1 * x);
* beta changes every update (the linear ramp at the midpoint of the update), so the acceptance
  probability is evaluated directly in float64 instead of being looked up in a per-beta table:
  num is an exact integer, h = num/(N-1) + g xi^1_i and sigmoid(2 beta h) are float64, in the
  operation order of ``reference.p_plus``.

Randomness: one torch.Generator per sampler; per block of updates it draws a (T, B) int64 site
array and a (T, B) float64 uniform array.  Every chain gets its own draws; no draw is shared.
"""
from __future__ import annotations

import numpy as np
import torch

from src.simple import design as D


class Sampler:
    """B independent chains on one device.

    xi_sets   (K, p, N) int8   pattern sets
    set_idx   (B,)             set of each chain
    g         (B,) float64     signed field of each chain
    """

    def __init__(self, xi_sets: np.ndarray, set_idx: np.ndarray, g: np.ndarray, *,
                 device: str | torch.device = "cuda", generator: torch.Generator | None = None,
                 block: int = 256) -> None:
        self.device = torch.device(device)
        self.xi_sets = np.asarray(xi_sets, dtype=np.int8)
        self.k_sets, self.p, self.n = self.xi_sets.shape
        self.b = int(np.asarray(set_idx).size)
        self.block = block
        self.gen = generator
        # xit[s*N + i, mu] = xi^mu_i of set s, int64: one gather per update yields every xi^mu_i
        xit = np.ascontiguousarray(self.xi_sets.transpose(0, 2, 1).reshape(-1, self.p))
        self.xit = torch.from_numpy(xit.astype(np.int64)).to(self.device)
        self.set_idx = torch.from_numpy(np.asarray(set_idx, dtype=np.int64)).to(self.device)
        self.g = torch.from_numpy(np.asarray(g, dtype=np.float64)).to(self.device)
        self.state = torch.ones((self.b, self.n), dtype=torch.int8, device=self.device)
        self.proj = torch.zeros((self.b, self.p), dtype=torch.int64, device=self.device)
        self._base_xi = self.set_idx * self.n                         # (B,) row offset into xit
        self._base_state = torch.arange(self.b, dtype=torch.int64, device=self.device) * self.n

    # ----------------------------------------------------------------- state
    def set_state(self, state: np.ndarray) -> None:
        self.state = torch.from_numpy(np.ascontiguousarray(state, dtype=np.int8)).to(self.device)
        self.proj = self.recompute_projections()

    def start(self) -> None:
        """Every spin independent and unbiased (exact equilibrium at beta = 0)."""
        bits = torch.randint(0, 2, (self.b, self.n), generator=self.gen, dtype=torch.int8,
                             device=self.device)
        self.state = bits * 2 - 1
        self.proj = self.recompute_projections()

    def recompute_projections(self, chunk: int = 4096) -> torch.Tensor:
        """c_mu = xi^mu . x from scratch in exact int32 arithmetic (no matmul: no TF32 risk)."""
        xi = torch.from_numpy(self.xi_sets.astype(np.int32)).to(self.device)   # (K, p, N)
        out = torch.empty((self.b, self.p), dtype=torch.int64, device=self.device)
        for lo in range(0, self.b, chunk):
            hi = min(lo + chunk, self.b)
            xs = xi[self.set_idx[lo:hi]]                                         # (c, p, N)
            st = self.state[lo:hi].to(torch.int32).unsqueeze(1)                  # (c, 1, N)
            out[lo:hi] = (xs * st).sum(dim=2).to(torch.int64)
        return out

    def projection_drift(self) -> int:
        """max |carried c - recomputed c| over chains and patterns; 0 if the carry is exact."""
        if self.b == 0:
            return 0
        return int((self.proj - self.recompute_projections()).abs().max())

    def projections(self) -> np.ndarray:
        return self.proj.cpu().numpy()

    # ---------------------------------------------------------------- update
    def p_plus_at(self, site: torch.Tensor, beta: float) -> torch.Tensor:
        """sigmoid(2 beta h_i) at ``site`` (B,), float64; the formula of reference.p_plus."""
        xi_i = self.xit[self._base_xi + site]                                    # (B, p)
        s_old = self.state.view(-1)[self._base_state + site].to(torch.int64)
        num = (xi_i * self.proj).sum(dim=1) - self.p * s_old
        h = num.to(torch.float64) / (self.n - 1) + self.g * xi_i[:, 0].to(torch.float64)
        return torch.sigmoid(h * (2.0 * beta))

    def run(self, n0: int, n1: int, *, sites: np.ndarray | None = None,
            unif: np.ndarray | None = None) -> None:
        """Updates n0..n1-1 of the design ramp at this N.

        ``sites`` (n1-n0, B) and ``unif`` (n1-n0, B) replace the generator (G0 check only).
        """
        if self.b == 0 or n1 <= n0:
            return
        betas = D.beta_of_update(np.arange(n0, n1), self.n).tolist()
        n, p = self.n, self.p
        state_flat = self.state.view(-1)
        one = torch.ones(self.b, dtype=torch.int8, device=self.device)
        neg = -one
        for lo in range(0, n1 - n0, self.block):
            hi = min(lo + self.block, n1 - n0)
            if sites is None:
                site_blk = torch.randint(0, n, (hi - lo, self.b), generator=self.gen,
                                         dtype=torch.int64, device=self.device)
                u_blk = torch.rand((hi - lo, self.b), generator=self.gen, dtype=torch.float64,
                                   device=self.device)
            else:
                site_blk = torch.from_numpy(np.ascontiguousarray(sites[lo:hi], dtype=np.int64)
                                            ).to(self.device)
                u_blk = torch.from_numpy(np.ascontiguousarray(unif[lo:hi], dtype=np.float64)
                                         ).to(self.device)
            for t in range(hi - lo):
                site = site_blk[t]
                flat = self._base_state + site
                xi_i = self.xit[self._base_xi + site]                            # (B, p) int64
                s_old = state_flat[flat].to(torch.int64)
                num = (xi_i * self.proj).sum(dim=1) - p * s_old                 # exact integer
                h = num.to(torch.float64) / (n - 1) + self.g * xi_i[:, 0].to(torch.float64)
                pp = torch.sigmoid(h * (2.0 * betas[lo + t]))
                s_new = torch.where(u_blk[t] < pp, one, neg)
                state_flat[flat] = s_new
                self.proj += xi_i * (s_new.to(torch.int64) - s_old).unsqueeze(1)

    # ---------------------------------------------------------------- labels
    def label_counts(self) -> dict[str, torch.Tensor]:
        """Per-chain booleans of the pre-registered label (all (B,)).

        tie       the largest |m^mu| is attained by more than one mu
        m1zero    m^1 = 0
        pattern1  no tie, argmax_mu |m^mu| = 1, and m^1 != 0   (the runs the estimator uses)
        correct   pattern1 and sign(m^1) = sign(g)
        winner    argmax_mu |m^mu| (0-based), meaningful where not tie
        """
        a = self.proj.abs()
        amax = a.max(dim=1).values
        tie = (a == amax.unsqueeze(1)).sum(dim=1) > 1
        m1 = self.proj[:, 0]
        winner = a.argmax(dim=1)
        pattern1 = (~tie) & (winner == 0) & (m1 != 0)
        correct = pattern1 & (torch.sign(m1).to(torch.float64) == torch.sign(self.g))
        return dict(tie=tie, m1zero=m1 == 0, pattern1=pattern1, correct=correct, winner=winner)
