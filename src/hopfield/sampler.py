"""Batched GPU heat-bath Glauber sampler for the Hopfield model.

The kernel is O(Np): the projection vector
``c_mu = xi^mu . s`` is carried and updated incrementally on every accepted flip,
and the local field is

    H_i = (sum_mu xi^mu_i c_mu - p s_i) / N + G xi^1_i .

**All field arithmetic is float64**.  It is in fact
done on the *host*, exactly once per (rung, condition): the numerator
``num_i = sum_mu xi^mu_i c_mu - p s_i`` is an integer in
``[-pN-p, pN+p]``, so the whole acceptance probability
``p_+(H_i) = sigmoid(2 beta H_i)`` takes only ``2 (2pN+2p+1)`` distinct values per
condition.  Those are tabulated in NumPy float64 and looked up on the GPU.  No
field arithmetic, and no transcendental, ever runs in reduced precision.

Chain randomness: every chain draws its own scan permutation and its own N
uniforms per sweep, from disjoint segments of one Philox stream.  No randomness
is shared between chains.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import expit


def acceptance_table(
    n_spins: int, n_patterns: int, beta: float, fields: np.ndarray
) -> np.ndarray:
    """p_+(H) for every reachable integer field numerator, shape ``(n_cond, 2, W)``.

    Axis 1 indexes ``xi^1_i``: slot 0 is ``xi^1_i = -1``, slot 1 is ``+1``.
    Axis 2 indexes ``num + pN + p`` with ``num`` the integer numerator.

    STATUS: exact formula, float64 evaluation with
    ``scipy.special.expit`` (stable at both tails, unlike ``(1+tanh)/2``).
    """
    off = n_patterns * n_spins + n_patterns
    num = np.arange(-off, off + 1, dtype=np.float64)
    xi1 = np.array([-1.0, 1.0])
    local_h = num[None, None, :] / n_spins + np.asarray(fields, dtype=np.float64)[
        :, None, None
    ] * xi1[None, :, None]
    return expit(2.0 * beta * local_h)


class BatchedHopfield:
    """A batch of independent Hopfield chains, one GPU tensor per quantity.

    ``xi_all``      (n_seed, p, N) int8 -- the pattern sets
    ``seed_idx``    (B,) which pattern set each chain uses
    ``cond_idx``    (B,) which grid condition (hence which G) each chain runs at
    ``fields``      (n_cond,) the field G of each condition
    """

    def __init__(
        self,
        xi_all: np.ndarray,
        seed_idx: np.ndarray,
        cond_idx: np.ndarray,
        fields: np.ndarray,
        *,
        device: str = "cuda",
        torch_seed: int = 0,
        perm_chunk: int = 32768,
    ) -> None:
        self.device = torch.device(device)
        self.n_seed, self.n_patterns, self.n_spins = xi_all.shape
        self.fields = np.asarray(fields, dtype=np.float64)
        self.n_cond = self.fields.size
        self.perm_chunk = perm_chunk

        self.xi_all = xi_all
        # (n_seed * N, p + 1) int64, so ONE index_select per step yields both
        # xi^mu_i for every mu and the table slot of xi^1_i.
        self.width = 2 * self.n_patterns * self.n_spins + 2 * self.n_patterns + 1
        self.offset = self.n_patterns * self.n_spins + self.n_patterns
        xit = np.ascontiguousarray(
            xi_all.transpose(0, 2, 1).reshape(-1, self.n_patterns)
        ).astype(np.int64)
        slot = (self.width * (xit[:, 0] > 0)).astype(np.int64)[:, None]
        self.xit = torch.from_numpy(np.ascontiguousarray(np.hstack([xit, slot]))).to(self.device)

        self.seed_idx = torch.from_numpy(np.asarray(seed_idx, dtype=np.int64)).to(self.device)
        self.cond_idx = torch.from_numpy(np.asarray(cond_idx, dtype=np.int64)).to(self.device)
        self.n_chains = int(self.seed_idx.numel())

        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(torch_seed))

        self.state = torch.empty((self.n_chains, self.n_spins), dtype=torch.int8, device=self.device)
        self.proj = torch.zeros((self.n_chains, self.n_patterns), dtype=torch.int64, device=self.device)
        self._arange = torch.arange(self.n_chains, dtype=torch.int64, device=self.device)
        self._perm = torch.empty((self.n_chains, self.n_spins), dtype=torch.int16, device=self.device)
        self._table_cache: dict[float, torch.Tensor] = {}

    # ------------------------------------------------------------------ state
    def cold_start(self) -> None:
        """Each spin +-1 with probability 1/2 (protocol step 1)."""
        bits = torch.randint(
            0, 2, (self.n_chains, self.n_spins), generator=self.gen,
            dtype=torch.int8, device=self.device,
        )
        self.state = bits * 2 - 1
        self._recompute_projections()

    def set_state(self, state: np.ndarray) -> None:
        """Force the state (used by (P2)'s warm start at s = xi^1, and by tests)."""
        self.state = torch.from_numpy(np.ascontiguousarray(state, dtype=np.int8)).to(self.device)
        self._recompute_projections()

    def _recompute_projections(self) -> None:
        """c_mu = xi^mu . s from scratch, in exact int32 arithmetic.

        Chunked over chains so the (chunk, p, N) intermediate stays small.  No
        matmul is used: a float matmul on this hardware may fall back to TF32,
        which would silently round an exactly-integral quantity.
        """
        xi = torch.from_numpy(self.xi_all.astype(np.int32)).to(self.device)  # (n_seed, p, N)
        proj = torch.empty((self.n_chains, self.n_patterns), dtype=torch.int64,
                           device=self.device)
        chunk = 4096
        for lo in range(0, self.n_chains, chunk):
            hi = min(lo + chunk, self.n_chains)
            xi_chunk = xi[self.seed_idx[lo:hi]]                              # (c, p, N)
            st = self.state[lo:hi].to(torch.int32).unsqueeze(1)              # (c, 1, N)
            proj[lo:hi] = (xi_chunk * st).sum(dim=2).to(torch.int64)
            del xi_chunk, st
        self.proj = proj
        del xi

    def projection_drift(self) -> int:
        """max |carried c - recomputed c|; must be 0."""
        carried = self.proj.clone()
        self._recompute_projections()
        drift = int((carried - self.proj).abs().max()) if self.n_chains else 0
        self.proj = carried
        return drift

    def projections(self) -> np.ndarray:
        """c_mu per chain, shape (B, p) int64."""
        return self.proj.to(torch.int64).cpu().numpy()

    def select(self, mask: torch.Tensor) -> None:
        """Keep only the chains where ``mask`` is true, compacting every buffer."""
        self.state = self.state[mask].contiguous()
        self.proj = self.proj[mask].contiguous()
        self.seed_idx = self.seed_idx[mask].contiguous()
        self.cond_idx = self.cond_idx[mask].contiguous()
        self.n_chains = int(self.seed_idx.numel())
        self._arange = torch.arange(self.n_chains, dtype=torch.int64, device=self.device)
        self._perm = torch.empty(
            (self.n_chains, self.n_spins), dtype=torch.int16, device=self.device
        )

    # ------------------------------------------------------------------ table
    def table_for(self, beta: float) -> torch.Tensor:
        """Flattened acceptance table for this rung, ``(n_cond * 2 * W,)`` float64."""
        key = float(beta)
        if key not in self._table_cache:
            tab = acceptance_table(self.n_spins, self.n_patterns, key, self.fields)
            self._table_cache[key] = torch.from_numpy(
                np.ascontiguousarray(tab.reshape(-1))
            ).to(self.device)
        return self._table_cache[key]

    # ------------------------------------------------------------------ sweep
    def _draw_permutations(self) -> None:
        """A fresh independent random permutation of {0..N-1} for every chain.

        Chunked so the transient int64 argsort output never dominates the pool.
        """
        for lo in range(0, self.n_chains, self.perm_chunk):
            hi = min(lo + self.perm_chunk, self.n_chains)
            keys = torch.rand((hi - lo, self.n_spins), generator=self.gen, device=self.device)
            self._perm[lo:hi] = torch.argsort(keys, dim=1).to(torch.int16)
            del keys

    def run_sweeps(
        self,
        beta: float,
        n_sweeps: int,
        *,
        perm_override: np.ndarray | None = None,
        unif_override: np.ndarray | None = None,
    ) -> None:
        """``n_sweeps`` sweeps at inverse temperature ``beta``, same G per chain.

        ``perm_override`` (B, N) and ``unif_override`` (n_sweeps, B, N) replace
        the internal RNG; ``sampler_checks.g1c_gpu_vs_reference`` uses them to reproduce the NumPy
        reference sweep bit for bit.
        """
        if self.n_chains == 0 or n_sweeps == 0:
            return
        table = self.table_for(beta)
        n = self.n_spins
        p = self.n_patterns
        state_flat = self.state.view(-1)
        # Loop-invariant index arithmetic, hoisted: at ~2e6 spin-steps per chain
        # batch the kernel is launch-bound, so every op removed from the body is
        # worth more than any op made cheaper.
        base_xi = self.seed_idx * n
        base_all = self.cond_idx * (2 * self.width) + self.offset
        arange_n = self._arange * n
        one8 = torch.ones(self.n_chains, dtype=torch.int8, device=self.device)
        neg8 = -one8

        for sweep_i in range(n_sweeps):
            if perm_override is None:
                self._draw_permutations()
                perm = self._perm
            else:
                perm = torch.from_numpy(
                    np.ascontiguousarray(perm_override, dtype=np.int16)
                ).to(self.device)
            if unif_override is not None:
                unif_sweep = torch.from_numpy(
                    np.ascontiguousarray(unif_override[sweep_i], dtype=np.float64)
                ).to(self.device)

            for t in range(n):
                site = perm[:, t].to(torch.int64)
                flat = arange_n + site
                s_old = state_flat[flat].to(torch.int64)
                gathered = self.xit[base_xi + site]                   # (B, p+1) int64
                xi_i = gathered[:, :p]
                num = (xi_i * self.proj).sum(dim=1) - p * s_old       # integer, exact
                idx = base_all + gathered[:, p] + num
                pp = table[idx]
                if unif_override is None:
                    u = torch.rand(
                        self.n_chains, generator=self.gen, dtype=torch.float64,
                        device=self.device,
                    )
                else:
                    u = unif_sweep[:, t]
                s_new = torch.where(u < pp, one8, neg8)               # int8
                state_flat[flat] = s_new
                self.proj += xi_i * (s_new.to(torch.int64) - s_old).unsqueeze(1)

    # ----------------------------------------------------------------- labels
    def labels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(mu_hat, sigma_hat) per chain.  ``mu_hat`` is 0-based."""
        absc = self.proj.abs()
        mu = absc.argmax(dim=1)
        picked = self.proj.gather(1, mu.unsqueeze(1)).squeeze(1)
        return mu, torch.sign(picked).to(torch.int32)

    def correct_mask(self) -> torch.Tensor:
        """(mu_hat, sigma_hat) = (1, +1) in 1-based labelling."""
        mu, sgn = self.labels()
        return (mu == 0) & (sgn == 1)
