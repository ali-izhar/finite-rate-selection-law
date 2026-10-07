"""The pre-runs and the field-grid driver.

Protocol, per run (paper Sec. 4, "Reheating after a filtered start"):
  1. start with independent unbiased spins;
  2. brief cooling, 2 sweeps at each of the 10 temperatures;
  3. keep only the runs not labeled (1, +);
  4. reheat, 30 sweeps at beta = 0.7;
  5. final cooling, K(beta) sweeps at each temperature, same energy, same G;
  6. re-label; retain runs with mu_hat = 1; among those count sigma_hat = +1.

Both pre-runs are at G = 0 only, so they cannot inform the field-grid results.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from . import protocol as D
from . import reference as H
from .sampler import BatchedHopfield


# --------------------------------------------------------------- utilities --
def pattern_bank(seed_indices: list[int]) -> np.ndarray:
    """The pattern sets, shape ``(len(seed_indices), p, N)``.

    "The same twenty pattern sets at every grid point and at the calibration
    point" (protocol).
    """
    return np.stack(
        [H.patterns(D.seed_of(s), D.N_SPINS, D.N_PATTERNS) for s in seed_indices]
    )


def _group_counts(values: torch.Tensor, group: torch.Tensor, n_groups: int) -> np.ndarray:
    """Sum a boolean tensor into ``n_groups`` bins."""
    out = torch.zeros(n_groups, dtype=torch.int64, device=values.device)
    out.scatter_add_(0, group, values.to(torch.int64))
    return out.cpu().numpy()


def _descend(sim: BatchedHopfield, budgets: tuple[int, ...], log=None) -> None:
    for beta, k in zip(D.LADDER, budgets, strict=True):
        t0 = time.perf_counter()
        sim.run_sweeps(beta, k)
        if log is not None:
            log(f"      beta={beta:.2f}  {k:4d} sweeps  "
                f"{time.perf_counter()-t0:7.1f}s  B={sim.n_chains}")


def run_protocol(
    xi_all: np.ndarray, fields: np.ndarray, chains_per_seed: int, *, torch_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The protocol (steps 1-6 above) on every (condition, draw) cell, no logging.

    ``xi_all`` is ``(n_draws, p, N)``; condition ``c`` runs at G = ``fields[c]``; every
    cell carries ``chains_per_seed`` chains, all in one batch.  Returns the counts
    ``(n_mis, n_ret, n_cor)``, each of shape ``(len(fields), n_draws)``: mis-committed
    after the scarce descent, retained (mu_hat = 1) after the re-descent, and retained
    with sigma_hat = +1.  The reruns in ``src/experiments`` call this;
    ``run_grid`` below is the logged driver of the paper's grid.
    """
    n_cond, n_seed = len(fields), len(xi_all)
    cond_idx = np.repeat(np.arange(n_cond), n_seed * chains_per_seed)
    seed_idx = np.tile(np.repeat(np.arange(n_seed), chains_per_seed), n_cond)
    sim = BatchedHopfield(xi_all, seed_idx, cond_idx, fields, torch_seed=torch_seed)
    n_groups = n_cond * n_seed
    sim.cold_start()
    _descend(sim, (D.K_SCARCE,) * len(D.LADDER))
    mis = ~sim.correct_mask()
    n_mis = _group_counts(mis, sim.cond_idx * n_seed + sim.seed_idx, n_groups)
    sim.select(mis)
    sim.run_sweeps(D.BETA_REHEAT, D.K_REHEAT)
    _descend(sim, D.K_REDESCENT)
    mu, sgn = sim.labels()
    retained = (mu == 0) & (sgn != 0)
    g = sim.cond_idx * n_seed + sim.seed_idx
    n_ret = _group_counts(retained, g, n_groups)
    n_cor = _group_counts(retained & (sgn == 1), g, n_groups)
    del sim
    torch.cuda.empty_cache()
    shape = (n_cond, n_seed)
    return n_mis.reshape(shape), n_ret.reshape(shape), n_cor.reshape(shape)


# ------------------------------------------------------------------- P2 -----
@dataclass
class OverlapCalibration:
    """(P2) per-seed bar m and its standard error."""

    seed_index: int
    mbar: float
    se_spec: float
    se_across_chains: float
    sd_in_well: float
    n_samples: int
    n_excluded: int


def run_p2_overlap_calibration(
    seed_indices: list[int], *, device: str = "cuda", torch_seed: int = 20260823,
    log=print,
) -> list[OverlapCalibration]:
    """(P2): per seed, 256 chains warm-started at xi^1, 60 burn-in then 200 measured
    sweeps at beta = 1.6, G = 0; reports bar m_seed = E[m^1 | R_+] and its SE.

    The standard error follows paper App. B: the
    in-well relaxation time tau_ex(1.6) = 1.695 sweeps sets an effective sample
    size ``n_chains * n_meas / tau``.  An independent across-chain standard error
    (the 256 chains are exactly independent, so no autocorrelation model is
    needed) is reported beside it as a check.

    All seeds are carried in one batch (20 x 256 chains).  That is a pure
    scheduling choice -- the chains are exactly the ones the protocol asks for, one
    warm-started set of 256 per seed, each with its own randomness -- and it
    turns a per-seed-serial run into a single 260-sweep pass.

    STATUS: numerical -- an equilibrium average inside one well of the *unbiased*
    model.: cannot leak.
    """
    xi_all = pattern_bank(seed_indices)
    n_seed = len(seed_indices)
    b_total = n_seed * D.B_CAL_CHAINS
    tau = D.TAU_EX_SPEC[-1]  # tau_ex(1.6) = 1.6946

    seed_idx = np.repeat(np.arange(n_seed), D.B_CAL_CHAINS)
    sim = BatchedHopfield(
        xi_all, seed_idx, np.zeros(b_total, dtype=np.int64), np.array([0.0]),
        device=device, torch_seed=torch_seed,
    )
    sim.set_state(np.repeat(xi_all[:, 0, :], D.B_CAL_CHAINS, axis=0))   # warm start at xi^1
    log(f"  (P2) {n_seed} seeds x {D.B_CAL_CHAINS} chains, "
        f"{D.K_BURN} burn-in + {D.K_MEAS} measured sweeps at beta = {D.BETA_MEASURE}")
    sim.run_sweeps(D.BETA_MEASURE, D.K_BURN)

    per_chain_sum = np.zeros(b_total)
    per_chain_n = np.zeros(b_total)
    per_chain_sq = np.zeros(b_total)
    for _ in range(D.K_MEAS):
        sim.run_sweeps(D.BETA_MEASURE, 1)
        proj = sim.projections()                          # (B, p)
        m = proj[:, 0] / D.N_SPINS
        mu = np.argmax(np.abs(proj), axis=1)
        keep = (mu == 0) & (m > 0.0)                      # the event R_+
        per_chain_sum += np.where(keep, m, 0.0)
        per_chain_sq += np.where(keep, m * m, 0.0)
        per_chain_n += keep

    results = []
    for j, s in enumerate(seed_indices):
        sl = slice(j * D.B_CAL_CHAINS, (j + 1) * D.B_CAL_CHAINS)
        total = int(per_chain_n[sl].sum())
        excluded = D.B_CAL_CHAINS * D.K_MEAS - total
        mbar = float(per_chain_sum[sl].sum() / total)
        var = float(per_chain_sq[sl].sum() / total) - mbar**2
        sd = float(np.sqrt(max(var, 0.0)))
        se_spec = sd / np.sqrt(D.B_CAL_CHAINS * D.K_MEAS / tau)
        ok = per_chain_n[sl] > 0
        chain_means = per_chain_sum[sl][ok] / per_chain_n[sl][ok]
        se_chain = float(np.std(chain_means, ddof=1) / np.sqrt(chain_means.size))
        results.append(
            OverlapCalibration(s, mbar, float(se_spec), se_chain, sd, total, excluded)
        )
        log(f"  (P2) seed s={s:2d} (rng {D.seed_of(s)}): bar m = {mbar:.7f}  "
            f"sd = {sd:.4e}  SE_spec = {se_spec:.2e}  SE_chain = {se_chain:.2e}  "
            f"excluded = {excluded}")
    log(f"  (P2) carried-vs-recomputed projection drift = {sim.projection_drift()}")
    del sim
    torch.cuda.empty_cache()
    return results


# ------------------------------------------------------------------- P1 -----
@dataclass
class BudgetPilot:
    """(P1) the budget pilot's two fractions and the B they imply."""

    seed_index: int
    n_chains: int
    f_miscommit: float
    c_retained: float
    fc: float
    b_chosen: int
    n_predicted_per_seed: float


def run_p1_budget_pilot(
    *, seed_index: int = D.PILOT_SEED_INDEX, device: str = "cuda",
    torch_seed: int = 20260823 + 1, log=print,
) -> BudgetPilot:
    """(P1): one seed, 2000 chains, the full protocol at G = 0.

    Reports ``f`` (mis-commitment fraction of the scarce descent) and ``c``
    (fraction of reheated chains whose new label is mu_hat = 1), and sets ``B``
.  Nothing else.

    STATUS: numerical.: at G = 0 the whole trajectory law is
    invariant under exchanging the correct and the wrong mode, so no statistic of
    this run carries information about q_model - q_eq at any G != 0.
    """
    xi_all = pattern_bank([seed_index])
    sim = BatchedHopfield(
        xi_all,
        np.zeros(D.B_PILOT, dtype=np.int64),
        np.zeros(D.B_PILOT, dtype=np.int64),
        np.array([0.0]),
        device=device, torch_seed=torch_seed,
    )
    sim.cold_start()
    _descend(sim, (D.K_SCARCE,) * len(D.LADDER))
    correct = sim.correct_mask()
    n_mis = int((~correct).sum())
    f = n_mis / D.B_PILOT
    sim.select(~correct)
    sim.run_sweeps(D.BETA_REHEAT, D.K_REHEAT)
    _descend(sim, D.K_REDESCENT)
    mu, sgn = sim.labels()
    retained = (mu == 0) & (sgn != 0)
    n_ret = int(retained.sum())
    c = n_ret / max(n_mis, 1)
    b = D.chains_per_seed(f, c)
    pilot = BudgetPilot(seed_index, D.B_PILOT, f, c, f * c, b, D.B_PILOT * f * c)
    log(f"  (P1) f = {f:.4f}  c = {c:.4f}  f*c = {f*c:.4f}  ->  B = {b}  "
        f"(predicted pooled n at {D.N_SEEDS} seeds = {D.N_SEEDS * b * f * c:.0f})")
    del sim
    torch.cuda.empty_cache()
    return pilot


# ------------------------------------------------------------------ grid ----
@dataclass
class GridRun:
    """Raw counts from one pass of the grid, per (condition, seed)."""

    conditions: list[float]                 # G of each condition
    seed_indices: list[int]
    chains_per_seed: int
    n_started: list[list[int]] = field(default_factory=list)
    n_miscommitted: list[list[int]] = field(default_factory=list)
    n_retained: list[list[int]] = field(default_factory=list)
    n_correct: list[list[int]] = field(default_factory=list)
    n_zero_overlap: list[list[int]] = field(default_factory=list)
    n_argmax_tie_scarce: int = 0
    n_argmax_tie_final: int = 0
    wall_seconds: float = 0.0


def run_grid(
    fields: list[float], seed_indices: list[int], chains_per_seed: int, *,
    device: str = "cuda", torch_seed: int = 20260823 + 2, log=print,
) -> GridRun:
    """One synchronised pass of the whole field grid.

    Every (condition, seed) cell runs ``chains_per_seed`` chains; all cells are
    carried in one batch so that the 10-rung ladder is walked once, not 180
    times (9 conditions x 20 seeds).  Chains never interact: each draws its own scan permutation and its
    own uniforms.
    """
    n_cond, n_seed = len(fields), len(seed_indices)
    total = n_cond * n_seed * chains_per_seed
    log(f"  grid: {n_cond} conditions x {n_seed} seeds x {chains_per_seed} chains "
        f"= {total} chains, {D.SWEEPS_PER_CHAIN} sweeps each")

    xi_all = pattern_bank(seed_indices)
    cond_idx = np.repeat(np.arange(n_cond), n_seed * chains_per_seed)
    seed_idx = np.tile(np.repeat(np.arange(n_seed), chains_per_seed), n_cond)
    sim = BatchedHopfield(
        xi_all, seed_idx, cond_idx, np.array(fields, dtype=np.float64),
        device=device, torch_seed=torch_seed,
    )
    t_start = time.perf_counter()
    sim.cold_start()

    def group(sim_: BatchedHopfield) -> torch.Tensor:
        return sim_.cond_idx * n_seed + sim_.seed_idx

    started = _group_counts(torch.ones_like(sim.cond_idx, dtype=torch.bool), group(sim),
                            n_cond * n_seed)

    log("    scarce descent (K_scarce = 2 per rung)")
    _descend(sim, (D.K_SCARCE,) * len(D.LADDER), log=log)

    absc = sim.proj.abs()
    top2 = absc.topk(2, dim=1).values
    ties_scarce = int((top2[:, 0] == top2[:, 1]).sum())
    correct = sim.correct_mask()
    mis = ~correct
    miscommitted = _group_counts(mis, group(sim), n_cond * n_seed)
    sim.select(mis)
    log(f"    kept {sim.n_chains} mis-committed chains of {total} "
        f"(f = {sim.n_chains/total:.4f}); argmax ties = {ties_scarce}")

    log(f"    reheat ({D.K_REHEAT} sweeps at beta = {D.BETA_REHEAT})")
    t0 = time.perf_counter()
    sim.run_sweeps(D.BETA_REHEAT, D.K_REHEAT)
    log(f"      {time.perf_counter()-t0:7.1f}s")

    log("    re-descent (K(beta) per rung)")
    _descend(sim, D.K_REDESCENT, log=log)

    absc = sim.proj.abs()
    top2 = absc.topk(2, dim=1).values
    ties_final = int((top2[:, 0] == top2[:, 1]).sum())
    mu, sgn = sim.labels()
    retained = (mu == 0) & (sgn != 0)
    zero_overlap = (mu == 0) & (sgn == 0)
    correct_final = retained & (sgn == 1)
    g = group(sim)
    n_ret = _group_counts(retained, g, n_cond * n_seed)
    n_cor = _group_counts(correct_final, g, n_cond * n_seed)
    n_zero = _group_counts(zero_overlap, g, n_cond * n_seed)

    wall = time.perf_counter() - t_start
    log(f"    done in {wall/60:.1f} min; argmax ties at re-label = {ties_final}")
    del sim
    torch.cuda.empty_cache()

    shape = (n_cond, n_seed)
    return GridRun(
        conditions=list(map(float, fields)),
        seed_indices=list(seed_indices),
        chains_per_seed=chains_per_seed,
        n_started=started.reshape(shape).tolist(),
        n_miscommitted=miscommitted.reshape(shape).tolist(),
        n_retained=n_ret.reshape(shape).tolist(),
        n_correct=n_cor.reshape(shape).tolist(),
        n_zero_overlap=n_zero.reshape(shape).tolist(),
        n_argmax_tie_scarce=ties_scarce,
        n_argmax_tie_final=ties_final,
        wall_seconds=wall,
    )


def save(obj, path: str) -> None:
    """Persist a dataclass (or list of them) as JSON."""
    def default(o):
        try:
            return asdict(o)
        except TypeError:
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)
