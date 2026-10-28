"""GPU steps of the simplified Hopfield test (docs/protocol-simple.md).  Called by ``python -m src.simple``.

g0       the G0 code gate: torch sampler == NumPy reference bit for bit on a supplied stream
         (N = 16, p = 3, 50 updates), plus carried == recomputed projections  -> g0.json
pilot    NOT part of the pre-registered analysis (own seed, counts never pooled): G0, throughput
         of every row for a few sweeps (hours extrapolated), and one full-ramp p = 1 check at
         N = 256 against the exact chain                                        -> pilot.json
row      one pre-registered row (N, p) with its frozen run count               -> row_N{N}_p{p}.json
rows     every row, in the order of design.ROWS
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from src.simple import design as D
from src.simple.provenance import provenance

OUT_DIR = os.path.join("results", "simple")
PRED = os.path.join(OUT_DIR, "predictions.json")


def _torch():
    os.environ["SIMPLE_WANT_TORCH"] = "1"
    import torch
    return torch


def _device(torch):
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dump(name: str, obj: dict) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
    return path


def predictions() -> dict:
    with open(PRED) as f:
        return json.load(f)


def pred_row(n_spins: int, p: int) -> dict:
    for r in predictions()["rows"]:
        if r["N"] == n_spins and r["p"] == p:
            return r
    raise KeyError((n_spins, p))


def pattern_sets(n_spins: int, p: int) -> np.ndarray:
    """(K, p, N) int8, the frozen sets s = 0..K-1."""
    return np.stack([D.patterns(n_spins, s, p) for s in range(D.N_SETS)])


def chain_layout(lo: int, hi: int, n_spins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chains lo..hi-1 of a row: set = c mod K, sign j = (c div K) mod 2 (0: +|G|, 1: -|G|).

    Every block of 2K consecutive chains holds each (set, sign) once, so a row of
    per_cell * 2K chains gives each cell exactly per_cell runs.
    """
    c = np.arange(lo, hi)
    s = c % D.N_SETS
    j = (c // D.N_SETS) % 2
    g = np.where(j == 0, 1.0, -1.0) * D.field(n_spins)
    return s, j, g


# ------------------------------------------------------------------------------------- G0
def g0(n_spins: int = 16, p: int = 3, n_upd: int = 50, n_chains: int = 400, seed: int = 0) -> dict:
    torch = _torch()
    from src.simple.reference import ReferenceSampler
    from src.simple.sampler import Sampler
    rng = np.random.default_rng([D.SEED, 0, seed])
    xi = (2 * rng.integers(0, 2, size=(D.N_SETS, p, n_spins)) - 1).astype(np.int8)
    s, _, g = chain_layout(0, n_chains, n_spins)
    x0 = (2 * rng.integers(0, 2, size=(n_chains, n_spins)) - 1).astype(np.int8)
    sites = rng.integers(0, n_spins, size=(n_upd, n_chains))
    unif = rng.random((n_upd, n_chains))
    # the 50 updates sit in the middle of the ramp at this N, where p_+ is not near 0 or 1
    n0 = D.n_updates(n_spins) // 2
    ref = ReferenceSampler(xi, s, g); ref.set_state(x0)
    gpu = Sampler(xi, s, g, device=_device(torch), block=16); gpu.set_state(x0)
    pp_ref = ref.p_plus_at(sites[0], float(D.beta_of_update(n0, n_spins)))
    pp_gpu = gpu.p_plus_at(torch.from_numpy(sites[0]).to(gpu.device),
                           float(D.beta_of_update(n0, n_spins))).cpu().numpy()
    ref.run(n0, n0 + n_upd, sites=sites, unif=unif)
    gpu.run(n0, n0 + n_upd, sites=sites, unif=unif)
    st_gpu = gpu.state.cpu().numpy()
    flips = int((ref.state != x0).sum())
    res = dict(
        N=n_spins, p=p, updates=n_upd, chains=n_chains, spins_changed=flips,
        states_equal=bool(np.array_equal(st_gpu, ref.state)),
        projections_equal=bool(np.array_equal(gpu.projections(), ref.proj)),
        drift_gpu=gpu.projection_drift(),
        drift_ref=int(np.abs(ref.proj - ref.recompute_projections()).max()),
        p_plus_max_abs_diff=float(np.abs(pp_ref - pp_gpu).max()),
    )
    res["passed"] = bool(res["states_equal"] and res["projections_equal"] and res["drift_gpu"] == 0
                         and res["drift_ref"] == 0 and flips > 0)
    return res


def step_g0() -> dict:
    res = g0()
    out = dict(step="g0", result=res, provenance=provenance(g0_stream_seed=[D.SEED, 0, 0]))
    print(f"G0: {'PASS' if res['passed'] else 'FAIL'}  {res}", flush=True)
    _dump("g0.json", out)
    return out


# ------------------------------------------------------------------------------------- rows
def run_row(n_spins: int, p: int, batch: int, seed_offset: int = 0, n_runs: int | None = None,
            torch_seed: int | None = None, write: str | None = None) -> dict:
    """Run R chains of row (N, p) in batches of ``batch``; return per-(set, sign) counts.

    R defaults to the frozen run count of predictions.json.  The seed defaults to
    design.torch_seed(N, p) + seed_offset (the stop rule's rerun uses offset 1).
    """
    torch = _torch()
    from src.simple.sampler import Sampler
    dev = _device(torch)
    pr = pred_row(n_spins, p) if n_runs is None else None
    runs = pr["runs"] * D.RUN_FACTOR if n_runs is None else n_runs
    seed = (D.torch_seed(n_spins, p) + seed_offset) if torch_seed is None else torch_seed
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    xi = pattern_sets(n_spins, p)
    m = D.n_updates(n_spins)
    k = D.N_SETS
    cnt = {key: np.zeros((k, 2), dtype=np.int64)
           for key in ("runs", "tie", "m1zero", "pattern1", "correct")}
    winners = np.zeros((k, 2, p), dtype=np.int64)
    drift_max, t0, done = 0, time.time(), 0
    out = dict(step="row", N=n_spins, p=p, M_updates=m, runs=runs, batch=batch, torch_seed=seed,
               seed_offset=seed_offset, device=dev, complete=False)
    for lo in range(0, runs, batch):
        hi = min(lo + batch, runs)
        s, j, g = chain_layout(lo, hi, n_spins)
        smp = Sampler(xi, s, g, device=dev, generator=gen)
        smp.start()
        smp.run(0, m)
        drift_max = max(drift_max, smp.projection_drift())
        lab = {key: v.cpu().numpy() for key, v in smp.label_counts().items()}
        np.add.at(cnt["runs"], (s, j), 1)
        for key in ("tie", "m1zero", "pattern1", "correct"):
            np.add.at(cnt[key], (s, j), lab[key].astype(np.int64))
        ok = ~lab["tie"]
        np.add.at(winners, (s[ok], j[ok], lab["winner"][ok]), 1)
        done = hi
        wall = time.time() - t0
        rate = done * m / wall
        print(f"N={n_spins} p={p}: {done}/{runs} runs  {rate:.3e} updates/s  "
              f"q so far={cnt['correct'].sum() / max(cnt['pattern1'].sum(), 1):.4f}  "
              f"drift={drift_max}  [{wall:.0f}s]", flush=True)
        del smp
        if write:
            out.update(counts={kk: v.tolist() for kk, v in cnt.items()}, winners=winners.tolist(),
                       runs_done=done, drift_max=drift_max, wall_seconds=wall,
                       updates_per_second=rate)
            _dump(write, out)
    out.update(counts={kk: v.tolist() for kk, v in cnt.items()}, winners=winners.tolist(),
               runs_done=done, drift_max=drift_max, wall_seconds=time.time() - t0,
               updates_per_second=done * m / (time.time() - t0), complete=(done == runs))
    return out


def step_row(n_spins: int, p: int, batch: int, seed_offset: int = 0) -> dict:
    name = f"row_N{n_spins}_p{p}" + (f"_rerun{seed_offset}" if seed_offset else "") + ".json"
    prov = provenance(torch_seed=D.torch_seed(n_spins, p) + seed_offset,
                      pattern_seed=f"default_rng([{D.SEED}, {n_spins}, s]), s=0..{D.N_SETS - 1}")
    out = run_row(n_spins, p, batch, seed_offset, write=name)
    out["provenance"] = prov
    out["provenance"]["utc_end"] = provenance()["utc"]
    _dump(name, out)
    print(f"wrote {os.path.join(OUT_DIR, name)}", flush=True)
    return out


# ------------------------------------------------------------------------------------ pilot
PILOT_SEED = 1          # distinct from every row seed (>= 2e7); pilot counts are never analysed


def step_pilot(batch: int, n_check: int = 256, runs_check: int = 100_000,
               timing_updates: int = 2) -> dict:
    torch = _torch()
    from scipy.special import logit
    from src.simple.predict import ramp_laws, sign_probs
    from src.simple.sampler import Sampler
    dev = _device(torch)
    res = dict(step="pilot", note="not part of the pre-registered analysis", g0=g0())
    print(f"G0: {'PASS' if res['g0']['passed'] else 'FAIL'}", flush=True)

    timing = []
    gen = torch.Generator(device=dev); gen.manual_seed(PILOT_SEED)
    for n_spins, p in D.ROWS:
        pr = pred_row(n_spins, p)
        b = min(batch, pr["runs"])
        s, _, g = chain_layout(0, b, n_spins)
        smp = Sampler(pattern_sets(n_spins, p), s, g, device=dev, generator=gen)
        smp.start()
        smp.run(0, 256)                                     # warm-up
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        upd = timing_updates * n_spins
        smp.run(256, 256 + upd)
        if dev == "cuda":
            torch.cuda.synchronize()
        rate = b * upd / (time.time() - t0)
        hours = pr["runs"] * D.n_updates(n_spins) / rate / 3600
        timing.append(dict(N=n_spins, p=p, batch=b, updates_per_second=rate, row_hours=hours,
                           drift=smp.projection_drift()))
        print(f"timing N={n_spins} p={p} B={b}: {rate:.3e} updates/s -> {hours:.2f} h for the row",
              flush=True)
        del smp
    res["timing"] = timing
    res["total_hours"] = sum(t["row_hours"] for t in timing)

    # full ramp, p = 1, N = n_check, against the exact chain of the same ramp
    t0 = time.time()
    r = run_row(n_check, 1, batch, n_runs=runs_check, torch_seed=PILOT_SEED + 1)
    g = D.field(n_check)
    qp, qm = sign_probs(ramp_laws([n_check], [1.0 / (n_check - 1)], n_check, g)[0], n_check)
    q_exact = qp / (qp + qm)
    n1 = int(np.sum(r["counts"]["pattern1"])); c1 = int(np.sum(r["counts"]["correct"]))
    q_hat = c1 / n1
    z = (q_hat - q_exact) / np.sqrt(q_exact * (1 - q_exact) / n1)
    res["p1_check"] = dict(N=n_check, runs=runs_check, n_pattern1=n1, q_hat=q_hat,
                           q_exact=float(q_exact), z=float(z),
                           logit_ratio=float(logit(q_hat) / logit(q_exact)),
                           passed=bool(abs(z) < 4), drift=r["drift_max"],
                           wall_seconds=time.time() - t0)
    print(f"p=1 N={n_check}: q_hat={q_hat:.5f}  exact={q_exact:.5f}  z={z:+.2f}  "
          f"{'PASS' if abs(z) < 4 else 'FAIL'}", flush=True)
    res["provenance"] = provenance(pilot_seed=PILOT_SEED, p1_check_seed=PILOT_SEED + 1)
    _dump("pilot.json", res)
    print(f"pilot: {res['total_hours']:.2f} GPU h for all rows at B={batch}", flush=True)
    return res
