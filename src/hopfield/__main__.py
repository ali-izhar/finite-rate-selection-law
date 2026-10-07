"""Driver for the field-grid experiment (paper Sec. 4, App. B; GPU).

    python -m src.hopfield static     # sampler check G1 and the schedule cross-check
    python -m src.hopfield pre        # pre-runs at G = 0: reference overlaps, run budget
    python -m src.hopfield grid       # the field grid, plus the top-up rule
    python -m src.hopfield analyse    # checks G2-G4 and the per-field analysis

Run the steps in this order.  Output goes to ``$HOPFIELD_OUT`` (default
``results/hopfield``); set it to a scratch folder to keep the saved results.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict


from . import analysis, pipeline, sampler_checks, equilibrium
from . import protocol as D
from .stats import wilson

OUT = os.environ.get("HOPFIELD_OUT", "results/hopfield")


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _dump(name: str, obj) -> None:
    os.makedirs(OUT, exist_ok=True)
    pipeline.save(obj, os.path.join(OUT, name))
    _log(f"  -> {os.path.join(OUT, name)}")


def _load(name: str):
    with open(os.path.join(OUT, name)) as fh:
        return json.load(fh)


# ------------------------------------------------------------------ static --
def cmd_static() -> None:
    _log("=" * 78)
    _log("SAMPLER CHECKS, STATIC  (no field-grid data is read)")
    _log("=" * 78)
    results = [
        sampler_checks.check_ladder_budget(),
        sampler_checks.check_interval_machinery(),
        sampler_checks.g1b_fast_vs_dense(),
        sampler_checks.g1a_detailed_balance(),
        sampler_checks.g1c_gpu_vs_reference(),
        sampler_checks.g1d_empirical_stationary(),
    ]
    for r in results:
        _log(str(r))
    _dump("sampler_checks_static.json", [asdict(r) for r in results])
    _log(f"\nG1 overall: {'PASS' if all(r.passed for r in results[1:]) else 'FAIL'}")


# --------------------------------------------------------------- pre-runs ---
def cmd_pre() -> None:
    _log("=" * 78)
    _log("PRE-RUNS, both at G = 0 only")
    _log("=" * 78)
    t0 = time.perf_counter()
    _log("(P2) overlap calibration, per seed")
    cal = pipeline.run_p2_overlap_calibration(list(range(D.N_SEEDS)), log=_log)
    _dump("overlap_run.json", [asdict(c) for c in cal])

    _log("\n(P1) budget pilot, one seed, 2000 chains, full protocol at G = 0")
    pilot = pipeline.run_p1_budget_pilot(log=_log)
    _dump("pilot_run.json", asdict(pilot))
    _log(f"\npre-runs took {(time.perf_counter()-t0)/60:.1f} min")


# ------------------------------------------------------------------- grid ---
def cmd_grid() -> None:
    pilot = _load("pilot_run.json")
    b = int(pilot["b_chosen"])
    _log("=" * 78)
    _log(f"FIELD GRID, ONE PASS.  B = {b} from (P1).")
    _log("=" * 78)
    fields = list(map(float, D.G_GRID)) + [D.G_CAL]
    run = pipeline.run_grid(fields, list(range(D.N_SEEDS)), b, log=_log)
    _dump("field_grid.json", asdict(run))

    # ---- top-up rule: reads n only, never q_model ----
    n_by_point = [sum(run.n_retained[k]) for k in range(D.N_GRID)]
    _log(f"\npooled n per grid point after {D.N_SEEDS} seeds: "
         + ", ".join(f"k={k}:{n}" for k, n in enumerate(n_by_point)))
    extra_runs = []
    s_next = D.N_SEEDS
    while any(n < D.POOLED_TARGET for n in n_by_point) and s_next < D.MAX_SEEDS:
        need = [k for k in range(D.N_GRID) if n_by_point[k] < D.POOLED_TARGET]
        _log(f"  top-up: seed s={s_next} at grid points {need} (stopping rule reads n only)")
        sub_fields = [float(D.G_GRID[k]) for k in need]
        extra = pipeline.run_grid(
            sub_fields, [s_next], b, torch_seed=20260823 + 100 + s_next, log=_log
        )
        extra_runs.append({"seed_index": s_next, "points": need, "run": asdict(extra)})
        for j, k in enumerate(need):
            n_by_point[k] += extra.n_retained[j][0]
        s_next += 1
    if extra_runs:
        _dump("field_grid_topup.json", extra_runs)
    _log("\nfinal pooled n per grid point: "
         + ", ".join(f"k={k}:{n}" for k, n in enumerate(n_by_point)))


# ---------------------------------------------------------------- analyse ---
def cmd_analyse() -> None:
    run = _load("field_grid.json")
    cal = _load("overlap_run.json")
    topup = []
    if os.path.exists(os.path.join(OUT, "field_grid_topup.json")):
        topup = _load("field_grid_topup.json")

    mbar = {c["seed_index"]: c["mbar"] for c in cal}
    se = {c["seed_index"]: c["se_spec"] for c in cal}
    seeds = run["seed_indices"]

    succ: dict[int, dict[int, int]] = {k: {} for k in range(D.N_GRID)}
    nret: dict[int, dict[int, int]] = {k: {} for k in range(D.N_GRID)}
    for k in range(D.N_GRID):
        for j, s in enumerate(seeds):
            succ[k][s] = run["n_correct"][k][j]
            nret[k][s] = run["n_retained"][k][j]
    for block in topup:
        s = block["seed_index"]
        for j, k in enumerate(block["points"]):
            succ[k][s] = block["run"]["n_correct"][j][0]
            nret[k][s] = block["run"]["n_retained"][j][0]
            if s not in mbar:
                # A top-up seed needs its own calibrated mbar (MBAR_CW_SPEC is
                # biased); the published run never reached this path.
                raise NotImplementedError(
                    f"top-up seed {s} has no (P2) calibration; a top-up "
                    "seed must be calibrated by its own P2-style run "
                    "before use. "
                    "This path is untested and must not silently fall "
                    "back to the population mbar."
                )

    _log("=" * 78)
    _log("CHECKS G2-G4")
    _log("=" * 78)
    cal_idx = D.N_GRID  # the calibration condition is the 9th
    g2 = sampler_checks.gate_g2(sum(run["n_correct"][cal_idx]), sum(run["n_retained"][cal_idx]))
    g3 = sampler_checks.gate_g3(
        [sum(succ[k].values()) for k in range(D.N_GRID)],
        [sum(nret[k].values()) for k in range(D.N_GRID)],
    )
    g4 = sampler_checks.gate_g4({c["seed_index"]: c["mbar"] for c in cal})
    for r in (g2, g3, g4):
        _log(str(r))
    _dump("sampler_checks.json", [asdict(r) for r in (g2, g3, g4)])

    _log("")
    _log("=" * 78)
    _log("FIELD-GRID ANALYSIS")
    _log("=" * 78)
    points = [analysis.analyse_point(k, succ[k], nret[k], mbar, se) for k in range(D.N_GRID)]
    result = analysis.decide(points)
    _log(analysis.report(result))
    _dump("field_grid_analysis.json", asdict(result))

    _log("")
    _log("MIRROR-POOLED estimate at each |u| level.")
    _log("  Licensed only because G3 passed: q(7-k) = 1 - q(k) exactly,")
    _log("  so the mirror point's failures are the same Bernoulli trial as k's")
    _log("  successes.  Reported as extra precision; no decision reads it.")
    _log(f"{'|u| level':>10} {'k pair':>8} {'n':>7} {'q_model':>9} {'[L':>9} {'U]':>9} "
         f"{'q_eq':>9} {'shortfall':>10}")
    for k in range(4):
        sk = sum(succ[k].values())
        nk = sum(nret[k].values())
        sm = sum(succ[7 - k].values())
        nm = sum(nret[7 - k].values())
        ci = wilson(sk + (nm - sm), nk + nm)
        qe = points[k].q_eq.point
        _log(f"{abs(equilibrium.tilt_u(D.N_SPINS, D.BETA_DEEP, float(D.G_GRID[k]), D.MBAR_CW_SPEC)):10.4f} "
             f"{f'{k}/{7-k}':>8} {nk+nm:7d} {ci.point:9.5f} {ci.lower:9.5f} {ci.upper:9.5f} "
             f"{qe:9.5f} {ci.point - qe:10.5f}")

    _log("")
    _log("=" * 78)
    _log("PROTOCOL DIAGNOSTICS (not part of any check)")
    _log("=" * 78)
    _log(f"{'cond':>5} {'G':>12} {'started':>9} {'mis f':>8} {'retain c':>9} "
         f"{'f*c':>7} {'n':>7} {'m1=0':>6}")
    for k in range(D.N_GRID + 1):
        st = sum(run["n_started"][k])
        mis = sum(run["n_miscommitted"][k])
        ret = sum(run["n_retained"][k])
        zero = sum(run["n_zero_overlap"][k])
        g = run["conditions"][k]
        name = f"k={k}" if k < D.N_GRID else "cal"
        _log(f"{name:>5} {g:12.5e} {st:9d} {mis/st:8.4f} {ret/max(mis,1):9.4f} "
             f"{ret/st:7.4f} {ret:7d} {zero:6d}")
    _log(f"argmax ties: after scarce descent {run['n_argmax_tie_scarce']}, "
         f"at re-label {run['n_argmax_tie_final']}")
    _log(f"grid wall time: {run['wall_seconds']/60:.1f} min")

    _log("")
    _log("calibration condition (G = 0), per seed:")
    for j, s in enumerate(seeds):
        ci = wilson(run["n_correct"][cal_idx][j], run["n_retained"][cal_idx][j])
        _log(f"  s={s:2d}: {run['n_correct'][cal_idx][j]}/{run['n_retained'][cal_idx][j]} = {ci}")

    _log("")
    _log("(P2) per-seed bar m and the q_eq it implies at k = 0:")
    for c in cal:
        s = c["seed_index"]
        _log(f"  s={s:2d}: bar m = {c['mbar']:.7f}  (dev from bar m_CW "
             f"{c['mbar'] - D.MBAR_CW_SPEC:+.3e})  q_eq^(0) = "
             f"{equilibrium.qeq_route_b(D.N_SPINS, D.BETA_DEEP, float(D.G_GRID[0]), c['mbar']):.7f}")


def main(argv: list[str]) -> int:
    cmds = {"static": cmd_static, "pre": cmd_pre, "grid": cmd_grid, "analyse": cmd_analyse}
    if len(argv) != 2 or argv[1] not in cmds:
        print(__doc__)
        return 2
    cmds[argv[1]]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
