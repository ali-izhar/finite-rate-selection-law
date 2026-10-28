# Page data

The numbers the page loads. Each file carries `source` and `paper` keys naming where its values come
from; nothing is typed by hand. Regenerate from the repository root:

```bash
python visual/data/make_exports.py          # table1.json, cw_linear.json, fork_trajectories.json (a few seconds)
python visual/data/ref/make_reference.py    # ref/reference.json, the oracle of lib/sim.test.mjs
node visual/data/odds/make_c_vs_bias.mjs    # odds/c_vs_bias.json, with lib/sim.js (about 10 s)
```

| File | Content | Source | Paper |
|---|---|---|---|
| `table1.json` | the eight protocol rows: predicted c, measured c with its 95% interval, ratio; the p = 3 size exponent | `results/simple/` via `src.simple.estimate` | Table 1, Fig. 2b |
| `cw_linear.json` | exact-chain c on nine linear ramps against the selection law | `results/experiments/cw_linear_sweep.json` | Table 2, Fig. 2a |
| `fork_trajectories.json` | 14 drift-plus-noise paths, N = 1024, NumPy seed 4 (the page's copy of the paper's input) | `results/figures/fork_trajectories.json` | Fig. 1a |
| `odds/c_vs_bias.json` | exact-chain c on the test's ramp at 16 field values, N = 2048 (`components/odds.js` carries these values inline) | `lib/sim.js` (tested against `src/`) | Sec. 4, Table 1 |
| `ref/reference.json` | test oracle: rates, log π, q_eq, exact evolution, linear ramps, A_p, Φ | `ref/make_reference.py` (`src/`, SciPy, mpmath) | none |

`make_exports.py` copies `fork_trajectories.json` from `results/figures/`, the input of paper Fig. 1a
(written by `paper/figures/make_fork.py`). The site build copies `data/*.json` only.
