"""python -m src.simple <step>   (run from the repository root)

  predict                     frozen predictions, CPU          (OVERWRITES results/simple/predictions.json)
  g0                          code gate G0, torch vs NumPy     -> results/simple/g0.json
  pilot  [--batch B]          throughput + p = 1 check at N = 256; not analysed -> pilot.json
  row N p [--batch B] [--seed-offset k]   one pre-registered row -> row_N{N}_p{p}[_rerun{k}].json
  rows   [--batch B] [--seed-offset k]    every row of design.ROWS, in order
  analyse [--seed-offset k]   gates G0-G2, then the A_p rows   -> analysis.json   (CPU)

GPU steps need torch.  On the GB10, from the repository root (files owned by the user):

  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$PWD:/workspace" -w /workspace nvcr.io/nvidia/pytorch:26.04-py3 python -m src.simple <step>

The batch size changes which random draw goes to which chain, so it is recorded in every row file;
the pre-registered runs use the default.
"""
from __future__ import annotations

import argparse
import sys

DEFAULT_BATCH = 131_000          # a multiple of 2K = 100, so each batch is balanced over cells


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.simple", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["predict", "g0", "pilot", "row", "rows", "analyse"])
    ap.add_argument("args", nargs="*", type=int)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--seed-offset", type=int, default=0)
    a = ap.parse_args(argv)
    if a.step == "predict":
        from src.simple.predict import main as predict
        predict()
    elif a.step == "analyse":
        from src.simple.analyse import main as analyse
        analyse(a.seed_offset)
    else:
        from src.simple import run
        if a.step == "g0":
            return 0 if run.step_g0()["result"]["passed"] else 1
        if a.step == "pilot":
            run.step_pilot(a.batch)
        elif a.step == "row":
            if len(a.args) != 2:
                ap.error("row needs N p")
            run.step_row(a.args[0], a.args[1], a.batch, a.seed_offset)
        elif a.step == "rows":
            from src.simple import design as D
            for n, p in D.ROWS:
                run.step_row(n, p, a.batch, a.seed_offset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
