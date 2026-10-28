# Checks

What each check of the paper verifies. Commands run from the repository root; none needs a GPU.
[docs/reproduce.md](../docs/reproduce.md) lists every command.

| Command | What it checks | Time |
|---|---|---|
| `python -m checks.claims` | Recomputes every number the paper prints, at the printed precision; [claims.md](claims.md) lists each row and its paper location. Exits 1 on any FAIL. | about 4 min |
| `python -m checks.paper_consistency [--paper paper/build]` | Compares Tables 1-3, the printed constants and seeds with `results/` and `src/simple/design.py`, checks that the earlier protocol's elements are gone, and checks the built PDF's pages, references and fonts. Needs the PDF build and poppler (`pdftotext`, `pdffonts`). | about 10 s |
| `python -m checks.reanalysis [--output FILE]` | Reanalyses the saved counts of the earlier protocol (field grid, pattern-count scans), which the paper no longer uses; kept for the visual edition. Prints JSON. | seconds |

`python -m src` checks the exact-chain implementation itself (82 checks); `checks.claims` runs it as one row.
`python -m pytest -q src/simple/tests` checks the code of the Hopfield test (kernel, gauge identity,
estimator); its GPU test is skipped without torch.
