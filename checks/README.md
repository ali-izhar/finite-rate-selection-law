# Checks

What each check of the paper verifies. Commands run from the repository root; none needs a GPU.
[docs/reproduce.md](../docs/reproduce.md) lists every command.

| Command | What it checks | Time |
|---|---|---|
| `python -m checks.claims` | Recomputes every number the paper prints, at the printed precision; [claims.md](claims.md) lists each row and its paper location. Exits 1 on any FAIL. | about 2 min |
| `python -m checks.paper_consistency [--paper paper/build]` | Compares the tables, captions and seeds with the saved results, and checks the built PDF's pages, references and fonts. Needs the PDF build and poppler (`pdftotext`, `pdffonts`). | seconds |
| `python -m checks.reanalysis [--output FILE]` | Reanalyses the saved counts: the field-grid slope under the paper's weights and three alternatives, both pattern-set bootstraps, the logistic fit, the p = 2 block sizes, the pattern-overlap eigenvalues. Prints JSON. | seconds |

`python -m src` checks the exact-chain implementation itself (82 checks); `checks.claims` runs it as one row.
