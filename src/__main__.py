"""Entry point: ``python -m src`` runs the exact-chain checks."""

from __future__ import annotations

import sys

from .validate import main

if __name__ == "__main__":
    sys.exit(main())
