#!/bin/sh
# Build the poster: figures (Python) -> QR code, inlined page, PDF and PNG preview (Node + Playwright).
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-../.venv/bin/python}"
[ -x "$PY" ] || PY=python3
[ -d node_modules/katex ] || npm install --no-audit --no-fund
"$PY" make_figures.py
node build.mjs
node check_qr.mjs
command -v pdfinfo >/dev/null && pdfinfo poster.pdf | grep -E "Pages|Page size"
