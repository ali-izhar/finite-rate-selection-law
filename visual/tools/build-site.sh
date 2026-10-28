#!/bin/sh
# Build the static site of the visual edition into visual/_site/ (GitHub Pages).
# Run from anywhere:  sh visual/tools/build-site.sh [--fragment]
# --fragment writes visual/dist/ instead: the page without <html>/<head>/<body>,
# for hosts that add their own document wrapper.
set -e
cd "$(dirname "$0")/.."
out=_site
if [ "${1:-}" = "--fragment" ]; then out=dist; fi
rm -rf "$out" && mkdir -p "$out"
if [ "$out" = dist ]; then
  sed -e '1{/^<!doctype html>$/d}' -e '/^<html lang="en">$/d' -e '/^<\/html>$/d' \
      -e '/^<head>$/d' -e '/^<\/head>$/d' -e '/^<body>$/d' -e '/^<\/body>$/d' index.html > "$out/index.html"
else
  cp index.html "$out/"
fi
cp -r lib components assets "$out/"
rm -f "$out/lib/sim.test.mjs"
# data: only what the page loads; generators and test oracles stay out
mkdir -p "$out/data" && cp data/*.json "$out/data/"
# fonts: the woff2 files the theme loads, KaTeX's stylesheet and fonts, and their licenses
mkdir -p "$out/fonts" && cp fonts/*.woff2 fonts/LICENSES.md fonts/OFL-*.txt "$out/fonts/" && cp -r fonts/katex "$out/fonts/"
# media: the narrated film and the chapter clips (committed; node tools/media.mjs rebuilds them)
if [ ! -f media/media.json ]; then echo "build-site: media/media.json is missing" >&2; exit 1; fi
cp -r media "$out/"
big=$(find "$out/media" -type f -size +14648k)
if [ -n "$big" ]; then echo "build-site: file over 15 MB: $big" >&2; exit 1; fi
touch "$out/.nojekyll"
echo "build-site: visual/$out, $(du -sh "$out" | cut -f1)"
