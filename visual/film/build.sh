#!/usr/bin/env bash
# Production pipeline for the narrated film. Run from anywhere:
#
#   ./build.sh tts [ids...]          narration for all (or these) scenes, cached; ledger-limited
#   ./build.sh voices                the two-sentence sample in George, Brian, Daniel, Chris
#   ./build.sh render [-l|-h] [ids]  render scenes/<id>.py (class = CamelCase id), then sync_check
#   ./build.sh check [ids...]        sync_check.py (all rendered scenes by default)
#   ./build.sh assemble              concat + loudnorm + captions + 720p web encodes
#   ./build.sh all [-l|-h]           tts, render every scene, check, assemble (stops on a failure)
#   ./build.sh test                  the pipeline proof: tests/synctest.* at -ql, check, cue frames
#
# -l = manim -ql (480p15, previews, default); -h = manim -qh (1080p60 render, assembled to 1080p30; release).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
# MANIM_ENV: the conda/micromamba env from environment.yml (default: manim on PATH).
# TEXLIVE_BIN: a TeX Live bin folder to put first on PATH (default: PATH as is).
MANIM_ENV=${MANIM_ENV:-}
manim() {
  if [[ -n $MANIM_ENV ]]; then PYTHONNOUSERSITE=1 "$MANIM_ENV/bin/manim" "$@"
  else PYTHONNOUSERSITE=1 command manim "$@"; fi
}
if [[ -n ${TEXLIVE_BIN:-} ]]; then export PATH="$TEXLIVE_BIN:$PATH"; fi

all_ids() { $PY -c 'import json;print(" ".join(s["id"] for s in json.load(open("scenes.json"))))'; }
class_of() { $PY -c 'import sys;print("".join(p.capitalize() for p in sys.argv[1].replace("_","-").split("-")))' "$1"; }

quality=l
parse_q() { case "${1:-}" in -l) quality=l; return 0;; -h) quality=h; return 0;; esac; return 1; }

render_one() {  # id
  local id=$1 cls file
  file="scenes/$id.py"; cls=$(class_of "$id")
  if [[ ! -f $file ]]; then echo "skip $id: no $file yet"; return 0; fi
  echo "render $id ($cls, -q$quality)"
  manim "-q$quality" --media_dir media "$file" "$cls" >"build/logs/render_$id.log" 2>&1 \
    || { tail -30 "build/logs/render_$id.log"; echo "render failed: $id"; return 1; }
  $PY sync_check.py "$id"
}

mkdir -p build/logs
cmd=${1:-help}; shift || true
case "$cmd" in
  tts)     $PY tts.py "$@" ;;
  voices)  $PY tts.py voices ;;
  render)  parse_q "${1:-}" && shift || true
           ids=("$@"); [[ ${#ids[@]} -eq 0 ]] && read -ra ids <<<"$(all_ids)"
           rc=0; for id in "${ids[@]}"; do render_one "$id" || rc=1; done; exit $rc ;;
  check)   $PY sync_check.py "$@" ;;
  assemble) $PY assemble.py "$@" && $PY sync_check.py --film ;;
  all)     parse_q "${1:-}" && shift || true
           $PY tts.py
           read -ra ids <<<"$(all_ids)"
           for id in "${ids[@]}"; do render_one "$id"; done
           $PY assemble.py && $PY sync_check.py --film ;;
  test)    $PY tts.py --script tests/synctest.md
           manim -ql --media_dir media tests/synctest.py SyncTest >build/logs/render_synctest.log 2>&1 \
             || { tail -30 build/logs/render_synctest.log; exit 1; }
           $PY sync_check.py synctest
           mkdir -p build/frames
           $PY - <<'PYEOF'
import json, subprocess
t = json.load(open("build/audio/synctest.timing.json"))
ev = json.load(open("build/sync/synctest.events.json"))
for name, at in t["cues"].items():
    for when, dt in (("at", 0.0), ("after", 0.9)):
        out = f"build/frames/synctest_{name}_{when}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at + dt:.3f}", "-i", ev["movie_file"],
                        "-frames:v", "1", out], check=True)
        print(f"frame {name:10s} {when:5s} t={at + dt:6.3f} s -> {out}")
PYEOF
           ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
