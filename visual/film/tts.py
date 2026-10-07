#!/usr/bin/env python3
"""Narration TTS: script.md -> build/audio/<id>.mp3 + build/audio/<id>.timing.json.

Every ElevenLabs call goes through ``synthesize()``, which
  * caches on disk: build/tts/<hash>.json (alignment + request metadata) and <hash>.mp3,
    hash = sha256(voice, model, settings, seed, text);
  * keeps a character ledger, build/tts/ledger.json, and refuses any call that would take the
    project past LEDGER_LIMIT characters (cache hits cost nothing).

Request stitching: each scene is sent with ``previous_text`` / ``next_text`` taken from the
neighbouring scenes' spoken text, so the prosody flows across scene boundaries. The context is
NOT part of the cache key: editing one scene regenerates only that scene.

Loudness is not touched here; assemble.py normalises the whole film.

Usage:
  python tts.py                    # every scene of script.md
  python tts.py hook race          # only these scenes
  python tts.py --script tests/synctest.md
  python tts.py --dry-run          # show what would be generated and the character cost
  python tts.py voices             # the same two-sentence sample in four voices
  python tts.py ledger             # print the ledger total

Voice: the voice id, model and voice settings (stability, similarity_boost, style,
use_speaker_boost, speed) come from voice.json (default: next to this file; override with FILM_VOICE_FILE). The file is re-read at
every run, so a new choice of voice only needs a rerun; the cache key includes the voice, the
model and every setting, so a change regenerates every scene and an unchanged choice costs nothing.

The API key is read at call time from ../../.env (ELEVEN_LABS_API_KEY; override the path with
FILM_ENV_FILE) and never printed or stored.
"""
from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from scriptparse import Scene, parse_script, word_spans

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ENV_FILE = Path(os.environ.get("FILM_ENV_FILE", REPO / ".env"))
VOICE_FILE = Path(os.environ.get("FILM_VOICE_FILE", HERE / "voice.json"))
BUILD = HERE / "build"
TTS_DIR = BUILD / "tts"
AUDIO_DIR = BUILD / "audio"
VOICE_DIR = BUILD / "voices"
LEDGER = TTS_DIR / "ledger.json"
LEDGER_LIMIT = 60_000

VOICES = {
    "george": "JBFqnCBsd6RMkjVDRZzb",
    "brian": "nPczCjzI2devNBz1zQrb",
    "daniel": "onwK4e9ZLuTAKqWW03F9",
    "bill": "pqHfZKP75CyOlxRsgubV",
    "chris": "iP95p4xoKVk53GoZ742B",
}
# Fallback when voice.json is missing: Brian, calm narration settings.
FALLBACK = {"voice_name": "brian", "voice_id": VOICES["brian"], "model_id": "eleven_multilingual_v2",
            "settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0,
                         "use_speaker_boost": True}}
SETTING_KEYS = ("stability", "similarity_boost", "style", "use_speaker_boost", "speed")


def load_voice() -> dict:
    """The current voice choice: {voice_name, voice_id, model_id, settings, status}."""
    try:
        v = json.loads(VOICE_FILE.read_text())
    except FileNotFoundError:
        v = dict(FALLBACK, status=f"fallback: {VOICE_FILE} not found")
    v["settings"] = {k: v["settings"][k] for k in SETTING_KEYS if k in v.get("settings", {})}
    v.setdefault("voice_name", v["voice_id"])
    v.setdefault("status", "")
    return v


VOICE = load_voice()
DEFAULT_VOICE = VOICE["voice_name"]
VOICES.setdefault(DEFAULT_VOICE, VOICE["voice_id"])
MODEL = VOICE["model_id"]
SETTINGS = VOICE["settings"]
SEED = 1983  # fixed seed: a re-request of the same text is as repeatable as the API allows
OUTPUT_FORMATS = ["mp3_44100_192", "mp3_44100_128"]  # first one the account allows wins
CONTEXT_CHARS = 1000  # characters of neighbouring text sent as previous_text / next_text

VOICE_SAMPLE = (
    "Here is a generative model with exactly two kinds of output. "
    "Picture them as a three, and an eight."
)  # script.md, scene hook, the first two sentences
VOICE_SAMPLE_NAMES = ["george", "brian", "daniel", "chris"]


# ---------------------------------------------------------------- key, ledger, cache
def _api_key() -> str:
    try:
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("ELEVEN_LABS_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    except FileNotFoundError:
        pass
    raise SystemExit(f"ELEVEN_LABS_API_KEY not found in {ENV_FILE}")


def load_ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"limit": LEDGER_LIMIT, "total_chars": 0, "entries": []}


def _save_ledger(led: dict) -> None:
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(led, indent=1))
    tmp.replace(LEDGER)


def cache_key(voice_id: str, text: str) -> str:
    blob = json.dumps(
        {"voice": voice_id, "model": MODEL, "settings": SETTINGS, "seed": SEED, "text": text},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _post(voice_id: str, body: dict, fmt: str) -> tuple[int, bytes]:
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
           f"?output_format={fmt}")
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"xi-api-key": _api_key(), "Content-Type": "application/json",
                 "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def synthesize(text: str, voice: str = DEFAULT_VOICE, previous_text: str = "",
               next_text: str = "", label: str = "") -> dict:
    """Return the cache record {key, mp3, alignment, ...}; call the API only on a cache miss."""
    voice_id = VOICES.get(voice, voice)
    key = cache_key(voice_id, text)
    meta_p, mp3_p = TTS_DIR / f"{key}.json", TTS_DIR / f"{key}.mp3"
    if meta_p.exists() and mp3_p.exists():
        rec = json.loads(meta_p.read_text())
        rec["cached"] = True
        return rec

    led = load_ledger()
    cost = len(text)
    if led["total_chars"] + cost > LEDGER_LIMIT:
        raise SystemExit(
            f"ledger refuses {label or key}: {led['total_chars']} + {cost} > {LEDGER_LIMIT} chars")

    body = {"text": text, "model_id": MODEL, "voice_settings": SETTINGS, "seed": SEED}
    if previous_text:
        body["previous_text"] = previous_text[-CONTEXT_CHARS:]
    if next_text:
        body["next_text"] = next_text[:CONTEXT_CHARS]

    status, payload, fmt_used = None, b"", None
    for fmt in OUTPUT_FORMATS:
        for attempt in range(4):
            try:
                status, payload = _post(voice_id, body, fmt)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                # the request may have been billed; count it conservatively
                led["total_chars"] += cost
                led["entries"].append({"key": key, "label": label, "chars": cost,
                                       "status": "network-error", "t": time.time()})
                _save_ledger(led)
                raise SystemExit(f"network error for {label or key}: {type(e).__name__}")
            if status in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt * 3)
                continue
            break
        if status == 200:
            fmt_used = fmt
            break
        # a format the plan does not allow comes back as 4xx: try the next format
        msg = payload[:300].decode("utf-8", "replace")
        if status in (400, 401, 403, 422) and "format" in msg.lower():
            continue
        raise SystemExit(f"TTS failed for {label or key}: HTTP {status}: {msg}")
    if status != 200:
        raise SystemExit(f"TTS failed for {label or key}: HTTP {status}")

    data = json.loads(payload)
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    mp3_p.write_bytes(base64.b64decode(data["audio_base64"]))
    rec = {
        "key": key, "label": label, "voice": voice, "voice_id": voice_id,
        "voice_status": VOICE.get("status", ""), "model": MODEL,
        "settings": SETTINGS, "seed": SEED, "output_format": fmt_used, "text": text,
        "previous_text_chars": len(body.get("previous_text", "")),
        "next_text_chars": len(body.get("next_text", "")),
        "alignment": data.get("alignment"),
        "normalized_alignment": data.get("normalized_alignment"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_p.write_text(json.dumps(rec))
    led["total_chars"] += cost
    led["entries"].append({"key": key, "label": label, "chars": cost, "voice": voice,
                           "voice_id": voice_id, "status": "ok", "t": time.time()})
    _save_ledger(led)
    rec["cached"] = False
    return rec


# ---------------------------------------------------------------- timing
def mp3_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def _sent_to_align(sent: str, chars: list[str]) -> list[int | None]:
    """Map every index of the sent text to an index of the alignment character list."""
    joined = "".join(chars)
    if joined == sent and all(len(c) == 1 for c in chars):
        return list(range(len(sent)))
    # alignment entries can be multi-char in rare cases; expand to a per-char list
    flat, owner = [], []
    for i, c in enumerate(chars):
        for ch in c:
            flat.append(ch)
            owner.append(i)
    sm = difflib.SequenceMatcher(a=sent, b="".join(flat), autojunk=False)
    out: list[int | None] = [None] * len(sent)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            out[a + k] = owner[b + k]
    # fill gaps from the nearest matched neighbour on the left (then right)
    last = None
    for i in range(len(out)):
        if out[i] is None:
            out[i] = last
        else:
            last = out[i]
    nxt = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] is None:
            out[i] = nxt
        else:
            nxt = out[i]
    return out


def build_timing(sc: Scene, rec: dict, mp3: Path) -> dict:
    al = rec["alignment"]
    chars = al["characters"]
    st, en = al["character_start_times_seconds"], al["character_end_times_seconds"]
    s2a = _sent_to_align(sc.sent, chars)

    def t_start(ci: int) -> float:
        return round(float(st[s2a[sc.clean_to_sent[ci]]]), 3)

    def t_end(ci: int) -> float:
        return round(float(en[s2a[sc.clean_to_sent[ci]]]), 3)

    words = [{"w": w, "start": t_start(a), "end": t_end(b - 1), "offset": a}
             for a, b, w in word_spans(sc.clean)]
    cues = {}
    for name, off in sc.cues.items():
        cues[name] = t_start(off)
    pauses = []
    for p in sc.pauses:
        before = [w for w in words if w["offset"] < p["offset"]]
        after = [w for w in words if w["offset"] >= p["offset"]]
        pauses.append({"offset": p["offset"], "seconds": p["seconds"],
                       "start": before[-1]["end"] if before else 0.0,
                       "end": after[0]["start"] if after else None})
    dur = mp3_duration(mp3)
    return {
        "id": sc.id, "title": sc.title, "chapter": sc.chapter,
        "audio": str(mp3.relative_to(HERE)), "duration": round(dur, 3),
        "speech_end": words[-1]["end"] if words else 0.0,
        "voice": rec["voice"], "voice_id": rec.get("voice_id"), "model": rec["model"],
        "settings": rec.get("settings"), "tts_key": rec["key"],
        "text": sc.clean, "cues": dict(sorted(cues.items(), key=lambda kv: kv[1])),
        "pauses": pauses, "words": words,
    }


# ---------------------------------------------------------------- commands
def run_scenes(script: Path, only: list[str], voice: str, dry: bool) -> None:
    scenes = parse_script(script)
    ids = [s.id for s in scenes]
    unknown = [i for i in only if i not in ids]
    if unknown:
        raise SystemExit(f"unknown scene ids: {unknown}")
    todo = [s for s in scenes if not only or s.id in only]
    led = load_ledger()
    new_chars = 0
    for s in todo:
        hit = (TTS_DIR / f"{cache_key(VOICES.get(voice, voice), s.sent)}.mp3").exists()
        if not hit:
            new_chars += len(s.sent)
        print(f"{s.id:10s} {len(s.sent):5d} chars  {'cached' if hit else 'NEW'}")
    print(f"voice {voice} ({VOICES.get(voice, voice)}), model {MODEL}, settings {SETTINGS}"
          + (f" [{VOICE['status']}]" if VOICE.get("status") else ""))
    print(f"new chars {new_chars}; ledger {led['total_chars']} / {LEDGER_LIMIT}")
    if dry:
        return
    if led["total_chars"] + new_chars > LEDGER_LIMIT:
        raise SystemExit("ledger limit would be exceeded; nothing generated")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for s in todo:
        i = ids.index(s.id)
        prev_t = scenes[i - 1].clean if i > 0 else ""
        next_t = scenes[i + 1].clean if i + 1 < len(scenes) else ""
        rec = synthesize(s.sent, voice, prev_t, next_t, label=f"scene:{s.id}")
        mp3 = AUDIO_DIR / f"{s.id}.mp3"
        shutil.copyfile(TTS_DIR / f"{rec['key']}.mp3", mp3)
        timing = build_timing(s, rec, mp3)
        (AUDIO_DIR / f"{s.id}.timing.json").write_text(json.dumps(timing, indent=1))
        print(f"  {s.id:10s} {timing['duration']:7.2f} s  "
              f"{'(cache)' if rec['cached'] else '(new)'}  cues={len(timing['cues'])}")
    print(f"ledger total {load_ledger()['total_chars']} / {LEDGER_LIMIT}")


def run_voices() -> None:
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    for name in VOICE_SAMPLE_NAMES:
        rec = synthesize(VOICE_SAMPLE, name, label=f"voice:{name}")
        out = VOICE_DIR / f"{name}.mp3"
        shutil.copyfile(TTS_DIR / f"{rec['key']}.mp3", out)
        print(f"{name:8s} {mp3_duration(out):5.2f} s {'(cache)' if rec['cached'] else '(new)'}")
    print(f"ledger total {load_ledger()['total_chars']} / {LEDGER_LIMIT}")


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*", help="scene ids, or 'voices', or 'ledger'")
    ap.add_argument("--script", default=str(HERE / "script.md"))
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(HERE)
    if a.scenes == ["voices"]:
        return run_voices()
    if a.scenes == ["ledger"]:
        led = load_ledger()
        print(f"ledger total {led['total_chars']} / {LEDGER_LIMIT} ({len(led['entries'])} calls)")
        return
    run_scenes(Path(a.script), a.scenes, a.voice, a.dry_run)


if __name__ == "__main__":
    main(sys.argv[1:])
