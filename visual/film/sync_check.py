#!/usr/bin/env python3
"""Verify narration/picture sync for rendered scenes. Exit 1 if any scene fails.

For every scene (default: all ids in scenes.json that have an events log):
  1. events   — every cue-bound event lands within TOL (0.12 s) of its cue;
  2. coverage — every cue in the script has at least one logged event;
  3. length   — video length = narration audio length + tail, tail in [0.5, 1.0] s (the film's
                last scene in scenes.json: [0.5, 3.0] s, so the closing frame holds before the end);
  4. silence  — no silence inside the scene longer than 1.8 s, except at a [[pause:s]] marker,
                where up to max(1.8, s + 0.6) s is allowed (measured on the rendered audio);
  5. cut      — the narration is complete in the rendered file: the audio stream is at least
                the narration length, and the last word ends at least 0.5 s before the video does.
With --film (or with no arguments), it also checks the assembled film (build/film.mp4 +
build/film.offsets.json): no scene's narration crosses into the next scene's slot; picture
and sound lengths = the sum of the scene slots; one chapter marker per scene; loudness -16 +/- 0.5
LUFS and a true peak below -1.5 dBTP; every scene boundary sits inside one silence (no narration
cut) of 0.5-1.8 s (a short natural pause); no other silence over 1.8 s except a marked pause.

Usage: python sync_check.py [scene ids...] [--json]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
TOL = 0.12
TAIL_MIN, TAIL_MAX = 0.5, 1.0
FINAL_TAIL_MAX = 3.0  # the last scene of the film: no next scene follows, so its close may hold longer
MAX_SILENCE = 1.8
PAUSE_SLACK = 0.6
SILENCE_DB = -45
LENGTH_TOL = 0.05  # one or two frames of rounding in the length checks


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    res = {"format": float(d["format"]["duration"])}
    for s in d.get("streams", []):
        if "duration" in s:
            res[s["codec_type"]] = float(s["duration"])
    return res


def silences(path: Path, min_len: float = 0.3) -> list[tuple[float, float]]:
    err = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af",
         f"silencedetect=noise={SILENCE_DB}dB:d={min_len}", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    out = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else None
        out.append((max(0.0, s), e))
    return out


def check_scene(sid: str) -> dict:
    timing_p = BUILD / "audio" / f"{sid}.timing.json"
    ev_p = BUILD / "sync" / f"{sid}.events.json"
    res = {"id": sid, "pass": True, "problems": [], "info": {}}

    def fail(msg):
        res["pass"] = False
        res["problems"].append(msg)

    if not timing_p.exists():
        fail("no timing.json (run tts.py)")
        return res
    if not ev_p.exists():
        fail("no events log (scene not rendered)")
        return res
    timing = json.loads(timing_p.read_text())
    ev = json.loads(ev_p.read_text())
    movie = Path(ev["movie_file"])
    if not movie.exists():
        fail(f"movie file missing: {movie}")
        return res

    # 1. events
    worst = 0.0
    for e in ev["events"]:
        worst = max(worst, abs(e["delta"]))
        if abs(e["delta"]) > TOL:
            fail(f"event {e['cue']} ({e['kind']}) at {e['actual']:.3f} s, cue {e['target']:.3f} s, "
                 f"off by {e['delta']:+.3f} s")
        if e["target"] != round(timing["cues"].get(e["cue"], -1), 4):
            fail(f"event {e['cue']} targets {e['target']} but the cue is now "
                 f"{timing['cues'].get(e['cue'])} (stale render: re-render the scene)")
    res["info"]["events"] = len(ev["events"])
    res["info"]["worst_event_offset"] = round(worst, 3)

    # 2. coverage
    bound = {e["cue"] for e in ev["events"]}
    missing = [c for c in timing["cues"] if c not in bound]
    if missing:
        fail(f"cues with no event: {missing}")

    # 3. length
    pr = probe(movie)
    vlen = pr.get("video", pr["format"])
    alen = timing["duration"]
    tail = vlen - alen
    res["info"].update(video=round(vlen, 3), narration=round(alen, 3), tail=round(tail, 3))
    order = [s["id"] for s in json.loads((HERE / "scenes.json").read_text())]
    tail_max = FINAL_TAIL_MAX if order and sid == order[-1] else TAIL_MAX
    if not (TAIL_MIN - LENGTH_TOL <= tail <= tail_max + LENGTH_TOL):
        fail(f"video {vlen:.3f} s = narration {alen:.3f} s + tail {tail:.3f} s; "
             f"tail must be {TAIL_MIN}-{tail_max} s")
    if abs(ev["audio_duration"] - alen) > 1e-3:
        fail("rendered against an older narration (re-render the scene)")

    # 4. silence (on the rendered audio, up to the end of speech)
    speech_end = timing["speech_end"]
    pauses = timing.get("pauses", [])
    long_ones = []
    for s, e in silences(movie):
        e_eff = min(e if e is not None else vlen, speech_end)
        if s >= speech_end:
            continue  # the tail
        length = e_eff - s
        if length <= MAX_SILENCE:
            continue
        allowed = MAX_SILENCE
        for p in pauses:
            p_end = p["end"] if p["end"] is not None else speech_end
            if s <= p_end and e_eff >= p["start"]:
                allowed = max(allowed, p["seconds"] + PAUSE_SLACK)
        if length > allowed:
            long_ones.append((round(s, 2), round(length, 2)))
    if long_ones:
        fail(f"silences over the limit (start, length): {long_ones}")
    res["info"]["longest_word_gap"] = round(max(
        [w2["start"] - w1["end"] for w1, w2 in zip(timing["words"], timing["words"][1:])] or [0]), 3)

    # 5. cut narration
    a_stream = pr.get("audio")
    if a_stream is None:
        fail("rendered file has no audio stream")
    elif a_stream + LENGTH_TOL < alen:
        fail(f"audio stream {a_stream:.3f} s is shorter than the narration {alen:.3f} s")
    if speech_end > vlen - TAIL_MIN + LENGTH_TOL:
        fail(f"last word ends at {speech_end:.3f} s, video ends at {vlen:.3f} s")
    return res


def check_film() -> list[str]:
    """The assembled film: slots, lengths, chapters, loudness, and the sound at every boundary."""
    film, off_p = BUILD / "film.mp4", BUILD / "film.offsets.json"
    if not (film.exists() and off_p.exists()):
        return []
    probs = []
    meta = json.loads(off_p.read_text())
    offs = meta["scenes"]
    pr = probe(film)
    total = sum(o["duration"] for o in offs)
    for kind in ("video", "audio"):
        if abs(pr.get(kind, pr["format"]) - total) > LENGTH_TOL:
            probs.append(f"film: {kind} is {pr.get(kind, pr['format']):.3f} s, scenes add to {total:.3f} s")
    for o1, o2 in zip(offs, offs[1:]):
        if abs(o1["start"] + o1["duration"] - o2["start"]) > 1e-3:
            probs.append(f"film: {o2['id']} does not start where {o1['id']} ends")
    if "loudness_I" in meta and not (abs(meta["loudness_I"] + 16) <= 0.5 and meta["true_peak"] < -1.5):
        probs.append(f"film: loudness {meta['loudness_I']} LUFS / true peak {meta['true_peak']} dBTP")
    chaps = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_chapters", "-of", "json",
                                       str(film)], capture_output=True, text=True).stdout)["chapters"]
    if len(chaps) != len(offs):
        probs.append(f"film: {len(chaps)} chapter markers for {len(offs)} scenes")

    sil = [(s, e if e is not None else pr["format"]) for s, e in silences(film)]
    timings = [json.loads((BUILD / "audio" / f"{o['id']}.timing.json").read_text()) for o in offs]
    # pauses the script marks, on the film clock
    allowed = []
    for o, t in zip(offs, timings):
        for p in t.get("pauses", []):
            p_end = p["end"] if p["end"] is not None else t["speech_end"]
            allowed.append((o["start"] + p["start"], o["start"] + p_end, p["seconds"] + PAUSE_SLACK))
    for i, (o, t) in enumerate(zip(offs, timings)):
        end_speech = o["start"] + t["speech_end"]
        slot_end = offs[i + 1]["start"] if i + 1 < len(offs) else pr["format"]
        if end_speech > slot_end - TAIL_MIN + LENGTH_TOL:
            probs.append(f"film: {o['id']} narration ends at {end_speech:.2f} s, slot ends {slot_end:.2f} s")
        if i + 1 == len(offs):
            continue
        # the boundary must sit inside one silence (no narration cut), and that pause must be natural
        b = slot_end
        gap = [(s, e) for s, e in sil if s <= b - 0.3 and e >= b - 0.05]
        if not gap:
            probs.append(f"film: sound at the {o['id']}/{offs[i+1]['id']} boundary ({b:.2f} s)")
            continue
        g = gap[0][1] - gap[0][0]
        if not (TAIL_MIN <= g <= MAX_SILENCE):
            probs.append(f"film: pause between {o['id']} and {offs[i+1]['id']} is {g:.2f} s")
    for s, e in sil:
        if e - s <= MAX_SILENCE or e >= pr["format"] - 0.05 or s <= 0.05:
            continue
        lim = max([m for a, z, m in allowed if s <= z and e >= a] or [MAX_SILENCE])
        if e - s > lim:
            probs.append(f"film: silence of {e - s:.2f} s at {s:.2f} s")
    return probs


def film_boundaries() -> list[dict]:
    """The pause at every scene boundary of the film (for the report)."""
    film, off_p = BUILD / "film.mp4", BUILD / "film.offsets.json"
    offs = json.loads(off_p.read_text())["scenes"]
    sil = [(s, e) for s, e in silences(film) if e is not None]
    out = []
    for o1, o2 in zip(offs, offs[1:]):
        b = o2["start"]
        g = [(s, e) for s, e in sil if s <= b - 0.3 and e >= b - 0.05]
        out.append({"from": o1["id"], "to": o2["id"], "at": b,
                    "pause": round(g[0][1] - g[0][0], 3) if g else None})
    return out


def main(argv):
    as_json = "--json" in argv
    ids = [a for a in argv if not a.startswith("--")]
    if not ids:
        ids = [s["id"] for s in json.loads((HERE / "scenes.json").read_text())
               if (BUILD / "sync" / f"{s['id']}.events.json").exists()]
    results = [check_scene(i) for i in ids]
    film_probs = check_film() if not argv or "--film" in argv or not ids else []
    ok = all(r["pass"] for r in results) and not film_probs
    if as_json:
        print(json.dumps({"pass": ok, "scenes": results, "film": film_probs}, indent=1))
    else:
        for r in results:
            i = r["info"]
            print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['id']:10s} events={i.get('events', '-')} "
                  f"worst={i.get('worst_event_offset', '-')}s video={i.get('video', '-')}s "
                  f"narration={i.get('narration', '-')}s tail={i.get('tail', '-')}s "
                  f"longest_gap={i.get('longest_word_gap', '-')}s")
            for p in r["problems"]:
                print(f"      - {p}")
        for p in film_probs:
            print(f"FAIL  {p}")
        if "--film" in argv and (BUILD / "film.offsets.json").exists():
            for g in film_boundaries():
                print(f"boundary {g['from']:>9s} -> {g['to']:9s} at {g['at']:8.3f} s  pause {g['pause']} s")
            print("film: " + ("PASS" if not film_probs else "FAIL"))
        if not results:
            print("no rendered scenes to check")
    return 0 if ok and results else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
