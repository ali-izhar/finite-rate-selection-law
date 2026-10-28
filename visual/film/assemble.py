#!/usr/bin/env python3
"""Assemble the film from rendered scenes: concat, transitions, loudness, captions, chapters,
poster and web encodes.

  python assemble.py [--scenes id ...] [--no-web] [--web-only] [--xfade 0.5]

  --web-only  redo step 7 only, from the existing build/film.mp4 and its offsets file

Steps
  1. Order = scenes.json (or --scenes). Each scene's picture = its rendered mp4 (path from
     build/sync/<id>.events.json, i.e. the last render). Its sound = the original narration mp3
     (build/audio/<id>.mp3, not manim's re-encoded AAC), padded with silence to the scene's
     length, so the narration starts at the scene's t = 0 exactly as NarratedScene placed it and
     the tail is the pause before the next scene. Each scene's length is floored to a whole number
     of film frames (30 fps), and picture and sound are cut to that same length. Where the
     narration file ends in more than OUT_PAUSE_MAX (1.5 s) of silence plus tail, the held last
     frame is cut so the boundary pause stays natural (never below narration + 0.5 s). The last
     scene has no boundary after it: its held closing frame is kept whole.
  2. Transitions: every boundary is a change of chapter, so the held last frame of a scene
     dissolves (--xfade s, default 0.5) into the first frame of the next. The dissolve sits inside
     the silent tail of the outgoing scene and the next scene starts at its own t = 0, so no scene
     is shortened or shifted: the film length is the sum of the scene lengths and the sound is
     untouched.
  3. Concat and re-encode at 1080p30: H.264 (yuv420p, CRF 18, slow), audio PCM in an intermediate.
  4. Loudness: two-pass ffmpeg loudnorm to -16 LUFS integrated, true peak -3.0 dBTP target (the AAC
     encode overshoots it by about 0.6 dB; requirement: below -1.5 dBTP), 48 kHz, AAC 192k -> build/film.mp4, with chapter
     markers (one per scene, named after the scene title); verified by an ebur128 pass, which must
     read -16 +/- 0.5 LU and a true peak below -1.5 dBTP.
  5. Captions (captions.py): the narration's word timings with display text (numerals and
     symbols for the spoken forms), cut at sentence and clause ends -> build/film.vtt and
     build/captions/<id>.vtt (<= 42 characters per line, <= 2 lines per cue).
  6. Poster: build/film-poster.png (the hook's title card, 1920x1080).
  7. Web encodes for the page (720p30, H.264 + AAC, faststart):
     build/web/film-720p.mp4, the whole film in one file under 14 MB (the page's
     same 14 MB cap as the clips): two-pass H.264 at the largest bitrate that fits (stepped down
     and re-encoded while the file is over), with
     AAC 64k (the narration is mono, so the two identical channels cost little more than one),
     after a 4x-oversampled limiter at -5 dBFS (low-bitrate AAC overshoots the true peak); each
     web file is measured: -16 +/- 0.5 LUFS and a true peak below -1.5 dBTP, or exit 1,
     + .vtt; and build/web/<id>.mp4 per scene (CRF 20, AAC 128k, < 14 MB) + <id>.vtt. A scene clip takes
     its picture from the scene's own render (it ends on the scene's held last frame, not on the
     dissolve) and its sound from the normalised film. One scene alone can sit 1-2 LU off the
     film, so each clip gets its own gain: measured on its slice of the film, then corrected once
     on the encoded clip (gain_db in the offsets file), so every clip reaches -16 LUFS.
Writes build/film.offsets.json (scene start times in the film) for sync_check.py --film.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

from captions import caption_cues, lint as caption_lint

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
TARGET_I, TARGET_TP, TARGET_LRA = -16.0, -3.0, 11.0
TP_LIMIT = -1.5
SILENCE_DB = -45
OUT_PAUSE_MAX = 1.5  # audible end of a scene's narration -> end of its slot, at most
WEB_LIMIT = 14_000_000
FILM_LIMIT = 14_000_000  # the whole film, one file: the same 14 MB cap as every web file
FILM_ABR_K = 64
# A low-bitrate AAC encode overshoots the master's true peak (the 64k film read -0.6 dBTP).
# Every web file therefore passes a 4x-oversampled limiter at -5 dBFS first: speech peaks only,
# so the loudness drops by 0.3-0.6 LU; a +0.4 dB make-up gain after it restores -16 LUFS (true
# peaks of the web files stay at -2.0 dBTP or below). Then each web file is measured like the master.
WEB_AF = ("aresample=192000,alimiter=limit=0.56:level=false:attack=2:release=40,volume=0.4dB,"
          "aresample=48000")
# The 64k film overshoots more (the -5 dBFS limiter read -1.5 dBTP; -6 dBFS read -1.4 dBTP): limit at -6.6 dBFS, +0.7 dB.
FILM_AF = ("aresample=192000,alimiter=limit=0.47:level=false:attack=2:release=40,volume=0.7dB,"
           "aresample=48000")


def sh(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-3000:])
        raise SystemExit(f"command failed: {' '.join(cmd[:6])} ...")
    return r


def probe_dur(path: Path, stream: str = "v:0") -> float:
    out = sh(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries",
              "stream=duration", "-of", "csv=p=0", str(path)]).stdout.strip()
    if out in ("", "N/A"):  # Matroska keeps stream lengths in a tag
        tag = sh(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries",
                  "stream_tags=DURATION", "-of", "csv=p=0", str(path)]).stdout.strip()
        h, m, sec = tag.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    return float(out)


def audible_end(path: Path) -> float:
    """End of the last sound above SILENCE_DB (the sync check's threshold) in a narration file."""
    err = sh(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
              f"silencedetect=noise={SILENCE_DB}dB:d=0.2", "-f", "null", "-"]).stderr
    total = probe_dur(path, "a:0")
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    if starts and (len(ends) < len(starts) or ends[-1] >= total - 0.05):
        return starts[-1]
    return total


def probe_size(path: Path) -> tuple[int, int, str]:
    out = sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
              "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(path)]).stdout.strip()
    w, h, r = out.split(",")
    return int(w), int(h), r


# ------------------------------------------------------------------ captions
def fmt_ts(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def write_vtt(path: Path, cues: list[tuple[float, float, str]]) -> None:
    out = ["WEBVTT", ""]
    for i, (a, b, t) in enumerate(cues, 1):
        out += [str(i), f"{fmt_ts(a)} --> {fmt_ts(b)}", t, ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))


# ------------------------------------------------------------------ helpers
def ffmeta_chapters(path: Path, offs: list[dict], titles: dict, film_title: str) -> None:
    def esc(x: str) -> str:
        return re.sub(r"([=;#\\\n])", r"\\\1", x)
    out = [";FFMETADATA1", f"title={esc(film_title)}", ""]
    for o in offs:
        a = int(round(o["start"] * 1000))
        b = int(round((o["start"] + o["duration"]) * 1000))
        out += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={a}", f"END={b}",
                f"title={esc(titles[o['id']])}", ""]
    path.write_text("\n".join(out))


def x264(crf: float | None = None, kbps: int | None = None, maxrate: int | None = None) -> list[str]:
    a = ["-c:v", "libx264", "-preset", "slow", "-tune", "animation", "-pix_fmt", "yuv420p",
         "-g", "120", "-profile:v", "high"]
    if crf is not None:
        a += ["-crf", str(crf)]
    if kbps is not None:
        a += ["-b:v", f"{kbps}k"]
    if maxrate is not None:
        a += ["-maxrate", f"{maxrate}k", "-bufsize", f"{2 * maxrate}k"]
    return a


def two_pass(src_args: list[str], vf: str, kbps: int, abr: str, dst: Path, tag: str,
             af: str = WEB_AF) -> None:
    log = BUILD / "logs" / f"x264_{tag}"
    log.parent.mkdir(parents=True, exist_ok=True)
    common = ["-vf", vf] + x264(kbps=kbps, maxrate=int(kbps * 2.5)) + ["-passlogfile", str(log)]
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + src_args + common
       + ["-pass", "1", "-an", "-f", "mp4", "/dev/null"])
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + src_args + common
       + ["-pass", "2", "-af", af, "-c:a", "aac", "-b:a", abr, "-movflags", "+faststart", str(dst)])
    for f in log.parent.glob(log.name + "*"):
        f.unlink()


def web_encode(src_args: list[str], dur: float, dst: Path, tag: str, abr_k: int = 128,
               gain_db: float = 0.0) -> dict:
    """720p30. CRF 20 when that fits under the limit; else a two-pass encode at the largest
    bitrate that fits. `gain_db` is the clip's own loudness gain, applied before WEB_AF.
    Returns the encode's video bitrate (kbps) and mode."""
    vf = "scale=-2:720:flags=lanczos,fps=30,format=yuv420p"
    af = f"volume={gain_db:.2f}dB,{WEB_AF}"
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + src_args + ["-vf", vf] + x264(crf=20)
       + ["-af", af, "-c:a", "aac", "-b:a", f"{abr_k}k", "-movflags", "+faststart", str(dst)])
    if dst.stat().st_size < WEB_LIMIT:
        mode = "crf20"
    else:
        kbps = int(WEB_LIMIT * 8 * 0.94 / dur / 1000 - abr_k - 4)
        two_pass(src_args, vf, kbps, f"{abr_k}k", dst, tag, af)
        mode = f"2pass {kbps}k"
    size = dst.stat().st_size
    vk = (size * 8 / dur / 1000) - abr_k
    return {"file": dst.name, "seconds": round(dur, 3), "bytes": size, "video_kbps": round(vk),
            "mode": mode, "under_limit": size < WEB_LIMIT}


# ------------------------------------------------------------------ main
def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--web-only", action="store_true")
    ap.add_argument("--xfade", type=float, default=0.5)
    ap.add_argument("--out", default=str(BUILD / "film.mp4"))
    a = ap.parse_args(argv)

    meta = {s["id"]: s for s in json.loads((HERE / "scenes.json").read_text())}
    if a.web_only:
        return web_only(Path(a.out))
    order = a.scenes or list(meta)
    items = []
    for sid in order:
        ev_p = BUILD / "sync" / f"{sid}.events.json"
        if not ev_p.exists():
            raise SystemExit(f"scene {sid} is not rendered (no {ev_p.name})")
        ev = json.loads(ev_p.read_text())
        timing = json.loads((BUILD / "audio" / f"{sid}.timing.json").read_text())
        movie = Path(ev["movie_file"])
        items.append({"id": sid, "movie": movie, "mp3": HERE / timing["audio"],
                      "vdur": probe_dur(movie), "timing": timing, "height": ev["pixel_height"]})
    heights = {it["height"] for it in items}
    if len(heights) > 1:
        print(f"warning: mixed render qualities {sorted(heights)}; scaling to the largest")
    W, H = max((probe_size(it["movie"])[:2] for it in items), key=lambda wh: wh[1])
    fps = 30 if H >= 720 else 15
    for it in items:  # whole film frames; drops at most one frame of the silent tail
        it["d"] = math.floor(it["vdur"] * fps + 0.01) / fps
        # Some narrations end in up to 1.1 s of digital silence that the TTS alignment counts
        # inside the last word; with the scene's tail that makes a boundary pause over 1.8 s.
        # Cut the held last frame so the audible end is at most OUT_PAUSE_MAX before the slot
        # end, never below the narration's length + 0.5 s (the scene-level tail minimum).
        heard = audible_end(it["mp3"])
        speech_end = it["timing"]["speech_end"]
        cap = math.ceil(max(heard + OUT_PAUSE_MAX, speech_end + 0.5) * fps - 0.01) / fps
        if it is not items[-1]:  # the film's last scene keeps its whole closing hold
            it["d"] = min(it["d"], cap)
        tail = it["d"] - heard
        if tail < a.xfade + 0.2:
            raise SystemExit(f"{it['id']}: tail {tail:.2f} s is too short for a {a.xfade} s dissolve")

    # 1-3. picture with dissolves, sound concatenated, both cut to the same scene lengths
    raw = BUILD / "film_raw.mkv"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for it in items:
        cmd += ["-i", str(it["movie"]), "-i", str(it["mp3"])]
    fc, acat = [], ""
    X = a.xfade
    for i, it in enumerate(items):
        d = f"{it['d']:.6f}"
        pre = f"tpad=start_duration={X:.3f}:start_mode=clone," if i > 0 and X > 0 else ""
        fc.append(f"[{2*i}:v]scale={W}:{H}:flags=lanczos,fps={fps},setsar=1,format=yuv420p,"
                  f"trim=duration={d},setpts=PTS-STARTPTS,{pre}settb=1/{fps * 1000}[v{i}]")
        fc.append(f"[{2*i+1}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                  f"apad=whole_dur={d},atrim=duration={d},asetpts=PTS-STARTPTS[a{i}]")
        acat += f"[a{i}]"
    if X > 0 and len(items) > 1:
        prev, t_end = "v0", items[0]["d"]
        for i in range(1, len(items)):
            lab = "v" if i == len(items) - 1 else f"x{i}"
            fc.append(f"[{prev}][v{i}]xfade=transition=fade:duration={X:.3f}"
                      f":offset={t_end - X:.6f}[{lab}]")
            prev, t_end = lab, t_end + items[i]["d"]
    else:
        fc.append("".join(f"[v{i}]" for i in range(len(items))) + f"concat=n={len(items)}:v=1:a=0[v]")
    fc.append(f"{acat}concat=n={len(items)}:v=0:a=1[a]")
    cmd += ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-c:a", "pcm_s24le", str(raw)]
    print(f"concat {len(items)} scene(s) at {W}x{H}@{fps}, dissolve {X} s ...")
    sh(cmd)

    offs, t0 = [], 0.0
    for it in items:
        offs.append({"id": it["id"], "title": meta.get(it["id"], {}).get("title", it["id"]),
                     "start": round(t0, 4), "duration": round(it["d"], 4)})
        t0 += it["d"]
    vlen, alen = probe_dur(raw, "v:0"), probe_dur(raw, "a:0")
    if abs(vlen - t0) > 1.5 / fps or abs(alen - t0) > 0.02:
        raise SystemExit(f"length mismatch: picture {vlen:.3f} s, sound {alen:.3f} s, scenes {t0:.3f} s")

    # 4. loudness, two-pass; chapters
    ln = f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}"
    err = sh(["ffmpeg", "-hide_banner", "-nostats", "-i", str(raw), "-vn", "-af",
              ln + ":print_format=json", "-f", "null", "-"]).stderr
    meas = json.loads(err[err.rindex("{"):err.rindex("}") + 1])
    ln2 = (f"{ln}:measured_I={meas['input_i']}:measured_TP={meas['input_tp']}"
           f":measured_LRA={meas['input_lra']}:measured_thresh={meas['input_thresh']}"
           f":offset={meas['target_offset']}:linear=true")
    out = Path(a.out)
    chap = BUILD / "film.chapters.txt"
    ffmeta_chapters(chap, offs, {o["id"]: o["title"] for o in offs},
                    "A Finite-Rate Selection Law for Annealed Gibbs Sampling")
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw), "-i", str(chap),
        "-map", "0:v", "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1", "-c:v", "copy",
        "-af", f"{ln2},aresample=48000,atrim=duration={t0:.6f}", "-t", f"{t0:.6f}", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        str(out)])
    err = sh(["ffmpeg", "-hide_banner", "-nostats", "-i", str(out), "-vn", "-af",
              "ebur128=peak=true", "-f", "null", "-"]).stderr
    summ = err[err.rindex("Summary:"):]
    li = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summ).group(1))
    lra = float(re.search(r"LRA:\s+(-?[\d.]+) LU", summ).group(1))
    tp = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summ).group(1))
    print(f"loudness: integrated {li:.1f} LUFS, LRA {lra:.1f} LU, true peak {tp:.1f} dBTP")
    loud_ok = abs(li - TARGET_I) <= 0.5 and tp < TP_LIMIT
    if not loud_ok:
        print("LOUDNESS FAIL: needs -16 +/- 0.5 LUFS and true peak below -1.5 dBTP")

    # 5. captions
    all_cues = []
    for it, o in zip(items, offs):
        cues = caption_cues(it["timing"]["words"], it["id"])
        for p in caption_lint(cues):
            print(f"caption warning {it['id']}: {p}")
        write_vtt(BUILD / "captions" / f"{it['id']}.vtt", cues)
        all_cues += [(s + o["start"], e + o["start"], t) for s, e, t in cues]
    write_vtt(out.with_suffix(".vtt"), all_cues)
    print(f"captions: {len(all_cues)} cues -> {out.with_suffix('.vtt').name}, build/captions/")

    # 6. poster: the hook's title card once the title, the venue and the route row are on
    #    (0.3 s after the "route" animation ends); else the frame before the dissolve
    first = offs[0]
    poster_t = first["start"] + first["duration"] - X - 0.2
    ev0 = json.loads((BUILD / "sync" / f"{first['id']}.events.json").read_text())["events"]
    route = [e for e in ev0 if e["cue"] == "route" and e["kind"] == "start"]
    if route and first["start"] + route[0]["end"] + 0.3 < poster_t:
        poster_t = first["start"] + route[0]["end"] + 0.3
    poster = BUILD / "film-poster.png"
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{poster_t:.3f}", "-i", str(out),
        "-frames:v", "1", str(poster)])
    print(f"poster: {poster.name}")

    # 7. web encodes
    web_info = {"film": [], "scenes": []}
    if not a.no_web:
        web_info = web_encodes(out, t0, offs, [it["movie"] for it in items])

    out.with_suffix(".offsets.json").write_text(json.dumps(
        {"film": str(out), "fps": fps, "xfade": X, "duration": round(t0, 4), "loudness_I": li,
         "loudness_LRA": lra, "true_peak": tp, "scenes": offs, "web": web_info}, indent=1))
    raw.unlink(missing_ok=True)
    print(f"film: {out} ({t0:.2f} s)")
    web_ok = all(r["under_limit"] for r in web_info["film"] + web_info["scenes"])
    return 0 if loud_ok and web_ok else 1


def web_encodes(out: Path, t0: float, offs: list[dict], movies: list[Path]) -> dict:
    """Step 7: the whole film as one 720p file, and one clip per scene."""
    web_info = {"film": [], "scenes": []}
    web = BUILD / "web"
    web.mkdir(parents=True, exist_ok=True)
    for f in web.glob("film-720p*"):
        f.unlink()
    whole = web / "film-720p.mp4"
    vf = "scale=-2:720:flags=lanczos,fps=30,format=yuv420p"
    kbps = int(FILM_LIMIT * 8 * 0.975 / t0 / 1000 - FILM_ABR_K - 3)
    for _ in range(4):  # two-pass x264 lands a few kb/s off its target: step down until it fits
        two_pass(["-i", str(out)], vf, kbps, f"{FILM_ABR_K}k", whole, "film", FILM_AF)
        size = whole.stat().st_size
        if size < FILM_LIMIT:
            break
        kbps -= math.ceil((size - 0.99 * FILM_LIMIT) * 8 / t0 / 1000) + 1
    web_info["film"].append({"file": whole.name, "seconds": round(t0, 3), "bytes": size,
                             "video_kbps": round(size * 8 / t0 / 1000 - FILM_ABR_K),
                             "mode": f"2pass {kbps}k", "under_limit": size < FILM_LIMIT})
    shutil.copyfile(out.with_suffix(".vtt"), web / "film-720p.vtt")
    for movie, o in zip(movies, offs):
        dst = web / f"{o['id']}.mp4"
        src = ["-i", str(movie), "-ss", f"{o['start']:.4f}", "-t", f"{o['duration']:.4f}",
               "-i", str(out), "-map", "0:v", "-map", "1:a", "-t", f"{o['duration']:.4f}"]
        # The master is -16 LUFS over the whole film, but one scene's narration alone can sit
        # 1-2 LU off it. Each clip gets its own gain: measured on its slice of the master, then
        # corrected once on the encoded clip (WEB_AF's limiter and the AAC encode move it a little).
        gain = TARGET_I - loudness(out, o["start"], o["duration"])[0]
        rr = web_encode(src, o["duration"], dst, o["id"], gain_db=gain)
        li = loudness(dst)[0]
        if abs(li - TARGET_I) > 0.2:
            gain += TARGET_I - li
            rr = web_encode(src, o["duration"], dst, o["id"], gain_db=gain)
        rr["gain_db"] = round(gain, 2)
        shutil.copyfile(BUILD / "captions" / f"{o['id']}.vtt", web / f"{o['id']}.vtt")
        web_info["scenes"].append(rr)
    for r in web_info["film"] + web_info["scenes"]:
        li, tp = loudness(web / r["file"])
        r["loudness_I"], r["true_peak"] = li, tp
        r["loud_ok"] = abs(li - TARGET_I) <= 0.5 and tp < TP_LIMIT
        r["under_limit"] = r["under_limit"] and r["loud_ok"]
        gain = f"  gain {r['gain_db']:+.2f} dB" if "gain_db" in r else ""
        print(f"web {r['file']:28s} {r['seconds']:8.2f} s {r['bytes']/1e6:6.2f} MB "
              f"{r['video_kbps']:5d} kbps {r['mode']:12s} {li:6.1f} LUFS {tp:5.1f} dBTP{gain}"
              f"{'' if r['under_limit'] else '  FAIL (size or loudness)'}")
    return web_info


def loudness(path: Path, start: float | None = None, dur: float | None = None) -> tuple[float, float]:
    """Integrated loudness (LUFS) and true peak (dBTP) of a file, or of the slice [start, start + dur)."""
    cut = [] if start is None else ["-ss", f"{start:.4f}", "-t", f"{dur:.4f}"]
    err = sh(["ffmpeg", "-hide_banner", "-nostats"] + cut + ["-i", str(path), "-vn", "-af",
              "ebur128=peak=true", "-f", "null", "-"]).stderr
    summ = err[err.rindex("Summary:"):]
    return (float(re.search(r"I:\s+(-?[\d.]+) LUFS", summ).group(1)),
            float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summ).group(1)))


def web_only(out: Path) -> int:
    offp = out.with_suffix(".offsets.json")
    info = json.loads(offp.read_text())
    offs = info["scenes"]
    movies = [Path(json.loads((BUILD / "sync" / f"{o['id']}.events.json").read_text())["movie_file"])
              for o in offs]
    info["web"] = web_encodes(out, info["duration"], offs, movies)
    offp.write_text(json.dumps(info, indent=1))
    return 0 if all(r["under_limit"] for r in info["web"]["film"] + info["web"]["scenes"]) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
