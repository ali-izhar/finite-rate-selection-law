"""Captions: readable display text on the narration's word timings, cut at sentence and clause ends.

The narration is written for the ear ("zero point one two zero", "N squared gamma, to the minus
one quarter"). A caption reader needs the written form ("0.120", "(N²γ)^(−1/4)"). This module

1. maps runs of spoken words to their display form (``DISPLAY``; the longest match wins, scene
   rules before the common ones). A replaced run becomes one caption token that starts on its
   first spoken word and ends on its last, so the display text keeps the TTS alignment exactly;
2. cuts the tokens into cues: one sentence per cue when it fits in two lines of 42 characters;
   a longer sentence is cut at the clause end (, ; : — ) nearest its middle, then at the word
   that leaves both parts at least three words; a short sentence (under three words) joins its
   neighbour unless a pause of more than 0.6 s isolates it (a deliberate beat, e.g. "61.").

Only the standard library; ``assemble.py`` calls ``caption_cues(words, scene_id)``.
"""
from __future__ import annotations

import re

LINE_MAX, LINES_MAX = 42, 2
MIN_WORDS = 3
BEAT_GAP = 0.6  # a pause longer than this around a short sentence marks a deliberate beat

# (scene id or "*", spoken words, display). Interior punctuation of the spoken words is ignored;
# the last spoken word's trailing punctuation is kept after the display text.
DISPLAY: list[tuple[str, str, str]] = [
    # ---------------------------------------------------------------- hook
    ("hook", "a hundred samples", "100 samples"),
    ("hook", "ninety-seven", "97"),
    ("hook", "sixty-one", "61"),
    ("hook", "fifty-nine", "59"),
    ("hook", "sixty-three", "63"),
    ("hook", "four times the paper's time, sixteen times the first", "4× the paper's time, 16× the first"),
    # ---------------------------------------------------------------- commit
    ("commit", "each plus or minus one", "each +1 or −1"),
    ("commit", "mostly plus ones", "mostly +1s"),
    # ---------------------------------------------------------------- odds
    ("odds", "toward plus one", "toward +1"),
    ("odds", "in ten steps", "in 10 steps"),
    ("odds", "N plus one probabilities", "N + 1 probabilities"),
    ("odds", "two to the N", "2^N"),
    ("odds", "ninety-seven point three percent", "97.3%"),
    ("odds", "ends at sixty point five", "ends at 60.5%"),
    ("odds", "the log of q over one minus q", "the log of q / (1 − q)"),
    ("odds", "three point five seven", "3.57"),
    ("odds", "zero point four three", "0.43"),
    ("odds", "past one hundred percent", "past 100%"),
    ("odds", "about thirty-six to one becomes about one and a half to one", "about 36:1 becomes about 1.5:1"),
    ("odds", "about twelve percent", "about 12%"),
    # ---------------------------------------------------------------- race
    ("race", "beta minus one, times m, plus G", "(β − 1)m + G"),
    ("race", "variance two over N", "variance 2/N"),
    ("race", "beta minus one, which is gamma times t", "β − 1, which is γt"),
    ("race", "gamma t squared over two", "γt²/2"),
    ("race", "gamma to the minus one half sweeps", "γ^(−1/2) sweeps"),
    ("race", "G root N, times gamma to the minus one quarter", "G√N · γ^(−1/4)"),
    ("race", "Phi of a", "Φ(a)"),
    ("race", "zero point two zero eight", "0.208"),
    ("race", "fifty-eight percent", "58%"),
    ("race", "sixty-one", "61"),
    # ---------------------------------------------------------------- scaling
    ("scaling", "like N times G", "like N·G"),
    ("scaling", "N squared gamma, to the minus one quarter", "(N²γ)^(−1/4)"),
    ("scaling", "sixteen times more slowly", "16× more slowly"),
    ("scaling", "sixteen times slower", "16× slower"),
    ("scaling", "zero point zero nine four", "0.094"),
    ("scaling", "zero point one eight nine", "0.189"),
    ("scaling", "two percent", "2%"),
    ("scaling", "fifty-nine to sixty-three", "59 to 63"),
    ("scaling", "one point four eight times", "1.48×"),
    # ---------------------------------------------------------------- modes
    ("modes", "patterns of plus and minus ones", "patterns of +1s and −1s"),
    ("modes", "from minus one at the negative", "from −1 at the negative"),
    ("modes", "to plus one at the pattern", "to +1 at the pattern"),
    ("modes", "one point six six two", "1.662"),
    ("modes", "zero point one nine nine", "0.199"),
    # ---------------------------------------------------------------- reheat
    ("reheat", "sixty-one percent", "61%"),
    # ---------------------------------------------------------------- test
    ("test", "sixty-six point nine percent", "66.9%"),
    ("test", "ninety-seven point two six", "97.26%"),
    ("test", "ninety-five percent", "95%"),
    ("test", "zero point one nine nine", "0.199"),
    ("test", "zero point two zero nine five", "0.2095"),
    ("test", "zero point two one two", "0.212"),
    ("test", "sixty-seven point one percent", "67.1%"),
    ("test", "sixty-six point nine measured", "66.9% measured"),
    ("test", "where it is minus one", "where it is −1"),
    ("test", "one point zero six three times", "1.063×"),
    ("test", "eleven percent", "11%"),
    # ---------------------------------------------------------------- limits
    ("limits", "N to the minus zero point three six", "N^(−0.36)"),
    ("limits", "the minus zero point four six", "the −0.46"),
    ("limits", "Ninety-seven asked for; sixty-one delivered", "97 asked for; 61 delivered"),
    ("limits", "zero point one two", "0.12"),
    ("limits", "sixteen times the sweeps", "16× the sweeps"),
    ("limits", "four positive samples in a hundred", "4 positive samples in 100"),
    # ---------------------------------------------------------------- common
    ("*", "beta equal to one", "β = 1"),
    ("*", "zero point one two zero", "0.120"),
    ("*", "root N", "√N"),
    ("*", "beta", "β"),
    ("*", "gamma", "γ"),
]


def _norm(w: str) -> str:
    return re.sub(r"[^\w\-'’–]", "", w).lower()


def _trail(w: str) -> str:
    m = re.search(r"[.,;:?!—]+$", w)
    return m.group(0) if m else ""


def display_tokens(words: list[dict], scene_id: str | None = None) -> list[dict]:
    """Spoken words -> caption tokens {w, start, end}, runs replaced by their display form."""
    rules = [(sc, [_norm(x) for x in sp.split()], disp) for sc, sp, disp in DISPLAY
             if sc in ("*", scene_id)]
    # scene rules first, then the longest spoken run
    rules.sort(key=lambda r: (r[0] == "*", -len(r[1])))
    norm = [_norm(w["w"]) for w in words]
    out, i = [], 0
    while i < len(words):
        for _, pat, disp in rules:
            n = len(pat)
            if norm[i:i + n] == pat:
                last = words[i + n - 1]
                out.append({"w": disp + _trail(last["w"]), "start": words[i]["start"], "end": last["end"]})
                i += n
                break
        else:
            out.append({"w": words[i]["w"], "start": words[i]["start"], "end": words[i]["end"]})
            i += 1
    return out


# ------------------------------------------------------------------ layout
WEAK = {"a", "an", "the", "of", "to", "and", "or", "in", "on", "at", "with", "from", "its", "that",
        "is", "by", "for", "as", "than", "like", "each", "every", "this", "their", "our", "his"}


def _bare(w: str) -> str:
    return re.sub(r"[^\w]", "", w).lower()


def _break_cost(left: dict, right: dict | None) -> float:
    """How bad it is to end a line or a cue after ``left``."""
    w = left["w"]
    if re.search(r"[.?!]$", w):
        return 0.0
    if re.search(r"[,;:—]$", w):
        return 1.0
    c = 5.0
    if right is not None and right["start"] - left["end"] > 0.25:
        c = 2.5
    if _bare(w) in WEAK:
        c += 4.0
    return c


def _layout(toks: list[dict]) -> tuple[float, list[str]] | None:
    """(cost, lines): one line if it fits; else the best two-line break at a token boundary
    (balanced, at a clause end, never after a weak word); None if the text does not fit."""
    t = _text(toks)
    if len(t) <= LINE_MAX:
        return 0.0, [t]
    best = None
    for k in range(1, len(toks)):
        a, b = _text(toks[:k]), _text(toks[k:])
        if len(a) > LINE_MAX or len(b) > LINE_MAX:
            continue
        score = abs(len(a) - len(b)) / 4 + 1.5 * _break_cost(toks[k - 1], toks[k])
        if min(len(a), len(b)) < 14:
            score += 4.0
        if best is None or score < best[0]:
            best = (score, [a, b])
    return best


def lines_of(toks: list[dict]) -> list[str] | None:
    r = _layout(toks)
    return None if r is None else r[1]


def balance(t: str) -> str | None:  # kept for callers that hold plain text
    return "\n".join(lines_of([{"w": w, "start": 0, "end": 0} for w in t.split(" ")]) or [t])


def _text(toks) -> str:
    return " ".join(t["w"] for t in toks)


def _fits(toks) -> bool:
    return lines_of(toks) is not None


def _sentences(toks: list[dict]) -> list[list[dict]]:
    out, cur = [], []
    for t in toks:
        cur.append(t)
        if re.search(r"[.?!]$", t["w"]):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _split(toks: list[dict]) -> list[list[dict]]:
    """Cut a sentence into cues that fit: fewest cues, cut at clause ends, no cue under three
    words, no line or cue ending on a weak word (DP over the cut points)."""
    n = len(toks)
    INF = float("inf")
    best = [INF] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            seg = toks[i:j]
            if len(_text(seg)) > LINE_MAX * LINES_MAX + 1:
                break
            lay = _layout(seg)
            if lay is None:
                continue
            c = 4.0 + 0.6 * lay[0] + (2.5 * _break_cost(seg[-1], toks[j]) if j < n else 0.0)
            if len(seg) < MIN_WORDS and not (i == 0 and j == n):
                c += 15.0
            dur = seg[-1]["end"] - seg[0]["start"]
            c += max(0.0, dur - 7.0) * 2
            if best[i] + c < best[j]:
                best[j], back[j] = best[i] + c, i
    if best[n] == INF:  # a single token wider than a cue: show it alone
        return [[t] for t in toks]
    out, j = [], n
    while j > 0:
        out.append(toks[back[j]:j])
        j = back[j]
    return out[::-1]


def caption_cues(words: list[dict], scene_id: str | None = None,
                 offset: float = 0.0) -> list[tuple[float, float, str]]:
    toks = display_tokens(words, scene_id)
    sents = _sentences(toks)
    # a short sentence joins a neighbour unless a pause isolates it (a deliberate beat)
    merged: list[list[dict]] = []
    i = 0
    while i < len(sents):
        s = sents[i]
        if len(s) < MIN_WORDS:
            gap_prev = s[0]["start"] - merged[-1][-1]["end"] if merged else 99
            gap_next = sents[i + 1][0]["start"] - s[-1]["end"] if i + 1 < len(sents) else 99
            beat = gap_prev > BEAT_GAP and gap_next > BEAT_GAP
            if not beat:
                if i + 1 < len(sents) and gap_next <= gap_prev and _fits(s + sents[i + 1]):
                    sents[i + 1] = s + sents[i + 1]
                    i += 1
                    continue
                if merged and _fits(merged[-1] + s):
                    merged[-1] = merged[-1] + s
                    i += 1
                    continue
                if i + 1 < len(sents) and _fits(s + sents[i + 1]):
                    sents[i + 1] = s + sents[i + 1]
                    i += 1
                    continue
        merged.append(s)
        i += 1
    groups = [g for s in merged for g in _split(s)]
    cues = []
    for j, g in enumerate(groups):
        a, last_end = g[0]["start"], g[-1]["end"]
        nxt = groups[j + 1][0]["start"] if j + 1 < len(groups) else None
        end = last_end + 0.35 if nxt is None else min(last_end + 0.35, nxt - 0.04)
        end = max(end, a + 0.8) if nxt is None else max(end, min(a + 0.8, nxt - 0.04))
        cues.append((a + offset, end + offset, "\n".join(lines_of(g) or [_text(g)])))
    return cues


def lint(cues) -> list[str]:
    """Problems a caption reader would see: a line over 42 characters, more than 2 lines,
    a cue under three words that is not a deliberate beat, a spelled-out decimal."""
    bad = []
    for k, (a, b, t) in enumerate(cues, 1):
        lines = t.split("\n")
        if len(lines) > LINES_MAX or any(len(l) > LINE_MAX for l in lines):
            bad.append(f"cue {k}: layout {lines}")
        if re.search(r"\bpoint (zero|one|two|three|four|five|six|seven|eight|nine)\b|\bhundred\b|\bthousand\b|\bpercent\b|\bbeta\b|\bgamma\b", t):
            bad.append(f"cue {k}: spelled number: {t!r}")
    return bad


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    ids = sys.argv[1:] or [s["id"] for s in json.loads((here / "scenes.json").read_text())]
    for sid in ids:
        t = json.loads((here / "build" / "audio" / f"{sid}.timing.json").read_text())
        cues = caption_cues(t["words"], sid)
        print(f"===== {sid} ({len(cues)} cues)")
        for a, b, txt in cues:
            print(f"{a:7.2f} {b:7.2f} | " + txt.replace("\n", " / "))
        for p in lint(cues):
            print("LINT", p)
