"""Parse the narration script (script.md) into scenes.

Conventions (script.md, "Conventions"):
  ## scene: <id>        starts a scene; the next lines give ``title:`` and ``chapter:``
  ### beat: <name>      starts a beat (not spoken)
  > text                spoken text; a bare ``>`` line is a sentence/paragraph break
  [[cue:<name>]]        an animation event lands on the first character of the next word
  [[pause:<seconds>]]   a deliberate pause, sent to TTS as <break time="Ns" />

For each scene this module builds:
  * ``clean``  — the spoken text with both markers removed (what captions show),
  * ``sent``   — the text sent to TTS (clean text plus break tags),
  * ``clean_to_sent`` — index map from a clean character to its sent character,
  * ``cues``   — {name: clean offset of the first non-space character after the marker},
  * ``pauses`` — [{offset, seconds}] clean offsets where a pause marker stood.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

MARKER = re.compile(r"\[\[(cue|pause):([^\]]+)\]\]")


@dataclass
class Scene:
    id: str
    title: str = ""
    chapter: str = ""
    raw: str = ""  # spoken text with markers, lines joined by single spaces
    clean: str = ""
    sent: str = ""
    clean_to_sent: list[int] = field(default_factory=list)
    cues: dict[str, int] = field(default_factory=dict)
    pauses: list[dict] = field(default_factory=list)
    beats: list[str] = field(default_factory=list)


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_script(path: str | Path) -> list[Scene]:
    scenes: list[Scene] = []
    cur: Scene | None = None
    spoken: list[str] = []

    def close():
        if cur is not None:
            cur.raw = _squash(" ".join(spoken))
            _build(cur)
            scenes.append(cur)

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## scene:\s*(\S+)\s*$", line)
        if m:
            close()
            cur, spoken = Scene(id=m.group(1)), []
            continue
        if cur is None:
            continue
        if line.startswith("## ") or line.strip() == "---":
            continue
        if line.startswith("title:") and not spoken and not cur.title:
            cur.title = line.split(":", 1)[1].strip()
        elif line.startswith("chapter:") and not spoken and not cur.chapter:
            cur.chapter = line.split(":", 1)[1].strip()
        elif line.startswith("### beat:"):
            cur.beats.append(line.split(":", 1)[1].strip())
        elif line.startswith(">"):
            text = line[1:].strip()
            if text:
                spoken.append(text)
    close()
    ids = [s.id for s in scenes]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate scene ids: {sorted(dup)}")
    return scenes


def _build(sc: Scene) -> None:
    clean: list[str] = []
    sent: list[str] = []
    c2s: list[int] = []
    pending_cues: list[str] = []
    pos = 0
    raw = sc.raw

    def emit_text(t: str):
        nonlocal pending_cues
        for ch in t:
            if pending_cues and not ch.isspace():
                for name in pending_cues:
                    if name in sc.cues:
                        raise ValueError(f"scene {sc.id}: duplicate cue {name!r}")
                    sc.cues[name] = len(clean)
                pending_cues = []
            c2s.append(len(sent))
            clean.append(ch)
            sent.append(ch)

    for m in MARKER.finditer(raw):
        emit_text(raw[pos:m.start()])
        kind, arg = m.group(1), m.group(2).strip()
        if kind == "cue":
            pending_cues.append(arg)
        else:
            secs = float(arg)
            sc.pauses.append({"offset": len(clean), "seconds": secs})
            tag = f'<break time="{secs:g}s" />'
            # keep a space on both sides so the tag never glues to a word
            if sent and not sent[-1].isspace():
                sent.append(" ")
            sent.extend(tag)
            sent.append(" ")
        pos = m.end()
    emit_text(raw[pos:])
    if pending_cues:
        raise ValueError(f"scene {sc.id}: cue(s) {pending_cues} at the end of the text")

    # collapse double spaces produced by marker removal, keeping the index map consistent
    out_clean, out_map = [], []
    remap = {}
    for i, ch in enumerate(clean):
        if ch == " " and out_clean and out_clean[-1] == " ":
            remap[i] = len(out_clean) - 1
            continue
        remap[i] = len(out_clean)
        out_clean.append(ch)
        out_map.append(c2s[i])
    # strip leading space
    lead = 0
    while lead < len(out_clean) and out_clean[lead] == " ":
        lead += 1
    sc.clean = "".join(out_clean[lead:]).rstrip()
    sc.clean_to_sent = out_map[lead:lead + len(sc.clean)]
    sc.cues = {k: remap[v] - lead for k, v in sc.cues.items()}
    for p in sc.pauses:
        p["offset"] = max(0, remap.get(p["offset"], len(out_clean)) - lead)
    sc.sent = re.sub(r" {2,}", " ", "".join(sent)).strip()
    # rebuild clean_to_sent against the squashed sent string
    sc.clean_to_sent = _map_clean_to_sent(sc.clean, sc.sent)


def _map_clean_to_sent(clean: str, sent: str) -> list[int]:
    """Greedy left-to-right map; ``sent`` is ``clean`` with break tags inserted."""
    out, j = [], 0
    tag = re.compile(r'<break time="[^"]+" />\s?')
    for ch in clean:
        while True:
            m = tag.match(sent, j)
            if m and ch != "<":
                j = m.end()
                continue
            break
        while j < len(sent) and sent[j] != ch:
            j += 1
        if j >= len(sent):
            raise ValueError("clean text does not embed in sent text")
        out.append(j)
        j += 1
    return out


def word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("script.md"))
    total = 0
    for s in parse_script(path):
        total += len(s.sent)
        print(f"{s.id:10s} chars={len(s.sent):5d} cues={len(s.cues):2d} pauses={len(s.pauses)} title={s.title!r}")
    print("total chars:", total)
