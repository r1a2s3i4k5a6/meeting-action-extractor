"""Transcript cleaning, speaker parsing and sentence segmentation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Utterance:
    speaker: Optional[str]
    text: str


@dataclass
class Segment:
    """One sentence, tagged with its speaker and the utterance it came from."""
    speaker: Optional[str]
    text: str
    index: int
    utt: int


_TS = r"(?:[\[\(]?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?[\]\)]?\s*[-–]?\s*)?"
_LINE = re.compile(
    r"^\s*" + _TS + r"(?P<speaker>[A-Z][\w.'’\-]*(?: [\w.'’\-]+){0,3}?)\s*:\s*(?P<text>\S.*)$"
)
_FILLER = re.compile(r"\s*\b(?:um+|uh+|uhm|erm|hmm+)\b[,.]?\s*", re.I)
_ABBREV = ["Mr.", "Mrs.", "Ms.", "Dr.", "vs.", "e.g.", "i.e.", "etc.", "approx.", "St."]
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'‘“(\[])")


def clean_text(text: str) -> str:
    """Remove disfluencies and normalise whitespace/punctuation spacing."""
    text = _FILLER.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    return text.strip()


def _strip_subtitles(text: str) -> str:
    """Convert WebVTT / SRT into plain 'Speaker: text' lines."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.upper().startswith("WEBVTT") or "-->" in s or s.isdigit():
            continue
        if s.startswith(("NOTE ", "STYLE")):
            continue
        m = re.match(r"^<v\s+([^>]+)>(.*?)(?:</v>)?$", s)
        if m:
            s = f"{m.group(1).strip()}: {m.group(2).strip()}"
        s = re.sub(r"</?[^>]+>", "", s)
        out.append(s)
    return "\n".join(out)


def parse_transcript(text: str) -> List[Utterance]:
    """Split a transcript into speaker utterances (supports txt, timestamps, VTT/SRT)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "-->" in text or text.lstrip().upper().startswith("WEBVTT"):
        text = _strip_subtitles(text)
    utterances: List[Utterance] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = _LINE.match(line)
        if m:
            utterances.append(Utterance(m.group("speaker").strip(), m.group("text").strip()))
        elif utterances:  # continuation of the previous speaker's turn
            utterances[-1].text += " " + line
        else:
            utterances.append(Utterance(None, line))
    for u in utterances:
        u.text = clean_text(u.text)
    return [u for u in utterances if u.text]


def split_sentences(text: str) -> List[str]:
    for i, a in enumerate(_ABBREV):
        text = text.replace(a, a.replace(".", f"\0{i}\0"))
    parts = _SENT_SPLIT.split(text)
    restored = []
    for p in parts:
        for i, a in enumerate(_ABBREV):
            p = p.replace(a.replace(".", f"\0{i}\0"), a)
        restored.append(p.strip())
    return [p for p in restored if p]


def segment(utterances: List[Utterance]) -> List[Segment]:
    segments: List[Segment] = []
    for ui, u in enumerate(utterances):
        for sent in split_sentences(u.text):
            segments.append(Segment(u.speaker, sent, len(segments), ui))
    return segments


def participants_of(utterances: List[Utterance]) -> List[str]:
    seen: List[str] = []
    for u in utterances:
        if u.speaker and u.speaker not in seen:
            seen.append(u.speaker)
    return seen
