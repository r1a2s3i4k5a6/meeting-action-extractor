"""Offline, rule-based action-item extractor (no API key required)."""
from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from .dates import parse_deadline
from .preprocess import Segment
from .schema import ActionItem

HEDGE = re.compile(
    r"\b(?:maybe|perhaps|might|probably|possibly|if we have time|if time permits|at some point|"
    r"sometime|eventually|not sure|someday)\b", re.I)
NEGATION = re.compile(
    r"\b(?:won['’]t|will not|can['’]t|cannot|not going to|don['’]t need to|no need to|"
    r"didn['’]t|haven['’]t)\b", re.I)
CONFIRM = re.compile(
    r"^\s*(?:sure|yes|yep|yeah|ok|okay|will do|on it|got it|sounds good|no problem|absolutely|"
    r"happy to|of course|can do|noted)\b", re.I)
CLAUSE_SPLIT = re.compile(
    r",?\s+(?:and then|and|then)\s+(?=(?:i['’]ll|i will|i['’]m going to|i am going to|let me)\b)", re.I)
_ADV = re.compile(r"\b(i['’]ll|i['’]m|i|we)\s+(?:also|just|really|actually|still|then)\b", re.I)
_LEAD = re.compile(
    r"^(?:(?:also|just|then|please|probably|maybe|go ahead and|to|actually|definitely|quickly)\s+)+", re.I)
EXCLUDE_TASK = re.compile(
    r"^(?:be|say|see|know|think|admit|mention|start|begin|stop|jump|hop|interrupt|step|hear|"
    r"share my screen|pull up|go first|get started|recap|summari[sz]e)\b", re.I)
VERBS = {
    "send", "review", "update", "create", "write", "draft", "schedule", "book", "fix", "prepare",
    "share", "follow", "check", "set", "make", "finish", "complete", "contact", "call", "email",
    "ask", "confirm", "add", "remove", "test", "deploy", "investigate", "look", "organize",
    "arrange", "collect", "compile", "submit", "publish", "post", "upload", "file", "order",
    "research", "coordinate", "sync", "reach", "ping", "circulate", "document", "design", "build",
    "run", "plan", "take", "get", "find", "put", "merge", "migrate", "loop", "decide", "finalize",
    "setup", "talk", "notify", "reply", "respond", "upgrade", "clean", "close", "open", "approve",
}
LETS_VERBS = VERBS - {"review", "look", "take", "run", "get", "check", "talk", "open", "close"}
_VERB_RX = "|".join(sorted(VERBS, key=len, reverse=True))
_LETS_RX = "|".join(sorted(LETS_VERBS, key=len, reverse=True))

Pattern = Tuple[str, "re.Pattern[str]", float, str]


def _name_forms(participants: Sequence[str]) -> List[str]:
    forms = set()
    for p in participants:
        forms.add(p)
        first = p.split()[0]
        if len(first) > 1:
            forms.add(first)
    return sorted(forms, key=len, reverse=True)


def _resolver(participants: Sequence[str]) -> Dict[str, str]:
    mapping = {p: p for p in participants}
    firsts: Dict[str, List[str]] = {}
    for p in participants:
        firsts.setdefault(p.split()[0], []).append(p)
    for first, full in firsts.items():
        if len(full) == 1:
            mapping.setdefault(first, full[0])
    return mapping


@lru_cache(maxsize=64)
def _build_patterns(names: Tuple[str, ...]) -> List[Pattern]:
    n = "|".join(re.escape(x) for x in names) if names else "(?!)"
    return [
        ("action_item", re.compile(
            rf"(?i:\baction items?\s*[:\-]\s*)(?:(?P<name>{n})\s*(?i:to\b|will\b)?\s*[:\-]?\s*)?(?P<task>.+)"), 0.95, "name"),
        ("name_request", re.compile(
            rf"\b(?P<name>{n}),?\s+(?i:(?:(?:can|could|would|will)\s+you(?:\s+please)?|please)\s+)(?P<task>.+)"), 0.85, "name"),
        ("name_imperative", re.compile(
            rf"^(?i:(?:(?:ok|okay|so|and|great|thanks|thank you|right|yes)[,.!]?\s+)*)(?P<name>{n}),\s+(?P<task>(?i:(?:{_VERB_RX})\b).+)"), 0.80, "name"),
        ("name_you", re.compile(
            rf"\b(?P<name>{n}),?\s+(?i:(?:please\s+)?(?:don['’]t forget to|remember to|make sure (?:you |to )|you(?:['’]ll| will)(?: need to)?|"
            rf"you need to|you should|i['’]d like you to|i need you to|i want you to))\s*(?P<task>.+)"), 0.85, "name"),
        ("lets_have", re.compile(
            rf"(?i:\b(?:let['’]s have|can we have|i['’]d like|i want|let['’]s get|let['’]s ask)\s+)(?P<name>{n})\s+(?:(?i:to)\s+)?(?P<task>.+)"), 0.80, "name"),
        ("first_person", re.compile(
            r"\b(?P<trig>i['’]ll(?: go ahead and)?|i will(?: go ahead and)?|i['’]m going to|i am going to|"
            r"i['’]m gonna|let me|i need to|i have to|i['’]d better|i['’]m happy to|i['’]d be happy to|"
            r"i volunteer to|i should(?=\s+(?:have|get)\b)|i can(?=\s+(?:take care of|take on|take over|handle|own)\b))\s+(?P<task>.+)", re.I), 0.85, "speaker"),
        ("name_will", re.compile(
            rf"\b(?P<name>{n})\s+(?i:(?P<modal>will|is going to|needs to|has to|should|is to)\s+)(?P<task>.+)"), 0.85, "name"),
        ("name_to", re.compile(
            rf"(?:^|[,:;]\s*|\b(?i:and|then|also)\s+)(?P<name>{n})\s+(?i:to)\s+(?P<task>.+)"), 0.80, "name"),
        ("you_request", re.compile(
            r"\b(?:can|could|would|will) you(?: please)?\s+(?P<task>.+)", re.I), 0.60, "prev"),
        ("reminder", re.compile(
            r"\b(?:please\s+)?(?:remember to|don['’]t forget to)\s+(?P<task>.+)", re.I), 0.60, "none"),
        ("group_we", re.compile(
            r"\b(?:we(?:\s+will)?\s+(?:need to|should|have to|must|ought to|need (?:someone|somebody) to)|(?:someone|somebody)(?:\s+else)?\s+(?:needs to|should|has to|must))\s+(?P<task>.+)", re.I), 0.50, "none"),
        ("group_lets", re.compile(
            rf"\blet['’]s\s+(?P<task>(?:{_LETS_RX})\b.*)", re.I), 0.50, "none"),
    ]


def _clean_task(task: str, phrase: Optional[str]) -> str:
    if phrase:
        task = task.replace(phrase, "", 1)
    task = re.sub(r"\s*\b(?:by|before|at|until|around)\s+\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)(?=\W|$)", " ", task, flags=re.I)
    task = re.sub(r"\s+", " ", task)
    task = re.sub(r"\s*,\s*(?:probably|maybe|hopefully|i think|i guess)\b.*$", "", task, flags=re.I)
    task = re.sub(r",?\s+(?:since|because)\s+(?:he|she|they|it|we|i|you|the)\b.*$", "", task, flags=re.I)
    task = _LEAD.sub("", task.strip())
    for _ in range(2):
        task = re.sub(r"[\s,;:.\-?!]+$", "", task)
        task = re.sub(r"\s*,\s*(?:everyone|everybody|folks|guys|all)$", "", task, flags=re.I)
        task = re.sub(r"\s*,?\s*\b(?:first thing|asap|please|okay|ok|right|alright|eod|cob)$", "", task, flags=re.I)
    task = re.sub(r"\s+(?:and|but|so)$", "", task, flags=re.I).strip()
    return task[:1].upper() + task[1:]


_PRON_TASK = re.compile(r"^(?P<verb>[A-Za-z][A-Za-z' ]{1,28}?)\s+(?P<pro>it|that|this|them)$", re.I)
_SUBJECT = re.compile(
    r"^(?P<art>the|our|this|that)\s+(?P<np>[A-Za-z][\w\-]*(?:\s+[\w\-]+){0,3}?)\s+"
    r"(?:is|are|was|were|needs?|has|have|isn['’]t|aren['’]t|broke|broken)\b", re.I)
_VAGUE = re.compile(r"^(?:do|handle|take|get|own|take care of)\s+(?:that|it|this)$", re.I)


def _resolve_pronoun(item: ActionItem, segments: List[Segment], k: int) -> None:
    """'Rewrite it' -> 'Rewrite the refund policy doc' using the subject of a recent sentence."""
    m = _PRON_TASK.match(item.task)
    if not m:
        return
    for j in range(k - 1, max(-1, k - 3), -1):
        sm = _SUBJECT.match(segments[j].text)
        if sm:
            item.task = f"{m.group('verb')} {sm.group('art').lower()} {sm.group('np')}"
            item.flags = sorted(set(item.flags) | {"pronoun_resolved"})
            item.confidence = round(max(0.05, item.confidence - 0.05), 2)
            return


def _prev_other(segments: List[Segment], k: int) -> Optional[str]:
    cur = segments[k].speaker
    for j in range(k - 1, -1, -1):
        if segments[j].speaker and segments[j].speaker != cur:
            return segments[j].speaker
    return None


def _match_clause(clause: str, seg: Segment, prev_other: Optional[str], pats: List[Pattern],
                  resolver: Dict[str, str], ref: date, first_of_utt: Dict[int, Segment]) -> Optional[ActionItem]:
    norm = _ADV.sub(r"\1", clause)
    for kind, rx, base, mode in pats:
        m = rx.search(norm)
        if not m:
            continue
        gd = m.groupdict()
        raw_task = _LEAD.sub("", (gd.get("task") or "").strip())
        if not raw_task or EXCLUDE_TASK.match(raw_task):
            continue
        if mode == "name":
            owner = resolver.get(gd.get("name") or "", gd.get("name"))
        elif mode == "speaker":
            owner = seg.speaker
        elif mode == "prev":
            owner = prev_other
        else:
            owner = None
        conf = base
        if kind == "first_person":
            trig = (gd["trig"] or "").lower()
            if trig.startswith(("i need", "i have", "i'd")) or trig.startswith("i’d"):
                conf = 0.75
            elif trig == "let me":
                conf = 0.80
        if kind == "name_will" and (gd.get("modal") or "").lower() in {"needs to", "has to", "should"}:
            conf = 0.75
        if kind == "you_request" and owner is None:
            conf = 0.45
        dl = parse_deadline(norm, ref)
        task = _clean_task(raw_task, dl.phrase if dl else None)
        if len(task.split()) < 2:
            continue
        if dl:
            conf += 0.05
        if HEDGE.search(clause):
            conf -= 0.25
        if owner and kind not in ("first_person", "action_item"):
            for du in (1, 2):  # owner acknowledges within the next two turns
                nxt = first_of_utt.get(seg.utt + du)
                if nxt and nxt.speaker == owner and CONFIRM.match(nxt.text):
                    conf += 0.08
                    break
        return ActionItem(
            task=task, owner=owner, deadline=dl.iso if dl else None,
            confidence=round(min(0.99, max(0.05, conf)), 2),
            source=f"{seg.speaker}: {clause.strip()}" if seg.speaker else clause.strip(),
        )
    return None


def extract_rules(segments: List[Segment], participants: Sequence[str], ref_date: date) -> List[ActionItem]:
    pats = _build_patterns(tuple(_name_forms(participants)))
    resolver = _resolver(participants)
    first_of_utt: Dict[int, Segment] = {}
    for s in segments:
        first_of_utt.setdefault(s.utt, s)
    items: List[ActionItem] = []
    item_seg: List[int] = []
    for k, seg in enumerate(segments):
        if NEGATION.search(seg.text):
            continue
        prev_other = _prev_other(segments, k)
        for clause in CLAUSE_SPLIT.split(seg.text):
            if clause and clause.strip():
                it = _match_clause(clause.strip(), seg, prev_other, pats, resolver, ref_date, first_of_utt)
                if it:
                    _resolve_pronoun(it, segments, k)
                    if _VAGUE.match(it.task):
                        # "I'll do that" / "I'll take it": the speaker volunteers for a recent unowned item
                        if it.owner and it.owner == seg.speaker:
                            for idx in range(len(items) - 1, -1, -1):
                                if item_seg[idx] < k - 3:
                                    break
                                if items[idx].owner is None:
                                    items[idx].owner = it.owner
                                    items[idx].confidence = round(min(0.99, items[idx].confidence + 0.1), 2)
                                    items[idx].flags = sorted(set(items[idx].flags) | {"owner_volunteered"})
                                    break
                        continue
                    items.append(it)
                    item_seg.append(k)
    return items
