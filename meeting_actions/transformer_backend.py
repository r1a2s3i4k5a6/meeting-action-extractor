"""Local transformer backend (google/flan-t5-base). No API key needed."""
from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from typing import List, Sequence

from .dates import parse_deadline
from .preprocess import Segment
from .rules_extractor import CONFIRM, VERBS, _clean_task
from .schema import ActionItem

MODEL_NAME = "google/flan-t5-base"
DETECT_THRESHOLD = 0.5

_FIRST_PERSON = re.compile(
    r"\b(i'll|i will|i can|i'm going to|i am going to|i'm gonna|let me|i need to|i have to)\b", re.I)
_NONE_WORDS = {"", "none", "nobody", "no one", "unknown", "n/a", "not specified", "no", "nothing"}


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tok, model


def _generate(prompt: str, max_new_tokens: int = 40) -> str:
    import torch
    tok, model = _load()
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def _yes_probability(prompt: str) -> float:
    import torch
    tok, model = _load()
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    yes_id = tok("yes", add_special_tokens=False).input_ids[0]
    no_id = tok("no", add_special_tokens=False).input_ids[0]
    dec = torch.tensor([[model.config.decoder_start_token_id]])
    with torch.no_grad():
        logits = model(**inputs, decoder_input_ids=dec).logits[0, -1]
    return float(torch.softmax(logits[[yes_id, no_id]], dim=0)[0])


def _clean_answer(ans: str):
    ans = ans.strip().strip(".").strip()
    return None if ans.lower() in _NONE_WORDS else ans


def _name_forms(participants: Sequence[str]):
    forms = []
    for p in participants:
        forms.append((p, p))
        first = p.split()[0]
        if len(first) > 1:
            forms.append((first, p))
    return sorted(forms, key=lambda x: len(x[0]), reverse=True)


def _find_owner(text: str, speaker, participants: Sequence[str]):
    if _FIRST_PERSON.search(text):
        return speaker
    best = None  # earliest mentioned participant
    for form, full in _name_forms(participants):
        m = re.search(rf"\b{re.escape(form)}\b", text, re.I)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), full)
    if best:
        return best[1]
    ans = _clean_answer(_generate(
        "Who is responsible for doing the task? Give only the person's name, or answer none.\n"
        f"Speaker: {speaker}\nSentence: {text}"))
    if ans:
        for form, full in _name_forms(participants):
            if form.lower() == ans.lower():
                return full
    return None


def extract_transformer(segments: List[Segment], participants: Sequence[str], ref_date: date,
                        threshold: float = DETECT_THRESHOLD) -> List[ActionItem]:
    items: List[ActionItem] = []
    name_rx = "|".join(re.escape(f) for f, _ in _name_forms(participants)) or "(?!)"
    for seg in segments:
        text = seg.text.strip()
        if len(text.split()) < 5 or CONFIRM.match(text):
            continue
        conf = _yes_probability(
            "Does this sentence from a meeting assign or commit to a task that someone "
            f"must do? Answer yes or no.\nSentence: {text}")
        if re.search(r"\b(?:can|could|would|will) you\b|\bplease\b", text, re.I):
            conf = max(conf, 0.6)  # direct requests are almost always tasks
        if conf < threshold:
            continue
        dl = parse_deadline(text, ref_date)
        task = _clean_answer(_generate(
            "Rewrite this meeting sentence as a short task starting with a verb.\n"
            f"Sentence: {text}")) or text
        task = re.sub(rf"^(?:{name_rx})\s*,?\s*", "", task, flags=re.I)
        task = _clean_task(task, dl.phrase if dl else None)
        task = re.sub(r"^(?:please\s+|i['\u2019]ll\s+|i will\s+|i['\u2019]m going to\s+|i am going to\s+|let me\s+|i need to\s+|i have to\s+)+(?:go ahead and\s+)?", "", task, flags=re.I)
        task = task[:1].upper() + task[1:]
        if len(task.split()) < 2 or task.split()[0].lower() not in VERBS:
            continue
        items.append(ActionItem(
            task=task,
            owner=_find_owner(text, seg.speaker, participants),
            deadline=dl.iso if dl else None,
            confidence=round(min(0.99, max(0.05, conf)), 2),
            source=f"{seg.speaker}: {text}" if seg.speaker else text,
        ))
    return items