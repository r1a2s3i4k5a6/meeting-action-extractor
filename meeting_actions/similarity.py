"""Text similarity helpers shared by validation (dedup) and evaluation (matching)."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List

STOPWORDS = {
    "a", "an", "the", "to", "of", "and", "for", "with", "on", "in", "at", "by", "it", "that",
    "this", "is", "be", "we", "i", "will", "then", "also", "so", "once", "up", "out", "our",
}


def tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if w not in STOPWORDS]


def similarity(a: str, b: str) -> float:
    """max(token Jaccard, sequence ratio) on stop-word-free tokens, in [0, 1]."""
    ta, tb = tokens(a), tokens(b)
    sa, sb = set(ta), set(tb)
    jac = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
    ratio = SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()
    return max(jac, ratio)
