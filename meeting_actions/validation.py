"""Validation rules: dates, missing owners, duplicate tasks, confidence & status."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence

from .dates import is_valid_iso
from .schema import STATUS_OPEN, STATUS_REVIEW, ActionItem
from .similarity import similarity

PLACEHOLDER_OWNERS = {"", "none", "n/a", "na", "unassigned", "unknown", "tbd", "team", "everyone",
                      "all", "we", "someone", "somebody", "null", "nobody"}
DUP_THRESHOLD = 0.8
PENALTIES = {"missing_owner": 0.15, "unknown_owner": 0.10, "invalid_deadline": 0.10, "deadline_in_past": 0.10}
BLOCKING_FLAGS = {"missing_owner", "unknown_owner", "invalid_deadline", "deadline_in_past"}


@dataclass
class ValidationReport:
    items: List[ActionItem]
    issues: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for i in self.issues:
            out[i["issue"]] = out.get(i["issue"], 0) + 1
        return out


def _resolve_owner(owner: Optional[str], roster: Sequence[str]) -> (Optional[str], bool):
    """Return (normalised_owner, is_unknown). Maps 'sarah' -> 'Sarah Lee' when unambiguous."""
    if owner is None:
        return None, False
    o = re.sub(r"\s+", " ", str(owner)).strip(" .,;:")
    if o.lower() in PLACEHOLDER_OWNERS:
        return None, False
    if not roster:
        return o, False
    for p in roster:
        if p.lower() == o.lower():
            return p, False
    first = o.split()[0].lower()
    hits = [p for p in roster if p.split()[0].lower() == first]
    if len(hits) == 1:
        return hits[0], False
    return o, True


def _compatible(a: Optional[str], b: Optional[str]) -> bool:
    return a is None or b is None or a.lower() == b.lower()


def _is_dup(a: ActionItem, b: ActionItem) -> bool:
    return (_compatible(a.owner, b.owner) and _compatible(a.deadline, b.deadline)
            and similarity(a.task, b.task) >= DUP_THRESHOLD)


def _merge(keep: ActionItem, other: ActionItem) -> ActionItem:
    base, extra = (keep, other) if keep.confidence >= other.confidence else (other, keep)
    base.owner = base.owner or extra.owner
    base.deadline = base.deadline or extra.deadline
    base.confidence = min(0.99, base.confidence + 0.05)  # restated => slightly more certain
    base.flags = sorted(set(base.flags) | set(extra.flags) | {"merged_duplicate"})
    return base


def validate_items(items: Sequence[ActionItem], participants: Sequence[str] = (),
                   ref_date: Optional[date] = None, review_threshold: float = 0.6) -> ValidationReport:
    ref = ref_date or date.today()
    issues: List[Dict[str, Any]] = []
    cleaned: List[ActionItem] = []

    for raw in items:
        it = copy.deepcopy(raw)
        it.task = re.sub(r"\s+", " ", it.task or "").strip(" .,;:-")
        it.flags = list(it.flags)
        try:
            it.confidence = min(1.0, max(0.0, float(it.confidence)))
        except (TypeError, ValueError):
            it.confidence = 0.5
        if len(it.task.split()) < 2:
            issues.append({"task": it.task, "issue": "task_too_short_dropped"})
            continue
        it.owner, unknown = _resolve_owner(it.owner, participants)
        if unknown:
            it.flags.append("unknown_owner")
        if it.deadline is not None and not is_valid_iso(it.deadline):
            issues.append({"task": it.task, "issue": "invalid_deadline", "value": str(it.deadline)})
            it.deadline = None
            it.flags.append("invalid_deadline")
        cleaned.append(it)

    merged: List[ActionItem] = []
    for it in cleaned:
        for idx, kept in enumerate(merged):
            if _is_dup(kept, it):
                merged[idx] = _merge(kept, it)
                issues.append({"task": it.task, "issue": "duplicate_merged", "into": merged[idx].task})
                break
        else:
            merged.append(it)

    for it in merged:
        flags = set(it.flags)
        if it.owner is None:
            flags.add("missing_owner")
        if it.deadline is None:
            flags.add("missing_deadline")
        else:
            d = date.fromisoformat(it.deadline)
            if d < ref:
                flags.add("deadline_in_past")
            elif d > ref + timedelta(days=365):
                flags.add("deadline_far_future")
        for f in flags:
            if f in PENALTIES:
                issues.append({"task": it.task, "issue": f})
        it.confidence = round(max(0.0, it.confidence - sum(PENALTIES.get(f, 0) for f in flags)), 2)
        it.flags = sorted(flags)
        needs_review = it.confidence < review_threshold or bool(flags & BLOCKING_FLAGS)
        it.status = STATUS_REVIEW if needs_review else STATUS_OPEN
    return ValidationReport(items=merged, issues=issues)
