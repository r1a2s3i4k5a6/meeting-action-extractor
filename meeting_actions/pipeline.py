"""End-to-end pipeline: transcript text -> validated, structured action items."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

from .llm_extractor import LLMExtractionError, extract_llm
from .preprocess import parse_transcript, participants_of, segment
from .rules_extractor import extract_rules
from .schema import ActionItem
from .validation import validate_items

BACKENDS = ("rules", "llm", "auto", "transformer")
CSV_FIELDS = ["task", "owner", "deadline", "status", "confidence", "flags", "source"]


@dataclass
class PipelineResult:
    items: List[ActionItem]
    participants: List[str]
    backend: str
    meeting_date: str
    warnings: List[str] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"meeting_date": self.meeting_date, "backend": self.backend,
                "participants": self.participants, "warnings": self.warnings,
                "action_items": [i.to_dict() for i in self.items], "validation_issues": self.issues}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
        w.writeheader()
        for i in self.items:
            row = i.to_dict()
            row["flags"] = ";".join(row["flags"])
            w.writerow({k: row[k] for k in CSV_FIELDS})
        return buf.getvalue()


def run_pipeline(text: str, reference_date: Optional[date] = None, backend: str = "rules",
                 min_confidence: float = 0.3, review_threshold: float = 0.6,
                 api_key: Optional[str] = None, model: Optional[str] = None,
                 extra_participants: Sequence[str] = ()) -> PipelineResult:
    """Extract action items. `reference_date` = meeting date (resolves 'Friday', 'next week', ...).
    `extra_participants` = attendee names that never speak but may be assigned tasks."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    ref = reference_date or date.today()
    utts = parse_transcript(text)
    participants = participants_of(utts)
    for name in extra_participants:  # attendees who never speak (so they can still own tasks)
        name = name.strip()
        if name and name.lower() not in {p.lower() for p in participants}:
            participants.append(name)
    segs = segment(utts)
    warnings: List[str] = []
    if not utts:
        warnings.append("Transcript is empty.")
    elif not participants:  # no speaker labels and no attendees given
        warnings.append("No 'Speaker: text' lines found; owners can only be inferred by the LLM backend.")

    used = backend
    raw_items: List[ActionItem] = []
    if backend in ("llm", "auto"):
        import os
        if backend == "auto" and not (api_key or os.getenv("ANTHROPIC_API_KEY")):
            used = "rules"
        else:
            clean = "\n".join(f"{u.speaker}: {u.text}" if u.speaker else u.text for u in utts)
            try:
                raw_items = extract_llm(clean, participants, ref, api_key=api_key, model=model)
            except LLMExtractionError as e:
                warnings.append(f"LLM backend failed ({e}); fell back to rules.")
                used = "rules"
    if backend == "transformer":
        from .transformer_backend import extract_transformer
        raw_items = extract_transformer(segs, participants, ref)
    if used == "rules":
        raw_items = extract_rules(segs, participants, ref)

    report = validate_items(raw_items, participants, ref, review_threshold)
    kept = []
    for it in report.items:
        if it.confidence < min_confidence:
            report.issues.append({"task": it.task, "issue": "below_min_confidence_dropped",
                                  "confidence": it.confidence})
        else:
            kept.append(it)
    return PipelineResult(kept, participants, used, ref.isoformat(), warnings, report.issues)
