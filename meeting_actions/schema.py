"""Structured output schema."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

STATUS_OPEN = "open"
STATUS_REVIEW = "needs_review"
STATUS_DONE = "done"  # never set by the extractor; a user marks it in the UI once the task is finished
STATUSES = (STATUS_OPEN, STATUS_REVIEW, STATUS_DONE)


@dataclass
class ActionItem:
    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None  # ISO date, YYYY-MM-DD
    status: str = STATUS_OPEN  # "open" | "needs_review" | "done" (done is set manually in the UI)
    confidence: float = 0.5  # 0..1
    source: str = ""  # supporting quote from the transcript
    flags: List[str] = field(default_factory=list)  # validation flags

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
