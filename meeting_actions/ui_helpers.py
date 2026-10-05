"""Pure-Python helpers for the Streamlit UI (no streamlit/pandas needed, so they are unit-tested)."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .schema import ActionItem

UNASSIGNED = "(unassigned)"
TABLE_COLUMNS = ["task", "owner", "deadline", "status", "confidence", "flags", "source"]


def items_to_rows(items: Sequence[ActionItem]) -> List[Dict[str, Any]]:
    """ActionItems -> flat table rows (flags joined into one string)."""
    rows = []
    for i in items:
        d = i.to_dict()
        d["flags"] = ", ".join(d["flags"])
        rows.append({k: d[k] for k in TABLE_COLUMNS})
    return rows


def owners_in(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({r["owner"] for r in rows if r.get("owner")})


def filter_rows(rows: Sequence[Dict[str, Any]], owners: Sequence[str] = (), only_review: bool = False) -> List[Dict[str, Any]]:
    """Keep rows whose owner is in `owners` (UNASSIGNED matches rows without an owner); optionally only needs_review."""
    out = list(rows)
    if owners:
        out = [r for r in out if (r.get("owner") in owners) or (not r.get("owner") and UNASSIGNED in owners)]
    if only_review:
        out = [r for r in out if r.get("status") == "needs_review"]
    return out


def summary_line(items: Sequence[ActionItem]) -> str:
    n_review = sum(i.status == "needs_review" for i in items)
    return f"{len(items)} action items, {n_review} need review (missing owner, past deadline, low confidence...)."
