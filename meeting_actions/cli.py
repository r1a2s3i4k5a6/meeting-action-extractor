"""Command line interface: python -m meeting_actions.cli transcript.txt --date 2026-09-14"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

from .pipeline import run_pipeline


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract action items from a meeting transcript.")
    ap.add_argument("transcript", help="Path to .txt / .vtt / .srt transcript ('-' for stdin)")
    ap.add_argument("--date", default=None, help="Meeting date YYYY-MM-DD (default: today)")
    ap.add_argument("--backend", default="rules", choices=["rules", "llm", "auto"])
    ap.add_argument("--format", default="table", choices=["table", "json", "csv"])
    ap.add_argument("--min-confidence", type=float, default=0.3)
    ap.add_argument("--attendees", default="", help="Comma-separated names of attendees who never speak")
    ap.add_argument("--out", default=None, help="Write output to this file instead of stdout")
    a = ap.parse_args(argv)

    text = sys.stdin.read() if a.transcript == "-" else Path(a.transcript).read_text(encoding="utf-8", errors="replace")
    try:
        ref = date.fromisoformat(a.date) if a.date else date.today()
    except ValueError:
        ap.error("--date must be YYYY-MM-DD")
    res = run_pipeline(text, ref, backend=a.backend, min_confidence=a.min_confidence,
                       extra_participants=[x for x in a.attendees.split(',') if x.strip()])

    if a.format == "json":
        out = res.to_json()
    elif a.format == "csv":
        out = res.to_csv()
    else:
        rows = [f"Meeting date: {res.meeting_date} | backend: {res.backend} | participants: {', '.join(res.participants)}"]
        rows += [f"! {w}" for w in res.warnings]
        rows.append(f"{'#':>2}  {'OWNER':<10} {'DEADLINE':<10} {'CONF':>4}  {'STATUS':<12} TASK")
        for n, i in enumerate(res.items, 1):
            rows.append(f"{n:>2}  {(i.owner or '-'):<10} {(i.deadline or '-'):<10} {i.confidence:>4.2f}  {i.status:<12} {i.task}")
        out = "\n".join(rows)
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8")
        print(f"Wrote {a.out}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
