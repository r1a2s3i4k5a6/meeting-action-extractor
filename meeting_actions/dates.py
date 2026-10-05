"""Deadline parsing: absolute and relative date expressions -> ISO dates."""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august",
                "september", "october", "november", "december"]
MONTHS = {n: i + 1 for i, n in enumerate(_MONTH_NAMES)}
MONTHS.update({n[:3]: i + 1 for i, n in enumerate(_MONTH_NAMES)})
MONTHS["sept"] = 9
_MON = "|".join(sorted(MONTHS, key=len, reverse=True))
_WD = "|".join(WEEKDAYS)
_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a couple of": 2}
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WEEK_OF = re.compile(rf"\bweek of\s+(?:(?:{_MON})\.?\s+)?$", re.I)
_PREP = re.compile(r"\b(?:by|before|on|until|till|due|no later than|for)\s+(?:the\s+)?$", re.I)


@dataclass
class DeadlineMatch:
    date: date
    phrase: str  # matched text, including a leading preposition such as "by"
    start: int
    end: int

    @property
    def iso(self) -> str:
        return self.date.isoformat()


def is_valid_iso(value: Optional[str]) -> bool:
    if not isinstance(value, str) or not ISO_RE.match(value):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def _safe(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    y += d.year
    m += 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _resolve_year(m: int, d: int, year: Optional[str], ref: date) -> Optional[date]:
    if year:
        return _safe(int(year), m, d)
    cand = _safe(ref.year, m, d)
    if cand and cand < ref:
        cand = _safe(ref.year + 1, m, d)
    return cand


def _h_iso(m, ref):
    return _safe(int(m["y"]), int(m["m"]), int(m["d"]))


def _h_month_day(m, ref):
    return _resolve_year(MONTHS[m["mon"].lower()], int(m["day"]), m["year"], ref)


def _h_numeric(m, ref):
    a, b = int(m["a"]), int(m["b"])
    mo, d = (b, a) if a > 12 and b <= 12 else (a, b)
    year = m["y"]
    if year and len(year) == 2:
        year = "20" + year
    return _resolve_year(mo, d, year, ref)


def _h_in(m, ref):
    raw = m["n"].lower()
    n = int(raw) if raw.isdigit() else _NUM_WORDS.get(raw)
    if not n:
        return None
    unit = m["u"].lower()
    if unit == "day":
        return ref + timedelta(days=n)
    if unit == "week":
        return ref + timedelta(weeks=n)
    return _add_months(ref, n)


def _h_next_wd(m, ref):
    monday_next = ref + timedelta(days=7 - ref.weekday())
    return monday_next + timedelta(days=WEEKDAYS.index(m["wd"].lower()))


def _h_ordinal(m, ref):
    """'the 15th' -> next occurrence of that day-of-month (this month if still ahead)."""
    day = int(m["day"])
    for add in range(3):
        y, mo = divmod(ref.month - 1 + add, 12)
        cand = _safe(ref.year + y, mo + 1, day)
        if cand and cand >= ref:
            return cand
    return None


def _h_end_next_week(m, ref):
    return ref + timedelta(days=7 - ref.weekday() + 4)


def _h_wd(m, ref):
    idx = WEEKDAYS.index(m["wd"].lower())
    return ref + timedelta(days=((idx - ref.weekday()) % 7) or 7)


_RULES = [
    (re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b"), _h_iso),
    (re.compile(rf"\b(?P<mon>{_MON})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(?P<year>\d{{4}}))?", re.I), _h_month_day),
    (re.compile(rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mon>{_MON})\b\.?(?:,?\s+(?P<year>\d{{4}}))?", re.I), _h_month_day),
    (re.compile(r"\b(?P<a>\d{1,2})/(?P<b>\d{1,2})(?:/(?P<y>\d{2,4}))?\b"), _h_numeric),
    (re.compile(r"\b(?:the\s+)?(?P<day>\d{1,2})(?:st|nd|rd|th)\b", re.I), _h_ordinal),
    (re.compile(r"\bin\s+(?P<n>\d+|a couple of|an|a|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?P<u>day|week|month)s?\b", re.I), _h_in),
    (re.compile(rf"\bnext\s+(?P<wd>{_WD})\b(?:\s+(?:morning|afternoon|evening))?", re.I), _h_next_wd),
    (re.compile(rf"\b(?:this\s+)?(?P<wd>{_WD})\b(?:\s+(?:morning|afternoon|evening))?", re.I), _h_wd),
    (re.compile(r"\btomorrow(?:\s+(?:morning|afternoon|evening))?\b", re.I), lambda m, r: r + timedelta(days=1)),
    (re.compile(r"\b(?:today|tonight|this (?:morning|afternoon|evening)|eod|cob|end of (?:the |this )?day)\b", re.I), lambda m, r: r),
    (re.compile(r"\b(?:by|before|until)\s+\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)(?=\W|$)", re.I), lambda m, r: r),
    (re.compile(r"\bend of next week\b", re.I), _h_end_next_week),
    (re.compile(r"\b(?:end of (?:the |this )?week|eow|this week)\b", re.I),
     lambda m, r: r + timedelta(days=(4 - r.weekday()) % 7)),
    (re.compile(r"\bnext week\b", re.I), lambda m, r: r + timedelta(days=7)),
    (re.compile(r"\b(?:end of (?:the |this )?month|eom)\b", re.I),
     lambda m, r: r.replace(day=calendar.monthrange(r.year, r.month)[1])),
    (re.compile(r"\bnext month\b", re.I), lambda m, r: _add_months(r, 1)),
]


def parse_deadline(text: str, ref: date) -> Optional[DeadlineMatch]:
    """Find the first deadline expression in `text`, resolved relative to `ref`."""
    for rx, handler in _RULES:
        for m in rx.finditer(text):
            if _WEEK_OF.search(text[:m.start()]):
                continue  # "the week of Nov 2" says when something happens, not when it is due
            d = handler(m, ref)
            if d:
                start = m.start()
                pm = _PREP.search(text[:start])
                if pm:
                    start = pm.start()
                return DeadlineMatch(d, text[start:m.end()], start, m.end())
    return None
