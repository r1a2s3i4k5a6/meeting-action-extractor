#!/usr/bin/env python3
"""Generate synthetic meeting transcripts with gold-annotated action items.

    python scripts/generate_synthetic.py --n 40 --seed 11 --out data/synthetic

Reproducible (seeded). Every gold item records the sentence `template` and deadline `when` kind that produced it, and
every meeting lists its `distractors` (sentences that look like action items but are not), so evaluation can report
*where* the extractor fails (see `python -m meeting_actions.evaluate --data-dir data/synthetic --breakdown`).

Design choices (so the benchmark is not a rubber stamp):
  * ~1/3 of assignment/commitment templates and ~15% of deadline expressions are phrasings the rule-based extractor does NOT cover.
  * Hard negatives: hedges, status updates, questions, cancellations, in-meeting activity ("let me explain..."), decisions.
  * Mixed formats: plain text, timestamped text, WebVTT. Speakers sometimes use full names but are addressed by first name.
  * Gold dates are computed here by independent code (not meeting_actions.dates).
Conventions: "by <weekday>" = the next such weekday strictly after the meeting date; "next <weekday>" = that weekday of the
following calendar week; "end of next week" = Friday of next week; "within two weeks" = meeting date + 14 days; deadlines that
cannot be resolved ("before the board meeting", "end of the sprint") are null.
"""
from __future__ import annotations

import argparse
import calendar
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FIRST = ["Maya", "Liam", "Noah", "Olivia", "Ethan", "Sofia", "Arjun", "Priya", "Mateo", "Chloe", "Daniel", "Hana", "Omar", "Zoe",
         "Lucas", "Amara", "Felix", "Isla", "Kenji", "Nadia", "Ravi", "Tessa", "Victor", "Wendy", "Yusuf", "Elena", "Gavin", "Ines"]
LAST = ["Patel", "Nguyen", "Garcia", "Kim", "Okafor", "Silva", "Novak", "Haddad", "Larsen", "Moreau", "Tanaka", "Reyes"]
WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
TASKS = [
    "send the revised budget to finance", "update the onboarding checklist", "review the pull request for the billing service",
    "book a room for the offsite", "draft the customer announcement", "schedule a call with the vendor",
    "fix the login bug on the mobile app", "prepare the quarterly metrics deck", "email the signed contract to legal",
    "write the test plan for the new API", "set up the staging environment", "collect feedback from the pilot users",
    "update the pricing page", "create tickets for the remaining bugs", "share the meeting notes with the team",
    "order the new laptops for the interns", "confirm the venue booking", "run the load test on the checkout service",
    "document the deployment process", "follow up with the Acme team", "migrate the old reports to the new dashboard",
    "renew the software licences", "interview two candidates for the designer role", "post the job description on the careers page",
    "audit the access permissions for the shared drive", "translate the help articles into Spanish",
    "benchmark the new search index", "refresh the sales forecast", "arrange travel for the conference", "clean up the support backlog",
]
THINGS = ["quarterly report", "vendor comparison", "design mockups", "release notes", "pricing sheet", "API documentation"]
CHATTER = ["Morning everyone.", "Thanks for joining.", "Great.", "Sounds good.", "Agreed.", "Okay.", "Right, that makes sense.",
           "Good point.", "Thanks, that's helpful.", "Yeah, I saw that."]
ACKS = ["Sure.", "Will do.", "On it.", "Got it.", "Okay, no problem.", "Sounds good."]


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def monday_next(ref: date) -> date:
    return ref + timedelta(days=7 - ref.weekday())


def make_when(rng: random.Random, ref: date) -> Tuple[str, Optional[str], str]:
    """Return (fragment, gold_iso_or_None, kind)."""
    kinds = ["none", "weekday", "next_weekday", "tomorrow", "eod", "eom", "eow_next", "month_day", "in_days", "ordinal",
             "end_of_sprint", "before_event", "within_weeks", "next_weekday_noon"]
    weights = [30, 14, 8, 6, 5, 4, 3, 8, 5, 4, 3, 3, 3, 2]
    kind = rng.choices(kinds, weights)[0]
    idx = rng.choice([i for i in range(5) if i != ref.weekday()])
    if kind == "none":
        return "", None, kind
    if kind == "weekday":
        return f" by {WD[idx]}", (ref + timedelta(days=((idx - ref.weekday()) % 7) or 7)).isoformat(), kind
    if kind in ("next_weekday", "next_weekday_noon"):
        iso = (monday_next(ref) + timedelta(days=idx)).isoformat()
        return f" by next {WD[idx]}" + (" at noon" if kind.endswith("noon") else ""), iso, kind
    if kind == "tomorrow":
        return " tomorrow", (ref + timedelta(days=1)).isoformat(), kind
    if kind == "eod":
        return " by end of day", ref.isoformat(), kind
    if kind == "eom":
        return " by the end of the month", ref.replace(day=calendar.monthrange(ref.year, ref.month)[1]).isoformat(), kind
    if kind == "eow_next":
        return " by the end of next week", (monday_next(ref) + timedelta(days=4)).isoformat(), kind
    if kind == "month_day":
        d = ref + timedelta(days=rng.randint(6, 40))
        return f" on {MONTHS[d.month - 1]} {d.day}", d.isoformat(), kind
    if kind == "in_days":
        n = rng.randint(2, 6)
        return f" in {n} days", (ref + timedelta(days=n)).isoformat(), kind
    if kind == "ordinal":
        d = ref + timedelta(days=rng.randint(4, 20))
        return f" by the {ordinal(d.day)}", d.isoformat(), kind
    if kind == "end_of_sprint":
        return " by the end of the sprint", None, kind
    if kind == "before_event":
        return rng.choice([" before the board meeting", " before the launch", " before the audit"]), None, kind
    return " within two weeks", (ref + timedelta(days=14)).isoformat(), kind  # within_weeks


# (id, template). {name}=first name of owner, {task}=imperative phrase, {Task}=capitalised, {when}=deadline fragment
ASSIGN = [
    ("can_you", "{name}, can you {task}{when}?"), ("please", "{name}, please {task}{when}."),
    ("will", "{name} will {task}{when}."), ("action_item", "Action item: {name} to {task}{when}."),
    ("lets_have", "Let's have {name} {task}{when}."), ("id_like", "I'd like {name} to {task}{when}."),
    ("youll_need", "{name}, you'll need to {task}{when}."), ("is_going_to", "{name} is going to {task}{when}."),
    ("over_to_you", "Over to you, {name}. {Task}{when}."), ("thats_yours", "{name}, that one's yours: {task}{when}."),
    ("agreed_to", "{name} agreed to {task}{when}."), ("can_i_get", "Can I get {name} to {task}{when}?"),
]
SELF = [
    ("ill", "I'll {task}{when}."), ("i_will", "I will {task}{when}."), ("let_me", "Let me {task}{when}."),
    ("happy_to", "I'm happy to {task}{when}."), ("going_to", "I'm going to {task}{when}."), ("need_to", "I need to {task}{when}."),
    ("on_it", "I'm on it. {Task}{when}."), ("promise", "I promise to {task}{when}."), ("on_me", "That's on me. {Task}{when}."),
]
GROUP = [("we_need", "We need to {task}{when}."), ("someone_should", "Someone should {task}{when}."), ("we_should", "We should {task}{when}.")]


def negatives(rng: random.Random, n: int) -> List[Tuple[str, str]]:
    pools = {
        "status": ["I finished the {t} yesterday.", "The {t} is blocked on legal.", "We closed twelve tickets last week.",
                   "The {t} went live on Monday."],
        "past": ["I already sent the {t}.", "I updated the {t} last week."],
        "hedge": ["Maybe we could {task} at some point.", "It might be worth it to {task} sometime.",
                  "We could probably {task} if we have time."],
        "meeting_activity": ["Let me share my screen.", "I'll go first.", "Let's move on to the next item.", "Let me know if that works.",
                             "Can you hear me okay?", "I'll walk you through the numbers.", "Let me explain how the process works.",
                             "I'll summarise the feedback quickly.", "Let me pull up the dashboard."],
        "question": ["Did you {task}?", "Should we {task}?", "Who is able to {task}?"],
        "negation": ["I won't be able to {task} this week.", "I can't {task} before Monday."],
        "decision": ["We decided to go with the second option.", "The team agreed the timeline looks fine."],
    }
    out = []
    for _ in range(n):
        kind = rng.choice(list(pools))
        out.append((kind, rng.choice(pools[kind]).format(t=rng.choice(THINGS), task=rng.choice(TASKS))))
    return out


def build_meeting(rng: random.Random, idx: int) -> dict:
    ref = date(2026, 9, 7) + timedelta(days=rng.randint(0, 100))
    while ref.weekday() > 4:
        ref += timedelta(days=1)
    firsts = rng.sample(FIRST, rng.randint(3, 5))
    full = rng.random() < 0.4
    label = {f: (f"{f} {rng.choice(LAST)}" if full else f) for f in firsts}
    tasks = rng.sample(TASKS, rng.randint(4, 7))
    events: List[dict] = []
    for task in tasks:
        when, iso, wkind = make_when(rng, ref)
        group = rng.choices(["assign", "self", "group"], [45, 40, 15])[0]
        owner = rng.choice(firsts)
        other = rng.choice([f for f in firsts if f != owner])
        tid, tpl = rng.choice({"assign": ASSIGN, "self": SELF, "group": GROUP}[group])
        text = tpl.format(name=owner, task=task, Task=task[0].upper() + task[1:], when=when)
        speaker = label[other] if group == "assign" else label[owner] if group == "self" else label[rng.choice(firsts)]
        lines = [(speaker, text)]
        if group == "assign" and rng.random() < 0.6:
            lines.append((label[owner], rng.choice(ACKS)))
        events.append({"lines": lines, "gold": {
            "task": task[0].upper() + task[1:], "owner": None if group == "group" else label[owner], "deadline": iso,
            "template": tid, "when": wkind}})
        if group != "group" and rng.random() < 0.25:  # later recap -> duplicate mention, no new gold
            events.append({"lines": [(label[other], f"To recap: {owner} will {task}{when}.")], "gold": None, "late": True})
    for kind, text in negatives(rng, rng.randint(8, 12)):
        events.append({"lines": [(label[rng.choice(firsts)], text)], "gold": None, "dis": (kind, text)})
    for _ in range(rng.randint(3, 5)):
        events.append({"lines": [(label[rng.choice(firsts)], rng.choice(CHATTER))], "gold": None})
    early = [e for e in events if not e.get("late")]
    late = [e for e in events if e.get("late")]
    rng.shuffle(early)
    lines = [(label[firsts[0]], "Okay, let's get started.")] + [l for e in early for l in e["lines"]] \
        + [l for e in late for l in e["lines"]] + [(label[firsts[0]], "Thanks everyone, that's it for today.")]
    return {"ref": ref, "participants": list(label.values()), "lines": lines,
            "gold": [e["gold"] for e in events if e["gold"]],
            "distractors": [{"kind": e["dis"][0], "text": e["dis"][1]} for e in events if e.get("dis")]}


def render(lines: List[Tuple[str, str]], style: str, rng: random.Random) -> Tuple[str, str]:
    if style == "vtt":
        out, t = ["WEBVTT", ""], 0
        for spk, txt in lines:
            d = rng.randint(3, 9)
            out += [f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}.000 --> {(t + d) // 3600:02d}:{(t + d) % 3600 // 60:02d}:{(t + d) % 60:02d}.000",
                    f"<v {spk}>{txt}", ""]
            t += d + 1
        return "\n".join(out), "vtt"
    if style == "timestamp":
        out, t = [], 0
        for spk, txt in lines:
            out.append(f"[{t // 60:02d}:{t % 60:02d}] {spk}: {txt}")
            t += rng.randint(4, 25)
        return "\n".join(out) + "\n", "txt"
    return "\n".join(f"{spk}: {txt}" for spk, txt in lines) + "\n", "txt"


def generate(n: int, seed: int, out: Path) -> int:
    rng = random.Random(seed)
    (out / "sample_transcripts").mkdir(parents=True, exist_ok=True)
    meetings = []
    for i in range(1, n + 1):
        m = build_meeting(rng, i)
        body, ext = render(m["lines"], rng.choice(["plain", "timestamp", "vtt"]), rng)
        fname = f"syn_{i:02d}_{m['ref'].isoformat()}.{ext}"
        (out / "sample_transcripts" / fname).write_text(body, encoding="utf-8")
        meetings.append({"file": fname, "meeting_date": m["ref"].isoformat(), "action_items": m["gold"], "distractors": m["distractors"]})
    (out / "annotations.json").write_text(json.dumps({
        "description": f"Synthetic meetings (seed {seed}) from scripts/generate_synthetic.py. Gold is exact by construction; "
                       "includes phrasings the rules do not cover and hard negatives. Not a real-world estimate.",
        "meetings": meetings}, indent=1, ensure_ascii=False), encoding="utf-8")
    return sum(len(m["action_items"]) for m in meetings)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/synthetic")
    a = ap.parse_args()
    print(f"Wrote {a.n} meetings, {generate(a.n, a.seed, Path(a.out))} gold action items -> {a.out}")
