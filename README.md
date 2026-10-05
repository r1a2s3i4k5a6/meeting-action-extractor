# AI Meeting Action-Item Extractor

Turns a meeting transcript into structured action items — **task, owner, deadline, status, confidence** —
with validation, evaluation and a simple upload-to-results UI.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py                                   # web UI (upload -> results)
python -m meeting_actions.cli data/sample_transcripts/01_launch_readiness.txt --date 2026-09-14
python -m meeting_actions.evaluate --all               # precision / recall / F1 on every dataset
python -m meeting_actions.evaluate --compare --data-dir data/synthetic --limit 10   # rules vs LLM, same meetings (needs API key)
python -m unittest discover -s tests -t .              # 57 tests (stdlib only, no installs needed)
```

The core pipeline, CLI, evaluation and tests use **only the Python standard library**. `streamlit`/`pandas`
are needed only for the UI, and `anthropic` only for the optional LLM backend.

## How it works (maps to the project steps)

| Step | Where |
|---|---|
| 1. Gather / generate annotated transcripts | **All data is generated:** hand-written curated sets (`data/`, `data/heldout/`, `data/blind/`: 10 meetings, 59 gold items) and a reproducible synthetic generator (`scripts/generate_synthetic.py` → `data/synthetic/`: 40 meetings, 220 gold items). Each dataset has `sample_transcripts/` + `annotations.json`. Formats: `.txt`, timestamped `.txt`, `.vtt` |
| 2. Clean & segment by speaker/sentence | `meeting_actions/preprocess.py` — speaker/timestamp parsing, VTT/SRT support, filler removal, abbreviation-safe sentence splitting |
| 3. LLM / model extraction | `rules_extractor.py` (offline, deterministic) and `llm_extractor.py` (Anthropic API, JSON output). `backend=auto` uses the LLM when `ANTHROPIC_API_KEY` is set, otherwise rules; LLM failures fall back to rules with a warning |
| 4. Output schema | `schema.py` → `{task, owner, deadline (YYYY-MM-DD), status, confidence, source, flags}` |
| 5. Validation rules | `validation.py` — see below |
| 6. Evaluation + UI | `evaluate.py`, `app.py` (Streamlit), `cli.py` |

### Validation rules
- **Dates:** must be real ISO dates (`2026-02-30` → dropped + `invalid_deadline`); deadlines before the meeting date → `deadline_in_past`; > 1 year out → `deadline_far_future`. Relative dates ("Friday", "next week", "end of month", "9/25", "in 2 weeks") are resolved from the meeting date.
- **Missing owners:** `missing_owner` flag, confidence −0.15; placeholder owners ("team", "TBD", "everyone") are treated as missing; names are mapped to the participant roster ("sarah" → "Sarah Lee"); names not in the roster → `unknown_owner`.
- **Duplicates:** items with similar text (≥ 0.8), compatible owner and compatible deadline are merged (gaps filled from the duplicate, confidence +0.05, `merged_duplicate`).
- **Too-short tasks** (< 2 words) are dropped.
- **Status:** the extractor sets `needs_review` if confidence < 0.6 or any blocking flag (missing/unknown owner, invalid/past deadline), otherwise `open`. `done` is never set automatically: a person picks it in the UI table when the task is finished.

### Confidence
Rules backend: pattern strength (explicit "action item" 0.95 > direct assignment 0.85 > self-commitment 0.85/0.75 > "can you…" without a name 0.60 > group "we should…" 0.50), +0.05 for a deadline, +0.08 if the owner acknowledges ("sure", "will do"), −0.25 for hedging ("maybe", "at some point"), then validation penalties. LLM backend: the model's own estimate, then validation penalties.

## Handy options
- `--attendees "Priya, Marcus"` (CLI) / sidebar field (UI): people who never speak but can still be assigned tasks. Without it, a name that isn't a speaker is not guessed as an owner.
- UI: filter by owner, show only items needing review, edit cells, download corrected CSV.
- Phrasing covered: "I'll…", "I can take care of…", "I'm happy to…", "Name, can you/please/you'll need to/don't forget to…", "Name will…", "Action item: Name to…", "we need to / someone should…", "I'll do that" (assigns the nearby unowned item to the speaker).

## Requirements checklist
| Requirement | Status |
|---|---|
| 1. Gather or generate annotated transcripts | Done: hand-written sets + a reproducible synthetic generator (all data is generated) |
| 2. Clean and segment by speaker and sentence | Done (`preprocess.py`) |
| 3. LLM or transformer extraction | Implemented: Anthropic LLM backend (JSON output, one retry on malformed JSON, fallback to rules) plus an offline rules backend. **Live LLM accuracy has not been measured yet** (the LLM path is tested with a fake client). Run `python -m meeting_actions.evaluate --compare --data-dir data/synthetic --limit 10` with your `ANTHROPIC_API_KEY` to produce the rules-vs-LLM table (`reports/comparison.json`) |
| 4. Output schema (task, person, date, status) | Done (+ confidence, source, flags). `owner` = person, `deadline` = date |
| 5. Validation (dates, missing owners, duplicates) | Done (`validation.py`, 7 tests) |
| 6. Evaluation + upload-to-results UI | Evaluation done (5 datasets, rules or LLM, side-by-side `--compare`). Streamlit UI (upload / sample, filter, edit, CSV + JSON download, Evaluate tab) is exercised end to end by `tests/test_app_smoke.py` against a fake Streamlit; **it has not been opened in a real browser by the author** - run `streamlit run app.py` once to confirm |

## Input format
`Speaker: text` per line, optional timestamps (`[00:01:02] Ann: …`), continuation lines, or WebVTT/SRT (`<v Ann>…`).
Owner detection needs speaker labels (participants are taken from them).

## Evaluation
`python -m meeting_actions.evaluate --all` matches predicted to gold tasks (text similarity >= 0.5, one-to-one) and reports
precision, recall, F1, owner accuracy, deadline accuracy, and a confidence-threshold sweep (saved to `reports/evaluation.json`).
A run with `--backend llm` **stops with an error** if the LLM could not be called, instead of silently scoring the rules backend
under the "llm" label.

### The number to quote: synthetic benchmark
`data/synthetic/` (40 meetings, 220 gold items), rules backend, scored once with no tuning:
**precision 0.92, recall 0.77, F1 0.84** (owner accuracy 1.00, deadline accuracy 0.98 on matched items).
Generate it yourself (or a fresh one with another seed): `python scripts/generate_synthetic.py --n 40 --seed 11 --out data/synthetic`.
The generator deliberately includes phrasings the rules do not cover (about 1/3 of templates, e.g. "Over to you, Sam. ...",
"Sam agreed to ...", "I promise to ..."), deadlines the rules cannot resolve ("within two weeks"), and hard negatives (hedges,
questions, status updates, in-meeting activity like "let me explain how it works"). `python -m meeting_actions.evaluate --data-dir data/synthetic`
prints a **breakdown**: rules find 100% of items in the templates they cover and <= 45% in the uncovered ones; all 14 false positives
were in-meeting activity. Because the template mix was chosen by the author, this is a diagnostic of *which* phrasings work,
not a real-world estimate.

### Curated sets (not a generalisation estimate)
| Dataset | Gold items | F1 when first scored | F1 now |
|---|---|---|---|
| `data/` (dev) | 27 | 1.00 (rules were written against it) | 1.00 |
| `data/heldout/` | 19 | 0.71 (new phrasing, before any changes) | 1.00 |
| `data/blind/` | 13 | 0.67 (written after the rules were frozen) | 1.00 |

After the first scoring the rules were extended to fix the failures on `heldout` and `blind`, so these three sets are now
effectively training data. The honest takeaway is the first-scored column: unseen phrasing dropped F1 to about 0.7.

### Rules vs LLM (to be filled in by you)
No live LLM run is included in this repository (it needs an API key and network access). To produce it:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m meeting_actions.evaluate --compare --data-dir data/synthetic --limit 10   # cheap first look
python -m meeting_actions.evaluate --compare --all                                   # every dataset
```
Output: a side-by-side table (P / R / F1 / owner / date) per dataset, saved to `reports/comparison.json`. Paste the table here.

**What these numbers do and don't show.** All data in this project is generated, so none of it is a real-world estimate.
The rules have not been evaluated on real recorded meetings, and spontaneous multi-party speech is where they are most likely
to struggle. The LLM backend is the intended answer for messy phrasing, which is why the comparison above matters.

Items below confidence 0.5 are mostly unassigned group tasks ("someone should..."); they are kept but marked `needs_review`.

Known limitations: pronoun resolution is a simple heuristic (subject of a recent sentence); soft or implicit
commitments ("I should have a fix by Friday" is covered, but "that'd be me" is not); a sentence containing a negation
("I can't...") is skipped entirely; dates like "the week of Nov 2" are deliberately not treated as deadlines.

## LLM backend
```bash
export ANTHROPIC_API_KEY=sk-ant-...     # optional: ANTHROPIC_MODEL=<model id>
python -m meeting_actions.cli my_meeting.txt --date 2026-10-05 --backend llm --format json
```
The model's JSON reply is parsed (one retry if it is malformed), then passes through the same validation rules as the rules backend. If the API fails the CLI/UI fall back to rules and show a warning.

## Project layout
```
app.py                    Streamlit UI
meeting_actions/          preprocess, dates, rules_extractor, llm_extractor, validation, pipeline, evaluate, cli, ui_helpers
data/                     dev set; data/heldout and data/blind = extra annotated sets
scripts/generate_synthetic.py  reproducible synthetic meetings + gold labels
tests/                    unit + regression tests
reports/evaluation.json   latest evaluation output
```

## Submitting
The git history has one commit per project step plus follow-up fix commits (`git log --oneline`). The commits carry a
**placeholder author** (`Your Name <you@example.com>`); put your own name on them before pushing:
```bash
./scripts/set_git_author.sh "Your Name" "you@example.com"     # rewrites every commit's author
git remote add origin https://github.com/<you>/<repo>.git     # create an EMPTY repo on GitHub first
git push -u origin main
```
