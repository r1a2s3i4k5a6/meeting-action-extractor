"""Optional LLM backend (Anthropic API). Falls back to rules if unavailable."""
from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Any, List, Optional, Sequence

from .schema import ActionItem

DEFAULT_MODEL = "claude-sonnet-5-5"


class LLMExtractionError(RuntimeError):
    pass


SYSTEM_PROMPT = """You extract action items from meeting transcripts.
An action item is a concrete task someone committed to or was asked to do. Ignore discussion,
status updates, past events, hedged ideas ("maybe we could") and meeting-management talk.
Return ONLY a JSON array (no prose, no markdown fences). Each element:
{"task": str (imperative, concise), "owner": str|null (use the participant's name exactly as listed;
null if unassigned), "deadline": "YYYY-MM-DD"|null (resolve relative dates from the meeting date),
"confidence": number 0-1 (how sure you are this is a real, correctly attributed action item),
"evidence": str (short quote from the transcript)}.
If there are no action items return []."""


def build_user_prompt(transcript: str, participants: Sequence[str], ref_date: date) -> str:
    return (f"Meeting date: {ref_date.isoformat()} ({ref_date.strftime('%A')})\n"
            f"Participants: {', '.join(participants) or 'unknown'}\n\nTranscript:\n{transcript}")


def parse_llm_json(raw: str) -> List[ActionItem]:
    """Parse the model's reply into ActionItems; tolerant of code fences and extra prose."""
    text = re.sub(r"```(?:json)?", "", raw).strip()
    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, list):
        raise LLMExtractionError("Model did not return a JSON array")
    items: List[ActionItem] = []
    for d in data:
        if not isinstance(d, dict) or not d.get("task"):
            continue
        try:
            conf = float(d.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        items.append(ActionItem(
            task=str(d["task"]), owner=d.get("owner") or None, deadline=d.get("deadline") or None,
            confidence=conf, source=str(d.get("evidence", "")),
        ))
    return items


def extract_llm(transcript: str, participants: Sequence[str], ref_date: date,
                api_key: Optional[str] = None, model: Optional[str] = None) -> List[ActionItem]:
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise LLMExtractionError("Install the SDK: pip install anthropic") from e
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise LLMExtractionError("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=key)
    messages = [{"role": "user", "content": build_user_prompt(transcript, participants, ref_date)}]
    last_err: Optional[LLMExtractionError] = None
    for attempt in range(2):  # one retry if the model replies with something that is not a JSON array
        try:
            resp = client.messages.create(
                model=model or os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as e:  # network/auth/model errors: retrying will not help
            raise LLMExtractionError(f"LLM request failed: {e}") from e
        raw = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        try:
            return parse_llm_json(raw)
        except LLMExtractionError as e:
            last_err = e
            messages = messages + [
                {"role": "assistant", "content": raw or "(empty)"},
                {"role": "user", "content": "That was not a valid JSON array. Reply again with ONLY the JSON array."},
            ]
    raise last_err  # type: ignore[misc]
