"""LLM backend plumbing, tested with a *fake* Anthropic client (no network, no API key).
This verifies prompt construction, response parsing, validation and fallback - NOT live model quality."""
import json
import sys
import types
import unittest
from datetime import date
from unittest import mock

from meeting_actions import run_pipeline
from meeting_actions.llm_extractor import LLMExtractionError, extract_llm

REPLY = json.dumps([
    {"task": "Send the budget to finance", "owner": "sarah", "deadline": "2026-10-09", "confidence": 0.9, "evidence": "..."},
    {"task": "Book the room", "owner": "Zed", "deadline": "2026-02-30", "confidence": 0.8, "evidence": "..."},
])


class FakeClient:
    calls = []
    fail = False

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.messages = self

    def create(self, **kw):
        if FakeClient.fail:
            raise RuntimeError("boom")
        FakeClient.calls.append(kw)
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=REPLY)])


def patched():
    FakeClient.calls, FakeClient.fail = [], False
    return mock.patch.dict(sys.modules, {"anthropic": types.SimpleNamespace(Anthropic=FakeClient)})


class LLMBackendTests(unittest.TestCase):
    def test_request_and_parsing(self):
        with patched():
            items = extract_llm("Sarah Lee: hi", ["Sarah Lee"], date(2026, 10, 5), api_key="k", model="m")
        call = FakeClient.calls[0]
        self.assertEqual(call["model"], "m")
        self.assertIn("JSON array", call["system"])
        self.assertIn("2026-10-05", call["messages"][0]["content"])
        self.assertIn("Sarah Lee", call["messages"][0]["content"])
        self.assertEqual(len(items), 2)

    def test_pipeline_validates_llm_output(self):
        with patched():
            r = run_pipeline("Sarah Lee: Hello.\nTom: Hi.", date(2026, 10, 5), backend="llm", api_key="k")
        self.assertEqual(r.backend, "llm")
        a, b = r.items
        self.assertEqual(a.owner, "Sarah Lee")           # 'sarah' mapped to roster
        self.assertIsNone(b.deadline)                    # impossible date dropped
        self.assertIn("invalid_deadline", b.flags)
        self.assertIn("unknown_owner", b.flags)
        self.assertEqual(b.status, "needs_review")

    def test_api_error_falls_back_to_rules(self):
        with patched():
            FakeClient.fail = True
            r = run_pipeline("Ann: I'll book the room tomorrow.", date(2026, 10, 5), backend="llm", api_key="k")
        self.assertEqual(r.backend, "rules")
        self.assertTrue(any("fell back" in w for w in r.warnings))
        self.assertEqual(len(r.items), 1)

    def test_missing_key(self):
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with patched(), self.assertRaises(LLMExtractionError):
            extract_llm("x", [], date(2026, 10, 5))


if __name__ == "__main__":
    unittest.main()
