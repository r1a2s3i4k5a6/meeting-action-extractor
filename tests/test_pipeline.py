import json
import unittest
from datetime import date

from meeting_actions import run_pipeline
from meeting_actions.evaluate import evaluate_dataset
from meeting_actions.llm_extractor import LLMExtractionError, parse_llm_json

REF = date(2026, 9, 14)


class PipelineTests(unittest.TestCase):
    def test_basic_extraction(self):
        t = "Ann: Bob, can you send the invoice to finance by Friday?\nBob: Sure.\nAnn: I'll book the room tomorrow."
        r = run_pipeline(t, REF)
        got = {(i.owner, i.task, i.deadline) for i in r.items}
        self.assertIn(("Bob", "Send the invoice to finance", "2026-09-18"), got)
        self.assertIn(("Ann", "Book the room", "2026-09-15"), got)

    def test_full_name_participants_resolved(self):
        t = "Sarah Lee: Hello.\nTom Roe: Sarah, please update the budget sheet by Friday.\nSarah Lee: Sure."
        r = run_pipeline(t, REF)
        self.assertEqual(r.items[0].owner, "Sarah Lee")

    def test_negation_hedge_and_questions_ignored(self):
        t = "Ann: I won't be able to send that.\nBob: Should we send a follow-up?\nAnn: Maybe we could look into it sometime."
        self.assertEqual(run_pipeline(t, REF).items, [])

    def test_empty_and_unlabelled_input(self):
        self.assertIn("Transcript is empty.", run_pipeline("   ", REF).warnings)
        self.assertTrue(run_pipeline("just some text without speakers", REF).warnings)

    def test_exports(self):
        r = run_pipeline("Ann: I'll book the room tomorrow.", REF)
        self.assertEqual(json.loads(r.to_json())["action_items"][0]["owner"], "Ann")
        self.assertTrue(r.to_csv().startswith("task,owner,deadline,status,confidence,flags,source"))

    def test_auto_falls_back_without_key(self):
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        self.assertEqual(run_pipeline("Ann: I'll book the room tomorrow.", REF, backend="auto").backend, "rules")

    def test_llm_without_key_falls_back_with_warning(self):
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        r = run_pipeline("Ann: I'll book the room tomorrow.", REF, backend="llm")
        self.assertEqual(r.backend, "rules")
        self.assertTrue(any("fell back" in w for w in r.warnings))

    def test_bad_backend(self):
        with self.assertRaises(ValueError):
            run_pipeline("x", REF, backend="nope")


class LLMParseTests(unittest.TestCase):
    def test_fenced_json(self):
        raw = '```json\n[{"task":"Send report","owner":"Ann","deadline":"2026-09-18","confidence":0.9,"evidence":"x"}]\n```'
        it = parse_llm_json(raw)[0]
        self.assertEqual((it.task, it.owner, it.deadline, it.confidence), ("Send report", "Ann", "2026-09-18", 0.9))

    def test_json_inside_prose_and_bad_confidence(self):
        it = parse_llm_json('Here you go: [{"task":"Do X now","confidence":"high"}] done')[0]
        self.assertEqual(it.confidence, 0.5)

    def test_non_json_raises(self):
        with self.assertRaises(LLMExtractionError):
            parse_llm_json("sorry, no")


class EvaluationRegression(unittest.TestCase):
    """Floors for every bundled dataset. 'blind' and 'heldout' were first scored BEFORE the rules were
    extended to cover them (0.67 / 0.71 F1), so treat them as regression tests, not fresh estimates."""

    def test_all_datasets(self):
        from meeting_actions.evaluate import DATA_DIR
        for d in [DATA_DIR, DATA_DIR / "heldout", DATA_DIR / "blind"]:
            row = next(r for r in evaluate_dataset(d)["by_threshold"] if r["min_confidence"] == 0.3)
            self.assertGreaterEqual(row["f1"], 0.9, d.name)
            self.assertGreaterEqual(row["owner_accuracy"], 0.9, d.name)
            self.assertGreaterEqual(row["deadline_accuracy"], 0.9, d.name)


if __name__ == "__main__":
    unittest.main()
