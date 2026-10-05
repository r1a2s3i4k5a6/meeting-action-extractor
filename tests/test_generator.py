import importlib.util
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from meeting_actions.dates import is_valid_iso
from meeting_actions.evaluate import evaluate_dataset
from meeting_actions.preprocess import parse_transcript

SPEC = importlib.util.spec_from_file_location("gen", Path(__file__).resolve().parent.parent / "scripts" / "generate_synthetic.py")
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)


class GeneratorTests(unittest.TestCase):
    def make(self, seed=3, n=8):
        d = Path(tempfile.mkdtemp())
        gen.generate(n, seed, d)
        return d, json.loads((d / "annotations.json").read_text())

    def test_deterministic(self):
        _, a = self.make()
        _, b = self.make()
        self.assertEqual(a, b)
        self.assertNotEqual(a, self.make(seed=4)[1])

    def test_gold_is_consistent_with_transcript(self):
        d, ann = self.make()
        for m in ann["meetings"]:
            text = (d / "sample_transcripts" / m["file"]).read_text()
            speakers = {u.speaker for u in parse_transcript(text)}
            self.assertGreaterEqual(len(m["action_items"]), 4)
            for g in m["action_items"]:
                self.assertIn(g["task"].lower(), text.lower())
                self.assertTrue(g["deadline"] is None or is_valid_iso(g["deadline"]))
                self.assertTrue(g["owner"] is None or g["owner"] in speakers)
            self.assertTrue(date.fromisoformat(m["meeting_date"]).weekday() < 5)

    def test_evaluation_with_breakdown(self):
        d, _ = self.make()
        rep = evaluate_dataset(d)
        self.assertIn("breakdown", rep)
        self.assertEqual(set(rep["breakdown"]), {"by_template", "by_deadline_kind", "false_positives_by_kind"})
        self.assertGreater(rep["by_threshold"][1]["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
