import unittest
from datetime import date

from meeting_actions.schema import ActionItem
from meeting_actions.validation import validate_items

REF = date(2026, 9, 14)
ROSTER = ["Sarah Lee", "Tom"]


class ValidationTests(unittest.TestCase):
    def run_v(self, items):
        return validate_items(items, ROSTER, REF)

    def test_missing_owner_flagged_and_review(self):
        r = self.run_v([ActionItem("Decide pricing tiers", None, "2026-09-20", confidence=0.9)])
        it = r.items[0]
        self.assertIn("missing_owner", it.flags)
        self.assertEqual(it.status, "needs_review")
        self.assertAlmostEqual(it.confidence, 0.75)

    def test_owner_normalisation_and_unknown(self):
        r = self.run_v([ActionItem("Send the report", "sarah", "2026-09-20", confidence=0.9),
                        ActionItem("Book the room", "Zed", "2026-09-20", confidence=0.9)])
        self.assertEqual(r.items[0].owner, "Sarah Lee")
        self.assertIn("unknown_owner", r.items[1].flags)

    def test_invalid_and_past_dates(self):
        r = self.run_v([ActionItem("Send the report", "Tom", "2026-02-30", confidence=0.9),
                        ActionItem("Book the room", "Tom", "2026-09-01", confidence=0.9),
                        ActionItem("Write the spec", "Tom", "next friday", confidence=0.9)])
        self.assertIsNone(r.items[0].deadline)
        self.assertIn("invalid_deadline", r.items[0].flags)
        self.assertIn("deadline_in_past", r.items[1].flags)
        self.assertIn("invalid_deadline", r.items[2].flags)
        self.assertTrue(all(i.status == "needs_review" for i in r.items))

    def test_duplicates_merged_and_fill_gaps(self):
        r = self.run_v([ActionItem("Draft the October newsletter", "Tom", None, confidence=0.8),
                        ActionItem("Draft the October newsletter.", "Tom", "2026-09-17", confidence=0.85)])
        self.assertEqual(len(r.items), 1)
        self.assertEqual(r.items[0].deadline, "2026-09-17")
        self.assertIn("merged_duplicate", r.items[0].flags)

    def test_different_owner_not_merged(self):
        r = self.run_v([ActionItem("Draft the newsletter", "Tom", "2026-09-17"),
                        ActionItem("Draft the newsletter", "Sarah Lee", "2026-09-17")])
        self.assertEqual(len(r.items), 2)

    def test_short_task_dropped_and_input_not_mutated(self):
        src = ActionItem("Ok", "Tom", "2026-09-17")
        r = self.run_v([src, ActionItem("Send the invoice", "Tom", "2026-09-17")])
        self.assertEqual(len(r.items), 1)
        self.assertEqual(src.flags, [])

    def test_clean_item_is_open(self):
        r = self.run_v([ActionItem("Send the invoice", "Tom", "2026-09-17", confidence=0.9)])
        self.assertEqual(r.items[0].status, "open")


if __name__ == "__main__":
    unittest.main()
