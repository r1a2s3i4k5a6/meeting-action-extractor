import unittest

from meeting_actions.schema import STATUSES, ActionItem
from meeting_actions.ui_helpers import UNASSIGNED, filter_rows, items_to_rows, owners_in, summary_line

ITEMS = [
    ActionItem("Fix the bug", "Ann", "2026-10-09", "open", 0.9, "Ann: I'll fix it", []),
    ActionItem("Book the room", None, None, "needs_review", 0.4, "Someone should book", ["missing_owner", "missing_deadline"]),
    ActionItem("Send the deck", "Bob", None, "open", 0.8, "Bob: will send", ["missing_deadline"]),
]


class UIHelperTests(unittest.TestCase):
    def test_rows_flatten_flags(self):
        rows = items_to_rows(ITEMS)
        self.assertEqual(rows[1]["flags"], "missing_owner, missing_deadline")
        self.assertEqual(list(rows[0]), ["task", "owner", "deadline", "status", "confidence", "flags", "source"])

    def test_owners_and_filters(self):
        rows = items_to_rows(ITEMS)
        self.assertEqual(owners_in(rows), ["Ann", "Bob"])
        self.assertEqual([r["task"] for r in filter_rows(rows, ["Ann"])], ["Fix the bug"])
        self.assertEqual([r["task"] for r in filter_rows(rows, [UNASSIGNED])], ["Book the room"])
        self.assertEqual(len(filter_rows(rows, ["Ann", UNASSIGNED])), 2)
        self.assertEqual([r["task"] for r in filter_rows(rows, only_review=True)], ["Book the room"])
        self.assertEqual(len(filter_rows(rows)), 3)

    def test_summary_and_statuses(self):
        self.assertTrue(summary_line(ITEMS).startswith("3 action items, 1 need review"))
        self.assertEqual(STATUSES, ("open", "needs_review", "done"))


if __name__ == "__main__":
    unittest.main()
