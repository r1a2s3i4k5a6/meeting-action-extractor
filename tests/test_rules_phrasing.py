import unittest
from datetime import date

from meeting_actions import run_pipeline

REF = date(2026, 10, 5)


def items(text, **kw):
    return run_pipeline(text, REF, min_confidence=0.0, **kw).items


class PhrasingTests(unittest.TestCase):
    def one(self, text, **kw):
        out = items(text, **kw)
        self.assertEqual(len(out), 1, [i.task for i in out])
        return out[0]

    def test_you_will_need_to(self):
        i = self.one("Al: Bea, you'll need to get the sign-off before the 15th.\nBea: Ok.")
        self.assertEqual((i.owner, i.task, i.deadline), ("Bea", "Get the sign-off", "2026-10-15"))

    def test_dont_forget_and_id_like_you_to(self):
        a = self.one("Al: Bea, don't forget to send the media list.\nBea: Ok.")
        self.assertEqual((a.owner, a.task), ("Bea", "Send the media list"))
        b = self.one("Al: Bea, I'd like you to look into the errors by Thursday.\nBea: Ok.")
        self.assertEqual((b.owner, b.deadline), ("Bea", "2026-10-08"))

    def test_volunteering_and_soft_commitment(self):
        a = self.one("Al: I can take care of the press release.\nBea: Ok.")
        self.assertEqual((a.owner, a.task), ("Al", "Take care of the press release"))
        b = self.one("Al: I should have a fix by Friday.\nBea: Ok.")
        self.assertEqual((b.owner, b.deadline), ("Al", "2026-10-09"))

    def test_someone_should_is_unassigned(self):
        i = self.one("Al: Someone should book a room.\nBea: Ok.")
        self.assertIsNone(i.owner)
        self.assertEqual(i.status, "needs_review")

    def test_pronoun_resolution(self):
        i = self.one("Al: The refund doc is outdated. We have to rewrite it before the end of the month.\nBea: Ok.")
        self.assertEqual(i.task, "Rewrite the refund doc")
        self.assertIn("pronoun_resolved", i.flags)

    def test_i_will_do_that_assigns_the_open_item(self):
        i = self.one("Al: We need someone to update the checklist.\nBea: I'll do that, probably early next week.")
        self.assertEqual((i.owner, i.task), ("Bea", "Update the checklist"))

    def test_time_of_day_removed_from_task(self):
        i = self.one("Al: I'll push the docs by 3 pm today.\nBea: Ok.")
        self.assertEqual((i.task, i.deadline), ("Push the docs", "2026-10-05"))

    def test_reminder_to_everyone(self):
        i = self.one("Al: Please remember to submit your reports by the 25th, everyone.\nBea: Ok.")
        self.assertEqual((i.task, i.deadline, i.owner), ("Submit your reports", "2026-10-25", None))

    def test_attendees_who_never_speak_can_own_tasks(self):
        text = "Al: Priya, can you send the deck by Friday?\nBea: Sounds good."
        self.assertEqual(self.one(text, extra_participants=["Priya"]).owner, "Priya")
        self.assertIsNone(self.one(text).owner)  # not a known participant -> not guessed

    def test_ambiguous_first_name_goes_to_review(self):
        t = "Sam Lee: Sam, can you book the room?\nSam Ortiz: Sure."
        i = self.one(t)
        self.assertIn("unknown_owner", i.flags)
        self.assertEqual(i.status, "needs_review")


if __name__ == "__main__":
    unittest.main()
