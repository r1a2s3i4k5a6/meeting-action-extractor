import unittest
from datetime import date

from meeting_actions.dates import is_valid_iso, parse_deadline

REF = date(2026, 9, 14)  # a Monday


class DateTests(unittest.TestCase):
    def check(self, text, expected):
        m = parse_deadline(text, REF)
        self.assertIsNotNone(m, text)
        self.assertEqual(m.iso, expected, text)

    def test_weekdays(self):
        self.check("by Friday", "2026-09-18")
        self.check("by next Wednesday", "2026-09-23")
        self.check("on Monday", "2026-09-21")  # same weekday -> next week

    def test_relative(self):
        self.check("tomorrow", "2026-09-15")
        self.check("by end of day", "2026-09-14")
        self.check("by the end of this week", "2026-09-18")
        self.check("next week", "2026-09-21")
        self.check("in 2 weeks", "2026-09-28")
        self.check("by end of month", "2026-09-30")

    def test_absolute(self):
        self.check("by September 25", "2026-09-25")
        self.check("on the 30th of September", "2026-09-30")
        self.check("by 9/25", "2026-09-25")
        self.check("by 25/9", "2026-09-25")
        self.check("by 2026-10-01", "2026-10-01")
        self.check("by January 5", "2027-01-05")  # already past this year -> next year

    def test_none_and_invalid(self):
        self.assertIsNone(parse_deadline("before the release", REF))
        self.assertIsNone(parse_deadline("by February 30", REF))
        self.assertFalse(is_valid_iso("2026-02-30"))
        self.assertTrue(is_valid_iso("2026-02-28"))
        self.assertFalse(is_valid_iso("next friday"))

    def test_phrase_includes_preposition(self):
        self.assertEqual(parse_deadline("send it by Friday", REF).phrase, "by Friday")


if __name__ == "__main__":
    unittest.main()


class MoreDateTests(unittest.TestCase):
    def test_new_expressions(self):
        r = date(2026, 10, 5)  # Monday
        for text, exp in [("before the 15th", "2026-10-15"), ("by the 3rd", "2026-11-03"),
                          ("this afternoon", "2026-10-05"), ("by 3 pm", "2026-10-05"),
                          ("tomorrow morning", "2026-10-06"), ("by end of next week", "2026-10-16"),
                          ("on Friday morning", "2026-10-09")]:
            self.assertEqual(parse_deadline(text, r).iso, exp, text)

    def test_week_of_is_not_a_deadline(self):
        self.assertIsNone(parse_deadline("for the week of November 2nd", date(2026, 10, 21)))


if __name__ == "__main__":
    unittest.main()
