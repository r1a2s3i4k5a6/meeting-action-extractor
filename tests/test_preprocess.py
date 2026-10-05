import unittest

from meeting_actions.preprocess import clean_text, parse_transcript, participants_of, segment, split_sentences


class PreprocessTests(unittest.TestCase):
    def test_filler_removal(self):
        self.assertEqual(clean_text("So, um, I'll draft it, uh, today."), "So, I'll draft it, today.")

    def test_speakers_timestamps_and_continuation(self):
        t = "[00:01:02] Ann: Hello there.\nBob Smith: Hi.\nstill Bob talking\n"
        u = parse_transcript(t)
        self.assertEqual([x.speaker for x in u], ["Ann", "Bob Smith"])
        self.assertEqual(u[1].text, "Hi. still Bob talking")

    def test_vtt(self):
        t = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n<v Ann>Hello.</v>\n"
        u = parse_transcript(t)
        self.assertEqual((u[0].speaker, u[0].text), ("Ann", "Hello."))

    def test_sentences_and_abbreviations(self):
        self.assertEqual(split_sentences("Ask Dr. Lee. Then go! Ok?"), ["Ask Dr. Lee.", "Then go!", "Ok?"])

    def test_segments_and_participants(self):
        u = parse_transcript("Ann: One. Two.\nBob: Three.")
        self.assertEqual(participants_of(u), ["Ann", "Bob"])
        self.assertEqual([(s.speaker, s.utt) for s in segment(u)], [("Ann", 0), ("Ann", 0), ("Bob", 1)])


if __name__ == "__main__":
    unittest.main()
