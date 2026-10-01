import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.generation import parse_generation_answer  # noqa: E402


class GenerationParserTest(unittest.TestCase):
    def test_valid_explicit_answer_forms(self):
        cases = (
            ("C", 2, "bare_label"),
            ("C.", 2, "label_period"),
            ("(C)", 2, "parenthesized_label"),
            ("C because it is correct", 2, "label_because"),
            ("Answer: C", 2, "answer_colon"),
            ("Answer is C", 2, "answer_is"),
            ("The answer is C", 2, "the_answer_is"),
            ("Option C", 2, "option_label"),
            ("Therefore, the answer is C.", 2, "the_answer_is"),
            ("The correct answer is:\n\n**B", 1, "the_correct_answer_is"),
            ("Answer: **C**", 2, "answer_colon"),
            ("\nC. Paris", 2, "label_period"),
        )
        for text, prediction, pattern in cases:
            with self.subTest(text=text):
                parsed = parse_generation_answer(text, option_count=4)
                self.assertEqual(parsed.prediction, prediction)
                self.assertEqual(parsed.pattern, pattern)
                self.assertIsNotNone(parsed.pattern_regex)
                self.assertIsNotNone(parsed.matched_text)

    def test_invalid_prose_is_not_an_answer(self):
        cases = (
            "a common phenomenon",
            "This is a common phenomenon.",
            "A common phenomenon",
            "The answer is a common phenomenon.",
            "Vitamin B is important.",
            "C is the chemical symbol for carbon.",
            "Choice C looks plausible.",
            "Option c",
            "E.",
            "",
        )
        for text in cases:
            with self.subTest(text=text):
                parsed = parse_generation_answer(text, option_count=4)
                self.assertIsNone(parsed.prediction)
                self.assertIsNone(parsed.pattern)
                self.assertIsNone(parsed.pattern_regex)


if __name__ == "__main__":
    unittest.main()
