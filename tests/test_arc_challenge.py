import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.arc_challenge import _convert_row, load_arc_challenge  # noqa: E402


def arc_row(index: int):
    return {
        "id": f"arc-{index}",
        "question": f"Question {index}?",
        "choices": {"text": ["first", "second"], "label": ["1", "2"]},
        "answerKey": 2,
    }


class ArcChallengeTest(unittest.TestCase):
    def test_numeric_choice_labels_map_to_gold_index(self):
        sample = _convert_row(arc_row(0))
        self.assertEqual(sample.options, ["first", "second"])
        self.assertEqual(sample.gold_index, 1)

    @patch("src.datasets.arc_challenge.load_dataset")
    def test_sampling_is_deterministic(self, mocked_load_dataset):
        mocked_load_dataset.return_value = [arc_row(index) for index in range(20)]
        first = load_arc_challenge(limit=5, seed=7)
        second = load_arc_challenge(limit=5, seed=7)
        self.assertEqual([sample.id for sample in first], [sample.id for sample in second])
        self.assertEqual(len({sample.id for sample in first}), 5)


if __name__ == "__main__":
    unittest.main()
