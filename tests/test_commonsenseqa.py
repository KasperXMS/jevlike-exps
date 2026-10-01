import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.commonsenseqa import (  # noqa: E402
    _convert_row,
    load_commonsenseqa,
    load_commonsenseqa_by_ids,
)
from src.schema import MultipleChoiceSample  # noqa: E402


def flat_row(index: int):
    return {
        "id": f"csqa-{index}",
        "question": f"Question {index}?",
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": ["a", "b", "c", "d", "e"],
        },
        "answerKey": "C",
    }


class CommonsenseQATest(unittest.TestCase):
    def test_flat_huggingface_conversion(self):
        sample = _convert_row(flat_row(1))
        self.assertIsInstance(sample, MultipleChoiceSample)
        self.assertEqual(sample.dataset, "commonsenseqa")
        self.assertEqual(sample.gold_index, 2)
        self.assertEqual(len(sample.options), 5)

    def test_nested_question_conversion(self):
        row = flat_row(2)
        row["question"] = {
            "stem": "Nested question?",
            "choices": row.pop("choices"),
        }
        sample = _convert_row(row)
        self.assertEqual(sample.question, "Nested question?")
        self.assertEqual(sample.options[2], "c")

    @patch("src.datasets.commonsenseqa.load_dataset")
    def test_sampling_is_deterministic(self, mocked_load_dataset):
        mocked_load_dataset.return_value = [flat_row(index) for index in range(150)]
        first = load_commonsenseqa(limit=25, seed=42)
        second = load_commonsenseqa(limit=25, seed=42)
        self.assertEqual([sample.id for sample in first], [sample.id for sample in second])

    @patch("src.datasets.commonsenseqa.load_dataset")
    def test_frozen_ids_preserve_order(self, mocked_load_dataset):
        mocked_load_dataset.return_value = [flat_row(index) for index in range(5)]
        restored = load_commonsenseqa_by_ids(["csqa-4", "csqa-0"])
        self.assertEqual([sample.id for sample in restored], ["csqa-4", "csqa-0"])


if __name__ == "__main__":
    unittest.main()
