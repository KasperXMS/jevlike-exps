import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.mmlu import (  # noqa: E402
    _convert_row,
    load_mmlu,
    load_mmlu_by_ids,
    resolve_mmlu_subject,
)
from src.schema import MultipleChoiceSample  # noqa: E402


def mmlu_row(index: int):
    return {
        "question": f"Question {index}?",
        "subject": "college_computer_science",
        "choices": ["zero", "one", "two", "three"],
        "answer": index % 4,
    }


class MMLUTest(unittest.TestCase):
    def test_conversion_is_schema_compatible(self):
        sample = _convert_row(
            mmlu_row(2), "computer_science", "college_computer_science", 2
        )
        self.assertIsInstance(sample, MultipleChoiceSample)
        self.assertEqual(sample.dataset, "mmlu")
        self.assertEqual(sample.subject, "computer_science")
        self.assertEqual(sample.gold_index, 2)
        self.assertEqual(len(sample.options), 4)
        self.assertTrue(sample.id.startswith("mmlu:computer_science:"))

    @patch("src.datasets.mmlu.load_dataset")
    def test_sampling_is_deterministic_per_subject(self, mocked_load_dataset):
        mocked_load_dataset.return_value = [mmlu_row(index) for index in range(150)]
        first = load_mmlu(["computer_science"], limit=20, seed=42)
        second = load_mmlu(["computer_science"], limit=20, seed=42)
        self.assertEqual([sample.id for sample in first], [sample.id for sample in second])
        self.assertEqual(mocked_load_dataset.call_args.args[1], "college_computer_science")

    @patch("src.datasets.mmlu.load_dataset")
    def test_frozen_ids_preserve_order(self, mocked_load_dataset):
        mocked_load_dataset.return_value = [mmlu_row(index) for index in range(5)]
        all_samples = load_mmlu(["computer_science"], limit=None)
        wanted = [all_samples[3].id, all_samples[1].id]
        restored = load_mmlu_by_ids(["computer_science"], wanted)
        self.assertEqual([sample.id for sample in restored], wanted)

    def test_subject_aliases_are_explicit(self):
        self.assertEqual(resolve_mmlu_subject("physics"), "college_physics")
        self.assertEqual(resolve_mmlu_subject("abstract_algebra"), "abstract_algebra")


if __name__ == "__main__":
    unittest.main()
