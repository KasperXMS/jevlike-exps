import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import (  # noqa: E402
    build_candidate_mass_analysis,
    build_milestone3_summary,
    build_paired_analysis,
)


def record(sample_id, method, correct, prediction, **metadata):
    return {
        "dataset": "test_dataset",
        "subject": None,
        "sample_id": sample_id,
        "method": method,
        "prediction": prediction,
        "gold_index": 0,
        "correct": correct,
        "probabilities": [0.7, 0.3] if method != "generation" else None,
        "confidence": None if method == "generation" else 0.7,
        "metadata": metadata,
    }


class MilestoneThreeMetricsTest(unittest.TestCase):
    def setUp(self):
        aggregation_correct = {
            "mean": {"correct": True},
            "summed": {"correct": False},
        }
        aggregation_wrong = {
            "mean": {"correct": False},
            "summed": {"correct": True},
        }
        self.records = [
            record("one", "generation", True, 0),
            record("two", "generation", False, None),
            record("one", "label_logit", True, 0, candidate_mass=0.2),
            record("two", "label_logit", True, 0, candidate_mass=0.4),
            record(
                "one",
                "option_likelihood_legacy",
                True,
                0,
                aggregations=aggregation_correct,
            ),
            record(
                "two",
                "option_likelihood_legacy",
                False,
                1,
                aggregations=aggregation_wrong,
            ),
            record(
                "one",
                "option_likelihood_exact_text",
                True,
                0,
                aggregations=aggregation_correct,
            ),
            record(
                "two",
                "option_likelihood_exact_text",
                False,
                1,
                aggregations=aggregation_wrong,
            ),
        ]

    def test_summary_generation(self):
        summary = build_milestone3_summary("model", self.records)
        metrics = summary["datasets"]["test_dataset"]
        self.assertEqual(metrics["generation"]["accuracy"], 0.5)
        self.assertEqual(metrics["generation"]["parse_rate"], 0.5)
        self.assertEqual(metrics["generation"]["conditional_accuracy"], 1.0)
        self.assertEqual(metrics["label_logit"]["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["label_logit"]["mean_candidate_mass"], 0.3)
        self.assertEqual(metrics["option_likelihood"]["accuracy_mean"], 0.5)
        self.assertEqual(metrics["option_likelihood"]["accuracy_sum"], 0.5)

    def test_paired_groups_and_candidate_mass(self):
        paired = build_paired_analysis(self.records)["test_dataset"]
        self.assertEqual(paired["G+L+"]["sample_ids"], ["one"])
        self.assertEqual(paired["G-L+"]["sample_ids"], ["two"])
        candidate = build_candidate_mass_analysis(self.records)["test_dataset"]
        self.assertAlmostEqual(candidate["mean"], 0.3)
        self.assertAlmostEqual(candidate["median"], 0.3)
        self.assertAlmostEqual(candidate["p10"], 0.22)
        self.assertAlmostEqual(candidate["p90"], 0.38)


if __name__ == "__main__":
    unittest.main()
