import pytest

from src.calibration import candidate_mass_analysis
from src.metrics import build_permutation_summary


def make_record(permutation_id, prediction_original, correct, probabilities, mass):
    permutation = [0, 1, 2, 3]
    return {
        "dataset": "arc_challenge",
        "subject": None,
        "sample_id": "one",
        "permutation_id": permutation_id,
        "permuted_to_original": permutation,
        "prediction_permuted_index": prediction_original,
        "prediction_original_index": prediction_original,
        "gold_permuted_index": 1,
        "gold_original_index": 1,
        "correct": correct,
        "probabilities": probabilities,
        "probabilities_original_order": probabilities,
        "candidate_mass": mass,
        "confidence": max(probabilities),
    }


def test_permutation_summary_uses_semantic_predictions():
    records = [
        make_record(0, 1, True, [0.1, 0.7, 0.1, 0.1], 0.2),
        make_record(1, 1, True, [0.1, 0.6, 0.2, 0.1], 0.3),
        make_record(2, 2, False, [0.1, 0.2, 0.6, 0.1], 0.4),
        make_record(3, 1, True, [0.1, 0.5, 0.3, 0.1], 0.5),
    ]
    metrics = build_permutation_summary("model", records)["datasets"]["arc_challenge"]
    assert metrics["accuracy"]["original"] == 1.0
    assert metrics["accuracy"]["non_original_permutations"] == pytest.approx(2 / 3)
    assert metrics["semantic_consistency"] == pytest.approx(2 / 3)
    assert metrics["gold_consistency"]["mean_fraction_correct"] == 0.75
    assert metrics["gold_consistency"]["at_least_3_of_4_correct_ratio"] == 1.0
    assert metrics["candidate_mass"]["mean"] == pytest.approx(0.35)


def test_candidate_mass_analysis_keeps_fixed_empty_bins():
    records = [
        {"correct": True, "confidence": 0.8, "candidate_mass": 0.08},
        {"correct": False, "confidence": 0.6, "candidate_mass": 0.30},
    ]
    analysis = candidate_mass_analysis(records)
    assert [row["count"] for row in analysis["fixed_bins"]] == [1, 0, 1, 0, 0]
    assert analysis["signal_separation"]["correct"]["candidate_mass"]["mean"] == 0.08
