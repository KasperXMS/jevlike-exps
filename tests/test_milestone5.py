import pytest

from src.metrics import (
    build_calibration_observation,
    build_invariance_summary,
    build_original_summary,
    build_paired,
)


def original_record(sample_id, method, correct, prediction=1, gold=1):
    metadata = {"candidate_mass": 0.4}
    return {
        "dataset": "arc_challenge",
        "subject": None,
        "sample_id": sample_id,
        "method": method,
        "prediction": prediction,
        "gold_index": gold,
        "correct": correct,
        "probabilities": [0.2, 0.6, 0.1, 0.1] if method == "label_logit" else None,
        "confidence": 0.6 if method == "label_logit" else None,
        "metadata": metadata,
    }


def permutation_record(sample_id, permutation_id, correct, prediction_original):
    probabilities = [0.1, 0.6, 0.2, 0.1]
    return {
        "dataset": "arc_challenge",
        "subject": None,
        "sample_id": sample_id,
        "permutation_id": permutation_id,
        "prediction_permuted_index": prediction_original,
        "prediction_original_index": prediction_original,
        "gold_permuted_index": 1,
        "gold_original_index": 1,
        "correct": correct,
        "probabilities": probabilities,
        "probabilities_original_order": probabilities,
        "confidence": 0.6,
        "candidate_mass": 0.4,
    }


def test_original_summary_and_paired_groups():
    records = [
        original_record("a", "generation", True),
        original_record("a", "label_logit", True),
        original_record("b", "generation", False, prediction=None),
        original_record("b", "label_logit", True),
    ]
    summary = build_original_summary("model", 10, records)
    metrics = summary["datasets"]["arc_challenge"]
    assert metrics["generation"]["accuracy"] == 0.5
    assert metrics["generation"]["parse_rate"] == 0.5
    assert metrics["label_logit"]["accuracy"] == 1.0
    paired = build_paired(records)["arc_challenge"]
    assert paired["G+L+"]["sample_ids"] == ["a"]
    assert paired["G-L+"]["sample_ids"] == ["b"]


def test_invariance_reports_flip_cancellation_separately():
    records = []
    records.extend(
        permutation_record("correct-original", index, index != 1, 1 if index != 1 else 2)
        for index in range(4)
    )
    records.extend(
        permutation_record("wrong-original", index, index in {1, 2}, 1 if index in {1, 2} else 2)
        for index in range(4)
    )
    summary = build_invariance_summary("model", 10, records)["datasets"]["arc_challenge"]
    assert summary["accuracy"]["original"] == 0.5
    assert summary["accuracy"]["non_original_permutations"] == pytest.approx(2 / 3)
    assert summary["flip_rates"]["correct_to_wrong"]["rate"] == pytest.approx(1 / 3)
    assert summary["flip_rates"]["wrong_to_correct"]["rate"] == pytest.approx(2 / 3)
    assert "median_probability_range" in summary["probability_stability"]["gold_option"]


def test_calibration_observation_uses_only_frozen_test_ids():
    records = [
        original_record("cal", "label_logit", True),
        original_record("test-good", "label_logit", True),
        original_record("test-bad", "label_logit", False, prediction=1, gold=0),
    ]
    split = {
        "datasets": {
            "arc_challenge": {
                "calibration_ids": ["cal"],
                "test_ids": ["test-good", "test-bad"],
            }
        }
    }
    output = build_calibration_observation("model", 10, records, split)
    assert output["datasets"]["arc_challenge"]["count"] == 2
    assert output["pooled"]["count"] == 2
