import math

import pytest

from src.calibration import (
    calibration_metrics,
    create_calibration_split,
    fit_temperature,
    softmax_temperature,
)


def test_temperature_softmax_is_normalized_and_preserves_argmax():
    logits = [1.0, 4.0, -2.0]
    for temperature in [0.1, 1.0, 10.0]:
        probabilities = softmax_temperature(logits, temperature)
        assert sum(probabilities) == pytest.approx(1.0)
        assert max(range(3), key=probabilities.__getitem__) == 1
    with pytest.raises(ValueError):
        softmax_temperature(logits, 0.0)


def test_fitted_temperature_is_positive_and_deterministic():
    logits = [[3.0, 0.0], [2.0, 1.0], [4.0, -1.0], [1.0, 2.0]]
    gold = [0, 1, 0, 1]
    first = fit_temperature(logits, gold)
    second = fit_temperature(logits, gold)
    assert first > 0
    assert first == pytest.approx(second)


def test_calibration_metrics_on_small_example():
    metrics = calibration_metrics([[0.8, 0.2], [0.4, 0.6]], [0, 0], bins=2)
    assert metrics["accuracy"] == 0.5
    assert metrics["nll"] == pytest.approx(-(math.log(0.8) + math.log(0.4)) / 2)
    assert metrics["brier"] == pytest.approx(0.4)
    assert metrics["ece"] == pytest.approx(0.2)
    assert len(metrics["ece_bins"]) == 2
    assert metrics["ece_bins"][1]["count"] == 2


def test_split_is_deterministic_stratified_and_has_no_overlap():
    records = []
    for subject in ["a", "b"]:
        for index in range(10):
            records.append(
                {
                    "dataset": "mmlu",
                    "subject": subject,
                    "sample_id": f"{subject}-{index}",
                }
            )
    first = create_calibration_split(records, seed=42)
    second = create_calibration_split(list(reversed(records)), seed=42)
    assert first == second
    entry = first["datasets"]["mmlu"]
    calibration = set(entry["calibration_ids"])
    test = set(entry["test_ids"])
    assert len(calibration) == 4
    assert len(test) == 16
    assert not calibration & test
    assert len([value for value in calibration if value.startswith("a-")]) == 2
    assert len([value for value in calibration if value.startswith("b-")]) == 2
