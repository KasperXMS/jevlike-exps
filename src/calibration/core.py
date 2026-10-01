from collections import defaultdict
import hashlib
import math
from statistics import mean, median
from typing import Any


def softmax_temperature(logits: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [value / temperature for value in logits]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _mean_nll(
    logits: list[list[float]], gold_indices: list[int], temperature: float
) -> float:
    return mean(
        -math.log(max(softmax_temperature(values, temperature)[gold], 1e-15))
        for values, gold in zip(logits, gold_indices)
    )


def fit_temperature(
    logits: list[list[float]], gold_indices: list[int]
) -> float:
    if not logits or len(logits) != len(gold_indices):
        raise ValueError("logits and gold_indices must be non-empty and aligned")
    # Bounded one-dimensional optimization in log-temperature space is stable,
    # deterministic, and supports a pooled set with different class counts.
    left, right = -5.0, 5.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = _mean_nll(logits, gold_indices, math.exp(x1))
    f2 = _mean_nll(logits, gold_indices, math.exp(x2))
    for _ in range(100):
        if f1 > f2:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = _mean_nll(logits, gold_indices, math.exp(x2))
        else:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = _mean_nll(logits, gold_indices, math.exp(x1))
    return math.exp((left + right) / 2.0)


def calibration_metrics(
    probabilities: list[list[float]], gold_indices: list[int], bins: int = 15
) -> dict[str, Any]:
    if not probabilities or len(probabilities) != len(gold_indices):
        raise ValueError("probabilities and gold_indices must be non-empty and aligned")
    predictions = [max(range(len(row)), key=row.__getitem__) for row in probabilities]
    confidences = [max(row) for row in probabilities]
    correct = [prediction == gold for prediction, gold in zip(predictions, gold_indices)]
    bin_rows = []
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            position
            for position, confidence in enumerate(confidences)
            if min(int(confidence * bins), bins - 1) == index
        ]
        mean_confidence = mean(confidences[position] for position in members) if members else None
        empirical_accuracy = mean(correct[position] for position in members) if members else None
        if members:
            ece += len(members) / len(probabilities) * abs(mean_confidence - empirical_accuracy)
        bin_rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
            }
        )
    return {
        "count": len(probabilities),
        "accuracy": mean(correct),
        "nll": mean(
            -math.log(max(row[gold], 1e-15))
            for row, gold in zip(probabilities, gold_indices)
        ),
        "brier": mean(
            sum((probability - (index == gold)) ** 2 for index, probability in enumerate(row))
            for row, gold in zip(probabilities, gold_indices)
        ),
        "ece": ece,
        "ece_bins": bin_rows,
    }


def _stable_key(seed: int, dataset: str, subject: str, sample_id: str) -> str:
    value = f"{seed}:{dataset}:{subject}:{sample_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def create_calibration_split(
    records: list[dict[str, Any]], seed: int = 42, fraction: float = 0.2
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        subject = record.get("subject") or ""
        stratum = subject if record["dataset"] == "mmlu" else ""
        grouped[(record["dataset"], stratum)].append(record["sample_id"])

    datasets: dict[str, dict[str, list[str]]] = {}
    for (dataset, subject), ids in sorted(grouped.items()):
        unique_ids = sorted(set(ids), key=lambda value: _stable_key(seed, dataset, subject, value))
        calibration_count = max(1, round(len(unique_ids) * fraction))
        entry = datasets.setdefault(dataset, {"calibration_ids": [], "test_ids": []})
        entry["calibration_ids"].extend(unique_ids[:calibration_count])
        entry["test_ids"].extend(unique_ids[calibration_count:])
    for entry in datasets.values():
        if set(entry["calibration_ids"]) & set(entry["test_ids"]):
            raise AssertionError("calibration and test IDs overlap")
    return {"seed": seed, "calibration_fraction": fraction, "datasets": datasets}


def _auroc(labels: list[bool], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _signal_summary(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "median": median(values)}


def candidate_mass_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    boundaries = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    bins = []
    for index, (lower, upper) in enumerate(zip(boundaries, boundaries[1:])):
        members = [
            record
            for record in records
            if lower <= record["candidate_mass"]
            and (record["candidate_mass"] < upper or (index == len(boundaries) - 2 and record["candidate_mass"] <= upper))
        ]
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "accuracy": mean(record["correct"] for record in members) if members else None,
                "mean_restricted_confidence": mean(record["confidence"] for record in members) if members else None,
                "mean_candidate_mass": mean(record["candidate_mass"] for record in members) if members else None,
            }
        )
    correct = [record for record in records if record["correct"]]
    incorrect = [record for record in records if not record["correct"]]
    labels = [bool(record["correct"]) for record in records]
    confidence = [record["confidence"] for record in records]
    mass = [record["candidate_mass"] for record in records]
    return {
        "fixed_bins": bins,
        "signal_separation": {
            "correct": {
                "count": len(correct),
                "restricted_confidence": _signal_summary([record["confidence"] for record in correct]),
                "candidate_mass": _signal_summary([record["candidate_mass"] for record in correct]),
            },
            "incorrect": {
                "count": len(incorrect),
                "restricted_confidence": _signal_summary([record["confidence"] for record in incorrect]),
                "candidate_mass": _signal_summary([record["candidate_mass"] for record in incorrect]),
            },
            "auroc": {
                "restricted_confidence": _auroc(labels, confidence),
                "candidate_mass": _auroc(labels, mass),
                "confidence_times_mass": _auroc(labels, [a * b for a, b in zip(confidence, mass)]),
            },
        },
    }
