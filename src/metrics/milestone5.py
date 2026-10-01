from collections import defaultdict
from statistics import mean
from typing import Any

from ..calibration import calibration_metrics, candidate_mass_analysis
from .milestone4 import build_permutation_summary


def _accuracy(records: list[dict[str, Any]]) -> float:
    return mean(bool(record["correct"]) for record in records)


def build_original_summary(
    model: str, parameter_count: int, records: list[dict[str, Any]]
) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)
    datasets = {}
    for dataset, dataset_records in by_dataset.items():
        generation = [record for record in dataset_records if record["method"] == "generation"]
        label = [record for record in dataset_records if record["method"] == "label_logit"]
        parsed = [record for record in generation if record["prediction"] is not None]
        datasets[dataset] = {
            "sample_count": len(generation),
            "generation": {
                "accuracy": _accuracy(generation),
                "parse_rate": len(parsed) / len(generation),
                "conditional_accuracy": _accuracy(parsed) if parsed else None,
            },
            "label_logit": {
                "accuracy": _accuracy(label),
                "mean_candidate_mass": mean(record["metadata"]["candidate_mass"] for record in label),
                "mean_restricted_confidence": mean(record["confidence"] for record in label),
            },
        }
    return {"model": model, "parameter_count": parameter_count, "datasets": datasets}


def build_paired(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)
    output = {}
    for dataset, dataset_records in by_dataset.items():
        generation = {r["sample_id"]: r for r in dataset_records if r["method"] == "generation"}
        label = {r["sample_id"]: r for r in dataset_records if r["method"] == "label_logit"}
        if generation.keys() != label.keys():
            raise ValueError(f"Unpaired original results for {dataset}")
        groups = {"G+L+": [], "G+L-": [], "G-L+": [], "G-L-": []}
        for sample_id, generation_record in generation.items():
            key = f"G{'+' if generation_record['correct'] else '-'}L{'+' if label[sample_id]['correct'] else '-'}"
            groups[key].append(sample_id)
        output[dataset] = {
            name: {"count": len(ids), "sample_ids": ids} for name, ids in groups.items()
        }
    return output


def _flip_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sample[record["sample_id"]].append(record)
    correct_to_wrong_count = 0
    correct_to_wrong_total = 0
    wrong_to_correct_count = 0
    wrong_to_correct_total = 0
    for sample_records in by_sample.values():
        original = next(record for record in sample_records if record["permutation_id"] == 0)
        permutations = [record for record in sample_records if record["permutation_id"] != 0]
        if original["correct"]:
            correct_to_wrong_count += sum(not record["correct"] for record in permutations)
            correct_to_wrong_total += len(permutations)
        else:
            wrong_to_correct_count += sum(record["correct"] for record in permutations)
            wrong_to_correct_total += len(permutations)
    return {
        "correct_to_wrong": {
            "count": correct_to_wrong_count,
            "opportunities": correct_to_wrong_total,
            "rate": correct_to_wrong_count / correct_to_wrong_total if correct_to_wrong_total else None,
        },
        "wrong_to_correct": {
            "count": wrong_to_correct_count,
            "opportunities": wrong_to_correct_total,
            "rate": wrong_to_correct_count / wrong_to_correct_total if wrong_to_correct_total else None,
        },
    }


def build_invariance_summary(
    model: str, parameter_count: int, records: list[dict[str, Any]]
) -> dict[str, Any]:
    summary = build_permutation_summary(model, records)
    summary["parameter_count"] = parameter_count
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)
    for dataset, dataset_records in by_dataset.items():
        summary["datasets"][dataset]["flip_rates"] = _flip_metrics(dataset_records)
        if dataset == "mmlu":
            for subject in summary["datasets"][dataset]["subjects"]:
                subject_records = [r for r in dataset_records if r["subject"] == subject]
                summary["datasets"][dataset]["subjects"][subject]["flip_rates"] = _flip_metrics(subject_records)
    return summary


def build_calibration_observation(
    model: str,
    parameter_count: int,
    records: list[dict[str, Any]],
    split: dict[str, Any],
    bins: int = 15,
) -> dict[str, Any]:
    labels = [record for record in records if record["method"] == "label_logit"]
    datasets = {}
    pooled = []
    for dataset, split_entry in split["datasets"].items():
        test_ids = set(split_entry["test_ids"])
        test_records = [
            record for record in labels
            if record["dataset"] == dataset and record["sample_id"] in test_ids
        ]
        probabilities = [record["probabilities"] for record in test_records]
        gold = [record["gold_index"] for record in test_records]
        metrics = calibration_metrics(probabilities, gold, bins=bins)
        signals = candidate_mass_analysis(
            [
                {
                    "correct": record["correct"],
                    "confidence": record["confidence"],
                    "candidate_mass": record["metadata"]["candidate_mass"],
                }
                for record in test_records
            ]
        )["signal_separation"]
        datasets[dataset] = {**metrics, "signals": signals}
        pooled.extend(test_records)
    pooled_metrics = calibration_metrics(
        [record["probabilities"] for record in pooled],
        [record["gold_index"] for record in pooled],
        bins=bins,
    )
    pooled_signals = candidate_mass_analysis(
        [
            {
                "correct": record["correct"],
                "confidence": record["confidence"],
                "candidate_mass": record["metadata"]["candidate_mass"],
            }
            for record in pooled
        ]
    )["signal_separation"]
    return {
        "model": model,
        "parameter_count": parameter_count,
        "datasets": datasets,
        "pooled": {**pooled_metrics, "signals": pooled_signals},
    }
