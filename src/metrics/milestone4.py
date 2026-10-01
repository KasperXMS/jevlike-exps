from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import Any


def _position_distribution(
    records: list[dict[str, Any]], key: str
) -> dict[str, dict[str, float | int]]:
    counts = Counter(record[key] for record in records if record[key] is not None)
    option_counts = Counter(len(record["probabilities"]) for record in records)
    output = {}
    for position in range(max(option_counts, default=0)):
        eligible = sum(count for size, count in option_counts.items() if size > position)
        output[chr(ord("A") + position)] = {
            "count": counts[position],
            "eligible_count": eligible,
            "normalized_frequency": counts[position] / eligible if eligible else 0.0,
        }
    return output


def _stability_summary(
    sample_records: list[dict[str, Any]], semantic_index: int
) -> tuple[float, float, float]:
    values = [record["probabilities_original_order"][semantic_index] for record in sample_records]
    return mean(values), pstdev(values), max(values) - min(values)


def _group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sample[record["sample_id"]].append(record)
    for sample_records in by_sample.values():
        sample_records.sort(key=lambda record: record["permutation_id"])
        if [record["permutation_id"] for record in sample_records] != [0, 1, 2, 3]:
            raise ValueError("Each sample must have exactly permutation IDs 0, 1, 2, 3")

    originals = [record for record in records if record["permutation_id"] == 0]
    non_originals = [record for record in records if record["permutation_id"] != 0]
    semantic_matches = 0
    gold_fractions = []
    correct_counts = []
    gold_stability = []
    predicted_stability = []
    candidate_stds = []
    candidate_ranges = []
    all_option_stds = []
    all_option_ranges = []
    for sample_records in by_sample.values():
        original_prediction = sample_records[0]["prediction_original_index"]
        semantic_matches += sum(
            record["prediction_original_index"] == original_prediction
            for record in sample_records[1:]
        )
        correct_count = sum(record["correct"] for record in sample_records)
        correct_counts.append(correct_count)
        gold_fractions.append(correct_count / len(sample_records))
        gold_index = sample_records[0]["gold_original_index"]
        gold_stability.append(_stability_summary(sample_records, gold_index))
        predicted_stability.append(
            _stability_summary(sample_records, original_prediction)
        )
        masses = [record["candidate_mass"] for record in sample_records]
        candidate_stds.append(pstdev(masses))
        candidate_ranges.append(max(masses) - min(masses))
        for option_index in range(len(sample_records[0]["probabilities"])):
            _, standard_deviation, value_range = _stability_summary(
                sample_records, option_index
            )
            all_option_stds.append(standard_deviation)
            all_option_ranges.append(value_range)

    conditioned = {}
    for position in range(max(len(record["probabilities"]) for record in records)):
        members = [record for record in records if record["gold_permuted_index"] == position]
        if members:
            conditioned[chr(ord("A") + position)] = {
                "count": len(members),
                "accuracy": mean(record["correct"] for record in members),
            }
    masses = [record["candidate_mass"] for record in records]
    return {
        "sample_count": len(by_sample),
        "ordering_count": len(records),
        "accuracy": {
            "original": mean(record["correct"] for record in originals),
            "non_original_permutations": mean(record["correct"] for record in non_originals),
            "all_orderings": mean(record["correct"] for record in records),
        },
        "semantic_consistency": semantic_matches / len(non_originals),
        "gold_consistency": {
            "mean_fraction_correct": mean(gold_fractions),
            "all_4_correct_ratio": mean(value == 4 for value in correct_counts),
            "at_least_3_of_4_correct_ratio": mean(value >= 3 for value in correct_counts),
            "never_correct_ratio": mean(value == 0 for value in correct_counts),
        },
        "position_preference": {
            "prediction": _position_distribution(records, "prediction_permuted_index"),
            "gold": _position_distribution(records, "gold_permuted_index"),
        },
        "label_conditioned_accuracy": conditioned,
        "probability_stability": {
            "gold_option": {
                "mean_probability": mean(value[0] for value in gold_stability),
                "mean_probability_std": mean(value[1] for value in gold_stability),
                "median_probability_std": median(value[1] for value in gold_stability),
                "mean_probability_range": mean(value[2] for value in gold_stability),
                "median_probability_range": median(value[2] for value in gold_stability),
            },
            "original_predicted_option": {
                "mean_probability": mean(value[0] for value in predicted_stability),
                "mean_probability_std": mean(value[1] for value in predicted_stability),
                "median_probability_std": median(value[1] for value in predicted_stability),
                "mean_probability_range": mean(value[2] for value in predicted_stability),
                "median_probability_range": median(value[2] for value in predicted_stability),
            },
            "all_semantic_options": {
                "mean_probability_std": mean(all_option_stds),
                "median_probability_std": median(all_option_stds),
                "mean_probability_range": mean(all_option_ranges),
                "median_probability_range": median(all_option_ranges),
            },
        },
        "candidate_mass": {
            "mean": mean(masses),
            "median": median(masses),
            "mean_within_sample_std": mean(candidate_stds),
            "median_within_sample_std": median(candidate_stds),
            "mean_within_sample_range": mean(candidate_ranges),
        },
    }


def build_permutation_summary(
    model: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)
    datasets = {}
    for dataset, dataset_records in by_dataset.items():
        metrics = _group_metrics(dataset_records)
        if dataset == "mmlu":
            metrics["subjects"] = {
                subject: _group_metrics(
                    [record for record in dataset_records if record["subject"] == subject]
                )
                for subject in sorted({record["subject"] for record in dataset_records})
            }
        datasets[dataset] = metrics
    return {"model": model, "milestone": "4", "datasets": datasets}
