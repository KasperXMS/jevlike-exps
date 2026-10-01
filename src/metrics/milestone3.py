from collections import defaultdict
from statistics import mean, median
from typing import Any


def _accuracy(records: list[dict[str, Any]]) -> float:
    return sum(bool(record["correct"]) for record in records) / len(records)


def _aggregation_accuracy(
    records: list[dict[str, Any]], aggregation: str
) -> float:
    return sum(
        bool(record["metadata"]["aggregations"][aggregation]["correct"])
        for record in records
    ) / len(records)


def _dataset_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_method[record["method"]].append(record)

    generation = by_method["generation"]
    label_logit = by_method["label_logit"]
    legacy = by_method["option_likelihood_legacy"]
    exact = by_method["option_likelihood_exact_text"]
    parsed = [record for record in generation if record["prediction"] is not None]

    return {
        "sample_count": len(generation),
        "generation": {
            "accuracy": _accuracy(generation),
            "parse_rate": len(parsed) / len(generation),
            "conditional_accuracy": _accuracy(parsed) if parsed else None,
        },
        "label_logit": {
            "accuracy": _accuracy(label_logit),
            "mean_candidate_mass": mean(
                record["metadata"]["candidate_mass"] for record in label_logit
            ),
            "mean_confidence": mean(
                record["confidence"] for record in label_logit
            ),
        },
        "option_likelihood": {
            "accuracy_mean": _aggregation_accuracy(legacy, "mean"),
            "accuracy_sum": _aggregation_accuracy(legacy, "summed"),
            "exact_mean": _aggregation_accuracy(exact, "mean"),
            "exact_sum": _aggregation_accuracy(exact, "summed"),
        },
    }


def build_milestone3_summary(
    model: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)

    datasets: dict[str, Any] = {}
    for dataset_name, dataset_records in by_dataset.items():
        metrics = _dataset_metrics(dataset_records)
        subjects = sorted(
            {record["subject"] for record in dataset_records if record["subject"]}
        )
        if subjects:
            metrics["subjects"] = {
                subject: _dataset_metrics(
                    [
                        record
                        for record in dataset_records
                        if record["subject"] == subject
                    ]
                )
                for subject in subjects
            }
        datasets[dataset_name] = metrics
    return {"model": model, "datasets": datasets}


def build_paired_analysis(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)

    output: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_name, dataset_records in by_dataset.items():
        generation = {
            record["sample_id"]: record
            for record in dataset_records
            if record["method"] == "generation"
        }
        label_logit = {
            record["sample_id"]: record
            for record in dataset_records
            if record["method"] == "label_logit"
        }
        if generation.keys() != label_logit.keys():
            raise ValueError(f"Paired sample IDs differ for {dataset_name}")
        groups = {"G+L+": [], "G+L-": [], "G-L+": [], "G-L-": []}
        for sample_id, generation_record in generation.items():
            label_record = label_logit[sample_id]
            generation_mark = "+" if generation_record["correct"] else "-"
            label_mark = "+" if label_record["correct"] else "-"
            groups[f"G{generation_mark}L{label_mark}"].append(sample_id)
        output[dataset_name] = {
            group: {"count": len(sample_ids), "sample_ids": sample_ids}
            for group, sample_ids in groups.items()
        }
    return output


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_candidate_mass_analysis(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    masses: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record["method"] == "label_logit":
            masses[record["dataset"]].append(record["metadata"]["candidate_mass"])
    return {
        dataset_name: {
            "mean": mean(values),
            "median": median(values),
            "p10": _percentile(values, 0.10),
            "p90": _percentile(values, 0.90),
        }
        for dataset_name, values in masses.items()
    }
