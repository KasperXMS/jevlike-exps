import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import (  # noqa: E402
    calibration_metrics,
    candidate_mass_analysis,
    create_calibration_split,
    fit_temperature,
    softmax_temperature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Milestone 4 calibration analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "milestone4_permutation_results.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ece-bins", type=int, default=15)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_original_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                record = json.loads(line)
                if record["permutation_id"] == 0:
                    records.append(record)
    keys = [(record["dataset"], record["sample_id"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Original-order calibration records contain duplicate sample IDs")
    return records


def _validate_split(split: dict[str, Any], records: list[dict[str, Any]]) -> None:
    expected: dict[str, set[str]] = {}
    for record in records:
        expected.setdefault(record["dataset"], set()).add(record["sample_id"])
    if set(split["datasets"]) != set(expected):
        raise ValueError("Calibration split datasets do not match input records")
    for dataset, ids in expected.items():
        entry = split["datasets"][dataset]
        calibration = set(entry["calibration_ids"])
        test = set(entry["test_ids"])
        if calibration & test:
            raise ValueError(f"Calibration/test overlap for {dataset}")
        if calibration | test != ids:
            raise ValueError(f"Calibration split IDs do not match {dataset}")


def _probabilities(records: list[dict[str, Any]], temperature: float) -> list[list[float]]:
    return [softmax_temperature(record["raw_scores"], temperature) for record in records]


def _metrics(records: list[dict[str, Any]], temperature: float, bins: int) -> dict[str, Any]:
    probabilities = _probabilities(records, temperature)
    return calibration_metrics(
        probabilities, [record["gold_original_index"] for record in records], bins=bins
    )


def _assert_argmax_invariant(records: list[dict[str, Any]], temperature: float) -> None:
    for record, probabilities in zip(records, _probabilities(records, temperature)):
        prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
        if prediction != record["prediction_original_index"]:
            raise AssertionError("Temperature scaling changed an argmax prediction")


def _plot_reliability(metrics: dict[str, Any], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    populated = [row for row in metrics["ece_bins"] if row["count"]]
    x = [row["mean_confidence"] for row in populated]
    y = [row["empirical_accuracy"] for row in populated]
    sizes = [20 + 2 * row["count"] for row in populated]
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="perfect calibration")
    axis.plot(x, y, color="#1f77b4", linewidth=1)
    axis.scatter(x, y, s=sizes, color="#1f77b4", alpha=0.8)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Empirical accuracy", title=title)
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _analysis_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "correct": record["correct"],
            "confidence": record["confidence"],
            "candidate_mass": record["candidate_mass"],
        }
        for record in records
    ]


def main() -> None:
    args = parse_args()
    records = read_original_records(args.input)
    if not records:
        raise ValueError("No original-order records found; run the permutation script first")

    split_path = PROJECT_ROOT / "results" / "milestone4_calibration_split.json"
    if split_path.exists():
        split = read_json(split_path)
    else:
        split = create_calibration_split(records, seed=args.seed, fraction=0.2)
        write_json(split_path, split)
    _validate_split(split, records)

    calibration_by_dataset = {}
    test_by_dataset = {}
    for dataset, entry in split["datasets"].items():
        calibration_ids = set(entry["calibration_ids"])
        test_ids = set(entry["test_ids"])
        calibration_by_dataset[dataset] = [
            record for record in records if record["dataset"] == dataset and record["sample_id"] in calibration_ids
        ]
        test_by_dataset[dataset] = [
            record for record in records if record["dataset"] == dataset and record["sample_id"] in test_ids
        ]

    temperatures = {}
    for dataset, calibration_records in calibration_by_dataset.items():
        temperature = fit_temperature(
            [record["raw_scores"] for record in calibration_records],
            [record["gold_original_index"] for record in calibration_records],
        )
        temperatures[dataset] = {
            "temperature": temperature,
            "calibration_count": len(calibration_records),
        }
    pooled_calibration = [record for values in calibration_by_dataset.values() for record in values]
    pooled_temperature = fit_temperature(
        [record["raw_scores"] for record in pooled_calibration],
        [record["gold_original_index"] for record in pooled_calibration],
    )
    temperatures["pooled"] = {
        "temperature": pooled_temperature,
        "calibration_count": len(pooled_calibration),
    }
    write_json(PROJECT_ROOT / "results" / "milestone4_temperatures.json", temperatures)

    summary = {"milestone": "4", "seed": args.seed, "ece_bin_count": args.ece_bins, "datasets": {}}
    figures = PROJECT_ROOT / "results" / "figures"
    aliases = {"arc_challenge": "arc", "mmlu": "mmlu", "commonsenseqa": "commonsenseqa"}
    mass_analysis = {"datasets": {}}
    for dataset, test_records in test_by_dataset.items():
        dataset_temperature = temperatures[dataset]["temperature"]
        raw = _metrics(test_records, 1.0, args.ece_bins)
        calibrated = _metrics(test_records, dataset_temperature, args.ece_bins)
        pooled = _metrics(test_records, pooled_temperature, args.ece_bins)
        _assert_argmax_invariant(test_records, dataset_temperature)
        _assert_argmax_invariant(test_records, pooled_temperature)
        if not (raw["accuracy"] == calibrated["accuracy"] == pooled["accuracy"]):
            raise AssertionError("Temperature scaling changed accuracy")
        summary["datasets"][dataset] = {
            "calibration_count": len(calibration_by_dataset[dataset]),
            "test_count": len(test_records),
            "dataset_temperature": dataset_temperature,
            "raw": raw,
            "dataset_temperature_scaled": calibrated,
            "pooled_temperature_scaled": pooled,
        }
        alias = aliases[dataset]
        _plot_reliability(raw, figures / f"{alias}_reliability_raw.png", f"{dataset}: raw")
        _plot_reliability(calibrated, figures / f"{alias}_reliability_calibrated.png", f"{dataset}: temperature scaled")
        mass_analysis["datasets"][dataset] = candidate_mass_analysis(_analysis_records(test_records))

    pooled_test = [record for values in test_by_dataset.values() for record in values]
    pooled_raw = _metrics(pooled_test, 1.0, args.ece_bins)
    pooled_scaled = _metrics(pooled_test, pooled_temperature, args.ece_bins)
    summary["pooled"] = {
        "calibration_count": len(pooled_calibration),
        "test_count": len(pooled_test),
        "temperature": pooled_temperature,
        "raw": pooled_raw,
        "pooled_temperature_scaled": pooled_scaled,
    }
    _plot_reliability(pooled_raw, figures / "pooled_reliability_raw.png", "Pooled: raw")
    _plot_reliability(pooled_scaled, figures / "pooled_reliability_calibrated.png", "Pooled: temperature scaled")
    mass_analysis["pooled"] = candidate_mass_analysis(_analysis_records(pooled_test))

    write_json(PROJECT_ROOT / "results" / "milestone4_calibration_summary.json", summary)
    write_json(PROJECT_ROOT / "results" / "milestone4_candidate_mass_analysis.json", mass_analysis)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
