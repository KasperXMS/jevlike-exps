import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import (  # noqa: E402
    load_arc_challenge_by_ids,
    load_commonsenseqa_by_ids,
    load_mmlu_by_ids,
)
from src.inference import run_label_logits  # noqa: E402
from src.metrics import build_permutation_summary  # noqa: E402
from src.models import load_model  # noqa: E402
from src.permutation import build_permutations, map_prediction_to_original  # noqa: E402
from src.schema import MultipleChoiceSample  # noqa: E402


DATASET_ALIASES = {
    "arc": "arc_challenge",
    "arc_challenge": "arc_challenge",
    "mmlu": "mmlu",
    "commonsenseqa": "commonsenseqa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Milestone 4 option permutations")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "qwen3_0.6b.yaml"
    )
    parser.add_argument("--datasets", default="arc,mmlu,commonsenseqa")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "milestone4_permutation_results.jsonl",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_datasets(value: str) -> list[str]:
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in DATASET_ALIASES]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return list(dict.fromkeys(DATASET_ALIASES[item] for item in requested))


def load_frozen_samples(selected: list[str]) -> dict[str, list[MultipleChoiceSample]]:
    results = PROJECT_ROOT / "results"
    output = {}
    if "arc_challenge" in selected:
        frozen = read_json(results / "arc_challenge_sample_ids.json")
        output["arc_challenge"] = load_arc_challenge_by_ids(frozen["sample_ids"], split="test")
    if "mmlu" in selected:
        frozen = read_json(results / "mmlu_sample_ids.json")
        output["mmlu"] = load_mmlu_by_ids(
            frozen["subjects"], frozen["sample_ids"], split=frozen.get("split", "test")
        )
    if "commonsenseqa" in selected:
        frozen = read_json(results / "commonsenseqa_sample_ids.json")
        output["commonsenseqa"] = load_commonsenseqa_by_ids(
            frozen["sample_ids"], split=frozen.get("split", "validation")
        )
    return output


def _manifest_entry(case, dataset: str, subject: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "subject": subject or None,
        "sample_id": case.sample.id,
        "permutation_id": case.permutation_id,
        "original_to_permuted": case.original_to_permuted,
        "permuted_to_original": case.permuted_to_original,
        "gold_original_index": case.gold_original_index,
        "gold_permuted_index": case.gold_permuted_index,
    }


def _read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    keys = [(r["dataset"], r["sample_id"], r["permutation_id"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Existing permutation results contain duplicate keys")
    return records


def main() -> None:
    args = parse_args()
    selected = parse_datasets(args.datasets)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    samples_by_dataset = load_frozen_samples(selected)
    cases = []
    manifest_entries = []
    for dataset in selected:
        for sample in samples_by_dataset[dataset]:
            for case in build_permutations(sample, count=3, seed=seed):
                cases.append((dataset, case))
                manifest_entries.append(_manifest_entry(case, dataset, sample.subject))

    manifest_path = PROJECT_ROOT / "results" / "milestone4_permutations.json"
    manifest = {
        "seed": seed,
        "permutations_per_sample": 3,
        "includes_original_ordering": True,
        "entries": manifest_entries,
    }
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Existing Milestone 4 permutation manifest does not match")
    write_json(manifest_path, manifest)

    records = _read_existing(args.output)
    completed = {
        (record["dataset"], record["sample_id"], record["permutation_id"])
        for record in records
    }
    expected = {(dataset, case.sample.id, case.permutation_id) for dataset, case in cases}
    unexpected = completed - expected
    if unexpected:
        raise ValueError(f"Output has records outside selected frozen experiment: {unexpected}")

    remaining = [
        (dataset, case)
        for dataset, case in cases
        if (dataset, case.sample.id, case.permutation_id) not in completed
    ]
    loaded = load_model(config) if remaining else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output_file:
        for completed_index, (dataset, case) in enumerate(remaining, start=1):
            result = run_label_logits(loaded, case.sample)
            if result.prediction is None or result.probabilities is None:
                raise RuntimeError(f"Unsupported label tokens for {case.sample.id}")
            prediction_original = map_prediction_to_original(
                result.prediction, case.permuted_to_original
            )
            probabilities_original = [0.0] * len(result.probabilities)
            for permuted_index, original_index in enumerate(case.permuted_to_original):
                probabilities_original[original_index] = result.probabilities[permuted_index]
            record = {
                "dataset": dataset,
                "subject": case.sample.subject or None,
                "sample_id": case.sample.id,
                "method": "label_logit",
                "permutation_id": case.permutation_id,
                "original_to_permuted": case.original_to_permuted,
                "permuted_to_original": case.permuted_to_original,
                "prediction_permuted_index": result.prediction,
                "prediction_original_index": prediction_original,
                "gold_permuted_index": case.gold_permuted_index,
                "gold_original_index": case.gold_original_index,
                "prediction": result.prediction,
                "gold_index": case.gold_permuted_index,
                "correct": prediction_original == case.gold_original_index,
                "raw_scores": result.raw_scores,
                "probabilities": result.probabilities,
                "probabilities_original_order": probabilities_original,
                "confidence": result.confidence,
                "candidate_mass": result.metadata["candidate_mass"],
                "metadata": {
                    **result.metadata,
                    "milestone": "4",
                    "seed": seed,
                },
            }
            records.append(record)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            if completed_index % 20 == 0 or completed_index == len(remaining):
                print(f"label-logit permutations: {completed_index}/{len(remaining)} remaining", flush=True)

    if len(records) != len(expected):
        raise RuntimeError(f"Expected {len(expected)} records, found {len(records)}")
    summary = build_permutation_summary(config["model_name"], records)
    summary.update({"seed": seed, "manifest": str(manifest_path.resolve())})
    write_json(PROJECT_ROOT / "results" / "milestone4_permutation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
