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
    load_commonsenseqa,
    load_commonsenseqa_by_ids,
    load_mmlu,
    load_mmlu_by_ids,
    resolve_mmlu_subject,
)
from src.inference import (  # noqa: E402
    run_exact_text_option_likelihood,
    run_generation,
    run_label_logits,
    run_option_likelihood,
)
from src.metrics import (  # noqa: E402
    build_candidate_mass_analysis,
    build_milestone3_summary,
    build_paired_analysis,
)
from src.models import load_model  # noqa: E402
from src.schema import InferenceResult, MultipleChoiceSample  # noqa: E402


DATASET_ALIASES = {
    "arc": "arc_challenge",
    "arc_challenge": "arc_challenge",
    "mmlu": "mmlu",
    "commonsenseqa": "commonsenseqa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Milestone 3 cross-dataset eval")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "qwen3_0.6b.yaml",
    )
    parser.add_argument("--datasets", default="arc,mmlu,commonsenseqa")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "milestone3_results.jsonl",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)


def parse_datasets(value: str) -> list[str]:
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in DATASET_ALIASES]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    resolved = [DATASET_ALIASES[item] for item in requested]
    return list(dict.fromkeys(resolved))


def load_arc_frozen() -> list[MultipleChoiceSample]:
    frozen = read_json(PROJECT_ROOT / "results" / "arc_challenge_sample_ids.json")
    if frozen.get("seed") != 42 or frozen.get("split") != "test":
        raise ValueError("Existing ARC frozen split is not seed 42 / test")
    return load_arc_challenge_by_ids(frozen["sample_ids"], split="test")


def load_or_create_mmlu_frozen(
    subjects: list[str], limit: int, seed: int
) -> list[MultipleChoiceSample]:
    path = PROJECT_ROOT / "results" / "mmlu_sample_ids.json"
    if path.exists():
        frozen = read_json(path)
        if frozen.get("seed") != seed or frozen.get("subjects") != subjects:
            raise ValueError("Existing MMLU frozen split metadata does not match config")
        return load_mmlu_by_ids(subjects, frozen["sample_ids"], split="test")

    samples = load_mmlu(subjects, split="test", limit=limit, seed=seed)
    write_json(
        path,
        {
            "dataset": "mmlu",
            "seed": seed,
            "split": "test",
            "subjects": subjects,
            "subject_configs": {
                subject: resolve_mmlu_subject(subject) for subject in subjects
            },
            "limit_per_subject": limit,
            "id_scheme": "logical_subject:source_index:content_sha256_20",
            "sample_ids": [sample.id for sample in samples],
        },
    )
    return samples


def load_or_create_commonsenseqa_frozen(
    limit: int, seed: int
) -> list[MultipleChoiceSample]:
    path = PROJECT_ROOT / "results" / "commonsenseqa_sample_ids.json"
    if path.exists():
        frozen = read_json(path)
        if frozen.get("seed") != seed or frozen.get("split") != "validation":
            raise ValueError(
                "Existing CommonsenseQA frozen split is not seed 42 / validation"
            )
        return load_commonsenseqa_by_ids(frozen["sample_ids"], split="validation")

    samples = load_commonsenseqa(split="validation", limit=limit, seed=seed)
    write_json(
        path,
        {
            "dataset": "commonsenseqa",
            "seed": seed,
            "split": "validation",
            "subjects": [],
            "sample_ids": [sample.id for sample in samples],
        },
    )
    return samples


def result_record(
    result: InferenceResult, sample: MultipleChoiceSample
) -> dict[str, Any]:
    metadata = dict(result.metadata)
    if result.mode == "generation":
        metadata["parser_rule"] = metadata["parser_pattern"]
    if result.mode.startswith("option_likelihood"):
        metadata["option_scores"] = {
            name: aggregation["scores"]
            for name, aggregation in metadata["aggregations"].items()
        }
    return {
        "dataset": sample.dataset,
        "subject": sample.subject or None,
        "sample_id": sample.id,
        "method": result.mode,
        "prediction": result.prediction,
        "gold_index": result.gold_index,
        "correct": result.correct,
        "probabilities": result.probabilities,
        "confidence": result.confidence,
        "raw_scores": result.raw_scores,
        "prompt": result.prompt,
        "options": result.options,
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    selected = parse_datasets(args.datasets)
    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    seed = int(config.get("seed", 42))
    milestone_config = config["milestone3"]
    subjects = list(milestone_config["mmlu_subjects"])

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    samples_by_dataset: dict[str, list[MultipleChoiceSample]] = {}
    if "arc_challenge" in selected:
        samples_by_dataset["arc_challenge"] = load_arc_frozen()
    if "mmlu" in selected:
        samples_by_dataset["mmlu"] = load_or_create_mmlu_frozen(
            subjects,
            int(milestone_config["mmlu_limit_per_subject"]),
            seed,
        )
    if "commonsenseqa" in selected:
        samples_by_dataset["commonsenseqa"] = load_or_create_commonsenseqa_frozen(
            int(milestone_config["commonsenseqa_limit"]), seed
        )

    loaded = load_model(config)
    max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", 8))
    methods = (
        ("generation", lambda sample: run_generation(loaded, sample, max_new_tokens)),
        ("label_logit", lambda sample: run_label_logits(loaded, sample)),
        ("option_likelihood_legacy", lambda sample: run_option_likelihood(loaded, sample)),
        (
            "option_likelihood_exact_text",
            lambda sample: run_exact_text_option_likelihood(loaded, sample),
        ),
    )

    records: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for dataset_name in selected:
            samples = samples_by_dataset[dataset_name]
            for sample_index, sample in enumerate(samples, start=1):
                for _, method in methods:
                    result = method(sample)
                    result.metadata.update(
                        {
                            "milestone": "3",
                            "split": (
                                "validation"
                                if dataset_name == "commonsenseqa"
                                else "test"
                            ),
                            "seed": seed,
                            "sample_index": sample_index - 1,
                            "batch_size": 1,
                        }
                    )
                    record = result_record(result, sample)
                    records.append(record)
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_file.flush()
                if sample_index % 10 == 0 or sample_index == len(samples):
                    print(
                        f"{dataset_name}: {sample_index}/{len(samples)} samples",
                        flush=True,
                    )

    summary = build_milestone3_summary(loaded.name, records)
    summary.update(
        {
            "milestone": "3",
            "seed": seed,
            "frozen_sample_id_files": {
                dataset_name: str(
                    (
                        PROJECT_ROOT
                        / "results"
                        / (
                            "arc_challenge_sample_ids.json"
                            if dataset_name == "arc_challenge"
                            else f"{dataset_name}_sample_ids.json"
                        )
                    ).resolve()
                )
                for dataset_name in selected
            },
        }
    )
    paired = build_paired_analysis(records)
    candidate_mass = build_candidate_mass_analysis(records)
    write_json(PROJECT_ROOT / "results" / "milestone3_summary.json", summary)
    write_json(PROJECT_ROOT / "results" / "milestone3_paired.json", paired)
    write_json(
        PROJECT_ROOT / "results" / "milestone3_candidate_mass.json",
        candidate_mass,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
