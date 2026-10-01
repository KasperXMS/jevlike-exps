import argparse
import gc
import json
import random
import re
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
from src.inference import run_generation, run_label_logits  # noqa: E402
from src.metrics import (  # noqa: E402
    build_calibration_observation,
    build_invariance_summary,
    build_original_summary,
    build_paired,
)
from src.models import load_model  # noqa: E402
from src.schema import InferenceResult, MultipleChoiceSample  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "milestone5"
MODEL_SPECS = {
    "Qwen/Qwen3-0.6B-Base": {"slug": "qwen3_0_6b_base", "nominal_billions": 0.6},
    "Qwen/Qwen3-1.7B-Base": {"slug": "qwen3_1_7b_base", "nominal_billions": 1.7},
    "Qwen/Qwen3-4B-Base": {"slug": "qwen3_4b_base", "nominal_billions": 4.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Milestone 5 model scaling")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "qwen3_0.6b.yaml"
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated Hugging Face model IDs; defaults to milestone5 config",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_jsonl_once(path: Path, records: list[dict[str, Any]]) -> None:
    if path.exists():
        if read_jsonl(path) != records:
            raise ValueError(f"Existing derived baseline differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_frozen_samples() -> dict[str, list[MultipleChoiceSample]]:
    results = PROJECT_ROOT / "results"
    arc = read_json(results / "arc_challenge_sample_ids.json")
    mmlu = read_json(results / "mmlu_sample_ids.json")
    commonsenseqa = read_json(results / "commonsenseqa_sample_ids.json")
    return {
        "arc_challenge": load_arc_challenge_by_ids(arc["sample_ids"], split=arc.get("split", "test")),
        "mmlu": load_mmlu_by_ids(mmlu["subjects"], mmlu["sample_ids"], split=mmlu.get("split", "test")),
        "commonsenseqa": load_commonsenseqa_by_ids(
            commonsenseqa["sample_ids"], split=commonsenseqa.get("split", "validation")
        ),
    }


def prepare_baseline() -> None:
    spec = MODEL_SPECS["Qwen/Qwen3-0.6B-Base"]
    originals = [
        record
        for record in read_jsonl(PROJECT_ROOT / "results" / "milestone3_results.jsonl")
        if record["method"] in {"generation", "label_logit"}
    ]
    permutations = read_jsonl(PROJECT_ROOT / "results" / "milestone4_permutation_results.jsonl")
    if len(originals) != 1400 or len(permutations) != 2800:
        raise ValueError("M3/M4 baseline raw result counts are not 1400/2800")
    write_jsonl_once(OUTPUT_DIR / f"model_results_{spec['slug']}.jsonl", originals)
    write_jsonl_once(OUTPUT_DIR / f"permutation_results_{spec['slug']}.jsonl", permutations)
    metadata_path = OUTPUT_DIR / f"model_metadata_{spec['slug']}.json"
    metadata = {
        "model": "Qwen/Qwen3-0.6B-Base",
        "parameter_count": 600_000_000,
        "parameter_count_source": "nominal model name",
        "original_source": "results/milestone3_results.jsonl",
        "permutation_source": "results/milestone4_permutation_results.jsonl",
    }
    if metadata_path.exists() and read_json(metadata_path) != metadata:
        raise ValueError("Existing 0.6B metadata differs")
    write_json(metadata_path, metadata)


def result_record(result: InferenceResult, sample: MultipleChoiceSample) -> dict[str, Any]:
    metadata = dict(result.metadata)
    if result.mode == "generation":
        metadata["parser_rule"] = metadata.get("parser_pattern")
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
        "metadata": {**metadata, "milestone": "5"},
    }


def run_originals(loaded, samples_by_dataset, path: Path, max_new_tokens: int) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    completed = {(r["dataset"], r["sample_id"], r["method"]) for r in records}
    expected = {
        (dataset, sample.id, method)
        for dataset, samples in samples_by_dataset.items()
        for sample in samples
        for method in ("generation", "label_logit")
    }
    if completed - expected or len(completed) != len(records):
        raise ValueError(f"Invalid or duplicate original records in {path}")
    remaining = len(expected - completed)
    path.parent.mkdir(parents=True, exist_ok=True)
    progress = 0
    with path.open("a", encoding="utf-8") as output_file:
        for dataset, samples in samples_by_dataset.items():
            for sample in samples:
                methods = (
                    ("generation", lambda: run_generation(loaded, sample, max_new_tokens)),
                    ("label_logit", lambda: run_label_logits(loaded, sample)),
                )
                for method, runner in methods:
                    key = (dataset, sample.id, method)
                    if key in completed:
                        continue
                    result = runner()
                    if method == "label_logit" and result.prediction is None:
                        raise RuntimeError(f"Unsupported label token representation for {sample.id}")
                    record = result_record(result, sample)
                    records.append(record)
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_file.flush()
                    completed.add(key)
                    progress += 1
                    if progress % 20 == 0 or progress == remaining:
                        print(f"{loaded.name} originals: {progress}/{remaining} remaining work", flush=True)
    if completed != expected:
        raise RuntimeError("Original-order evaluation is incomplete")
    return records


def _permutation_record(
    result_record_value: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    prediction_permuted = result_record_value["prediction"]
    permuted_to_original = manifest["permuted_to_original"]
    prediction_original = permuted_to_original[prediction_permuted]
    probabilities = result_record_value["probabilities"]
    probabilities_original = [0.0] * len(probabilities)
    for permuted_index, original_index in enumerate(permuted_to_original):
        probabilities_original[original_index] = probabilities[permuted_index]
    candidate_mass = result_record_value["metadata"]["candidate_mass"]
    return {
        "dataset": manifest["dataset"],
        "subject": manifest.get("subject"),
        "sample_id": manifest["sample_id"],
        "method": "label_logit",
        "permutation_id": manifest["permutation_id"],
        "original_to_permuted": manifest["original_to_permuted"],
        "permuted_to_original": permuted_to_original,
        "prediction_permuted_index": prediction_permuted,
        "prediction_original_index": prediction_original,
        "gold_permuted_index": manifest["gold_permuted_index"],
        "gold_original_index": manifest["gold_original_index"],
        "prediction": prediction_permuted,
        "gold_index": manifest["gold_permuted_index"],
        "correct": prediction_original == manifest["gold_original_index"],
        "raw_scores": result_record_value["raw_scores"],
        "probabilities": probabilities,
        "probabilities_original_order": probabilities_original,
        "confidence": result_record_value["confidence"],
        "candidate_mass": candidate_mass,
        "metadata": {**result_record_value["metadata"], "milestone": "5"},
    }


def run_permutations(
    loaded,
    samples_by_dataset: dict[str, list[MultipleChoiceSample]],
    original_records: list[dict[str, Any]],
    path: Path,
) -> list[dict[str, Any]]:
    manifest_entries = read_json(PROJECT_ROOT / "results" / "milestone4_permutations.json")["entries"]
    sample_map = {
        (sample.dataset, sample.id): sample
        for samples in samples_by_dataset.values()
        for sample in samples
    }
    label_originals = {
        (record["dataset"], record["sample_id"]): record
        for record in original_records
        if record["method"] == "label_logit"
    }
    records = read_jsonl(path)
    completed = {(r["dataset"], r["sample_id"], r["permutation_id"]) for r in records}
    expected = {(m["dataset"], m["sample_id"], m["permutation_id"]) for m in manifest_entries}
    if completed - expected or len(completed) != len(records):
        raise ValueError(f"Invalid or duplicate permutation records in {path}")
    remaining = len(expected - completed)
    progress = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        for manifest in manifest_entries:
            key = (manifest["dataset"], manifest["sample_id"], manifest["permutation_id"])
            if key in completed:
                continue
            original = sample_map[(manifest["dataset"], manifest["sample_id"])]
            if manifest["permutation_id"] == 0:
                label_record = label_originals[(manifest["dataset"], manifest["sample_id"])]
            else:
                sample = MultipleChoiceSample(
                    id=original.id,
                    dataset=original.dataset,
                    question=original.question,
                    options=[original.options[index] for index in manifest["permuted_to_original"]],
                    gold_index=manifest["gold_permuted_index"],
                    context=original.context,
                    subject=original.subject,
                )
                label_record = result_record(run_label_logits(loaded, sample), sample)
                if label_record["prediction"] is None:
                    raise RuntimeError(f"Unsupported labels for {sample.id}")
            record = _permutation_record(label_record, manifest)
            records.append(record)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            completed.add(key)
            progress += 1
            if progress % 20 == 0 or progress == remaining:
                print(f"{loaded.name} permutations: {progress}/{remaining} remaining work", flush=True)
    if completed != expected:
        raise RuntimeError("Permutation evaluation is incomplete")
    return records


def _available_models() -> list[dict[str, Any]]:
    available = []
    for model, spec in MODEL_SPECS.items():
        metadata_path = OUTPUT_DIR / f"model_metadata_{spec['slug']}.json"
        original_path = OUTPUT_DIR / f"model_results_{spec['slug']}.jsonl"
        permutation_path = OUTPUT_DIR / f"permutation_results_{spec['slug']}.jsonl"
        if metadata_path.exists() and original_path.exists() and permutation_path.exists():
            metadata = read_json(metadata_path)
            originals = read_jsonl(original_path)
            permutations = read_jsonl(permutation_path)
            if len(originals) == 1400 and len(permutations) == 2800:
                available.append({"model": model, "spec": spec, "metadata": metadata, "originals": originals, "permutations": permutations})
    return available


def _plot_lines(models, values, ylabel, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    x = [item["metadata"]["parameter_count"] / 1e9 for item in models]
    for label, series in values.items():
        axis.plot(x, series, marker="o", label=label)
    axis.set(xlabel="Model parameters (billions)", ylabel=ylabel, title=title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def build_outputs() -> None:
    models = sorted(_available_models(), key=lambda item: item["metadata"]["parameter_count"])
    if not models:
        return
    split = read_json(PROJECT_ROOT / "results" / "milestone4_calibration_split.json")
    summaries = {}
    paired = {}
    invariance = {}
    calibration = {}
    for item in models:
        model = item["model"]
        parameters = item["metadata"]["parameter_count"]
        summaries[model] = build_original_summary(model, parameters, item["originals"])
        paired[model] = build_paired(item["originals"])
        invariance[model] = build_invariance_summary(model, parameters, item["permutations"])
        calibration[model] = build_calibration_observation(model, parameters, item["originals"], split)
    write_json(OUTPUT_DIR / "summary.json", {"milestone": "5", "models": summaries})
    write_json(OUTPUT_DIR / "paired.json", {"milestone": "5", "models": paired})
    write_json(OUTPUT_DIR / "invariance.json", {"milestone": "5", "models": invariance})
    write_json(OUTPUT_DIR / "calibration.json", {"milestone": "5", "models": calibration})

    datasets = ["arc_challenge", "mmlu", "commonsenseqa"]
    figures = OUTPUT_DIR / "figures"
    _plot_lines(
        models,
        {
            f"{dataset} label": [summaries[m["model"]]["datasets"][dataset]["label_logit"]["accuracy"] for m in models]
            for dataset in datasets
        },
        "Accuracy",
        "Label-logit accuracy scaling",
        figures / "model_size_vs_accuracy.png",
    )
    _plot_lines(
        models,
        {dataset: [invariance[m["model"]]["datasets"][dataset]["semantic_consistency"] for m in models] for dataset in datasets},
        "Semantic consistency",
        "Permutation semantic consistency",
        figures / "model_size_vs_semantic_consistency.png",
    )
    for key, filename, title in [
        ("correct_to_wrong", "model_size_vs_correct_to_wrong.png", "Correct-to-wrong permutation flips"),
        ("wrong_to_correct", "model_size_vs_wrong_to_correct.png", "Wrong-to-correct permutation flips"),
    ]:
        _plot_lines(
            models,
            {dataset: [invariance[m["model"]]["datasets"][dataset]["flip_rates"][key]["rate"] for m in models] for dataset in datasets},
            "Flip rate",
            title,
            figures / filename,
        )
    _plot_lines(
        models,
        {dataset: [invariance[m["model"]]["datasets"][dataset]["probability_stability"]["gold_option"]["mean_probability_range"] for m in models] for dataset in datasets},
        "Mean gold probability range",
        "Gold probability instability",
        figures / "model_size_vs_gold_probability_range.png",
    )
    _plot_lines(
        models,
        {dataset: [calibration[m["model"]]["datasets"][dataset]["ece"] for m in models] for dataset in datasets},
        "ECE (15 bins)",
        "Natural calibration scaling",
        figures / "model_size_vs_ece.png",
    )
    _plot_position_preference(models, invariance, figures / "position_preference_by_model.png")


def _plot_position_preference(models, invariance, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = ["arc_challenge", "mmlu", "commonsenseqa"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for axis, dataset in zip(axes, datasets):
        labels = list(invariance[models[0]["model"]]["datasets"][dataset]["position_preference"]["prediction"])
        x = np.arange(len(labels))
        width = 0.8 / len(models)
        for index, item in enumerate(models):
            distribution = invariance[item["model"]]["datasets"][dataset]["position_preference"]["prediction"]
            values = [distribution[label]["normalized_frequency"] for label in labels]
            axis.bar(x + (index - (len(models) - 1) / 2) * width, values, width, label=item["model"].split("/")[-1])
        gold = invariance[models[0]["model"]]["datasets"][dataset]["position_preference"]["gold"]
        axis.plot(x, [gold[label]["normalized_frequency"] for label in labels], color="black", marker="x", linestyle="--", label="gold")
        axis.set(title=dataset, xticks=x, xticklabels=labels, xlabel="Position")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Normalized frequency")
    axes[-1].legend(fontsize=8)
    figure.suptitle("Prediction position preference by model")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    requested = (
        [value.strip() for value in args.models.split(",") if value.strip()]
        if args.models
        else list(config["milestone5"]["models"])
    )
    unknown = [model for model in requested if model not in MODEL_SPECS or model.endswith("0.6B-Base")]
    if unknown:
        raise ValueError(f"Unsupported or baseline-only models: {unknown}")
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    prepare_baseline()
    samples_by_dataset = load_frozen_samples()
    max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", 8))
    for model_name in requested:
        spec = MODEL_SPECS[model_name]
        model_config = dict(config)
        model_config.update(
            {
                "model_name": model_name,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "dtype": "auto",
                "cache_dir": str(PROJECT_ROOT / ".cache" / "huggingface" / "hub"),
            }
        )
        loaded = load_model(model_config)
        parameter_count = sum(parameter.numel() for parameter in loaded.model.parameters())
        metadata = {
            "model": model_name,
            "parameter_count": parameter_count,
            "parameter_count_source": "loaded model parameters",
            "nominal_billions": spec["nominal_billions"],
            "device": str(loaded.device),
            "dtype": str(next(loaded.model.parameters()).dtype),
        }
        write_json(OUTPUT_DIR / f"model_metadata_{spec['slug']}.json", metadata)
        originals = run_originals(
            loaded,
            samples_by_dataset,
            OUTPUT_DIR / f"model_results_{spec['slug']}.jsonl",
            max_new_tokens,
        )
        run_permutations(
            loaded,
            samples_by_dataset,
            originals,
            OUTPUT_DIR / f"permutation_results_{spec['slug']}.jsonl",
        )
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        build_outputs()
    build_outputs()


if __name__ == "__main__":
    main()
