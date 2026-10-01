import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import load_arc_challenge_by_ids  # noqa: E402
from src.inference import (  # noqa: E402
    run_exact_text_option_likelihood,
    run_generation,
    run_label_logits,
    run_option_likelihood,
)
from src.models import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARC Milestone 2.5")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "qwen3_0.6b.yaml",
    )
    parser.add_argument(
        "--frozen-ids",
        type=Path,
        default=PROJECT_ROOT / "results" / "arc_challenge_sample_ids.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "arc_challenge_milestone25.jsonl",
    )
    return parser.parse_args()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)


def accuracy(results) -> float:
    return sum(result.correct for result in results) / len(results)


def aggregation_accuracy(results, aggregation: str) -> float:
    return sum(
        result.metadata["aggregations"][aggregation]["correct"]
        for result in results
    ) / len(results)


def main() -> None:
    args = parse_args()
    frozen_bytes = args.frozen_ids.read_bytes()
    frozen = json.loads(frozen_bytes.decode("utf-8"))
    sample_ids = frozen["sample_ids"]
    if len(sample_ids) != 100 or len(set(sample_ids)) != 100:
        raise ValueError("Milestone 2.5 requires exactly 100 unique frozen sample IDs")

    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    seed = int(config.get("seed", 42))
    if frozen.get("seed") != seed or frozen.get("split") != "test":
        raise ValueError("Frozen sample metadata does not match seed 42 / test split")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    samples = load_arc_challenge_by_ids(sample_ids, split="test")
    if [sample.id for sample in samples] != sample_ids:
        raise RuntimeError("Loaded ARC samples do not preserve frozen ID order")

    loaded = load_model(config)
    methods = (
        (
            "generation",
            lambda sample: run_generation(
                loaded,
                sample,
                max_new_tokens=int(
                    config.get("generation", {}).get("max_new_tokens", 8)
                ),
            ),
        ),
        ("label_logit", lambda sample: run_label_logits(loaded, sample)),
        ("option_likelihood_legacy", lambda sample: run_option_likelihood(loaded, sample)),
        (
            "option_likelihood_exact_text",
            lambda sample: run_exact_text_option_likelihood(loaded, sample),
        ),
    )
    results_by_method = {method_name: [] for method_name, _ in methods}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for sample_index, sample in enumerate(samples, start=1):
            for method_name, method in methods:
                result = method(sample)
                result.metadata.update(
                    {
                        "milestone": "2.5",
                        "split": "test",
                        "seed": seed,
                        "sample_index": sample_index - 1,
                        "batch_size": 1,
                    }
                )
                results_by_method[method_name].append(result)
                output_file.write(
                    json.dumps(result.to_dict(), ensure_ascii=False) + "\n"
                )
                output_file.flush()
            if sample_index % 10 == 0:
                print(f"Completed {sample_index}/{len(samples)} samples", flush=True)

    generation = results_by_method["generation"]
    label_logit = results_by_method["label_logit"]
    legacy = results_by_method["option_likelihood_legacy"]
    exact_text = results_by_method["option_likelihood_exact_text"]
    parsed_generation = [result for result in generation if result.prediction is not None]
    parsed_correct = sum(result.correct for result in parsed_generation)

    paired = {
        "G+L+": [],
        "G+L-": [],
        "G-L+": [],
        "G-L-": [],
    }
    for generation_result, label_result in zip(generation, label_logit, strict=True):
        generation_mark = "+" if generation_result.correct else "-"
        label_mark = "+" if label_result.correct else "-"
        paired[f"G{generation_mark}L{label_mark}"].append(generation_result.sample_id)

    paired_output = {
        group: {"count": len(ids), "sample_ids": ids}
        for group, ids in paired.items()
    }
    paired_path = args.output.with_name(args.output.stem + "_paired.json")
    write_json(paired_path, paired_output)

    summary = {
        "milestone": "2.5",
        "model": loaded.name,
        "dataset": "arc_challenge",
        "split": "test",
        "seed": seed,
        "sample_count": len(samples),
        "frozen_sample_ids": {
            "source": str(args.frozen_ids.resolve()),
            "sha256": hashlib.sha256(frozen_bytes).hexdigest(),
            "unchanged": True,
        },
        "generation": {
            "end_to_end_accuracy": accuracy(generation),
            "parse_rate": len(parsed_generation) / len(generation),
            "parsed_count": len(parsed_generation),
            "conditional_accuracy": (
                parsed_correct / len(parsed_generation) if parsed_generation else None
            ),
            "parsed_correct": parsed_correct,
        },
        "label_logit": {
            "accuracy": accuracy(label_logit),
            "mean_candidate_mass": sum(
                result.metadata["candidate_mass"] for result in label_logit
            )
            / len(label_logit),
        },
        "option_likelihood": {
            "legacy_mean_accuracy": aggregation_accuracy(legacy, "mean"),
            "legacy_summed_accuracy": aggregation_accuracy(legacy, "summed"),
            "exact_text_mean_accuracy": aggregation_accuracy(exact_text, "mean"),
            "exact_text_summed_accuracy": aggregation_accuracy(exact_text, "summed"),
        },
        "paired_generation_vs_label_logit": {
            group: value["count"] for group, value in paired_output.items()
        },
        "paired_sample_ids_file": str(paired_path.resolve()),
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    write_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Raw results: {args.output.resolve()}")
    print(f"Paired analysis: {paired_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
