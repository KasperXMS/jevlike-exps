import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import load_arc_challenge  # noqa: E402
from src.inference import (  # noqa: E402
    run_generation,
    run_label_logits,
    run_option_likelihood,
)
from src.models import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ARC-Challenge")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "qwen3_0.6b.yaml",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "arc_challenge.jsonl",
    )
    return parser.parse_args()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    samples = load_arc_challenge(split=args.split, limit=args.limit, seed=seed)
    ids_path = args.output.with_name(args.output.stem + "_sample_ids.json")
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    _write_json(
        ids_path,
        {
            "dataset": "arc_challenge",
            "split": args.split,
            "seed": seed,
            "requested_limit": args.limit,
            "sample_ids": [sample.id for sample in samples],
        },
    )

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
        ("option_likelihood", lambda sample: run_option_likelihood(loaded, sample)),
    )
    aggregates = defaultdict(lambda: {"correct": 0, "parsed": 0, "latencies": []})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for sample_index, sample in enumerate(samples, start=1):
            for method_name, method in methods:
                result = method(sample)
                result.metadata.update(
                    {
                        "split": args.split,
                        "seed": seed,
                        "sample_index": sample_index - 1,
                        "batch_size": 1,
                    }
                )
                output_file.write(
                    json.dumps(result.to_dict(), ensure_ascii=False) + "\n"
                )
                output_file.flush()
                stats = aggregates[method_name]
                stats["correct"] += int(result.correct)
                stats["parsed"] += int(result.prediction is not None)
                stats["latencies"].append(result.latency_ms)
            if sample_index % 10 == 0 or sample_index == len(samples):
                print(f"Completed {sample_index}/{len(samples)} samples", flush=True)

    summary = {
        "model": loaded.name,
        "dataset": "arc_challenge",
        "split": args.split,
        "seed": seed,
        "sample_count": len(samples),
        "methods": {},
    }
    for method_name, stats in aggregates.items():
        count = len(samples)
        summary["methods"][method_name] = {
            "accuracy": stats["correct"] / count,
            "correct": stats["correct"],
            "parsed_or_supported": stats["parsed"],
            "mean_latency_ms": sum(stats["latencies"]) / count,
        }
    _write_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Raw results: {args.output.resolve()}")
    print(f"Sample IDs: {ids_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
