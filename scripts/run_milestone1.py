import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (  # noqa: E402
    run_generation,
    run_label_logits,
    run_option_likelihood,
)
from src.models import load_model  # noqa: E402
from src.prompting import option_label  # noqa: E402
from src.schema import MultipleChoiceSample  # noqa: E402


SAMPLE = MultipleChoiceSample(
    id="manual-geography-001",
    dataset="manual",
    question="What is the capital of France?",
    options=["Berlin", "Madrid", "Paris", "Rome"],
    gold_index=2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Milestone 1 smoke experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "qwen3_0.6b.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "milestone1.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    loaded = load_model(config)
    results = [
        run_generation(
            loaded,
            SAMPLE,
            max_new_tokens=int(config.get("generation", {}).get("max_new_tokens", 8)),
        ),
        run_label_logits(loaded, SAMPLE),
        run_option_likelihood(loaded, SAMPLE),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for result in results:
            output_file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    for result in results:
        prediction = (
            option_label(result.prediction) if result.prediction is not None else "UNPARSED"
        )
        status = "correct" if result.correct else "wrong"
        print(
            f"{result.mode:18} prediction={prediction} "
            f"confidence={result.confidence!s:>8} latency={result.latency_ms:.2f} ms "
            f"[{status}]"
        )
        if result.mode == "generation":
            print(f"  generated: {result.metadata['generated_text']!r}")
        if result.mode == "label_logit" and not result.metadata["supported"]:
            print(f"  unsupported: {result.metadata['unsupported_reason']}")

    print(f"Raw results: {args.output.resolve()}")


if __name__ == "__main__":
    main()

