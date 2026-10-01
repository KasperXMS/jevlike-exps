import argparse
import gc
import json
import platform
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
from src.inference.label_logits import inspect_label_tokens  # noqa: E402
from src.models import load_model  # noqa: E402
from src.prompting import build_prompt, build_prompt_parts  # noqa: E402
from src.schema import MultipleChoiceSample  # noqa: E402
from src.shared_state import summarize_shared_state  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "milestone7"
MODELS = [
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Milestone 7 shared-state probe")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "qwen3_0.6b.yaml"
    )
    parser.add_argument("--sample-groups", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=10)
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


def load_samples() -> dict[tuple[str, str], MultipleChoiceSample]:
    results = PROJECT_ROOT / "results"
    arc = read_json(results / "arc_challenge_sample_ids.json")
    mmlu = read_json(results / "mmlu_sample_ids.json")
    csqa = read_json(results / "commonsenseqa_sample_ids.json")
    samples = [
        *load_arc_challenge_by_ids(arc["sample_ids"], split=arc.get("split", "test")),
        *load_mmlu_by_ids(mmlu["subjects"], mmlu["sample_ids"], split=mmlu.get("split", "test")),
        *load_commonsenseqa_by_ids(csqa["sample_ids"], split=csqa.get("split", "validation")),
    ]
    return {(sample.dataset, sample.id): sample for sample in samples}


def branch_samples(
    original: MultipleChoiceSample, entries: list[dict[str, Any]]
) -> list[MultipleChoiceSample]:
    return [
        MultipleChoiceSample(
            id=original.id,
            dataset=original.dataset,
            question=original.question,
            options=[original.options[index] for index in entry["permuted_to_original"]],
            gold_index=entry["gold_permuted_index"],
            context=original.context,
            subject=original.subject,
        )
        for entry in entries
    ]


def prepare_group(loaded, original, entries):
    branches = branch_samples(original, entries)
    prefix, _ = build_prompt_parts(branches[0])
    prefix_ids = loaded.tokenizer.encode(prefix, add_special_tokens=False)
    suffix_rows = []
    full_rows = []
    candidate_rows = []
    for branch in branches:
        branch_prefix, suffix = build_prompt_parts(branch)
        if branch_prefix != prefix:
            raise AssertionError("Permutation branches do not share an exact prefix")
        suffix_ids = loaded.tokenizer.encode(suffix, add_special_tokens=False)
        full_ids = loaded.tokenizer.encode(build_prompt(branch), add_special_tokens=False)
        if full_ids != prefix_ids + suffix_ids:
            raise ValueError("Tokenizer does not preserve the selected prefix boundary")
        inspection = inspect_label_tokens(loaded.tokenizer, build_prompt(branch), len(branch.options))
        if not inspection.supported or inspection.token_ids is None:
            raise RuntimeError(f"Unsupported labels for {branch.id}")
        suffix_rows.append(suffix_ids)
        full_rows.append(full_ids)
        candidate_rows.append(inspection.token_ids)
    if len({len(row) for row in suffix_rows}) != 1:
        raise ValueError("Permutation suffix token lengths differ")
    device = loaded.device
    return {
        "prefix_ids": torch.tensor([prefix_ids], dtype=torch.long, device=device),
        "suffix_ids": torch.tensor(suffix_rows, dtype=torch.long, device=device),
        "full_ids": torch.tensor(full_rows, dtype=torch.long, device=device),
        "candidate_ids": torch.tensor(candidate_rows, dtype=torch.long, device=device),
        "prefix_tokens": len(prefix_ids),
        "suffix_tokens": len(suffix_rows[0]),
        "full_prompt_tokens": len(full_rows[0]),
    }


def restricted(logits: torch.Tensor, candidate_ids: torch.Tensor):
    candidate_logits = logits.float().gather(1, candidate_ids)
    probabilities = candidate_logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    return probabilities, predictions


@torch.inference_mode()
def full_forward_trial(loaded, tensors):
    torch.cuda.reset_peak_memory_stats(loaded.device)
    torch.cuda.synchronize(loaded.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    outputs = loaded.model(
        input_ids=tensors["full_ids"],
        attention_mask=torch.ones_like(tensors["full_ids"]),
        use_cache=False,
        logits_to_keep=1,
    )
    probabilities, predictions = restricted(outputs.logits[:, -1], tensors["candidate_ids"])
    end.record()
    end.synchronize()
    return {
        "latency_ms": start.elapsed_time(end),
        "peak_memory": torch.cuda.max_memory_allocated(loaded.device),
        "probabilities": probabilities.cpu().tolist(),
        "predictions": predictions.cpu().tolist(),
    }


@torch.inference_mode()
def shared_forward_trial(loaded, tensors):
    branch_count = tensors["suffix_ids"].shape[0]
    prefix_attention = torch.ones_like(tensors["prefix_ids"])
    full_attention = torch.ones(
        (branch_count, tensors["full_prompt_tokens"]),
        dtype=torch.long,
        device=loaded.device,
    )
    torch.cuda.reset_peak_memory_stats(loaded.device)
    torch.cuda.synchronize(loaded.device)
    start = torch.cuda.Event(enable_timing=True)
    prefix_end = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    prefix_outputs = loaded.model.model(
        input_ids=tensors["prefix_ids"],
        attention_mask=prefix_attention,
        use_cache=True,
    )
    prefix_end.record()
    cache = prefix_outputs.past_key_values
    cache.batch_repeat_interleave(branch_count)
    cache_position = torch.arange(
        tensors["prefix_tokens"],
        tensors["full_prompt_tokens"],
        device=loaded.device,
    )
    outputs = loaded.model(
        input_ids=tensors["suffix_ids"],
        attention_mask=full_attention,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=False,
        logits_to_keep=1,
    )
    probabilities, predictions = restricted(outputs.logits[:, -1], tensors["candidate_ids"])
    end.record()
    end.synchronize()
    return {
        "prefix_ms": start.elapsed_time(prefix_end),
        "suffix_ms": prefix_end.elapsed_time(end),
        "latency_ms": start.elapsed_time(end),
        "peak_memory": torch.cuda.max_memory_allocated(loaded.device),
        "probabilities": probabilities.cpu().tolist(),
        "predictions": predictions.cpu().tolist(),
    }


def semantic_predictions(predictions: list[int], entries: list[dict[str, Any]]) -> list[int]:
    return [
        entry["permuted_to_original"][prediction]
        for prediction, entry in zip(predictions, entries)
    ]


def max_probability_delta(first: list[list[float]], second: list[list[float]]) -> float:
    return max(
        abs(left - right)
        for first_row, second_row in zip(first, second)
        for left, right in zip(first_row, second_row)
    )


def plot_summary(summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(summary["models"])
    labels = [model.split("/")[-1].replace("Qwen3-", "").replace("-Base", "") for model in models]
    speedups = [summary["models"][model]["latency_speedup_full_over_shared"] for model in models]
    full_memory = [summary["models"][model]["full_peak_cuda_memory_mib"] for model in models]
    shared_memory = [summary["models"][model]["shared_peak_cuda_memory_mib"] for model in models]
    figures = OUTPUT_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 4.2))
    axis.bar(labels, speedups, color="#1f77b4")
    axis.axhline(1.0, color="0.5", linestyle="--")
    axis.set(ylabel="Full batch / shared-KV latency", title="Shared-prefix latency speedup")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures / "shared_prefix_speedup.png", dpi=150)
    plt.close(figure)

    x = range(len(models))
    figure, axis = plt.subplots(figsize=(6, 4.2))
    axis.bar([value - 0.18 for value in x], full_memory, 0.36, label="full batch")
    axis.bar([value + 0.18 for value in x], shared_memory, 0.36, label="shared KV")
    axis.set(xticks=list(x), xticklabels=labels, ylabel="Peak CUDA memory (MiB)", title="Shared-prefix peak memory")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "shared_prefix_memory.png", dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Shared-state timing requires CUDA")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sample_map = load_samples()
    benchmark_ids = read_json(
        PROJECT_ROOT / "results" / "milestone6a" / "benchmark_sample_ids.json"
    )["sample_ids"]
    key_by_id = {key[1]: key for key in sample_map}
    selected_keys = [key_by_id[sample_id] for sample_id in benchmark_ids[: args.sample_groups]]
    manifest_entries = read_json(
        PROJECT_ROOT / "results" / "milestone4_permutations.json"
    )["entries"]
    entries_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in manifest_entries:
        key = (entry["dataset"], entry["sample_id"])
        if key in selected_keys:
            entries_by_key.setdefault(key, []).append(entry)
    for entries in entries_by_key.values():
        entries.sort(key=lambda entry: entry["permutation_id"])
    selection = {
        "source": "first 16 matching IDs from the frozen Milestone 6A four-option sample manifest",
        "sample_groups": [sample_id for _, sample_id in selected_keys],
        "branches_per_group": 4,
        "branch_source": "results/milestone4_permutations.json",
    }
    selection_path = OUTPUT_DIR / "shared_state_sample_ids.json"
    if selection_path.exists() and read_json(selection_path) != selection:
        raise ValueError("Existing shared-state sample selection differs")
    write_json(selection_path, selection)

    raw_path = OUTPUT_DIR / "shared_state_raw.jsonl"
    records = read_jsonl(raw_path)
    completed = {(r["model"], r["sample_id"], r["trial"]) for r in records}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8") as output_file:
        for model_name in MODELS:
            model_config = dict(config)
            model_config.update(
                {
                    "model_name": model_name,
                    "device": "cuda",
                    "dtype": "auto",
                    "cache_dir": str(PROJECT_ROOT / ".cache" / "huggingface" / "hub"),
                }
            )
            loaded = load_model(model_config)
            for group_index, key in enumerate(selected_keys, start=1):
                original = sample_map[key]
                entries = entries_by_key[key]
                if len(entries) != 4:
                    raise ValueError(f"Expected four frozen branches for {key}")
                tensors = prepare_group(loaded, original, entries)
                pending = [
                    trial
                    for trial in range(args.trials)
                    if (model_name, original.id, trial) not in completed
                ]
                if pending:
                    for _ in range(args.warmups):
                        full_forward_trial(loaded, tensors)
                        shared_forward_trial(loaded, tensors)
                for trial in pending:
                    if trial % 2:
                        shared = shared_forward_trial(loaded, tensors)
                        full = full_forward_trial(loaded, tensors)
                    else:
                        full = full_forward_trial(loaded, tensors)
                        shared = shared_forward_trial(loaded, tensors)
                    record = {
                        "model": model_name,
                        "dataset": original.dataset,
                        "subject": original.subject or None,
                        "sample_id": original.id,
                        "trial": trial,
                        "branch_count": 4,
                        "prefix_tokens": tensors["prefix_tokens"],
                        "suffix_tokens": tensors["suffix_tokens"],
                        "full_prompt_tokens": tensors["full_prompt_tokens"],
                        "full_forward_ms": full["latency_ms"],
                        "shared_prefix_ms": shared["prefix_ms"],
                        "shared_suffix_ms": shared["suffix_ms"],
                        "shared_total_ms": shared["latency_ms"],
                        "full_peak_cuda_memory_bytes": full["peak_memory"],
                        "shared_peak_cuda_memory_bytes": shared["peak_memory"],
                        "full_probabilities": full["probabilities"],
                        "shared_probabilities": shared["probabilities"],
                        "full_predictions": full["predictions"],
                        "shared_predictions": shared["predictions"],
                        "full_semantic_predictions": semantic_predictions(full["predictions"], entries),
                        "shared_semantic_predictions": semantic_predictions(shared["predictions"], entries),
                        "max_probability_delta": max_probability_delta(full["probabilities"], shared["probabilities"]),
                    }
                    records.append(record)
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_file.flush()
                    completed.add((model_name, original.id, trial))
                print(f"{model_name}: shared group {group_index}/{len(selected_keys)}", flush=True)
            del loaded
            gc.collect()
            torch.cuda.empty_cache()
    summary = summarize_shared_state(records)
    summary.update(
        {
            "milestone": "7",
            "experiment": "shared-prefix KV branching probe",
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "warmups_per_group": args.warmups,
                "trials_per_group": args.trials,
            },
        }
    )
    write_json(OUTPUT_DIR / "shared_state_summary.json", summary)
    plot_summary(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
