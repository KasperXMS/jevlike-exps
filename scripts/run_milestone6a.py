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

from src.benchmarking import summarize_benchmark  # noqa: E402
from src.datasets import (  # noqa: E402
    load_arc_challenge_by_ids,
    load_commonsenseqa_by_ids,
    load_mmlu_by_ids,
)
from src.inference.label_logits import inspect_label_tokens  # noqa: E402
from src.models import load_model  # noqa: E402
from src.prompting import build_prompt  # noqa: E402
from src.schema import MultipleChoiceSample  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "milestone6a"
MODEL_SPECS = {
    "Qwen/Qwen3-0.6B-Base": "qwen3_0_6b_base",
    "Qwen/Qwen3-1.7B-Base": "qwen3_1_7b_base",
    "Qwen/Qwen3-4B-Base": "qwen3_4b_base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Milestone 6A CUDA benchmark")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "qwen3_0.6b.yaml"
    )
    parser.add_argument("--batch-sizes", default="1,8,32,64")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--generated-tokens", type=int, default=8)
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


def load_samples() -> list[MultipleChoiceSample]:
    results = PROJECT_ROOT / "results"
    arc = read_json(results / "arc_challenge_sample_ids.json")
    mmlu = read_json(results / "mmlu_sample_ids.json")
    csqa = read_json(results / "commonsenseqa_sample_ids.json")
    return [
        *load_arc_challenge_by_ids(arc["sample_ids"], split=arc.get("split", "test")),
        *load_mmlu_by_ids(mmlu["subjects"], mmlu["sample_ids"], split=mmlu.get("split", "test")),
        *load_commonsenseqa_by_ids(csqa["sample_ids"], split=csqa.get("split", "validation")),
    ]


def benchmark_samples(samples: list[MultipleChoiceSample], seed: int) -> list[MultipleChoiceSample]:
    ordered = [sample for sample in samples if len(sample.options) == 4]
    random.Random(seed).shuffle(ordered)
    return ordered[:64]


def _batch_inputs(tokenizer, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return {
        key: value.to(device)
        for key, value in tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).items()
    }


def _events(count: int) -> list[torch.cuda.Event]:
    return [torch.cuda.Event(enable_timing=True) for _ in range(count)]


@torch.inference_mode()
def generation_trial(loaded, inputs, generated_tokens: int) -> dict[str, float | int]:
    torch.cuda.reset_peak_memory_stats(loaded.device)
    torch.cuda.synchronize(loaded.device)
    start, prefill_end, end = _events(3)
    start.record()
    outputs = loaded.model(**inputs, use_cache=True, logits_to_keep=1)
    prefill_end.record()
    next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    cache = outputs.past_key_values
    attention_mask = inputs["attention_mask"]
    for _ in range(1, generated_tokens):
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
        outputs = loaded.model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = outputs.past_key_values
        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    end.record()
    end.synchronize()
    return {
        "prefill_ms": start.elapsed_time(prefill_end),
        "decode_ms": prefill_end.elapsed_time(end),
        "total_ms": start.elapsed_time(end),
        "generated_tokens_per_sample": generated_tokens,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(loaded.device),
    }


@torch.inference_mode()
def label_trial(loaded, inputs, token_ids: torch.Tensor) -> dict[str, float | int]:
    torch.cuda.reset_peak_memory_stats(loaded.device)
    torch.cuda.synchronize(loaded.device)
    start, forward_end, end = _events(3)
    start.record()
    outputs = loaded.model(**inputs, use_cache=False, logits_to_keep=1)
    forward_end.record()
    final_logits = outputs.logits[:, -1].float()
    candidate_logits = final_logits.gather(1, token_ids)
    probabilities = candidate_logits.softmax(dim=1)
    _ = probabilities.argmax(dim=1)
    end.record()
    end.synchronize()
    return {
        "forward_ms": start.elapsed_time(forward_end),
        "scoring_ms": forward_end.elapsed_time(end),
        "total_ms": start.elapsed_time(end),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(loaded.device),
    }


def _label_token_ids(loaded, prompts: list[str], samples: list[MultipleChoiceSample]) -> torch.Tensor:
    rows = []
    option_counts = {len(sample.options) for sample in samples}
    if len(option_counts) != 1:
        raise ValueError("A benchmark batch must have a fixed option count")
    for prompt, sample in zip(prompts, samples):
        inspection = inspect_label_tokens(loaded.tokenizer, prompt, len(sample.options))
        if not inspection.supported or inspection.token_ids is None:
            raise RuntimeError(f"Unsupported labels for benchmark sample {sample.id}")
        rows.append(inspection.token_ids)
    return torch.tensor(rows, dtype=torch.long, device=loaded.device)


def _record_failure(model: str, batch_size: int, method: str, exc: Exception) -> dict[str, Any]:
    return {
        "model": model,
        "batch_size": batch_size,
        "method": method,
        "trial": None,
        "status": "oom" if isinstance(exc, torch.OutOfMemoryError) else "error",
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_method(
    loaded,
    samples: list[MultipleChoiceSample],
    batch_size: int,
    method: str,
    warmups: int,
    trials: int,
    generated_tokens: int,
    output_file,
    existing_trials: set[tuple[str, int, str, int | None]],
) -> None:
    # CommonsenseQA has five options. Keeping each timed batch shape-homogeneous
    # avoids padding the candidate-label matrix or changing scoring work by model.
    eligible = [sample for sample in samples if len(sample.options) == 4]
    if len(eligible) < batch_size:
        raise ValueError(f"Only {len(eligible)} four-option benchmark samples available")
    batch = eligible[:batch_size]
    prompts = [build_prompt(sample) for sample in batch]
    inputs = _batch_inputs(loaded.tokenizer, prompts, loaded.device)
    token_ids = _label_token_ids(loaded, prompts, batch) if method == "label_logit" else None
    runner = (
        (lambda: generation_trial(loaded, inputs, generated_tokens))
        if method == "generation"
        else (lambda: label_trial(loaded, inputs, token_ids))
    )
    pending = [
        trial
        for trial in range(trials)
        if (loaded.name, batch_size, method, trial) not in existing_trials
    ]
    if (loaded.name, batch_size, method, None) in existing_trials:
        return
    if not pending:
        return
    try:
        for _ in range(warmups):
            runner()
        for trial in pending:
            metrics = runner()
            record = {
                "model": loaded.name,
                "batch_size": batch_size,
                "method": method,
                "trial": trial,
                "status": "ok",
                "sequence_length": int(inputs["input_ids"].shape[1]),
                **metrics,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            existing_trials.add((loaded.name, batch_size, method, trial))
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
        if not is_oom:
            raise
        failure = _record_failure(loaded.name, batch_size, method, torch.OutOfMemoryError(str(exc)))
        output_file.write(json.dumps(failure, ensure_ascii=False) + "\n")
        output_file.flush()
        existing_trials.add((loaded.name, batch_size, method, None))
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Milestone 6A requires CUDA; no benchmark was run")
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    samples = benchmark_samples(load_samples(), int(config.get("seed", 42)))
    sample_manifest = {
        "seed": int(config.get("seed", 42)),
        "selection": "deterministic shuffle of four-option M3 frozen samples; first 64",
        "timed_batches": "first B samples from selection",
        "sample_ids": [sample.id for sample in samples],
    }
    manifest_path = OUTPUT_DIR / "benchmark_sample_ids.json"
    if manifest_path.exists() and read_json(manifest_path) != sample_manifest:
        raise ValueError("Existing benchmark sample manifest differs")
    write_json(manifest_path, sample_manifest)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "warmups": args.warmups,
        "trials": args.trials,
        "generated_tokens_per_sample": args.generated_tokens,
        "generation_contract": "fixed-step greedy decode; no early EOS termination",
        "timing": "CUDA events with synchronization",
    }
    write_json(OUTPUT_DIR / "environment.json", environment)

    raw_path = OUTPUT_DIR / "benchmark_raw.jsonl"
    skip_path = OUTPUT_DIR / "skip_combinations.json"
    skips = read_json(skip_path).get("combinations", []) if skip_path.exists() else []
    existing = read_jsonl(raw_path)
    existing_trials = {
        (r["model"], r["batch_size"], r["method"], r["trial"]) for r in existing
    }
    with raw_path.open("a", encoding="utf-8") as output_file:
        for model_name in MODEL_SPECS:
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
            for batch_size in batch_sizes:
                for method in ("generation", "label_logit"):
                    skip = next(
                        (
                            item
                            for item in skips
                            if item["model"] == model_name
                            and item["batch_size"] == batch_size
                            and item["method"] == method
                        ),
                        None,
                    )
                    if skip and (model_name, batch_size, method, None) not in existing_trials:
                        record = {
                            "model": model_name,
                            "batch_size": batch_size,
                            "method": method,
                            "trial": None,
                            "status": skip["status"],
                            "error": skip["reason"],
                        }
                        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output_file.flush()
                        existing_trials.add((model_name, batch_size, method, None))
                    elif not skip:
                        run_method(
                            loaded,
                            samples,
                            batch_size,
                            method,
                            args.warmups,
                            args.trials,
                            args.generated_tokens,
                            output_file,
                            existing_trials,
                        )
                    torch.cuda.empty_cache()
                    print(f"{model_name}: batch {batch_size} {method} complete", flush=True)
            del loaded
            gc.collect()
            torch.cuda.empty_cache()
    records = read_jsonl(raw_path)
    summary = summarize_benchmark(records)
    summary.update({"milestone": "6A", "environment": environment})
    write_json(OUTPUT_DIR / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
