from collections import defaultdict
from statistics import mean, median
from typing import Any


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": mean(values),
        "p50_ms": median(values),
        "p95_ms": _percentile(values, 0.95),
    }


def summarize_benchmark(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    failures: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in records:
        key = (record["model"], record["batch_size"], record["method"])
        if record["status"] == "ok":
            grouped[key].append(record)
        else:
            failures[key] = {
                "status": record["status"],
                "error": record.get("error"),
            }

    models: dict[str, Any] = {}
    all_keys = sorted(set(grouped) | set(failures))
    for model, batch_size, method in all_keys:
        batch = models.setdefault(model, {"batch_sizes": {}})["batch_sizes"].setdefault(
            str(batch_size), {}
        )
        key = (model, batch_size, method)
        method_records = grouped.get(key, [])
        if not method_records:
            batch[method] = failures[key]
            continue
        total = [record["total_ms"] for record in method_records]
        method_summary = {
            "status": "ok",
            "trials": len(method_records),
            "total_latency": _latency(total),
            "throughput_samples_per_second": batch_size * 1000.0 / mean(total),
            "peak_cuda_memory_mib": max(record["peak_cuda_memory_bytes"] for record in method_records) / 2**20,
        }
        if method == "generation":
            method_summary["prefill_latency"] = _latency(
                [record["prefill_ms"] for record in method_records]
            )
            method_summary["decode_latency"] = _latency(
                [record["decode_ms"] for record in method_records]
            )
            method_summary["generated_tokens_per_sample"] = method_records[0][
                "generated_tokens_per_sample"
            ]
        else:
            method_summary["forward_latency"] = _latency(
                [record["forward_ms"] for record in method_records]
            )
            method_summary["scoring_overhead"] = _latency(
                [record["scoring_ms"] for record in method_records]
            )
        batch[method] = method_summary

    for model_data in models.values():
        for batch in model_data["batch_sizes"].values():
            generation = batch.get("generation", {})
            label = batch.get("label_logit", {})
            if generation.get("status") == label.get("status") == "ok":
                generation_total = generation["total_latency"]["mean_ms"]
                label_total = label["total_latency"]["mean_ms"]
                batch["comparison"] = {
                    "latency_speedup_generation_over_label": generation_total / label_total,
                    "throughput_ratio_label_over_generation": label["throughput_samples_per_second"] / generation["throughput_samples_per_second"],
                }
    return {"models": models}
