from statistics import mean, median
from typing import Any


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "p50": median(values),
        "p95": _percentile(values, 0.95),
    }


def summarize_shared_state(records: list[dict[str, Any]]) -> dict[str, Any]:
    models = {}
    for model in sorted({record["model"] for record in records}):
        selected = [record for record in records if record["model"] == model]
        full = [record["full_forward_ms"] for record in selected]
        shared = [record["shared_total_ms"] for record in selected]
        models[model] = {
            "sample_groups": len({record["sample_id"] for record in selected}),
            "trials": len(selected),
            "branch_count": selected[0]["branch_count"],
            "mean_prefix_token_fraction": mean(
                record["prefix_tokens"] / record["full_prompt_tokens"]
                for record in selected
            ),
            "full_forward_latency_ms": _stats(full),
            "shared_total_latency_ms": _stats(shared),
            "shared_prefix_latency_ms": _stats(
                [record["shared_prefix_ms"] for record in selected]
            ),
            "shared_suffix_latency_ms": _stats(
                [record["shared_suffix_ms"] for record in selected]
            ),
            "latency_speedup_full_over_shared": mean(full) / mean(shared),
            "full_peak_cuda_memory_mib": max(
                record["full_peak_cuda_memory_bytes"] for record in selected
            )
            / 2**20,
            "shared_peak_cuda_memory_mib": max(
                record["shared_peak_cuda_memory_bytes"] for record in selected
            )
            / 2**20,
            "prediction_agreement": mean(
                record["full_predictions"] == record["shared_predictions"]
                for record in selected
            ),
            "semantic_prediction_agreement": mean(
                record["full_semantic_predictions"]
                == record["shared_semantic_predictions"]
                for record in selected
            ),
            "mean_max_probability_delta": mean(
                record["max_probability_delta"] for record in selected
            ),
            "maximum_probability_delta": max(
                record["max_probability_delta"] for record in selected
            ),
        }
    return {"models": models}
