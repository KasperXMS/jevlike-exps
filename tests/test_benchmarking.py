import pytest

from src.benchmarking import summarize_benchmark


def record(model, batch, method, trial, total):
    value = {
        "model": model,
        "batch_size": batch,
        "method": method,
        "trial": trial,
        "status": "ok",
        "total_ms": total,
        "peak_cuda_memory_bytes": 2**20,
    }
    if method == "generation":
        value.update(prefill_ms=total * 0.4, decode_ms=total * 0.6, generated_tokens_per_sample=8)
    else:
        value.update(forward_ms=total - 0.1, scoring_ms=0.1)
    return value


def test_benchmark_summary_latency_throughput_and_ratios():
    records = [
        record("model", 8, "generation", 0, 10.0),
        record("model", 8, "generation", 1, 20.0),
        record("model", 8, "label_logit", 0, 4.0),
        record("model", 8, "label_logit", 1, 6.0),
    ]
    batch = summarize_benchmark(records)["models"]["model"]["batch_sizes"]["8"]
    assert batch["generation"]["total_latency"]["mean_ms"] == 15.0
    assert batch["label_logit"]["throughput_samples_per_second"] == 1600.0
    assert batch["comparison"]["latency_speedup_generation_over_label"] == 3.0
    assert batch["comparison"]["throughput_ratio_label_over_generation"] == 3.0


def test_benchmark_summary_preserves_oom():
    records = [
        {
            "model": "model",
            "batch_size": 64,
            "method": "generation",
            "trial": None,
            "status": "oom",
            "error": "out of memory",
        }
    ]
    result = summarize_benchmark(records)["models"]["model"]["batch_sizes"]["64"]
    assert result["generation"]["status"] == "oom"
