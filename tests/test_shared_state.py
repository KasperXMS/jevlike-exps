from src.prompting import build_prompt, build_prompt_parts
from src.schema import MultipleChoiceSample
from src.shared_state import summarize_shared_state


def test_prompt_parts_reconstruct_standard_prompt():
    sample = MultipleChoiceSample(
        id="one",
        dataset="test",
        context="Some context.",
        question="Question?",
        options=["First", "Second", "Third", "Fourth"],
        gold_index=1,
    )
    prefix, suffix = build_prompt_parts(sample)
    assert prefix + suffix == build_prompt(sample)
    assert prefix.endswith("Options:\n")
    assert suffix.startswith("A. First")


def test_shared_state_summary():
    records = [
        {
            "model": "model",
            "sample_id": "one",
            "branch_count": 4,
            "prefix_tokens": 20,
            "full_prompt_tokens": 100,
            "full_forward_ms": 10.0,
            "shared_total_ms": 8.0,
            "shared_prefix_ms": 2.0,
            "shared_suffix_ms": 6.0,
            "full_peak_cuda_memory_bytes": 2 * 2**20,
            "shared_peak_cuda_memory_bytes": 3 * 2**20,
            "full_predictions": [0, 1, 2, 3],
            "shared_predictions": [0, 1, 2, 3],
            "full_semantic_predictions": [0, 0, 0, 0],
            "shared_semantic_predictions": [0, 0, 0, 0],
            "max_probability_delta": 1e-5,
        }
    ]
    summary = summarize_shared_state(records)["models"]["model"]
    assert summary["latency_speedup_full_over_shared"] == 1.25
    assert summary["prediction_agreement"] == 1.0
    assert summary["shared_peak_cuda_memory_mib"] == 3.0
