import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.shared_state import summarize_shared_state  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "milestone7"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def main() -> None:
    records = read_jsonl(OUTPUT_DIR / "shared_state_raw.jsonl")
    keys = {(r["model"], r["sample_id"], r["trial"]) for r in records}
    if len(records) != 480 or len(keys) != 480:
        raise AssertionError("Expected 480 unique shared-state trials")
    selection = read_json(OUTPUT_DIR / "shared_state_sample_ids.json")
    if len(selection["sample_groups"]) != 16 or len(set(selection["sample_groups"])) != 16:
        raise AssertionError("Expected 16 unique shared-state groups")
    for record in records:
        if record["branch_count"] != 4:
            raise AssertionError("Every shared-state group must have four branches")
        for rows in [record["full_probabilities"], record["shared_probabilities"]]:
            if any(abs(sum(row) - 1.0) > 1e-5 for row in rows):
                raise AssertionError("Restricted probabilities are not normalized")
    saved = read_json(OUTPUT_DIR / "shared_state_summary.json")
    rebuilt = summarize_shared_state(records)
    if rebuilt["models"] != saved["models"]:
        raise AssertionError("Shared-state summary differs from raw recomputation")

    audit = read_jsonl(OUTPUT_DIR / "precision_audit_raw.jsonl")
    audit_keys = {(r["model"], r["sample_id"], r["branch_id"]) for r in audit}
    if len(audit) != 16 or len(audit_keys) != 16:
        raise AssertionError("Expected 16 unique float32 precision-audit branches")
    if any(r["full_prediction"] != r["shared_prediction"] for r in audit):
        raise AssertionError("Float32 cache audit still changes a prediction")
    if max(r["max_probability_delta"] for r in audit) >= 1e-4:
        raise AssertionError("Float32 cache audit probability delta is too large")
    audit_summary = read_json(OUTPUT_DIR / "precision_audit_summary.json")["models"]
    for model in {record["model"] for record in audit}:
        selected = [record for record in audit if record["model"] == model]
        rebuilt_audit = {
            "mismatch_groups_audited": len({record["sample_id"] for record in selected}),
            "branches": len(selected),
            "prediction_agreement": sum(
                record["full_prediction"] == record["shared_prediction"]
                for record in selected
            )
            / len(selected),
            "maximum_probability_delta": max(
                record["max_probability_delta"] for record in selected
            ),
        }
        if rebuilt_audit != audit_summary[model]:
            raise AssertionError("Precision audit summary differs from raw recomputation")

    conclusion = read_json(OUTPUT_DIR / "conclusion.json")["evidence"]
    m5 = read_json(PROJECT_ROOT / "results" / "milestone5" / "invariance.json")["models"]
    model_keys = [
        ("qwen3_0_6b", "Qwen/Qwen3-0.6B-Base"),
        ("qwen3_1_7b", "Qwen/Qwen3-1.7B-Base"),
        ("qwen3_4b", "Qwen/Qwen3-4B-Base"),
    ]
    for short, model in model_keys:
        datasets = m5[model]["datasets"].values()
        total = sum(dataset["sample_count"] for dataset in datasets)
        semantic = sum(
            dataset["semantic_consistency"] * dataset["sample_count"]
            for dataset in datasets
        ) / total
        if semantic != conclusion["label_logit_scaling"]["sample_weighted_semantic_consistency"][short]:
            raise AssertionError("Conclusion semantic consistency differs from M5")
        flip_count = sum(dataset["flip_rates"]["correct_to_wrong"]["count"] for dataset in datasets)
        opportunities = sum(dataset["flip_rates"]["correct_to_wrong"]["opportunities"] for dataset in datasets)
        if flip_count / opportunities != conclusion["label_logit_scaling"]["sample_weighted_correct_to_wrong_rate"][short]:
            raise AssertionError("Conclusion C-to-W rate differs from M5")

    m6 = read_json(PROJECT_ROOT / "results" / "milestone6a" / "summary.json")["models"]
    batch_one = sorted(
        model["batch_sizes"]["1"]["comparison"]["latency_speedup_generation_over_label"]
        for model in m6.values()
    )
    if [batch_one[0], batch_one[-1]] != conclusion["zero_decoding_efficiency"]["batch_1_speedup_range"]:
        raise AssertionError("Conclusion batch-1 range differs from M6")
    for short, model in model_keys:
        if rebuilt["models"][model]["latency_speedup_full_over_shared"] != conclusion["shared_prefix_probe"]["full_over_shared_latency_speedup"][short]:
            raise AssertionError("Conclusion shared-prefix speedup differs from M7")
    print("Milestone 7 validation passed: 480 BF16 trials and 16 float32 audit branches")


if __name__ == "__main__":
    main()
