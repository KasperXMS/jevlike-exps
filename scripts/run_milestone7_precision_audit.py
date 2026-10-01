import gc
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_milestone7_shared_state as shared  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "milestone7"


def main() -> None:
    source = shared.read_jsonl(OUTPUT_DIR / "shared_state_raw.jsonl")
    mismatch_groups = sorted(
        {
            (record["model"], record["sample_id"])
            for record in source
            if record["full_predictions"] != record["shared_predictions"]
        }
    )
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "qwen3_0.6b.yaml").read_text(encoding="utf-8")
    )
    sample_map = shared.load_samples()
    sample_by_id = {sample_id: sample for (_, sample_id), sample in sample_map.items()}
    manifest = shared.read_json(
        PROJECT_ROOT / "results" / "milestone4_permutations.json"
    )["entries"]
    output_path = OUTPUT_DIR / "precision_audit_raw.jsonl"
    records = shared.read_jsonl(output_path)
    completed = {(r["model"], r["sample_id"], r["branch_id"]) for r in records}
    with output_path.open("a", encoding="utf-8") as output_file:
        for model_name in sorted({model for model, _ in mismatch_groups}):
            model_config = dict(config)
            model_config.update(
                {
                    "model_name": model_name,
                    "device": "cuda",
                    "dtype": "float32",
                    "cache_dir": str(PROJECT_ROOT / ".cache" / "huggingface" / "hub"),
                }
            )
            loaded = shared.load_model(model_config)
            for _, sample_id in [value for value in mismatch_groups if value[0] == model_name]:
                original = sample_by_id[sample_id]
                entries = sorted(
                    [entry for entry in manifest if entry["sample_id"] == sample_id],
                    key=lambda entry: entry["permutation_id"],
                )
                tensors = shared.prepare_group(loaded, original, entries)
                for branch_id in range(4):
                    key = (model_name, sample_id, branch_id)
                    if key in completed:
                        continue
                    one = {
                        **tensors,
                        "suffix_ids": tensors["suffix_ids"][branch_id : branch_id + 1],
                        "full_ids": tensors["full_ids"][branch_id : branch_id + 1],
                        "candidate_ids": tensors["candidate_ids"][branch_id : branch_id + 1],
                    }
                    full = shared.full_forward_trial(loaded, one)
                    branched = shared.shared_forward_trial(loaded, one)
                    record = {
                        "model": model_name,
                        "dtype": "float32",
                        "sample_id": sample_id,
                        "branch_id": branch_id,
                        "full_prediction": full["predictions"][0],
                        "shared_prediction": branched["predictions"][0],
                        "full_probabilities": full["probabilities"][0],
                        "shared_probabilities": branched["probabilities"][0],
                        "max_probability_delta": shared.max_probability_delta(
                            full["probabilities"], branched["probabilities"]
                        ),
                    }
                    records.append(record)
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_file.flush()
                    completed.add(key)
            del loaded
            gc.collect()
            torch.cuda.empty_cache()
    summary = {}
    for model_name in sorted({record["model"] for record in records}):
        selected = [record for record in records if record["model"] == model_name]
        summary[model_name] = {
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
    shared.write_json(
        OUTPUT_DIR / "precision_audit_summary.json",
        {
            "purpose": "Distinguish BF16 segmented-attention numerical sensitivity from a cache-position correctness bug",
            "models": summary,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
