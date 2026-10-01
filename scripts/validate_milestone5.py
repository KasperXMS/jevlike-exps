import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import (  # noqa: E402
    build_calibration_observation,
    build_invariance_summary,
    build_original_summary,
    build_paired,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "milestone5"
INPUT_FILES = [
    "results/arc_challenge_sample_ids.json",
    "results/mmlu_sample_ids.json",
    "results/commonsenseqa_sample_ids.json",
    "results/milestone4_permutations.json",
    "results/milestone4_calibration_split.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently validate Milestone 5 outputs")
    parser.add_argument("--initialize-hashes", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    hashes_path = OUTPUT_DIR / "input_hashes.json"
    current_hashes = {name: sha256(PROJECT_ROOT / name) for name in INPUT_FILES}
    if args.initialize_hashes:
        if hashes_path.exists():
            raise ValueError("Input hashes are already initialized")
        hashes_path.write_text(
            json.dumps(current_hashes, indent=2) + "\n", encoding="utf-8"
        )
    if not hashes_path.exists() or read_json(hashes_path) != current_hashes:
        raise AssertionError("Frozen IDs, permutation manifest, or calibration split changed")

    manifest = read_json(PROJECT_ROOT / "results" / "milestone4_permutations.json")["entries"]
    manifest_by_key = {
        (entry["dataset"], entry["sample_id"], entry["permutation_id"]): entry
        for entry in manifest
    }
    split = read_json(PROJECT_ROOT / "results" / "milestone4_calibration_split.json")
    saved_summary = read_json(OUTPUT_DIR / "summary.json")
    saved_paired = read_json(OUTPUT_DIR / "paired.json")
    saved_invariance = read_json(OUTPUT_DIR / "invariance.json")
    saved_calibration = read_json(OUTPUT_DIR / "calibration.json")
    rebuilt_summary = {"milestone": "5", "models": {}}
    rebuilt_paired = {"milestone": "5", "models": {}}
    rebuilt_invariance = {"milestone": "5", "models": {}}
    rebuilt_calibration = {"milestone": "5", "models": {}}

    metadata_files = sorted(OUTPUT_DIR.glob("model_metadata_*.json"))
    for metadata_path in metadata_files:
        suffix = metadata_path.stem.removeprefix("model_metadata_")
        metadata = read_json(metadata_path)
        model = metadata["model"]
        originals = read_jsonl(OUTPUT_DIR / f"model_results_{suffix}.jsonl")
        permutations = read_jsonl(OUTPUT_DIR / f"permutation_results_{suffix}.jsonl")
        original_keys = {(r["dataset"], r["sample_id"], r["method"]) for r in originals}
        permutation_keys = {(r["dataset"], r["sample_id"], r["permutation_id"]) for r in permutations}
        if len(originals) != 1400 or len(original_keys) != 1400:
            raise AssertionError(f"{model}: expected 1,400 unique original records")
        if len(permutations) != 2800 or permutation_keys != manifest_by_key.keys():
            raise AssertionError(f"{model}: permutation keys differ from frozen manifest")
        for record in permutations:
            key = (record["dataset"], record["sample_id"], record["permutation_id"])
            entry = manifest_by_key[key]
            if record["permuted_to_original"] != entry["permuted_to_original"]:
                raise AssertionError(f"{model}: changed permutation mapping at {key}")
            if record["prediction_original_index"] != record["permuted_to_original"][record["prediction_permuted_index"]]:
                raise AssertionError(f"{model}: invalid semantic prediction mapping at {key}")
        parameters = metadata["parameter_count"]
        rebuilt_summary["models"][model] = build_original_summary(model, parameters, originals)
        rebuilt_paired["models"][model] = build_paired(originals)
        rebuilt_invariance["models"][model] = build_invariance_summary(model, parameters, permutations)
        rebuilt_calibration["models"][model] = build_calibration_observation(model, parameters, originals, split)

    comparisons = [
        ("summary", rebuilt_summary, saved_summary),
        ("paired", rebuilt_paired, saved_paired),
        ("invariance", rebuilt_invariance, saved_invariance),
        ("calibration", rebuilt_calibration, saved_calibration),
    ]
    for name, rebuilt, saved in comparisons:
        if rebuilt != saved:
            raise AssertionError(f"Saved {name}.json differs from independent raw recomputation")
    print(
        f"Milestone 5 validation passed: {len(metadata_files)} models, "
        "4,200 original records, 8,400 permutation records"
    )


if __name__ == "__main__":
    main()
