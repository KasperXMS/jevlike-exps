import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking import summarize_benchmark  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "milestone6a"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    with (OUTPUT_DIR / "benchmark_raw.jsonl").open("r", encoding="utf-8") as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]
    keys = [(r["model"], r["batch_size"], r["method"], r["trial"]) for r in records]
    if len(keys) != len(set(keys)):
        raise AssertionError("Duplicate benchmark trial keys")
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["batch_size"], record["method"])].append(record)
        if record["status"] == "ok":
            if abs(record["total_ms"] - sum(
                record[name]
                for name in (
                    ("prefill_ms", "decode_ms")
                    if record["method"] == "generation"
                    else ("forward_ms", "scoring_ms")
                )
            )) > 0.01:
                raise AssertionError("Latency components do not sum to total")
    if len(grouped) != 24:
        raise AssertionError(f"Expected 24 model/batch/method combinations, found {len(grouped)}")
    for key, values in grouped.items():
        successful = [value for value in values if value["status"] == "ok"]
        failed = [value for value in values if value["status"] != "ok"]
        if successful and (len(successful) != 20 or failed):
            raise AssertionError(f"Incomplete successful trial group: {key}")
        if failed and len(failed) != 1:
            raise AssertionError(f"Expected one explicit failure record: {key}")

    frozen_ids = set()
    for filename in [
        "arc_challenge_sample_ids.json",
        "mmlu_sample_ids.json",
        "commonsenseqa_sample_ids.json",
    ]:
        frozen_ids.update(read_json(PROJECT_ROOT / "results" / filename)["sample_ids"])
    benchmark_ids = read_json(OUTPUT_DIR / "benchmark_sample_ids.json")["sample_ids"]
    if len(benchmark_ids) != 64 or len(set(benchmark_ids)) != 64 or not set(benchmark_ids) <= frozen_ids:
        raise AssertionError("Benchmark sample IDs are not 64 unique M3 frozen IDs")

    saved = read_json(OUTPUT_DIR / "summary.json")
    rebuilt = summarize_benchmark(records)
    if rebuilt["models"] != saved["models"]:
        raise AssertionError("Saved benchmark summary differs from raw recomputation")
    environment = saved["environment"]
    if environment["timing"] != "CUDA events with synchronization" or environment["trials"] != 20:
        raise AssertionError("Benchmark environment does not document required timing/trials")
    print("Milestone 6A validation passed: 24 combinations, 460 timed trials, 1 memory-limit record")


if __name__ == "__main__":
    main()
