import hashlib
import json
import random
from typing import Any

from datasets import load_dataset

from ..schema import MultipleChoiceSample


MMLU_SUBJECT_CONFIGS = {
    "abstract_algebra": "abstract_algebra",
    "computer_science": "college_computer_science",
    "physics": "college_physics",
    "history": "high_school_world_history",
    "psychology": "high_school_psychology",
}


def resolve_mmlu_subject(subject: str) -> str:
    return MMLU_SUBJECT_CONFIGS.get(subject, subject)


def _stable_id(
    subject: str, config_name: str, source_index: int, row: dict[str, Any]
) -> str:
    payload = json.dumps(
        {
            "config": config_name,
            "question": row["question"],
            "choices": list(row["choices"]),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"mmlu:{subject}:{source_index}:{digest}"


def _convert_row(
    row: dict[str, Any], subject: str, config_name: str, source_index: int
) -> MultipleChoiceSample:
    return MultipleChoiceSample(
        id=_stable_id(subject, config_name, source_index, row),
        dataset="mmlu",
        subject=subject,
        context="",
        question=str(row["question"]),
        options=[str(choice) for choice in row["choices"]],
        gold_index=int(row["answer"]),
    )


def load_mmlu(
    subjects: list[str],
    split: str = "test",
    limit: int | None = None,
    seed: int = 42,
) -> list[MultipleChoiceSample]:
    samples: list[MultipleChoiceSample] = []
    for subject in subjects:
        config_name = resolve_mmlu_subject(subject)
        dataset = load_dataset("cais/mmlu", config_name, split=split)
        if limit is None or limit >= len(dataset):
            indices = list(range(len(dataset)))
        else:
            indices = random.Random(f"{seed}:{subject}").sample(
                range(len(dataset)), limit
            )
        samples.extend(
            _convert_row(dataset[index], subject, config_name, index) for index in indices
        )
    ids = [sample.id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("MMLU generated duplicate stable sample IDs")
    return samples


def load_mmlu_by_ids(
    subjects: list[str], sample_ids: list[str], split: str = "test"
) -> list[MultipleChoiceSample]:
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Frozen MMLU sample IDs must be unique")
    samples_by_id: dict[str, MultipleChoiceSample] = {}
    for subject in subjects:
        config_name = resolve_mmlu_subject(subject)
        dataset = load_dataset("cais/mmlu", config_name, split=split)
        for source_index, row in enumerate(dataset):
            sample = _convert_row(row, subject, config_name, source_index)
            samples_by_id[sample.id] = sample
    missing = [sample_id for sample_id in sample_ids if sample_id not in samples_by_id]
    if missing:
        raise ValueError(f"Frozen MMLU sample IDs not found: {missing}")
    return [samples_by_id[sample_id] for sample_id in sample_ids]
