import random
from typing import Any

from datasets import load_dataset

from ..schema import MultipleChoiceSample


def _convert_row(row: dict[str, Any]) -> MultipleChoiceSample:
    question_value = row["question"]
    if isinstance(question_value, dict):
        question = str(question_value["stem"])
        choices = question_value.get("choices", row.get("choices"))
    else:
        question = str(question_value)
        choices = row["choices"]
    labels = [str(label) for label in choices["label"]]
    answer_key = str(row["answerKey"])
    try:
        gold_index = labels.index(answer_key)
    except ValueError as exc:
        raise ValueError(
            f"CommonsenseQA sample {row['id']} answerKey {answer_key!r} "
            f"is not in {labels!r}"
        ) from exc
    return MultipleChoiceSample(
        id=str(row["id"]),
        dataset="commonsenseqa",
        context="",
        question=question,
        options=[str(text) for text in choices["text"]],
        gold_index=gold_index,
    )


def load_commonsenseqa(
    split: str = "validation", limit: int | None = 100, seed: int = 42
) -> list[MultipleChoiceSample]:
    dataset = load_dataset("tau/commonsense_qa", split=split)
    if limit is None or limit >= len(dataset):
        indices = list(range(len(dataset)))
    else:
        indices = random.Random(seed).sample(range(len(dataset)), limit)
    return [_convert_row(dataset[index]) for index in indices]


def load_commonsenseqa_by_ids(
    sample_ids: list[str], split: str = "validation"
) -> list[MultipleChoiceSample]:
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Frozen CommonsenseQA sample IDs must be unique")
    dataset = load_dataset("tau/commonsense_qa", split=split)
    rows_by_id = {str(row["id"]): row for row in dataset}
    missing = [sample_id for sample_id in sample_ids if sample_id not in rows_by_id]
    if missing:
        raise ValueError(f"Frozen CommonsenseQA sample IDs not found: {missing}")
    return [_convert_row(rows_by_id[sample_id]) for sample_id in sample_ids]
