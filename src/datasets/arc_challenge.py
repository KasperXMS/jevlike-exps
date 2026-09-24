import random
from typing import Any

from datasets import load_dataset

from ..schema import MultipleChoiceSample


def _convert_row(row: dict[str, Any]) -> MultipleChoiceSample:
    choices = row["choices"]
    labels = [str(label) for label in choices["label"]]
    answer_key = str(row["answerKey"])
    try:
        gold_index = labels.index(answer_key)
    except ValueError as exc:
        raise ValueError(
            f"ARC sample {row['id']} answerKey {answer_key!r} is not in {labels!r}"
        ) from exc

    return MultipleChoiceSample(
        id=str(row["id"]),
        dataset="arc_challenge",
        context="",
        question=str(row["question"]),
        options=[str(text) for text in choices["text"]],
        gold_index=gold_index,
    )


def load_arc_challenge(
    split: str = "test", limit: int | None = 100, seed: int = 42
) -> list[MultipleChoiceSample]:
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    if limit is None or limit >= len(dataset):
        indices = list(range(len(dataset)))
    else:
        indices = random.Random(seed).sample(range(len(dataset)), limit)
    return [_convert_row(dataset[index]) for index in indices]

