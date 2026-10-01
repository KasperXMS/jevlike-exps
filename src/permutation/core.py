from dataclasses import dataclass
import hashlib
import random

from ..schema import MultipleChoiceSample


@dataclass(frozen=True)
class PermutationCase:
    sample: MultipleChoiceSample
    permutation_id: int
    original_to_permuted: list[int]
    permuted_to_original: list[int]
    gold_original_index: int
    gold_permuted_index: int


def _rng(seed: int, sample: MultipleChoiceSample) -> random.Random:
    value = f"{seed}:{sample.dataset}:{sample.id}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(value).digest()[:8], "big"))


def _invert(permuted_to_original: list[int]) -> list[int]:
    original_to_permuted = [0] * len(permuted_to_original)
    for permuted_index, original_index in enumerate(permuted_to_original):
        original_to_permuted[original_index] = permuted_index
    return original_to_permuted


def build_permutations(
    sample: MultipleChoiceSample, count: int = 3, seed: int = 42
) -> list[PermutationCase]:
    """Return the original ordering followed by unique deterministic permutations."""
    option_count = len(sample.options)
    maximum = 1
    for value in range(2, option_count + 1):
        maximum *= value
    if count + 1 > maximum:
        raise ValueError("Requested more unique orderings than the options permit")

    identity = tuple(range(option_count))
    orderings = [identity]
    seen = {identity}
    rng = _rng(seed, sample)
    while len(orderings) < count + 1:
        candidate = list(identity)
        rng.shuffle(candidate)
        ordering = tuple(candidate)
        if ordering not in seen:
            seen.add(ordering)
            orderings.append(ordering)

    cases = []
    for permutation_id, ordering in enumerate(orderings):
        permuted_to_original = list(ordering)
        original_to_permuted = _invert(permuted_to_original)
        gold_permuted_index = original_to_permuted[sample.gold_index]
        cases.append(
            PermutationCase(
                sample=MultipleChoiceSample(
                    id=sample.id,
                    dataset=sample.dataset,
                    question=sample.question,
                    options=[sample.options[index] for index in permuted_to_original],
                    gold_index=gold_permuted_index,
                    context=sample.context,
                    subject=sample.subject,
                ),
                permutation_id=permutation_id,
                original_to_permuted=original_to_permuted,
                permuted_to_original=permuted_to_original,
                gold_original_index=sample.gold_index,
                gold_permuted_index=gold_permuted_index,
            )
        )
    return cases


def map_prediction_to_original(
    prediction_permuted_index: int | None, permuted_to_original: list[int]
) -> int | None:
    if prediction_permuted_index is None:
        return None
    return permuted_to_original[prediction_permuted_index]
