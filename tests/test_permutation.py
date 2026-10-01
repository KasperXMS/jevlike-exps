from src.permutation import build_permutations, map_prediction_to_original
from src.schema import MultipleChoiceSample


def make_sample(options=None, gold_index=2):
    return MultipleChoiceSample(
        id="sample-1",
        dataset="test",
        question="Question?",
        options=options or ["zero", "one", "two", "three"],
        gold_index=gold_index,
        subject="subject",
    )


def test_permutations_are_deterministic_and_preserve_original_sample():
    sample = make_sample()
    original_options = list(sample.options)
    first = build_permutations(sample, seed=42)
    second = build_permutations(sample, seed=42)

    assert [case.permuted_to_original for case in first] == [
        case.permuted_to_original for case in second
    ]
    assert len({tuple(case.permuted_to_original) for case in first}) == 4
    assert first[0].permuted_to_original == [0, 1, 2, 3]
    assert sample.options == original_options
    assert sample.gold_index == 2


def test_gold_and_prediction_map_by_index_in_both_directions():
    sample = make_sample()
    for case in build_permutations(sample, seed=7):
        assert case.gold_permuted_index == case.original_to_permuted[2]
        assert case.permuted_to_original[case.gold_permuted_index] == 2
        for permuted_index, original_index in enumerate(case.permuted_to_original):
            assert map_prediction_to_original(permuted_index, case.permuted_to_original) == original_index


def test_duplicate_option_text_does_not_affect_semantic_mapping():
    sample = make_sample(options=["same", "same", "different", "other"], gold_index=1)
    for case in build_permutations(sample, seed=99):
        permuted_index = case.original_to_permuted[1]
        assert map_prediction_to_original(permuted_index, case.permuted_to_original) == 1
        assert case.gold_permuted_index == permuted_index


def test_none_prediction_maps_to_none():
    assert map_prediction_to_original(None, [2, 0, 1]) is None
