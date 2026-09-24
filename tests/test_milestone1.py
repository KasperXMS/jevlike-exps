import sys
import types
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# The core inference tests use a deterministic fake model and do not require the
# optional Hugging Face dependency to be installed.
transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoModelForCausalLM = object
transformers_stub.AutoTokenizer = object
sys.modules.setdefault("transformers", transformers_stub)

from src.inference import (  # noqa: E402
    inspect_label_tokens,
    run_generation,
    run_label_logits,
    run_option_likelihood,
)
from src.models import LoadedModel  # noqa: E402
from src.schema import MultipleChoiceSample  # noqa: E402


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    label_ids = {"A": 10, "B": 11, "C": 12, "D": 13}
    option_ids = {"Berlin": 30, "Madrid": 31, "Paris": 32, "Rome": 33}

    def encode(self, text, add_special_tokens=False):
        for label, token_id in self.label_ids.items():
            suffix = " " + label
            if text.endswith(suffix) and len(text) > len(suffix):
                return self.encode(text[: -len(suffix)]) + [token_id]
            if text == suffix:
                return [token_id]
            if text == label:
                return [token_id + 10]
        for option, token_id in self.option_ids.items():
            if text == " " + option:
                return [token_id]
        return [1, 2]

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = torch.tensor([self.encode(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, token_ids, skip_special_tokens=True):
        labels = {token_id: " " + label for label, token_id in self.label_ids.items()}
        return "".join(labels.get(token_id, "") for token_id in token_ids)


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits
        self.past_key_values = ("fake-cache",)


class FakeModel:
    def __init__(self):
        self.generation_config = types.SimpleNamespace(eos_token_id=99)

    def __call__(self, input_ids, past_key_values=None, **kwargs):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 128), -5.0)
        if past_key_values is not None:
            logits[:, -1, 99] = 10.0
        else:
            logits[:, :, 12] = 4.0
            if not kwargs.get("use_cache", False):
                logits[:, :, 32] = 8.0
        return FakeOutput(logits)


class MultiTokenLabelTokenizer(FakeTokenizer):
    def encode(self, text, add_special_tokens=False):
        if text in {" A", " B", " C", " D", "A", "B", "C", "D"}:
            return [70, 71]
        return super().encode(text, add_special_tokens=add_special_tokens)


class MilestoneOneTest(unittest.TestCase):
    def setUp(self):
        self.sample = MultipleChoiceSample(
            id="test-1",
            dataset="manual",
            question="What is the capital of France?",
            options=["Berlin", "Madrid", "Paris", "Rome"],
            gold_index=2,
        )
        self.loaded = LoadedModel(
            name="fake-model",
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            device=torch.device("cpu"),
        )

    def test_all_three_modes_choose_the_gold_option(self):
        generation = run_generation(self.loaded, self.sample)
        label_logit = run_label_logits(self.loaded, self.sample)
        likelihood = run_option_likelihood(self.loaded, self.sample)

        self.assertEqual(generation.prediction, 2)
        self.assertEqual(label_logit.prediction, 2)
        self.assertEqual(likelihood.prediction, 2)
        self.assertTrue(generation.correct)
        self.assertTrue(label_logit.correct)
        self.assertTrue(likelihood.correct)
        self.assertEqual(likelihood.metadata["option_token_counts"], [1, 1, 1, 1])

    def test_multi_token_labels_are_unsupported(self):
        inspection = inspect_label_tokens(
            MultiTokenLabelTokenizer(), "Question:\nQ\n\nAnswer:", 4
        )
        self.assertFalse(inspection.supported)
        self.assertIsNone(inspection.token_ids)


if __name__ == "__main__":
    unittest.main()
