from dataclasses import dataclass

import torch

from ..models import LoadedModel
from ..prompting import build_prompt, option_label
from ..schema import InferenceResult, MultipleChoiceSample
from .common import model_inputs, timed


@dataclass(frozen=True)
class LabelTokenInspection:
    supported: bool
    formatting: str | None
    token_ids: list[int] | None
    candidates: dict[str, dict[str, list[int]]]
    reason: str | None = None


def inspect_label_tokens(tokenizer, prompt: str, option_count: int) -> LabelTokenInspection:
    labels = [option_label(index) for index in range(option_count)]
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    formats = (("leading_space", " "), ("bare", ""))
    candidate_details = {
        prefix_name: {
            label: tokenizer.encode(prefix + label, add_special_tokens=False)
            for label in labels
        }
        for prefix_name, prefix in formats
    }

    for prefix_name, prefix in formats:
        encoded = candidate_details[prefix_name]
        token_ids = [encoded[label][0] for label in labels if len(encoded[label]) == 1]
        all_single = len(token_ids) == len(labels) and len(set(token_ids)) == len(labels)
        boundary_stable = all(
            tokenizer.encode(prompt + prefix + label, add_special_tokens=False)
            == prompt_ids + encoded[label]
            for label in labels
        )
        if all_single and boundary_stable:
            return LabelTokenInspection(
                supported=True,
                formatting=prefix_name,
                token_ids=token_ids,
                candidates=candidate_details,
            )

    return LabelTokenInspection(
        supported=False,
        formatting=None,
        token_ids=None,
        candidates=candidate_details,
        reason="No shared, boundary-stable single-token representation for all labels",
    )


@torch.inference_mode()
def run_label_logits(
    loaded: LoadedModel, sample: MultipleChoiceSample
) -> InferenceResult:
    prompt = build_prompt(sample)
    inspection = inspect_label_tokens(
        loaded.tokenizer, prompt, len(sample.options)
    )
    inspection_metadata = {
        "supported": inspection.supported,
        "label_formatting": inspection.formatting,
        "label_token_ids": inspection.token_ids,
        "tokenizer_candidates": inspection.candidates,
        "unsupported_reason": inspection.reason,
    }
    if not inspection.supported or inspection.token_ids is None:
        return InferenceResult(
            sample_id=sample.id,
            dataset=sample.dataset,
            model=loaded.name,
            mode="label_logit",
            prompt=prompt,
            options=sample.options,
            gold_index=sample.gold_index,
            prediction=None,
            correct=False,
            metadata=inspection_metadata,
        )

    inputs = model_inputs(loaded.tokenizer, prompt, loaded.device)

    def score() -> torch.Tensor:
        logits = loaded.model(**inputs).logits[0, -1]
        indices = torch.tensor(inspection.token_ids, device=loaded.device)
        return logits.index_select(0, indices).float()

    scores, latency_ms = timed(loaded.device, score)
    probabilities = scores.softmax(dim=0)
    prediction = int(probabilities.argmax().item())

    return InferenceResult(
        sample_id=sample.id,
        dataset=sample.dataset,
        model=loaded.name,
        mode="label_logit",
        prompt=prompt,
        options=sample.options,
        gold_index=sample.gold_index,
        prediction=prediction,
        correct=prediction == sample.gold_index,
        raw_scores=scores.cpu().tolist(),
        probabilities=probabilities.cpu().tolist(),
        confidence=float(probabilities.max().item()),
        latency_ms=latency_ms,
        metadata=inspection_metadata,
    )
