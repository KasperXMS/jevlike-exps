import re
from dataclasses import dataclass

import torch

from ..models import LoadedModel
from ..prompting import build_prompt
from ..schema import InferenceResult, MultipleChoiceSample
from .common import model_inputs, timed


@dataclass(frozen=True)
class ParsedAnswer:
    prediction: int | None
    pattern: str | None
    pattern_regex: str | None
    matched_text: str | None


_ANSWER_PATTERNS = (
    (
        "the_correct_answer_is",
        re.compile(
            r"(?i:\bthe\s+correct\s+answer\s+is)\s*[:\-]?\s*"
            r"(?:\*{1,2})?\(?([A-Z])\)?(?:\*{1,2})?"
            r"(?=[\s.,;:!?)*]|$)"
        ),
    ),
    (
        "the_answer_is",
        re.compile(
            r"(?i:\bthe\s+answer\s+is)\s*[:\-]?\s*"
            r"(?:\*{1,2})?\(?([A-Z])\)?(?:\*{1,2})?"
            r"(?=[\s.,;:!?)*]|$)"
        ),
    ),
    (
        "answer_is",
        re.compile(
            r"(?i:\banswer\s+is)\s*[:\-]?\s*"
            r"(?:\*{1,2})?\(?([A-Z])\)?(?:\*{1,2})?"
            r"(?=[\s.,;:!?)*]|$)"
        ),
    ),
    (
        "answer_colon",
        re.compile(
            r"(?i:\banswer)\s*:\s*(?:\*{1,2})?\(?([A-Z])\)?"
            r"(?:\*{1,2})?(?=[\s.,;:!?)*]|$)"
        ),
    ),
    (
        "option_label",
        re.compile(
            r"(?i:\boption)\s+(?:\*{1,2})?\(?([A-Z])\)?"
            r"(?:\*{1,2})?(?=[\s.,;:!?)*]|$)"
        ),
    ),
    (
        "parenthesized_label",
        re.compile(r"^\s*\(([A-Z])\)(?=[\s.,;:!?]|$)"),
    ),
    (
        "label_period",
        re.compile(r"^\s*([A-Z])\.(?=\s|$)"),
    ),
    (
        "label_because",
        re.compile(r"^\s*([A-Z])\s+(?i:because)\b"),
    ),
    (
        "bare_label",
        re.compile(r"^\s*([A-Z])\s*$"),
    ),
)


def parse_generation_answer(text: str, option_count: int) -> ParsedAnswer:
    if not 1 <= option_count <= 26:
        raise ValueError("option_count must be between 1 and 26")
    for pattern_name, pattern in _ANSWER_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        prediction = ord(match.group(1)) - ord("A")
        if prediction < option_count:
            return ParsedAnswer(
                prediction, pattern_name, pattern.pattern, match.group(0)
            )
    return ParsedAnswer(None, None, None, None)


@torch.inference_mode()
def run_generation(
    loaded: LoadedModel,
    sample: MultipleChoiceSample,
    max_new_tokens: int = 8,
) -> InferenceResult:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    prompt = build_prompt(sample)
    inputs = model_inputs(loaded.tokenizer, prompt, loaded.device)

    def prefill():
        return loaded.model(**inputs, use_cache=True)

    outputs, prefill_ms = timed(loaded.device, prefill)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [int(next_token.item())]
    past_key_values = outputs.past_key_values

    def decode() -> None:
        nonlocal next_token, past_key_values
        eos_ids = loaded.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = loaded.tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_ids = set(eos_ids or [])

        for _ in range(max_new_tokens - 1):
            if generated[-1] in eos_ids:
                break
            step = loaded.model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = step.past_key_values
            next_token = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))

    _, decode_ms = timed(loaded.device, decode)
    generated_text = loaded.tokenizer.decode(generated, skip_special_tokens=True)
    parsed = parse_generation_answer(generated_text, len(sample.options))
    prediction = parsed.prediction

    return InferenceResult(
        sample_id=sample.id,
        dataset=sample.dataset,
        model=loaded.name,
        mode="generation",
        prompt=prompt,
        options=sample.options,
        gold_index=sample.gold_index,
        prediction=prediction,
        correct=prediction == sample.gold_index,
        confidence=None,
        latency_ms=prefill_ms + decode_ms,
        metadata={
            "generated_text": generated_text,
            "generated_token_ids": generated,
            "parse_succeeded": prediction is not None,
            "parser_pattern": parsed.pattern,
            "parser_pattern_regex": parsed.pattern_regex,
            "parser_match": parsed.matched_text,
            "prefill_latency_ms": prefill_ms,
            "decode_latency_ms": decode_ms,
            "total_latency_ms": prefill_ms + decode_ms,
        },
    )
