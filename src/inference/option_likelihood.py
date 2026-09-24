import torch

from ..models import LoadedModel
from ..prompting import build_prompt
from ..schema import InferenceResult, MultipleChoiceSample
from .common import timed


def _option_token_ids(tokenizer, option: str) -> list[int]:
    token_ids = tokenizer.encode(" " + option.strip(), add_special_tokens=False)
    if not token_ids:
        raise ValueError("An option encoded to zero tokens")
    return token_ids


@torch.inference_mode()
def run_option_likelihood(
    loaded: LoadedModel, sample: MultipleChoiceSample
) -> InferenceResult:
    prompt = build_prompt(sample)
    tokenizer = loaded.tokenizer
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    option_ids = [_option_token_ids(tokenizer, option) for option in sample.options]
    sequences = [prompt_ids + ids for ids in option_ids]
    max_length = max(len(sequence) for sequence in sequences)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has neither a pad token nor an EOS token")

    input_ids = torch.full(
        (len(sequences), max_length), pad_id, dtype=torch.long, device=loaded.device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, device=loaded.device)
        attention_mask[row, :length] = 1

    def score() -> tuple[torch.Tensor, torch.Tensor]:
        logits = loaded.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        sums: list[torch.Tensor] = []
        for row, ids in enumerate(option_ids):
            start = len(prompt_ids) - 1
            positions = torch.arange(start, start + len(ids), device=loaded.device)
            targets = input_ids[row, len(prompt_ids) : len(prompt_ids) + len(ids)]
            token_log_probs = logits[row, positions].log_softmax(
                dim=-1, dtype=torch.float32
            )
            sums.append(token_log_probs.gather(1, targets.unsqueeze(1)).sum())
        raw_sums = torch.stack(sums)
        lengths = torch.tensor(
            [len(ids) for ids in option_ids], device=loaded.device
        )
        return raw_sums, raw_sums / lengths

    (raw_sums, normalized_scores), latency_ms = timed(loaded.device, score)
    probabilities = normalized_scores.softmax(dim=0)
    prediction = int(probabilities.argmax().item())

    return InferenceResult(
        sample_id=sample.id,
        dataset=sample.dataset,
        model=loaded.name,
        mode="option_likelihood",
        prompt=prompt,
        options=sample.options,
        gold_index=sample.gold_index,
        prediction=prediction,
        correct=prediction == sample.gold_index,
        raw_scores=normalized_scores.cpu().tolist(),
        probabilities=probabilities.cpu().tolist(),
        confidence=float(probabilities.max().item()),
        latency_ms=latency_ms,
        metadata={
            "summed_log_likelihoods": raw_sums.cpu().tolist(),
            "normalized_log_likelihoods": normalized_scores.cpu().tolist(),
            "option_token_counts": [len(ids) for ids in option_ids],
            "continuation_prefix": " ",
        },
    )
