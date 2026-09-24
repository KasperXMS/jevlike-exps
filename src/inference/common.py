import time
from collections.abc import Callable
from typing import TypeVar

import torch

T = TypeVar("T")


def timed(device: torch.device, operation: Callable[[], T]) -> tuple[T, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, (time.perf_counter() - started) * 1000.0


def model_inputs(tokenizer, text: str, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).items()
    }
