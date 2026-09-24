from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LoadedModel:
    name: str
    model: Any
    tokenizer: Any
    device: torch.device


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        return getattr(torch, value)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype: {value}") from exc


def load_model(config: dict[str, Any]) -> LoadedModel:
    name = config["model_name"]
    device = _resolve_device(config.get("device", "auto"))
    dtype = _resolve_dtype(config.get("dtype", "auto"), device)
    trust_remote_code = bool(config.get("trust_remote_code", False))

    tokenizer = AutoTokenizer.from_pretrained(
        name, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    return LoadedModel(name=name, model=model, tokenizer=tokenizer, device=device)
