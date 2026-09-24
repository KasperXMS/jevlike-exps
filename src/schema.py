from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MultipleChoiceSample:
    id: str
    dataset: str
    question: str
    options: list[str]
    gold_index: int
    context: str = ""

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("A multiple-choice sample needs at least two options")
        if len(self.options) > 26:
            raise ValueError("At most 26 options are supported")
        if not 0 <= self.gold_index < len(self.options):
            raise ValueError("gold_index is outside the options list")


@dataclass
class InferenceResult:
    sample_id: str
    dataset: str
    model: str
    mode: str
    prompt: str
    options: list[str]
    gold_index: int
    prediction: int | None
    correct: bool
    raw_scores: list[float] | None = None
    probabilities: list[float] | None = None
    confidence: float | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

