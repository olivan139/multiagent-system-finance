from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from tqdm import tqdm


@dataclass
class AgentResult:
    sentiment: str
    confidence: float
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)


class SentimentAgent(ABC):
    """Base class for all LLM-based sentiment agents."""

    name: str = "base"

    @abstractmethod
    def analyze(self, text: str) -> AgentResult: ...

    def analyze_batch(self, texts: list[str], show_progress: bool = True) -> list[AgentResult]:
        iterator = tqdm(texts, desc=self.name) if show_progress else texts
        return [self.analyze(t) for t in iterator]

    def predict_labels(self, texts: list[str], **kwargs) -> list[str]:
        results = self.analyze_batch(texts, **kwargs)
        return [r.sentiment for r in results]

    @property
    def total_cost_usd(self) -> float:
        return 0.0

    @property
    def total_tokens(self) -> int:
        return 0
