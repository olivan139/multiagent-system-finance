from __future__ import annotations

import json
import time

from openai import OpenAI

from ..config import LABELS, LLMConfig
from .base import AgentResult, SentimentAgent

ZERO_SHOT_SYSTEM = """\
You are a financial sentiment analysis expert. Classify the sentiment of the \
given financial/crypto news text.

Respond ONLY with a JSON object:
{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": <float 0-1>,
  "reasoning": "<brief explanation>"
}"""

FEW_SHOT_EXAMPLES = [
    {
        "text": "Bitcoin surges past $100,000 as institutional demand hits record high",
        "sentiment": "positive",
        "confidence": 0.95,
        "reasoning": "Price surge and record institutional demand are strongly positive",
    },
    {
        "text": "SEC delays decision on crypto ETF applications again",
        "sentiment": "negative",
        "confidence": 0.7,
        "reasoning": "Regulatory delays create uncertainty and are generally bearish",
    },
    {
        "text": "Bitcoin trading volume remains steady at $30 billion daily average",
        "sentiment": "neutral",
        "confidence": 0.85,
        "reasoning": "Steady volume with no directional movement is a neutral observation",
    },
]


class SingleLLMAgent(SentimentAgent):
    """Single LLM agent for sentiment analysis (zero-shot or few-shot)."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        mode: str = "zero_shot",
    ):
        self.config = config or LLMConfig()
        self.mode = mode
        self.name = f"SingleLLM-{mode}"
        self.client = OpenAI(api_key=self.config.api_key)
        self._total_tokens = 0
        self._total_cost = 0.0

    def _build_messages(self, text: str) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": ZERO_SHOT_SYSTEM}]

        if self.mode == "few_shot":
            for ex in FEW_SHOT_EXAMPLES:
                messages.append({"role": "user", "content": ex["text"]})
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "sentiment": ex["sentiment"],
                                "confidence": ex["confidence"],
                                "reasoning": ex["reasoning"],
                            }
                        ),
                    }
                )

        messages.append({"role": "user", "content": text})
        return messages

    def _estimate_cost(self, usage) -> float:
        input_cost = (usage.prompt_tokens / 1_000_000) * 0.15
        output_cost = (usage.completion_tokens / 1_000_000) * 0.60
        return input_cost + output_cost

    def analyze(self, text: str) -> AgentResult:
        messages = self._build_messages(text)

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                parsed = json.loads(content)

                self._total_tokens += response.usage.total_tokens
                self._total_cost += self._estimate_cost(response.usage)

                sentiment = parsed.get("sentiment", "neutral").lower().strip()
                if sentiment not in LABELS:
                    sentiment = "neutral"

                return AgentResult(
                    sentiment=sentiment,
                    confidence=float(parsed.get("confidence", 0.5)),
                    reasoning=parsed.get("reasoning", ""),
                    metadata={"tokens": response.usage.total_tokens},
                )
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    return AgentResult(
                        sentiment="neutral",
                        confidence=0.0,
                        reasoning=f"Error after {self.config.max_retries} retries: {e}",
                        metadata={"error": str(e)},
                    )
                time.sleep(self.config.retry_delay * (attempt + 1))

        return AgentResult(sentiment="neutral", confidence=0.0, reasoning="unreachable")

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        return self._total_tokens
